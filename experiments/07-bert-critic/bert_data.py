"""07-bert-critic: BERT scoring-input construction (BERT-Critic-Rerank-Plan §8.1).

For one (stock_uid, date, selection_position p) prediction the exact BERT input is

    [BOS] + last W historical tokens (positions up to p) + [MASK]

where the MASK slot is position p+1 — it predicts feature row p, the next
trading day.  Alignment is inherited from pack_stocks_v2 / build_prepared_batches:

  - input position j holds the token of feature row j-1, its own calendar
    (time_ids[j] = calendar of row j-1) and its own va (va_values[j]).
  - the MASK slot at position p+1 has time_ids = calendar of feature row p
    (the next day / target date, which is deterministic and known) and
    va_values = 0 — the next day's volume/amount is NOT known.  Hard leakage
    red line; any non-zero va on the MASK row voids the arm.
  - the sequence is truncated at the MASK slot (no rows after t+1).
  - position_ids restart at 0 (RoPE matches the truncated training sequences).

The caller (score_bert.py) walks the 06 prepared inputs (prepared_inputs.pt)
and calls ``selection_history_window`` + ``build_bert_input`` per selection.
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VOCAB_BASE = 128          # tokenizer.vocab_coarse (F1-wired)
CALIB_START, CALIB_STOP = "2023-02-01", "2024-02-01"


# ============================================================================
# Calendar helpers
# ============================================================================

def in_audit_calibration(date_str: str) -> bool:
    """True if ``date_str`` (YYYY-MM-DD) lies in the audit calibration slice.

    The calibration slice is audit_uids x [2023-02-01, 2024-02-01) — the ONLY
    region where fusion weights (lambda / stacking / acceptance threshold) may
    be fit.  T6 pins this boundary.
    """
    return CALIB_START <= date_str < CALIB_STOP


def bert_time_for_target_date(date_str: str) -> torch.Tensor:
    """Calendar (day, month, year_enc) for a 'YYYY-MM-DD' target date.

    year_enc = year - 2010 clipped to [0, 99], matching data_processor's year
    embedding convention.  The next trading day's calendar is deterministic and
    known at prediction time (not leakage).
    """
    y, m, d = (int(x) for x in str(date_str)[:10].split("-"))
    y_enc = max(0, min(99, y - 2010))
    return torch.tensor([d, m, y_enc], dtype=torch.long)


# ============================================================================
# Per-selection history window
# ============================================================================

def selection_history_window(input_ids_row, time_ids_row, va_row, pos, window):
    """Slice the last ``window`` GPT input positions ending at ``pos``.

    ``input_ids_row`` [N] is the packed input (index 0 = BOS, index j = token
    of feature row j-1); the selection at position ``pos`` predicts feature row
    ``pos``.  Returns the real history window that the BERT critic scores
    against — the same window GPT used.

    Returns ``(hist_ids, hist_time, hist_va)``:
      hist_ids   [L] long  — real tokens (starts with BOS if the window reaches
                             the packed BOS, otherwise tokens only)
      hist_time  [L, 3]    — each position's own calendar
      hist_va    [L, 2]    — each position's real va
    """
    start = max(0, pos + 1 - window)
    hist_ids = input_ids_row[start:pos + 1]
    hist_time = time_ids_row[start:pos + 1]
    hist_va = va_row[start:pos + 1]
    return hist_ids, hist_time, hist_va


def build_bert_input(hist_ids, hist_time, hist_va, target_time,
                     vocab_base=VOCAB_BASE, mask_id=VOCAB_BASE + 2):
    """Build one BERT scoring input from a history window (plan §8.1).

    Returns ``(input_ids, time_ids, va_values, position_ids)``:
      input_ids [L+1]      — history + [MASK] at the last slot
      time_ids  [L+1, 3]   — history calendars + target-time for the MASK slot
      va_values [L+1, 2]   — history va (real) + 0 for the MASK slot
      position_ids [L+1]   — contiguous 0..L
    The history window is expected to start with BOS when it reaches the packed
    BOS; if the caller passes tokens-only history the model still scores the
    MASK slot (BOS is a soft feature — never masked, never predicted).
    """
    ids = torch.cat([hist_ids, torch.tensor([mask_id], dtype=torch.long)])
    tids = torch.cat([hist_time, target_time.unsqueeze(0)], dim=0)
    va = torch.cat([hist_va, torch.zeros(1, 2, dtype=hist_va.dtype)], dim=0)
    pos = torch.arange(ids.shape[0], dtype=torch.long)
    return ids, tids, va, pos


# ============================================================================
# Batch-level construction for scoring (padded, single forward)
# ============================================================================

def selection_bert_batch(selections, window=512, vocab_base=VOCAB_BASE,
                         mask_id=VOCAB_BASE + 2):
    """Build a padded BERT batch for a list of selections.

    ``selections``: list of dicts with keys ``input_ids_row``, ``time_ids_row``,
    ``va_row``, ``pos``, ``date_key``.  The MASK slot's calendar is parsed
    exactly from ``date_key`` (the deterministic next-day calendar).

    Returns ``(input_ids, time_ids, va_values, lengths, mask_positions)``
    with ``mask_positions[i]`` = the index of the [MASK] slot in padded row i.
    """
    rows = []
    for sel in selections:
        hist_ids, hist_time, hist_va = selection_history_window(
            sel["input_ids_row"], sel["time_ids_row"], sel["va_row"],
            sel["pos"], window)
        target_time = bert_time_for_target_date(sel["date_key"])
        ids, tids, va, _ = build_bert_input(
            hist_ids, hist_time, hist_va, target_time,
            vocab_base=vocab_base, mask_id=mask_id)
        rows.append((ids, tids, va))
    max_len = max(r[0].shape[0] for r in rows)
    B = len(rows)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    time_ids = torch.zeros(B, max_len, 3, dtype=torch.long)
    va_values = torch.zeros(B, max_len, 2, dtype=torch.float32)
    lengths = torch.empty(B, dtype=torch.long)
    mask_positions = torch.empty(B, dtype=torch.long)
    for i, (ids, tids, va) in enumerate(rows):
        L = ids.shape[0]
        lengths[i] = L
        input_ids[i, :L] = ids
        time_ids[i, :L] = tids
        va_values[i, :L] = va
        mask_positions[i] = L - 1
    return input_ids, time_ids, va_values, lengths, mask_positions
