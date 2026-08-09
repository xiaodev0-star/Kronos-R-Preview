"""t1_masking.py — plan §4 T1: scoring-aligned MLM masking.

Three alignment changes vs Stage-1 F2 80/10/10 (fixes §3 E-2):

  1. final-position masking  — 50% of examples are the scoring shape: a
     truncated window with [MASK] at the LAST position (no trailing EOS), the
     exact input the critic sees at scoring time (§8.1).  BERT never saw
     "last position is a masked day token" during Stage 1.
  2. va-dropout  — masked rows get va_values = 0 with 50% probability, making
     "day token + va=0" (the leakage-safe scoring form) an in-training shape.
     (The scoring-shape examples always zero the MASK-row va, matching §8.1.)
  3. recency-weighted masking  — mask probability ramps linearly to 2x over the
     last ``recency_window`` positions (strengthen near-term dynamics, §3 E-3).

The random-replacement branch of 80/10/10 is preserved everywhere (it is the
distribution-robustness device that teaches BERT to tolerate GPT-inserted
tokens), now recency-weighted.
"""
from __future__ import annotations

import torch


def _recency_weights(seq_len, recency_window, device):
    """Linear ramp 1.0 -> 2.0 over the last ``recency_window`` positions."""
    dist = seq_len - 1 - torch.arange(seq_len, device=device)   # 0 at the last
    w = torch.ones(seq_len, device=device)
    if recency_window > 1:
        near = dist < recency_window
        w[near] = 1.0 + (recency_window - 1 - dist[near]).clamp(min=0).float() \
            / float(recency_window - 1)
    return w


def _corrupt_801010(ids, vocab_base, mask_id, mlm_prob, corrupt_fracs,
                    recency_window, generator):
    """Recency-weighted 80/10/10 corruption over a 1-D day-token slice.

    Returns (corrupted_ids, labels) with the same length; specials untouched.
    """
    N = ids.shape[0]
    device = ids.device
    is_special = ids >= vocab_base
    w = _recency_weights(N, recency_window, device)
    rand = torch.rand(N, device=device, generator=generator)
    can_mask = (~is_special) & (rand < (mlm_prob * w).clamp(max=0.5))

    f_mask, f_rnd, f_keep = corrupt_fracs
    pick = torch.rand(N, device=device, generator=generator)
    do_mask = can_mask & (pick < f_mask)
    do_random = can_mask & (pick >= f_mask) & (pick < f_mask + f_rnd)

    out = ids.clone()
    out[do_mask] = mask_id
    if bool(do_random.any()):
        n_rnd = int(do_random.sum())
        out[do_random] = torch.randint(0, vocab_base, (n_rnd,), device=device)
    labels = torch.full((N,), -100, dtype=torch.long, device=device)
    labels[can_mask] = ids[can_mask]
    return out, labels


def make_t1_batch(input_ids, time_id, va, vocab_base, mask_id,
                  window=512, mlm_prob=0.15, corrupt_fracs=(0.8, 0.1, 0.1),
                  final_pos_frac=0.5, va_zero_frac=0.5, recency_window=64,
                  generator=None):
    """Build one scoring-aligned MLM example.

    Args (all [N] 1-D, or [N,3]/[N,2] for time/va):
        input_ids: original packed v2 ids (BOS ... tokens ... EOS)
        time_id: [N,3] day/month/year calendar
        va: [N,2] volume/amount
    Returns:
        (mlm_ids [L], time_t [L,3], va_t [L,2], labels [L], pos_out [L], offset)
    L <= window; pos_out restarts at 0 for truncated windows (scoring shape);
    ``offset`` is the window's first index in the original sequence (0 for the
    full-sequence random path) — used by T2 to align cached GPT proposals.
    """
    device = input_ids.device
    if generator is not None:
        r = torch.rand(1, device=device, generator=generator).item()
    else:
        r = torch.rand(1, device=device).item()

    day_mask = input_ids < vocab_base
    valid_pos = torch.nonzero(day_mask, as_tuple=False).squeeze(-1)
    if valid_pos.numel() == 0:
        # degenerate (no day token): fall back to a masked full sequence
        valid_pos = torch.arange(input_ids.shape[0], device=device)

    if r < final_pos_frac and valid_pos.numel() > 1:
        # ---- scoring shape: truncated window, MASK at the last position ----
        p = int(valid_pos[torch.randint(valid_pos.numel(), (1,),
                                        generator=generator).item()])
        start = max(0, p + 1 - window)
        ids = input_ids[start:p + 1].clone()
        tids = time_id[start:p + 1]
        v = va[start:p + 1].clone()
        L = ids.shape[0]
        ids[-1] = mask_id
        labels = torch.full((L,), -100, dtype=torch.long, device=device)
        labels[-1] = input_ids[p]
        if L > 1:
            hist, hlab = _corrupt_801010(
                input_ids[start:p], vocab_base, mask_id, mlm_prob,
                corrupt_fracs, recency_window, generator)
            ids[:-1] = hist
            labels[:-1] = hlab
        v[-1] = 0.0                                   # §8.1 leakage-safe form
        pos_out = torch.arange(L, device=device)      # restart at 0 (scoring)
        return ids, tids, v, labels, pos_out, start

    # ---- random path: full sequence, recency-weighted 80/10/10 + va-dropout ----
    ids, labels = _corrupt_801010(input_ids, vocab_base, mask_id, mlm_prob,
                                  corrupt_fracs, recency_window, generator)
    v = va.clone()
    masked = ids == mask_id
    vdrop = torch.rand(v.shape[0], device=device, generator=generator) \
        < va_zero_frac
    v = v.masked_fill((masked & vdrop).unsqueeze(-1), 0.0)
    pos_out = torch.arange(ids.shape[0], device=device)
    return ids, time_id, v, labels, pos_out, 0
