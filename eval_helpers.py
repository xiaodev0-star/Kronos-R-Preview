"""Shared evaluation helpers.

Centralizes the (previously duplicated) GPT/BERT forward path and model loading
used by all eval scripts:
  - eval_batch_1step.py
  - eval_cross_loss.py
  - eval_bert_calibration.py
  - eval_bert_calibration_v2.py

The previous copies had two bugs that biased the eval distribution vs training:
  1. `va_values` was passed as zeros — training feeds the real (per-stock) VA.
  2. `time_ids` in BERT was set to the predicted-position's date at every history
     position, leaking a uniform date and destroying positional context.

This module fixes both. All eval scripts should import `build_gpt_eval_inputs` and
`build_bert_eval_inputs` from here.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
import torch

from config import DataConfig, NormConfig
from data_processor import document_normalize, _stock_cutoff_idx
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from model.kronos_bert import KronosBert

BOS_ID = 1024
EOS_ID = 1025
MASK_ID = 1026
VOCAB_BASE = 1024


def load_tokenizer(path, device):
    """Load a frozen BSQ tokenizer for evaluation."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def load_gpt(path, device):
    """Load a KronosPreview GPT model for evaluation."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_bert(path, device):
    """Load a KronosBert calibrator for evaluation (reads size config from ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    from config import ModelConfig as _MC

    class _Cfg:
        pass

    cfg = _Cfg()
    for attr in dir(_MC):
        if attr.startswith("_"):
            continue
        setattr(cfg, attr, getattr(_MC, attr))
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    model = KronosBert(cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def attach_close_prices(test_stocks):
    """Attach `close_prices` [T] float64 to each stock by re-reading its CSV.

    Falls back to exp(cumsum(log_ret)) when the CSV is unavailable.
    """
    csv_map = {os.path.basename(f).split(".")[0]: f for f in sorted(glob("dataset/*.csv"))}
    for s in test_stocks:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            s["close_prices"] = df["close"].values.astype(np.float64)
        else:
            lr = s["features_raw"][:, 0]
            s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)


def _cutoff_idx(stock):
    return int(np.searchsorted(stock["dates_dt"],
                               np.datetime64(pd.Timestamp(DataConfig.cutoff_date)), side="left"))


def build_stock_arrays(stock):
    """Build (token_ids, day, month, year, va_values, p_mean, p_std, test_start, test_end, T_total)
    for one stock, matching the v2 training pipeline exactly.

    Returns None if the stock is too short / has no test period.
    """
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    close = stock["close_prices"]
    ci = _cutoff_idx(stock)
    T_total = len(feat)
    m = NormConfig.min_lookback
    if T_total < m + 10 or ci < m:
        return None

    price_feat = feat[:, :4]  # [T, 4] OHLC (log_ret, log_high, log_low, log_open)
    # Per-stock historical Z-Score, stats from train-only data — same as training
    price_normed, va_normed = document_normalize(feat, cutoff_idx=ci)
    # NOTE: va_normed already has first-day baseline + Z-Score, length T_total, [T, 2]
    # document_normalize with cutoff_idx=ci uses feat[:ci] for stats (train-only), then
    # applies them to the full feat. The returned va_normed has shape [T_total, 2].
    p_mean = price_feat[:ci].mean(axis=0)
    p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)

    return {
        "feat": feat,
        "close": close,
        "ci": ci,
        "T_total": T_total,
        "p_mean": p_mean,
        "p_std": p_std,
        "price_normed": price_normed,
        "va_normed": va_normed,
        "day": day,
        "month": month,
        "year": year,
    }


def build_gpt_eval_inputs(arrays, tokenizer, device):
    """Build the GPT forward inputs (matching training):
        - input_ids: [1, S-1]  (BOS + tokens[:-1])
        - time_ids:  [1, S-1, 3]  (per-position day/month/year)
        - position_ids: [1, S-1]
        - attn_mask: [1, S-1, S-1] causal (BOS visible to all)
        - va_values: [1, S-1, 2]  real (NOT zero!) per-stock VA
    Returns the dict and a forward-able model output is computed by the caller.
    """
    price_normed = arrays["price_normed"]
    day, month, year = arrays["day"], arrays["month"], arrays["year"]
    T_total = arrays["T_total"]
    ci = arrays["ci"]

    idx_c, _ = tokenizer.encode(
        torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
    token_ids = idx_c[0].cpu().numpy()  # [T_total]
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab

    N = T_total
    ids = [bos_id] + token_ids[:N].tolist()
    d_l = [day[0]] + day[:N].tolist()
    m_l = [month[0]] + month[:N].tolist()
    y_l = [year[0]] + year[:N].tolist()
    S = len(ids)

    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tids = torch.stack([
        torch.tensor([d_l[:-1]], dtype=torch.long),
        torch.tensor([m_l[:-1]], dtype=torch.long),
        torch.tensor([y_l[:-1]], dtype=torch.long),
    ], dim=-1).to(device)
    pos = torch.arange(S - 1, device=device).unsqueeze(0)
    mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))

    # VA: build [S-1, 2] tensor matching the v2 packing in data_processor.py:
    #   inp = [BOS, content[0], ..., content[T-2]]  (length S-1 = T_total)
    #   corresponding va = [zero, va_normed[0], ..., va_normed[T-2]]
    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        arrays["va_normed"][:N - 1],  # va for content[0..T-2]
    ], axis=0)  # [T_total, 2]
    va = torch.tensor(va_seq, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "inp": inp,
        "tids": tids,
        "pos": pos,
        "mask": mask,
        "va_values": va,
        "S": S,
        "token_ids": token_ids,
        "test_start": ci,
        "test_end": T_total - 2,
    }


def build_bert_position_arrays(arrays, prefix_len, candidate_token_id, mask_positions,
                                token_ids=None):
    """Build a [1, S] BERT forward input for one (prefix_len, candidate, mask_positions) triple.

    Length S = 1 (BOS) + prefix_len (history) + 1 (candidate). Mask positions
    are replaced with MASK_ID. Per-position day/month/year from the original stock
    are used (NOT the predicted-position's date copied to all).

    Args:
        arrays: stock arrays dict (from build_stock_arrays) — supplies day/month/year/va
        prefix_len: number of history tokens (positions [0..prefix_len-1] of stock's token_ids)
        candidate_token_id: the candidate token to append as "future" context
        mask_positions: positions (1-indexed in the BERT input, where 0=BOS, S-1=candidate)
                        at which to insert MASK_ID
        token_ids: full stock token sequence; if None, only dates/VA are used (no prefix tokens).
                   If provided, the prefix is built from it.

    Returns the input tensors ready for `bert(...)`.
    """
    day, month, year = arrays["day"], arrays["month"], arrays["year"]
    if token_ids is None:
        # When no token sequence is provided, fall back to BOS-padded prefix.
        # BERT will still see correct per-position dates and VA values.
        token_ids = np.full(prefix_len, BOS_ID, dtype=np.int64)

    # prefix: [BOS, tok_0, ..., tok_{prefix_len-1}]  (length prefix_len+1)
    prefix = [BOS_ID] + token_ids[:prefix_len].tolist()
    # Insert masks
    inp_list = list(prefix)
    for mp in mask_positions:
        if mp < 0 or mp >= len(inp_list):
            raise ValueError(f"mask_pos {mp} out of range [0, {len(inp_list)})")
        inp_list[mp] = MASK_ID
    inp_list.append(int(candidate_token_id))  # candidate as "future"
    S = len(inp_list)
    inp = torch.tensor([inp_list], dtype=torch.long)

    # Time embedding: per-position dates from the original stock sequence
    #   BOS (position 0) is "synthetic" — copy day[0] to it
    #   position i (1..prefix_len) maps to day[i-1] (so position 1 = day of tok_0)
    #   candidate position (S-1) maps to day[prefix_len] (the predicted position)
    d_list = [int(day[0])] + [int(day[i]) for i in range(prefix_len)] + [int(day[prefix_len])]
    m_list = [int(month[0])] + [int(month[i]) for i in range(prefix_len)] + [int(month[prefix_len])]
    y_list = [int(year[0])] + [int(year[i]) for i in range(prefix_len)] + [int(year[prefix_len])]
    tids = torch.tensor([list(zip(d_list, m_list, y_list))], dtype=torch.long)

    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)

    # VA: prefix is from stock data; candidate position uses va_normed at predicted day
    # BOS uses zero VA; positions 1..prefix_len use va_normed[0..prefix_len-1];
    # candidate position uses va_normed[prefix_len]
    va_data = arrays["va_normed"]
    va_prefix = va_data[:prefix_len]
    va_candidate = va_data[prefix_len:prefix_len + 1]
    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        va_prefix,
        va_candidate,
    ], axis=0)
    va = torch.tensor(va_seq, dtype=torch.float32).unsqueeze(0)
    return {
        "inp": inp,
        "tids": tids,
        "pos": pos,
        "va_values": va,
        "S": S,
    }
