"""Shared evaluation helpers for GPT/BERT evaluation scripts.

Functions extracted from eval_bert_calibration_v2.py and eval_gpt.py
to eliminate code duplication.
"""
import os
import numpy as np
import pandas as pd
import torch
from glob import glob

from config import DataConfig, NormConfig
from data_processor import document_normalize, _stock_cutoff_idx, load_stocks, split_stocks
from model import load_tokenizer
from model.kronos_preview import KronosPreview


# ============================================================================
# Model loading
# ============================================================================

def load_gpt(path, device):
    """Load a KronosPreview GPT model for evaluation."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


# ============================================================================
# Data preparation
# ============================================================================

def attach_close_prices(test_stocks):
    """Attach `close_prices` [T] float64 to each stock by re-reading its CSV."""
    csv_map = {os.path.basename(f).split(".")[0]: f
               for f in sorted(glob("dataset/*.csv"))}
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


def build_stock_arrays(stock):
    """Build per-stock arrays matching the v2 training pipeline.

    Returns dict with normalization stats, test_start/test_end, raw features.
    None if stock is too short.
    """
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    close = stock["close_prices"]
    ci = _stock_cutoff_idx(stock, DataConfig.cutoff_date)
    T_total = len(feat)
    m = NormConfig.min_lookback
    if T_total < m + 10 or ci < m:
        return None

    price_feat = feat[:, :4]
    price_normed, va_normed = document_normalize(feat, cutoff_idx=ci)
    p_mean = price_feat[:ci].mean(axis=0)
    p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)

    return {
        "feat": feat, "close": close, "ci": ci, "T_total": T_total,
        "p_mean": p_mean, "p_std": p_std,
        "price_normed": price_normed, "va_normed": va_normed,
        "day": day, "month": month, "year": year,
    }


def build_gpt_eval_inputs(arrays, tokenizer, device):
    """Build the GPT forward inputs matching v2 training exactly.

    Returns dict with inp, tids, pos, mask, va_values, S, token_ids,
    test_start, test_end.
    """
    price_normed = arrays["price_normed"]
    day, month, year = arrays["day"], arrays["month"], arrays["year"]
    T_total = arrays["T_total"]

    idx_c, _ = tokenizer.encode(
        torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
    token_ids = idx_c[0].cpu().numpy()
    vocab = tokenizer.vocab_size
    bos_id = vocab

    N = T_total
    ids = [bos_id] + token_ids[:N].tolist()
    day_list = [day[0]] + day[:N].tolist()
    month_list = [month[0]] + month[:N].tolist()
    year_list = [year[0]] + year[:N].tolist()
    S = len(ids)

    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tids = torch.stack([
        torch.tensor([day_list[:-1]], dtype=torch.long),
        torch.tensor([month_list[:-1]], dtype=torch.long),
        torch.tensor([year_list[:-1]], dtype=torch.long),
    ], dim=-1).to(device)
    pos = torch.arange(S - 1, device=device).unsqueeze(0)
    mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))

    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        arrays["va_normed"][:N - 1],
    ], axis=0)
    va = torch.tensor(va_seq, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "inp": inp, "tids": tids, "pos": pos, "mask": mask, "va_values": va,
        "S": S, "token_ids": token_ids,
        "test_start": arrays["ci"], "test_end": T_total - 2,
    }


def decode_predicted_token(token_id, tokenizer, device):
    """Decode a single predicted token to a feature vector."""
    pred_indices = (torch.tensor([token_id], dtype=torch.long, device=device)
                    .unsqueeze(0).unsqueeze(-1)
                    .expand(-1, -1, 2).contiguous())
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]


# ============================================================================
# GPT inference
# ============================================================================

AMP_DTYPE = torch.bfloat16


@torch.no_grad()
def get_gpt_full_seq_logits(gpt, tokenizer, stock, device):
    """Run GPT once on the full sequence. Returns coarse logits at every position."""
    arrays = build_stock_arrays(stock)
    if arrays is None:
        return None
    inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        coarse_logits, fine_logits = gpt(inputs["inp"], inputs["tids"], inputs["pos"],
                                         inputs["mask"], va_values=inputs["va_values"])
    return {
        "logits": coarse_logits.float().cpu(),
        "fine_logits": fine_logits.float().cpu(),
        "token_ids": inputs["token_ids"],
        "test_start": inputs["test_start"],
        "test_end": inputs["test_end"],
        "p_mean": arrays["p_mean"], "p_std": arrays["p_std"],
        "feat": arrays["feat"], "close": arrays["close"],
        "T_total": arrays["T_total"],
        "_arrays": arrays,
    }


def decode_coarse_token(coarse_id, fine_logits_at_pos, tokenizer, device):
    """Decode a coarse prediction + fine logits to a feature vector.

    Uses the fine head's argmax to select the fine token, then decodes
    the (coarse, fine) pair through the tokenizer's full decoder.
    """
    # Get fine token from fine head
    fine_id = int(fine_logits_at_pos.argmax().item())

    # Build [1, 1, 2] index tensor for tokenizer.decode_all
    pred_indices = torch.tensor([[[coarse_id, fine_id]]], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]
