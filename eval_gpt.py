"""GPT-only evaluation: DA, MAPE, AmpRatio, Collapse, Unique, Rank-IC.

Uses eval_helpers.py for shared data preparation and GPT inference.

Usage:
    python eval_gpt.py --gpt_ckpt checkpoints/sweep/bits_7_6_gpt.pt \
                       --tokenizer checkpoints/sweep/bits_7_6_tok.pt
    python eval_gpt.py --n_stocks 10  # quick test
"""
import argparse
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

from config import ModelConfig, set_global_seed
from data_processor import load_stocks, split_stocks
from model import load_tokenizer
from eval_helpers import (
    load_gpt, attach_close_prices, get_gpt_full_seq_logits, decode_coarse_token,
)

N_TEST_STOCKS = 30
SEED = 42


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(pred_lrs, true_lrs, pred_toks, base_closes, true_closes):
    pred_lr_arr = np.array(pred_lrs)
    true_lr_arr = np.array(true_lrs)
    pred_toks_arr = np.array(pred_toks, dtype=np.int64)
    base_close_arr = np.array(base_closes)
    true_close_arr = np.array(true_closes)
    eps = 1e-8

    if len(pred_lr_arr) == 0:
        return None

    da = (np.sign(pred_lr_arr) == np.sign(true_lr_arr)).mean()

    pred_close_arr = base_close_arr * np.exp(pred_lr_arr.astype(np.float64))
    mape_pt = np.abs(pred_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps) * 100
    mape = mape_pt.mean()
    bl_mape = np.mean(np.abs(base_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps)) * 100

    ampratio = np.mean(np.abs(pred_lr_arr)) / max(np.mean(np.abs(true_lr_arr)), eps)

    unique, counts = np.unique(pred_toks_arr, return_counts=True)
    collapse = counts.max() / len(pred_toks_arr)
    n_unique = len(unique)

    rank_ic, _ = spearmanr(pred_lr_arr, true_lr_arr) if len(pred_lr_arr) > 2 else (0.0, 1.0)

    return {
        "da": float(da), "mape": float(mape), "baseline_mape": float(bl_mape),
        "ampratio": float(ampratio), "collapse_rate": float(collapse),
        "n_unique_tokens": int(n_unique),
        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        "n_predictions": len(pred_lr_arr),
    }


# ============================================================================
# Main
# ============================================================================

def evaluate(gpt_ckpt, tokenizer_ckpt, device, n_stocks=N_TEST_STOCKS,
             max_test_pos=0, silent=False):
    """Run GPT-only evaluation with dual-head decode. Returns metrics dict."""
    tokenizer = load_tokenizer(tokenizer_ckpt, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    gpt = load_gpt(gpt_ckpt, device)

    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)
    attach_close_prices(test_stocks_all)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks_all),
                         min(n_stocks, len(test_stocks_all)), replace=False)
    test_stocks = [test_stocks_all[i] for i in sorted(indices)]

    pred_toks, pred_lrs, true_lrs = [], [], []
    base_closes, true_closes = [], []
    t0 = time.time()

    for si, stock in enumerate(test_stocks):
        info = get_gpt_full_seq_logits(gpt, tokenizer, stock, device)
        if info is None:
            continue

        test_start = info["test_start"]
        test_end = info["test_end"]
        if test_end <= test_start:
            continue
        if max_test_pos > 0 and (test_end - test_start + 1) > max_test_pos:
            test_end = test_start + max_test_pos - 1

        coarse_logits = info["logits"]       # [1, T, vocab_coarse]
        fine_logits = info["fine_logits"]     # [1, T, vocab_fine]
        vocab = tokenizer.vocab_coarse
        p_mean, p_std = info["p_mean"], info["p_std"]
        feat, close = info["feat"], info["close"]

        for p in range(test_start, test_end + 1):
            coarse_lp = coarse_logits[0, p, :vocab].float()
            chosen_coarse = int(torch.argmax(coarse_lp).item())

            # Decode with fine head
            pred_feat = decode_coarse_token(chosen_coarse, fine_logits[0, p], tokenizer, device)
            pred_lr = pred_feat[0] * p_std[0] + p_mean[0]

            pred_toks.append(chosen_coarse)
            pred_lrs.append(pred_lr)
            true_lrs.append(feat[p + 1, 0])
            base_closes.append(close[p])
            true_closes.append(close[p + 1])

        if not silent and ((si + 1) % 5 == 0 or si == len(test_stocks) - 1):
            elapsed = time.time() - t0
            n_total = len(pred_lrs)
            running_da = (np.sign(np.array(pred_lrs)) == np.sign(np.array(true_lrs))).mean() * 100 if n_total > 0 else 0.0
            print(f"  [{si+1}/{len(test_stocks)}] {elapsed:.0f}s n_pred={n_total} DA={running_da:.2f}%",
                  flush=True)

    metrics = compute_metrics(pred_lrs, true_lrs, pred_toks, base_closes, true_closes)
    if metrics is None:
        return {"da": 0, "mape": 999, "ampratio": 0, "collapse_rate": 1.0,
                "n_unique_tokens": 0, "rank_ic": 0, "n_predictions": 0}

    if not silent:
        print(f"\n  DA={metrics['da']*100:.2f}%  MAPE={metrics['mape']:.2f}%  "
              f"AmpRatio={metrics['ampratio']:.3f}x  Collapse={metrics['collapse_rate']*100:.1f}%  "
              f"Unique={metrics['n_unique_tokens']}  RankIC={metrics['rank_ic']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="GPT-only evaluation")
    parser.add_argument("--gpt_ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--n_stocks", type=int, default=N_TEST_STOCKS)
    parser.add_argument("--max_test_pos", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(args.gpt_ckpt, args.tokenizer, device,
                       n_stocks=args.n_stocks, max_test_pos=args.max_test_pos)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
