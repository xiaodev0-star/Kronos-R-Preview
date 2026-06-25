"""Evaluation with multi-day sliding window: fixes single-date cross-section bias.

The original eval_gpt.py only predicts test_pos (cutoff+1) per stock. Since all
stocks share the same cutoff_date=2024-02-01, all 4540+ predictions share the
same target date 2024-02-02 — making DA a cross-section-direction metric, not
a time-series-prediction metric.

Usage:
    python eval_windowed.py --gpt_ckpt checkpoints/sweep/bits_7_7_gpt.pt \\
                            --tokenizer checkpoints/sweep/bits_7_7_tok.pt \\
                            --n_days 20 --output results_windowed.json
"""
import argparse
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from config import ModelConfig, set_global_seed
from data_processor import load_stocks, split_stocks
from model import load_tokenizer
from eval_helpers import (
    load_gpt, attach_close_prices,
    batched_gpt_eval_windowed, compute_windowed_metrics,
)

SEED = 42
N_STOCKS = 2400


def evaluate_windowed(gpt_ckpt, tokenizer_ckpt, device,
                      n_stocks=N_STOCKS, n_days=20, batch_size=4, silent=False):
    """Run multi-day windowed GPT evaluation. Returns metrics dict."""
    tokenizer = load_tokenizer(tokenizer_ckpt, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    gpt = load_gpt(gpt_ckpt, device, tokenizer=tokenizer)

    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks = split_stocks(stocks)
    attach_close_prices(test_stocks)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks),
                         min(n_stocks, len(test_stocks)), replace=False)
    test_sample = [test_stocks[i] for i in sorted(indices)]

    t0 = time.time()
    preds = batched_gpt_eval_windowed(
        gpt, tokenizer, test_sample, device,
        batch_size=batch_size, n_days=n_days, silent=silent)

    elapsed = time.time() - t0
    metrics = compute_windowed_metrics(preds)

    if not silent:
        print(f"\n  Eval done in {elapsed:.1f}s ({len(preds)} predictions, "
              f"{metrics.get('n_dates', 0)} dates)")
        print(f"  Avg DA per date:  {metrics['avg_da_per_date']*100:.2f}%")
        print(f"  DA above baseline: {metrics['avg_da_above_baseline']*100:+.2f}%")
        print(f"  DA std across dates: {metrics.get('da_std', 0)*100:.2f}%")
        print(f"  Collapse: {metrics['collapse_rate']*100:.1f}%  "
              f"Unique: {metrics['n_unique_tokens']}  "
              f"RankIC: {metrics['rank_ic']:.4f}  "
              f"AmpRatio: {metrics.get('ampratio', 0):.3f}x")
        print(f"  Signal score: {metrics.get('signal_score', 0)*100:+.2f}%")

        # Show some per-date detail
        per_date = metrics.get("per_date", {})
        if per_date:
            print(f"\n  Per-date detail (first 5 dates):")
            for i, (d, v) in enumerate(sorted(per_date.items())):
                if i >= 5:
                    break
                print(f"    {d}: DA={v['da']*100:5.1f}%  "
                      f"base={v['baseline_da']*100:5.1f}%  "
                      f"excess={v['da_above_baseline']*100:+5.1f}%  "
                      f"n={v['n']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-day windowed GPT evaluation (fixes single-date bias)")
    parser.add_argument("--gpt_ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--n_stocks", type=int, default=N_STOCKS)
    parser.add_argument("--n_days", type=int, default=20,
                        help="Number of consecutive test days to evaluate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default="",
                        help="Save metrics to JSON file")
    parser.add_argument("--per_date_output", type=str, default="",
                        help="Save per-date results separately (for plotting)")
    args = parser.parse_args()

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, n_stocks={args.n_stocks}, n_days={args.n_days}")

    metrics = evaluate_windowed(args.gpt_ckpt, args.tokenizer, device,
                                n_stocks=args.n_stocks, n_days=args.n_days,
                                batch_size=args.batch_size, silent=False)

    # Save
    if args.output:
        # Strip per_date from main output (too verbose)
        output_clean = {k: v for k, v in metrics.items() if k != "per_date"}
        output_clean["gpt_ckpt"] = args.gpt_ckpt
        output_clean["tokenizer"] = args.tokenizer
        output_clean["n_days"] = args.n_days
        output_clean["n_stocks"] = args.n_stocks
        with open(args.output, "w") as f:
            json.dump(output_clean, f, indent=2)
        print(f"  Saved: {args.output}")

    if args.per_date_output and "per_date" in metrics:
        with open(args.per_date_output, "w") as f:
            json.dump(metrics["per_date"], f, indent=2)
        print(f"  Per-date results saved: {args.per_date_output}")

    return metrics


if __name__ == "__main__":
    main()
