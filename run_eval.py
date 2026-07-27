"""独立评估脚本：绕过 compute_windowed_metrics 的 bug，直接计算所有指标。

用法:
    python run_eval.py --ckpt experiments/04a-loss-ablation/gpt_focal.pt
    python run_eval.py --ckpt experiments/04a-loss-ablation/gpt_ce.pt
    python run_eval.py --ckpt experiments/04a-loss-ablation/gpt_focal.pt --n_stocks 500  # 快速
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from config import DataConfig, ModelConfig, set_global_seed
from data_processor import load_stocks, split_stocks
from model import load_tokenizer
from model.kronos_preview import KronosPreview
from eval_helpers import batched_gpt_eval_windowed, attach_close_prices

TOK_PATH = "experiments/02-tokenizer-tuning/tok_sweep_emb64_hid192.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(predictions):
    """直接从 predictions 列表计算所有指标，绕过 compute_windowed_metrics。"""
    if not predictions:
        return {"n_predictions": 0}

    pred_lrs = np.array([p["pred_logret"] for p in predictions])
    true_lrs = np.array([p["true_logret"] for p in predictions])
    pred_toks = np.array([p["coarse_id"] for p in predictions])
    base_closes = np.array([p.get("base_close", 0) for p in predictions])
    true_closes = np.array([p.get("true_close", 0) for p in predictions])

    # Per-date DA
    by_date = {}
    for p in predictions:
        dk = p.get("date_key", "unknown")
        by_date.setdefault(dk, []).append(p)

    per_date_da = []
    for dk, preds in sorted(by_date.items()):
        plr = np.array([p["pred_logret"] for p in preds])
        tlr = np.array([p["true_logret"] for p in preds])
        valid = ~(np.isnan(plr) | np.isnan(tlr))
        if valid.sum() < 5:
            continue
        da = float((np.sign(plr[valid]) == np.sign(tlr[valid])).mean())
        up_frac = float((tlr[valid] > 0).mean())
        baseline = max(up_frac, 1 - up_frac)
        per_date_da.append({"da": da, "baseline": baseline, "n": int(valid.sum())})

    if not per_date_da:
        return {"n_predictions": len(predictions), "error": "no valid dates"}

    avg_da = float(np.mean([d["da"] for d in per_date_da]))
    da_std = float(np.std([d["da"] for d in per_date_da]))
    avg_baseline = float(np.mean([d["baseline"] for d in per_date_da]))

    # RankIC (all predictions, exclude NaN)
    valid = ~(np.isnan(pred_lrs) | np.isnan(true_lrs))
    rank_ic = float(spearmanr(pred_lrs[valid], true_lrs[valid])[0]) if valid.sum() > 2 else 0.0
    rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic

    # Collapse / Unique
    unique_toks, counts = np.unique(pred_toks, return_counts=True)
    collapse_rate = float(counts.max() / max(len(pred_toks), 1))
    n_unique = int(len(unique_toks))

    # AmpRatio
    eps = 1e-8
    amp_ratio = float(np.mean(np.abs(pred_lrs)) / max(np.mean(np.abs(true_lrs)), eps))

    # MAPE (price space)
    has_close = (base_closes > 0) & (true_closes > 0)
    if has_close.sum() > 0:
        pred_prices = base_closes[has_close] * np.exp(pred_lrs[has_close].astype(np.float64))
        true_prices = true_closes[has_close]
        mape = float(np.mean(np.abs(pred_prices - true_prices) / np.maximum(np.abs(true_prices), eps))) * 100
        bl_mape = float(np.mean(np.abs(base_closes[has_close] - true_prices) / np.maximum(np.abs(true_prices), eps))) * 100
    else:
        mape = 0.0
        bl_mape = 0.0

    return {
        "avg_da_per_date": round(avg_da * 100, 2),
        "da_std": round(da_std * 100, 2),
        "avg_baseline_da": round(avg_baseline * 100, 2),
        "da_above_baseline": round((avg_da - avg_baseline) * 100, 2),
        "rank_ic": round(rank_ic, 4),
        "collapse_rate": round(collapse_rate * 100, 2),
        "n_unique_tokens": n_unique,
        "amp_ratio": round(amp_ratio, 4),
        "mape": round(mape, 2),
        "baseline_mape": round(bl_mape, 2),
        "n_dates": len(per_date_da),
        "n_predictions": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="GPT checkpoint path")
    parser.add_argument("--n_stocks", type=int, default=0, help="0=all")
    parser.add_argument("--n_days", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print(f"Loading tokenizer...")
    tok = load_tokenizer(TOK_PATH, device=DEVICE)

    print(f"Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = ModelConfig()
    mc.vocab_size = cfg.get("vocab_size", tok.vocab_coarse)
    mc.vocab_fine = cfg.get("vocab_fine", tok.bsq_fine.vocab_size)
    if "dim" in cfg:
        mc.dim = cfg["dim"]
    if "depth" in cfg:
        mc.depth = cfg["depth"]
    if "heads" in cfg:
        mc.heads = cfg["heads"]
    if "num_kv_heads" in cfg:
        mc.num_kv_heads = cfg["num_kv_heads"]
    model = KronosPreview(mc).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loading stocks...")
    stocks = load_stocks(max_stocks=args.n_stocks)
    _, _, test_stocks = split_stocks(stocks)
    attach_close_prices(test_stocks)
    print(f"  Test stocks: {len(test_stocks)}")

    print(f"Running eval (n_days={args.n_days}, batch_size={args.batch_size})...")
    t0 = time.time()
    predictions = batched_gpt_eval_windowed(
        model, tok, test_stocks, DEVICE,
        batch_size=args.batch_size, n_days=args.n_days)
    elapsed = time.time() - t0
    print(f"  {len(predictions)} predictions in {elapsed:.0f}s")

    print(f"Computing metrics...")
    metrics = compute_metrics(predictions)

    print(f"\n{'='*50}")
    print(f"  Results: {os.path.basename(args.ckpt)}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
