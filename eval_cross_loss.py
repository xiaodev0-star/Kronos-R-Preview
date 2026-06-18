"""Cross-Loss Fair Evaluation: Compare models across different loss functions.

Different losses (CE, Focal, Het) produce different val_loss scales, so val_loss
is NOT a fair comparison metric. This script uses downstream metrics instead:
  - 1-Step DA (Directional Accuracy)
  - 1-Step MAPE
  - Collapse Rate (token distribution concentration)
  - Token Diversity (unique tokens predicted)

Usage:
    python eval_cross_loss.py                           # Evaluate all checkpoints
    python eval_cross_loss.py --dirs checkpoints/hpo_fast_phase2
    python eval_cross_loss.py --checkpoints path1.pt path2.pt
    python eval_cross_loss.py --top_n 3                 # Top 3 per loss family
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from glob import glob

from config import NormConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, stratified_split_stocks
from reproducibility import set_global_seed
from eval_helpers import (
    build_stock_arrays, build_gpt_eval_inputs, load_tokenizer, load_gpt, attach_close_prices,
    _cutoff_idx,
)

N_TEST_STOCKS = 30
SEED = 42
AMP_DTYPE = torch.bfloat16
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@torch.no_grad()
def eval_1step(model, tokenizer, test_stocks, device):
    """1-step prediction evaluation — returns DA, MAPE, collapse_rate, etc."""
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab
    m = NormConfig.min_lookback

    stock_mape, stock_da, stock_baseline = [], [], []
    stock_ampratio = []
    all_pred_toks = []

    for si, stock in enumerate(test_stocks):
        arrays = build_stock_arrays(stock)
        if arrays is None:
            continue
        ci = arrays["ci"]
        T_total = arrays["T_total"]
        m = NormConfig.min_lookback
        n_test = T_total - ci
        if n_test < 5:
            continue

        # Build GPT inputs (uses real va_values, not zeros — see eval_helpers.py)
        with torch.no_grad():
            inputs = build_gpt_eval_inputs(arrays, tokenizer, device)

        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc, _ = model(inputs["inp"], inputs["tids"], inputs["pos"],
                          inputs["mask"], va_values=inputs["va_values"])

        p_mean = arrays["p_mean"]
        p_std = arrays["p_std"]
        feat = arrays["feat"]
        close = arrays["close"]
        test_start = inputs["test_start"]
        test_end = inputs["test_end"]
        if test_end <= test_start:
            continue
        n_pred = test_end - test_start + 1

        pred_c = lc[0, test_start:test_end + 1].argmax(dim=-1)
        all_pred_toks.append(pred_c.cpu().numpy())

        # Replicate coarse id to both tokenizer levels for decoding
        pred_indices = pred_c.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous()
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()

        pred_lr = pred_feat[:, 0] * p_std[0] + p_mean[0]
        true_lr = feat[test_start + 1:test_end + 2, 0]

        base_close = close[test_start:test_end + 1]
        pred_close = base_close * np.exp(pred_lr.astype(np.float64))
        true_close = close[test_start + 1:test_end + 2]

        eps = 1e-8
        mape_pt = np.abs(pred_close - true_close) / (np.abs(true_close) + eps) * 100
        da_pt = (np.sign(pred_lr) == np.sign(true_lr)).astype(float)

        bl = float(np.mean(np.abs(base_close - true_close) / (np.abs(true_close) + eps)) * 100)

        # AmpRatio: predicted amplitude / true amplitude
        pred_amp = float(np.mean(np.abs(pred_lr)))
        true_amp = float(np.mean(np.abs(true_lr)))
        ampratio = pred_amp / max(true_amp, 1e-8)

        stock_mape.append(float(np.mean(mape_pt)))
        stock_da.append(float(np.mean(da_pt)))
        stock_baseline.append(bl)
        stock_ampratio.append(ampratio)

    # Compute collapse metrics
    all_toks = np.concatenate(all_pred_toks) if all_pred_toks else np.array([])
    total = len(all_toks)
    if total > 0:
        unique, counts = np.unique(all_toks, return_counts=True)
        collapse_rate = counts.max() / total
        n_unique = len(unique)
        # Top-3 concentration
        top3_count = np.sort(counts)[-3:].sum() if len(counts) >= 3 else counts.sum()
        top3_rate = top3_count / total
    else:
        collapse_rate, n_unique, top3_rate = 0.0, 0, 0.0

    return {
        "mape": float(np.mean(stock_mape)) if stock_mape else 0,
        "da": float(np.mean(stock_da)) if stock_da else 0,
        "ampratio": float(np.mean(stock_ampratio)) if stock_ampratio else 0,
        "baseline_mape": float(np.mean(stock_baseline)) if stock_baseline else 0,
        "collapse_rate": float(collapse_rate),
        "top3_rate": float(top3_rate),
        "n_unique_tokens": int(n_unique),
        "n_stocks": len(stock_mape),
        "total_predictions": int(total),
    }


def discover_checkpoints(dirs, checkpoint_list):
    """Discover checkpoints from directories and/or explicit paths."""
    checkpoints = []

    # Explicit paths
    for path in checkpoint_list:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                # Default: only completed checkpoints. KRONOS_FORCE_EVAL=1 overrides.
                if not ckpt.get("completed", False) and not os.environ.get("KRONOS_FORCE_EVAL"):
                    continue
                checkpoints.append({
                    "name": os.path.basename(path).replace(".pt", ""),
                    "path": os.path.abspath(path),
                    "val_loss": ckpt.get("val_loss", float("inf")),
                    "loss_type": ckpt.get("loss_type", "unknown"),
                    "gamma": ckpt.get("gamma", 0),
                    "heteroscedastic": ckpt.get("heteroscedastic", False),
                    "het_weight": ckpt.get("het_weight", 0),
                    "collapse_rate": ckpt.get("collapse_rate", 0),
                })
            except Exception as e:
                print(f"  Skip {path}: {e}")

    # Scan directories
    for d in dirs:
        for f in sorted(glob(os.path.join(d, "*.pt"))):
            if f.endswith(".ckpt") or "_override_" in f:
                continue
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                if ckpt.get("completed", False):
                    checkpoints.append({
                        "name": ckpt.get("tag", os.path.basename(f).replace(".pt", "")),
                        "path": os.path.abspath(f),
                        "val_loss": ckpt.get("val_loss", float("inf")),
                        "loss_type": ckpt.get("loss_type", "unknown"),
                        "gamma": ckpt.get("gamma", 0),
                        "heteroscedastic": ckpt.get("heteroscedastic", False),
                        "het_weight": ckpt.get("het_weight", 0),
                        "collapse_rate": ckpt.get("collapse_rate", 0),
                    })
            except Exception:
                pass

    return checkpoints


def get_loss_family(info):
    """Determine loss family from checkpoint metadata."""
    if info.get("heteroscedastic"):
        if info.get("loss_type") == "focal":
            return "focal_het"
        return "ce_het"
    if info.get("loss_type") == "focal":
        return "focal"
    return "ce"


def main():
    parser = argparse.ArgumentParser(description="Cross-Loss Fair Evaluation")
    parser.add_argument("--dirs", nargs="*", default=[],
                        help="Directories to scan for checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Explicit checkpoint paths")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Top N per loss family (by val_loss)")
    parser.add_argument("--n_stocks", type=int, default=N_TEST_STOCKS)
    parser.add_argument("--output", type=str, default="eval_cross_loss_results.json")
    args = parser.parse_args()

    # Default directories
    if not args.dirs and not args.checkpoints:
        args.dirs = [
            "checkpoints/hpo_fast_phase2",
            "checkpoints/hpo_v4_het_refined",
            "checkpoints/hpo_v3_het",
        ]
        # Also include baseline
        if os.path.exists("checkpoints/v2_model.pt"):
            args.checkpoints.append("checkpoints/v2_model.pt")

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and test stocks
    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(TrainingConfig.tokenizer_path, device)

    print("Loading test stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)

    # Attach close_prices
    attach_close_prices(test_stocks_all)

    rng = np.random.RandomState(SEED)
    stratified_n = int(os.environ.get("KRONOS_STRATIFIED_N", "0") or "0")
    if stratified_n > 0:
        test_stocks = stratified_split_stocks(test_stocks_all, n=stratified_n, seed=SEED)
        print(f"Test stocks (stratified): {len(test_stocks)} (n={stratified_n}, seed={SEED})")
    else:
        indices = rng.choice(len(test_stocks_all), min(args.n_stocks, len(test_stocks_all)), replace=False)
        test_stocks = [test_stocks_all[i] for i in sorted(indices)]
        print(f"Test stocks: {len(test_stocks)}")

    # Discover checkpoints
    all_checkpoints = discover_checkpoints(args.dirs, args.checkpoints)
    print(f"Discovered {len(all_checkpoints)} completed checkpoints")

    if not all_checkpoints:
        print("No checkpoints found!")
        return

    # Group by family and select top N
    families = {}
    for ckpt in all_checkpoints:
        family = get_loss_family(ckpt)
        ckpt["family"] = family
        families.setdefault(family, []).append(ckpt)

    selected = []
    for family, ckpts in sorted(families.items()):
        ckpts.sort(key=lambda x: x["val_loss"])
        top = ckpts[:args.top_n]
        selected.extend(top)
        print(f"  {family}: {len(ckpts)} found, selected top {len(top)}")

    print(f"\nEvaluating {len(selected)} checkpoints ...")
    print("=" * 100)

    # Evaluate
    results = []
    for i, ckpt_info in enumerate(selected):
        name = ckpt_info["name"]
        path = ckpt_info["path"]
        family = ckpt_info["family"]
        print(f"\n[{i+1}/{len(selected)}] {name} [{family}]")

        try:
            model = load_gpt(path, device)
            res = eval_1step(model, tokenizer, test_stocks, device)
            res["name"] = name
            res["path"] = path
            res["family"] = family
            res["val_loss"] = ckpt_info["val_loss"]
            res["loss_type"] = ckpt_info.get("loss_type", "unknown")
            res["gamma"] = ckpt_info.get("gamma", 0)
            res["het_weight"] = ckpt_info.get("het_weight", 0)
            results.append(res)

            print(f"  DA={res['da']*100:.2f}%  MAPE={res['mape']:.2f}%  "
                  f"AmpRatio={res.get('ampratio',0):.3f}x  "
                  f"Collapse={res['collapse_rate']*100:.1f}%  Unique={res['n_unique_tokens']}")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # === CROSS-LOSS COMPARISON TABLE ===
    print("\n" + "=" * 130)
    print("  CROSS-LOSS FAIR COMPARISON (Anti-Collapse Priority)")
    print("  Sort: |AmpRatio-1| → MAPE → DA")
    print("=" * 130)
    print(f"  {'Name':<30} {'Family':<10} {'AmpRatio':>9} {'DA':>8} {'MAPE':>8} {'Baseline':>9} "
          f"{'Collapse':>9} {'Unique':>7}")
    print("  " + "-" * 120)

    # 3-variable sort: |AmpRatio-1| (ascending) → MAPE (ascending) → DA (descending)
    results.sort(key=lambda x: (abs(x.get("ampratio", 0) - 1.0),
                                 x.get("mape", 999),
                                 -x.get("da", 0)))
    for r in results:
        print(f"  {r['name']:<30} {r['family']:<10} {r.get('ampratio',0):>8.3f}x "
              f"{r['da']*100:>7.2f}% {r['mape']:>7.2f}% "
              f"{r['baseline_mape']:>8.2f}% {r['collapse_rate']*100:>8.1f}% "
              f"{r['n_unique_tokens']:>6}")

    # Best per family (by |AmpRatio-1|)
    print(f"\n  Best per family (by |AmpRatio-1| → MAPE → DA):")
    for family in sorted(set(r["family"] for r in results)):
        fam_results = [r for r in results if r["family"] == family]
        best = min(fam_results, key=lambda x: (abs(x.get("ampratio", 0) - 1.0),
                                                x.get("mape", 999)))
        print(f"    {family:<12}: {best['name']} AR={best.get('ampratio',0):.3f}x "
              f"DA={best['da']*100:.2f}% MAPE={best['mape']:.2f}%")

    # Overall winner by 3-variable sort
    if results:
        winner = results[0]  # Already sorted by |AmpRatio-1| → MAPE → DA
        print(f"\n  >>> WINNER: {winner['name']} [{winner['family']}]")
        print(f"     AmpRatio={winner.get('ampratio',0):.3f}x  "
              f"DA={winner['da']*100:.2f}%  MAPE={winner['mape']:.2f}%  "
              f"Collapse={winner['collapse_rate']*100:.1f}%")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
