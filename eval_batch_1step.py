"""Batch 1-step evaluation with FULL historical context.
Feeds the model the entire stock history (train+test), predicts at test positions.

Usage:
    python eval_batch_1step.py
"""
import os, sys, json, warnings
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
def eval_1step_full(model, tokenizer, test_stocks, device):
    """1-step prediction with FULL historical context.

    The model sees the entire stock history (train + test period),
    and we only evaluate predictions at test-period positions.
    """
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab
    m = NormConfig.min_lookback

    stock_mape, stock_da, stock_baseline = [], [], []
    all_pred_toks = []

    for si, stock in enumerate(test_stocks):
        symbol = stock["symbol"]
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

        # ---- Forward pass on FULL sequence ----
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc, _ = model(inputs["inp"], inputs["tids"], inputs["pos"],
                          inputs["mask"], va_values=inputs["va_values"])
        # Extract the precomputed rollouts from `inputs`
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

        # Denormalize: pred_lr = pred_feat[:, 0] * rstd + rmean
        # Use rolling stats at position t+1 (the position whose return we predicted)
        pred_lr = pred_feat[:, 0] * p_std[0] + p_mean[0]
        # True log return at position t+1
        true_lr = feat[test_start + 1:test_end + 2, 0]

        # ---- Convert to price space ----
        base_close = close[test_start:test_end + 1]  # close at position t
        pred_close = base_close * np.exp(pred_lr.astype(np.float64))
        true_close = close[test_start + 1:test_end + 2]  # close at position t+1

        eps = 1e-8
        mape_pt = np.abs(pred_close - true_close) / (np.abs(true_close) + eps) * 100
        da_pt = (np.sign(pred_lr) == np.sign(true_lr)).astype(float)

        # Baseline: predict zero return
        bl = float(np.mean(np.abs(base_close - true_close) / (np.abs(true_close) + eps)) * 100)

        stock_mape.append(float(np.mean(mape_pt)))
        stock_da.append(float(np.mean(da_pt)))
        stock_baseline.append(bl)

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(test_stocks)}] {symbol}: DA={np.mean(da_pt)*100:.1f}% "
                  f"MAPE={np.mean(mape_pt):.2f}% (n_pred={n_pred})")

    # Zero-collapse: dominant token fraction
    all_toks = np.concatenate(all_pred_toks) if all_pred_toks else np.array([])
    total = len(all_toks)
    if total > 0:
        unique, counts = np.unique(all_toks, return_counts=True)
        top_count = counts.max()
        top_token = unique[np.argmax(counts)]
        collapse_rate = top_count / total
        n_unique = len(unique)
    else:
        collapse_rate = 0.0
        n_unique = 0
        top_token = -1

    return {
        "mape": float(np.mean(stock_mape)) if stock_mape else 0,
        "da": float(np.mean(stock_da)) if stock_da else 0,
        "baseline_mape": float(np.mean(stock_baseline)) if stock_baseline else 0,
        "collapse_rate": float(collapse_rate),
        "n_unique_tokens": int(n_unique),
        "top_token": int(top_token),
        "n_stocks": len(stock_mape),
        "total_predictions": int(total),
    }


def main():
    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(TrainingConfig.tokenizer_path, device)

    print("Loading stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)
    print(f"Total test stocks: {len(test_stocks_all)}")

    # Attach close_prices
    print("Attaching close prices ...")
    attach_close_prices(test_stocks_all)

    # Select test stocks (30 fixed OR stratified_n via env var)
    stratified_n = int(os.environ.get("KRONOS_STRATIFIED_N", "0") or "0")
    if stratified_n > 0:
        test_stocks = stratified_split_stocks(test_stocks_all, n=stratified_n, seed=SEED)
        print(f"Selected {len(test_stocks)} stocks via stratified_split_stocks (n={stratified_n}, seed={SEED})")
    else:
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(test_stocks_all), min(N_TEST_STOCKS, len(test_stocks_all)), replace=False)
        test_stocks = [test_stocks_all[i] for i in sorted(indices)]
        print(f"Selected {len(test_stocks)} stocks for evaluation (seed={SEED})")

    # Show context info
    sample = test_stocks[0]
    ci = _cutoff_idx(sample)
    print(f"Example: {sample['symbol']} total={len(sample['features_raw'])} days, "
          f"history={ci}, test={len(sample['features_raw'])-ci}")

    # Collect checkpoints
    checkpoints = []

    # Baseline model — look in conventional locations.
    # expA_v2_hpo.pt is the HPO 2026-06-18 best (phase3_t000, DA 48.12% with V2).
    for cand in ["checkpoints/expA_v2_hpo.pt", "checkpoints/expA_v2.pt",
                 "checkpoints/v2_model.pt", "checkpoints/baseline_v2.pt"]:
        if os.path.exists(cand):
            checkpoints.append({"name": "baseline_expA_v2", "path": os.path.abspath(cand), "val_loss": 0})
            break

    # v3 checkpoints
    for f in sorted(glob("checkpoints/hpo_v3_het/het_t*.pt")):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            if ckpt.get("completed", False):
                checkpoints.append({
                    "name": f"v3_{ckpt.get('tag', os.path.basename(f))}",
                    "path": os.path.abspath(f),
                    "val_loss": ckpt.get("val_loss", 0),
                })
        except:
            pass

    # v4 checkpoints
    for f in sorted(glob("checkpoints/hpo_v4_het_refined/v4_t*.pt")):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            if ckpt.get("completed", False):
                checkpoints.append({
                    "name": f"v4_{ckpt.get('tag', os.path.basename(f))}",
                    "path": os.path.abspath(f),
                    "val_loss": ckpt.get("val_loss", 0),
                })
        except:
            pass

    # Select top checkpoints by val_loss to save time (max 8)
    checkpoints.sort(key=lambda x: x["val_loss"] if x["val_loss"] > 0 else 999)
    # Always include baseline + top 7
    selected = checkpoints[:8]
    print(f"\nEvaluating {len(selected)} checkpoints (top by val_loss)")
    print("=" * 90)

    results = []
    for i, ckpt_info in enumerate(selected):
        name = ckpt_info["name"]
        path = ckpt_info["path"]
        print(f"\n[{i+1}/{len(selected)}] {name}")
        print(f"  Path: {path}")

        try:
            model = load_gpt(path, device)
            res = eval_1step_full(model, tokenizer, test_stocks, device)
            res["name"] = name
            res["path"] = path
            res["val_loss"] = ckpt_info["val_loss"]
            results.append(res)

            print(f"  >>> DA={res['da']*100:.2f}%  MAPE={res['mape']:.2f}%  "
                  f"Baseline={res['baseline_mape']:.2f}%  "
                  f"Collapse={res['collapse_rate']*100:.1f}%  "
                  f"Unique={res['n_unique_tokens']}")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison table
    print("\n" + "=" * 100)
    print("  1-STEP EVALUATION WITH FULL HISTORICAL CONTEXT (30 test stocks)")
    print("=" * 100)
    print(f"  {'Name':<28} {'ValLoss':>8} {'DA':>8} {'MAPE':>8} {'Baseline':>9} {'ΔMAPE':>8} {'Collapse':>9} {'Unique':>7}")
    print("  " + "-" * 95)

    results.sort(key=lambda x: x["da"], reverse=True)
    for r in results:
        delta = r["mape"] - r["baseline_mape"]
        print(f"  {r['name']:<28} {r['val_loss']:>8.4f} {r['da']*100:>7.2f}% {r['mape']:>7.2f}% "
              f"{r['baseline_mape']:>8.2f}% {delta:>+7.2f}% {r['collapse_rate']*100:>8.1f}% {r['n_unique_tokens']:>6}")

    if results:
        best_da = max(results, key=lambda x: x["da"])
        best_mape = min(results, key=lambda x: x["mape"])
        print(f"\n  Best DA:     {best_da['name']} ({best_da['da']*100:.2f}%)")
        print(f"  Best MAPE:   {best_mape['name']} ({best_mape['mape']:.2f}%)")
        print(f"  Baseline:    {results[0]['baseline_mape']:.2f}%")

    out_path = "eval_batch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
