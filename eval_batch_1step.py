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
import pandas as pd
from glob import glob

from config import DataConfig, NormConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, document_normalize, _stock_cutoff_idx, stratified_split_stocks
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from reproducibility import set_global_seed

N_TEST_STOCKS = 30
SEED = 42
AMP_DTYPE = torch.bfloat16
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _cutoff_idx(stock):
    return int(np.searchsorted(stock["dates_dt"],
                               np.datetime64(pd.Timestamp(DataConfig.cutoff_date)), side="left"))


def _rolling_stats(features, window=NormConfig.lookback_window):
    T, D = features.shape
    cs = np.cumsum(features, axis=0)
    cs2 = np.cumsum(features ** 2, axis=0)
    idx = np.arange(T)
    starts = np.maximum(idx - window + 1, 0)
    counts = (idx - starts + 1).astype(np.float32)
    shifted = np.zeros_like(cs); shifted[1:] = cs[:-1]
    shifted2 = np.zeros_like(cs2); shifted2[1:] = cs2[:-1]
    mask_arr = (starts > 0).astype(np.float32)[:, None]
    win_sum = cs - shifted * mask_arr
    win_sum2 = cs2 - shifted2 * mask_arr
    means = win_sum / counts[:, None]
    var = win_sum2 / counts[:, None] - means ** 2
    stds = np.sqrt(np.maximum(var, 1e-08))
    return means.astype(np.float32), stds.astype(np.float32)


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
        feat = stock["features_raw"]       # FULL history: [T_total, 6]
        day, month, year = stock["day"], stock["month"], stock["year"]
        close = stock["close_prices"]      # FULL close prices
        ci = _cutoff_idx(stock)            # cutoff index

        # Full feature set for the entire history
        T_total = len(feat)
        if T_total < m + 10 or ci < m:
            continue

        # Test period length
        n_test = T_total - ci
        if n_test < 5:
            continue

        # ---- Normalize FULL history (4D OHLC only) ----
        price_feat = feat[:, :4]  # [T_total, 4]
        price_normed, _ = document_normalize(feat, cutoff_idx=ci)
        normed = price_normed  # [T_total, 4] normalized

        # ---- Tokenize FULL history ----
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()  # [T_total]

        # ---- Rolling stats from FULL history (for denormalization) ----
        # Document-level stats (fixed) for denormalization
        p_mean = price_feat[:ci].mean(axis=0)  # [4]
        p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)  # [4]

        # ---- Build full sequence: BOS + all tokens ----
        # We need the model to see everything up to each test position.
        # Feed the FULL sequence and extract predictions at test positions.
        N = T_total  # use all available tokens
        ids = [bos_id] + token_ids[:N].tolist()
        d_l = [day[0]] + day[:N].tolist()
        m_l = [month[0]] + month[:N].tolist()
        y_l = [year[0]] + year[:N].tolist()

        S = len(ids)
        # Input: all tokens except the last (we predict the next token)
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        tids = torch.stack([
            torch.tensor([d_l[:-1]], dtype=torch.long),
            torch.tensor([m_l[:-1]], dtype=torch.long),
            torch.tensor([y_l[:-1]], dtype=torch.long),
        ], dim=-1).to(device)
        pos = torch.arange(S - 1, device=device).unsqueeze(0)
        mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))

        # ---- Forward pass on FULL sequence ----
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc, lf, _ = model(inp, tids, pos, mask)

        # ---- Extract predictions at TEST positions only ----
        # Test positions: indices ci..T_total-1 in the original sequence
        # In the logits tensor (which has S-1 positions), position i predicts token at i+1
        # So to get prediction for original position ci, we need logit at position ci-1
        # (because logit at position ci-1 predicts the token at position ci)
        # Wait, let me re-think:
        #   ids = [BOS, tok_0, tok_1, ..., tok_{N-1}]
        #   inp = ids[:-1] = [BOS, tok_0, ..., tok_{N-2}]
        #   lc[0, i] predicts the token at position i+1 in ids
        #   So lc[0, ci-1] predicts tok_ci (the first test token)
        #   lc[0, ci] predicts tok_{ci+1}
        #   etc.

        # We want predictions for positions ci..T_total-1
        # lc[0, ci-1] predicts tok_ci → compare with true tok_ci
        # But the "true" log return at position ci is feat[ci, 0]
        # The predicted log return is decoded from the predicted token

        # Actually, for 1-step prediction, we want:
        #   At position t, predict the next price change (t → t+1)
        #   The model at position t predicts the token at position t+1
        #   So we need: for t in test positions, predict token at t+1

        # Test positions: t = ci, ci+1, ..., T_total-2 (predicting t+1)
        # In logits: lc[0, t] predicts token at position t+1
        # We want: lc[0, ci], lc[0, ci+1], ..., lc[0, T_total-2]
        # These predict tokens at positions ci+1, ci+2, ..., T_total-1

        # But wait, the true log return at position t is feat[t, 0]
        # The "next price" prediction at position t is about going from t to t+1
        # So: predicted lr at position t = decode(lc[0, t].argmax())
        #      true lr at position t = feat[t+1, 0] (the actual return from t to t+1)
        # Wait no... feat[t, 0] IS the log return AT position t (= log(close[t]/close[t-1]))

        # For standard 1-step: at position t, predict the return from t to t+1
        # The model's output at position t predicts the NEXT token (position t+1)
        # The token at position t+1 encodes the return at position t+1 = log(close[t+1]/close[t])
        # So: predicted_return[t] = decode(argmax(lc[0, t]))
        #      true_return[t] = feat[t+1, 0]

        # Test positions: t = ci, ci+1, ..., T_total-2
        test_start = ci
        test_end = T_total - 2  # last position where we can predict t+1
        if test_end <= test_start:
            continue

        n_pred = test_end - test_start + 1
        pred_c = lc[0, test_start:test_end + 1].argmax(dim=-1)
        all_pred_toks.append(pred_c.cpu().numpy())

        # Experiment A: 2-level joint argmax with lf.std guard for backward-compat
        lf_std = lf[0, test_start:test_end + 1].float().std().item()
        if lf_std < 0.1:
            # Old behavior: replicate coarse to both levels
            pred_indices = pred_c.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous()
        else:
            pred_f = lf[0, test_start:test_end + 1].argmax(dim=-1)
            pred_indices = torch.stack([pred_c, pred_f], dim=-1).unsqueeze(0)

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
    csv_map = {os.path.basename(f).split(".")[0]: f for f in sorted(glob("dataset/*.csv"))}
    for s in test_stocks_all:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            prev = df["close"].shift(1)
            df["log_ret"] = np.log(df["close"] / prev).replace([np.inf, -np.inf], np.nan)
            df = df.dropna().reset_index(drop=True)
            s["close_prices"] = df["close"].values.astype(np.float64)
        else:
            lr = s["features_raw"][:, 0]
            s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)

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

    # Baseline model
    baseline_path = "checkpoints/v2_model.pt"
    if os.path.exists(baseline_path):
        checkpoints.append({"name": "baseline_v2", "path": os.path.abspath(baseline_path), "val_loss": 0})

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
            model = load_model(path, device)
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
