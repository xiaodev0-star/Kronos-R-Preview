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

def load_gpt(path, device, tokenizer=None):
    """Load a KronosPreview GPT model for evaluation, restoring config from checkpoint."""
    from config import ModelConfig
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Restore config from checkpoint
    cfg = ckpt.get("config", {})
    if "vocab_size" in cfg:
        ModelConfig.vocab_size = cfg["vocab_size"]
    if "vocab_fine" in cfg:
        ModelConfig.vocab_fine = cfg["vocab_fine"]
    elif tokenizer is not None:
        # Fallback: infer from tokenizer
        ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    if "dim" in cfg:
        ModelConfig.dim = cfg["dim"]
    if "depth" in cfg:
        ModelConfig.depth = cfg["depth"]
    if "heads" in cfg:
        ModelConfig.heads = cfg["heads"]
    if "num_kv_heads" in cfg:
        ModelConfig.num_kv_heads = cfg["num_kv_heads"]
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
        "day": day, "month": month, "year": year, "dates_dt": stock.get("dates_dt", None),
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
    vocab = tokenizer.vocab_coarse
    bos_id = vocab  # BOS = vocab_coarse (matches model's nn.Embedding(vocab_size + 2))

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
    """Decode a single coarse prediction + fine logits to a feature vector."""
    coarse_id = max(0, min(coarse_id, tokenizer.bsq_coarse.vocab_size - 1))
    fine_id = int(fine_logits_at_pos.argmax().item())
    fine_id = max(0, min(fine_id, tokenizer.bsq_fine.vocab_size - 1))
    pred_indices = torch.tensor([[[coarse_id, fine_id]]], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]


def decode_coarse_batch(coarse_ids, fine_logits_batch, tokenizer, device):
    """Batch decode: [B] coarse IDs + [B, V_fine] fine logits -> [B, 4] features."""
    fine_ids = fine_logits_batch.argmax(dim=-1).cpu().numpy()  # [B]
    coarse_ids_np = np.array(coarse_ids)
    B = len(coarse_ids_np)
    pred_indices = torch.zeros(B, 1, 2, dtype=torch.long, device=device)
    pred_indices[:, 0, 0] = torch.tensor(np.clip(coarse_ids_np, 0, tokenizer.bsq_coarse.vocab_size - 1), dtype=torch.long)
    pred_indices[:, 0, 1] = torch.tensor(np.clip(fine_ids, 0, tokenizer.bsq_fine.vocab_size - 1), dtype=torch.long)
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)  # [B, 1, 4]
    return pred_feat[:, 0, :].cpu().numpy()  # [B, 4]


# ============================================================================
# Batched GPT inference (eval acceleration)
# ============================================================================

def _prepare_stocks_batch(stocks, tokenizer, device):
    """Batch-tokenize all stocks. Tokenize on CPU to avoid GPU OOM."""
    all_arrays = []
    for stock in stocks:
        arrays = build_stock_arrays(stock)
        if arrays is not None:
            all_arrays.append(arrays)

    if not all_arrays:
        return []

    # Tokenize on CPU to avoid GPU memory spike
    cpu_device = torch.device("cpu")
    max_T = max(a["T_total"] for a in all_arrays)
    batch_np = np.zeros((len(all_arrays), max_T, 4), dtype=np.float32)
    lengths = []
    for j, a in enumerate(all_arrays):
        T = a["T_total"]
        batch_np[j, :T] = a["price_normed"][:T]
        lengths.append(T)

    # Process in chunks of 256 to avoid CPU memory spike
    chunk_size = 256
    all_idx_chunks = []
    for i in range(0, len(batch_np), chunk_size):
        chunk = torch.from_numpy(batch_np[i:i+chunk_size]).float()
        with torch.no_grad():
            idx, _ = tokenizer.encode(chunk.to(device))
        all_idx_chunks.append(idx.cpu().numpy())
    all_idx_np = np.concatenate(all_idx_chunks, axis=0)

    vocab = tokenizer.vocab_coarse
    bos_id = vocab
    results = []
    for j, arrays in enumerate(all_arrays):
        T = lengths[j]
        token_ids = all_idx_np[j, :T]
        day, month, year = arrays["day"], arrays["month"], arrays["year"]

        inp_ids = [bos_id] + token_ids.tolist()
        seq_len = len(inp_ids) - 1

        va_seq = np.concatenate([
            np.zeros((1, 2), dtype=np.float32),
            arrays["va_normed"][:T - 1],
        ], axis=0)

        # Store raw dates for date-aware evaluation (for per-date DA aggregation)
        dates_np = arrays.get("dates_dt", None)
        if dates_np is not None and len(dates_np) >= T:
            # Convert numpy datetime64 to string for JSON serialization
            dates_full = [str(d)[:10] if "T" in str(d) else str(d) for d in dates_np]
        else:
            dates_full = ["unknown"] * T

        # Build aligned dates matching inp_ids[:-1] positions for BOS/position reference
        dates_aligned = [dates_full[0]] + dates_full[:T-1]

        results.append({
            "inp_ids": inp_ids[:-1],
            "day": [int(day[0])] + day[:T-1].tolist(),
            "month": [int(month[0])] + month[:T-1].tolist(),
            "year": [int(year[0])] + year[:T-1].tolist(),
            "dates": dates_aligned,       # aligns with inp_ids[:-1] positions
            "dates_raw": dates_full,       # raw dates indexible by feature index
            "va": va_seq,
            "seq_len": seq_len,
            "test_pos": arrays["ci"],
            "p_mean": arrays["p_mean"], "p_std": arrays["p_std"],
            "feat": arrays["feat"], "close": arrays["close"],
        })
    return results


def _bucket_by_length(prepped, tolerance=100):
    """Group stocks by similar sequence length (±tolerance)."""
    valid = [p for p in prepped if p is not None]
    valid.sort(key=lambda x: x["seq_len"])
    buckets = []
    bucket = [valid[0]]
    for s in valid[1:]:
        if s["seq_len"] - bucket[0]["seq_len"] <= tolerance:
            bucket.append(s)
        else:
            buckets.append(bucket)
            bucket = [s]
    if bucket:
        buckets.append(bucket)
    return buckets


@torch.no_grad()
def batched_gpt_eval(gpt, tokenizer, test_stocks, device, batch_size=16, silent=False):
    """Batched GPT evaluation: predict cutoff+1 for each test stock.

    All three hot steps are batched: tokenize, forward, decode.
    """
    # Phase 1: batch tokenize all stocks in one GPU call
    if not silent:
        print(f"  Preparing {len(test_stocks)} stocks (batched tokenize)...")
    prepped = _prepare_stocks_batch(test_stocks, tokenizer, device)
    if not silent:
        print(f"  {len(prepped)} valid stocks")

    # Phase 2: bucket by length
    buckets = _bucket_by_length(prepped, tolerance=100)
    if not silent:
        print(f"  {len(buckets)} length buckets, batch_size={batch_size}")

    # Phase 3: batched forward + collect coarse IDs and fine logits
    all_coarse_ids = []
    all_fine_logits = []
    all_meta = []  # p_mean, p_std, feat, close, test_pos
    n_forward = 0

    for bucket in buckets:
        for batch_start in range(0, len(bucket), batch_size):
            batch = bucket[batch_start:batch_start + batch_size]
            B = len(batch)
            max_len = max(s["seq_len"] for s in batch)

            inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
            tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
            pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.zeros(B, max_len, max_len, dtype=torch.bool, device=device)
            va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

            for j, s in enumerate(batch):
                L = s["seq_len"]
                inp[j, :L] = torch.tensor(s["inp_ids"], dtype=torch.long)
                tids[j, :L, 0] = torch.tensor(s["day"], dtype=torch.long)
                tids[j, :L, 1] = torch.tensor(s["month"], dtype=torch.long)
                tids[j, :L, 2] = torch.tensor(s["year"], dtype=torch.long)
                va[j, :L] = torch.tensor(s["va"], dtype=torch.float32)
                mask[j, :L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))
                mask[j, L:, 0] = True  # padded positions attend to BOS (prevents NaN in SDPA)

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                coarse_logits, fine_logits = gpt(inp, tids, pos, mask, va_values=va)
            n_forward += 1

            coarse_cpu = coarse_logits.float().cpu()
            fine_cpu = fine_logits.float().cpu()
            vocab = tokenizer.vocab_coarse

            for j, s in enumerate(batch):
                p = s["test_pos"]
                if p < coarse_cpu.shape[1]:
                    cid = int(coarse_cpu[j, p, :vocab].argmax().item())
                    fl = fine_cpu[j, p] if p < fine_cpu.shape[1] else torch.zeros(tokenizer.bsq_fine.vocab_size)
                else:
                    cid = 0
                    fl = torch.zeros(tokenizer.bsq_fine.vocab_size)
                all_coarse_ids.append(cid)
                all_fine_logits.append(fl)
                all_meta.append({
                    "p_mean": s["p_mean"], "p_std": s["p_std"],
                    "feat": s["feat"], "close": s["close"], "test_pos": p,
                })

    if not silent:
        print(f"  {n_forward} forward passes")

    # Phase 4: batch decode all predictions in one GPU call
    if not silent:
        print(f"  Batch decoding {len(all_coarse_ids)} predictions...")
    fine_logits_tensor = torch.stack(all_fine_logits)  # [N, V_fine]
    decoded = decode_coarse_batch(all_coarse_ids, fine_logits_tensor, tokenizer, device)  # [N, 4]

    # Assemble results
    results = []
    for i, meta in enumerate(all_meta):
        p = meta["test_pos"]
        r = {
            "coarse_id": all_coarse_ids[i],
            "decoded_feat": decoded[i],  # [4] already decoded
            "true_logret": float(meta["feat"][p, 0]) if p < len(meta["feat"]) else 0.0,
            "base_close": float(meta["close"][p - 1]) if p - 1 >= 0 else float(meta["close"][0]),
            "true_close": float(meta["close"][p]) if p < len(meta["close"]) else 0.0,
            "pred_logret": float(decoded[i][0]) * meta["p_std"][0] + meta["p_mean"][0],
            **meta,
        }
        results.append(r)

    if not silent:
        print(f"  Done: {len(results)} predictions from {n_forward} forwards")
    return results


@torch.no_grad()
def batched_gpt_eval_windowed(gpt, tokenizer, test_stocks, device,
                              batch_size=16, n_days=20, silent=False):
    """Batched GPT evaluation with multi-day sliding window.

    For each test stock, predicts positions test_pos .. test_pos+n_days-1.
    Returns a list of dicts, one per prediction, with date info attached.

    Args:
        n_days: number of consecutive days to predict (default 20)
    """
    # Phase 1: batch tokenize
    if not silent:
        print(f"  Preparing {len(test_stocks)} stocks (batched tokenize)...")
    prepped = _prepare_stocks_batch(test_stocks, tokenizer, device)
    if not silent:
        print(f"  {len(prepped)} valid stocks, n_days={n_days}")

    # Filter stocks that have enough test data
    valid = [p for p in prepped if p["test_pos"] + n_days < p["seq_len"]]
    if not valid:
        return []
    if not silent:
        print(f"  {len(valid)} stocks with >= {n_days} test days")

    # Phase 2: bucket by length
    buckets = _bucket_by_length(valid, tolerance=100)
    if not silent:
        print(f"  {len(buckets)} length buckets, batch_size={batch_size}")

    # Phase 3: single forward pass, collect all positions
    predictions = []  # list of dicts with date_key, meta, etc.
    n_forward = 0
    vocab = tokenizer.vocab_coarse

    for bucket in buckets:
        batch_start = 0
        while batch_start < len(bucket):
            batch = bucket[batch_start:batch_start + batch_size]
            B = len(batch)
            max_len = max(s["seq_len"] for s in batch)

            try:
                inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
                tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
                pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
                mask = torch.zeros(B, max_len, max_len, dtype=torch.bool, device=device)
                va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

                for j, s in enumerate(batch):
                    L = s["seq_len"]
                    inp[j, :L] = torch.tensor(s["inp_ids"], dtype=torch.long)
                    tids[j, :L, 0] = torch.tensor(s["day"], dtype=torch.long)
                    tids[j, :L, 1] = torch.tensor(s["month"], dtype=torch.long)
                    tids[j, :L, 2] = torch.tensor(s["year"], dtype=torch.long)
                    va[j, :L] = torch.tensor(s["va"], dtype=torch.float32)
                    mask[j, :L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))
                    mask[j, L:, 0] = True

                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    coarse_logits, fine_logits = gpt(inp, tids, pos, mask, va_values=va)
                n_forward += 1

                coarse_cpu = coarse_logits.float().cpu()
                fine_cpu = fine_logits.float().cpu()

                for j, s in enumerate(batch):
                    p_start = s["test_pos"]
                    for offset in range(n_days):
                        p = p_start + offset
                        if p >= coarse_cpu.shape[1]:
                            break  # out of bounds for this batch item

                        cid = int(coarse_cpu[j, p, :vocab].argmax().item())
                        fl = fine_cpu[j, p] if p < fine_cpu.shape[1] else torch.zeros(tokenizer.bsq_fine.vocab_size)

                        # Date for this prediction's target: pred is for feat[p], so true is feat[p, 0]
                        date_key = s["dates_raw"][p] if p < len(s["dates_raw"]) else "unknown"

                        predictions.append({
                            "coarse_id": cid,
                            "fine_logits": fl,
                            "test_pos": p,
                            "date_key": date_key,
                            "day_offset": offset,
                            "p_mean": s["p_mean"], "p_std": s["p_std"],
                            "feat": s["feat"], "close": s["close"],
                        })

                batch_start += batch_size

            except torch.cuda.OutOfMemoryError:
                # OOM fallback: halve batch_size and retry; if already 1, skip this stock
                torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    if not silent:
                        print(f"  [OOM] Reducing eval batch_size to {batch_size} "
                              f"(seq_len={max_len})")
                    continue  # retry same batch_start with smaller batch
                else:
                    if not silent:
                        print(f"  [OOM] Skipping stock (seq_len={max_len}, bs=1)")
                    batch_start += 1

    if not silent:
        print(f"  {n_forward} forward passes, {len(predictions)} predictions")

    # Phase 4: batch decode all coarse IDs
    if not silent:
        print(f"  Batch decoding {len(predictions)} predictions...")
    fine_tensor = torch.stack([p["fine_logits"] for p in predictions])
    coarse_ids = [p["coarse_id"] for p in predictions]
    decoded = decode_coarse_batch(coarse_ids, fine_tensor, tokenizer, device)

    for i, pred in enumerate(predictions):
        pred["decoded_feat"] = decoded[i]
        pred["pred_logret"] = float(decoded[i][0]) * pred["p_std"][0] + pred["p_mean"][0]
        # Ground truth: pred is for feat[tp, 0] = log_ret[tp]
        tp = pred["test_pos"]
        pred["true_logret"] = float(pred["feat"][tp, 0]) if tp < len(pred["feat"]) else 0.0
        pred["base_close"] = float(pred["close"][tp - 1]) if tp - 1 >= 0 else float(pred["close"][0])
        pred["true_close"] = float(pred["close"][tp]) if tp < len(pred["close"]) else 0.0
        # Clean up heavy metadata to save memory
        del pred["fine_logits"], pred["feat"], pred["close"]

    return predictions


def compute_windowed_metrics(predictions):
    """Compute multi-day-averaged metrics from windowed predictions.

    Returns dict with per-date DA then cross-date aggregation.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from collections import defaultdict

    if not predictions:
        return {"avg_da_per_date": 0, "n_predictions": 0}

    # Group predictions by date
    by_date = defaultdict(list)
    for p in predictions:
        by_date[p["date_key"]].append(p)

    per_date = {}
    all_pred_lrs = []
    all_true_lrs = []
    all_pred_toks = []

    for date_key, group in sorted(by_date.items()):
        pred_lrs = np.array([g["pred_logret"] for g in group])
        true_lrs = np.array([g["true_logret"] for g in group])
        pred_toks = np.array([g["coarse_id"] for g in group], dtype=np.int64)
        eps = 1e-8

        if len(pred_lrs) < 5:
            continue

        da = float((np.sign(pred_lrs) == np.sign(true_lrs)).mean())

        # Always-up / always-down baselines for this date
        up_frac = float((true_lrs > 0).mean())
        down_frac = float((true_lrs < 0).mean())
        always_up_da = up_frac
        always_down_da = down_frac
        baseline_da = max(always_up_da, always_down_da)

        da_above_baseline = da - baseline_da

        # Baseline for "always the same sign" (more conservative)
        # If we predict the majority direction for *every* stock
        majority_dir = 1.0 if up_frac > down_frac else -1.0
        majority_da = up_frac if majority_dir == 1.0 else down_frac

        per_date[date_key] = {
            "da": float(da),
            "n": len(pred_lrs),
            "up_frac": float(up_frac),
            "always_up_da": float(always_up_da),
            "always_down_da": float(always_down_da),
            "baseline_da": float(baseline_da),
            "da_above_baseline": float(da_above_baseline),
            "majority_dir": float(majority_dir),
            "majority_da": float(majority_da),
        }

        all_pred_lrs.extend(pred_lrs.tolist())
        all_true_lrs.extend(true_lrs.tolist())
        all_pred_toks.extend(pred_toks.tolist())

    if not per_date:
        return {"avg_da_per_date": 0, "n_predictions": 0}

    # Cross-date aggregation
    das = np.array([v["da"] for v in per_date.values()])

    avg_da = float(das.mean())
    da_std = float(das.std()) if len(das) > 1 else 0.0
    da_values = list(das)
    da_vol = float(np.std(da_values)) if len(da_values) > 1 else 0.0

    # Aggregate baseline-relative metrics
    above_baseline_vals = [v["da_above_baseline"] for v in per_date.values()]
    avg_da_above_baseline = float(np.mean(above_baseline_vals))

    # Overall collapse/unique across ALL predictions
    all_pred_toks_arr = np.array(all_pred_toks)
    unique, counts = np.unique(all_pred_toks_arr, return_counts=True)
    collapse_rate = float(counts.max() / max(len(all_pred_toks_arr), 1))
    n_unique_tokens = int(len(unique))

    # Overall RankIC (across all predictions, not per-date)
    all_pred_lrs_arr = np.array(all_pred_lrs)
    all_true_lrs_arr = np.array(all_true_lrs)
    valid = ~np.isnan(all_pred_lrs_arr) & ~np.isnan(all_true_lrs_arr)
    rank_ic = float(spearmanr(all_pred_lrs_arr[valid], all_true_lrs_arr[valid])[0]) if valid.sum() > 2 else 0.0
    rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic

    # AmpRatio
    eps = 1e-8
    amp_ratio = float(np.mean(np.abs(all_pred_lrs_arr)) / max(np.mean(np.abs(all_true_lrs_arr)), eps))

    # Signal score: average DA above baseline across dates
    # Positive = better than always-predicting-majority
    signal_score = avg_da_above_baseline

    # Collapse penalty: reduce signal if diversity is low
    total_days = len(per_date)
    n_pred = len(predictions)

    return {
        "avg_da_per_date": avg_da,
        "da_std": da_std,
        "da_vol": da_vol,
        "n_dates": total_days,
        "n_predictions": n_pred,
        "avg_da_above_baseline": avg_da_above_baseline,
        "signal_score": signal_score,
        "collapse_rate": collapse_rate,
        "n_unique_tokens": n_unique_tokens,
        "rank_ic": rank_ic,
        "ampratio": amp_ratio,
        "per_date": per_date,
    }
