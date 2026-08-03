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
    if "ffn_multiplier" in cfg:
        ModelConfig.ffn_multiplier = cfg["ffn_multiplier"]
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


# ============================================================================
# Data preparation
# ============================================================================

def attach_close_prices(test_stocks):
    """Attach date-aligned ``close_prices`` [T] float64 to each stock.

    ``features_raw`` loses the first CSV row when the initial log return is
    undefined, so copying the raw close column positionally introduces a
    one-day MAPE shift. Align by date instead. Numeric symbols are indexed both
    with and without leading zeroes; unresolved/incomplete series fall back to
    a scale-free close path reconstructed from log returns.
    """
    csv_map = {}
    for f in sorted(glob("dataset/*.csv")):
        stem = os.path.basename(f).split(".")[0]
        csv_map[stem] = f
        if stem.isdigit():
            csv_map.setdefault(str(int(stem)), f)
    for s in test_stocks:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (df.dropna(subset=["date", "close"])
                    .sort_values("date")
                    .drop_duplicates("date", keep="last"))
            close_by_date = df.set_index("date")["close"]
            target_dates = pd.to_datetime(s.get("dates_dt", []), errors="coerce")
            aligned = close_by_date.reindex(target_dates).to_numpy(dtype=np.float64)
            if len(aligned) == len(s["features_raw"]) and np.isfinite(aligned).all():
                s["close_prices"] = aligned
                continue

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
        "symbol": str(stock.get("symbol", "unknown")),
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
    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        arrays["va_normed"][:N - 1],
    ], axis=0)
    va = torch.tensor(va_seq, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        # Right padding is always after every real query.  Passing no explicit
        # mask lets SDPA use its native causal path without allocating [N, N].
        "inp": inp, "tids": tids, "pos": pos, "mask": None, "va_values": va,
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
    fine_logits = torch.as_tensor(fine_logits_batch, device=device)
    fine_ids = fine_logits.float().argmax(dim=-1)
    decoded = decode_code_ids_tensor(
        coarse_ids, fine_ids, tokenizer, device
    )
    return decoded.cpu().numpy()


@torch.no_grad()
def decode_code_ids_tensor(coarse_ids, fine_ids, tokenizer, device):
    """Decode already-selected coarse/fine IDs and keep the result on device."""
    coarse = torch.as_tensor(
        coarse_ids, dtype=torch.long, device=device
    ).clamp(0, tokenizer.bsq_coarse.vocab_size - 1)
    fine = torch.as_tensor(
        fine_ids, dtype=torch.long, device=device
    ).clamp(0, tokenizer.bsq_fine.vocab_size - 1)
    pred_indices = torch.stack((coarse, fine), dim=-1).unsqueeze(1)
    return tokenizer.decode_all(pred_indices)[:, 0, :]


@torch.no_grad()
def gather_prediction_ids(
    coarse_logits,
    fine_logits,
    row_indices,
    positions,
    tokenizer,
    device,
):
    """Gather selected coarse/fine IDs without copying full logits to CPU.

    ``row_indices`` and ``positions`` are aligned flattened vectors.  The
    historical final-position behaviour is retained: if no fine logit exists,
    fine ID zero is used (argmax of the former all-zero fallback vector).
    """
    rows = torch.as_tensor(row_indices, dtype=torch.long, device=device)
    pos = torch.as_tensor(positions, dtype=torch.long, device=device)
    selected_coarse = coarse_logits[
        rows, pos, : tokenizer.vocab_coarse
    ].float()
    coarse_ids = selected_coarse.argmax(dim=-1)

    valid_fine = pos < fine_logits.shape[1]
    safe_fine_positions = pos.clamp(max=fine_logits.shape[1] - 1)
    gathered_fine_ids = (
        fine_logits[rows, safe_fine_positions]
        .float()
        .argmax(dim=-1)
    )
    fine_ids = torch.where(
        valid_fine, gathered_fine_ids, torch.zeros_like(gathered_fine_ids)
    )
    return coarse_ids, fine_ids


@torch.no_grad()
def predict_selected_ids(model, input_ids, time_ids, position_ids, va_values,
                         row_indices, positions, tokenizer, device):
    """Coarse/fine IDs at selected positions, without projecting every position.

    Same contract and the same outputs as running the full forward and calling
    ``gather_prediction_ids`` on it, including the rule that a position with no
    fine logit (the last one) yields fine ID zero.  Models without a selective
    forward fall back to the full path automatically.
    """
    rows = torch.as_tensor(row_indices, dtype=torch.long)
    pos = torch.as_tensor(positions, dtype=torch.long)
    selective = getattr(model, "forward_selected", None)
    if selective is None:
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE,
                                enabled=device.type == "cuda"):
            coarse_logits, fine_logits = model(
                input_ids, time_ids, position_ids, None, va_values=va_values)
        return gather_prediction_ids(
            coarse_logits, fine_logits, rows, pos, tokenizer, device)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
        coarse_logits, fine_logits = selective(
            input_ids, time_ids, position_ids, rows, pos, va_values=va_values)
    coarse_ids = coarse_logits[:, : tokenizer.vocab_coarse].float().argmax(dim=-1)
    # The full forward's fine head spans N-1 positions; the final position has no
    # fine logit and historically resolves to ID zero.
    n_positions = input_ids.shape[-1]
    valid_fine = (pos < n_positions - 1).to(device)
    fine_ids = fine_logits.float().argmax(dim=-1)
    return coarse_ids, torch.where(valid_fine, fine_ids, torch.zeros_like(fine_ids))


@torch.no_grad()
def gather_decode_positions(
    coarse_logits,
    fine_logits,
    row_indices,
    positions,
    tokenizer,
    device,
):
    """Gather and immediately decode positions with one host transfer."""
    coarse_ids, fine_ids = gather_prediction_ids(
        coarse_logits,
        fine_logits,
        row_indices,
        positions,
        tokenizer,
        device,
    )
    decoded = decode_code_ids_tensor(
        coarse_ids, fine_ids, tokenizer, device
    )
    # Coarse IDs are exactly representable in float32 for all project vocabs.
    # Packing both outputs produces one device-to-host transfer per batch.
    return torch.cat((coarse_ids.float().unsqueeze(1), decoded), dim=1)


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
    N = len(all_arrays)

    # Pre-allocate output directly (avoids building a list of chunks).  Keep
    # both hierarchy levels: most inference paths only consume coarse IDs, but
    # capacity experiments need the aligned fine targets to measure use of the
    # complete joint codebook.
    all_idx_np = np.zeros((N, max_T), dtype=np.int32)
    all_fine_idx_np = np.zeros((N, max_T), dtype=np.int16)

    # Process in chunks to avoid GPU memory spike
    chunk_size = 64  # smaller chunks = less peak memory
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        # Build chunk tensor from stocks one-by-one (avoid large batch_np)
        chunk_len = end - i
        chunk_np = np.zeros((chunk_len, max_T, 4), dtype=np.float32)
        for k, a in enumerate(all_arrays[i:end]):
            T = a["T_total"]
            chunk_np[k, :T] = a["price_normed"][:T]
        chunk_t = torch.from_numpy(chunk_np)
        with torch.no_grad():
            all_idx = tokenizer.encode_all(chunk_t.to(device))
        all_idx_host = all_idx.cpu().numpy()
        all_idx_np[i:end, :] = all_idx_host[..., 0].astype(np.int32)
        all_fine_idx_np[i:end, :] = all_idx_host[..., 1].astype(np.int16)
        del chunk_np, chunk_t, all_idx, all_idx_host  # free immediately

    vocab = tokenizer.vocab_coarse
    bos_id = vocab
    results = []
    for j, arrays in enumerate(all_arrays):
        T = arrays["T_total"]
        token_ids = all_idx_np[j, :T]
        fine_token_ids = all_fine_idx_np[j, :T]
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
            "symbol": arrays["symbol"],
            "inp_ids": inp_ids[:-1],
            # Raw feature-aligned targets. At selected causal position p both
            # hierarchy levels target these arrays at p. Keeping coarse targets
            # explicitly also preserves the final predictable token, which is
            # intentionally absent from ``inp_ids[:-1]``.
            "coarse_token_ids": token_ids,
            "fine_token_ids": fine_token_ids,
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

    # Phase 3: batched forward and compact position-ID gather
    all_coarse_ids = []
    all_fine_ids = []
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
            va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

            for j, s in enumerate(batch):
                length = s["seq_len"]
                inp[j, :length] = torch.tensor(
                    s["inp_ids"][:length], dtype=torch.long
                )
                tids[j, :length, 0] = torch.tensor(
                    s["day"][:length], dtype=torch.long
                )
                tids[j, :length, 1] = torch.tensor(
                    s["month"][:length], dtype=torch.long
                )
                tids[j, :length, 2] = torch.tensor(
                    s["year"][:length], dtype=torch.long
                )
                va[j, :length] = torch.tensor(
                    s["va"][:length], dtype=torch.float32
                )

            selected_rows = []
            selected_positions = []
            for j, s in enumerate(batch):
                p = s["test_pos"]
                if p < s["seq_len"]:
                    selected_rows.append(j)
                    selected_positions.append(p)
                    all_meta.append({
                        "p_mean": s["p_mean"], "p_std": s["p_std"],
                        "feat": s["feat"], "close": s["close"], "test_pos": p,
                    })
            coarse_ids, fine_ids = predict_selected_ids(
                gpt, inp, tids, pos, va,
                selected_rows, selected_positions, tokenizer, device,
            )
            n_forward += 1
            id_pairs = torch.stack(
                (coarse_ids, fine_ids), dim=1
            ).cpu().tolist()
            all_coarse_ids.extend(item[0] for item in id_pairs)
            all_fine_ids.extend(item[1] for item in id_pairs)

    if not silent:
        print(f"  {n_forward} forward passes")

    decoded = decode_code_ids_tensor(
        all_coarse_ids, all_fine_ids, tokenizer, device
    ).cpu().numpy()

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
                              batch_size=16, n_days=20, start_offset=0,
                              silent=False):
    """Batched GPT evaluation with multi-day sliding window.

    For each test stock, predicts up to n_days consecutive positions starting
    from test_pos.  Stocks with fewer than n_days positions are still included
    (evaluated for as many positions as available).

    Args:
        n_days: maximum number of consecutive days to predict per stock (default 20)
        start_offset: begin this many test observations after the cutoff
    """
    # Phase 1: batch tokenize
    if not silent:
        print(f"  Preparing {len(test_stocks)} stocks (batched tokenize)...")
    prepped = _prepare_stocks_batch(test_stocks, tokenizer, device)
    if not silent:
        print(f"  {len(prepped)} valid stocks, n_days={n_days}")

    # Filter stocks that have at least some test data
    min_required = min(n_days, 10)
    valid = [p for p in prepped
             if p["test_pos"] + start_offset + min_required <= p["seq_len"]]
    if not valid:
        return []
    if not silent:
        print(f"  {len(valid)} stocks with >= {min_required} test days")

    # Phase 2: bucket by length
    buckets = _bucket_by_length(valid, tolerance=100)
    if not silent:
        print(f"  {len(buckets)} length buckets, batch_size={batch_size}")

    # Phase 3: single forward pass, gather and decode requested positions
    predictions = []  # list of dicts with date_key, meta, etc.
    n_forward = 0

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
                va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

                for j, s in enumerate(batch):
                    length = s["seq_len"]
                    inp[j, :length] = torch.tensor(
                        s["inp_ids"][:length], dtype=torch.long
                    )
                    tids[j, :length, 0] = torch.tensor(
                        s["day"][:length], dtype=torch.long
                    )
                    tids[j, :length, 1] = torch.tensor(
                        s["month"][:length], dtype=torch.long
                    )
                    tids[j, :length, 2] = torch.tensor(
                        s["year"][:length], dtype=torch.long
                    )
                    va[j, :length] = torch.tensor(
                        s["va"][:length], dtype=torch.float32
                    )

                selected_rows = []
                selected_positions = []
                selected_meta = []
                for j, s in enumerate(batch):
                    p_start = s["test_pos"] + start_offset
                    for offset in range(n_days):
                        p = p_start + offset
                        if p >= s["seq_len"]:
                            break  # out of bounds for this stock (not padded batch length)

                        selected_rows.append(j)
                        selected_positions.append(p)
                        selected_meta.append((s, p, offset))

                coarse_ids, fine_ids = predict_selected_ids(
                    gpt, inp, tids, pos, va,
                    selected_rows, selected_positions, tokenizer, device,
                )
                n_forward += 1
                id_pairs = torch.stack(
                    (coarse_ids, fine_ids), dim=1
                ).cpu().tolist()
                for ids, (stock, position, day_offset) in zip(
                    id_pairs, selected_meta
                ):
                    predictions.append({
                        "coarse_id": int(ids[0]),
                        "fine_id": int(ids[1]),
                        "test_pos": position,
                        "date_key": (
                            stock["dates_raw"][position]
                            if position < len(stock["dates_raw"])
                            else "unknown"
                        ),
                        "day_offset": day_offset,
                        "p_mean": stock["p_mean"],
                        "p_std": stock["p_std"],
                        "feat": stock["feat"],
                        "close": stock["close"],
                    })

                batch_start += B

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

    # Keep the historical 10k decode chunking so the decoder GEMM shape and
    # therefore every reconstructed float remain exactly comparable.
    decode_bs = 10_000
    for start in range(0, len(predictions), decode_bs):
        end = min(start + decode_bs, len(predictions))
        chunk = predictions[start:end]
        decoded = decode_code_ids_tensor(
            [item["coarse_id"] for item in chunk],
            [item["fine_id"] for item in chunk],
            tokenizer,
            device,
        ).cpu().numpy()
        for feature, prediction in zip(decoded, chunk):
            position = prediction["test_pos"]
            close = prediction["close"]
            prediction["decoded_feat"] = feature
            prediction["pred_logret"] = (
                float(feature[0]) * prediction["p_std"][0]
                + prediction["p_mean"][0]
            )
            prediction["true_logret"] = float(
                prediction["feat"][position, 0]
            )
            prediction["base_close"] = float(close[position - 1])
            prediction["true_close"] = float(close[position])
            del (
                prediction["fine_id"],
                prediction["feat"],
                prediction["close"],
            )

    return predictions


def compute_windowed_metrics(predictions, min_date_coverage_ratio=0.80):
    """Compute multi-day-averaged metrics from windowed predictions.

    Returns dict with per-date DA then cross-date aggregation.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from collections import defaultdict

    def token_distribution_metrics(pred_tokens, true_tokens):
        """Compare argmax-token use with the empirical target distribution.

        Raw unique-token counts reward even one-off, potentially wrong tokens.
        These diagnostics instead report support overlap, effective vocabulary
        size (exp entropy), collapse alignment, and Jensen-Shannon divergence.
        JSD uses base-2 logs and is therefore bounded to [0, 1].
        """
        pred_tokens = np.asarray(pred_tokens, dtype=np.int64)
        true_tokens = np.asarray(true_tokens, dtype=np.int64)
        paired = true_tokens >= 0
        pred_tokens = pred_tokens[paired]
        true_tokens = true_tokens[paired]
        if not len(true_tokens):
            return {}

        # ``[(tokens == label).sum() for label in labels]`` was acceptable
        # for the ~175-token daily coarse support but becomes quadratic-like
        # work for thousands of joint codes.  Unique once, then align both
        # sparse count vectors onto the same sorted support.
        pred_labels, pred_nonzero_counts = np.unique(
            pred_tokens, return_counts=True
        )
        true_labels, true_nonzero_counts = np.unique(
            true_tokens, return_counts=True
        )
        labels = np.union1d(pred_labels, true_labels)
        pred_counts = np.zeros(len(labels), dtype=np.float64)
        true_counts = np.zeros(len(labels), dtype=np.float64)
        pred_counts[np.searchsorted(labels, pred_labels)] = (
            pred_nonzero_counts
        )
        true_counts[np.searchsorted(labels, true_labels)] = (
            true_nonzero_counts
        )
        pred_prob = pred_counts / pred_counts.sum()
        true_prob = true_counts / true_counts.sum()
        midpoint = 0.5 * (pred_prob + true_prob)

        def entropy_bits(prob):
            nonzero = prob > 0
            return float(-np.sum(prob[nonzero] * np.log2(prob[nonzero])))

        def kl_bits(left, right):
            nonzero = left > 0
            return float(
                np.sum(
                    left[nonzero]
                    * np.log2(left[nonzero] / right[nonzero])
                )
            )

        pred_entropy = entropy_bits(pred_prob)
        true_entropy = entropy_bits(true_prob)
        pred_effective = float(2.0 ** pred_entropy)
        true_effective = float(2.0 ** true_entropy)
        jsd = float(
            0.5 * kl_bits(pred_prob, midpoint)
            + 0.5 * kl_bits(true_prob, midpoint)
        )

        pred_support = set(pred_labels.tolist())
        true_support = set(true_labels.tolist())
        overlap = len(pred_support & true_support)
        precision = overlap / len(pred_support)
        recall = overlap / len(true_support)
        support_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        pred_collapse = float(pred_counts.max() / pred_counts.sum())
        true_collapse = float(true_counts.max() / true_counts.sum())

        def symmetric_ratio(left, right):
            if left <= 0 or right <= 0:
                return 0.0
            return float(min(left / right, right / left))

        unique_alignment = symmetric_ratio(
            len(pred_support), len(true_support)
        )
        effective_alignment = symmetric_ratio(
            pred_effective, true_effective
        )
        collapse_alignment = symmetric_ratio(
            pred_collapse, true_collapse
        )
        distribution_alignment = max(0.0, 1.0 - jsd)
        # A diagnostic balance index, not a downstream quality score.  The
        # geometric mean prevents one strong marginal from hiding a failure in
        # support, frequency shape, effective use, or collapse behaviour.
        codebook_balance = float(
            (
                support_f1
                * distribution_alignment
                * effective_alignment
                * collapse_alignment
            )
            ** 0.25
        )
        return {
            "token_accuracy": float(np.mean(pred_tokens == true_tokens)),
            "n_unique_tokens": int(len(pred_support)),
            "collapse_rate": pred_collapse,
            "target_n_unique_tokens": int(len(true_support)),
            "target_collapse_rate": true_collapse,
            "pred_token_entropy_bits": pred_entropy,
            "target_token_entropy_bits": true_entropy,
            "pred_effective_tokens": pred_effective,
            "target_effective_tokens": true_effective,
            "prediction_support_precision": float(precision),
            "target_support_recall": float(recall),
            "token_support_f1": float(support_f1),
            "token_jsd": jsd,
            "distribution_alignment": distribution_alignment,
            "unique_token_alignment": unique_alignment,
            "effective_token_alignment": effective_alignment,
            "collapse_alignment": collapse_alignment,
            "codebook_balance_score": codebook_balance,
        }

    def named_token_distribution_metrics(
        pred_tokens, true_tokens, *, level
    ):
        """Name one hierarchy level while preserving legacy coarse fields."""
        metrics = token_distribution_metrics(pred_tokens, true_tokens)
        if not metrics:
            return {}
        if level == "coarse":
            # collapse_rate/n_unique_tokens already exist as the historical
            # coarse behaviour fields and are populated below.
            metrics.pop("collapse_rate")
            metrics.pop("n_unique_tokens")
            metrics["coarse_token_accuracy"] = metrics.pop("token_accuracy")
            return metrics
        prefix = f"{level}_"
        return {f"{prefix}{key}": value for key, value in metrics.items()}

    if not predictions:
        return {"avg_da_per_date": 0, "da_std": 0, "da_vol": 0,
                "n_dates": 0, "n_predictions": 0,
                "avg_da_above_baseline": 0, "signal_score": 0,
                "collapse_rate": 0, "n_unique_tokens": 0,
                "max_daily_collapse_rate": 0, "min_daily_unique_tokens": 0,
                "rank_ic": 0, "avg_daily_rank_ic": 0,
                "daily_rank_ic_std": 0, "ampratio": 0,
                "mape": 0, "baseline_mape": 0,
                "min_date_coverage_ratio": min_date_coverage_ratio,
                "max_cross_section_size": 0, "min_cross_section_size": 0,
                "dropped_sparse_dates": 0,
                "per_date": {}}

    # Group predictions by date
    by_date = defaultdict(list)
    for p in predictions:
        by_date[p["date_key"]].append(p)

    finite_counts = {}
    for date_key, group in by_date.items():
        pred_lrs = np.array([g["pred_logret"] for g in group])
        true_lrs = np.array([g["true_logret"] for g in group])
        finite_counts[date_key] = int(
            (np.isfinite(pred_lrs) & np.isfinite(true_lrs)).sum())
    max_cross_section_size = max(finite_counts.values(), default=0)
    min_cross_section_size = max(
        5, int(np.ceil(max_cross_section_size * min_date_coverage_ratio)))
    dropped_sparse_dates = sum(
        count < min_cross_section_size for count in finite_counts.values())

    per_date = {}
    all_pred_lrs = []
    all_true_lrs = []
    all_pred_toks = []

    for date_key, group in sorted(by_date.items()):
        pred_lrs = np.array([g["pred_logret"] for g in group])
        true_lrs = np.array([g["true_logret"] for g in group])
        pred_toks = np.array([g["coarse_id"] for g in group], dtype=np.int64)
        true_toks = np.array(
            [g.get("true_coarse_id", -1) for g in group],
            dtype=np.int64,
        )
        pred_fine_toks = np.array(
            [g.get("fine_id", -1) for g in group], dtype=np.int64
        )
        true_fine_toks = np.array(
            [g.get("true_fine_id", -1) for g in group], dtype=np.int64
        )
        pred_joint_toks = np.array(
            [g.get("joint_id", -1) for g in group], dtype=np.int64
        )
        true_joint_toks = np.array(
            [g.get("true_joint_id", -1) for g in group], dtype=np.int64
        )
        eps = 1e-8

        valid = np.isfinite(pred_lrs) & np.isfinite(true_lrs)
        if valid.sum() < min_cross_section_size:
            continue

        pred_lrs = pred_lrs[valid]
        true_lrs = true_lrs[valid]
        pred_toks = pred_toks[valid]
        true_toks = true_toks[valid]
        pred_fine_toks = pred_fine_toks[valid]
        true_fine_toks = true_fine_toks[valid]
        pred_joint_toks = pred_joint_toks[valid]
        true_joint_toks = true_joint_toks[valid]
        da = float((np.sign(pred_lrs) == np.sign(true_lrs)).mean())
        daily_rank_ic = float(spearmanr(pred_lrs, true_lrs)[0])
        if np.isnan(daily_rank_ic):
            daily_rank_ic = 0.0
        _, daily_token_counts = np.unique(pred_toks, return_counts=True)
        daily_collapse_rate = float(daily_token_counts.max() / len(pred_toks))
        daily_unique_tokens = int(len(daily_token_counts))

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
            "rank_ic": daily_rank_ic,
            "collapse_rate": daily_collapse_rate,
            "n_unique_tokens": daily_unique_tokens,
        }
        per_date[date_key].update(
            named_token_distribution_metrics(
                pred_toks, true_toks, level="coarse"
            )
        )
        if np.any(true_fine_toks >= 0):
            per_date[date_key].update(
                named_token_distribution_metrics(
                    pred_fine_toks, true_fine_toks, level="fine"
                )
            )
        if np.any(true_joint_toks >= 0):
            per_date[date_key].update(
                named_token_distribution_metrics(
                    pred_joint_toks, true_joint_toks, level="joint"
                )
            )

        all_pred_lrs.extend(pred_lrs.tolist())
        all_true_lrs.extend(true_lrs.tolist())
        all_pred_toks.extend(pred_toks.tolist())

    if not per_date:
        return {
            "avg_da_per_date": 0, "da_std": 0, "da_vol": 0,
            "n_dates": 0, "n_predictions": 0,
            "avg_da_above_baseline": 0, "signal_score": 0,
            "collapse_rate": 0, "n_unique_tokens": 0,
            "max_daily_collapse_rate": 0, "min_daily_unique_tokens": 0,
            "rank_ic": 0, "avg_daily_rank_ic": 0,
            "daily_rank_ic_std": 0, "ampratio": 0,
            "mape": 0, "baseline_mape": 0,
            "min_date_coverage_ratio": min_date_coverage_ratio,
            "max_cross_section_size": max_cross_section_size,
            "min_cross_section_size": min_cross_section_size,
            "dropped_sparse_dates": dropped_sparse_dates,
            "per_date": {},
        }

    # Cross-date aggregation
    das = np.array([v["da"] for v in per_date.values()])

    avg_da = float(das.mean())
    da_std = float(das.std()) if len(das) > 1 else 0.0
    da_values = list(das)
    da_vol = float(np.std(da_values)) if len(da_values) > 1 else 0.0

    # Aggregate baseline-relative metrics
    above_baseline_vals = [v["da_above_baseline"] for v in per_date.values()]
    avg_da_above_baseline = float(np.mean(above_baseline_vals))
    daily_rank_ics = np.array([v["rank_ic"] for v in per_date.values()],
                              dtype=np.float64)
    avg_daily_rank_ic = float(daily_rank_ics.mean())
    daily_rank_ic_std = (float(daily_rank_ics.std())
                         if len(daily_rank_ics) > 1 else 0.0)
    max_daily_collapse_rate = max(
        value["collapse_rate"] for value in per_date.values())
    min_daily_unique_tokens = min(
        value["n_unique_tokens"] for value in per_date.values())

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

    # MAPE (price space), using the same dense dates as DA/RankIC.
    included_dates = set(per_date)
    mape_rows = [p for p in predictions if p["date_key"] in included_dates]
    all_pred_lrs_for_mape = np.array(
        [p.get("pred_logret", np.nan) for p in mape_rows])
    all_base_closes = np.array([p.get("base_close", 0) for p in mape_rows])
    all_true_closes = np.array([p.get("true_close", 0) for p in mape_rows])
    has_close = (
        (all_base_closes > 0) & (all_true_closes > 0)
        & np.isfinite(all_pred_lrs_for_mape)
        & np.isfinite(all_base_closes) & np.isfinite(all_true_closes)
    )
    if has_close.sum() > 0:
        pred_prices = all_base_closes[has_close] * np.exp(all_pred_lrs_for_mape[has_close].astype(np.float64))
        true_prices = all_true_closes[has_close]
        mape = float(np.mean(np.abs(pred_prices - true_prices) / np.maximum(np.abs(true_prices), eps))) * 100
        # Baseline MAPE: if we always predict "no change" (pred = base_close)
        bl_mape = float(np.mean(np.abs(all_base_closes[has_close] - true_prices) / np.maximum(np.abs(true_prices), eps))) * 100
    else:
        mape = 0.0
        bl_mape = 0.0

    # Signal score: average DA above baseline across dates
    # Positive = better than always-predicting-majority
    signal_score = avg_da_above_baseline

    # Collapse penalty: reduce signal if diversity is low
    total_days = len(per_date)
    n_pred = len(all_pred_lrs)

    return {
        "avg_da_per_date": avg_da,
        "da_std": da_std,
        "da_vol": da_vol,
        "n_dates": total_days,
        "n_predictions": n_pred,
        "min_date_coverage_ratio": min_date_coverage_ratio,
        "max_cross_section_size": max_cross_section_size,
        "min_cross_section_size": min_cross_section_size,
        "dropped_sparse_dates": dropped_sparse_dates,
        "avg_da_above_baseline": avg_da_above_baseline,
        "signal_score": signal_score,
        "collapse_rate": collapse_rate,
        "n_unique_tokens": n_unique_tokens,
        "max_daily_collapse_rate": max_daily_collapse_rate,
        "min_daily_unique_tokens": min_daily_unique_tokens,
        "rank_ic": rank_ic,
        "avg_daily_rank_ic": avg_daily_rank_ic,
        "daily_rank_ic_std": daily_rank_ic_std,
        "ampratio": amp_ratio,
        "mape": mape,
        "baseline_mape": bl_mape,
        "per_date": per_date,
    }


# ============================================================================
# High-level evaluation entry point
# ============================================================================

def evaluate_windowed(gpt_ckpt, tokenizer_ckpt, device,
                      n_stocks=0, n_days=20, batch_size=4,
                      start_offset=0, silent=False, seed=42,
                      sample_strategy="random"):
    """Run multi-day windowed GPT evaluation. Returns metrics dict.

    Args:
        gpt_ckpt: path to GPT checkpoint
        tokenizer_ckpt: path to tokenizer checkpoint
        device: torch device
        n_stocks: number of test stocks to evaluate (0 = all)
        n_days: number of consecutive test days per stock
        batch_size: evaluation batch size
        start_offset: begin this many test observations after the cutoff
        silent: suppress print output
        seed: random seed for stock sampling
        sample_strategy: ``random`` (default) or ``shortest`` (smoke tests)
    """
    import time as _time
    ModelConfig = __import__("config", fromlist=["ModelConfig"]).ModelConfig

    tokenizer = load_tokenizer(tokenizer_ckpt, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    gpt = load_gpt(gpt_ckpt, device, tokenizer=tokenizer)

    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks = split_stocks(stocks)

    n_sample = len(test_stocks) if n_stocks <= 0 else min(n_stocks, len(test_stocks))
    if sample_strategy == "shortest" and n_stocks > 0:
        test_sample = sorted(
            test_stocks, key=lambda stock: len(stock["features_raw"])
        )[:n_sample]
    elif sample_strategy == "random":
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(test_stocks), n_sample, replace=False)
        test_sample = [test_stocks[i] for i in sorted(indices)]
    else:
        raise ValueError(f"Unknown sample_strategy: {sample_strategy}")
    attach_close_prices(test_sample)

    t0 = _time.time()
    preds = batched_gpt_eval_windowed(
        gpt, tokenizer, test_sample, device,
        batch_size=batch_size, n_days=n_days,
        start_offset=start_offset, silent=silent)

    elapsed = _time.time() - t0
    metrics = compute_windowed_metrics(preds)

    if not silent:
        print(f"\n  Eval done in {elapsed:.1f}s ({len(preds)} predictions, "
              f"{metrics.get('n_dates', 0)} dates)")
        print(f"  Avg DA per date:  {metrics['avg_da_per_date']*100:.2f}%")
        print(f"  DA above baseline: {metrics['avg_da_above_baseline']*100:+.2f}%")
        print(f"  DA std across dates: {metrics.get('da_std', 0)*100:.2f}%")
        print(f"  Dense-date coverage: >={metrics.get('min_cross_section_size', 0)} stocks "
              f"({metrics.get('min_date_coverage_ratio', 0)*100:.0f}% of max); "
              f"dropped dates={metrics.get('dropped_sparse_dates', 0)}")
        print(f"  Collapse: {metrics['collapse_rate']*100:.1f}%  "
              f"Unique: {metrics['n_unique_tokens']}  "
              f"RankIC: {metrics['rank_ic']:.4f}  "
              f"AmpRatio: {metrics.get('ampratio', 0):.3f}x  "
              f"MAPE: {metrics.get('mape', 0):.2f}%  "
              f"BL-MAPE: {metrics.get('baseline_mape', 0):.2f}%")

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
