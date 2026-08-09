"""PT-00C/PT-03: causal frozen-hidden cache over the formal 400-day window.

Builds the prepared inputs (offsets 0..399, n_days=1) with stock_uid identity,
runs the frozen upstream backbone's ``encode_selected`` over every selected
position, and writes a hidden cache to the weights root keyed by
upstream/tokenizer/data/split hashes.

Output NPZ fields (one row per date x stock_uid prediction):
    hidden            [N, dim] float32   final-norm hidden states
    stock_uid/date_key/symbol [N] unicode
    offset            [N] int16
    position          [N] int32
    p_mean0 / p_std0  [N] float64
    true_logret       [N] float64
    true_coarse_id / true_fine_id [N] int16
    base_close / true_close [N] float64
    dense_threshold   scalar (from the shared reference universe)

The hidden cache is the common input for PT-01 (exact joint decode) and PT-03
(independent probes); it is never a results-tree artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig  # noqa: E402
from experiment_io import file_sha256  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402
from data_processor import split_stocks  # noqa: E402

from posttrain_common import (  # noqa: E402
    load_reviewed_selection, upstream_checkpoint_path, resolve_roots,
    VALIDATION_OFFSETS, require_offsets, dict_sha256,
)
from posttrain_data import (  # noqa: E402
    load_stocks_uid, attach_close_prices_uid, prepare_stocks_uid,
    build_prepared_batches,
)


def prepared_cache_key(sel, tokenizer_path, split_sha):
    """Hash key for the prepared-input cache (uid + upstream + data + split)."""
    ckpt = upstream_checkpoint_path(sel)
    payload = {
        "schema": 2,
        "upstream_sha256": file_sha256(ckpt),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "data_uid_fingerprint": None,  # set by caller
        "split_sha256": split_sha,
        "offsets": list(VALIDATION_OFFSETS),
        "n_days": 1,
    }
    return payload


def build_and_extract_hidden(
    *,
    device,
    batch_size=4,
    seed=42,
    roots=None,
):
    """Build prepared inputs and extract hidden states; returns cache paths.

    Returns (prepared_path, hidden_path, n_rows, dense_threshold).
    """
    import numpy as np
    from eval_helpers import _bucket_by_length  # noqa: F401

    sel = load_reviewed_selection()
    ckpt_path = upstream_checkpoint_path(sel)
    tok_path = Path(sel["upstream"]["tokenizer"])
    tokenizer = load_tokenizer(str(tok_path), device)

    # ---- stocks + split ----
    stocks = load_stocks_uid(DataConfig.data_dir)
    _, _, test_stocks = split_stocks(stocks)  # train/val/test by cutoff date
    attach_close_prices_uid(test_stocks)
    prepped = prepare_stocks_uid(test_stocks, tokenizer, device)
    batches = build_prepared_batches(
        prepped, list(VALIDATION_OFFSETS), n_days=1, batch_size=batch_size)
    n_rows = sum(len(b["stock_uids"]) for b in batches)
    print(f"[hidden] prepared {len(batches)} batches, {n_rows} rows")

    # dense threshold from the reference universe (one date's max cross-section)
    by_date = {}
    for b in batches:
        for d, u in zip(b["date_keys"], b["stock_uids"]):
            by_date.setdefault(d, set()).add(u)
    max_cs = max(len(v) for v in by_date.values())
    dense_threshold = max(5, int(np.ceil(0.8 * max_cs)))
    print(f"[hidden] max cross-section {max_cs}, dense_threshold {dense_threshold}")

    # ---- model ----
    model = load_gpt(str(ckpt_path), device, tokenizer=tokenizer)
    model.eval()

    # ---- extract hidden ----
    dim = model.head_coarse.in_features
    hidden_list = []
    meta = {k: [] for k in ("stock_uid", "date_key", "symbol", "offset", "position",
                            "p_mean0", "p_std0", "true_logret", "true_coarse_id",
                            "true_fine_id", "base_close", "true_close")}
    total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for bi, batch in enumerate(batches):
            inp = batch["input_ids"].to(device)
            tids = batch["time_ids"].to(device)
            va = batch["va_values"].to(device)
            pos_ids = torch.arange(inp.shape[1], device=device).unsqueeze(0).expand(inp.shape[0], -1)
            rows = batch["selection_rows"].to(device)
            poss = batch["selection_positions"].to(device)
            h = model.encode_selected(inp, tids, pos_ids, rows, poss, va_values=va)
            hidden_list.append(h.float().cpu())
            cnt = h.shape[0]
            total += cnt
            meta["stock_uid"].extend(batch["stock_uids"])
            meta["date_key"].extend(batch["date_keys"])
            meta["symbol"].extend(batch["symbols"])
            meta["offset"].extend(batch["window_starts"].tolist())
            meta["position"].extend(batch["selection_positions"].tolist())
            meta["p_mean0"].extend(batch["p_means"].tolist())
            meta["p_std0"].extend(batch["p_stds"].tolist())
            meta["true_logret"].extend(batch["true_logrets"].tolist())
            meta["true_coarse_id"].extend(batch["true_coarse_ids"].tolist())
            meta["true_fine_id"].extend(batch["true_fine_ids"].tolist())
            meta["base_close"].extend(batch["base_closes"].tolist())
            meta["true_close"].extend(batch["true_closes"].tolist())
            if (bi + 1) % 200 == 0:
                print(f"[hidden] batch {bi+1}/{len(batches)} rows {total}")

    hidden = torch.cat(hidden_list, dim=0)  # [N, dim]
    print(f"[hidden] total hidden rows {hidden.shape}")

    # ---- write caches ----
    weights_root = roots.weights_root if roots else resolve_roots().weights_root
    weights_root.mkdir(parents=True, exist_ok=True)
    prep_path = weights_root / "prepared_inputs.pt"
    torch.save({"batches": batches, "dense_threshold": int(dense_threshold),
                "n_rows": int(n_rows)}, prep_path)

    hidden_path = weights_root / "hidden_cache.npz"
    np.savez(
        hidden_path,
        hidden=hidden.numpy(),
        stock_uid=np.array(meta["stock_uid"]),
        date_key=np.array(meta["date_key"]),
        symbol=np.array(meta["symbol"]),
        offset=np.array(meta["offset"], dtype=np.int16),
        position=np.array(meta["position"], dtype=np.int32),
        p_mean0=np.array(meta["p_mean0"], dtype=np.float64),
        p_std0=np.array(meta["p_std0"], dtype=np.float64),
        true_logret=np.array(meta["true_logret"], dtype=np.float64),
        true_coarse_id=np.array(meta["true_coarse_id"], dtype=np.int16),
        true_fine_id=np.array(meta["true_fine_id"], dtype=np.int16),
        base_close=np.array(meta["base_close"], dtype=np.float64),
        true_close=np.array(meta["true_close"], dtype=np.float64),
        dense_threshold=np.array([dense_threshold]),
    )
    print(f"[hidden] wrote {hidden_path} ({hidden_path.stat().st_size/1e9:.2f} GB)")
    return prep_path, hidden_path, int(n_rows), int(dense_threshold)


def main():
    ap = argparse.ArgumentParser(description="Build the PostTrain frozen hidden cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal hidden cache")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    roots = resolve_roots(seed=args.seed)
    build_and_extract_hidden(device=device, batch_size=args.batch_size,
                             seed=args.seed, roots=roots)


if __name__ == "__main__":
    main()
