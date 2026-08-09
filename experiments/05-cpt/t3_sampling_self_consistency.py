"""T3: Sampling-based self-consistency inference (ToDo T3).

For every (stock, date) position in the 400-window protocol, sample K coarse
tokens from the model's coarse logits (multinomial at temperature T), decode
each to a log-return, then compare three policies against the realized return:

    - argmax : single-path greedy (the existing windowed evaluator baseline)
    - vote   : majority sign across K sampled log-returns
    - mean   : mean log-return across K sampled paths

Report per-window and aggregate DA / RankIC for each policy. Pure inference,
zero training. No holdout positions are used (offsets 0-399).

Reuses the exact prepared-input packing from evaluate_epoch_trajectory so the
input tensors are identical to the formal 400-window protocol.

Usage:
    python experiments/05-cpt/t3_sampling_self_consistency.py \\
        --ckpt checkpoints/exp04b_best_ep100.pt \\
        --tokenizer checkpoints/tokenizer_v2_ohlc.pt \\
        --k 8 --temperature 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "experiments" / "04" / "b-hpo"))

from config import ModelConfig, set_global_seed  # noqa: E402
from eval_helpers import (  # noqa: E402
    AMP_DTYPE,
    decode_code_ids_tensor,
    load_gpt,
    load_tokenizer,
)
from evaluate_epoch_trajectory import (  # noqa: E402
    parse_int_spec,
    prepare_stocks,
    prepared_cache_metadata,
    prepared_cache_path,
    load_prepared_cache,
    save_prepared_cache,
    resolve_tokenizer,
    load_json,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--trial_dir", type=Path, default=None)
    parser.add_argument("--offsets", default=",".join(str(v) for v in range(400)))
    parser.add_argument("--n_days", type=int, default=1)
    parser.add_argument("--n_stocks", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--prepared_cache_dir",
        type=Path,
        default=None,
        help="Reuse the trajectory evaluator's prepared-input cache.",
    )
    return parser.parse_args()


def logret_from_ids(coarse_ids: torch.Tensor, fine_ids: torch.Tensor,
                    tokenizer: Any, device: Any,
                    p_means: np.ndarray, p_stds: np.ndarray) -> np.ndarray:
    """Decode (coarse, fine) id pairs to price-space log-returns."""
    decoded = decode_code_ids_tensor(coarse_ids, fine_ids, tokenizer, device)
    logrets = decoded.cpu().numpy()[:, 0] * p_stds + p_means
    return logrets


@torch.no_grad()
def main() -> int:
    args = parse_args()
    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Resolve tokenizer + trial dir -----------------------------------
    if args.tokenizer is None:
        if args.trial_dir is None:
            raise ValueError("--tokenizer or --trial_dir required")
        args.tokenizer = resolve_tokenizer(args, args.trial_dir)
    tokenizer = load_tokenizer(str(args.tokenizer), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    model = load_gpt(str(args.ckpt), device, tokenizer=tokenizer)
    model.eval()

    offsets = parse_int_spec(args.offsets)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)
    print(f"K={args.k}, temperature={args.temperature}", flush=True)
    print(f"Windows: {len(offsets)} (0-{offsets[-1]}), days/window={args.n_days}", flush=True)

    # ---- Prepare stocks once (reuse trajectory prep) -----------------------
    trial_dir = args.trial_dir or Path(
        "server_runs/results/04b-cpt/seed42/trials/local_cpt"
    )
    settings = {
        "trial_dir": str(trial_dir),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": file_sha256(Path(args.tokenizer)),
        "override_path": str((trial_dir / "override.json").resolve()),
        "override_sha256": file_sha256(trial_dir / "override.json"),
        "seed": args.seed,
        "offsets": offsets,
        "n_days": args.n_days,
        "n_stocks": args.n_stocks,
        "batch_size": args.batch_size,
        "sample_strategy": "random",
        "max_collapse_rate": 0.35,
        "min_unique_tokens": 32,
        "evaluator_schema": 5,
        "token_distribution_schema": 3,
        "prediction_record_schema": 1,
        "holdout_used": False,
    }
    cache_root = args.prepared_cache_dir or (trial_dir / "cache")
    metadata = prepared_cache_metadata(settings)
    cache_path = prepared_cache_path(cache_root, metadata)
    prepared_batches = None
    if cache_path.exists():
        try:
            prepared_batches, sampled_count, valid_count = load_prepared_cache(
                cache_path, metadata
            )
            print(f"  Reused prepared cache ({valid_count} stocks, {len(prepared_batches)} batches)", flush=True)
        except Exception as exc:
            print(f"  Cache invalid: {exc}", flush=True)
            prepared_batches = None
    if prepared_batches is None:
        prepared_batches, sampled_count, valid_count = prepare_stocks(
            tokenizer=tokenizer, device=device, n_stocks=args.n_stocks,
            sample_strategy="random", seed=args.seed, offsets=offsets,
            n_days=args.n_days, batch_size=args.batch_size,
        )
        save_prepared_cache(cache_path, metadata, prepared_batches, sampled_count, valid_count)
        print(f"  Built prepared cache ({valid_count} stocks)", flush=True)

    # ---- Forward + K-sample loop over prepared batches ---------------------
    # Mirrors evaluate_prepared but draws K multinomial coarse samples per
    # selected position and decodes each to a log-return.
    all_rows: list[dict[str, Any]] = []
    selective = getattr(model, "forward_selected", None)
    if selective is None:
        raise RuntimeError("T3 requires forward_selected (KronosPreview v2)")

    started = time.monotonic()
    vocab = tokenizer.vocab_coarse
    for bi, prepared in enumerate(prepared_batches):
        n_rows = prepared["input_ids"].shape[0]
        max_len = int(prepared["lengths"].max())
        selection_ptr = prepared["selection_ptr"]
        row_start = 0
        while row_start < n_rows:
            row_end = min(row_start + args.batch_size, n_rows)
            batch_count = row_end - row_start
            max_len_b = int(prepared["lengths"][row_start:row_end].max())
            selection_start = int(selection_ptr[row_start])
            selection_end = int(selection_ptr[row_end])
            inp = prepared["input_ids"][row_start:row_end, :max_len_b].to(device)
            tids = prepared["time_ids"][row_start:row_end, :max_len_b].to(device)
            va = prepared["va_values"][row_start:row_end, :max_len_b].to(device)
            positions = torch.arange(max_len_b, device=device).unsqueeze(0).expand(batch_count, -1)
            # selection_rows are offsets within the (row_start:row_end) slice
            sel_rows = prepared["selection_rows"][selection_start:selection_end] - row_start
            sel_pos = prepared["selection_positions"][selection_start:selection_end]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                coarse_logits, fine_logits = selective(
                    inp, tids, positions, sel_rows, sel_pos, va_values=va
                )
            # coarse_logits: [n_sel, vocab_coarse]
            logits = coarse_logits[:, :vocab].float()
            if args.temperature != 1.0:
                logits = logits / args.temperature
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs, args.k, replacement=True
            )  # [n_sel, K]
            fine_ids = fine_logits.float().argmax(dim=-1)  # [n_sel]
            n_sel = sampled.shape[0]
            if n_sel == 0:
                row_start = row_end
                continue
            coarse_flat = sampled.reshape(-1)  # [n_sel*K]
            fine_flat = fine_ids.unsqueeze(1).expand(-1, args.k).reshape(-1)
            span = slice(selection_start, selection_end)
            p_means = prepared["p_means"][span].numpy()
            p_stds = prepared["p_stds"][span].numpy()
            p_means_k = np.repeat(p_means, args.k)
            p_stds_k = np.repeat(p_stds, args.k)
            decoded = logret_from_ids(
                coarse_flat, fine_flat, tokenizer, device, p_means_k, p_stds_k
            ).reshape(n_sel, args.k)
            argmax_id = logits.argmax(dim=-1)
            argmax_lr = logret_from_ids(
                argmax_id, fine_ids, tokenizer, device, p_means, p_stds
            )
            true_lr = prepared["true_logrets"][span].numpy()
            base_close = prepared["base_closes"][span].numpy()
            true_close = prepared["true_closes"][span].numpy()
            window_starts = prepared["window_starts"][span].numpy()
            symbols = prepared["symbols"][span]
            dates = prepared["date_keys"][span]

            for i in range(n_sel):
                sample_lrs = decoded[i]
                argmax_vote = 1.0 if argmax_lr[i] >= 0 else -1.0
                n_pos = int((sample_lrs >= 0).sum())
                n_neg = args.k - n_pos
                vote_dir = 1.0 if n_pos > n_neg else (-1.0 if n_neg > n_pos else argmax_vote)
                all_rows.append({
                    "symbol": symbols[i],
                    "date": dates[i],
                    "window_start": int(window_starts[i]),
                    "true_lr": float(true_lr[i]),
                    "base_close": float(base_close[i]),
                    "true_close": float(true_close[i]),
                    "argmax_lr": float(argmax_lr[i]),
                    "vote_lr": vote_dir,
                    "mean_lr": float(sample_lrs.mean()),
                    "k_samples": args.k,
                })
            row_start = row_end
        if (bi + 1) % 100 == 0:
            print(f"  batch {bi+1}/{len(prepared_batches)}, elapsed={time.monotonic()-started:.1f}s", flush=True)

    print(f"Collected {len(all_rows)} positions in {time.monotonic()-started:.1f}s", flush=True)

    # ---- Aggregate per-date DA / RankIC for each policy --------------------
    from collections import defaultdict
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["date"]].append(row)

    def spearman(x, y):
        from scipy.stats import spearmanr
        if len(x) < 2:
            return float("nan")
        return float(spearmanr(x, y).statistic)

    policies = {
        "argmax": lambda r: r["argmax_lr"],
        "vote": lambda r: r["vote_lr"],
        "mean": lambda r: r["mean_lr"],
    }
    results: dict[str, Any] = {"n_positions": len(all_rows), "k": args.k,
                               "temperature": args.temperature}
    for pname, pfunc in policies.items():
        da_vals, ic_vals = [], []
        for date, rows in grouped.items():
            if len(rows) < 3:
                continue
            preds = np.asarray([pfunc(r) for r in rows])
            trues = np.asarray([r["true_lr"] for r in rows])
            pred_sign = np.sign(preds)
            true_sign = np.sign(trues)
            # DA: direction agreement — same convention as compute_windowed_metrics
            da = float((pred_sign == true_sign).mean())
            da_vals.append(da)
            ic = spearman(preds, trues)
            if not math.isnan(ic):
                ic_vals.append(ic)
        results[f"{pname}_mean_da"] = float(np.mean(da_vals)) if da_vals else float("nan")
        results[f"{pname}_median_da"] = float(np.median(da_vals)) if da_vals else float("nan")
        results[f"{pname}_mean_ic"] = float(np.mean(ic_vals)) if ic_vals else float("nan")
        results[f"{pname}_median_ic"] = float(np.median(ic_vals)) if ic_vals else float("nan")
        results[f"{pname}_n_dates"] = len(da_vals)

    # Output
    out_dir = args.output_dir or Path(
        "server_runs/results/04b-cpt/seed42/trials/local_cpt/t3_self_consistency"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_position.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=== T3 self-consistency results ===")
    for pname in policies:
        print(
            f"{pname:>6}: mean_DA={results[f'{pname}_mean_da']*100:.2f}% "
            f"median_DA={results[f'{pname}_median_da']*100:.2f}% "
            f"mean_IC={results[f'{pname}_mean_ic']:.4f} "
            f"(n_dates={results[f'{pname}_n_dates']})"
        )
    print()
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
