"""Branch D (MTP) post-train consistency filter inference.

The Branch D checkpoint adds three MTP future heads (``head_future``) that share
the backbone hidden state with the main coarse head and predict the next-next
(t+2), t+3 and t+4 coarse tokens.  This script reuses the formal 400-window
prepared-input cache and, for every selected (stock, date) position, runs
``forward_selected(..., return_future=True)`` once to obtain the main head
coarse/fine logits plus the three future-head coarse logits.

The main head is decoded exactly as T3 decodes its argmax path:

    main coarse id = coarse_logits[:, :vocab].argmax(-1)
    fine id        = fine_logits.argmax(-1)
    decode_code_ids_tensor(coarse, fine) -> [*, 0] * p_std + p_mean  (t+1)

Each future head uses the same fine id and its own coarse argmax, giving the
t+2/3/4 log-return.  A position is *accepted* when at least ``min_agree`` of the
three future-head directions agree with the main-head direction
(``agree >= min_agree``).  We then compare the accepted subset against the full
set on per-date DA / RankIC and report a paired (acted_da - full_da) block
bootstrap over dates.

Pure inference, zero training.  No holdout positions are used (offsets 0-399).

Usage:
    python experiments/05-cpt/d_consistency_filter.py \\
        --ckpt checkpoints/branchD_mtp.pt \\
        --tokenizer checkpoints/tokenizer_v2_ohlc.pt \\
        --offsets 0-399 --batch_size 4 --min_agree 3 --bootstrap 2000
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

# Formal 400-window protocol cache lives under the CPT baseline trial dir.  The
# runtime override JSON must be applied before ``config`` is imported so a
# rebuilt cache (fallback when the shared cache is absent) matches the exact
# DataConfig/NormConfig/TokenizerConfig used by the protocol.
DEFAULT_TRIAL_DIR = (
    ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt"
)
os.environ.setdefault("KRONOS_PREVIEW_OVERRIDE_JSON", str((DEFAULT_TRIAL_DIR / "override.json").resolve()))

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
    file_sha256,
)

# The 400-window protocol evaluates one day per window start (n_days=1).
N_DAYS = 1
N_STOCKS = 0  # 0 = the complete test universe (matches the formal protocol)
DEFAULT_CACHE_DIR = DEFAULT_TRIAL_DIR / "cache"
DEFAULT_OUTPUT_DIR = DEFAULT_TRIAL_DIR / "d_consistency_filter"

# Minimum per-date cross-section sizes (T3 uses 3 rows for DA aggregation).
MIN_DA_ROWS = 3
MIN_IC_ROWS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--offsets", default=",".join(str(v) for v in range(400)))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_agree", type=int, default=3,
                        help="Future heads agreeing with the main direction "
                             "required to accept a position (default 3; 2 to relax).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--prepared_cache_dir",
        type=Path,
        default=None,
        help="Reuse the trajectory evaluator's prepared-input cache.",
    )
    return parser.parse_args()


def direction(value: float) -> float:
    """Sign of a log-return, ties counted as +1 (matches T3's argmax_vote)."""
    return 1.0 if value >= 0 else -1.0


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(x) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def decode_logrets(coarse_ids: torch.Tensor, fine_ids: torch.Tensor,
                   tokenizer: Any, device: Any,
                   p_means: np.ndarray, p_stds: np.ndarray) -> np.ndarray:
    """Decode (coarse, fine) id pairs to price-space log-returns [N]."""
    decoded = decode_code_ids_tensor(coarse_ids, fine_ids, tokenizer, device)
    logrets = decoded.cpu().numpy()[:, 0] * p_stds + p_means
    return logrets


def aggregate_per_date(all_rows: list[dict[str, Any]], min_agree: int) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Compute per-date full/acted/rejected DA + acted RankIC.

    Returns ``(results, full_da_by_date, acted_da_by_date)`` where the two
    per-date maps only contain dates that satisfied the minimum cross-section
    size, so they can be paired for the block bootstrap.
    """
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["date"]].append(row)
    dates = sorted(grouped)

    n_total = len(all_rows)
    n_accepted = int(sum(1 for row in all_rows if row["accepted"]))
    n_rejected = n_total - n_accepted

    full_da_vals: list[float] = []
    acted_da_vals: list[float] = []
    rejected_da_vals: list[float] = []
    acted_ic_vals: list[float] = []
    daily_coverage: list[float] = []
    full_da_by_date: dict[str, float] = {}
    acted_da_by_date: dict[str, float] = {}
    per_date: dict[str, dict[str, Any]] = {}

    for date in dates:
        rows = grouped[date]
        n = len(rows)
        main_arr = np.asarray([r["main_lr"] for r in rows])
        true_arr = np.asarray([r["true_logret"] for r in rows])
        acc_mask = np.asarray([r["accepted"] for r in rows])
        n_acc = int(acc_mask.sum())
        n_rej = n - n_acc

        full_da = float((np.sign(main_arr) == np.sign(true_arr)).mean())
        if n >= MIN_DA_ROWS:
            full_da_vals.append(full_da)
            full_da_by_date[date] = full_da

        acted_da = float("nan")
        rejected_da = float("nan")
        acted_ic = float("nan")
        if n_acc >= MIN_DA_ROWS:
            am = main_arr[acc_mask]
            at = true_arr[acc_mask]
            acted_da = float((np.sign(am) == np.sign(at)).mean())
            acted_da_vals.append(acted_da)
            acted_da_by_date[date] = acted_da
            if n_acc >= MIN_IC_ROWS:
                ic = spearman(am, at)
                if not math.isnan(ic):
                    acted_ic = ic
                    acted_ic_vals.append(ic)
        if n_rej >= MIN_DA_ROWS:
            rm = main_arr[~acc_mask]
            rt = true_arr[~acc_mask]
            rejected_da = float((np.sign(rm) == np.sign(rt)).mean())
            rejected_da_vals.append(rejected_da)

        daily_coverage.append(n_acc / n if n else 0.0)
        per_date[date] = {
            "n": n,
            "n_accepted": n_acc,
            "n_rejected": n_rej,
            "full_da": full_da,
            "acted_da": acted_da,
            "rejected_da": rejected_da,
            "acted_rank_ic": acted_ic,
            "coverage": n_acc / n if n else 0.0,
        }

    def mean_of(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    results: dict[str, Any] = {
        "min_agree": min_agree,
        "n_positions": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "coverage": n_accepted / n_total if n_total else 0.0,
        "mean_daily_coverage": mean_of(daily_coverage),
        "n_dates": len(dates),
        "full_da": mean_of(full_da_vals),
        "acted_da": mean_of(acted_da_vals),
        "rejected_da": mean_of(rejected_da_vals),
        "acted_rank_ic": mean_of(acted_ic_vals),
        "n_full_da_dates": len(full_da_vals),
        "n_acted_da_dates": len(acted_da_vals),
        "n_rejected_da_dates": len(rejected_da_vals),
        "n_acted_ic_dates": len(acted_ic_vals),
        "per_date": per_date,
    }
    return results, full_da_by_date, acted_da_by_date


def paired_block_bootstrap(
    full_da_by_date: dict[str, float],
    acted_da_by_date: dict[str, float],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Block-bootstrap per-date (acted_da - full_da) deltas over dates.

    Mirrors the loop in ``bootstrap_compare.py``: resample the date axis with
    replacement, mean the deltas, then take the 2.5/97.5 percentiles.
    """
    common = sorted(set(full_da_by_date) & set(acted_da_by_date))
    if not common:
        return {
            "delta_n_dates": 0,
            "delta_mean": float("nan"),
            "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"),
            "delta_significant": False,
        }
    deltas = np.asarray(
        [acted_da_by_date[d] - full_da_by_date[d] for d in common]
    )
    rng = np.random.RandomState(seed)
    n = len(common)
    means = np.empty(bootstrap)
    for b in range(bootstrap):
        idx = rng.randint(0, n, n)
        means[b] = deltas[idx].mean()
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    mean_delta = float(deltas.mean())
    return {
        "delta_n_dates": n,
        "delta_mean": mean_delta,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "delta_ci_95": [lo, hi],
        "delta_significant": not (lo <= 0 <= hi),
    }


@torch.no_grad()
def main() -> int:
    args = parse_args()
    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Resolve tokenizer --------------------------------------------------
    if args.tokenizer is None:
        args.tokenizer = ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt"
    tokenizer = load_tokenizer(str(args.tokenizer), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    model = load_gpt(str(args.ckpt), device, tokenizer=tokenizer)
    model.eval()

    offsets = parse_int_spec(args.offsets)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)
    print(f"min_agree={args.min_agree}, bootstrap={args.bootstrap}", flush=True)
    print(f"Windows: {len(offsets)} (0-{offsets[-1]}), days/window={N_DAYS}", flush=True)

    # ---- Reuse / build the 400-window prepared-input cache ------------------
    settings = {
        "trial_dir": str(DEFAULT_TRIAL_DIR),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": file_sha256(Path(args.tokenizer)),
        "override_path": str((DEFAULT_TRIAL_DIR / "override.json").resolve()),
        "override_sha256": file_sha256(DEFAULT_TRIAL_DIR / "override.json"),
        "seed": args.seed,
        "offsets": offsets,
        "n_days": N_DAYS,
        "n_stocks": N_STOCKS,
        "batch_size": args.batch_size,
        "sample_strategy": "random",
        "max_collapse_rate": 0.35,
        "min_unique_tokens": 32,
        "evaluator_schema": 5,
        "token_distribution_schema": 3,
        "prediction_record_schema": 1,
        "holdout_used": False,
    }
    cache_root = args.prepared_cache_dir or DEFAULT_CACHE_DIR
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
            tokenizer=tokenizer, device=device, n_stocks=N_STOCKS,
            sample_strategy="random", seed=args.seed, offsets=offsets,
            n_days=N_DAYS, batch_size=args.batch_size,
        )
        save_prepared_cache(cache_path, metadata, prepared_batches, sampled_count, valid_count)
        print(f"  Built prepared cache ({valid_count} stocks)", flush=True)

    # ---- Forward + future-head consistency loop -----------------------------
    selective = getattr(model, "forward_selected", None)
    if selective is None:
        raise RuntimeError("Branch D filter requires forward_selected (KronosPreview v2)")

    all_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    vocab = tokenizer.vocab_coarse
    n_future = len(model.future_offsets)
    if n_future == 0:
        raise RuntimeError("Checkpoint model has no head_future heads (not a Branch D ckpt)")

    for bi, prepared in enumerate(prepared_batches):
        n_rows = prepared["input_ids"].shape[0]
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
                coarse_logits, fine_logits, future_logits = selective(
                    inp, tids, positions, sel_rows, sel_pos, va_values=va, return_future=True
                )
            n_sel = coarse_logits.shape[0]
            if n_sel == 0:
                row_start = row_end
                continue
            span = slice(selection_start, selection_end)
            p_means = prepared["p_means"][span].numpy()
            p_stds = prepared["p_stds"][span].numpy()

            # Main head (t+1): coarse argmax over the real vocab + fine argmax.
            coarse_main = coarse_logits[:, :vocab].float().argmax(dim=-1)  # [n_sel]
            fine_ids = fine_logits.float().argmax(dim=-1)  # [n_sel]
            main_lrs = decode_logrets(
                coarse_main, fine_ids, tokenizer, device, p_means, p_stds
            )  # [n_sel]

            # Future heads (t+2/3/4): each own coarse argmax + the same fine id.
            future_coarse = torch.stack(
                [future_logits[j][:, :vocab].float().argmax(dim=-1)
                 for j in range(n_future)],
                dim=1,
            )  # [n_sel, n_future]
            future_flat = future_coarse.reshape(-1)  # [n_sel*n_future]
            fine_flat = fine_ids.unsqueeze(1).expand(-1, n_future).reshape(-1)
            p_means_fut = np.repeat(p_means, n_future)
            p_stds_fut = np.repeat(p_stds, n_future)
            decoded_fut = decode_logrets(
                future_flat, fine_flat, tokenizer, device, p_means_fut, p_stds_fut
            ).reshape(n_sel, n_future)  # [n_sel, n_future]

            true_lrs = prepared["true_logrets"][span].numpy()
            symbols = prepared["symbols"][span]
            dates = prepared["date_keys"][span]

            for i in range(n_sel):
                main_lr = float(main_lrs[i])
                main_dir = direction(main_lr)
                fut_lrs = [float(v) for v in decoded_fut[i]]
                fut_dirs = [direction(v) for v in fut_lrs]
                agree = int(sum(1 for d in fut_dirs if d == main_dir))
                all_rows.append({
                    "symbol": symbols[i],
                    "date": dates[i],
                    "true_logret": float(true_lrs[i]),
                    "main_lr": main_lr,
                    "fut_lrs": fut_lrs,
                    "directions": fut_dirs,
                    "agree": agree,
                    "accepted": agree >= args.min_agree,
                })
            row_start = row_end
        if (bi + 1) % 100 == 0:
            print(f"  batch {bi+1}/{len(prepared_batches)}, elapsed={time.monotonic()-started:.1f}s", flush=True)

    print(f"Collected {len(all_rows)} positions in {time.monotonic()-started:.1f}s", flush=True)

    # ---- Aggregate + bootstrap ----------------------------------------------
    results, full_da_by_date, acted_da_by_date = aggregate_per_date(
        all_rows, args.min_agree
    )
    bootstrap_results = paired_block_bootstrap(
        full_da_by_date, acted_da_by_date, args.bootstrap, args.seed
    )
    results.update(bootstrap_results)
    results.update({
        "ckpt": str(args.ckpt),
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "future_offsets": list(model.future_offsets),
    })

    # ---- Output ---------------------------------------------------------------
    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_position.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "date", "true_logret", "main_lr",
                        "fut_lrs", "directions", "accepted"],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                "symbol": r["symbol"],
                "date": r["date"],
                "true_logret": r["true_logret"],
                "main_lr": r["main_lr"],
                "fut_lrs": json.dumps(r["fut_lrs"]),
                "directions": json.dumps(r["directions"]),
                "accepted": int(r["accepted"]),
            })

    print()
    print("=== Branch D consistency filter results ===")
    print(f"  min_agree        : {args.min_agree}  (future offsets {list(model.future_offsets)})")
    print(f"  positions        : {results['n_positions']}  "
          f"accepted={results['n_accepted']}  rejected={results['n_rejected']}  "
          f"coverage={results['coverage']*100:.1f}%")
    print(f"  full_da          : {results['full_da']*100:.2f}%  "
          f"(n_dates={results['n_full_da_dates']})")
    print(f"  acted_da         : {results['acted_da']*100:.2f}%  "
          f"(n_dates={results['n_acted_da_dates']})")
    print(f"  rejected_da      : {results['rejected_da']*100:.2f}%  "
          f"(n_dates={results['n_rejected_da_dates']})")
    print(f"  acted_rank_ic    : {results['acted_rank_ic']:.4f}  "
          f"(n_dates={results['n_acted_ic_dates']})")
    if results["delta_n_dates"] > 0:
        print(f"  delta (acted-full): {results['delta_mean']*100:+.2f}pp  "
              f"95% CI [{results['delta_ci_lo']*100:+.2f}, "
              f"{results['delta_ci_hi']*100:+.2f}]pp  "
              f"significant={results['delta_significant']}  "
              f"(n_dates={results['delta_n_dates']})")
    print()
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
