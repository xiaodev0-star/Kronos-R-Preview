"""Evaluate every saved epoch of one GPT trial efficiently.

The regular windowed evaluator performs one full-sequence model forward for
each validation window.  This script performs one forward per checkpoint and
collects only the positions belonging to all requested windows.  Tokenization
and stock preparation are also shared across checkpoints.

Outputs are resumable: one JSON file and one compressed coarse/fine/joint
distribution sidecar are written per epoch, followed by a summary CSV/JSON and
separate quality/behaviour plots.  No holdout positions are used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import default_study_roots

_, DEFAULT_STUDY_ROOT = default_study_roots("04b-hpo", seed=42)
# Standalone default: full-coverage single-day tiling of the pre-holdout region
# [0, 400). Runners pass --offsets explicitly.
DEFAULT_OFFSETS = tuple(range(0, 400, 1))
PREPARED_CACHE_SCHEMA = 6
TOKEN_DISTRIBUTION_SCHEMA = 3
PREDICTION_RECORD_SCHEMA = 1


def parse_int_spec(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges, preserving order."""
    values: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            step = 1 if end >= start else -1
            expanded = range(start, end + step, step)
        else:
            expanded = (int(part),)
        for item in expanded:
            if item not in seen:
                values.append(item)
                seen.add(item)
    if not values:
        raise argparse.ArgumentTypeError("integer specification is empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast multi-window checkpoint trajectory evaluation."
    )
    parser.add_argument(
        "--trial_dir",
        type=Path,
        default=None,
        help="Trial/config directory. Default resolves the current HPO leader.",
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--prepared_cache_dir",
        type=Path,
        default=None,
        help="Optional shared directory for the ~500 MiB prepared-input cache. "
        "Safe to share across trials when the metadata digest matches.",
    )
    parser.add_argument(
        "--epochs",
        default="",
        help="Comma-separated epochs/ranges. Default evaluates every epoch "
        "listed by model_checkpoints.json.",
    )
    parser.add_argument(
        "--offsets",
        default=",".join(str(value) for value in DEFAULT_OFFSETS),
    )
    parser.add_argument("--n_days", type=int, default=1)
    parser.add_argument("--n_stocks", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_strategy", choices=("random", "shortest"), default="random"
    )
    parser.add_argument("--max_collapse_rate", type=float, default=0.35)
    parser.add_argument("--min_unique_tokens", type=int, default=32)
    parser.add_argument(
        "--reference_epoch",
        type=int,
        default=0,
        help="Compare this epoch against existing formal window JSONs; 0 uses "
        "the last indexed epoch. Missing reference files are reported and skipped.",
    )
    parser.add_argument(
        "--no_reference_check",
        action="store_true",
        help="Skip equivalence check against existing formal evaluation outputs.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--rebuild_prepared_cache",
        action="store_true",
        help="Rebuild the persistent tokenized/padded inference input cache.",
    )
    parser.add_argument(
        "--experiment_label",
        default="Exp 04-B",
        help="Label used in generated plot titles.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    replace_with_retry(temporary, path)


def replace_with_retry(temporary: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trial_dir(args: argparse.Namespace) -> Path:
    if args.trial_dir is not None:
        return args.trial_dir.resolve()
    leaderboard_path = DEFAULT_STUDY_ROOT / "leaderboard.json"
    if not leaderboard_path.is_file():
        raise FileNotFoundError(
            "No --trial_dir was supplied and the refreshed HPO leaderboard "
            f"does not exist: {leaderboard_path}"
        )
    leaderboard = load_json(leaderboard_path)
    rows = leaderboard.get("rows", [])
    if not rows:
        raise RuntimeError(f"HPO leaderboard has no completed rows: {leaderboard_path}")
    return (DEFAULT_STUDY_ROOT / "trials" / rows[0]["tid"]).resolve()


def resolve_tokenizer(args: argparse.Namespace, trial_dir: Path) -> Path:
    if args.tokenizer is not None:
        return args.tokenizer.resolve()
    manifest_path = trial_dir.parents[1] / "study_manifest.json"
    manifest = load_json(manifest_path)
    tokenizer = manifest["settings"]["tokenizer"]
    if isinstance(tokenizer, dict):
        tokenizer = tokenizer["path"]
    return Path(tokenizer).resolve()


def make_settings(
    args: argparse.Namespace,
    trial_dir: Path,
    tokenizer_path: Path,
    offsets: list[int],
) -> dict[str, Any]:
    override_path = (trial_dir / "override.json").resolve()
    return {
        "trial_dir": str(trial_dir),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "override_path": str(override_path),
        "override_sha256": file_sha256(override_path),
        "seed": args.seed,
        "offsets": offsets,
        "n_days": args.n_days,
        "n_stocks": args.n_stocks,
        "batch_size": args.batch_size,
        "sample_strategy": args.sample_strategy,
        "max_collapse_rate": args.max_collapse_rate,
        "min_unique_tokens": args.min_unique_tokens,
        "evaluator_schema": 5,
        "token_distribution_schema": TOKEN_DISTRIBUTION_SCHEMA,
        "prediction_record_schema": PREDICTION_RECORD_SCHEMA,
        "holdout_used": False,
    }


def cache_matches(
    payload: dict[str, Any],
    settings: dict[str, Any],
    epoch: int,
    checkpoint: Path,
) -> bool:
    checkpoint_meta = payload.get("checkpoint", {})
    distribution_meta = payload.get("token_distributions", {})
    distribution_path = distribution_meta.get("path")
    records_meta = payload.get("prediction_records", {})
    records_path = records_meta.get("path")
    targets_path = records_meta.get("target_table", {}).get("path")
    return (
        payload.get("status") == "completed"
        and payload.get("epoch") == epoch
        and payload.get("settings") == settings
        and checkpoint_meta.get("path") == str(checkpoint.resolve())
        and checkpoint_meta.get("size_bytes") == checkpoint.stat().st_size
        and payload.get("aggregate", {}).get("n_dates", 0) > 0
        and distribution_meta.get("schema") == TOKEN_DISTRIBUTION_SCHEMA
        and isinstance(distribution_path, str)
        and Path(distribution_path).is_file()
        and records_meta.get("schema") == PREDICTION_RECORD_SCHEMA
        and isinstance(records_path, str)
        and Path(records_path).is_file()
        and isinstance(targets_path, str)
        and Path(targets_path).is_file()
    )


def prepare_stocks(
    *,
    tokenizer: Any,
    device: Any,
    n_stocks: int,
    sample_strategy: str,
    seed: int,
    offsets: list[int],
    n_days: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, int]:
    import numpy as np
    import torch

    from data_processor import load_stocks, split_stocks
    from eval_helpers import (
        _bucket_by_length,
        _prepare_stocks_batch,
        attach_close_prices,
    )

    # Smoke evaluations request the shortest few stocks only to exercise the
    # pipeline. Loading and parsing all 4695 CSVs first adds several minutes
    # without improving that check. Formal/random evaluations still sample
    # from the complete universe.
    load_limit = (
        max(24, n_stocks * 4)
        if sample_strategy == "shortest" and n_stocks > 0
        else 0
    )
    stocks = load_stocks(max_stocks=load_limit)
    _, _, test_stocks = split_stocks(stocks)
    n_sample = (
        len(test_stocks)
        if n_stocks <= 0
        else min(n_stocks, len(test_stocks))
    )
    if sample_strategy == "shortest" and n_stocks > 0:
        test_sample = sorted(
            test_stocks, key=lambda stock: len(stock["features_raw"])
        )[:n_sample]
    else:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(test_stocks), n_sample, replace=False)
        test_sample = [test_stocks[index] for index in sorted(indices)]

    attach_close_prices(test_sample)
    prepped = _prepare_stocks_batch(test_sample, tokenizer, device)
    min_required = min(n_days, 10)
    valid: list[dict[str, Any]] = []

    for stock in prepped:
        selections: list[tuple[int, int, str]] = []
        for window_start in offsets:
            first_position = stock["test_pos"] + window_start
            if first_position + min_required > stock["seq_len"]:
                continue
            for within_window in range(n_days):
                position = first_position + within_window
                if position >= stock["seq_len"]:
                    break
                date_key = (
                    stock["dates_raw"][position]
                    if position < len(stock["dates_raw"])
                    else "unknown"
                )
                selections.append((window_start, position, date_key))
        if not selections:
            continue

        original_length = stock["seq_len"]
        # In exact arithmetic the causal sequence could end immediately after
        # the final requested position.  In BF16, changing N can select a
        # different SDPA tiling and move rare near-tie argmaxes, so retain the
        # historical full length for strict prediction equivalence.
        required_length = original_length
        positions = np.asarray(
            [item[1] for item in selections], dtype=np.int64
        )
        features = stock["feat"]
        coarse_token_ids = stock["coarse_token_ids"]
        fine_token_ids = stock["fine_token_ids"]
        closes = stock["close"]
        p_mean_0 = float(stock["p_mean"][0])
        p_std_0 = float(stock["p_std"][0])

        time_ids = torch.empty(required_length, 3, dtype=torch.long)
        time_ids[:, 0] = torch.as_tensor(
            stock["day"][:required_length], dtype=torch.long
        )
        time_ids[:, 1] = torch.as_tensor(
            stock["month"][:required_length], dtype=torch.long
        )
        time_ids[:, 2] = torch.as_tensor(
            stock["year"][:required_length], dtype=torch.long
        )
        valid.append(
            {
                "symbol": stock["symbol"],
                "seq_len": required_length,
                "input_ids": torch.as_tensor(
                    stock["inp_ids"][:required_length], dtype=torch.long
                ),
                "time_ids": time_ids,
                "va_values": torch.as_tensor(
                    stock["va"][:required_length], dtype=torch.float32
                ),
                "selection_positions": torch.from_numpy(positions),
                "window_starts": torch.as_tensor(
                    [item[0] for item in selections], dtype=torch.int32
                ),
                "date_keys": [item[2] for item in selections],
                "p_mean_0": p_mean_0,
                "p_std_0": p_std_0,
                "true_logret": torch.as_tensor(
                    [float(features[position, 0]) for position in positions],
                    dtype=torch.float64,
                ),
                "true_coarse_ids": torch.as_tensor(
                    [
                        int(coarse_token_ids[position])
                        for position in positions
                    ],
                    dtype=torch.int16,
                ),
                "true_fine_ids": torch.as_tensor(
                    [int(fine_token_ids[position]) for position in positions],
                    dtype=torch.int16,
                ),
                "base_close": torch.as_tensor(
                    [float(closes[position - 1]) for position in positions],
                    dtype=torch.float64,
                ),
                "true_close": torch.as_tensor(
                    [float(closes[position]) for position in positions],
                    dtype=torch.float64,
                ),
            }
        )

    if not valid:
        raise RuntimeError("No stocks have usable observations for the requested windows")
    buckets = _bucket_by_length(valid, tolerance=100)
    packed_batches: list[dict[str, Any]] = []
    for bucket in buckets:
        for start in range(0, len(bucket), batch_size):
            stocks_batch = bucket[start : start + batch_size]
            batch_count = len(stocks_batch)
            max_len = max(stock["seq_len"] for stock in stocks_batch)
            input_ids = torch.zeros(
                batch_count, max_len, dtype=torch.long
            )
            time_ids = torch.zeros(
                batch_count, max_len, 3, dtype=torch.long
            )
            va_values = torch.zeros(
                batch_count, max_len, 2, dtype=torch.float32
            )
            lengths = torch.empty(batch_count, dtype=torch.long)
            selection_ptr = [0]
            selection_rows = []
            selection_positions = []
            window_starts = []
            date_keys: list[str] = []
            symbols: list[str] = []
            p_means = []
            p_stds = []
            true_logrets = []
            true_coarse_ids = []
            true_fine_ids = []
            base_closes = []
            true_closes = []

            for row, stock in enumerate(stocks_batch):
                length = stock["seq_len"]
                lengths[row] = length
                input_ids[row, :length] = stock["input_ids"]
                time_ids[row, :length] = stock["time_ids"]
                va_values[row, :length] = stock["va_values"]
                count = len(stock["selection_positions"])
                selection_rows.append(
                    torch.full((count,), row, dtype=torch.long)
                )
                selection_positions.append(stock["selection_positions"])
                window_starts.append(stock["window_starts"])
                date_keys.extend(stock["date_keys"])
                symbols.extend([stock["symbol"]] * count)
                p_means.append(
                    torch.full(
                        (count,), stock["p_mean_0"], dtype=torch.float64
                    )
                )
                p_stds.append(
                    torch.full(
                        (count,), stock["p_std_0"], dtype=torch.float64
                    )
                )
                true_logrets.append(stock["true_logret"])
                true_coarse_ids.append(stock["true_coarse_ids"])
                true_fine_ids.append(stock["true_fine_ids"])
                base_closes.append(stock["base_close"])
                true_closes.append(stock["true_close"])
                selection_ptr.append(selection_ptr[-1] + count)

            packed_batches.append(
                {
                    "input_ids": input_ids,
                    "time_ids": time_ids,
                    "va_values": va_values,
                    "lengths": lengths,
                    "selection_ptr": torch.as_tensor(
                        selection_ptr, dtype=torch.long
                    ),
                    "selection_rows": torch.cat(selection_rows),
                    "selection_positions": torch.cat(selection_positions),
                    "window_starts": torch.cat(window_starts),
                    "date_keys": date_keys,
                    "symbols": symbols,
                    "p_means": torch.cat(p_means),
                    "p_stds": torch.cat(p_stds),
                    "true_logrets": torch.cat(true_logrets),
                    "true_coarse_ids": torch.cat(true_coarse_ids),
                    "true_fine_ids": torch.cat(true_fine_ids),
                    "base_closes": torch.cat(base_closes),
                    "true_closes": torch.cat(true_closes),
                }
            )
    return packed_batches, len(test_sample), len(valid)


def prepared_cache_metadata(
    settings: dict[str, Any],
) -> dict[str, Any]:
    from config import DataConfig
    from data_processor import _csv_fingerprint

    stock_cache = ROOT / "dataset" / ".cache" / "stocks_all.pkl"
    cache_stat = stock_cache.stat() if stock_cache.exists() else None
    csv_count, csv_total_bytes = _csv_fingerprint(
        DataConfig.data_dir, 0
    )
    override = load_json(Path(settings["override_path"]))
    # Model width/depth and training knobs cannot affect prepared inference
    # inputs.  Fingerprinting only the input-relevant override sections lets
    # controlled architecture sweeps reuse one exact cache instead of writing
    # a ~500 MiB duplicate per model.
    input_override = {
        section: override.get(section, {})
        for section in ("DataConfig", "NormConfig", "TokenizerConfig")
    }
    return {
        "schema": PREPARED_CACHE_SCHEMA,
        "tokenizer_sha256": settings["tokenizer_sha256"],
        "input_override": input_override,
        "seed": settings["seed"],
        "offsets": settings["offsets"],
        "n_days": settings["n_days"],
        "n_stocks": settings["n_stocks"],
        "batch_size": settings["batch_size"],
        "sample_strategy": settings["sample_strategy"],
        "dataset_csv": {
            "file_count": csv_count,
            "total_bytes": csv_total_bytes,
        },
        "stock_cache": (
            {
                "size_bytes": cache_stat.st_size,
                "mtime_ns": cache_stat.st_mtime_ns,
            }
            if cache_stat is not None
            else None
        ),
    }


def prepared_cache_path(
    cache_root: Path,
    metadata: dict[str, Any],
) -> Path:
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return cache_root / f"trajectory_inputs_{digest}.pt"


def save_prepared_cache(
    path: Path,
    metadata: dict[str, Any],
    batches: list[dict[str, Any]],
    sampled_count: int,
    valid_count: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "metadata": metadata,
            "batches": batches,
            "sampled_count": sampled_count,
            "valid_count": valid_count,
        },
        temporary,
    )
    replace_with_retry(temporary, path)


def load_prepared_cache(
    path: Path,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(
            path, map_location="cpu", weights_only=False
        )
    if payload.get("metadata") != metadata:
        raise RuntimeError("prepared inference cache metadata mismatch")
    return (
        payload["batches"],
        int(payload["sampled_count"]),
        int(payload["valid_count"]),
    )


def token_distribution_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"token_distributions_epoch_{epoch:03d}.npz"


def prediction_records_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"prediction_records_epoch_{epoch:03d}.npz"


def evaluation_targets_path(output_dir: Path) -> Path:
    return output_dir / "evaluation_targets.npz"


def _top_count_rows(
    counts: Any,
    *,
    level: str,
    vocab_fine: int,
    limit: int = 50,
) -> list[dict[str, Any]]:
    import numpy as np

    nonzero = np.flatnonzero(counts)
    if nonzero.size == 0:
        return []
    ordered = nonzero[
        np.argsort(counts[nonzero], kind="stable")[::-1][:limit]
    ]
    total = max(int(counts.sum()), 1)
    output: list[dict[str, Any]] = []
    for raw_id in ordered.tolist():
        row: dict[str, Any] = {
            "token_id": int(raw_id),
            "count": int(counts[raw_id]),
            "frequency": float(counts[raw_id] / total),
        }
        if level == "joint":
            row["coarse_id"] = int(raw_id // vocab_fine)
            row["fine_id"] = int(raw_id % vocab_fine)
        output.append(row)
    return output


def write_token_distributions(
    path: Path,
    predictions: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Persist exact true/predicted coarse, fine, and joint count vectors.

    Metrics in the epoch JSON are convenient summaries; this compressed sidecar
    preserves the complete empirical distributions so conclusions about the
    65,536-way joint codebook remain auditable.
    """
    import numpy as np

    offsets = np.asarray(sorted(predictions), dtype=np.int32)
    vocab_coarse = int(tokenizer.bsq_coarse.vocab_size)
    vocab_fine = int(tokenizer.bsq_fine.vocab_size)
    vocab_joint = vocab_coarse * vocab_fine

    arrays: dict[str, Any] = {
        "schema": np.asarray([TOKEN_DISTRIBUTION_SCHEMA], dtype=np.int16),
        "offsets": offsets,
        "vocab_coarse": np.asarray([vocab_coarse], dtype=np.int32),
        "vocab_fine": np.asarray([vocab_fine], dtype=np.int32),
        "vocab_joint": np.asarray([vocab_joint], dtype=np.int32),
    }
    preview: dict[str, Any] = {}
    for level, vocab in (
        ("coarse", vocab_coarse),
        ("fine", vocab_fine),
        ("joint", vocab_joint),
    ):
        pred_key = f"{level}_id" if level != "coarse" else "coarse_id"
        true_key = (
            f"true_{level}_id"
            if level != "coarse"
            else "true_coarse_id"
        )
        pred_counts = []
        true_counts = []
        for offset in offsets.tolist():
            paired = [
                row
                for row in predictions[int(offset)]
                if pred_key in row
                and true_key in row
                and 0 <= int(row[pred_key]) < vocab
                and 0 <= int(row[true_key]) < vocab
            ]
            pred_ids = np.asarray(
                [int(row[pred_key]) for row in paired], dtype=np.int64
            )
            true_ids = np.asarray(
                [int(row[true_key]) for row in paired], dtype=np.int64
            )
            pred_counts.append(
                np.bincount(pred_ids, minlength=vocab).astype(np.int32)
                if pred_ids.size
                else np.zeros(vocab, dtype=np.int32)
            )
            true_counts.append(
                np.bincount(true_ids, minlength=vocab).astype(np.int32)
                if true_ids.size
                else np.zeros(vocab, dtype=np.int32)
            )
        arrays[f"{level}_pred_counts"] = np.stack(pred_counts)
        arrays[f"{level}_true_counts"] = np.stack(true_counts)
        preview[level] = {
            "overall": {
                "pred": _top_count_rows(
                    arrays[f"{level}_pred_counts"].sum(axis=0),
                    level=level,
                    vocab_fine=vocab_fine,
                ),
                "true": _top_count_rows(
                    arrays[f"{level}_true_counts"].sum(axis=0),
                    level=level,
                    vocab_fine=vocab_fine,
                ),
            },
            "by_offset": {
                str(offset): {
                    "pred": _top_count_rows(
                        arrays[f"{level}_pred_counts"][index],
                        level=level,
                        vocab_fine=vocab_fine,
                    ),
                    "true": _top_count_rows(
                        arrays[f"{level}_true_counts"][index],
                        level=level,
                        vocab_fine=vocab_fine,
                    ),
                }
                for index, offset in enumerate(offsets.tolist())
            },
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    replace_with_retry(temporary, path)
    return {
        "schema": TOKEN_DISTRIBUTION_SCHEMA,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "offset_axis": offsets.tolist(),
        "vocab": {
            "coarse": vocab_coarse,
            "fine": vocab_fine,
            "joint": vocab_joint,
        },
        "count_arrays": sorted(
            key for key in arrays if key.endswith("_counts")
        ),
        "count_semantics": "paired predictions with an available true ID",
        "top_50_preview": preview,
    }


def write_prediction_records(
    path: Path,
    targets_path: Path,
    predictions: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Persist every evaluated stock/date prediction in a compact table.

    This sidecar is sufficient to reconstruct per-stock, per-date, per-window,
    coarse/fine/joint distributions and sparse confusion matrices offline.
    """
    import numpy as np

    rows = [
        row
        for offset in sorted(predictions)
        for row in predictions[offset]
    ]
    symbols = sorted({str(row["symbol"]) for row in rows})
    dates = sorted({str(row["date_key"]) for row in rows})
    symbol_index = {value: index for index, value in enumerate(symbols)}
    date_index = {value: index for index, value in enumerate(dates)}

    def integer_array(key: str, dtype: Any, default: int = -1) -> Any:
        return np.asarray(
            [int(row.get(key, default)) for row in rows], dtype=dtype
        )

    target_arrays: dict[str, Any] = {
        "schema": np.asarray([PREDICTION_RECORD_SCHEMA], dtype=np.int16),
        "symbols": np.asarray(symbols),
        "dates": np.asarray(dates),
        "symbol_index": np.asarray(
            [symbol_index[str(row["symbol"])] for row in rows],
            dtype=np.int32,
        ),
        "date_index": np.asarray(
            [date_index[str(row["date_key"])] for row in rows],
            dtype=np.int16,
        ),
        "position": integer_array("position", np.int32),
        "true_coarse_id": integer_array("true_coarse_id", np.int32),
        "true_fine_id": integer_array("true_fine_id", np.int32),
        "true_joint_id": integer_array("true_joint_id", np.int32),
        "true_logret": np.asarray(
            [float(row["true_logret"]) for row in rows], dtype=np.float64
        ),
        "base_close": np.asarray(
            [float(row["base_close"]) for row in rows], dtype=np.float64
        ),
        "true_close": np.asarray(
            [float(row["true_close"]) for row in rows], dtype=np.float64
        ),
    }
    # The offset is the dictionary key rather than a field in each in-memory
    # row; fill it in deterministically without bloating those row dictionaries.
    target_arrays["window_offset"] = np.concatenate(
        [
            np.full(len(predictions[offset]), offset, dtype=np.int16)
            for offset in sorted(predictions)
        ]
    )
    prediction_arrays: dict[str, Any] = {
        "schema": np.asarray([PREDICTION_RECORD_SCHEMA], dtype=np.int16),
        "pred_coarse_id": integer_array("coarse_id", np.int32),
        "pred_fine_id": integer_array("fine_id", np.int32),
        "pred_joint_id": integer_array("joint_id", np.int32),
        "pred_logret": np.asarray(
            [float(row["pred_logret"]) for row in rows], dtype=np.float64
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    targets_temporary = targets_path.with_suffix(
        targets_path.suffix + ".tmp"
    )
    with targets_temporary.open("wb") as handle:
        np.savez_compressed(handle, **target_arrays)
    replace_with_retry(targets_temporary, targets_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **prediction_arrays)
    replace_with_retry(temporary, path)
    return {
        "schema": PREDICTION_RECORD_SCHEMA,
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "n_records": len(rows),
        "n_symbols": len(symbols),
        "n_dates": len(dates),
        "prediction_columns": sorted(prediction_arrays),
        "target_table": {
            "path": str(targets_path.resolve()),
            "filename": targets_path.name,
            "size_bytes": targets_path.stat().st_size,
            "sha256": file_sha256(targets_path),
            "columns": sorted(target_arrays),
        },
        "reconstructable_views": [
            "per-stock predictions",
            "per-date predictions",
            "per-window predictions",
            "coarse/fine/joint distributions",
            "coarse/fine/joint sparse confusion matrices",
        ],
    }


def _flush_pending_ids(
    pending: list[dict[str, Any]],
    predictions: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
    device: Any,
) -> None:
    """Decode a pending chunk and file each prediction under its window.

    ``pending`` holds one entry per stock, each carrying whole arrays rather than
    one dict per prediction: a stock's metadata is a contiguous slice of the
    prepared tensors, so it never needed to be taken apart element by element.
    """
    if not pending:
        return
    import numpy as np
    import torch

    from eval_helpers import decode_code_ids_tensor

    coarse_ids = torch.cat([item["coarse_ids"] for item in pending]).to(device)
    fine_ids = torch.cat([item["fine_ids"] for item in pending]).to(device)
    decoded = decode_code_ids_tensor(
        coarse_ids, fine_ids, tokenizer, device
    ).cpu().numpy()
    coarse_host = coarse_ids.cpu().numpy()
    fine_host = fine_ids.cpu().numpy()
    vocab_fine = int(tokenizer.bsq_fine.vocab_size)

    cursor = 0
    for item in pending:
        count = int(item["coarse_ids"].numel())
        stop = cursor + count
        pred_logret = (
            decoded[cursor:stop, 0].astype(np.float64) * item["p_stds"]
            + item["p_means"]
        )
        window_starts = item["window_starts"]
        date_keys = item["date_keys"]
        symbols = item["symbols"]
        positions = item["positions"]
        true_logrets = item["true_logrets"]
        base_closes = item["base_closes"]
        true_closes = item["true_closes"]
        true_coarse_ids = item.get("true_coarse_ids")
        true_fine_ids = item.get("true_fine_ids")
        for offset in range(count):
            coarse_id = int(coarse_host[cursor + offset])
            fine_id = int(fine_host[cursor + offset])
            prediction = {
                "coarse_id": coarse_id,
                "fine_id": fine_id,
                "joint_id": coarse_id * vocab_fine + fine_id,
                "date_key": date_keys[offset],
                "symbol": symbols[offset],
                "position": int(positions[offset]),
                "pred_logret": float(pred_logret[offset]),
                "true_logret": float(true_logrets[offset]),
                "base_close": float(base_closes[offset]),
                "true_close": float(true_closes[offset]),
            }
            if true_coarse_ids is not None:
                true_coarse_id = int(true_coarse_ids[offset])
                if true_coarse_id >= 0:
                    prediction["true_coarse_id"] = true_coarse_id
                    if true_fine_ids is not None:
                        true_fine_id = int(true_fine_ids[offset])
                        if true_fine_id >= 0:
                            prediction["true_fine_id"] = true_fine_id
                            prediction["true_joint_id"] = (
                                true_coarse_id * vocab_fine + true_fine_id
                            )
            predictions[int(window_starts[offset])].append(prediction)
        cursor = stop
    pending.clear()


def evaluate_prepared(
    *,
    model: Any,
    tokenizer: Any,
    prepared_batches: list[dict[str, Any]],
    device: Any,
    offsets: list[int],
    batch_size: int,
    epoch: int,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    import torch

    from eval_helpers import predict_selected_ids

    predictions: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in offsets
    }
    pending: list[dict[str, Any]] = []
    pending_rows = 0
    decode_batch_size = 10_000
    effective_batch_size = batch_size
    estimated_batches = len(prepared_batches)
    completed_batches = 0
    n_forward = 0
    started = time.monotonic()

    with torch.inference_mode():
        for prepared in prepared_batches:
            row_start = 0
            total_rows = prepared["input_ids"].shape[0]
            selection_ptr = prepared["selection_ptr"]
            while row_start < total_rows:
                row_end = min(
                    row_start + effective_batch_size, total_rows
                )
                batch_count = row_end - row_start
                max_len = int(
                    prepared["lengths"][row_start:row_end].max()
                )
                selection_start = int(selection_ptr[row_start])
                selection_end = int(selection_ptr[row_end])
                try:
                    inp = prepared["input_ids"][
                        row_start:row_end, :max_len
                    ].to(device)
                    tids = prepared["time_ids"][
                        row_start:row_end, :max_len
                    ].to(device)
                    va = prepared["va_values"][
                        row_start:row_end, :max_len
                    ].to(device)
                    positions = torch.arange(max_len, device=device)
                    positions = positions.unsqueeze(0).expand(batch_count, -1)

                    selected_rows = (
                        prepared["selection_rows"][
                            selection_start:selection_end
                        ]
                        - row_start
                    )
                    selected_positions = prepared[
                        "selection_positions"
                    ][selection_start:selection_end]
                    # Only the selected positions are ever read, so the vocabulary
                    # heads run on those rows instead of every position.  Identical
                    # IDs to the full projection; see predict_selected_ids.
                    coarse_ids, fine_ids = predict_selected_ids(
                        model,
                        inp,
                        tids,
                        positions,
                        va,
                        selected_rows,
                        selected_positions,
                        tokenizer,
                        device,
                    )
                    n_forward += 1

                    # A stock's selections are a contiguous span of the prepared
                    # arrays, so each stock is carried as slices instead of being
                    # unpacked into one dict per prediction.  Staying at stock
                    # granularity keeps the historical 10k flush boundary, and
                    # with it the decode matmul's row counts: the decode is not
                    # invariant to chunk size (up to 5e-6 across shapes).
                    coarse_host = coarse_ids.cpu()
                    fine_host = fine_ids.cpu()
                    for original_row in range(row_start, row_end):
                        begin = int(selection_ptr[original_row])
                        stop = int(selection_ptr[original_row + 1])
                        if stop == begin:
                            continue
                        local = slice(begin - selection_start, stop - selection_start)
                        span = slice(begin, stop)
                        pending.append(
                            {
                                "coarse_ids": coarse_host[local],
                                "fine_ids": fine_host[local],
                                # Store selected targets directly: input_ids is
                                # [BOS, tok_0, ..., tok_(T-2)] and cannot recover
                                # the final predictable tok_(T-1).
                                "true_coarse_ids": prepared[
                                    "true_coarse_ids"
                                ][span].numpy(),
                                "true_fine_ids": prepared[
                                    "true_fine_ids"
                                ][span].numpy(),
                                "window_starts": prepared["window_starts"][span].numpy(),
                                "date_keys": prepared["date_keys"][span],
                                "symbols": prepared["symbols"][span],
                                "positions": prepared[
                                    "selection_positions"
                                ][span].numpy(),
                                "p_means": prepared["p_means"][span].numpy(),
                                "p_stds": prepared["p_stds"][span].numpy(),
                                "true_logrets": prepared["true_logrets"][span].numpy(),
                                "base_closes": prepared["base_closes"][span].numpy(),
                                "true_closes": prepared["true_closes"][span].numpy(),
                            }
                        )
                        pending_rows += stop - begin
                        if pending_rows >= decode_batch_size:
                            _flush_pending_ids(
                                pending, predictions, tokenizer, device
                            )
                            pending_rows = 0

                    del (
                        coarse_ids,
                        fine_ids,
                        inp,
                        tids,
                        va,
                        positions,
                    )
                    row_start = row_end
                    completed_batches += 1
                    if completed_batches % 100 == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"  epoch {epoch:02d}: batches "
                            f"{completed_batches}/{estimated_batches}, "
                            f"elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if effective_batch_size <= 1:
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    estimated_batches += math.ceil(
                        (total_rows - row_start)
                        / effective_batch_size
                    )
                    print(
                        f"  OOM: reducing evaluation batch size to "
                        f"{effective_batch_size}",
                        flush=True,
                    )

    _flush_pending_ids(pending, predictions, tokenizer, device)
    return predictions, n_forward


def augment_per_date_metrics(
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    import numpy as np

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["date_key"]].append(prediction)
    for date_key, date_metrics in metrics.get("per_date", {}).items():
        rows = grouped[date_key]
        pred_lrs = np.asarray([row["pred_logret"] for row in rows])
        true_lrs = np.asarray([row["true_logret"] for row in rows])
        base_closes = np.asarray([row["base_close"] for row in rows])
        true_closes = np.asarray([row["true_close"] for row in rows])
        valid = (
            np.isfinite(pred_lrs)
            & np.isfinite(true_lrs)
            & np.isfinite(base_closes)
            & np.isfinite(true_closes)
            & (base_closes > 0)
            & (true_closes > 0)
        )
        eps = 1e-8
        if valid.any():
            pred_prices = base_closes[valid] * np.exp(
                pred_lrs[valid].astype(np.float64)
            )
            date_metrics["mape"] = float(
                np.mean(
                    np.abs(pred_prices - true_closes[valid])
                    / np.maximum(np.abs(true_closes[valid]), eps)
                )
                * 100
            )
            date_metrics["baseline_mape"] = float(
                np.mean(
                    np.abs(base_closes[valid] - true_closes[valid])
                    / np.maximum(np.abs(true_closes[valid]), eps)
                )
                * 100
            )
            date_metrics["ampratio"] = float(
                np.mean(np.abs(pred_lrs[valid]))
                / max(np.mean(np.abs(true_lrs[valid])), eps)
            )


COARSE_CODEBOOK_FIELDS = (
    "coarse_token_accuracy",
    "target_n_unique_tokens",
    "target_collapse_rate",
    "pred_token_entropy_bits",
    "target_token_entropy_bits",
    "pred_effective_tokens",
    "target_effective_tokens",
    "prediction_support_precision",
    "target_support_recall",
    "token_support_f1",
    "token_jsd",
    "distribution_alignment",
    "unique_token_alignment",
    "effective_token_alignment",
    "collapse_alignment",
    "codebook_balance_score",
)
HIERARCHICAL_CODEBOOK_BASE_FIELDS = (
    "token_accuracy",
    "n_unique_tokens",
    "collapse_rate",
    "target_n_unique_tokens",
    "target_collapse_rate",
    "pred_token_entropy_bits",
    "target_token_entropy_bits",
    "pred_effective_tokens",
    "target_effective_tokens",
    "prediction_support_precision",
    "target_support_recall",
    "token_support_f1",
    "token_jsd",
    "distribution_alignment",
    "unique_token_alignment",
    "effective_token_alignment",
    "collapse_alignment",
    "codebook_balance_score",
)


def aggregate_windows(
    windows: dict[int, dict[str, Any]],
    *,
    max_collapse_rate: float,
    min_unique_tokens: int,
) -> dict[str, Any]:
    import numpy as np

    values = list(windows.values())
    weights = np.asarray([item["n_dates"] for item in values], dtype=np.float64)
    if weights.sum() <= 0:
        raise RuntimeError("Evaluation produced zero usable dates")

    def weighted_mean(field: str) -> float:
        return float(
            np.average(
                np.asarray([item[field] for item in values], dtype=np.float64),
                weights=weights,
            )
        )

    daily = [
        date_metrics
        for window in values
        for date_metrics in window["per_date"].values()
    ]
    daily_collapse = np.asarray(
        [item["collapse_rate"] for item in daily], dtype=np.float64
    )
    daily_unique = np.asarray(
        [item["n_unique_tokens"] for item in daily], dtype=np.float64
    )
    window_da = np.asarray(
        [item["avg_da_per_date"] for item in values], dtype=np.float64
    )
    window_ic = np.asarray(
        [item["avg_daily_rank_ic"] for item in values], dtype=np.float64
    )
    window_amp = np.asarray(
        [item["ampratio"] for item in values], dtype=np.float64
    )
    worst_collapse = float(daily_collapse.max())
    minimum_unique = int(daily_unique.min())
    avg_amp = weighted_mean("ampratio")
    aggregate = {
        "healthy": (
            worst_collapse <= max_collapse_rate
            and minimum_unique >= min_unique_tokens
        ),
        "avg_da_per_date": weighted_mean("avg_da_per_date"),
        "min_window_da": float(window_da.min()),
        "window_da_std": float(window_da.std()),
        "avg_da_above_baseline": weighted_mean("avg_da_above_baseline"),
        "avg_daily_rank_ic": weighted_mean("avg_daily_rank_ic"),
        "min_window_daily_rank_ic": float(window_ic.min()),
        "window_daily_rank_ic_std": float(window_ic.std()),
        "avg_mape": weighted_mean("mape"),
        "avg_baseline_mape": weighted_mean("baseline_mape"),
        "avg_ampratio": avg_amp,
        "ampratio_log_error": abs(math.log(avg_amp)) if avg_amp > 0 else None,
        "min_window_ampratio": float(window_amp.min()),
        "max_window_ampratio": float(window_amp.max()),
        "median_daily_collapse_rate": float(np.median(daily_collapse)),
        "p90_daily_collapse_rate": float(
            np.quantile(daily_collapse, 0.90)
        ),
        "worst_daily_collapse_rate": worst_collapse,
        "median_daily_unique_tokens": float(np.median(daily_unique)),
        "min_daily_unique_tokens": minimum_unique,
        "n_windows": len(values),
        "n_dates": int(sum(item["n_dates"] for item in values)),
        "n_predictions": int(
            sum(item["n_predictions"] for item in values)
        ),
        "health_gate": {
            "max_collapse_rate": max_collapse_rate,
            "min_unique_tokens": min_unique_tokens,
        },
    }
    codebook_groups = {
        "coarse": COARSE_CODEBOOK_FIELDS,
        "fine": tuple(
            f"fine_{field}" for field in HIERARCHICAL_CODEBOOK_BASE_FIELDS
        ),
        "joint": tuple(
            f"joint_{field}" for field in HIERARCHICAL_CODEBOOK_BASE_FIELDS
        ),
    }
    for level, codebook_fields in codebook_groups.items():
        if not daily or not all(
            all(field in item for field in codebook_fields)
            for item in daily
        ):
            continue
        for field in codebook_fields:
            field_values = np.asarray(
                [item[field] for item in daily], dtype=np.float64
            )
            aggregate[f"median_daily_{field}"] = float(
                np.median(field_values)
            )
        balance_field = (
            "codebook_balance_score"
            if level == "coarse"
            else f"{level}_codebook_balance_score"
        )
        balance_values = np.asarray(
            [item[balance_field] for item in daily],
            dtype=np.float64,
        )
        aggregate[f"p10_daily_{balance_field}"] = float(
            np.quantile(balance_values, 0.10)
        )
    return aggregate


REFERENCE_FLOAT_KEYS = (
    "avg_da_per_date",
    "avg_daily_rank_ic",
    "rank_ic",
    "ampratio",
    "mape",
    "baseline_mape",
    "collapse_rate",
    "max_daily_collapse_rate",
)
REFERENCE_INT_KEYS = (
    "n_dates",
    "n_predictions",
    "n_unique_tokens",
    "min_daily_unique_tokens",
)


def validate_reference(
    *,
    trial_dir: Path,
    windows: dict[int, dict[str, Any]],
    offsets: list[int],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for offset in offsets:
        reference_path = trial_dir / f"eval_offset_{offset:04d}.json"
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference evaluation is missing: {reference_path}"
            )
        reference = load_json(reference_path)
        candidate = windows[offset]
        float_diffs = {
            key: abs(float(candidate[key]) - float(reference[key]))
            for key in REFERENCE_FLOAT_KEYS
        }
        integer_matches = {
            key: int(candidate[key]) == int(reference[key])
            for key in REFERENCE_INT_KEYS
        }
        max_difference = max(float_diffs.values(), default=0.0)
        passed = (
            max_difference <= tolerance and all(integer_matches.values())
        )
        comparisons[str(offset)] = {
            "passed": passed,
            "max_float_abs_difference": max_difference,
            "float_abs_differences": float_diffs,
            "integer_matches": integer_matches,
            "reference_path": str(reference_path.resolve()),
        }
        if not passed:
            raise RuntimeError(
                f"Optimized evaluator does not match offset {offset}: "
                f"max float diff={max_difference}, ints={integer_matches}"
            )
    return {"passed": True, "windows": comparisons}


_SUMMARY_PREFIX_FIELDS = (
    "epoch",
    "train_loss",
    "train_coarse_loss",
    "train_fine_loss",
    "train_het_loss",
    "val_loss",
    "val_coarse_loss",
    "val_fine_loss",
    "val_het_loss",
    "learning_rate",
    "learning_rate_adam",
    "optimizer_steps_this_epoch",
    "global_step",
    "avg_da_per_date",
    "avg_da_above_baseline",
    "min_window_da",
    "window_da_std",
    "avg_daily_rank_ic",
    "min_window_daily_rank_ic",
    "window_daily_rank_ic_std",
    "avg_mape",
    "avg_baseline_mape",
    "avg_ampratio",
    "ampratio_log_error",
    "min_window_ampratio",
    "max_window_ampratio",
    "median_daily_collapse_rate",
    "p90_daily_collapse_rate",
    "worst_daily_collapse_rate",
    "median_daily_unique_tokens",
    "min_daily_unique_tokens",
)
_COARSE_SUMMARY_FIELDS = tuple(
    f"median_daily_{field}" for field in COARSE_CODEBOOK_FIELDS
) + ("p10_daily_codebook_balance_score",)
_FINE_SUMMARY_FIELDS = tuple(
    f"median_daily_fine_{field}"
    for field in HIERARCHICAL_CODEBOOK_BASE_FIELDS
) + ("p10_daily_fine_codebook_balance_score",)
_JOINT_SUMMARY_FIELDS = tuple(
    f"median_daily_joint_{field}"
    for field in HIERARCHICAL_CODEBOOK_BASE_FIELDS
) + ("p10_daily_joint_codebook_balance_score",)
_SUMMARY_SUFFIX_FIELDS = (
    "healthy",
    "n_dates",
    "n_predictions",
    "elapsed_sec",
)
SUMMARY_FIELDS = (
    _SUMMARY_PREFIX_FIELDS
    + _COARSE_SUMMARY_FIELDS
    + _FINE_SUMMARY_FIELDS
    + _JOINT_SUMMARY_FIELDS
    + _SUMMARY_SUFFIX_FIELDS
)


def flatten_result(payload: dict[str, Any]) -> dict[str, Any]:
    training = payload["training"]
    row = {
        "epoch": payload["epoch"],
        "train_loss": training["train_loss"],
        "train_coarse_loss": training.get("train_coarse_loss"),
        "train_fine_loss": training.get("train_fine_loss"),
        "train_het_loss": training.get("train_het_loss"),
        "val_loss": training["val_loss"],
        "val_coarse_loss": training.get(
            "val_coarse_loss", training["val_loss"]
        ),
        "val_fine_loss": training.get("val_fine_loss"),
        "val_het_loss": training.get("val_het_loss"),
        "learning_rate": training["learning_rate"],
        "learning_rate_adam": training.get("learning_rate_adam"),
        "optimizer_steps_this_epoch": training.get(
            "optimizer_steps_this_epoch"
        ),
        "global_step": training.get("global_step"),
        "elapsed_sec": payload["elapsed_sec"],
    }
    row.update(payload["aggregate"])
    return {key: row.get(key) for key in SUMMARY_FIELDS}


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = [flatten_result(item) for item in sorted(results, key=lambda x: x["epoch"])]
    atomic_write_json(output_dir / "epoch_summary.json", rows)
    temporary = output_dir / "epoch_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temporary, output_dir / "epoch_summary.csv")


def make_plots(
    output_dir: Path,
    results: list[dict[str, Any]],
    experiment_label: str = "Exp 04-B",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [flatten_result(item) for item in sorted(results, key=lambda x: x["epoch"])]
    epochs = np.asarray([row["epoch"] for row in rows])

    def series(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    figure, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(epochs, series("val_loss"), color="0.25", label="Val loss")
    axes[0].set_ylabel("Validation loss")
    axes[0].legend()

    da = series("avg_da_per_date") * 100
    min_da = series("min_window_da") * 100
    axes[1].plot(epochs, da, label="Mean daily DA", color="#1f77b4")
    axes[1].plot(
        epochs, min_da, label="Worst-window DA", color="#1f77b4", alpha=0.45
    )
    axes[1].axhline(50, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_ylabel("DA (%)")
    axes[1].legend()

    axes[2].plot(epochs, series("avg_mape"), label="Model MAPE", color="#d62728")
    axes[2].plot(
        epochs,
        series("avg_baseline_mape"),
        label="No-change MAPE",
        color="0.45",
        linestyle="--",
    )
    axes[2].set_ylabel("MAPE (%)")
    axes[2].legend()

    axes[3].plot(
        epochs,
        series("avg_daily_rank_ic"),
        label="Mean daily RankIC",
        color="#2ca02c",
    )
    axes[3].plot(
        epochs,
        series("min_window_daily_rank_ic"),
        label="Worst-window RankIC",
        color="#2ca02c",
        alpha=0.45,
    )
    axes[3].axhline(0, color="0.5", linestyle="--", linewidth=1)
    axes[3].set_ylabel("RankIC")
    axes[3].set_xlabel("Epoch")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{experiment_label} checkpoint trajectory: prediction quality"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "quality_trajectory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(
        epochs,
        series("median_daily_collapse_rate") * 100,
        label="Median daily collapse",
    )
    axes[0].plot(
        epochs,
        series("p90_daily_collapse_rate") * 100,
        label="P90 daily collapse",
    )
    axes[0].plot(
        epochs,
        series("worst_daily_collapse_rate") * 100,
        label="Worst daily collapse",
        alpha=0.55,
    )
    axes[0].axhline(35, color="#d62728", linestyle="--", label="Health gate")
    axes[0].set_ylabel("Collapse (%)")
    axes[0].legend(ncol=2)

    axes[1].plot(
        epochs,
        series("median_daily_unique_tokens"),
        label="Median daily unique",
        color="#9467bd",
    )
    axes[1].plot(
        epochs,
        series("min_daily_unique_tokens"),
        label="Minimum daily unique",
        color="#9467bd",
        alpha=0.5,
    )
    axes[1].axhline(32, color="#d62728", linestyle="--", label="Health gate")
    axes[1].set_ylabel("Unique coarse tokens")
    axes[1].legend()

    amp = series("avg_ampratio")
    axes[2].plot(epochs, amp, label="Mean AmpRatio", color="#ff7f0e")
    axes[2].fill_between(
        epochs,
        series("min_window_ampratio"),
        series("max_window_ampratio"),
        color="#ff7f0e",
        alpha=0.18,
        label="Window range",
    )
    axes[2].axhline(1, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_ylabel("AmpRatio")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{experiment_label} checkpoint trajectory: collapse and amplitude"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "behaviour_trajectory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    colour = epochs
    points = axes[0].scatter(
        series("median_daily_collapse_rate") * 100,
        series("avg_da_per_date") * 100,
        c=colour,
        cmap="viridis",
    )
    axes[0].set_xlabel("Median daily collapse (%)")
    axes[0].set_ylabel("DA (%)")
    axes[1].scatter(
        series("median_daily_collapse_rate") * 100,
        series("avg_daily_rank_ic"),
        c=colour,
        cmap="viridis",
    )
    axes[1].set_xlabel("Median daily collapse (%)")
    axes[1].set_ylabel("Daily RankIC")
    axes[2].scatter(
        series("avg_ampratio"),
        series("avg_mape"),
        c=colour,
        cmap="viridis",
    )
    axes[2].axvline(1, color="0.5", linestyle="--", linewidth=1)
    axes[2].set_xlabel("AmpRatio")
    axes[2].set_ylabel("MAPE (%)")
    figure.colorbar(points, ax=axes, label="Epoch", shrink=0.85)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{experiment_label} checkpoint trade-offs (no composite score)"
    )
    figure.savefig(
        output_dir / "tradeoff_scatter.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    trial_dir = resolve_trial_dir(args)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else trial_dir / "epoch_trajectory"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_index = load_json(trial_dir / "model_checkpoints.json")
    checkpoint_rows = {
        int(item["epoch"]): item
        for item in checkpoint_index.get("checkpoints", [])
    }
    available_epochs = sorted(checkpoint_rows)
    if not available_epochs:
        raise RuntimeError(f"No indexed checkpoints found in {trial_dir}")
    epochs = (
        parse_int_spec(args.epochs)
        if args.epochs.strip()
        else available_epochs
    )
    reference_epoch = args.reference_epoch or available_epochs[-1]
    offsets = parse_int_spec(args.offsets)
    if min(epochs) < 1:
        raise ValueError("Epoch numbers must be >= 1")
    if min(offsets) < 0:
        raise ValueError("Offsets must be >= 0")
    if args.n_days <= 0 or args.batch_size <= 0:
        raise ValueError("n_days and batch_size must be positive")

    tokenizer_path = resolve_tokenizer(args, trial_dir)
    override_path = (trial_dir / "override.json").resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if not override_path.is_file():
        raise FileNotFoundError(override_path)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(override_path)

    import torch

    from config import ModelConfig, set_global_seed
    from eval_helpers import compute_windowed_metrics, load_gpt, load_tokenizer

    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = make_settings(
        args, trial_dir, tokenizer_path, offsets
    )
    missing_epochs = [epoch for epoch in epochs if epoch not in checkpoint_rows]
    if missing_epochs:
        raise RuntimeError(f"Checkpoint index is missing epochs: {missing_epochs}")

    print(f"Device: {device}", flush=True)
    print(f"Trial: {trial_dir}", flush=True)
    print(f"Epoch order: {epochs}", flush=True)
    print(f"Validation windows: {offsets}, days/window={args.n_days}", flush=True)
    print("Loading tokenizer and preparing stocks once...", flush=True)
    tokenizer = load_tokenizer(str(tokenizer_path), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    prep_started = time.monotonic()
    cache_metadata = prepared_cache_metadata(settings)
    prepared_cache_root = (
        args.prepared_cache_dir.resolve()
        if args.prepared_cache_dir is not None
        else trial_dir / "cache"
    )
    input_cache_path = prepared_cache_path(prepared_cache_root, cache_metadata)
    cache_hit = False
    if input_cache_path.exists() and not args.rebuild_prepared_cache:
        try:
            prepared_batches, sampled_count, valid_count = (
                load_prepared_cache(input_cache_path, cache_metadata)
            )
            cache_hit = True
        except Exception as exc:
            print(
                f"  Prepared-input cache ignored: {type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )
    if not cache_hit:
        prepared_batches, sampled_count, valid_count = prepare_stocks(
            tokenizer=tokenizer,
            device=device,
            n_stocks=args.n_stocks,
            sample_strategy=args.sample_strategy,
            seed=args.seed,
            offsets=offsets,
            n_days=args.n_days,
            batch_size=args.batch_size,
        )
        save_prepared_cache(
            input_cache_path,
            cache_metadata,
            prepared_batches,
            sampled_count,
            valid_count,
        )
    preparation_sec = time.monotonic() - prep_started
    print(
        f"Prepared {valid_count}/{sampled_count} stocks in "
        f"{preparation_sec:.1f}s ({len(prepared_batches)} eval batches, "
        f"cache={'hit' if cache_hit else 'built'})",
        flush=True,
    )

    manifest = {
        "status": "running",
        "settings": settings,
        "epochs_requested": epochs,
        "preparation_sec": preparation_sec,
        "sampled_stocks": sampled_count,
        "valid_stocks": valid_count,
        "prepared_input_cache": str(input_cache_path.resolve()),
        "prepared_input_cache_hit": cache_hit,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "started_at_unix": time.time(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    completed_results: list[dict[str, Any]] = []
    model = None
    run_started = time.monotonic()
    for sequence_index, epoch in enumerate(epochs, start=1):
        checkpoint = Path(checkpoint_rows[epoch]["path"]).resolve()
        output_path = output_dir / f"epoch_{epoch:03d}.json"
        if (
            not args.force
            and output_path.exists()
            and cache_matches(load_json(output_path), settings, epoch, checkpoint)
        ):
            payload = load_json(output_path)
            completed_results.append(payload)
            print(
                f"[{sequence_index}/{len(epochs)}] epoch {epoch:02d}: "
                "reusing cached result",
                flush=True,
            )
            continue

        print(
            f"[{sequence_index}/{len(epochs)}] epoch {epoch:02d}: "
            f"loading {checkpoint.name}",
            flush=True,
        )
        if model is None:
            model = load_gpt(str(checkpoint), device, tokenizer=tokenizer)
        else:
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(
                checkpoint_payload["model_state_dict"], strict=False
            )
            model.eval()
            del checkpoint_payload

        epoch_started = time.monotonic()
        raw_predictions, n_forward = evaluate_prepared(
            model=model,
            tokenizer=tokenizer,
            prepared_batches=prepared_batches,
            device=device,
            offsets=offsets,
            batch_size=args.batch_size,
            epoch=epoch,
        )
        windows: dict[int, dict[str, Any]] = {}
        for offset in offsets:
            metrics = compute_windowed_metrics(raw_predictions[offset])
            augment_per_date_metrics(raw_predictions[offset], metrics)
            windows[offset] = metrics
        aggregate = aggregate_windows(
            windows,
            max_collapse_rate=args.max_collapse_rate,
            min_unique_tokens=args.min_unique_tokens,
        )
        distribution_metadata = write_token_distributions(
            token_distribution_path(output_dir, epoch),
            raw_predictions,
            tokenizer,
        )
        prediction_record_metadata = write_prediction_records(
            prediction_records_path(output_dir, epoch),
            evaluation_targets_path(output_dir),
            raw_predictions,
        )
        reference_validation = None
        if epoch == reference_epoch and not args.no_reference_check:
            references_exist = all(
                (trial_dir / f"eval_offset_{offset:04d}.json").is_file()
                for offset in offsets
            )
            if references_exist:
                reference_validation = validate_reference(
                    trial_dir=trial_dir,
                    windows=windows,
                    offsets=offsets,
                )
                print("  Reference equivalence check passed.", flush=True)
            else:
                print(
                    "  Reference outputs are absent; equivalence check skipped.",
                    flush=True,
                )

        elapsed_sec = time.monotonic() - epoch_started
        index_row = checkpoint_rows[epoch]
        payload = {
            "status": "completed",
            "epoch": epoch,
            "checkpoint": {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "training": {
                "train_loss": float(index_row["train_loss"]),
                "train_coarse_loss": float(
                    index_row.get(
                        "train_coarse_loss", index_row["train_loss"]
                    )
                ),
                "train_fine_loss": (
                    float(index_row["train_fine_loss"])
                    if index_row.get("train_fine_loss") is not None
                    else None
                ),
                "train_het_loss": (
                    float(index_row["train_het_loss"])
                    if index_row.get("train_het_loss") is not None
                    else None
                ),
                "val_loss": float(index_row["val_loss"]),
                "val_coarse_loss": float(
                    index_row.get("val_coarse_loss", index_row["val_loss"])
                ),
                "val_fine_loss": (
                    float(index_row["val_fine_loss"])
                    if index_row.get("val_fine_loss") is not None
                    else None
                ),
                "val_het_loss": (
                    float(index_row["val_het_loss"])
                    if index_row.get("val_het_loss") is not None
                    else None
                ),
                "learning_rate": float(index_row["learning_rate"]),
                "learning_rate_adam": index_row.get(
                    "learning_rate_adam"
                ),
                "optimizer_steps_this_epoch": int(
                    index_row.get("optimizer_steps_this_epoch", 0)
                ),
                "global_step": int(index_row.get("global_step", 0)),
                "best_so_far": bool(index_row["best_so_far"]),
            },
            "settings": settings,
            "aggregate": aggregate,
            "windows": {str(key): value for key, value in windows.items()},
            "token_distributions": distribution_metadata,
            "prediction_records": prediction_record_metadata,
            "reference_validation": reference_validation,
            "n_forward_passes": n_forward,
            "elapsed_sec": elapsed_sec,
        }
        atomic_write_json(output_path, payload)
        completed_results.append(payload)
        write_summary(output_dir, completed_results)
        print(
            f"  epoch {epoch:02d} done in {elapsed_sec:.1f}s | "
            f"DA={aggregate['avg_da_per_date'] * 100:.2f}% "
            f"MAPE={aggregate['avg_mape']:.3f}% "
            f"IC={aggregate['avg_daily_rank_ic']:.4f} "
            f"collapse(med/p90)="
            f"{aggregate['median_daily_collapse_rate'] * 100:.1f}/"
            f"{aggregate['p90_daily_collapse_rate'] * 100:.1f}% "
            f"amp={aggregate['avg_ampratio']:.3f} "
            f"joint_balance="
            f"{aggregate.get('median_daily_joint_codebook_balance_score', float('nan')):.3f}",
            flush=True,
        )
        del raw_predictions, windows
        gc.collect()
        # Keep the CUDA allocator warm across checkpoints.  The model object
        # and input shapes are reused; empty_cache() here only forces costly
        # reallocation on the next epoch.

    sorted_results = sorted(completed_results, key=lambda item: item["epoch"])
    write_summary(output_dir, sorted_results)
    make_plots(output_dir, sorted_results, args.experiment_label)
    manifest.update(
        {
            "status": "completed",
            "completed_at_unix": time.time(),
            "evaluation_sec": time.monotonic() - run_started,
            "epochs_completed": [item["epoch"] for item in sorted_results],
            "reference_check_passed": any(
                item.get("reference_validation", {}).get("passed", False)
                for item in sorted_results
                if item.get("reference_validation")
            ),
        }
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(f"Completed. Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
