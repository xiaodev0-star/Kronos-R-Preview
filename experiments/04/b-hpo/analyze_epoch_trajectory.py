"""Reproducible, non-composite analysis of an Exp 04-B epoch trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import default_study_roots

_, DEFAULT_STUDY_ROOT = default_study_roots("04b-hpo", seed=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory_dir", type=Path, default=None,
        help="Default resolves the current refreshed-HPO leader trajectory",
    )
    parser.add_argument(
        "--reference_epoch", type=int, default=0,
        help="0 uses the selected mature-window representative epoch",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="Comma-separated epochs; default chooses representative points",
    )
    parser.add_argument("--block_days", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
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


def resolve_trajectory_dir(raw: Path | None) -> Path:
    if raw is not None:
        return raw.resolve()
    leaderboard_path = DEFAULT_STUDY_ROOT / "leaderboard.json"
    leaderboard = load_json(leaderboard_path)
    rows = leaderboard.get("rows", [])
    if not rows:
        raise RuntimeError(f"No completed HPO rows in {leaderboard_path}")
    return (
        DEFAULT_STUDY_ROOT
        / "trials"
        / rows[0]["tid"]
        / "epoch_trajectory"
    ).resolve()


def load_epoch(directory: Path, epoch: int) -> dict[str, Any]:
    return load_json(directory / f"epoch_{epoch:03d}.json")


def per_date_array(
    payload: dict[str, Any], offset: str, metric: str
) -> np.ndarray:
    values = payload["windows"][offset]["per_date"]
    return np.asarray(
        [values[date][metric] for date in sorted(values)],
        dtype=np.float64,
    )


def bootstrap_indices(
    reference: dict[str, Any],
    *,
    samples: int,
    block_days: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap separately inside each validation window."""
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for offset, window in reference["windows"].items():
        n_dates = len(window["per_date"])
        n_blocks = math.ceil(n_dates / block_days)
        starts = rng.integers(0, n_dates, size=(samples, n_blocks))
        indices = (
            starts[:, :, None]
            + np.arange(block_days, dtype=np.int64)[None, None, :]
        ) % n_dates
        output[offset] = indices.reshape(samples, -1)[:, :n_dates]
    return output


def paired_mean_difference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    indices: dict[str, np.ndarray],
    metric: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    point_values: list[float] = []
    window_bootstrap: list[np.ndarray] = []
    for offset in sorted(reference["windows"]):
        candidate_values = per_date_array(candidate, offset, metric)
        reference_values = per_date_array(reference, offset, metric)
        if transform is not None:
            candidate_values = transform(candidate_values)
            reference_values = transform(reference_values)
        differences = candidate_values - reference_values
        point_values.extend(differences.tolist())
        window_bootstrap.append(
            differences[indices[offset]].mean(axis=1)
        )
    bootstrap = np.stack(window_bootstrap, axis=1).mean(axis=1)
    return {
        "difference": float(np.mean(point_values)),
        "ci95": [
            float(value)
            for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def paired_median_difference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    indices: dict[str, np.ndarray],
    metric: str,
) -> dict[str, Any]:
    candidate_samples: list[np.ndarray] = []
    reference_samples: list[np.ndarray] = []
    candidate_point: list[np.ndarray] = []
    reference_point: list[np.ndarray] = []
    for offset in sorted(reference["windows"]):
        candidate_values = per_date_array(candidate, offset, metric)
        reference_values = per_date_array(reference, offset, metric)
        candidate_point.append(candidate_values)
        reference_point.append(reference_values)
        candidate_samples.append(candidate_values[indices[offset]])
        reference_samples.append(reference_values[indices[offset]])
    bootstrap = np.median(
        np.concatenate(candidate_samples, axis=1), axis=1
    ) - np.median(
        np.concatenate(reference_samples, axis=1), axis=1
    )
    point = np.median(np.concatenate(candidate_point)) - np.median(
        np.concatenate(reference_point)
    )
    return {
        "difference": float(point),
        "ci95": [
            float(value)
            for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def daily_descriptives(payload: dict[str, Any]) -> dict[str, float]:
    daily = [
        value
        for window in payload["windows"].values()
        for value in window["per_date"].values()
    ]
    collapse = np.asarray(
        [value["collapse_rate"] for value in daily], dtype=np.float64
    )
    unique = np.asarray(
        [value["n_unique_tokens"] for value in daily], dtype=np.float64
    )
    amplitude_error = np.asarray(
        [
            abs(math.log(max(value["ampratio"], 1e-12)))
            for value in daily
        ],
        dtype=np.float64,
    )
    output = {
        "mean_daily_collapse_rate": float(collapse.mean()),
        "median_daily_collapse_rate": float(np.median(collapse)),
        "p90_daily_collapse_rate": float(np.quantile(collapse, 0.90)),
        "worst_daily_collapse_rate": float(collapse.max()),
        "mean_daily_unique_tokens": float(unique.mean()),
        "median_daily_unique_tokens": float(np.median(unique)),
        "min_daily_unique_tokens": int(unique.min()),
        "mean_daily_ampratio_log_error": float(amplitude_error.mean()),
    }
    optional = {
        "codebook_balance_score": "mean_daily_codebook_balance_score",
        "target_support_recall": "mean_daily_target_support_recall",
        "pred_effective_tokens": "mean_daily_pred_effective_tokens",
        "token_jsd": "mean_daily_token_jsd",
    }
    for source, destination in optional.items():
        values = [
            float(value[source]) for value in daily if source in value
        ]
        if values:
            output[destination] = float(np.mean(values))
    return output


def correlations(
    rows: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["epoch"] <= end]
    val_loss = np.asarray([row["val_loss"] for row in selected])
    metrics = (
        "avg_da_per_date",
        "avg_mape",
        "avg_daily_rank_ic",
        "median_daily_collapse_rate",
        "p90_daily_collapse_rate",
        "avg_ampratio",
        "median_daily_unique_tokens",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in selected])
        result = spearmanr(val_loss, values)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
        output[metric] = {
            "spearman_rho_with_val_loss": statistic,
            "pvalue": pvalue,
        }
    return {
        "epoch_range": [start, end],
        "n_epochs": len(selected),
        "metrics": output,
    }


def extrema(
    rows: list[dict[str, Any]], metric: str, maximize: bool
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row[metric], reverse=maximize)
    return {
        "metric": metric,
        "direction": "max" if maximize else "min",
        "top5": [
            {"epoch": row["epoch"], "value": row[metric]}
            for row in ordered[:5]
        ],
    }


def main() -> int:
    args = parse_args()
    directory = resolve_trajectory_dir(args.trajectory_dir)
    rows = load_json(directory / "epoch_summary.json")
    if not rows:
        raise RuntimeError(f"No evaluated epochs in {directory}")
    available = {int(row["epoch"]) for row in rows}
    final_epoch = max(available)
    result_path = directory.parent / "result.json"
    validation_aggregate = {}
    if result_path.is_file():
        validation_aggregate = load_json(result_path).get(
            "validation", {}
        ).get("aggregate", {})
    mature_window = validation_aggregate.get(
        "mature_window_epochs",
        [max(1, final_epoch - min(4, final_epoch - 1)), final_epoch],
    )
    representative_epoch = int(
        validation_aggregate.get(
            "representative_epoch",
            (int(mature_window[0]) + int(mature_window[1])) // 2,
        )
    )
    reference_epoch = args.reference_epoch or representative_epoch
    if reference_epoch not in available:
        raise ValueError(f"Reference epoch {reference_epoch} is not available")
    if args.candidates.strip():
        candidates = [
            int(value.strip())
            for value in args.candidates.split(",")
            if value.strip()
        ]
    else:
        candidates = sorted(
            {
                value
                for value in (
                    1,
                    3,
                    5,
                    10,
                    15,
                    20,
                    int(mature_window[0]),
                    representative_epoch,
                    int(mature_window[1]),
                    final_epoch,
                )
                if value in available
            }
        )
    missing = sorted(set(candidates) - available)
    if missing:
        raise ValueError(f"Candidate epochs are unavailable: {missing}")
    reference = load_epoch(directory, reference_epoch)
    indices = bootstrap_indices(
        reference,
        samples=args.bootstrap_samples,
        block_days=args.block_days,
        seed=args.seed,
    )

    comparisons: list[dict[str, Any]] = []
    for epoch in candidates:
        payload = load_epoch(directory, epoch)
        versus_reference = {
            "reference_epoch": reference_epoch,
            "daily_da": paired_mean_difference(
                payload, reference, indices, "da"
            ),
            "daily_rank_ic": paired_mean_difference(
                payload, reference, indices, "rank_ic"
            ),
            "daily_mape": paired_mean_difference(
                payload, reference, indices, "mape"
            ),
            "mean_daily_collapse_rate": paired_mean_difference(
                payload, reference, indices, "collapse_rate"
            ),
            "median_daily_collapse_rate": paired_median_difference(
                payload, reference, indices, "collapse_rate"
            ),
            "daily_unique_tokens": paired_mean_difference(
                payload, reference, indices, "n_unique_tokens"
            ),
            "daily_ampratio_log_error": paired_mean_difference(
                payload,
                reference,
                indices,
                "ampratio",
                transform=lambda value: np.abs(
                    np.log(np.maximum(value, 1e-12))
                ),
            ),
        }
        first_offset = next(iter(reference["windows"]))
        first_daily = next(
            iter(reference["windows"][first_offset]["per_date"].values())
        )
        for name, metric in (
            ("daily_codebook_balance", "codebook_balance_score"),
            ("daily_target_support_recall", "target_support_recall"),
            ("daily_effective_tokens", "pred_effective_tokens"),
            ("daily_token_jsd", "token_jsd"),
        ):
            if metric in first_daily:
                versus_reference[name] = paired_mean_difference(
                    payload, reference, indices, metric
                )
        comparison = {
            "epoch": epoch,
            "training": payload["training"],
            "aggregate": payload["aggregate"],
            "daily": daily_descriptives(payload),
            "versus_reference": versus_reference,
        }
        comparisons.append(comparison)

    report = {
        "method": {
            "composite_score_used": False,
            "reference_epoch": reference_epoch,
            "selected_mature_window": mature_window,
            "bootstrap_samples": args.bootstrap_samples,
            "block_days": args.block_days,
            "bootstrap_seed": args.seed,
            "bootstrap_stratified_by_validation_window": True,
            "holdout_used": False,
        },
        "health": {
            "healthy_epoch_count": sum(
                bool(row["healthy"]) for row in rows
            ),
            "best_worst_daily_collapse": extrema(
                rows, "worst_daily_collapse_rate", maximize=False
            ),
            "best_min_daily_unique": extrema(
                rows, "min_daily_unique_tokens", maximize=True
            ),
        },
        "quality_extrema": {
            "da": extrema(rows, "avg_da_per_date", maximize=True),
            "mape": extrema(rows, "avg_mape", maximize=False),
            "rank_ic": extrema(
                rows, "avg_daily_rank_ic", maximize=True
            ),
            "worst_window_da": extrema(
                rows, "min_window_da", maximize=True
            ),
        },
        "behaviour_extrema": {
            "median_collapse": extrema(
                rows, "median_daily_collapse_rate", maximize=False
            ),
            "p90_collapse": extrema(
                rows, "p90_daily_collapse_rate", maximize=False
            ),
            "median_unique": extrema(
                rows, "median_daily_unique_tokens", maximize=True
            ),
        },
        "loss_correlations": {
            "all_epochs": correlations(rows, 1, final_epoch),
            "early": correlations(rows, 1, min(10, final_epoch)),
            "middle": correlations(
                rows,
                min(11, final_epoch),
                max(min(11, final_epoch), final_epoch - 10),
            ),
            "mature": correlations(
                rows, int(mature_window[0]), int(mature_window[1])
            ),
        },
        "candidate_comparisons": comparisons,
    }
    atomic_write_json(directory / "analysis.json", report)

    fields = (
        "epoch",
        "da",
        "mape",
        "rank_ic",
        "ampratio",
        "median_collapse",
        "p90_collapse",
        "worst_collapse",
        "median_unique",
        "min_unique",
        "da_delta_vs_reference",
        "da_ci_low",
        "da_ci_high",
        "rank_ic_delta_vs_reference",
        "mape_delta_vs_reference",
        "mean_collapse_delta_vs_reference",
        "mean_unique_delta_vs_reference",
        "ampratio_log_error_delta_vs_reference",
    )
    temporary = directory / "candidate_comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            aggregate = item["aggregate"]
            daily = item["daily"]
            paired = item["versus_reference"]
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "da": aggregate["avg_da_per_date"],
                    "mape": aggregate["avg_mape"],
                    "rank_ic": aggregate["avg_daily_rank_ic"],
                    "ampratio": aggregate["avg_ampratio"],
                    "median_collapse": daily[
                        "median_daily_collapse_rate"
                    ],
                    "p90_collapse": daily["p90_daily_collapse_rate"],
                    "worst_collapse": daily[
                        "worst_daily_collapse_rate"
                    ],
                    "median_unique": daily[
                        "median_daily_unique_tokens"
                    ],
                    "min_unique": daily["min_daily_unique_tokens"],
                    "da_delta_vs_reference": paired["daily_da"]["difference"],
                    "da_ci_low": paired["daily_da"]["ci95"][0],
                    "da_ci_high": paired["daily_da"]["ci95"][1],
                    "rank_ic_delta_vs_reference": paired[
                        "daily_rank_ic"
                    ]["difference"],
                    "mape_delta_vs_reference": paired["daily_mape"][
                        "difference"
                    ],
                    "mean_collapse_delta_vs_reference": paired[
                        "mean_daily_collapse_rate"
                    ]["difference"],
                    "mean_unique_delta_vs_reference": paired[
                        "daily_unique_tokens"
                    ]["difference"],
                    "ampratio_log_error_delta_vs_reference": paired[
                        "daily_ampratio_log_error"
                    ]["difference"],
                }
            )
    replace_with_retry(temporary, directory / "candidate_comparison.csv")
    print(f"Saved analysis to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
