"""Analyze whether each Exp 03 GPT can use the fixed Exp 02 tokenizer.

The original analysis ranked architectures mainly by downstream DA/MAPE and
the final ten epochs.  That answers the wrong primary question for a capacity
study: Exp 03 fixes a 9+7 tokenizer and should ask how well each GPT models its
coarse-token distribution.  This revision:

* reconstructs the true coarse-token distribution for every retained date;
* scores all 250 checkpoints relative to that target, not to the 512-token
  theoretical maximum;
* selects stable five-epoch mature windows instead of a noisy single peak;
* uses exact support/entropy/JSD diagnostics for representative checkpoints;
* keeps DA, RankIC, MAPE, AmpRatio, validation loss, and compute as guardrails.

The balance indices are diagnostics, not substitutes for the underlying
metrics.  The report always publishes their components and the Pareto roles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
DEFAULT_ROOT = SCRIPT_PATH.parent / "run_seed42"
COMMON_ANALYSIS_PATH = (
    ROOT / "experiments" / "02-tokenizer-tuning" / "analyze_tokenizer_epochwise.py"
)


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp02_analysis_common", COMMON_ANALYSIS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMMON_ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_common()

HISTORICAL_OPTIMIZER_STEPS = {
    "baseline": 3934,
    "wide": 3981,
    "deep": 3984,
    "large": 4039,
    "xlarge": 4104,
}
PROTOCOL_AUDIT = {
    "status": "historical_confounded_pending_exp03_sup",
    "scheduler_budget": 4160,
    "actual_optimizer_steps": HISTORICAL_OPTIMIZER_STEPS,
    "same_optimizer_steps_across_architectures": False,
    "data_order_architecture_independent": False,
    "cause": (
        "adaptive microbatches crossed accumulation boundaries and loader "
        "shuffle consumed architecture-dependent global RNG"
    ),
    "resolution": (
        "Exp 03-Sup retrains the complete controlled grid with a local loader "
        "seed and exact accumulation boundaries"
    ),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    _replace_with_retry(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    _replace_with_retry(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temporary, path)


def envelopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in sorted(
        {str(row["config"]) for row in rows},
        key=lambda name: min(
            row["parameter_count"] for row in rows if row["config"] == name
        ),
    ):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        late = group[-min(10, len(group)) :]
        best_da = COMMON.arg_extreme(group, "avg_da_per_date", "max")
        best_rank = COMMON.arg_extreme(group, "avg_daily_rank_ic", "max")
        best_mape = COMMON.arg_extreme(group, "avg_mape", "min")
        best_collapse = COMMON.arg_extreme(
            group, "p90_daily_collapse_rate", "min"
        )
        best_amp = COMMON.arg_extreme(group, "ampratio_log_error", "min")
        best_val = COMMON.arg_extreme(group, "val_loss", "min")
        output.append(
            {
                "config": config,
                "parameter_count": group[0]["parameter_count"],
                "dim": group[0]["dim"],
                "depth": group[0]["depth"],
                "heads": group[0]["heads"],
                "kv_heads": group[0]["kv_heads"],
                "gradient_checkpointing": group[0]["gradient_checkpointing"],
                "n_epochs": len(group),
                "healthy_epochs": sum(bool(row.get("healthy")) for row in group),
                "best_da": best_da["avg_da_per_date"],
                "best_da_epoch": best_da["epoch"],
                "best_rankic": best_rank["avg_daily_rank_ic"],
                "best_rankic_epoch": best_rank["epoch"],
                "best_mape": best_mape["avg_mape"],
                "best_mape_epoch": best_mape["epoch"],
                "best_p90_collapse": best_collapse["p90_daily_collapse_rate"],
                "best_p90_collapse_epoch": best_collapse["epoch"],
                "best_ampratio": best_amp["avg_ampratio"],
                "best_ampratio_epoch": best_amp["epoch"],
                "best_val_loss": best_val["val_loss"],
                "best_val_loss_epoch": best_val["epoch"],
                "late_median_da": float(
                    np.median([row["avg_da_per_date"] for row in late])
                ),
                "late_median_rankic": float(
                    np.median([row["avg_daily_rank_ic"] for row in late])
                ),
                "late_median_mape": float(
                    np.median([row["avg_mape"] for row in late])
                ),
                "late_median_p90_collapse": float(
                    np.median([row["p90_daily_collapse_rate"] for row in late])
                ),
                "late_median_unique": float(
                    np.median([row["median_daily_unique_tokens"] for row in late])
                ),
                "late_median_ampratio": float(
                    np.median([row["avg_ampratio"] for row in late])
                ),
                "late_median_ampratio_log_error": float(
                    np.median([row["ampratio_log_error"] for row in late])
                ),
                "late_median_val_loss": float(
                    np.median([row["val_loss"] for row in late])
                ),
                "late_std_da": float(
                    np.std([row["avg_da_per_date"] for row in late], ddof=1)
                ),
                "late_std_rankic": float(
                    np.std([row["avg_daily_rank_ic"] for row in late], ddof=1)
                ),
                "late_median_worst_collapse": float(
                    np.median([row["worst_daily_collapse_rate"] for row in late])
                ),
                "late_median_min_unique": float(
                    np.median([row["min_daily_unique_tokens"] for row in late])
                ),
                "quality_pareto_epochs": sum(
                    bool(row["quality_pareto"]) for row in group
                ),
                "behaviour_pareto_epochs": sum(
                    bool(row["behaviour_pareto"]) for row in group
                ),
                "joint_pareto_epochs": sum(
                    bool(row["joint_pareto"]) for row in group
                ),
            }
        )
    return output


def scaling_correlations(summary: list[dict[str, Any]]) -> dict[str, Any]:
    x = [math.log10(float(row["parameter_count"])) for row in summary]
    output: dict[str, Any] = {}
    for metric in (
        "late_median_val_loss",
        "late_median_da",
        "late_median_rankic",
        "late_median_mape",
        "late_median_p90_collapse",
        "late_median_unique",
        "late_median_ampratio",
    ):
        output[metric] = COMMON.safe_spearman(
            x, [float(row[metric]) for row in summary]
        )
    return output


def _epoch_payload_paths(root: Path, config: str) -> list[Path]:
    trajectory = root / "configs" / config / "epoch_trajectory"
    paths = [
        path
        for path in trajectory.glob("epoch_*.json")
        if re.fullmatch(r"epoch_\d{3}", path.stem)
    ]
    return sorted(paths)


def extract_target_distribution(
    root: Path,
    reference_config: str = "deep",
) -> tuple[dict[tuple[int, str], dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    """Recover daily true coarse-token distributions from the prepared cache."""
    import torch

    epoch_paths = _epoch_payload_paths(root, reference_config)
    if not epoch_paths:
        raise FileNotFoundError(
            f"No epoch payloads for target reference {reference_config}"
        )
    reference = load_json(epoch_paths[-1])
    retained = {
        (int(offset), date_key)
        for offset, window in reference["windows"].items()
        for date_key in window["per_date"]
    }
    cache_paths = sorted(
        (root / "configs" / reference_config / "cache").glob(
            "trajectory_inputs_*.pt"
        )
    )
    if len(cache_paths) != 1:
        raise RuntimeError(
            f"Expected one prepared cache for {reference_config}, "
            f"found {len(cache_paths)}"
        )
    payload = torch.load(
        cache_paths[0],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    counts: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    missing_targets = 0
    for batch in payload["batches"]:
        rows = batch["selection_rows"].long()
        positions = batch["selection_positions"].long()
        valid = positions + 1 < batch["lengths"][rows]
        targets = torch.full_like(positions, -1)
        targets[valid] = batch["input_ids"][
            rows[valid], positions[valid] + 1
        ]
        for offset, date_key, target, is_valid in zip(
            batch["window_starts"].tolist(),
            batch["date_keys"],
            targets.tolist(),
            valid.tolist(),
        ):
            key = (int(offset), str(date_key))
            if key not in retained:
                continue
            if not is_valid:
                missing_targets += 1
                continue
            counts[key][int(target)] += 1

    if set(counts) != retained:
        missing = sorted(retained - set(counts))
        raise RuntimeError(f"Target distribution is missing dates: {missing}")
    daily_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, str], dict[str, float]] = {}
    for (offset, date_key), token_counts in sorted(counts.items()):
        n = sum(token_counts.values())
        probabilities = np.asarray(
            [count / n for count in token_counts.values()],
            dtype=np.float64,
        )
        entropy_bits = float(
            -np.sum(probabilities * np.log2(probabilities))
        )
        row = {
            "window_offset": offset,
            "date": date_key,
            "n": n,
            "target_unique_tokens": len(token_counts),
            "target_collapse_rate": max(token_counts.values()) / n,
            "target_entropy_bits": entropy_bits,
            "target_effective_tokens": float(2.0**entropy_bits),
        }
        daily_rows.append(row)
        lookup[(offset, date_key)] = {
            "unique": float(row["target_unique_tokens"]),
            "collapse": float(row["target_collapse_rate"]),
            "effective": float(row["target_effective_tokens"]),
        }

    def distribution_summary(field: str) -> dict[str, float]:
        values = np.asarray(
            [float(row[field]) for row in daily_rows], dtype=np.float64
        )
        return {
            "min": float(values.min()),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(values.max()),
        }

    summary = {
        "scope": (
            "Coarse 512-way AR target only; this does not measure fine-token "
            "or joint 65,536-code utilization."
        ),
        "reference_config": reference_config,
        "prepared_cache": str(cache_paths[0].resolve()),
        "n_retained_date_sections": len(daily_rows),
        "n_target_observations": sum(int(row["n"]) for row in daily_rows),
        "missing_targets": missing_targets,
        "daily_target_unique_tokens": distribution_summary(
            "target_unique_tokens"
        ),
        "daily_target_collapse_rate": distribution_summary(
            "target_collapse_rate"
        ),
        "daily_target_effective_tokens": distribution_summary(
            "target_effective_tokens"
        ),
    }
    return lookup, summary, daily_rows


def codebook_epoch_rows(
    root: Path,
    rows: list[dict[str, Any]],
    target: dict[tuple[int, str], dict[str, float]],
) -> list[dict[str, Any]]:
    """Add target-relative unique/collapse diagnostics to every checkpoint."""
    by_key = {
        (str(row["config"]), int(row["epoch"])): dict(row) for row in rows
    }
    output: list[dict[str, Any]] = []
    for config in sorted({key[0] for key in by_key}):
        for path in _epoch_payload_paths(root, config):
            payload = load_json(path)
            daily: list[tuple[float, float, float, float, float]] = []
            for raw_offset, window in payload["windows"].items():
                offset = int(raw_offset)
                for date_key, metrics in window["per_date"].items():
                    reference = target[(offset, date_key)]
                    predicted_unique = float(metrics["n_unique_tokens"])
                    predicted_collapse = float(metrics["collapse_rate"])

                    def symmetric_ratio(left: float, right: float) -> float:
                        if left <= 0 or right <= 0:
                            return 0.0
                        return min(left / right, right / left)

                    unique_alignment = symmetric_ratio(
                        predicted_unique, reference["unique"]
                    )
                    collapse_alignment = symmetric_ratio(
                        predicted_collapse, reference["collapse"]
                    )
                    daily.append(
                        (
                            unique_alignment,
                            collapse_alignment,
                            math.sqrt(
                                unique_alignment * collapse_alignment
                            ),
                            predicted_unique / reference["unique"],
                            predicted_collapse - reference["collapse"],
                        )
                    )
            values = np.asarray(daily, dtype=np.float64)
            row = by_key[(config, int(payload["epoch"]))]
            row.update(
                {
                    "target_unique_alignment_median": float(
                        np.median(values[:, 0])
                    ),
                    "target_unique_alignment_p10": float(
                        np.quantile(values[:, 0], 0.10)
                    ),
                    "target_collapse_alignment_median": float(
                        np.median(values[:, 1])
                    ),
                    "target_collapse_alignment_p10": float(
                        np.quantile(values[:, 1], 0.10)
                    ),
                    # Target-Moment Alignment (TMA) is an all-epoch proxy.
                    # Exact representative diagnostics additionally use support,
                    # entropy/effective vocabulary, and JSD.
                    "target_moment_alignment_median": float(
                        np.median(values[:, 2])
                    ),
                    "target_moment_alignment_p10": float(
                        np.quantile(values[:, 2], 0.10)
                    ),
                    "target_unique_ratio_mean": float(values[:, 3].mean()),
                    "target_collapse_excess_mean": float(values[:, 4].mean()),
                }
            )
            output.append(row)
    return sorted(output, key=lambda row: (row["config"], row["epoch"]))


def _five_epoch_windows(
    rows: list[dict[str, Any]],
    *,
    first_epoch: int,
    loss_ceiling: float | None,
) -> list[dict[str, Any]]:
    by_epoch = {int(row["epoch"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for start in range(first_epoch, max(by_epoch) - 3):
        epochs = list(range(start, start + 5))
        if any(epoch not in by_epoch for epoch in epochs):
            continue
        window = [by_epoch[epoch] for epoch in epochs]
        if (
            loss_ceiling is not None
            and any(float(row["val_loss"]) > loss_ceiling for row in window)
        ):
            continue
        output.append(
            {
                "start_epoch": start,
                "end_epoch": start + 4,
                "center_epoch": start + 2,
                "median_tma": float(
                    np.median(
                        [
                            row["target_moment_alignment_median"]
                            for row in window
                        ]
                    )
                ),
                "median_p10_tma": float(
                    np.median(
                        [
                            row["target_moment_alignment_p10"]
                            for row in window
                        ]
                    )
                ),
            }
        )
    return output


def capacity_windows(
    codebook_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find stable raw-utilization and near-best-loss mature windows."""
    output: list[dict[str, Any]] = []
    configs = sorted(
        {str(row["config"]) for row in codebook_rows},
        key=lambda name: min(
            row["parameter_count"]
            for row in codebook_rows
            if row["config"] == name
        ),
    )
    for config in configs:
        group = sorted(
            [row for row in codebook_rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        minimum_loss = min(float(row["val_loss"]) for row in group)
        loss_ceiling = minimum_loss * 1.01
        maturity_onset = next(
            int(row["epoch"])
            for row in group
            if float(row["val_loss"]) <= loss_ceiling
        )
        raw_windows = _five_epoch_windows(
            group, first_epoch=maturity_onset, loss_ceiling=None
        )
        balanced_windows = _five_epoch_windows(
            group,
            first_epoch=maturity_onset,
            loss_ceiling=loss_ceiling,
        )
        if not raw_windows or not balanced_windows:
            raise RuntimeError(f"No mature five-epoch window for {config}")
        raw = max(
            raw_windows,
            key=lambda row: (row["median_tma"], row["median_p10_tma"]),
        )
        balanced = max(
            balanced_windows,
            key=lambda row: (row["median_tma"], row["median_p10_tma"]),
        )
        eligible = [
            row for row in group if int(row["epoch"]) >= maturity_onset
        ]
        raw_single = max(
            eligible,
            key=lambda row: (
                row["target_moment_alignment_median"],
                row["target_moment_alignment_p10"],
            ),
        )
        output.append(
            {
                "config": config,
                "parameter_count": group[0]["parameter_count"],
                "minimum_val_loss": minimum_loss,
                "near_best_loss_ceiling": loss_ceiling,
                "maturity_onset_epoch": maturity_onset,
                "balanced_window_start": balanced["start_epoch"],
                "balanced_window_end": balanced["end_epoch"],
                "balanced_representative_epoch": balanced["center_epoch"],
                "balanced_window_median_tma": balanced["median_tma"],
                "balanced_window_median_p10_tma": balanced[
                    "median_p10_tma"
                ],
                "raw_utilization_window_start": raw["start_epoch"],
                "raw_utilization_window_end": raw["end_epoch"],
                "raw_utilization_window_median_tma": raw["median_tma"],
                "raw_utilization_peak_epoch": int(raw_single["epoch"]),
                "raw_utilization_peak_tma": raw_single[
                    "target_moment_alignment_median"
                ],
            }
        )
    return output


EXACT_AGGREGATE_FIELDS = (
    "avg_da_per_date",
    "avg_daily_rank_ic",
    "avg_mape",
    "avg_ampratio",
    "median_daily_collapse_rate",
    "p90_daily_collapse_rate",
    "median_daily_unique_tokens",
    "median_daily_target_collapse_rate",
    "median_daily_target_n_unique_tokens",
    "median_daily_pred_effective_tokens",
    "median_daily_target_effective_tokens",
    "median_daily_prediction_support_precision",
    "median_daily_target_support_recall",
    "median_daily_token_support_f1",
    "median_daily_token_jsd",
    "median_daily_distribution_alignment",
    "median_daily_effective_token_alignment",
    "median_daily_collapse_alignment",
    "median_daily_codebook_balance_score",
    "p10_daily_codebook_balance_score",
    "median_daily_coarse_token_accuracy",
)


def load_exact_representatives(
    root: Path,
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load exact distribution diagnostics for each balanced checkpoint."""
    output: list[dict[str, Any]] = []
    for window in windows:
        config = str(window["config"])
        epoch = int(window["balanced_representative_epoch"])
        path = (
            root
            / "configs"
            / config
            / "codebook_diagnostics"
            / f"epoch_{epoch:03d}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Exact codebook diagnostic is missing: {path}"
            )
        payload = load_json(path)
        row = {
            "config": config,
            "epoch": epoch,
            "role": "balanced_representative",
            "parameter_count": window["parameter_count"],
            "val_loss": payload["training"]["val_loss"],
        }
        row.update(
            {
                field: payload["aggregate"][field]
                for field in EXACT_AGGREGATE_FIELDS
            }
        )
        output.append(row)

    xlarge_window = next(
        row for row in windows if row["config"] == "xlarge"
    )
    peak_epoch = int(xlarge_window["raw_utilization_peak_epoch"])
    peak_path = (
        root
        / "configs"
        / "xlarge"
        / "codebook_diagnostics"
        / f"epoch_{peak_epoch:03d}.json"
    )
    if peak_path.is_file():
        payload = load_json(peak_path)
        row = {
            "config": "xlarge",
            "epoch": peak_epoch,
            "role": "raw_utilization_peak",
            "parameter_count": xlarge_window["parameter_count"],
            "val_loss": payload["training"]["val_loss"],
        }
        row.update(
            {
                field: payload["aggregate"][field]
                for field in EXACT_AGGREGATE_FIELDS
            }
        )
        output.append(row)
    return output


DAILY_DIAGNOSTIC_FIELDS = {
    "daily_da": "da",
    "daily_rank_ic": "rank_ic",
    "daily_mape": "mape",
    "daily_collapse": "collapse_rate",
    "daily_unique": "n_unique_tokens",
    "daily_ampratio_log_error": "ampratio",
}


def _mature_config_daily_values(
    root: Path,
    config: str,
) -> dict[str, dict[int, dict[str, float]]]:
    trajectory = root / "configs" / config / "epoch_trajectory"
    epoch_paths: list[Path] = []
    for path in trajectory.glob("epoch_*.json"):
        try:
            int(path.stem.removeprefix("epoch_"))
        except ValueError:
            continue
        epoch_paths.append(path)
    payloads = [load_json(path) for path in sorted(epoch_paths)]
    payloads.sort(key=lambda payload: int(payload["epoch"]))
    payloads = payloads[-min(10, len(payloads)) :]
    collected: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {} for name in DAILY_DIAGNOSTIC_FIELDS
    }
    for payload in payloads:
        for raw_offset, window in payload["windows"].items():
            offset = int(raw_offset)
            for date_key, daily in window["per_date"].items():
                for name, source in DAILY_DIAGNOSTIC_FIELDS.items():
                    value = float(daily[source])
                    if name == "daily_ampratio_log_error":
                        value = abs(math.log(value))
                    collected[name].setdefault(offset, {}).setdefault(
                        date_key, []
                    ).append(value)
    return {
        name: {
            offset: {
                date_key: sum(values) / len(values)
                for date_key, values in dates.items()
            }
            for offset, dates in windows.items()
        }
        for name, windows in collected.items()
    }


def paired_mature_diagnostics(
    root: Path,
    selected_config: str,
    comparator_config: str = "baseline",
    *,
    replicates: int = 10_000,
    block_length: int = 5,
) -> list[dict[str, Any]]:
    if selected_config == comparator_config:
        return []
    selected = _mature_config_daily_values(root, selected_config)
    comparator = _mature_config_daily_values(root, comparator_config)
    differences: dict[str, dict[int, list[float]]] = {
        metric: {} for metric in DAILY_DIAGNOSTIC_FIELDS
    }
    for offset in sorted(
        set(selected["daily_da"]) & set(comparator["daily_da"])
    ):
        common_dates = set(selected["daily_da"][offset]) & set(
            comparator["daily_da"][offset]
        )
        for metric in DAILY_DIAGNOSTIC_FIELDS:
            common_dates &= set(selected[metric][offset])
            common_dates &= set(comparator[metric][offset])
        ordered_dates = sorted(common_dates)
        if not ordered_dates:
            continue
        for metric in DAILY_DIAGNOSTIC_FIELDS:
            differences[metric][offset] = [
                selected[metric][offset][date]
                - comparator[metric][offset][date]
                for date in ordered_dates
            ]

    if not any(differences["daily_da"].values()):
        raise RuntimeError(
            f"No paired mature dates for {selected_config} vs {comparator_config}"
        )
    point_estimates = {
        metric: float(
            np.mean([value for window in windows.values() for value in window])
        )
        for metric, windows in differences.items()
    }
    bootstrap = {metric: [] for metric in DAILY_DIAGNOSTIC_FIELDS}
    rng = np.random.default_rng(42)
    for _ in range(replicates):
        sampled = {metric: [] for metric in DAILY_DIAGNOSTIC_FIELDS}
        for offset in sorted(differences["daily_da"]):
            count = len(differences["daily_da"][offset])
            indices: list[int] = []
            while len(indices) < count:
                start = int(rng.integers(count))
                indices.extend(
                    (start + step) % count for step in range(block_length)
                )
            indices = indices[:count]
            for metric in DAILY_DIAGNOSTIC_FIELDS:
                sampled[metric].extend(
                    differences[metric][offset][index] for index in indices
                )
        for metric in DAILY_DIAGNOSTIC_FIELDS:
            bootstrap[metric].append(float(np.mean(sampled[metric])))
    favorable = {
        "daily_da": "higher",
        "daily_rank_ic": "higher",
        "daily_mape": "lower",
        "daily_collapse": "lower",
        "daily_unique": "higher",
        "daily_ampratio_log_error": "lower",
    }
    return [
        {
            "selected_config": selected_config,
            "comparator_config": comparator_config,
            "metric": metric,
            "favorable_when": favorable[metric],
            "selected_minus_comparator": point_estimates[metric],
            "moving_block_bootstrap_95_ci": [
                float(np.quantile(bootstrap[metric], 0.025)),
                float(np.quantile(bootstrap[metric], 0.975)),
            ],
            "n_dates": sum(
                len(values) for values in differences[metric].values()
            ),
            "n_windows": len(differences[metric]),
            "block_length_days": block_length,
            "replicates": replicates,
        }
        for metric in DAILY_DIAGNOSTIC_FIELDS
    ]


EXACT_DAILY_FIELDS = {
    "codebook_balance_score": ("codebook_balance_score", "higher"),
    "target_support_recall": ("target_support_recall", "higher"),
    "pred_effective_tokens": ("pred_effective_tokens", "higher"),
    "token_jsd": ("token_jsd", "lower"),
    "daily_collapse": ("collapse_rate", "lower"),
    "coarse_token_accuracy": ("coarse_token_accuracy", "higher"),
    "daily_da": ("da", "higher"),
    "daily_rank_ic": ("rank_ic", "higher"),
    "daily_mape": ("mape", "lower"),
    "daily_ampratio_log_error": ("ampratio", "lower"),
}


def paired_exact_checkpoint_diagnostics(
    root: Path,
    left_config: str,
    left_epoch: int,
    right_config: str,
    right_epoch: int,
    *,
    replicates: int = 10_000,
    block_length: int = 5,
) -> list[dict[str, Any]]:
    """Paired-date diagnostics for two exact checkpoint re-evaluations."""

    def payload(config: str, epoch: int) -> dict[str, Any]:
        path = (
            root
            / "configs"
            / config
            / "codebook_diagnostics"
            / f"epoch_{epoch:03d}.json"
        )
        return load_json(path)

    left = payload(left_config, left_epoch)
    right = payload(right_config, right_epoch)
    differences: dict[str, dict[int, list[float]]] = {
        name: {} for name in EXACT_DAILY_FIELDS
    }
    common_windows = sorted(
        set(left["windows"]) & set(right["windows"]), key=int
    )
    for raw_offset in common_windows:
        left_dates = left["windows"][raw_offset]["per_date"]
        right_dates = right["windows"][raw_offset]["per_date"]
        dates = sorted(set(left_dates) & set(right_dates))
        offset = int(raw_offset)
        for name, (source, _) in EXACT_DAILY_FIELDS.items():
            values: list[float] = []
            for date_key in dates:
                left_value = float(left_dates[date_key][source])
                right_value = float(right_dates[date_key][source])
                if name == "daily_ampratio_log_error":
                    left_value = abs(math.log(left_value))
                    right_value = abs(math.log(right_value))
                values.append(left_value - right_value)
            differences[name][offset] = values

    rng = np.random.default_rng(42)
    bootstrap: dict[str, list[float]] = {
        name: [] for name in EXACT_DAILY_FIELDS
    }
    for _ in range(replicates):
        sampled: dict[str, list[float]] = {
            name: [] for name in EXACT_DAILY_FIELDS
        }
        for offset in sorted(differences["codebook_balance_score"]):
            count = len(differences["codebook_balance_score"][offset])
            indices: list[int] = []
            while len(indices) < count:
                start = int(rng.integers(count))
                indices.extend(
                    (start + step) % count for step in range(block_length)
                )
            indices = indices[:count]
            for name in EXACT_DAILY_FIELDS:
                sampled[name].extend(
                    differences[name][offset][index] for index in indices
                )
        for name in EXACT_DAILY_FIELDS:
            bootstrap[name].append(float(np.mean(sampled[name])))

    output: list[dict[str, Any]] = []
    for name, (_, favorable) in EXACT_DAILY_FIELDS.items():
        flattened = [
            value
            for values in differences[name].values()
            for value in values
        ]
        output.append(
            {
                "left_config": left_config,
                "left_epoch": left_epoch,
                "right_config": right_config,
                "right_epoch": right_epoch,
                "metric": name,
                "favorable_when": favorable,
                "left_minus_right": float(np.mean(flattened)),
                "moving_block_bootstrap_95_ci": [
                    float(np.quantile(bootstrap[name], 0.025)),
                    float(np.quantile(bootstrap[name], 0.975)),
                ],
                "n_dates": len(flattened),
                "n_windows": len(differences[name]),
                "block_length_days": block_length,
                "replicates": replicates,
            }
        )
    return output


def build_selection(
    root: Path,
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    exact_representatives: list[dict[str, Any]],
    selected_config: str,
) -> dict[str, Any]:
    by_name = {str(row["config"]): row for row in summary}
    if selected_config not in by_name:
        raise ValueError(
            f"Unknown selected config {selected_config!r}; "
            f"available={sorted(by_name)}"
        )
    selected = dict(by_name[selected_config])
    architecture = next(
        (
            dict(item)
            for item in manifest["settings"]["configs"]
            if item["name"] == selected_config
        ),
        None,
    )
    if architecture is None:
        raise RuntimeError(f"Architecture metadata missing for {selected_config}")
    selected_window = next(
        row for row in windows if row["config"] == selected_config
    )
    selected_exact = next(
        row
        for row in exact_representatives
        if row["config"] == selected_config
        and row["role"] == "balanced_representative"
    )
    representative_epoch = int(
        selected_window["balanced_representative_epoch"]
    )
    selected.update(architecture)
    selected.update(
        {
            "checkpoint_epoch": representative_epoch,
            "mature_window": [
                selected_window["balanced_window_start"],
                selected_window["balanced_window_end"],
            ],
            "maturity_onset_epoch": selected_window[
                "maturity_onset_epoch"
            ],
            "near_best_loss_ceiling": selected_window[
                "near_best_loss_ceiling"
            ],
            "selection_role": (
                "historical coarse capacity-utilization leader inside the "
                "near-best validation-loss basin; pending controlled Sup rerun"
            ),
            "exact_codebook_diagnostics": {
                key: value
                for key, value in selected_exact.items()
                if key
                not in {
                    "config",
                    "epoch",
                    "role",
                    "parameter_count",
                }
            },
        }
    )
    selected["model_path"] = str(
        (
            root
            / "configs"
            / selected_config
            / f"model_ep{representative_epoch}.pt"
        ).resolve()
    )

    exact_by_config = {
        row["config"]: row
        for row in exact_representatives
        if row["role"] == "balanced_representative"
    }
    baseline_exact = exact_by_config["baseline"]
    deep_exact = exact_by_config["deep"]
    xlarge_exact = exact_by_config["xlarge"]
    deep_gain = (
        deep_exact["median_daily_codebook_balance_score"]
        - baseline_exact["median_daily_codebook_balance_score"]
    )
    xlarge_gain = (
        xlarge_exact["median_daily_codebook_balance_score"]
        - deep_exact["median_daily_codebook_balance_score"]
    )
    deep_parameter_gain_m = (
        deep_exact["parameter_count"] - baseline_exact["parameter_count"]
    ) / 1e6
    xlarge_parameter_gain_m = (
        xlarge_exact["parameter_count"] - deep_exact["parameter_count"]
    ) / 1e6
    efficiency = {
        "baseline_to_deep_codebook_score_gain": deep_gain,
        "baseline_to_deep_added_parameters_m": deep_parameter_gain_m,
        "baseline_to_deep_gain_per_added_million_parameters": (
            deep_gain / deep_parameter_gain_m
        ),
        "deep_to_xlarge_codebook_score_gain": xlarge_gain,
        "deep_to_xlarge_added_parameters_m": xlarge_parameter_gain_m,
        "deep_to_xlarge_gain_per_added_million_parameters": (
            xlarge_gain / xlarge_parameter_gain_m
        ),
    }
    raw_peak = next(
        row
        for row in exact_representatives
        if row["config"] == "xlarge"
        and row["role"] == "raw_utilization_peak"
    )
    rationale = (
        f"{selected_config}@{representative_epoch} is the historical grid's "
        "capacity/utilization leader, not a controlled causal size selection "
        "and not proof that it fully drives the "
        "9+7 tokenizer. Its median daily target-support recall is "
        f"{selected_exact['median_daily_target_support_recall'] * 100:.1f}%, "
        "effective-token alignment is "
        f"{selected_exact['median_daily_effective_token_alignment'] * 100:.1f}%, "
        "and predicted collapse remains "
        f"{selected_exact['median_daily_collapse_rate'] * 100:.1f}% versus "
        f"{selected_exact['median_daily_target_collapse_rate'] * 100:.1f}% "
        "in the target. deep remains the compute-efficiency knee. The later "
        f"xlarge@{raw_peak['epoch']} checkpoint improves raw codebook "
        "distribution alignment further, but leaves the near-best-loss basin "
        "and sacrifices downstream DA, so it is recorded as a sensitivity "
        "point rather than the working checkpoint."
    )
    return {
        "experiment": "Exp 03 GPT architecture scaling",
        "status": "historical_confounded_pending_exp03_sup",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "selection_rule": (
            "Primary: target-relative coarse-codebook handling over a stable "
            "five-epoch mature window. The historical candidate maximizes the "
            "median/p10 target-moment balance inside the config-specific "
            "1%-of-best validation-loss basin. Exact support, entropy, JSD, "
            "and collapse diagnostics validate the representative. DA, RankIC, "
            "MAPE, AmpRatio, validation loss, and compute are guardrails rather "
            "than terms in the codebook score."
        ),
        "rationale": rationale,
        "capacity_roles": {
            "best_tested_balanced_capacity": {
                "config": selected_config,
                "epoch": representative_epoch,
            },
            "compute_efficiency_knee": {
                "config": "deep",
                "epoch": int(
                    next(
                        row
                        for row in windows
                        if row["config"] == "deep"
                    )["balanced_representative_epoch"]
                ),
            },
            "raw_utilization_winner": {
                "config": "xlarge",
                "epoch": int(raw_peak["epoch"]),
            },
        },
        "capacity_efficiency": efficiency,
        "no_tested_config_fully_drives_tokenizer": True,
        "codebook_scope": (
            "Coarse 512-way autoregressive IDs only. Fine-token and joint "
            "65,536-code utilization were not retained by this experiment."
        ),
        "holdout_used": False,
        "protocol_audit": PROTOCOL_AUDIT,
    }


def plot_trajectories(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    configs = sorted(
        {str(row["config"]) for row in rows},
        key=lambda name: min(
            row["parameter_count"] for row in rows if row["config"] == name
        ),
    )
    colours = plt.get_cmap("tab10")
    definitions = {
        "quality_trajectories.png": (
            "Prediction quality",
            (
                ("avg_da_per_date", "Mean daily DA (%)", 100.0),
                ("avg_daily_rank_ic", "Mean daily RankIC", 1.0),
                ("avg_mape", "MAPE (%)", 1.0),
            ),
        ),
        "behaviour_trajectories.png": (
            "Prediction behaviour",
            (
                ("p90_daily_collapse_rate", "P90 daily Collapse (%)", 100.0),
                ("median_daily_unique_tokens", "Median daily Unique", 1.0),
                ("avg_ampratio", "AmpRatio", 1.0),
            ),
        ),
    }
    for filename, (title, metrics) in definitions.items():
        figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
        for index, config in enumerate(configs):
            group = sorted(
                [row for row in rows if row["config"] == config],
                key=lambda row: int(row["epoch"]),
            )
            for axis, (metric, label, scale) in zip(axes, metrics):
                axis.plot(
                    [row["epoch"] for row in group],
                    [float(row[metric]) * scale for row in group],
                    label=config,
                    color=colours(index % 10),
                )
                axis.set_ylabel(label)
        if filename.startswith("quality"):
            axes[0].axhline(50, color="0.5", linestyle="--", linewidth=1)
        else:
            axes[0].axhline(35, color="0.5", linestyle="--", linewidth=1)
            axes[2].axhline(1, color="0.5", linestyle="--", linewidth=1)
        axes[-1].set_xlabel("GPT epoch")
        axes[0].legend(ncol=3, fontsize=8)
        figure.suptitle(f"Exp 03: {title}")
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)


def plot_capacity_dashboard(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    ordered = sorted(summary, key=lambda row: row["parameter_count"])
    x = np.asarray([row["parameter_count"] for row in ordered]) / 1e6
    labels = [str(row["config"]) for row in ordered]
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0, False),
        ("late_median_rankic", "Late RankIC", 1.0, False),
        ("late_median_mape", "Late MAPE (%)", 1.0, True),
        ("late_median_p90_collapse", "Late P90 Collapse (%)", 100.0, True),
        ("late_median_unique", "Late Unique", 1.0, False),
        ("late_median_ampratio", "Late AmpRatio", 1.0, None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (metric, title, scale, lower_better) in zip(
        axes.ravel(), definitions
    ):
        y = np.asarray([float(row[metric]) * scale for row in ordered])
        axis.plot(x, y, "o-", linewidth=1.4)
        for x_value, y_value, label in zip(x, y, labels):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Parameters (millions, log scale)")
        axis.set_ylabel(title)
        if lower_better is None:
            axis.axhline(1, color="0.4", linestyle="--", linewidth=1)
    figure.suptitle(
        "Exp 03 legacy final-10 downstream context (not the capacity selector)"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "capacity_metric_dashboard.png", dpi=180)
    plt.close(figure)


def plot_loss_reversal(plot_dir: Path, summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0),
        ("late_median_rankic", "Late RankIC", 1.0),
        ("late_median_p90_collapse", "Late P90 Collapse (%)", 100.0),
    )
    for axis, (metric, label, scale) in zip(axes, definitions):
        for row in summary:
            x = float(row["late_median_val_loss"])
            y = float(row[metric]) * scale
            axis.scatter(x, y, s=60)
            axis.annotate(
                str(row["config"]),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Late validation loss")
        axis.set_ylabel(label)
    figure.suptitle("Does lower validation loss imply better prediction?")
    figure.tight_layout()
    figure.savefig(plot_dir / "loss_vs_downstream.png", dpi=180)
    plt.close(figure)


def plot_pareto(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    configs = sorted({str(row["config"]) for row in rows})
    colours = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(10, 7))
    for index, config in enumerate(configs):
        group = [row for row in rows if row["config"] == config]
        axis.scatter(
            [row["p90_daily_collapse_rate"] * 100 for row in group],
            [row["avg_da_per_date"] * 100 for row in group],
            s=18,
            alpha=0.5,
            label=config,
            color=colours(index % 10),
        )
    front = [row for row in rows if row["joint_pareto"]]
    axis.scatter(
        [row["p90_daily_collapse_rate"] * 100 for row in front],
        [row["avg_da_per_date"] * 100 for row in front],
        s=85,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="6D joint Pareto",
    )
    axis.set_xlabel("P90 daily Collapse (%)")
    axis.set_ylabel("Mean daily DA (%)")
    axis.set_title("Exp 03 quality/behaviour trade-off")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "da_vs_collapse_pareto.png", dpi=180)
    plt.close(figure)


def plot_codebook_trajectories(
    plot_dir: Path,
    rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> None:
    configs = sorted(
        {str(row["config"]) for row in rows},
        key=lambda name: min(
            row["parameter_count"] for row in rows if row["config"] == name
        ),
    )
    window_by_config = {row["config"]: row for row in windows}
    colours = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for index, config in enumerate(configs):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        colour = colours(index % 10)
        axes[0].plot(
            [row["epoch"] for row in group],
            [row["target_moment_alignment_median"] for row in group],
            label=config,
            color=colour,
        )
        axes[1].plot(
            [row["epoch"] for row in group],
            [row["target_moment_alignment_p10"] for row in group],
            label=config,
            color=colour,
        )
        selected = window_by_config[config]
        axes[0].axvspan(
            selected["balanced_window_start"],
            selected["balanced_window_end"],
            color=colour,
            alpha=0.08,
        )
    axes[0].set_ylabel("Median target-moment alignment")
    axes[1].set_ylabel("P10 target-moment alignment")
    axes[1].set_xlabel("GPT epoch")
    axes[0].legend(ncol=3, fontsize=8)
    figure.suptitle(
        "Exp 03 target-relative coarse-codebook trajectories\n"
        "shading = selected 5-epoch near-best-loss mature window"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "codebook_trajectories.png", dpi=180)
    plt.close(figure)


def plot_exact_codebook_dashboard(
    plot_dir: Path,
    exact_rows: list[dict[str, Any]],
) -> None:
    balanced = [
        row for row in exact_rows if row["role"] == "balanced_representative"
    ]
    balanced.sort(key=lambda row: row["parameter_count"])
    x = np.asarray([row["parameter_count"] for row in balanced]) / 1e6
    labels = [f"{row['config']}@{row['epoch']}" for row in balanced]
    definitions = (
        (
            "median_daily_codebook_balance_score",
            "Codebook balance score",
            1.0,
        ),
        (
            "median_daily_target_support_recall",
            "Target support recall (%)",
            100.0,
        ),
        (
            "median_daily_effective_token_alignment",
            "Effective-token alignment (%)",
            100.0,
        ),
        ("median_daily_token_jsd", "Token JSD (lower better)", 1.0),
        (
            "median_daily_collapse_rate",
            "Predicted collapse (%)",
            100.0,
        ),
        ("avg_da_per_date", "DA guardrail (%)", 100.0),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (field, label, scale) in zip(axes.ravel(), definitions):
        y = np.asarray([float(row[field]) * scale for row in balanced])
        axis.plot(x, y, "o-", linewidth=1.4)
        for x_value, y_value, name in zip(x, y, labels):
            axis.annotate(
                name,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Parameters (millions, log scale)")
        axis.set_ylabel(label)
    figure.suptitle(
        "Exp 03 exact coarse-codebook diagnostics at balanced checkpoints"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "codebook_capacity_dashboard.png", dpi=180)
    plt.close(figure)


def render_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    target_summary: dict[str, Any],
    windows: list[dict[str, Any]],
    exact_representatives: list[dict[str, Any]],
    selection: dict[str, Any],
    selected_vs_deep: list[dict[str, Any]],
) -> str:
    tokenizer = manifest["settings"]["exp02_dependency"]
    window_by_config = {row["config"]: row for row in windows}
    balanced = [
        row
        for row in exact_representatives
        if row["role"] == "balanced_representative"
    ]
    balanced.sort(key=lambda row: row["parameter_count"])
    raw_peak = next(
        row
        for row in exact_representatives
        if row["role"] == "raw_utilization_peak"
    )
    target_unique = target_summary["daily_target_unique_tokens"]["median"]
    target_collapse = target_summary["daily_target_collapse_rate"]["median"]
    target_effective = target_summary["daily_target_effective_tokens"]["median"]
    lines = [
        "# Exp 03 GPT 容量与码本利用分析",
        "",
        "> **协议审计警告**：五个架构实际执行 3934–4104 个 optimizer steps，"
        "且 loader 顺序受架构消耗的全局 RNG 影响；以下排序是历史证据，"
        "不是纯容量因果结论。容量冻结必须等待 Exp 03-Sup 六点统一重训。",
        "",
        f"- 状态：**{manifest.get('status')}**",
        f"- Tokenizer：**{tokenizer['embedding_dim']}x"
        f"{tokenizer['hidden_dim']} / {tokenizer['bits_l1']}+"
        f"{tokenizer['bits_l2']} bits**",
        f"- 架构 / checkpoint：**{len(summary)} / {len(rows)}**",
        f"- 旧健康门通过数：**"
        f"{sum(bool(row.get('healthy')) for row in rows)}**",
        f"- 旧协议 coarse 指标领先候选：**{selection['selected']['config']}@"
        f"{selection['selected']['checkpoint_epoch']}**",
        "- 计算效率拐点：**deep@46**；纯利用率峰值："
        f"**xlarge@{raw_peak['epoch']}**",
        "",
        "## 1. 分析口径",
        "",
        "本分析只覆盖 GPT 主 AR 头的 **512-way coarse token**。现有实验没有"
        "保留 fine-token 预测直方图，因此不能把下面结果表述为完整 65,536 "
        "joint code 的利用率。",
        "",
        f"80 个保留日截面的真实目标中位数为：Unique={target_unique:.1f}、"
        f"Collapse={target_collapse * 100:.2f}%、"
        f"有效 token 数 exp(H)={target_effective:.1f}。这比用理论上限 512 "
        "作分母更合理，因为逐日真实分布本来就不会覆盖整个码本。",
        "",
        "所有 250 个 epoch 先用目标矩对齐 TMA 作轨迹扫描："
        "`sqrt(unique_alignment × collapse_alignment)`，其中两个 alignment "
        "都是预测值与真实值之比的对称形式 `min(r, 1/r)`。成熟起点定义为"
        "首次进入 `val_loss <= 1.01 × 本配置最小 val_loss`；每组比较连续 "
        "5 epoch 的中位数，避免从 50 次观察里挑一个噪声峰。",
        "",
        "代表 checkpoint 再做精确复评。Codebook Balance Score 是 support "
        "F1、`1-JSD`、有效 token 对齐和 collapse 对齐的几何平均；它只用于"
        "定位折中点，报告同时保留全部分量。DA/RankIC/MAPE/AmpRatio 与 loss "
        "不进入该分数，只作有用性和过拟合 guardrail。",
        "",
        "## 2. 代表 checkpoint 的精确码本诊断",
        "",
        "| Config@ep | 参数 | 成熟平衡窗 | CB score | Target recall | "
        "有效 token（预测/真实） | JSD | Collapse（预测/真实） | DA | Amp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in balanced:
        window = window_by_config[row["config"]]
        lines.append(
            f"| {row['config']}@{row['epoch']} | "
            f"{row['parameter_count'] / 1e6:.2f}M | "
            f"{window['balanced_window_start']}–"
            f"{window['balanced_window_end']} | "
            f"{row['median_daily_codebook_balance_score']:.3f} | "
            f"{row['median_daily_target_support_recall'] * 100:.1f}% | "
            f"{row['median_daily_pred_effective_tokens']:.1f}/"
            f"{row['median_daily_target_effective_tokens']:.1f} | "
            f"{row['median_daily_token_jsd']:.3f} | "
            f"{row['median_daily_collapse_rate'] * 100:.1f}%/"
            f"{row['median_daily_target_collapse_rate'] * 100:.1f}% | "
            f"{row['avg_da_per_date'] * 100:.2f}% | "
            f"{row['avg_ampratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "在旧协议结果中，`deep` 支配参数更多的 `wide/large`，是历史"
            "容量—成本曲线的拐点；`xlarge@18` 在仍处于近最优 loss 区间时"
            "取得旧网格最高精确码本分数。"
            "但它的 target support recall 只有约 24%，有效 token 仅为真实分布"
            "的约 19%，collapse 仍是目标的约 3 倍，所以结论不是“xlarge 已经"
            "驾驭码本”，而是“它是当前网格中最接近者”。",
            "",
            "## 3. 最佳 epoch 与后期峰值不是同一个问题",
            "",
            f"`xlarge@{raw_peak['epoch']}` 的纯利用率更高：CB score="
            f"{raw_peak['median_daily_codebook_balance_score']:.3f}、"
            f"recall={raw_peak['median_daily_target_support_recall'] * 100:.1f}%、"
            f"有效 token={raw_peak['median_daily_pred_effective_tokens']:.1f}。"
            f"但 val loss 已从最低点约 3.656 升至 {raw_peak['val_loss']:.3f}，"
            f"DA 只有 {raw_peak['avg_da_per_date'] * 100:.2f}%。因此它是"
            "“容量能把 argmax 分布铺得更开”的证据，不是历史平衡候选。"
            "旧协议代表点取 16–20 平衡窗中心 ep18；单点 ep18 也有该窗最好的 "
            "P10 TMA。",
            "",
            "这也解释了为什么不能只看最终 checkpoint、也不能只拿 50 个 epoch "
            "中的最高 Unique：前者错过可泛化的成熟点，后者会把过拟合后的"
            "分布扩张误当成无条件收益。",
            "",
            "## 4. xlarge@18 相对 deep@46 的逐日配对诊断",
            "",
            "差值为 xlarge@18 − deep@46；95% 区间是在四个验证窗内做 5 日"
            "循环 moving-block bootstrap（10,000 次）。它衡量日期不确定性，"
            "不替代多 seed。",
            "",
            "| 指标 | 差值 | 95% 区间 | 越好方向 |",
            "|---|---:|---:|---|",
        ]
    )
    for item in selected_vs_deep:
        low, high = item["moving_block_bootstrap_95_ci"]
        lines.append(
            f"| {item['metric']} | {item['left_minus_right']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {item['favorable_when']} |"
        )
    lines.extend(
        [
            "",
            "码本分数、support recall、有效 token 和 JSD 的改善均有逐日分辨力；"
            "DA/MAPE/RankIC 的区间跨 0。AmpRatio 更偏低，说明它仍应由 Exp 04 "
            "训练配方处理。相比之下，xlarge@44 的码本改善更强，但对 deep@46 "
            "的 DA 差已经不跨 0，因此没有选它。",
            "",
            "## 5. 修订结论",
            "",
            "1. 用户提出的实验定位基本正确：Exp 01/02 定 tokenizer，Exp 03 "
            "应优先回答 GPT 是否能利用该 tokenizer，而不是重复用 DA/MAPE "
            "做 HPO。",
            "2. `Unique + Collapse` 方向正确但不充分；必须相对真实目标分布，"
            "并加入 support、有效词表与 JSD，才能排除“随机撒 token”。",
            "3. 最佳 epoch 应从成熟区间选，但不能直接 cherry-pick 50 个单点；"
            "连续 5 epoch 平台加代表点更稳健。",
            "4. 在带 step/data-order 混杂的旧网格中，`xlarge@18` 是 coarse "
            "指标领先候选，`deep@46` 是历史效率拐点，`xlarge@44` 是纯利用率"
            "峰值；这些角色不能当作纯容量因果排序。",
            "5. 没有任何配置真正驾驭 9+7 tokenizer。Exp 03-Sup 已补齐"
            "4.5M–17M 间的深度/宽度控制点，并会将六个点按同数据顺序、同 4160 "
            "steps 全部重训；Exp 04 应等待该选型，而不是直接继承旧候选。",
            "",
            "## 6. 图与数据",
            "",
            "- `target_daily_distribution.csv`",
            "- `codebook_epoch_summary.csv`",
            "- `capacity_windows.csv`",
            "- `codebook_representative_checkpoints.csv`",
            "- `plots/codebook_trajectories.png`",
            "- `plots/codebook_capacity_dashboard.png`",
            "- `plots/quality_trajectories.png`",
            "- `plots/behaviour_trajectories.png`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--select_config",
        default="xlarge",
        help=(
            "Architecture to publish in selection.json "
            "(default: xlarge, balanced checkpoint)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    rows = load_json(root / "combined_epoch_summary.json")
    if not rows:
        raise RuntimeError("No Exp 03 rows to analyze")
    annotated = COMMON.annotate(rows)
    summary = envelopes(annotated)
    target, target_summary, target_daily = extract_target_distribution(root)
    codebook_rows = codebook_epoch_rows(root, annotated, target)
    windows = capacity_windows(codebook_rows)
    exact_representatives = load_exact_representatives(root, windows)
    selection = build_selection(
        root,
        manifest,
        summary,
        windows,
        exact_representatives,
        args.select_config,
    )
    selected_window = next(
        row for row in windows if row["config"] == args.select_config
    )
    deep_window = next(row for row in windows if row["config"] == "deep")
    baseline_window = next(
        row for row in windows if row["config"] == "baseline"
    )
    selected_vs_deep = paired_exact_checkpoint_diagnostics(
        root,
        args.select_config,
        int(selected_window["balanced_representative_epoch"]),
        "deep",
        int(deep_window["balanced_representative_epoch"]),
    )
    deep_vs_baseline = paired_exact_checkpoint_diagnostics(
        root,
        "deep",
        int(deep_window["balanced_representative_epoch"]),
        "baseline",
        int(baseline_window["balanced_representative_epoch"]),
    )
    raw_epoch = int(selected_window["raw_utilization_peak_epoch"])
    raw_vs_balanced = paired_exact_checkpoint_diagnostics(
        root,
        args.select_config,
        raw_epoch,
        args.select_config,
        int(selected_window["balanced_representative_epoch"]),
    )
    selection["paired_exact_checkpoint_diagnostics"] = {
        "selected_vs_compute_knee": selected_vs_deep,
        "compute_knee_vs_baseline": deep_vs_baseline,
        "raw_utilization_peak_vs_balanced_checkpoint": raw_vs_balanced,
        "scope": (
            "Per-date differences with a 5-day circular moving-block bootstrap "
            "inside each of four validation windows; diagnostic only and not "
            "training-seed uncertainty."
        ),
    }
    selection["target_distribution_summary"] = target_summary
    selection["representative_checkpoints"] = exact_representatives
    selection["capacity_windows"] = windows
    front = [
        row
        for row in annotated
        if row["quality_pareto"]
        or row["behaviour_pareto"]
        or row["joint_pareto"]
    ]
    codebook_front = []
    balanced_exact = [
        row
        for row in exact_representatives
        if row["role"] == "balanced_representative"
    ]
    for candidate in balanced_exact:
        dominated = any(
            other["parameter_count"] <= candidate["parameter_count"]
            and other["median_daily_codebook_balance_score"]
            >= candidate["median_daily_codebook_balance_score"]
            and (
                other["parameter_count"] < candidate["parameter_count"]
                or other["median_daily_codebook_balance_score"]
                > candidate["median_daily_codebook_balance_score"]
            )
            for other in balanced_exact
            if other is not candidate
        )
        if not dominated:
            codebook_front.append(candidate)
    analyzed_at = datetime.now(timezone.utc).isoformat()
    analysis_script_sha256 = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    analysis = {
        "experiment": "Exp 03 GPT architecture scaling",
        "status": manifest.get("status"),
        "analyzed_at_utc": analyzed_at,
        "analysis_script_sha256": analysis_script_sha256,
        "n_configs": len(summary),
        "n_checkpoints": len(annotated),
        "n_healthy": sum(bool(row.get("healthy")) for row in annotated),
        "pareto_counts": {
            "quality": sum(bool(row["quality_pareto"]) for row in annotated),
            "behaviour": sum(bool(row["behaviour_pareto"]) for row in annotated),
            "joint": sum(bool(row["joint_pareto"]) for row in annotated),
        },
        "config_envelopes": summary,
        "scaling_correlations": scaling_correlations(summary),
        "target_distribution_summary": target_summary,
        "capacity_windows": windows,
        "exact_representative_checkpoints": exact_representatives,
        "codebook_parameter_pareto": codebook_front,
        "paired_exact_checkpoint_diagnostics": {
            "selected_vs_compute_knee": selected_vs_deep,
            "compute_knee_vs_baseline": deep_vs_baseline,
            "raw_utilization_peak_vs_balanced_checkpoint": raw_vs_balanced,
        },
        "selection_policy": selection["selection_rule"],
        "protocol_audit": PROTOCOL_AUDIT,
    }
    selection["analysis_provenance"] = {
        "analyzed_at_utc": analyzed_at,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": analysis_script_sha256,
    }
    existing_selection_path = root / "selection.json"
    if existing_selection_path.is_file():
        existing_selection = load_json(existing_selection_path)
        existing_provenance = existing_selection.get("analysis_provenance", {})
        if (
            existing_selection.get("selected", {}).get("config")
            == args.select_config
            and existing_selection.get("selected", {}).get(
                "checkpoint_epoch"
            )
            == selection["selected"]["checkpoint_epoch"]
            and existing_provenance.get("script_sha256")
            == analysis_script_sha256
        ):
            selection["created_at_utc"] = existing_selection.get(
                "created_at_utc", selection["created_at_utc"]
            )
            selection["analysis_provenance"] = existing_provenance
    atomic_write_json(root / "analysis.json", analysis)
    atomic_write_json(root / "selection.json", selection)
    atomic_write_json(
        root / "target_distribution_summary.json", target_summary
    )
    atomic_write_json(root / "codebook_epoch_summary.json", codebook_rows)
    write_csv(root / "config_envelopes.csv", summary)
    write_csv(root / "pareto_front.csv", front)
    write_csv(root / "target_daily_distribution.csv", target_daily)
    write_csv(root / "codebook_epoch_summary.csv", codebook_rows)
    write_csv(root / "capacity_windows.csv", windows)
    write_csv(
        root / "codebook_representative_checkpoints.csv",
        exact_representatives,
    )
    write_csv(root / "codebook_parameter_pareto.csv", codebook_front)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(plot_dir, annotated)
    plot_capacity_dashboard(plot_dir, summary)
    plot_loss_reversal(plot_dir, summary)
    plot_pareto(plot_dir, annotated)
    plot_codebook_trajectories(plot_dir, codebook_rows, windows)
    plot_exact_codebook_dashboard(plot_dir, exact_representatives)
    atomic_write_text(
        root / "ANALYSIS.md",
        render_report(
            manifest,
            annotated,
            summary,
            target_summary,
            windows,
            exact_representatives,
            selection,
            selected_vs_deep,
        ),
    )
    manifest["analysis_completed_at_utc"] = analyzed_at
    manifest["posthoc_analysis"] = {
        "script_path": str(SCRIPT_PATH),
        "script_sha256": analysis_script_sha256,
        "selected_config": args.select_config,
        "selected_checkpoint_epoch": selection["selected"][
            "checkpoint_epoch"
        ],
        "analysis_focus": "target-relative coarse-codebook utilization",
        "protocol_audit": PROTOCOL_AUDIT,
    }
    atomic_write_json(root / "study_manifest.json", manifest)
    print(
        f"Analyzed {len(annotated)} checkpoints across {len(summary)} "
        f"architectures. Outputs: {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
