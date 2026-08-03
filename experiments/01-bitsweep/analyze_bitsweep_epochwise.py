"""Non-composite analysis for the epoch-wise BitSweep rerun.

The analysis deliberately does not collapse metrics into a weighted score.
Prediction behaviour (Collapse/AmpRatio/Unique) and prediction quality
(DA/MAPE/daily RankIC) remain separate.  Pareto membership and metric-specific
representatives are reported instead of one synthetic winner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import default_study_roots

_, DEFAULT_ROOT = default_study_roots("01-bitsweep", seed=42)


QUALITY_OBJECTIVES = (
    ("avg_da_per_date", "max"),
    ("avg_daily_rank_ic", "max"),
    ("avg_mape", "min"),
)
BEHAVIOUR_OBJECTIVES = (
    ("p90_daily_collapse_rate", "min"),
    ("ampratio_log_error", "min"),
    ("median_daily_unique_tokens", "max"),
)
JOINT_OBJECTIVES = QUALITY_OBJECTIVES + BEHAVIOUR_OBJECTIVES


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def finite_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def dominates(
    candidate: dict[str, Any],
    target: dict[str, Any],
    objectives: Iterable[tuple[str, str]],
) -> bool:
    at_least_as_good = True
    strictly_better = False
    for key, direction in objectives:
        left = finite_value(candidate, key)
        right = finite_value(target, key)
        if left is None or right is None:
            return False
        if direction == "max":
            if left < right:
                at_least_as_good = False
                break
            strictly_better |= left > right
        else:
            if left > right:
                at_least_as_good = False
                break
            strictly_better |= left < right
    return at_least_as_good and strictly_better


def pareto_flags(
    rows: list[dict[str, Any]],
    objectives: tuple[tuple[str, str], ...],
) -> list[bool]:
    flags: list[bool] = []
    for index, row in enumerate(rows):
        dominated = any(
            other_index != index and dominates(other, row, objectives)
            for other_index, other in enumerate(rows)
        )
        flags.append(not dominated)
    return flags


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
    os.replace(temporary, path)


def arg_extreme(
    rows: list[dict[str, Any]], key: str, direction: str
) -> dict[str, Any]:
    valid = [row for row in rows if finite_value(row, key) is not None]
    if not valid:
        raise RuntimeError(f"No finite values for {key}")
    function = max if direction == "max" else min
    return function(valid, key=lambda row: float(row[key]))


def representative_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions = (
        ("quality_da", "avg_da_per_date", "max"),
        ("quality_rankic", "avg_daily_rank_ic", "max"),
        ("quality_mape", "avg_mape", "min"),
        ("behaviour_collapse", "p90_daily_collapse_rate", "min"),
        ("behaviour_amplitude", "ampratio_log_error", "min"),
        ("behaviour_unique", "median_daily_unique_tokens", "max"),
    )
    output: list[dict[str, Any]] = []
    configs = sorted({str(row["config"]) for row in rows})
    for config in configs:
        group = [row for row in rows if row["config"] == config]
        seen: set[tuple[str, int]] = set()
        for role, key, direction in definitions:
            selected = arg_extreme(group, key, direction)
            identity = (role, int(selected["epoch"]))
            if identity in seen:
                continue
            seen.add(identity)
            output.append(
                {
                    "config": config,
                    "role": role,
                    "selection_metric": key,
                    "selection_direction": direction,
                    **selected,
                }
            )
        last = max(group, key=lambda row: int(row["epoch"]))
        output.append(
            {
                "config": config,
                "role": "last_epoch",
                "selection_metric": "epoch",
                "selection_direction": "max",
                **last,
            }
        )
    return output


def safe_spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "pvalue": None, "n": len(x)}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "n": len(x),
    }


def loss_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "avg_da_per_date",
        "avg_daily_rank_ic",
        "avg_mape",
        "p90_daily_collapse_rate",
        "avg_ampratio",
        "median_daily_unique_tokens",
    )

    def one_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in metrics:
            pairs = [
                (finite_value(row, "val_loss"), finite_value(row, metric))
                for row in group
            ]
            finite_pairs = [
                (float(left), float(right))
                for left, right in pairs
                if left is not None and right is not None
            ]
            result[metric] = safe_spearman(
                [pair[0] for pair in finite_pairs],
                [pair[1] for pair in finite_pairs],
            )
        return result

    output = {"all_config_epochs": one_group(rows), "per_config": {}}
    for config in sorted({str(row["config"]) for row in rows}):
        output["per_config"][config] = one_group(
            [row for row in rows if row["config"] == config]
        )
    return output


def config_envelopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in sorted({str(row["config"]) for row in rows}):
        group = [row for row in rows if row["config"] == config]
        epochs = sorted(int(row["epoch"]) for row in group)
        late_start = max(min(epochs), max(epochs) - 9)
        late = [row for row in group if int(row["epoch"]) >= late_start]

        max_da = arg_extreme(group, "avg_da_per_date", "max")
        max_ic = arg_extreme(group, "avg_daily_rank_ic", "max")
        min_mape = arg_extreme(group, "avg_mape", "min")
        min_collapse = arg_extreme(
            group, "p90_daily_collapse_rate", "min"
        )
        best_amp = arg_extreme(group, "ampratio_log_error", "min")
        first = group[0]
        output.append(
            {
                "config": config,
                "bits_l1": first["bits_l1"],
                "bits_l2": first["bits_l2"],
                "joint_vocab": first["joint_vocab"],
                "tokenizer_mae": first["tokenizer_mae"],
                "tokenizer_joint_entropy_bits": first[
                    "tokenizer_joint_entropy_bits"
                ],
                "tokenizer_joint_utilization": first[
                    "tokenizer_joint_utilization"
                ],
                "n_epochs": len(group),
                "healthy_epochs": sum(bool(row.get("healthy")) for row in group),
                "max_da": max_da["avg_da_per_date"],
                "max_da_epoch": max_da["epoch"],
                "max_rankic": max_ic["avg_daily_rank_ic"],
                "max_rankic_epoch": max_ic["epoch"],
                "min_mape": min_mape["avg_mape"],
                "min_mape_epoch": min_mape["epoch"],
                "min_p90_collapse": min_collapse[
                    "p90_daily_collapse_rate"
                ],
                "min_p90_collapse_epoch": min_collapse["epoch"],
                "best_ampratio": best_amp["avg_ampratio"],
                "best_ampratio_epoch": best_amp["epoch"],
                "late_epoch_start": late_start,
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
                    np.median(
                        [row["p90_daily_collapse_rate"] for row in late]
                    )
                ),
                "late_median_ampratio": float(
                    np.median([row["avg_ampratio"] for row in late])
                ),
            }
        )
    return output


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted({str(row["config"]) for row in rows})
    colour_map = plt.get_cmap("tab10")

    figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for index, config in enumerate(configs):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        epoch = [row["epoch"] for row in group]
        colour = colour_map(index % 10)
        axes[0].plot(
            epoch,
            [row["avg_da_per_date"] * 100 for row in group],
            label=config,
            color=colour,
        )
        axes[1].plot(
            epoch,
            [row["avg_daily_rank_ic"] for row in group],
            label=config,
            color=colour,
        )
        axes[2].plot(
            epoch,
            [row["avg_mape"] for row in group],
            label=config,
            color=colour,
        )
    axes[0].axhline(50, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean daily DA (%)")
    axes[1].set_ylabel("Mean daily RankIC")
    axes[2].set_ylabel("MAPE (%)")
    axes[2].set_xlabel("GPT epoch")
    axes[0].legend(ncol=5, fontsize=8)
    figure.suptitle("BitSweep epoch trajectories: prediction quality")
    figure.tight_layout()
    figure.savefig(output_dir / "quality_trajectories.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for index, config in enumerate(configs):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        epoch = [row["epoch"] for row in group]
        colour = colour_map(index % 10)
        axes[0].plot(
            epoch,
            [row["p90_daily_collapse_rate"] * 100 for row in group],
            label=config,
            color=colour,
        )
        axes[1].plot(
            epoch,
            [row["median_daily_unique_tokens"] for row in group],
            label=config,
            color=colour,
        )
        axes[2].plot(
            epoch,
            [row["avg_ampratio"] for row in group],
            label=config,
            color=colour,
        )
    axes[0].axhline(35, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylabel("P90 daily Collapse (%)")
    axes[1].set_ylabel("Median daily Unique")
    axes[2].axhline(1, color="0.5", linestyle="--", linewidth=1)
    axes[2].set_ylabel("AmpRatio")
    axes[2].set_xlabel("GPT epoch")
    axes[0].legend(ncol=5, fontsize=8)
    figure.suptitle("BitSweep epoch trajectories: prediction behaviour")
    figure.tight_layout()
    figure.savefig(output_dir / "behaviour_trajectories.png", dpi=170)
    plt.close(figure)

    first_rows = [
        min(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        for config in configs
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = [row["joint_vocab"] for row in first_rows]
    axes[0].plot(x, [row["tokenizer_mae"] for row in first_rows], "o-")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Joint vocabulary")
    axes[0].set_ylabel("Tokenizer validation MAE")
    axes[1].plot(
        x,
        [row["tokenizer_joint_entropy_bits"] for row in first_rows],
        "o-",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Joint vocabulary")
    axes[1].set_ylabel("Joint code entropy (bits)")
    for axis in axes:
        for row in first_rows:
            y_key = (
                "tokenizer_mae"
                if axis is axes[0]
                else "tokenizer_joint_entropy_bits"
            )
            axis.annotate(
                row["config"],
                (row["joint_vocab"], row[y_key]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
    figure.suptitle("Tokenizer capacity and reconstruction")
    figure.tight_layout()
    figure.savefig(output_dir / "tokenizer_capacity.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 7))
    for index, config in enumerate(configs):
        group = [row for row in rows if row["config"] == config]
        axis.scatter(
            [row["p90_daily_collapse_rate"] * 100 for row in group],
            [row["avg_da_per_date"] * 100 for row in group],
            s=18,
            alpha=0.55,
            label=config,
            color=colour_map(index % 10),
        )
    joint_front = [row for row in rows if row["joint_pareto"]]
    axis.scatter(
        [row["p90_daily_collapse_rate"] * 100 for row in joint_front],
        [row["avg_da_per_date"] * 100 for row in joint_front],
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="6D joint Pareto",
    )
    axis.set_xlabel("P90 daily Collapse (%)")
    axis.set_ylabel("Mean daily DA (%)")
    axis.legend(ncol=3, fontsize=8)
    axis.set_title("Quality/behaviour trade-off (no composite score)")
    figure.tight_layout()
    figure.savefig(output_dir / "da_vs_collapse_pareto.png", dpi=170)
    plt.close(figure)


def render_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    envelopes: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
) -> str:
    completed = manifest.get("status") == "completed"
    lines = [
        "# Exp 01 BitSweep rerun: epoch-wise analysis",
        "",
        f"> Status: **{'complete' if completed else 'partial'}**. "
        "No weighted composite score is used.",
        "",
        "The two metric groups remain separate:",
        "",
        "- Behaviour: daily Collapse, AmpRatio, and Unique tokens.",
        "- Quality: daily DA, MAPE, and daily cross-sectional RankIC.",
        "",
        f"- Config-epoch points analysed: **{len(rows)}**",
        f"- Health-passing points: **{sum(bool(row.get('healthy')) for row in rows)}**",
        f"- Joint six-objective Pareto points: **{len(pareto_rows)}**",
        f"- Holdout used: **{manifest.get('holdout_used', False)}**",
        "",
        "## Per-configuration envelopes",
        "",
        "| Config | Tok MAE | Best DA (ep) | Best RankIC (ep) | Best MAPE (ep) | Best P90 Collapse (ep) | Healthy ep |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in envelopes:
        lines.append(
            f"| {row['config']} | {row['tokenizer_mae']:.4f} | "
            f"{row['max_da'] * 100:.2f}% ({row['max_da_epoch']}) | "
            f"{row['max_rankic']:.4f} ({row['max_rankic_epoch']}) | "
            f"{row['min_mape']:.3f}% ({row['min_mape_epoch']}) | "
            f"{row['min_p90_collapse'] * 100:.1f}% "
            f"({row['min_p90_collapse_epoch']}) | "
            f"{row['healthy_epochs']}/{row['n_epochs']} |"
        )
    lines.extend(
        [
            "",
            "The table reports metric-specific extrema, not a winner. "
            "Epoch selection should use a stable Pareto region and must not "
            "open the offset-400 holdout during this analysis.",
            "",
            "## Artifacts",
            "",
            "- `analysis.json`: Pareto counts, envelopes, and loss correlations.",
            "- `pareto_front.csv`: all non-dominated points and front labels.",
            "- `metric_representatives.csv`: per-config extrema for each metric.",
            "- `plots/`: separate quality, behaviour, tokenizer, and trade-off figures.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse an epoch-wise BitSweep without a composite score"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    rows = load_json(root / "combined_epoch_summary.json")
    if not rows:
        raise RuntimeError("No config-epoch results are available")

    quality = pareto_flags(rows, QUALITY_OBJECTIVES)
    behaviour = pareto_flags(rows, BEHAVIOUR_OBJECTIVES)
    joint = pareto_flags(rows, JOINT_OBJECTIVES)
    annotated: list[dict[str, Any]] = []
    for row, quality_flag, behaviour_flag, joint_flag in zip(
        rows, quality, behaviour, joint
    ):
        annotated.append(
            {
                **row,
                "quality_pareto": quality_flag,
                "behaviour_pareto": behaviour_flag,
                "joint_pareto": joint_flag,
            }
        )

    pareto_rows = [
        row
        for row in annotated
        if row["quality_pareto"]
        or row["behaviour_pareto"]
        or row["joint_pareto"]
    ]
    representatives = representative_rows(annotated)
    envelopes = config_envelopes(annotated)
    correlations = loss_correlations(annotated)
    analysis = {
        "status": (
            "completed"
            if manifest.get("status") == "completed"
            else "partial"
        ),
        "n_config_epochs": len(annotated),
        "n_configs": len({row["config"] for row in annotated}),
        "n_healthy": sum(bool(row.get("healthy")) for row in annotated),
        "objectives": {
            "quality": QUALITY_OBJECTIVES,
            "behaviour": BEHAVIOUR_OBJECTIVES,
            "joint": JOINT_OBJECTIVES,
        },
        "pareto_counts": {
            "quality": sum(quality),
            "behaviour": sum(behaviour),
            "joint": sum(joint),
        },
        "config_envelopes": envelopes,
        "loss_metric_spearman": correlations,
        "holdout_used": bool(manifest.get("holdout_used", False)),
        "weighted_score_used": False,
    }
    atomic_write_json(root / "analysis.json", analysis)
    write_csv(root / "pareto_front.csv", pareto_rows)
    write_csv(root / "metric_representatives.csv", representatives)
    write_csv(root / "config_envelopes.csv", envelopes)
    plot_dir = root / "plots"
    make_plots(plot_dir, annotated)
    report = render_report(manifest, annotated, envelopes, pareto_rows)
    atomic_write_text(root / "README.md", report)
    print(
        f"Analysed {len(annotated)} config-epoch points; "
        f"joint Pareto={sum(joint)}, healthy={analysis['n_healthy']}. "
        f"Holdout used=False."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
