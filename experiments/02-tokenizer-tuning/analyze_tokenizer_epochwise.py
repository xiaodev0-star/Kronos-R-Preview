"""Analyze the current Exp 02 tokenizer sweep without a composite score."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parent / "rerun_seed42"

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def finite(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def dominates(
    candidate: dict[str, Any],
    target: dict[str, Any],
    objectives: Iterable[tuple[str, str]],
) -> bool:
    strictly_better = False
    for key, direction in objectives:
        left = finite(candidate, key)
        right = finite(target, key)
        if left is None or right is None:
            return False
        if direction == "max":
            if left < right:
                return False
            strictly_better |= left > right
        else:
            if left > right:
                return False
            strictly_better |= left < right
    return strictly_better


def pareto_flags(
    rows: list[dict[str, Any]],
    objectives: tuple[tuple[str, str], ...],
) -> list[bool]:
    return [
        not any(
            other_index != index and dominates(other, row, objectives)
            for other_index, other in enumerate(rows)
        )
        for index, row in enumerate(rows)
    ]


def safe_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return {"rho": None, "pvalue": None, "n": len(left)}
    result = spearmanr(left, right)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "n": len(left),
    }


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


def annotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = pareto_flags(rows, QUALITY_OBJECTIVES)
    behaviour = pareto_flags(rows, BEHAVIOUR_OBJECTIVES)
    joint = pareto_flags(rows, JOINT_OBJECTIVES)
    return [
        {
            **row,
            "quality_pareto": quality[index],
            "behaviour_pareto": behaviour[index],
            "joint_pareto": joint[index],
        }
        for index, row in enumerate(rows)
    ]


def arg_extreme(
    rows: list[dict[str, Any]], key: str, direction: str
) -> dict[str, Any]:
    valid = [row for row in rows if finite(row, key) is not None]
    function = max if direction == "max" else min
    return function(valid, key=lambda row: float(row[key]))


def envelopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in sorted({str(row["config"]) for row in rows}):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        late = group[-min(10, len(group)) :]
        best_da = arg_extreme(group, "avg_da_per_date", "max")
        best_rank = arg_extreme(group, "avg_daily_rank_ic", "max")
        best_mape = arg_extreme(group, "avg_mape", "min")
        best_collapse = arg_extreme(
            group, "p90_daily_collapse_rate", "min"
        )
        best_amp = arg_extreme(group, "ampratio_log_error", "min")
        output.append(
            {
                "config": config,
                "embedding_dim": group[0]["embedding_dim"],
                "hidden_dim": group[0]["hidden_dim"],
                "n_epochs": len(group),
                "healthy_epochs": sum(bool(row.get("healthy")) for row in group),
                "tokenizer_mae": group[0]["tokenizer_mae"],
                "tokenizer_rmse": group[0]["tokenizer_rmse"],
                "tokenizer_joint_unique": group[0]["tokenizer_joint_unique"],
                "tokenizer_joint_entropy_bits": group[0][
                    "tokenizer_joint_entropy_bits"
                ],
                "tokenizer_joint_collapse": group[0][
                    "tokenizer_joint_collapse"
                ],
                "best_da": best_da["avg_da_per_date"],
                "best_da_epoch": best_da["epoch"],
                "best_rankic": best_rank["avg_daily_rank_ic"],
                "best_rankic_epoch": best_rank["epoch"],
                "best_mape": best_mape["avg_mape"],
                "best_mape_epoch": best_mape["epoch"],
                "best_p90_collapse": best_collapse[
                    "p90_daily_collapse_rate"
                ],
                "best_p90_collapse_epoch": best_collapse["epoch"],
                "best_ampratio": best_amp["avg_ampratio"],
                "best_ampratio_epoch": best_amp["epoch"],
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
                "late_median_unique": float(
                    np.median(
                        [row["median_daily_unique_tokens"] for row in late]
                    )
                ),
                "late_median_ampratio": float(
                    np.median([row["avg_ampratio"] for row in late])
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


def correlations(
    rows: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = (
        "avg_da_per_date",
        "avg_daily_rank_ic",
        "avg_mape",
        "p90_daily_collapse_rate",
        "avg_ampratio",
        "median_daily_unique_tokens",
    )
    loss: dict[str, Any] = {}
    for metric in metrics:
        pairs = [
            (finite(row, "val_loss"), finite(row, metric)) for row in rows
        ]
        valid = [
            (float(left), float(right))
            for left, right in pairs
            if left is not None and right is not None
        ]
        loss[metric] = safe_spearman(
            [item[0] for item in valid],
            [item[1] for item in valid],
        )
    tokenizer_mae = [float(row["tokenizer_mae"]) for row in summary]
    downstream = {}
    for metric in (
        "late_median_da",
        "late_median_rankic",
        "late_median_mape",
        "late_median_p90_collapse",
        "late_median_unique",
        "late_median_ampratio",
    ):
        downstream[metric] = safe_spearman(
            tokenizer_mae, [float(row[metric]) for row in summary]
        )
    return {
        "val_loss_vs_epoch_metrics": loss,
        "tokenizer_mae_vs_late_downstream": downstream,
    }


def plot_trajectories(
    plot_dir: Path, rows: list[dict[str, Any]]
) -> None:
    configs = sorted({str(row["config"]) for row in rows})
    colours = plt.get_cmap("tab10")
    quality = (
        ("avg_da_per_date", "Mean daily DA (%)", 100.0),
        ("avg_daily_rank_ic", "Mean daily RankIC", 1.0),
        ("avg_mape", "MAPE (%)", 1.0),
    )
    behaviour = (
        ("p90_daily_collapse_rate", "P90 daily Collapse (%)", 100.0),
        ("median_daily_unique_tokens", "Median daily Unique", 1.0),
        ("avg_ampratio", "AmpRatio", 1.0),
    )
    for filename, title, definitions in (
        ("quality_trajectories.png", "Prediction quality", quality),
        ("behaviour_trajectories.png", "Prediction behaviour", behaviour),
    ):
        figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
        for index, config in enumerate(configs):
            group = sorted(
                [row for row in rows if row["config"] == config],
                key=lambda row: int(row["epoch"]),
            )
            for axis, (metric, label, scale) in zip(axes, definitions):
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
        figure.suptitle(f"Exp 02: {title}")
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)


def plot_tokenizer_grid(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    embeddings = sorted({int(row["embedding_dim"]) for row in summary})
    hiddens = sorted({int(row["hidden_dim"]) for row in summary})
    mae = np.full((len(hiddens), len(embeddings)), np.nan)
    collapse = np.full_like(mae, np.nan)
    for row in summary:
        i = hiddens.index(int(row["hidden_dim"]))
        j = embeddings.index(int(row["embedding_dim"]))
        mae[i, j] = float(row["tokenizer_mae"])
        collapse[i, j] = float(row["tokenizer_joint_collapse"]) * 100
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, matrix, title, fmt in (
        (axes[0], mae, "Validation reconstruction MAE", ".4f"),
        (axes[1], collapse, "Joint-code collapse (%)", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
        axis.set_xticks(range(len(embeddings)), embeddings)
        axis.set_yticks(range(len(hiddens)), hiddens)
        axis.set_xlabel("Embedding dimension")
        axis.set_ylabel("Hidden dimension")
        axis.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j,
                    i,
                    format(matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white"
                    if matrix[i, j] > np.nanmedian(matrix)
                    else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Exp 02 tokenizer-side evidence")
    figure.tight_layout()
    figure.savefig(plot_dir / "tokenizer_grid.png", dpi=180)
    plt.close(figure)


def plot_late_dashboard(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    ordered = sorted(summary, key=lambda row: str(row["config"]))
    labels = [str(row["config"]) for row in ordered]
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0, False),
        ("late_median_rankic", "Late RankIC", 1.0, False),
        ("late_median_mape", "Late MAPE (%)", 1.0, True),
        (
            "late_median_p90_collapse",
            "Late P90 Collapse (%)",
            100.0,
            True,
        ),
        ("late_median_unique", "Late Unique", 1.0, False),
        ("late_median_ampratio", "Late AmpRatio", 1.0, None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (metric, title, scale, lower_better) in zip(
        axes.ravel(), definitions
    ):
        values = [float(row[metric]) * scale for row in ordered]
        bars = axis.bar(labels, values, color=plt.get_cmap("tab10").colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        if lower_better is None:
            axis.axhline(1, color="0.4", linestyle="--", linewidth=1)
            best_index = int(np.argmin(np.abs(np.asarray(values) - 1)))
        elif lower_better:
            best_index = int(np.argmin(values))
        else:
            best_index = int(np.argmax(values))
        bars[best_index].set_edgecolor("black")
        bars[best_index].set_linewidth(2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    figure.suptitle(
        "Exp 02 mature-checkpoint comparison (last 10 epochs; no composite)"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "late_metric_dashboard.png", dpi=180)
    plt.close(figure)


def plot_reversal(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0),
        ("late_median_rankic", "Late RankIC", 1.0),
        ("late_median_p90_collapse", "Late P90 Collapse (%)", 100.0),
    )
    for axis, (metric, label, scale) in zip(axes, definitions):
        for row in summary:
            x = float(row["tokenizer_mae"])
            y = float(row[metric]) * scale
            axis.scatter(x, y, s=55)
            axis.annotate(
                str(row["config"]),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Tokenizer reconstruction MAE")
        axis.set_ylabel(label)
    figure.suptitle("Does tokenizer reconstruction predict GPT behaviour?")
    figure.tight_layout()
    figure.savefig(plot_dir / "tokenizer_vs_downstream.png", dpi=180)
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
    axis.set_title("Exp 02 quality/behaviour trade-off")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "da_vs_collapse_pareto.png", dpi=180)
    plt.close(figure)


def render_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    correlation_payload: dict[str, Any],
) -> str:
    lines = [
        "# Exp 02 tokenizer architecture rerun",
        "",
        f"- Status: **{manifest.get('status')}**",
        f"- Bits inherited from Exp 01: **{manifest['settings']['bits_l1']}+"
        f"{manifest['settings']['bits_l2']}**",
        f"- Configurations: **{len(summary)}**",
        f"- Evaluated GPT checkpoints: **{len(rows)}**",
        f"- Health-passing checkpoints: **"
        f"{sum(bool(row.get('healthy')) for row in rows)}**",
        "",
        "## Mature-checkpoint envelope",
        "",
        "| Config | Tok MAE | Best DA | Late DA | Late RankIC | Late MAPE | "
        "Late P90 collapse | Late unique | Late amp | Joint Pareto epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['config']} | {row['tokenizer_mae']:.4f} | "
            f"{row['best_da'] * 100:.2f}%@{row['best_da_epoch']} | "
            f"{row['late_median_da'] * 100:.2f}% | "
            f"{row['late_median_rankic']:.4f} | "
            f"{row['late_median_mape']:.3f} | "
            f"{row['late_median_p90_collapse'] * 100:.1f}% | "
            f"{row['late_median_unique']:.1f} | "
            f"{row['late_median_ampratio']:.3f} | "
            f"{row['joint_pareto_epochs']} |"
        )
    lines.extend(
        [
            "",
            "No weighted score is used. The tokenizer-side reconstruction "
            "table, downstream quality group, and prediction-behaviour group "
            "must be read separately.",
            "",
            "## Figures",
            "",
            "- `plots/tokenizer_grid.png`: tokenizer reconstruction and code use.",
            "- `plots/quality_trajectories.png`: DA, RankIC, and MAPE by epoch.",
            "- `plots/behaviour_trajectories.png`: collapse, diversity, amplitude.",
            "- `plots/late_metric_dashboard.png`: mature last-10-epoch comparison.",
            "- `plots/tokenizer_vs_downstream.png`: reconstruction/downstream reversal.",
            "- `plots/da_vs_collapse_pareto.png`: non-composite trade-off.",
            "",
            "## Correlation audit",
            "",
            "Raw Spearman results are stored in `analysis.json`; correlation "
            "is descriptive and is not used as a selection score.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    rows = load_json(root / "combined_epoch_summary.json")
    if not rows:
        raise RuntimeError("No Exp 02 rows to analyze")
    annotated = annotate(rows)
    summary = envelopes(annotated)
    correlation_payload = correlations(annotated, summary)
    front = [
        row
        for row in annotated
        if row["quality_pareto"]
        or row["behaviour_pareto"]
        or row["joint_pareto"]
    ]
    analysis = {
        "experiment": "Exp 02 Tokenizer architecture rerun",
        "status": manifest.get("status"),
        "n_configs": len(summary),
        "n_checkpoints": len(annotated),
        "n_healthy": sum(bool(row.get("healthy")) for row in annotated),
        "pareto_counts": {
            "quality": sum(bool(row["quality_pareto"]) for row in annotated),
            "behaviour": sum(
                bool(row["behaviour_pareto"]) for row in annotated
            ),
            "joint": sum(bool(row["joint_pareto"]) for row in annotated),
        },
        "config_envelopes": summary,
        "correlations": correlation_payload,
        "selection_policy": manifest["settings"]["evaluation"][
            "selection_policy"
        ],
    }
    atomic_write_json(root / "analysis.json", analysis)
    write_csv(root / "config_envelopes.csv", summary)
    write_csv(root / "pareto_front.csv", front)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(plot_dir, annotated)
    plot_tokenizer_grid(plot_dir, summary)
    plot_late_dashboard(plot_dir, summary)
    plot_reversal(plot_dir, summary)
    plot_pareto(plot_dir, annotated)
    atomic_write_text(
        root / "ANALYSIS.md",
        render_report(manifest, annotated, summary, correlation_payload),
    )
    print(
        f"Analyzed {len(annotated)} checkpoints across {len(summary)} configs. "
        f"Outputs: {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
