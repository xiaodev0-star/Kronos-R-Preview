"""Create narrative Exp 01 figures and record the Exp 02 bit dependency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = EXPERIMENT_DIR / "rerun_seed42"
ANALYSIS_PATH = EXPERIMENT_DIR / "analyze_bitsweep_epochwise.py"


def _load_analysis_module():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp01_analysis_for_narrative", ANALYSIS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis_module()


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


def mature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in sorted({str(row["config"]) for row in rows}):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        late = group[-min(10, len(group)) :]
        output.append(
            {
                "config": config,
                "bits_l1": int(group[0]["bits_l1"]),
                "bits_l2": int(group[0]["bits_l2"]),
                "joint_vocab": int(group[0]["joint_vocab"]),
                "tokenizer_mae": float(group[0]["tokenizer_mae"]),
                "tokenizer_entropy": float(
                    group[0]["tokenizer_joint_entropy_bits"]
                ),
                "late_da": float(
                    np.median([row["avg_da_per_date"] for row in late])
                ),
                "late_rankic": float(
                    np.median([row["avg_daily_rank_ic"] for row in late])
                ),
                "late_mape": float(
                    np.median([row["avg_mape"] for row in late])
                ),
                "late_p90_collapse": float(
                    np.median(
                        [row["p90_daily_collapse_rate"] for row in late]
                    )
                ),
                "late_unique": float(
                    np.median(
                        [row["median_daily_unique_tokens"] for row in late]
                    )
                ),
                "late_ampratio": float(
                    np.median([row["avg_ampratio"] for row in late])
                ),
                "best_da": float(
                    max(row["avg_da_per_date"] for row in group)
                ),
                "best_da_epoch": int(
                    max(group, key=lambda row: row["avg_da_per_date"])[
                        "epoch"
                    ]
                ),
                "healthy_epochs": int(
                    sum(bool(row.get("healthy")) for row in group)
                ),
            }
        )
    return output


def plot_mature_dashboard(
    plot_dir: Path,
    summary: list[dict[str, Any]],
    selected: str | None,
) -> None:
    definitions = (
        ("late_da", "Late DA (%)", 100.0, "max"),
        ("late_rankic", "Late RankIC", 1.0, "max"),
        ("late_mape", "Late MAPE (%)", 1.0, "min"),
        ("late_p90_collapse", "Late P90 Collapse (%)", 100.0, "min"),
        ("late_unique", "Late median Unique", 1.0, "max"),
        ("late_ampratio", "Late AmpRatio", 1.0, "one"),
    )
    labels = [row["config"] for row in summary]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    colours = [
        "#d62728" if label == selected else "#4c78a8" for label in labels
    ]
    for axis, (metric, title, scale, direction) in zip(
        axes.ravel(), definitions
    ):
        values = np.asarray([row[metric] * scale for row in summary])
        bars = axis.bar(labels, values, color=colours)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=40)
        if direction == "max":
            best = int(np.argmax(values))
        elif direction == "min":
            best = int(np.argmin(values))
        else:
            axis.axhline(1, color="0.4", linestyle="--", linewidth=1)
            best = int(np.argmin(np.abs(values - 1)))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2)
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
        "Exp 01 mature checkpoint evidence (last 10 epochs; no composite)"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "mature_metric_dashboard.png", dpi=190)
    plt.close(figure)


def plot_capacity_story(
    plot_dir: Path,
    summary: list[dict[str, Any]],
    selected: str | None,
) -> None:
    ordered = sorted(summary, key=lambda row: row["joint_vocab"])
    x = np.asarray([row["joint_vocab"] for row in ordered], dtype=float)
    definitions = (
        ("tokenizer_mae", "Tokenizer MAE", 1.0),
        ("late_da", "Late DA (%)", 100.0),
        ("late_rankic", "Late RankIC", 1.0),
        ("late_p90_collapse", "Late P90 Collapse (%)", 100.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (metric, title, scale) in zip(axes.ravel(), definitions):
        y = np.asarray([row[metric] * scale for row in ordered])
        axis.scatter(x, y, s=55, color="#4c78a8", alpha=0.8)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Joint vocabulary size")
        axis.set_ylabel(title)
        for row, x_value, y_value in zip(ordered, x, y):
            colour = "#d62728" if row["config"] == selected else "black"
            weight = "bold" if row["config"] == selected else "normal"
            axis.annotate(
                row["config"],
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=colour,
                fontweight=weight,
            )
    figure.suptitle(
        "Exp 01 capacity story: reconstruction improves monotonically; "
        "forecast quality does not"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "capacity_story.png", dpi=190)
    plt.close(figure)


def plot_quality_behaviour_map(
    plot_dir: Path,
    summary: list[dict[str, Any]],
    selected: str | None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    for row in summary:
        is_selected = row["config"] == selected
        axis.scatter(
            row["late_p90_collapse"] * 100,
            row["late_da"] * 100,
            s=130 if is_selected else 70,
            color="#d62728" if is_selected else "#4c78a8",
            edgecolor="black" if is_selected else "none",
        )
        axis.annotate(
            row["config"],
            (
                row["late_p90_collapse"] * 100,
                row["late_da"] * 100,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Late P90 daily Collapse (%) — lower is better")
    axis.set_ylabel("Late median daily DA (%) — higher is better")
    axis.set_title("Exp 01 mature quality/behaviour map")
    figure.tight_layout()
    figure.savefig(plot_dir / "mature_quality_behaviour_map.png", dpi=190)
    plt.close(figure)


def render_narrative(
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    selected: str | None,
    rationale: str,
) -> str:
    lines = [
        "# Exp 01 narrative: how much codebook is enough?",
        "",
        f"- Study status: **{manifest.get('status')}**",
        f"- Configurations: **{len(summary)}**",
        f"- GPT checkpoints evaluated: **"
        f"{sum(manifest['settings']['gpt']['epochs'] for _ in summary)}**",
        f"- Health-passing checkpoints: **"
        f"{sum(row['healthy_epochs'] for row in summary)}**",
        "",
        "## Mature checkpoint comparison",
        "",
        "| Bits | Joint vocab | Tok MAE | Late DA | Late RankIC | Late MAPE | "
        "Late P90 collapse | Late unique | Late amp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        marker = " **(selected)**" if row["config"] == selected else ""
        lines.append(
            f"| {row['config']}{marker} | {row['joint_vocab']:,} | "
            f"{row['tokenizer_mae']:.4f} | {row['late_da'] * 100:.2f}% | "
            f"{row['late_rankic']:.4f} | {row['late_mape']:.3f} | "
            f"{row['late_p90_collapse'] * 100:.1f}% | "
            f"{row['late_unique']:.1f} | {row['late_ampratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The six metrics are deliberately not combined into one score. "
            "Tokenizer reconstruction, prediction quality, and prediction "
            "behaviour tell different stories.",
            "",
        ]
    )
    if selected is not None:
        lines.extend(
            [
                "## Dependency decision for Exp 02",
                "",
                f"- Selected bits: **{selected}**",
                f"- Rationale: {rationale}",
                "",
            ]
        )
    lines.extend(
        [
            "## Narrative figures",
            "",
            "- `plots/capacity_story.png`",
            "- `plots/mature_metric_dashboard.png`",
            "- `plots/mature_quality_behaviour_map.png`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--select_config", default="")
    parser.add_argument("--rationale", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    rows = load_json(root / "combined_epoch_summary.json")
    expected = len(manifest["settings"]["configs"]) * int(
        manifest["settings"]["gpt"]["epochs"]
    )
    if manifest.get("status") != "completed" or len(rows) != expected:
        raise RuntimeError(
            f"Exp 01 is incomplete: status={manifest.get('status')}, "
            f"rows={len(rows)}/{expected}"
        )
    summary = mature_rows(rows)
    selected = args.select_config.strip() or None
    known = {row["config"] for row in summary}
    if selected is not None and selected not in known:
        raise ValueError(f"Unknown selected config {selected}; expected {known}")
    rationale = args.rationale.strip()
    if selected is not None and not rationale:
        raise ValueError("--rationale is required with --select_config")

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_mature_dashboard(plot_dir, summary, selected)
    plot_capacity_story(plot_dir, summary, selected)
    plot_quality_behaviour_map(plot_dir, summary, selected)
    atomic_write_json(root / "mature_summary.json", summary)
    if selected is not None:
        chosen = next(row for row in summary if row["config"] == selected)
        config_dir = (
            root
            / "configs"
            / f"bits_{chosen['bits_l1']:02d}_{chosen['bits_l2']:02d}"
        )
        selection = {
            "experiment": "Exp 01 BitSweep rerun",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected": {
                **chosen,
                "bits_l1": chosen["bits_l1"],
                "bits_l2": chosen["bits_l2"],
                "tokenizer_path": str(
                    (config_dir / "tokenizer.pt").resolve()
                ),
            },
            "selection_rule": (
                "Staged non-composite judgment over mature quality, mature "
                "behaviour, tokenizer sufficiency, and codebook complexity."
            ),
            "rationale": rationale,
            "holdout_used": False,
        }
        atomic_write_json(root / "selection.json", selection)
    atomic_write_text(
        root / "NARRATIVE.md",
        render_narrative(manifest, summary, selected, rationale),
    )
    print(
        f"Narrative outputs written to {root}; "
        f"selection={selected or 'not recorded'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
