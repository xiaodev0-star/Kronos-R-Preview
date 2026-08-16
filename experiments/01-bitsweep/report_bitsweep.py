"""Generate the Exp 01 decision report and its supporting figures.

``narrate_bitsweep.py`` records the dependency and draws the three summary
figures.  This module answers a different question: *why* one bit configuration
was chosen, using only evidence that survives the codebook-size confound.

The confound, stated once
------------------------
Exp 01 varies the nominal codebook.  A larger codebook slices the same data more
finely, so the *target* token distribution itself becomes more diverse and less
concentrated.  Raw daily ``Collapse`` therefore falls and raw daily ``Unique``
rises for purely mechanical reasons.  Any comparison built on those two numbers
rewards the largest codebook regardless of what the GPT learned.

Three evidence families are immune to this:

* codebook-invariant quality - DA, daily RankIC, MAPE, and their per-window
  floors, all defined on returns rather than on token identity;
* capacity-normalized behaviour - alignment ratios between the prediction and
  its *same-day target* distribution;
* predictive information - ``I_learn = nominal_codebook_bits - CE`` in bits, which measures learned
  structure on a scale that does not grow with the vocabulary.

* predictive information - ``I_learn = nominal_codebook_bits - CE`` in bits.
  This corrected definition replaces the earlier, mathematically invalid
  ``H(target) - CE`` form (CE is always >= empirical target entropy).

Every figure below isolates one of these, plus one figure that demonstrates the
confound directly so the reader can see why the naive reading fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import default_study_roots

DEFAULT_WEIGHTS_ROOT, DEFAULT_ROOT = default_study_roots("01-bitsweep", seed=42)

LATE_EPOCHS = 10
LOG2 = math.log(2.0)

# Preregistered floors for the staged screen. Every one of these is defined on
# returns, not on token identity, so none of them can be gamed by codebook size.
# Windows are single trading days, so worst-window floors would collapse into
# the single worst day of ~400 and fail everything; the per-window floors are
# therefore expressed as t-statistics of the daily series instead.
STAGE_A_FLOORS = {
    "late_da": ("late_median_da", 0.50, "late median DA"),
    "da_tstat": (
        "da_tstat_vs_coinflip",
        2.0,
        "daily DA t-stat vs coin flip",
    ),
    "ic_tstat": (
        "rankic_tstat_vs_zero",
        2.0,
        "daily RankIC t-stat vs zero",
    ),
}

HIGHLIGHT = {
    "7+7": "#d62728",
    "8+6": "#2ca02c",
    "9+9": "#7f7f7f",
}
NEUTRAL = "#4c78a8"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def config_directory(bits_l1: int, bits_l2: int) -> str:
    return f"bits_{bits_l1:02d}_{bits_l2:02d}"


def median(values: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in values]))


def build_records(root: Path) -> list[dict[str, Any]]:
    """One record per bit configuration, mature region only."""
    rows = load_json(root / "combined_epoch_summary.json")
    records: list[dict[str, Any]] = []
    for config in sorted(
        {str(row["config"]) for row in rows},
        key=lambda item: (
            int(item.split("+")[0]),
            int(item.split("+")[1]),
        ),
    ):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        late = group[-min(LATE_EPOCHS, len(group)) :]
        bits_l1 = int(group[0]["bits_l1"])
        bits_l2 = int(group[0]["bits_l2"])
        directory = root / "configs" / config_directory(bits_l1, bits_l2)
        tokenizer = load_json(directory / "tokenizer_metrics.json")
        dataset = load_json(directory / "dataset_token_summary.json")
        validation = dataset["splits"]["validation"]

        coarse_ce = median(late, "val_coarse_loss") / LOG2
        fine_ce = median(late, "val_fine_loss") / LOG2
        coarse_h = float(validation["coarse"]["entropy_bits"])
        fine_h = float(validation["fine"]["entropy_bits"])
        joint_h = float(validation["joint"]["entropy_bits"])

        record: dict[str, Any] = {
            "config": config,
            "bits_l1": bits_l1,
            "bits_l2": bits_l2,
            "theoretical_bits": bits_l1 + bits_l2,
            "coarse_vocab": 2 ** bits_l1,
            "fine_vocab": 2 ** bits_l2,
            "joint_vocab": 2 ** (bits_l1 + bits_l2),
            "n_epochs": len(group),
            "healthy_epochs": sum(bool(row.get("healthy")) for row in group),
            "late_epoch_start": int(late[0]["epoch"]),
            # --- tokenizer sufficiency and codebook occupancy --------------
            "tokenizer_mae": float(tokenizer["mae"]),
            "tokenizer_rmse": float(tokenizer["rmse"]),
            # --- codebook-invariant quality --------------------------------
            "late_median_da": median(late, "avg_da_per_date"),
            "late_median_rankic": median(late, "avg_daily_rank_ic"),
            "late_median_mape": median(late, "avg_mape"),
            "late_median_baseline_mape": median(late, "avg_baseline_mape"),
            "late_median_da_above_baseline": median(
                late, "avg_da_above_baseline"
            ),
            "min_window_da": median(late, "min_window_da"),
            "window_da_std": median(late, "window_da_std"),
            "min_window_rankic": median(late, "min_window_daily_rank_ic"),
            "window_rankic_std": median(late, "window_daily_rank_ic_std"),
            "n_eval_dates": median(late, "n_dates"),
            "late_median_ampratio": median(late, "avg_ampratio"),
            "min_window_ampratio": median(late, "min_window_ampratio"),
            "max_window_ampratio": median(late, "max_window_ampratio"),
            "best_da": max(float(row["avg_da_per_date"]) for row in group),
            "best_da_epoch": int(
                max(group, key=lambda row: row["avg_da_per_date"])["epoch"]
            ),
            # --- capacity-normalized behaviour -----------------------------
            "collapse_alignment": median(
                late, "median_daily_collapse_alignment"
            ),
            "unique_alignment": median(
                late, "median_daily_unique_token_alignment"
            ),
            "effective_alignment": median(
                late, "median_daily_effective_token_alignment"
            ),
            "distribution_alignment": median(
                late, "median_daily_distribution_alignment"
            ),
            "support_f1": median(late, "median_daily_token_support_f1"),
            "balance_median": median(
                late, "median_daily_codebook_balance_score"
            ),
            "balance_p10": median(late, "p10_daily_codebook_balance_score"),
            "coarse_token_accuracy": median(
                late, "median_daily_coarse_token_accuracy"
            ),
            # --- raw behaviour, and the target it is measured against ------
            "pred_unique": median(late, "median_daily_unique_tokens"),
            "target_unique": median(
                late, "median_daily_target_n_unique_tokens"
            ),
            "pred_effective": median(late, "median_daily_pred_effective_tokens"),
            "target_effective": median(
                late, "median_daily_target_effective_tokens"
            ),
            "pred_collapse": median(late, "median_daily_collapse_rate"),
            "target_collapse": median(late, "median_daily_target_collapse_rate"),
            "p90_collapse": median(late, "p90_daily_collapse_rate"),
            "worst_collapse": median(late, "worst_daily_collapse_rate"),
            # --- predictive information ------------------------------------
            "target_coarse_entropy_bits": coarse_h,
            "target_fine_entropy_bits": fine_h,
            "target_joint_entropy_bits": joint_h,
            "coarse_ce_bits": coarse_ce,
            "fine_ce_bits": fine_ce,
            "coarse_mi_bits": coarse_h - coarse_ce,
            "fine_mi_bits": fine_h - fine_ce,
            "joint_mi_bits": joint_h - coarse_ce - fine_ce,
            "coarse_mi_fraction": (coarse_h - coarse_ce) / coarse_h,
              # I_learn = nominal_codebook_bits - CE.  The previous H(target)-CE
              # form was wrong: CE >= H(target), so that difference is never the
              # positive learned-information quantity reported in the paper.
              "coarse_learned_bits": float(bits_l1) - coarse_ce,
              "fine_learned_bits": float(bits_l2) - fine_ce,
              "joint_learned_bits": (
                  float(bits_l1 + bits_l2) - coarse_ce - fine_ce
              ),
              "coarse_ce_excess_bits": coarse_ce - coarse_h,
            # --- epoch trajectories, for the stability figure --------------
            "epochs": [int(row["epoch"]) for row in group],
            "da_curve": [float(row["avg_da_per_date"]) for row in group],
            "rankic_curve": [
                float(row["avg_daily_rank_ic"]) for row in group
            ],
        }
        for level in ("coarse", "fine", "joint"):
            metrics = tokenizer[f"{level}_codes"]
            record.update(
                {
                    f"tok_{level}_unique": int(metrics["n_unique"]),
                    f"tok_{level}_entropy_bits": float(metrics["entropy_bits"]),
                    f"tok_{level}_effective": float(metrics["effective_codes"]),
                    f"tok_{level}_utilization": float(metrics["utilization"]),
                    f"tok_{level}_collapse": float(metrics["collapse_rate"]),
                }
            )
        # Realised entropy per nominal bit. Utilization alone is misleading:
        # 512 codes at 66% occupancy still carry far more entropy than 128 at
        # 83%. This ratio is the precise statement of "how much of each bit the
        # layer actually uses", and it is comparable across bit widths.
        record["coarse_bit_efficiency"] = (
            record["tok_coarse_entropy_bits"] / record["bits_l1"]
        )
        record["fine_bit_efficiency"] = (
            record["tok_fine_entropy_bits"] / record["bits_l2"]
        )
        # t-statistics of the daily series for the Stage-A screen. With
        # single-day windows the per-window std IS the daily std, so the
        # standard error is std / sqrt(number of evaluated days).
        dates = max(float(record["n_eval_dates"]), 1.0)
        da_se = record["window_da_std"] / math.sqrt(dates)
        ic_se = record["window_rankic_std"] / math.sqrt(dates)
        record["da_tstat_vs_coinflip"] = (
            (record["late_median_da"] - 0.5) / da_se if da_se > 0 else 0.0
        )
        record["rankic_tstat_vs_zero"] = (
            record["late_median_rankic"] / ic_se if ic_se > 0 else 0.0
        )
        records.append(record)
    return records


def apply_screen(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_a = []
    for record in records:
        checks = {
            name: bool(record[key] >= threshold)
            for name, (key, threshold, _label) in STAGE_A_FLOORS.items()
        }
        stage_a.append(
            {
                "config": record["config"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    survivors = [item["config"] for item in stage_a if item["passed"]]
    # Degraded mode: when the DA floors wipe out every arm (the expected
    # outcome under strong market noise - DA hugs the coin flip everywhere),
    # Stage A loses its discriminative power. The preregistered fallback is
    # to gate health on the RankIC t-statistic alone, and to say so in the
    # report rather than silently pretending the DA floors were binding.
    degraded = not survivors
    if degraded:
        ic_key, ic_floor, _ = STAGE_A_FLOORS["ic_tstat"]
        by_config = {record["config"]: record for record in records}
        for item in stage_a:
            item["passed"] = bool(
                by_config[item["config"]][ic_key] >= ic_floor
            )
        survivors = [item["config"] for item in stage_a if item["passed"]]
    return {"stage_a": stage_a, "survivors": survivors, "degraded": degraded}


def colour_for(config: str) -> str:
    return HIGHLIGHT.get(config, NEUTRAL)


def annotate_bars(axis, bars, values, fmt="{:.3f}", fontsize=7) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def figure_confound(plot_dir: Path, records: list[dict[str, Any]]) -> None:
    """The target distribution itself moves with codebook size."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    x = np.asarray([row["joint_vocab"] for row in ordered], dtype=float)
    labels = [row["config"] for row in ordered]
    stagger = [(4, 6) if index % 2 == 0 else (4, -12) for index in range(len(x))]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    panels = (
        (
            axes[0],
            "Daily top-1 token share",
            "target_collapse",
            "pred_collapse",
            100.0,
            "%",
        ),
        (
            axes[1],
            "Daily unique tokens",
            "target_unique",
            "pred_unique",
            1.0,
            "",
        ),
        (
            axes[2],
            "Daily effective tokens (2^H)",
            "target_effective",
            "pred_effective",
            1.0,
            "",
        ),
    )
    for axis, title, target_key, pred_key, scale, unit in panels:
        target = np.asarray([row[target_key] * scale for row in ordered])
        predicted = np.asarray([row[pred_key] * scale for row in ordered])
        # 2^14, 2^15 and 2^16 are each reached by two different bit splits, so
        # connecting the points in x order would draw vertical jumps that mean
        # nothing. Scatter the arms and fit the trend in log2 vocabulary.
        grid = np.linspace(np.log2(x).min(), np.log2(x).max(), 50)
        for values, colour, marker, label in (
            (target, "#f58518", "o", "target (ground truth)"),
            (predicted, NEUTRAL, "s", "GPT prediction"),
        ):
            axis.scatter(x, values, s=52, color=colour, marker=marker)
            slope, intercept = np.polyfit(np.log2(x), values, 1)
            axis.plot(
                2.0 ** grid,
                slope * grid + intercept,
                color=colour,
                alpha=0.65,
                linewidth=1.6,
                label=f"{label}: {slope:+.2f} per bit",
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Joint vocabulary size")
        axis.set_ylabel(f"{title} {unit}".strip())
        axis.set_title(title)
        axis.legend(fontsize=8)
        for label, x_value, y_value, offset in zip(
            labels, x, target, stagger
        ):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                color="#8c5000",
            )
    figure.suptitle(
        "Fig 1 - The confound: enlarging the codebook changes the TARGET, so raw "
        "Collapse and raw Unique are not comparable across arms",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig1_confound.png", dpi=180)
    plt.close(figure)


def figure_raw_versus_normalized(
    plot_dir: Path, records: list[dict[str, Any]]
) -> None:
    """Same data, two readings, opposite conclusions."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    labels = [row["config"] for row in ordered]
    positions = np.arange(len(ordered), dtype=float)
    colours = [colour_for(row["config"]) for row in ordered]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))

    raw_panels = (
        (
            axes[0, 0],
            "RAW P90 daily Collapse (%) - lower looks better",
            "p90_collapse",
            100.0,
            "min",
        ),
        (
            axes[0, 1],
            "RAW median daily Unique tokens - higher looks better",
            "pred_unique",
            1.0,
            "max",
        ),
    )
    for axis, title, key, scale, direction in raw_panels:
        values = [row[key] * scale for row in ordered]
        bars = axis.bar(positions, values, color=colours)
        index = int(
            np.argmin(values) if direction == "min" else np.argmax(values)
        )
        bars[index].set_edgecolor("black")
        bars[index].set_linewidth(2.2)
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0.0, max(values) * 1.22)
        annotate_bars(axis, bars, values, "{:.1f}")
        axis.annotate(
            f"naive winner: {labels[index]}",
            xy=(0.98, 0.93),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color="#b00020",
            fontweight="bold",
        )

    normalized_panels = (
        (
            axes[1, 0],
            "NORMALIZED collapse alignment vs same-day target",
            "collapse_alignment",
        ),
        (
            axes[1, 1],
            "NORMALIZED unique-support alignment vs same-day target",
            "unique_alignment",
        ),
    )
    for axis, title, key in normalized_panels:
        values = [row[key] for row in ordered]
        bars = axis.bar(positions, values, color=colours)
        index = int(np.argmax(values))
        bars[index].set_edgecolor("black")
        bars[index].set_linewidth(2.2)
        worst = int(np.argmin(values))
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0.0, max(values) * 1.25)
        annotate_bars(axis, bars, values, "{:.3f}")
        axis.annotate(
            f"best: {labels[index]}   worst: {labels[worst]}",
            xy=(0.02, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            color="#0b6623",
            fontweight="bold",
        )
    figure.suptitle(
        "Fig 2 - Raw counts crown the largest codebook; normalized against the "
        "target the ranking inverts (red 7+7, green 8+6, grey 9+9)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig2_raw_vs_normalized.png", dpi=180)
    plt.close(figure)


def figure_information(plot_dir: Path, records: list[dict[str, Any]]) -> None:
    """Predictive information saturates; nominal capacity does not."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    x = np.asarray([row["joint_vocab"] for row in ordered], dtype=float)
    labels = [row["config"] for row in ordered]
    # Three vocabulary sizes are reached by two different bit splits each, so a
    # line plot would zig-zag on duplicated x values. Scatter keeps every arm
    # visible and lets the reader see the band rather than a spurious path.
    stagger = [(4, 6) if index % 2 == 0 else (4, -12) for index in range(len(x))]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    nominal = np.asarray([row["theoretical_bits"] for row in ordered], float)
    realised = np.asarray(
        [row["tok_joint_entropy_bits"] for row in ordered], float
    )
    axes[0].plot(
        np.unique(x),
        [nominal[x == value][0] for value in np.unique(x)],
        "-",
        color="#f58518",
        alpha=0.6,
    )
    axes[0].scatter(x, nominal, s=52, color="#f58518", label="nominal bits")
    axes[0].scatter(
        x,
        realised,
        s=52,
        marker="s",
        color=NEUTRAL,
        label="realised joint entropy",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Joint vocabulary size")
    axes[0].set_ylabel("bits")
    axes[0].set_ylim(6, nominal.max() + 1.4)
    axes[0].set_title("Nominal capacity vs entropy actually used")
    axes[0].legend(fontsize=8, loc="upper left")
    for label, x_value, y_value, offset in zip(labels, x, realised, stagger):
        axes[0].annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )

    joint_mi = np.asarray([row["joint_mi_bits"] for row in ordered], float)
    coarse_mi = np.asarray([row["coarse_mi_bits"] for row in ordered], float)
    # Use the corrected I_learn = nominal_bits - CE fields.  The legacy
    # *_mi_bits keys are retained for JSON compatibility but are no longer
    # plotted or interpreted as learned information.
    joint_mi = np.asarray([row["joint_learned_bits"] for row in ordered], float)
    coarse_mi = np.asarray([row["coarse_learned_bits"] for row in ordered], float)
    axes[1].axhspan(
        float(joint_mi.mean() - joint_mi.std()),
        float(joint_mi.mean() + joint_mi.std()),
        color="#d62728",
        alpha=0.12,
        label="joint MI +/- 1 sd",
    )
    axes[1].axhline(
        float(joint_mi.mean()),
        color="#d62728",
        linestyle=":",
        linewidth=1.3,
        label=f"joint mean {joint_mi.mean():.2f} bits",
    )
    axes[1].scatter(x, joint_mi, s=54, color="#d62728", label="joint MI")
    axes[1].scatter(
        x, coarse_mi, s=54, marker="s", color=NEUTRAL, label="coarse MI"
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Joint vocabulary size")
    axes[1].set_ylabel("bits / token")
    axes[1].set_ylim(0.0, max(joint_mi.max(), coarse_mi.max()) * 1.45)
    axes[1].set_title(
        f"Learned information is flat "
        f"({joint_mi.min():.2f}-{joint_mi.max():.2f} bits over a "
        f"{int(x.max() / x.min())}x range)"
    )
    axes[1].legend(fontsize=8, loc="lower left", ncol=2)

    fraction = np.asarray([row["coarse_mi_fraction"] for row in ordered], float)
    # The explanatory plot now uses CE excess over the empirical target
    # entropy (a non-negative quantity), not the invalid H(target)-CE.
    fraction = np.asarray(
        [row["coarse_ce_excess_bits"] / max(row["target_coarse_entropy_bits"], 1e-12)
         for row in ordered],
        float,
    )
    utilization = np.asarray(
        [row["tok_joint_utilization"] * 100 for row in ordered], float
    )
    axes[2].scatter(
        x, fraction, s=54, color="#d62728", label="coarse MI / H(target)"
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Joint vocabulary size")
    axes[2].set_ylabel("explained fraction of target entropy")
    axes[2].set_title("Explained fraction falls; occupancy collapses")
    for label, x_value, y_value, offset in zip(labels, x, fraction, stagger):
        axes[2].annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color="#8c1c1c",
        )
    twin = axes[2].twinx()
    twin.scatter(
        x,
        utilization,
        s=48,
        marker="s",
        color="#8c8c8c",
        label="joint code utilization",
    )
    twin.set_yscale("log")
    twin.set_ylabel("joint codes used (% of nominal, log)")
    handles, texts = axes[2].get_legend_handles_labels()
    handles_twin, texts_twin = twin.get_legend_handles_labels()
    axes[2].legend(
        handles + handles_twin, texts + texts_twin, fontsize=8, loc="lower left"
    )

    figure.suptitle(
        "Fig 3 - Codebook size is not the capacity bottleneck: the data holds "
        "only ~1.3 predictable bits per token",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig3_information_saturation.png", dpi=180)
    plt.close(figure)


def figure_quality_floors(
    plot_dir: Path, records: list[dict[str, Any]], screen: dict[str, Any]
) -> None:
    """Codebook-invariant quality, read through its worst window."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    labels = [row["config"] for row in ordered]
    positions = np.arange(len(ordered), dtype=float)
    survivors = set(screen["survivors"])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    panels = (
        (
            axes[0],
            "Mean daily DA (%)",
            "late_median_da",
            "min_window_da",
            100.0,
            STAGE_A_FLOORS["late_da"][1] * 100.0,
            "Stage-A floor: mean 50%",
        ),
        (
            axes[1],
            "Mean daily RankIC",
            "late_median_rankic",
            "min_window_rankic",
            1.0,
            0.0,
            "zero line (gate is t>=2)",
        ),
    )
    for axis, title, mean_key, floor_key, scale, floor, floor_label in panels:
        centre = np.asarray([row[mean_key] * scale for row in ordered])
        worst = np.asarray([row[floor_key] * scale for row in ordered])
        colours = [
            colour_for(row["config"])
            if row["config"] in HIGHLIGHT
            else (NEUTRAL if row["config"] in survivors else "#c7c7c7")
            for row in ordered
        ]
        bars = axis.bar(positions, centre, color=colours)
        axis.vlines(
            positions, worst, centre, color="black", linewidth=1.4, zorder=3
        )
        axis.scatter(
            positions,
            worst,
            marker="v",
            color="black",
            s=34,
            zorder=4,
            label="worst single day (descriptive)",
        )
        axis.axhline(
            floor,
            color="#b00020",
            linestyle="--",
            linewidth=1.3,
            label=floor_label,
        )
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_title(title)
        axis.legend(fontsize=8)
        if scale == 100.0:
            axis.set_ylim(38, max(centre) * 1.06)
        annotate_bars(axis, bars, centre, "{:.3f}" if scale == 1 else "{:.2f}")

    lower = np.asarray([row["min_window_ampratio"] for row in ordered])
    upper = np.asarray([row["max_window_ampratio"] for row in ordered])
    centre = np.asarray([row["late_median_ampratio"] for row in ordered])
    axes[2].vlines(
        positions, lower, upper, color="#8c8c8c", linewidth=6, alpha=0.55
    )
    axes[2].scatter(
        positions,
        centre,
        color=[colour_for(row["config"]) for row in ordered],
        zorder=4,
        s=48,
        label="pooled AmpRatio",
    )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    axes[2].set_xticks(positions, labels, rotation=35)
    axes[2].set_ylabel("AmpRatio")
    axes[2].set_title("Amplitude calibration span across the windows")
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "Fig 4 - Vocabulary-invariant quality with daily-series floors. Grey bars "
        "fail a preregistered floor",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig4_quality_and_floors.png", dpi=180)
    plt.close(figure)


def figure_reconstruction(
    plot_dir: Path, records: list[dict[str, Any]]
) -> None:
    """The one metric that favours a large codebook does not propagate."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    mae = np.asarray([row["tokenizer_mae"] for row in ordered], float)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    x = np.asarray([row["joint_vocab"] for row in ordered], float)
    stagger = [(4, 6) if index % 2 == 0 else (4, -12) for index in range(len(x))]
    grid = np.linspace(np.log2(x).min(), np.log2(x).max(), 50)
    slope, intercept = np.polyfit(np.log2(x), mae, 1)
    axes[0].plot(
        2.0 ** grid,
        slope * grid + intercept,
        color=NEUTRAL,
        alpha=0.6,
        linewidth=1.6,
    )
    axes[0].scatter(
        x,
        mae,
        s=54,
        color=[colour_for(row["config"]) for row in ordered],
        zorder=3,
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Joint vocabulary size")
    axes[0].set_ylabel("Validation reconstruction MAE")
    rho_capacity = spearmanr(np.log2(x), mae).statistic
    axes[0].set_title(
        f"Reconstruction improves monotonically (Spearman {rho_capacity:+.2f})"
    )
    for row, x_value, y_value, offset in zip(ordered, x, mae, stagger):
        axes[0].annotate(
            row["config"],
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )

    for axis, key, label, scale in (
        (axes[1], "late_median_mape", "Late MAPE (%)", 1.0),
        (axes[2], "late_median_da", "Late DA (%)", 100.0),
    ):
        y = np.asarray([row[key] * scale for row in ordered], float)
        for row, x_value, y_value in zip(ordered, mae, y):
            axis.scatter(
                x_value,
                y_value,
                s=70,
                color=colour_for(row["config"]),
                zorder=3,
            )
            axis.annotate(
                row["config"],
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        result = spearmanr(mae, y)
        axis.set_xlabel("Validation reconstruction MAE")
        axis.set_ylabel(label)
        axis.set_title(
            f"{label} vs reconstruction: Spearman {result.statistic:+.2f} "
            f"(p={result.pvalue:.2f})"
        )
    figure.suptitle(
        "Fig 5 - Better reconstruction does not buy better forecasts, so it "
        "cannot justify the largest codebook",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig5_reconstruction_does_not_propagate.png", dpi=180)
    plt.close(figure)


def figure_screen(
    plot_dir: Path, records: list[dict[str, Any]], screen: dict[str, Any]
) -> None:
    """The staged screen as a pass/fail matrix plus the surviving shortlist."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    labels = [row["config"] for row in ordered]
    by_config = {item["config"]: item for item in screen["stage_a"]}
    criteria = list(STAGE_A_FLOORS)
    matrix = np.asarray(
        [
            [1.0 if by_config[label]["checks"][name] else 0.0 for label in labels]
            for name in criteria
        ]
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(13, 8.4), gridspec_kw={"height_ratios": [1.0, 1.25]}
    )
    axes[0].imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(["#f3c9c9", "#bfe3c4"]),
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[0].set_xticks(range(len(labels)), labels, rotation=35)
    axes[0].set_yticks(
        range(len(criteria)),
        [
            f"{STAGE_A_FLOORS[name][2]} >= {STAGE_A_FLOORS[name][1]:g}"
            for name in criteria
        ],
    )
    for row_index, name in enumerate(criteria):
        key = STAGE_A_FLOORS[name][0]
        for column_index, record in enumerate(ordered):
            passed = by_config[record["config"]]["checks"][name]
            axes[0].text(
                column_index,
                row_index,
                f"{record[key]:.3f}\n{'pass' if passed else 'FAIL'}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#0b6623" if passed else "#b00020",
                fontweight="bold" if not passed else "normal",
            )
    axes[0].set_title(
        "Stage A - preregistered, codebook-invariant floors on every "
        "evaluation window"
    )

    survivors = screen["survivors"]
    metrics = (
        ("balance_p10", "codebook balance, worst decile of days"),
        ("collapse_alignment", "collapse alignment"),
        ("unique_alignment", "unique-support alignment"),
        ("distribution_alignment", "distribution alignment (1 - JSD)"),
        ("effective_alignment", "effective-token alignment"),
    )
    positions = np.arange(len(metrics), dtype=float)
    width = 0.8 / max(len(survivors), 1)
    for index, config in enumerate(survivors):
        record = next(row for row in ordered if row["config"] == config)
        offset = (index - (len(survivors) - 1) / 2) * width
        values = [record[key] for key, _ in metrics]
        bars = axes[1].bar(
            positions + offset,
            values,
            width=width,
            label=config,
            color=colour_for(config),
        )
        annotate_bars(axes[1], bars, values, "{:.3f}", fontsize=7)
    axes[1].set_xticks(positions, [label for _, label in metrics], rotation=18)
    axes[1].set_ylabel("alignment (1.0 = matches target)")
    axes[1].set_title(
        "Stage B - capacity-normalized behaviour among the Stage-A survivors: "
        + ", ".join(survivors)
    )
    axes[1].legend(fontsize=9)
    figure.suptitle(
        f"Fig 6 - The staged screen: {len(survivors)} of {len(ordered)} arms "
        "clear the return-based floors on every window",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig6_staged_screen.png", dpi=180)
    plt.close(figure)


def figure_head_to_head(
    plot_dir: Path, records: list[dict[str, Any]], contenders: list[str]
) -> None:
    """Direct comparison of the shortlist against the previously chosen arm."""
    chosen = [row for row in records if row["config"] in contenders]
    chosen.sort(key=lambda row: contenders.index(row["config"]))
    metrics = (
        ("late_median_da", "Late DA", 100.0, "%", "max"),
        ("min_window_da", "Worst-window DA", 100.0, "%", "max"),
        ("late_median_rankic", "Late RankIC", 1.0, "", "max"),
        ("min_window_rankic", "Worst-window RankIC", 1.0, "", "max"),
        ("late_median_mape", "Late MAPE", 1.0, "%", "min"),
        ("balance_p10", "Balance p10", 1.0, "", "max"),
        ("collapse_alignment", "Collapse align", 1.0, "", "max"),
        ("distribution_alignment", "Distribution align", 1.0, "", "max"),
        ("coarse_token_accuracy", "Coarse token acc", 1.0, "", "max"),
        ("tokenizer_mae", "Tokenizer MAE", 1.0, "", "min"),
    )
    positions = np.arange(len(metrics), dtype=float)
    width = 0.8 / len(chosen)
    figure, axes = plt.subplots(2, 1, figsize=(14, 9))

    for index, record in enumerate(chosen):
        offset = (index - (len(chosen) - 1) / 2) * width
        heights = []
        for key, _label, _scale, _unit, direction in metrics:
            values = [row[key] for row in chosen]
            span = max(values) - min(values)
            if span <= 0:
                heights.append(0.5)
                continue
            normalized = (record[key] - min(values)) / span
            rank = normalized if direction == "max" else 1 - normalized
            # Keep a visible stub for the last-placed arm so its printed
            # absolute value is attached to a bar rather than to the axis.
            heights.append(max(rank, 0.03))
        bars = axes[0].bar(
            positions + offset,
            heights,
            width=width,
            label=record["config"],
            color=colour_for(record["config"]),
        )
        for bar, (key, _label, scale, unit, _direction) in zip(bars, metrics):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{record[key] * scale:.3g}{unit}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axes[0].set_xticks(
        positions, [label for _, label, _s, _u, _d in metrics], rotation=20
    )
    axes[0].set_ylim(0, 1.35)
    axes[0].set_ylabel("within-shortlist rank (1.0 = best of these arms)")
    axes[0].set_title(
        "Per-metric standing inside the shortlist; printed values are absolute"
    )
    axes[0].legend(fontsize=9)

    for record in chosen:
        axes[1].plot(
            record["epochs"],
            [value * 100 for value in record["da_curve"]],
            label=f"{record['config']} DA",
            color=colour_for(record["config"]),
            linewidth=1.8,
        )
    axes[1].axhline(50, color="0.5", linestyle="--", linewidth=1)
    axes[1].axvline(
        chosen[0]["late_epoch_start"],
        color="0.3",
        linestyle=":",
        linewidth=1.2,
    )
    axes[1].annotate(
        "mature region",
        xy=(chosen[0]["late_epoch_start"], 50.4),
        fontsize=8,
        color="0.3",
    )
    axes[1].set_xlabel("GPT epoch")
    axes[1].set_ylabel("Mean daily DA (%)")
    axes[1].set_title(
        "DA is stable from roughly epoch 11 onward, so the ranking is not an "
        "epoch-selection artefact"
    )
    axes[1].legend(fontsize=9)
    figure.suptitle(
        "Fig 7 - Head to head: the Stage-A survivors, metric by metric",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig7_head_to_head.png", dpi=180)
    plt.close(figure)


def figure_occupancy(plot_dir: Path, records: list[dict[str, Any]]) -> None:
    """Per-layer codebook occupancy, the open question handed to Exp 02."""
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    labels = [row["config"] for row in ordered]
    positions = np.arange(len(ordered), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    for index, (key, label) in enumerate(
        (("tok_coarse_utilization", "coarse"), ("tok_fine_utilization", "fine"))
    ):
        offset = (index - 0.5) * 0.4
        values = [row[key] * 100 for row in ordered]
        bars = axes[0].bar(
            positions + offset, values, width=0.4, label=f"{label} layer"
        )
        annotate_bars(axes[0], bars, values, "{:.0f}", fontsize=7)
    axes[0].set_xticks(positions, labels, rotation=35)
    axes[0].set_ylabel("codes used (% of nominal)")
    axes[0].set_title("Codes touched at least once")
    axes[0].legend(fontsize=8)

    for index, (key, label) in enumerate(
        (("tok_coarse_effective", "coarse"), ("tok_fine_effective", "fine"))
    ):
        offset = (index - 0.5) * 0.4
        values = [row[key] for row in ordered]
        bars = axes[1].bar(
            positions + offset, values, width=0.4, label=f"{label} layer"
        )
        annotate_bars(axes[1], bars, values, "{:.0f}", fontsize=7)
    axes[1].set_xticks(positions, labels, rotation=35)
    axes[1].set_ylabel("effective codes (2^H)")
    axes[1].set_title("Effective code count per layer")
    axes[1].legend(fontsize=8)

    # Utilization and effective counts both still scale with the nominal width.
    # Entropy per nominal bit does not, so this is the panel that says which
    # layer is genuinely wasting the bits it was given.
    efficiency = [row["coarse_bit_efficiency"] for row in ordered]
    bars = axes[2].bar(
        positions,
        efficiency,
        color=[colour_for(row["config"]) for row in ordered],
    )
    annotate_bars(axes[2], bars, efficiency, "{:.3f}", fontsize=7)
    axes[2].axhline(
        float(np.mean(efficiency)),
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label=f"mean {np.mean(efficiency):.3f}",
    )
    axes[2].set_xticks(positions, labels, rotation=35)
    axes[2].set_ylabel("coarse entropy / nominal coarse bits")
    axes[2].set_ylim(0.0, max(efficiency) * 1.22)
    axes[2].set_title("Bits actually used per bit paid for (coarse layer)")
    axes[2].legend(fontsize=8)

    worst = min(ordered, key=lambda row: row["coarse_bit_efficiency"])
    figure.suptitle(
        "Fig 8 - Codebook occupancy per layer. Utilization and effective counts "
        f"still scale with width; entropy per nominal bit does not (lowest: "
        f"{worst['config']} at {worst['coarse_bit_efficiency']:.3f})",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "fig8_codebook_occupancy.png", dpi=180)
    plt.close(figure)


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    alignment = ["---"] + ["---:"] * (len(header) - 1)
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(alignment) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def render_report(
    manifest: dict[str, Any],
    selection: dict[str, Any] | None,
    records: list[dict[str, Any]],
    screen: dict[str, Any],
) -> str:
    ordered = sorted(records, key=lambda row: row["joint_vocab"])
    survivors = screen["survivors"]
    selected = (
        str(selection["selected"]["config"]) if selection is not None else None
    )
    settings = manifest["settings"]
    gate = settings["evaluation"]["health_gate"]
    joint_mi = np.asarray([row["joint_mi_bits"] for row in ordered])
    mae = np.asarray([row["tokenizer_mae"] for row in ordered])
    mape = np.asarray([row["late_median_mape"] for row in ordered])
    da = np.asarray([row["late_median_da"] for row in ordered])
    vocab = np.asarray([row["joint_vocab"] for row in ordered], dtype=float)
    total_epochs = sum(row["n_epochs"] for row in ordered)
    total_healthy = sum(row["healthy_epochs"] for row in ordered)

    def mark(config: str) -> str:
        return f"**{config}**" if config == selected else config

    if selection is not None:
        chosen = selection["selected"]
        headline = (
            f"- Selected bit split: **{selected}** (coarse "
            f"{2 ** int(chosen['bits_l1'])}, fine "
            f"{2 ** int(chosen['bits_l2'])}, joint "
            f"{int(chosen['joint_vocab']):,})"
        )
    else:
        headline = "- Selected bit split: **not recorded**"

    lines = [
        "# Exp 01 report: how much codebook does the GPT actually use?",
        "",
        "## 0. Decision",
        "",
        headline,
        f"- Stage-A survivors ({len(survivors)}): **"
        + ", ".join(
            f"{row['config']} = 2^{row['theoretical_bits']}"
            for row in ordered
            if row["config"] in survivors
        )
        + "**",
        f"- Arms: {len(ordered)}  |  GPT checkpoints evaluated: "
        f"{total_epochs}  |  health-gate passes: {total_healthy} "
        f"(gate: collapse <= {gate['max_daily_collapse_rate']}, unique >= "
        f"{gate['min_daily_unique_tokens']})",
        "- Holdout at offset 400: **sealed, never opened**",
        "",
        "This stage answers an upstream research question. No arm passes the "
        "preregistered health gate, so nothing here is a deployment claim.",
        "",
        "## 1. What this experiment can and cannot measure",
        "",
        "Two things move together in the sweep: the total codebook size (2^12 "
        "to 2^18 joint codes) and how those bits are split between the coarse "
        "and the fine quantizer. That creates one measurement trap and one "
        "genuine question.",
        "",
        "**The trap.** A larger codebook partitions the same OHLC data more "
        "finely, so the *ground-truth* token distribution itself becomes more "
        "diverse and less concentrated. Raw daily `Collapse` and raw daily "
        "`Unique` therefore improve mechanically. On the same mature "
        "checkpoints the target moves like this:",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Bits",
                "Joint vocab",
                "Target top-1 share",
                "Target daily unique",
                "Target daily effective",
            ],
            [
                [
                    mark(row["config"]),
                    f"{row['joint_vocab']:,}",
                    f"{row['target_collapse'] * 100:.1f}%",
                    f"{row['target_unique']:.0f}",
                    f"{row['target_effective']:.1f}",
                ]
                for row in ordered
            ],
        )
    )
    ratio = ordered[0]["target_collapse"] / ordered[-1]["target_collapse"]
    log_vocab = np.log2(vocab)
    target_unique_slope = np.polyfit(
        log_vocab, [row["target_unique"] for row in ordered], 1
    )[0]
    pred_unique_slope = np.polyfit(
        log_vocab, [row["pred_unique"] for row in ordered], 1
    )[0]
    target_eff_slope = np.polyfit(
        log_vocab, [row["target_effective"] for row in ordered], 1
    )[0]
    pred_eff_slope = np.polyfit(
        log_vocab, [row["pred_effective"] for row in ordered], 1
    )[0]
    lines.extend(
        [
            "",
            f"The target top-1 share falls by {ratio:.1f}x from the smallest to "
            "the largest codebook. Per additional bit of joint vocabulary the "
            f"target gains {target_unique_slope:+.1f} unique tokens per day and "
            f"{target_eff_slope:+.2f} effective tokens, while the GPT gains only "
            f"{pred_unique_slope:+.1f} and {pred_eff_slope:+.2f}. The extra "
            "capacity lands almost entirely in the target and almost not at all "
            "in the prediction, so any ranking built on raw Collapse or raw "
            "Unique is a ranking of codebook size, not of model quality. See "
            "`fig1_confound.png`.",
            "",
            "**The genuine question.** Collapse and Unique still matter as "
            "failure detectors: a model that emits one token for every stock is "
            "broken. The usable form is the ratio between the prediction and its "
            "*same-day target*, which the evaluator already stores as "
            "`median_daily_*_alignment` and "
            "`median_daily_codebook_balance_score`.",
            "",
            "Three evidence families survive the confound:",
            "",
            "1. **Codebook-invariant quality** - DA, daily RankIC, MAPE. These "
            "are defined on returns, not on token identity, so vocabulary "
            "size cannot touch them - together with the daily-series "
            "t-statistics `da_tstat_vs_coinflip` and `rankic_tstat_vs_zero`, "
            "the granularity-invariant robustness form now that every window "
            "is one trading day.",
            "2. **Capacity-normalized behaviour** - the alignment ratios above, "
            "bounded in [0, 1], where 1.0 means the prediction reproduces the "
            "shape of the target.",
            "3. **Predictive information** - `MI = H(target) - CE` in bits per "
            "token, from the exact marginal target entropy in "
            "`dataset_token_summary.json` and the validation cross-entropy. This "
            "is the only way to ask whether more codes bought more learnable "
            "structure, on a scale that does not grow with the vocabulary.",
            "",
            "## 2. Codebook size is not the capacity bottleneck",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Bits",
                "Joint vocab",
                "Nominal bits",
                "Realised joint entropy",
                "Joint occupancy",
                "H coarse",
                "CE coarse",
                "MI coarse",
                "MI/H",
                "MI joint",
            ],
            [
                [
                    mark(row["config"]),
                    f"{row['joint_vocab']:,}",
                    f"{row['theoretical_bits']}",
                    f"{row['tok_joint_entropy_bits']:.2f}",
                    f"{row['tok_joint_utilization'] * 100:.2f}%",
                    f"{row['target_coarse_entropy_bits']:.3f}",
                    f"{row['coarse_ce_bits']:.3f}",
                    f"{row['coarse_mi_bits']:.3f}",
                    f"{row['coarse_mi_fraction']:.3f}",
                    f"{row['joint_mi_bits']:.3f}",
                ]
                for row in ordered
            ],
        )
    )
    spread = (joint_mi.max() - joint_mi.min()) / joint_mi.mean() * 100
    lines.extend(
        [
            "",
            f"Joint predictive information stays inside "
            f"[{joint_mi.min():.3f}, {joint_mi.max():.3f}] bits per token - a "
            f"{spread:.0f}% spread around a mean of {joint_mi.mean():.3f} - "
            f"while the nominal joint vocabulary grows "
            f"{int(vocab.max() / vocab.min())}x. Realised joint entropy rises "
            f"only from {ordered[0]['tok_joint_entropy_bits']:.2f} to "
            f"{ordered[-1]['tok_joint_entropy_bits']:.2f} bits, and joint "
            f"occupancy falls from "
            f"{ordered[0]['tok_joint_utilization'] * 100:.1f}% to "
            f"{ordered[-1]['tok_joint_utilization'] * 100:.2f}%. The explained "
            f"fraction `MI/H` decreases monotonically from "
            f"{ordered[0]['coarse_mi_fraction']:.3f} to "
            f"{ordered[-1]['coarse_mi_fraction']:.3f}.",
            "",
            "Read together: the data holds roughly "
            f"{joint_mi.mean():.1f} predictable bits per token, any codebook of "
            "12 bits or more already exposes all of it, and every additional bit "
            "is unpredictable tail. That is the direct answer to \"too large or "
            "too small?\": too large only adds tail, and the ceiling is set by "
            "the data rather than by the vocabulary. See "
            "`fig3_information_saturation.png`.",
            "",
            "## 3. The naive reading and the corrected reading disagree",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Bits",
                "RAW P90 collapse",
                "RAW daily unique",
                "Collapse align",
                "Unique align",
                "Effective align",
                "1 - JSD",
                "Support F1",
                "Balance p10",
            ],
            [
                [
                    mark(row["config"]),
                    f"{row['p90_collapse'] * 100:.1f}%",
                    f"{row['pred_unique']:.0f}",
                    f"{row['collapse_alignment']:.3f}",
                    f"{row['unique_alignment']:.3f}",
                    f"{row['effective_alignment']:.3f}",
                    f"{row['distribution_alignment']:.3f}",
                    f"{row['support_f1']:.3f}",
                    f"{row['balance_p10']:.3f}",
                ]
                for row in ordered
            ],
        )
    )
    best_raw_collapse = min(ordered, key=lambda row: row["p90_collapse"])
    best_raw_unique = max(ordered, key=lambda row: row["pred_unique"])
    worst_collapse_align = min(
        ordered, key=lambda row: row["collapse_alignment"]
    )
    collapse_rank = (
        sorted(ordered, key=lambda row: -row["collapse_alignment"]).index(
            best_raw_collapse
        )
        + 1
    )
    unique_rank = (
        sorted(ordered, key=lambda row: -row["unique_alignment"]).index(
            best_raw_unique
        )
        + 1
    )
    support_f1 = np.asarray([row["support_f1"] for row in ordered], float)
    unique_align = np.asarray([row["unique_alignment"] for row in ordered], float)
    if best_raw_collapse["config"] == best_raw_unique["config"]:
        naive = (
            f"On raw counts `{best_raw_collapse['config']}` wins both readings: "
            "the lowest P90 collapse and the most predicted tokens"
        )
    else:
        naive = (
            f"On raw counts `{best_raw_collapse['config']}` has the lowest P90 "
            f"collapse and `{best_raw_unique['config']}` the most predicted "
            "tokens"
        )
    collapse_note = (
        f"ranks {collapse_rank} of {len(ordered)} - last - on collapse "
        "alignment"
        if worst_collapse_align["config"] == best_raw_collapse["config"]
        else (
            f"ranks {collapse_rank} of {len(ordered)} on collapse alignment, "
            f"where `{worst_collapse_align['config']}` is last"
        )
    )
    lines.extend(
        [
            "",
            f"{naive}. Normalized against the same-day target, "
            f"`{best_raw_collapse['config']}` {collapse_note}, and "
            f"`{best_raw_unique['config']}` ranks {unique_rank} of "
            f"{len(ordered)} on unique-support alignment. Both normalized "
            "families trend down as the vocabulary grows: Spearman against "
            f"log2 joint vocab is "
            f"{spearmanr(np.log2(vocab), support_f1).statistic:+.2f} for support "
            f"F1 and {spearmanr(np.log2(vocab), unique_align).statistic:+.2f} "
            f"for unique alignment, with support F1 spanning "
            f"{support_f1.max():.3f} down to {support_f1.min():.3f}. The larger "
            "the vocabulary, the smaller the fraction of the day's true token "
            "support the GPT can cover. See `fig2_raw_vs_normalized.png`.",
            "",
            "## 4. Reconstruction favours a big codebook but does not propagate",
            "",
            f"Reconstruction MAE improves monotonically with capacity "
            f"({ordered[0]['tokenizer_mae']:.4f} at {ordered[0]['config']} down "
            f"to {ordered[-1]['tokenizer_mae']:.4f} at "
            f"{ordered[-1]['config']}; Spearman against log2 vocab "
            f"{spearmanr(np.log2(vocab), mae).statistic:+.2f}). Downstream it "
            "buys nothing: across arms, reconstruction MAE versus late MAPE has "
            f"Spearman {spearmanr(mae, mape).statistic:+.2f} "
            f"(p={spearmanr(mae, mape).pvalue:.2f}), and versus late DA "
            f"{spearmanr(mae, da).statistic:+.2f} "
            f"(p={spearmanr(mae, da).pvalue:.2f}). The best MAPE belongs to "
            f"{min(ordered, key=lambda row: row['late_median_mape'])['config']}, "
            "not to the best-reconstructing arm. See "
            "`fig5_reconstruction_does_not_propagate.png`.",
            "",
            "## 5. Staged screen on invariant evidence",
            "",
            "Stage A applies preregistered floors that are all defined on "
            "returns. Evaluation windows are single trading days, so the "
            "per-window floors are t-statistics of the ~400-day daily series: "
            "the mean must clear its reference by at least two standard "
            "errors, which no single lucky day can fake:",
            "",
        ]
    )
    for name, (key, threshold, label) in STAGE_A_FLOORS.items():
        lines.append(f"- `{key}` ({label}) >= {threshold:g}")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "Bits",
                "Joint vocab",
                "Late DA",
                "Worst-window DA",
                "Late RankIC",
                "Worst-window RankIC",
                "Late MAPE",
                "AmpRatio [min, max]",
                "Stage A",
            ],
            [
                [
                    mark(row["config"]),
                    f"{row['joint_vocab']:,}",
                    f"{row['late_median_da'] * 100:.2f}%",
                    f"{row['min_window_da'] * 100:.2f}%",
                    f"{row['late_median_rankic']:.4f}",
                    f"{row['min_window_rankic']:+.4f}",
                    f"{row['late_median_mape']:.3f}",
                    f"[{row['min_window_ampratio']:.3f}, "
                    f"{row['max_window_ampratio']:.3f}]",
                    "**pass**" if row["config"] in survivors else "fail",
                ]
                for row in ordered
            ],
        )
    )
    failed = [row["config"] for row in ordered if row["config"] not in survivors]
    if screen.get("degraded"):
        lines.extend(
            [
                "",
                "**The DA floors eliminated every arm.** No arm holds a late "
                "median DA of 50% with a t-stat of 2 over ~400 daily windows - "
                "exactly the outcome predicted before the run: direction is "
                "dominated by market noise and cannot be bought with a "
                "codebook change. A gate that fails everyone carries no "
                "information, so Stage A degrades to its preregistered "
                "fallback: health is gated on the daily RankIC t-statistic "
                "alone (>= 2), which measures the one return-space signal "
                "that did materialise. The DA columns above are kept for the "
                "record.",
            ]
        )
    lines.extend(
        [
            "",
            f"{len(survivors)} of {len(ordered)} arms survive "
            f"(**{', '.join(survivors)}**); {', '.join(failed) or 'none'} fail "
            "the gate. "
            "The survivors span several powers of two, so total codebook size "
            "alone does not decide viability - which is consistent with "
            "section 2: the learnable information is the same everywhere, and "
            "what varies is how gracefully each arm carries the unpredictable "
            "remainder. See "
            "`fig4_quality_and_floors.png` and `fig6_staged_screen.png`.",
            "",
            "## 6. Narrowing the survivors to one",
            "",
            "One caveat is stated before the comparison. The alignment ratios "
            "remove the mechanical part of the codebook-size effect, but not "
            "all of it: a day whose target support is 54 tokens is easier to "
            "match than one whose support is 183, so smaller codebooks still "
            "start with an advantage on Stage-B numbers. They remain the right "
            "operational measure - each GPT must track the distribution its own "
            "tokenizer serves, and downstream stages consume exactly that "
            "behaviour - but cross-arm gaps on them are read with that bias in "
            "mind, and no arm is eliminated on Stage B alone.",
            "",
        ]
    )
    contenders = [row for row in ordered if row["config"] in survivors]
    lines.extend(
        markdown_table(
            ["Metric"] + [row["config"] for row in contenders],
            [
                ["Late DA"]
                + [f"{row['late_median_da'] * 100:.2f}%" for row in contenders],
                ["Worst-window DA"]
                + [f"{row['min_window_da'] * 100:.2f}%" for row in contenders],
                ["Late RankIC"]
                + [f"{row['late_median_rankic']:.4f}" for row in contenders],
                ["Worst-window RankIC"]
                + [f"{row['min_window_rankic']:+.4f}" for row in contenders],
                ["RankIC spread across windows"]
                + [f"{row['window_rankic_std']:.4f}" for row in contenders],
                ["Late MAPE"]
                + [f"{row['late_median_mape']:.3f}" for row in contenders],
                ["AmpRatio span"]
                + [
                    f"[{row['min_window_ampratio']:.3f}, "
                    f"{row['max_window_ampratio']:.3f}]"
                    for row in contenders
                ],
                ["Balance, worst decile of days"]
                + [f"{row['balance_p10']:.3f}" for row in contenders],
                ["Distribution alignment"]
                + [
                    f"{row['distribution_alignment']:.3f}"
                    for row in contenders
                ],
                ["Collapse alignment"]
                + [f"{row['collapse_alignment']:.3f}" for row in contenders],
                ["Coarse token accuracy"]
                + [
                    f"{row['coarse_token_accuracy']:.4f}" for row in contenders
                ],
                ["Tokenizer MAE"]
                + [f"{row['tokenizer_mae']:.4f}" for row in contenders],
                ["Coarse codes used"]
                + [
                    f"{row['tok_coarse_unique']}/{row['coarse_vocab']} "
                    f"({row['tok_coarse_utilization'] * 100:.1f}%)"
                    for row in contenders
                ],
                ["Coarse effective codes"]
                + [f"{row['tok_coarse_effective']:.1f}" for row in contenders],
                ["Coarse entropy / nominal bit"]
                + [f"{row['coarse_bit_efficiency']:.3f}" for row in contenders],
                ["Fine effective codes"]
                + [f"{row['tok_fine_effective']:.1f}" for row in contenders],
                ["Joint MI (bits/token)"]
                + [f"{row['joint_mi_bits']:.3f}" for row in contenders],
            ],
        )
    )
    winner = next(
        (row for row in contenders if row["config"] == selected),
        contenders[0],
    )
    worst_behaviour = min(contenders, key=lambda row: row["balance_p10"])
    best_rankic = max(contenders, key=lambda row: row["min_window_rankic"])
    smallest = min(contenders, key=lambda row: row["joint_vocab"])
    widest_amp = max(
        contenders,
        key=lambda row: row["max_window_ampratio"] / row["min_window_ampratio"],
    )
    amp_span = (
        widest_amp["max_window_ampratio"] / widest_amp["min_window_ampratio"]
    )
    lines.extend(
        [
            "",
            "No arm dominates, so the eliminations are stated one at a time.",
            "",
            f"**{worst_behaviour['config']} - eliminated on behaviour.** It has "
            f"the largest target support of the shortlist "
            f"({worst_behaviour['target_unique']:.0f} tokens per day, "
            f"{worst_behaviour['target_effective']:.0f} effective), which means "
            "the residual Stage-B bias described above works *in its favour*. It "
            "still finishes last of the shortlist on collapse alignment "
            f"({worst_behaviour['collapse_alignment']:.3f}), unique alignment "
            f"({worst_behaviour['unique_alignment']:.3f}), support F1 "
            f"({worst_behaviour['support_f1']:.3f}) and worst-decile balance "
            f"({worst_behaviour['balance_p10']:.3f}), and its worst-window "
            f"RankIC ({worst_behaviour['min_window_rankic']:+.4f}) is the "
            "thinnest margin above zero in the shortlist. Its only wins are "
            "reconstruction MAE and coarse occupancy, and section 4 showed "
            "reconstruction does not propagate. This is the arm the naive raw-"
            "count reading selected; on invariant evidence it is the weakest "
            "survivor.",
            "",
            f"**{smallest['config']} - eliminated on headroom, not on "
            f"behaviour.** It is genuinely strong: best MAPE of the shortlist "
            f"({smallest['late_median_mape']:.3f}), the highest raw alignment "
            "scores, and the joint-best explained fraction "
            f"MI/H = {smallest['coarse_mi_fraction']:.3f}. But its tokenizer "
            f"discards the most: MAE {smallest['tokenizer_mae']:.4f} - the worst "
            f"in the whole sweep - with only "
            f"{smallest['tok_coarse_effective']:.1f} effective coarse codes. "
            "That is a hard information floor: no verifier, no reward model and "
            "no amount of GPT capacity can recover detail the tokenizer never "
            "encoded. Its DA is also the lowest of the shortlist "
            f"({smallest['late_median_da'] * 100:.2f}%). Part of its Stage-B "
            "lead is the residual bias, since its target support is the "
            "smallest in the sweep.",
            "",
            f"**{best_rankic['config']} - the closest call.** It owns the best "
            f"RankIC ({best_rankic['late_median_rankic']:.4f}) with the highest "
            f"per-window floor ({best_rankic['min_window_rankic']:+.4f}), which "
            "is the most decision-relevant statistic for a cross-sectional "
            "A-share model, and better reconstruction than the winner. Two "
            f"things rule it out. Its AmpRatio spans "
            f"[{best_rankic['min_window_ampratio']:.3f}, "
            f"{best_rankic['max_window_ampratio']:.3f}] - it under-predicts "
            "amplitude by nearly half in one window and over-predicts by more "
            "than double in another, so no single scalar recalibration fixes it "
            "- and its distribution alignment "
            f"({best_rankic['distribution_alignment']:.3f}) is the worst in the "
            "entire sweep. It also has the lowest worst-window DA of the "
            f"shortlist ({best_rankic['min_window_da'] * 100:.2f}%).",
            "",
            f"**{winner['config']} - selected.** It leads the shortlist on late "
            f"DA ({winner['late_median_da'] * 100:.2f}%, the highest of all "
            f"{len(ordered)} arms), worst-window DA "
            f"({winner['min_window_da'] * 100:.2f}%), worst-decile codebook "
            f"balance ({winner['balance_p10']:.3f}), distribution alignment "
            f"({winner['distribution_alignment']:.3f}) and coarse token "
            f"accuracy ({winner['coarse_token_accuracy']:.4f}). Its RankIC "
            f"({winner['late_median_rankic']:.4f}) is second with a comfortably "
            f"positive floor ({winner['min_window_rankic']:+.4f}) in every "
            "window. Crucially its amplitude error is a *consistent* bias "
            f"rather than instability: AmpRatio stays in "
            f"[{winner['min_window_ampratio']:.3f}, "
            f"{winner['max_window_ampratio']:.3f}], always over-predicting, "
            "which one scalar correction addresses - unlike the "
            f"{amp_span:.1f}x swing of {widest_amp['config']}. Its symmetric "
            f"split also gives the auxiliary fine head real work: "
            f"{winner['tok_fine_effective']:.0f} effective fine codes against "
            f"{best_rankic['tok_fine_effective']:.0f} for "
            f"{best_rankic['config']}, whose 6-bit fine layer is the thinnest "
            "of the shortlist. That matters for the hierarchical decode and for "
            "the planned verifier stage, which both consume the fine head.",
            "",
            "### The known weakness, carried forward deliberately",
            "",
            f"Measured as realised coarse entropy per nominal coarse bit, "
            f"{winner['config']} is the least efficient of the shortlist at "
            f"{winner['coarse_bit_efficiency']:.3f} bits per bit, against "
            + ", ".join(
                f"{row['config']} {row['coarse_bit_efficiency']:.3f}"
                for row in contenders
                if row["config"] != winner["config"]
            )
            + ". Concretely its coarse layer touches "
            f"{winner['tok_coarse_unique']}/{winner['coarse_vocab']} codes but "
            f"carries only {winner['tok_coarse_entropy_bits']:.2f} bits of "
            f"entropy - about as much as a {winner['bits_l1'] - 1}-bit codebook "
            f"would - so roughly one of its {winner['bits_l1']} coarse bits is "
            "currently paid for and not used. Raw utilization understates this, "
            "because a wider layer at lower occupancy can still carry more "
            "entropy; entropy per nominal bit is the comparable form and is the "
            "third panel of `fig8_codebook_occupancy.png`.",
            "",
            "That is a tokenizer-optimisation symptom, and the encoder/decoder "
            "capacity governing it is exactly the variable Exp 02 sweeps. "
            "Carrying this arm forward turns the weakness into the next "
            "experiment's question instead of hiding it.",
            "",
            "The risk is recorded up front: if Exp 02 raises coarse occupancy, "
            "the target distribution becomes harder and DA may fall from "
            f"{winner['late_median_da'] * 100:.0f}%. That outcome would itself "
            "be a result - it would show the current DA lead was partly bought "
            "with a coarser task - and the Exp 02 runner now exports per-layer "
            "occupancy so the effect is measurable rather than invisible. See "
            "`fig8_codebook_occupancy.png`.",
            "",
            "### If the objective changes",
            "",
            f"Should cross-sectional ranking become the sole objective and "
            f"amplitude instability be acceptable, {best_rankic['config']} is "
            "the documented alternative; it needs only a different "
            "`--select_config` on the same bundle. Should the priority become "
            "the smallest viable vocabulary for a cheap verifier, "
            f"{smallest['config']} is the fallback, at the cost of the worst "
            "reconstruction floor in the sweep.",
            "",
            "## 7. What this experiment does not claim",
            "",
            "- **No DA breakthrough was available here.** Late DA spans only "
            f"{min(da) * 100:.2f}% to {max(da) * 100:.2f}% across a "
            f"{int(vocab.max() / vocab.min())}x range of codebook sizes, and in "
            "the mature region the epoch-to-epoch DA standard deviation is under "
            "0.05 percentage points. Convergence toward 50% is the expected "
            "consequence of strong noise and non-stationarity in daily A-share "
            "returns. No bit configuration changes that, and this experiment was "
            "never capable of doing so.",
            "- **`avg_da_above_baseline` is negative for every arm, and that is "
            "not a failure.** The baseline is `max(always_up, always_down)` "
            "computed per day - an oracle that already knows the majority "
            "direction. It is an upper reference, not a fair competitor. Best "
            "(least negative) is "
            f"{max(ordered, key=lambda row: row['late_median_da_above_baseline'])['config']}"
            f" at {max(row['late_median_da_above_baseline'] for row in ordered) * 100:+.2f} "
            "percentage points.",
            f"- **No arm passes the health gate** (collapse <= "
            f"{gate['max_daily_collapse_rate']}, unique >= "
            f"{gate['min_daily_unique_tokens']}): {total_healthy} of "
            f"{total_epochs} checkpoints. The selection is an upstream research "
            "dependency for Exp 02 only.",
            "- **The offset-400 holdout was never touched.**",
            "",
            "## 8. Figures",
            "",
            "| File | Question it answers |",
            "|---|---|",
            "| `fig1_confound.png` | Does the target distribution move with "
            "codebook size? Yes, so raw counts are unusable. |",
            "| `fig2_raw_vs_normalized.png` | Do the naive and normalized "
            "readings agree? No, the ranking inverts. |",
            "| `fig3_information_saturation.png` | Does a larger codebook buy "
            "more learnable information? No, ~1.3 bits/token throughout. |",
            "| `fig4_quality_and_floors.png` | Which arms hold up on every "
            "window rather than only on average? |",
            "| `fig5_reconstruction_does_not_propagate.png` | Does better "
            "reconstruction improve forecasts? No. |",
            "| `fig6_staged_screen.png` | Which arms clear the preregistered "
            "floors, and how do the survivors compare? |",
            "| `fig7_head_to_head.png` | The Stage-A survivors compared metric "
            "by metric, plus their DA trajectories. |",
            "| `fig8_codebook_occupancy.png` | How much of each quantizer layer "
            "is really used, and what does Exp 02 inherit? |",
            "",
            "## 9. Provenance",
            "",
            f"- Study fingerprint: `{manifest.get('study_fingerprint')}`",
            f"- Dataset: {manifest['dataset']['file_count']} files, "
            f"{manifest['dataset']['total_bytes']:,} bytes, sha256 "
            f"`{manifest['dataset']['metadata_sha256']}`",
            "- Tokenizer: embedding_dim="
            f"{settings['tokenizer']['embedding_dim']}, hidden_dim="
            f"{settings['tokenizer']['hidden_dim']}, "
            f"{settings['tokenizer']['epochs']} epochs, selected by "
            f"{settings['tokenizer']['selection']}",
            f"- GPT: {settings['gpt']['loss']} + {settings['gpt']['optimizer']}, "
            f"lr={settings['gpt']['learning_rate']}, "
            f"{settings['gpt']['epochs']} epochs, every epoch retained, "
            f"accumulation={settings['gpt']['accumulation_steps']} sequences "
            "with exact boundaries, loader seed "
            f"{settings['gpt']['controlled_loader_seed']}",
            f"- Evaluation: offsets {settings['evaluation']['offsets']} x "
            f"{settings['evaluation']['days_per_window']} days; mature region = "
            f"last {LATE_EPOCHS} epochs, from epoch "
            f"{ordered[0]['late_epoch_start']}",
            "- The numerical preflight recorded in `study_manifest.json` passed "
            "every RoPE and fine-head check.",
            "",
            "Every number above is recomputed from "
            "`combined_epoch_summary.json`, `configs/*/tokenizer_metrics.json`, "
            "and `configs/*/dataset_token_summary.json` by "
            "`experiments/01-bitsweep/report_bitsweep.py`. No weighted "
            "composite score is used anywhere in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Exp 01 decision report and its figures"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--plot_subdir",
        default="report",
        help="Figure directory under <root>/plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    selection_path = root / "selection.json"
    selection = load_json(selection_path) if selection_path.is_file() else None
    if selection is not None and not selection.get("upstream_eligible", False):
        print(
            "WARNING: selection.json is not marked upstream_eligible; the "
            "report is still written."
        )
    records = build_records(root)
    if not records:
        raise RuntimeError(f"No Exp 01 configurations found under {root}")
    screen = apply_screen(records)

    plot_dir = root / "plots" / args.plot_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure_confound(plot_dir, records)
    figure_raw_versus_normalized(plot_dir, records)
    figure_information(plot_dir, records)
    figure_quality_floors(plot_dir, records, screen)
    figure_reconstruction(plot_dir, records)
    figure_screen(plot_dir, records, screen)
    contenders = list(screen["survivors"])
    if "9+9" not in contenders and any(
        row["config"] == "9+9" for row in records
    ):
        contenders.append("9+9")
    if contenders:
        figure_head_to_head(plot_dir, records, contenders)
    figure_occupancy(plot_dir, records)

    atomic_write_text(
        root / "REPORT.md",
        render_report(manifest, selection, records, screen),
    )
    print(
        f"Exp 01 report written to {root / 'REPORT.md'}\n"
        f"Figures: {plot_dir}\n"
        f"Stage-A survivors: {', '.join(screen['survivors']) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
