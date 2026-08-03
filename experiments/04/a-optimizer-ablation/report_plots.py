"""Exp 04-A single simplified selection plot.

Left panel: token-quality metrics (bar comparison, muon vs adamw).
Right panel: paired moving-block bootstrap differences with 95% CIs.

Run:
    python experiments/04/a-optimizer-ablation/report_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "server_runs" / "results" / "04a-optimizer-ablation" / "seed42"

COLORS = {"adamw": "#7f8c8d", "muon": "#3498db"}
SELECTED = "muon"


def main() -> None:
    analysis = json.loads((RESULTS / "analysis.json").read_text(encoding="utf-8"))
    envelopes = {row["arm"]: row for row in analysis["arm_envelopes"]}
    paired = analysis["paired_epoch_diagnostics"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- Left: token-quality bar comparison ---
    metrics = [
        ("mature_median_codebook_balance", "CB balance", 1.0),
        ("mature_median_token_support_f1", "Support F1", 1.0),
        ("mature_median_effective_token_alignment", "Eff. align", 1.0),
        ("mature_median_fine_codebook_balance", "Fine balance", 1.0),
        ("mature_median_fine_n_unique_tokens", "Fine unique", 1.0),
        ("mature_median_joint_n_unique_tokens", "Joint unique", 1.0),
    ]
    labels = [m[1] for m in metrics]
    x_pos = range(len(metrics))
    width = 0.36
    for offset, arm in enumerate(("adamw", "muon")):
        env = envelopes[arm]
        values = [
            (env.get(field) or 0.0) * scale
            for field, _, scale in metrics
        ]
        bars = left.bar(
            [x + (offset - 0.5) * width for x in x_pos],
            values,
            width=width,
            color=COLORS[arm],
            label=arm + (" (selected)" if arm == SELECTED else ""),
            edgecolor="black" if arm == SELECTED else "none",
            linewidth=1.0 if arm == SELECTED else 0.0,
        )
        for bar, value in zip(bars, values):
            left.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if value < 5 else f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if arm == SELECTED else "normal",
            )
    left.set_xticks(list(x_pos))
    left.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    left.set_ylabel("Score / count")
    left.set_title("Token-quality envelopes (all 50 epochs, median)")
    left.grid(alpha=0.25, axis="y")
    left.legend(fontsize=9)

    # --- Right: paired bootstrap differences ---
    # Favorable direction normalized: positive = muon better
    favorable_higher = {
        "daily_codebook_balance",
        "daily_target_support_recall",
        "daily_effective_tokens",
        "daily_unique",
        "daily_rank_ic",
        "daily_da",
    }
    ordered = [
        "daily_codebook_balance",
        "daily_target_support_recall",
        "daily_token_jsd",
        "daily_effective_tokens",
        "daily_unique",
        "daily_rank_ic",
        "daily_mape",
        "daily_ampratio_log_error",
    ]
    items = {item["metric"]: item for item in paired}
    y_pos = list(range(len(ordered)))
    for i, metric in enumerate(ordered):
        item = items.get(metric)
        if item is None:
            continue
        delta = item["selected_minus_comparator"]
        ci = item["moving_block_bootstrap_95_ci"]
        # Normalize so positive = muon better
        if metric not in favorable_higher:
            delta = -delta
            ci = [-ci[1], -ci[0]]
        color = "#27ae60" if (ci[0] > 0 or ci[1] < 0) else "#95a5a6"
        right.errorbar(
            delta,
            i,
            xerr=[[delta - ci[0]], [ci[1] - delta]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            linewidth=1.5,
        )
    right.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
    right.set_yticks(list(y_pos))
    right.set_yticklabels(
        [m.replace("daily_", "").replace("_", " ") for m in ordered],
        fontsize=9,
    )
    right.invert_yaxis()
    right.set_xlabel("Muon - AdamW (positive = Muon better)")
    right.set_title("Paired bootstrap (10000 reps, 5-day block)")
    right.grid(alpha=0.25, axis="x")

    figure.suptitle(
        "Exp 04-A: selection = muon (token-balance dominant, guardrails pass)",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout()
    out = RESULTS / "report_selection.png"
    figure.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
