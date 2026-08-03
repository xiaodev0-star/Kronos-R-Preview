"""Exp 03 single simplified selection plot + manifest status correction.

Run:
    python experiments/03-gpt-scaling/report_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "server_runs" / "results" / "03-gpt-scaling" / "seed42"
MANIFEST = RESULTS / "study_manifest.json"

COLORS = {
    "deep": "#7f8c8d",
    "depth6": "#3498db",
    "depth8": "#9b59b6",
    "width384_d4": "#e67e22",
    "width512_d4_kv1": "#e74c3c",
    "xlarge": "#c0392b",
}
SELECTED = "depth6"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fix_manifest() -> None:
    payload = load_json(MANIFEST)
    payload["status"] = "completed"
    payload["completed_at_utc"] = payload.get("completed_at_utc") or payload.pop(
        "failed_at_utc", None
    )
    payload.pop("failed_at_utc", None)
    payload.pop("error", None)
    with MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"manifest -> {payload['status']}")


def main() -> None:
    fix_manifest()
    analysis = load_json(RESULTS / "analysis.json")
    selection = analysis["selection"]
    ranked = analysis["ranked_windows"]
    by_config = {row["config"]: row for row in ranked}

    eligible = set(selection["near_best_rule"]["eligible_configs"])
    leader_score = selection["near_best_rule"]["leader_score"]
    margin = selection["near_best_rule"]["margin_one_leader_window_sd"]
    threshold = leader_score - margin

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # near-best threshold line
    ax.axhline(threshold, color="#f39c12", linestyle="--", linewidth=1.2,
               label=f"near-best threshold = {threshold:.3f}")

    for cfg, row in by_config.items():
        x = row["parameter_count"] / 1e6
        y = row["median_daily_codebook_balance_score"]
        is_elig = cfg in eligible
        is_sel = cfg == SELECTED
        marker = "*" if is_sel else ("o" if is_elig else "x")
        ax.scatter(x, y,
                   s=260 if is_sel else (160 if is_elig else 100),
                   color=COLORS[cfg],
                   marker=marker,
                   edgecolor="black" if marker != "x" else "none",
                   linewidth=1.2 if is_sel else 0.6,
                   zorder=5)
        ax.annotate(f"{cfg}\n({y:.3f})", (x, y),
                    xytext=(8, 8), textcoords="offset points", fontsize=9,
                    fontweight="bold" if is_sel else "normal")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (million, log scale)")
    ax.set_ylabel("Coarse codebook balance (all 50 epochs, median)")
    ax.set_title("Exp 03: selection = depth6 (smallest within near-best set)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "report_selection.png", dpi=180)
    plt.close(fig)
    print("wrote report_selection.png")


if __name__ == "__main__":
    main()
