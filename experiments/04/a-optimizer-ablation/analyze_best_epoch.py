"""Exp 04-A per-epoch token-quality scan + best-epoch comparison.

For each arm, scan all 50 epochs and rank them by a token-quality composite
key (mirroring Exp 03's primary_key). Then compare each arm's best epoch
head-to-head. Writes:
  - best_epoch_scan.json  (full per-epoch ranking per arm)
  - best_epoch_comparison.png  (left: trajectories with best markers;
    right: best-epoch metric bars)

Run:
    python experiments/04/a-optimizer-ablation/analyze_best_epoch.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "server_runs" / "results" / "04a-optimizer-ablation" / "seed42"
COLORS = {"adamw": "#7f8c8d", "muon": "#3498db"}


def fnum(row: dict, key: str, default: float) -> float:
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def token_quality_key(row: dict) -> tuple[float, ...]:
    """Token-quality composite ranking key (higher = better).

    Mirrors Exp 03 primary_key. Lower-is-better fields are negated.
    Perplexity proxy = pred_token_entropy_bits (higher = codebook used more
    evenly, closer to target entropy ~4.84 bits, i.e. less collapsed).
    """
    return (
        fnum(row, "median_daily_codebook_balance_score", -math.inf),
        fnum(row, "p10_daily_codebook_balance_score", -math.inf),
        fnum(row, "median_daily_token_support_f1", -math.inf),
        -fnum(row, "median_daily_token_jsd", math.inf),
        fnum(row, "median_daily_effective_token_alignment", -math.inf),
        -fnum(row, "median_daily_collapse_rate", math.inf),
        fnum(row, "median_daily_unique_tokens", -math.inf),
        fnum(row, "median_daily_pred_token_entropy_bits", -math.inf),
    )


DISPLAY_METRICS = [
    ("median_daily_codebook_balance_score", "CB balance", 1.0, True),
    ("median_daily_token_support_f1", "Support F1", 1.0, True),
    ("median_daily_token_jsd", "Token JSD", 1.0, False),
    ("median_daily_effective_token_alignment", "Eff. align", 1.0, True),
    ("median_daily_collapse_rate", "Collapse", 1.0, False),
    ("median_daily_unique_tokens", "Unique", 1.0, True),
    ("median_daily_pred_token_entropy_bits", "Pred entropy (bits)", 1.0, True),
    ("median_daily_fine_codebook_balance_score", "Fine balance", 1.0, True),
    ("median_daily_joint_codebook_balance_score", "Joint balance", 1.0, True),
]


def main() -> None:
    rows = json.loads(
        (RESULTS / "combined_epoch_summary.json").read_text(encoding="utf-8")
    )
    arms = list(dict.fromkeys(str(r["arm"]) for r in rows))

    # Per-arm per-epoch ranking
    scan: dict[str, list[dict]] = {}
    best: dict[str, dict] = {}
    for arm in arms:
        group = sorted(
            [r for r in rows if r["arm"] == arm],
            key=lambda r: int(r["epoch"]),
        )
        ranked = sorted(group, key=token_quality_key, reverse=True)
        scan[arm] = [
            {
                "epoch": int(r["epoch"]),
                "rank": i + 1,
                **{
                    m[0]: r.get(m[0])
                    for m in DISPLAY_METRICS
                },
                "composite_key": list(token_quality_key(r)),
            }
            for i, r in enumerate(ranked)
        ]
        best[arm] = ranked[0]

    # Write scan json
    out_json = {
        "arm_best_epoch": {arm: int(best[arm]["epoch"]) for arm in arms},
        "scan": scan,
    }
    (RESULTS / "best_epoch_scan.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # Print summary
    print("=== Best epoch per arm (token-quality composite) ===")
    for arm in arms:
        b = best[arm]
        print(
            f"[{arm}] best epoch = {b['epoch']}  "
            f"CB={b.get('median_daily_codebook_balance_score'):.4f}  "
            f"F1={b.get('median_daily_token_support_f1'):.4f}  "
            f"JSD={b.get('median_daily_token_jsd'):.4f}  "
            f"entropy={b.get('median_daily_pred_token_entropy_bits'):.3f}  "
            f"unique={b.get('median_daily_unique_tokens')}"
        )

    # === Plot ===
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: trajectories of CB balance + entropy, mark best epoch
    ax1 = left
    ax2 = ax1.twinx()
    for arm in arms:
        group = sorted(
            [r for r in rows if r["arm"] == arm],
            key=lambda r: int(r["epoch"]),
        )
        epochs = [int(r["epoch"]) for r in group]
        cb = [r.get("median_daily_codebook_balance_score", 0) for r in group]
        ent = [r.get("median_daily_pred_token_entropy_bits", 0) for r in group]
        ax1.plot(epochs, cb, color=COLORS[arm], linewidth=1.8, label=f"{arm} CB balance")
        ax2.plot(
            epochs, ent, color=COLORS[arm], linewidth=1.4, linestyle="--",
            label=f"{arm} pred entropy",
        )
        be = int(best[arm]["epoch"])
        bcb = best[arm].get("median_daily_codebook_balance_score", 0)
        ax1.scatter([be], [bcb], color=COLORS[arm], s=180, marker="*",
                    edgecolor="black", linewidth=1.0, zorder=5)
        ax1.annotate(f"best ep{be}", (be, bcb), xytext=(8, -14),
                     textcoords="offset points", fontsize=9, fontweight="bold",
                     color=COLORS[arm])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Coarse codebook balance (median)")
    ax2.set_ylabel("Pred token entropy (bits)", color="#555")
    ax1.set_title("Token quality over epochs (★ = best)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    # Right: best-epoch metric comparison (normalized to adamw=1.0)
    labels = [m[1] for m in DISPLAY_METRICS]
    x_pos = range(len(DISPLAY_METRICS))
    width = 0.36
    adamw_best = best["adamw"]
    muon_best = best["muon"]
    for offset, arm in enumerate(("adamw", "muon")):
        b = best[arm]
        # Normalize: for higher-better, ratio vs adamw; for lower-better, inverse ratio
        vals = []
        for field, _, _, higher_better in DISPLAY_METRICS:
            v_arm = b.get(field) or 0.0
            v_ref = adamw_best.get(field) or 0.0
            if v_ref == 0:
                vals.append(0.0)
            elif higher_better:
                vals.append(v_arm / v_ref)
            else:
                vals.append(v_ref / v_arm if v_arm != 0 else 0.0)
        bars = right.bar(
            [x + (offset - 0.5) * width for x in x_pos],
            vals,
            width=width,
            color=COLORS[arm],
            label=f"{arm} (ep{int(b['epoch'])})",
            edgecolor="black" if arm == "muon" else "none",
            linewidth=1.0 if arm == "muon" else 0.0,
        )
        for bar, val in zip(bars, vals):
            right.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold" if arm == "muon" else "normal",
            )
    right.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
    right.set_xticks(list(x_pos))
    right.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    right.set_ylabel("Ratio vs AdamW best (1.0 = tie)")
    right.set_title("Best-epoch comparison (normalized, >1 = Muon better)")
    right.grid(alpha=0.25, axis="y")
    right.legend(fontsize=9)

    fig.suptitle(
        "Exp 04-A: per-epoch token-quality scan + best-epoch comparison",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_png = RESULTS / "best_epoch_comparison.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
