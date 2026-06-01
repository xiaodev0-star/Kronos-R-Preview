"""Generate comprehensive HPO result visualizations for Kronos-R-Preview."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

OUT_DIR = "HPO_Report"
COLORS = {
    "wave1": "#3498db",
    "wave2": "#e74c3c",
    "wave2b": "#f39c12",
    "wave3": "#9b59b6",
    "wave4": "#2ecc71",
    "best": "#e74c3c",
    "baseline": "#7f8c8d",
    "collapse_neg": "#e74c3c",
    "collapse_pos": "#27ae60",
    "collapse_zero": "#2c3e50",
}

WAVE_MAP = {
    "w1_": "wave1", "w2_": "wave2", "w2b_": "wave2b",
    "w3_": "wave3", "w4_": "wave4",
}


def get_wave(name):
    for prefix, wave in WAVE_MAP.items():
        if name.startswith(prefix):
            return wave
    return "wave1"


def load_results():
    with open("hpo_14h_results.json", "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("results", [])
    return data


def enrich(r):
    """Add computed fields."""
    r["amp_ratio"] = (r.get("pred_amplitude", 0) or 1) / max(r.get("acc_amplitude", 0) or 1, 1e-6)
    r["wave"] = get_wave(r["name"])
    r["collapse"] = r.get("collapse", 0) or 0
    return r


def plot_1_overview(results):
    """4-panel overview: MAPE, DA, Collapse, AmpRatio across all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Kronos-R-Preview HPO: Complete Experiment Overview (22 Runs, 13.1h)",
                 fontsize=14, weight="bold", y=0.98)

    names = [r["name"].replace("w1_", "").replace("w2_", "").replace("w2b_", "")
             .replace("w3_ft_", "ft_").replace("w4_", "") for r in results]
    x = np.arange(len(results))
    colors = [COLORS[r["wave"]] for r in results]

    # MAPE
    ax = axes[0, 0]
    mapes = [r.get("mape", 0) or 0 for r in results]
    bars = ax.bar(x, mapes, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(564.9, color=COLORS["baseline"], ls="--", lw=1.5, label="Baseline (564.9%)")
    best_idx = np.argmin(mapes)
    bars[best_idx].set_edgecolor(COLORS["best"])
    bars[best_idx].set_linewidth(2.5)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("10-Step MAPE (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.legend(fontsize=8)
    ax.annotate(f"Best: {mapes[best_idx]:.1f}%", xy=(best_idx, mapes[best_idx]),
                xytext=(best_idx + 1, mapes[best_idx] * 1.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["best"]),
                fontsize=9, color=COLORS["best"], weight="bold")

    # DA
    ax = axes[0, 1]
    das = [r.get("da", 0) or 0 for r in results]
    ax.bar(x, das, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random (50%)")
    ax.axhline(0.625, color=COLORS["baseline"], ls="--", lw=1.5, label="Baseline (62.5%)")
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Directional Accuracy (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 0.72)

    # Collapse
    ax = axes[1, 0]
    collapses = [r["collapse"] for r in results]
    bar_colors = [COLORS["collapse_pos"] if c > 0.05 else
                  COLORS["collapse_neg"] if c < -0.05 else
                  COLORS["collapse_zero"] for c in collapses]
    ax.bar(x, collapses, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", ls="-", lw=1.5)
    ax.axhline(-0.101, color=COLORS["baseline"], ls="--", lw=1, alpha=0.7, label="Baseline (-0.101)")
    ax.set_ylabel("Collapse (Pred - Acc)")
    ax.set_title("Zero-Collapse Metric (closer to 0 is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.legend(fontsize=8)
    red_patch = mpatches.Patch(color=COLORS["collapse_neg"], label="Under-predicting")
    green_patch = mpatches.Patch(color=COLORS["collapse_pos"], label="Over-predicting")
    ax.legend(handles=[red_patch, green_patch, plt.Line2D([0], [0], color=COLORS["baseline"],
              ls="--", label="Baseline")], fontsize=8)

    # AmpRatio
    ax = axes[1, 1]
    ars = [r["amp_ratio"] for r in results]
    ax.bar(x, ars, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", ls="-", lw=1.5, label="Perfect (1.0x)")
    ax.axhline(0.70, color=COLORS["baseline"], ls="--", lw=1, label="Baseline (0.70x)")
    ax.set_ylabel("Amplitude Ratio (Pred/Acc)")
    ax.set_title("Amplitude Calibration (closer to 1.0x is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 4.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT_DIR}/01_overview_4panel.png")
    plt.close()
    print("  Saved 01_overview_4panel.png")


def plot_2_mape_vs_collapse(results):
    """Scatter: MAPE vs Collapse with AmpRatio as bubble size."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("MAPE vs Collapse Trade-off (bubble size = |AmpRatio - 1|)",
                 fontsize=13, weight="bold")

    for r in results:
        c = COLORS[r["wave"]]
        mape = r.get("mape", 0) or 0
        collapse = r["collapse"]
        ar_err = abs(r["amp_ratio"] - 1.0)
        size = max(ar_err * 200, 30)
        alpha = 0.8 if ar_err < 0.5 else 0.5
        ax.scatter(mape, collapse, s=size, c=c, alpha=alpha, edgecolors="white", linewidth=0.8)
        # Label top candidates
        if mape < 580 or abs(collapse) < 0.04:
            ax.annotate(r["name"].replace("w1_", "").replace("w2_", "").replace("w2b_", "")
                       .replace("w3_ft_", "ft_").replace("w4_", ""),
                       (mape, collapse), fontsize=7, ha="left",
                       xytext=(5, 3), textcoords="offset points")

    ax.axhline(0, color="black", ls="-", lw=1)
    ax.axvline(564.9, color=COLORS["baseline"], ls="--", lw=1, alpha=0.5, label="Baseline MAPE")
    ax.set_xlabel("MAPE (%) — lower is better")
    ax.set_ylabel("Collapse (Pred - Acc) — closer to 0 is better")
    ax.set_xscale("log")

    # Legend for waves
    patches = [mpatches.Patch(color=COLORS[w], label=w.replace("wave", "Wave "))
               for w in ["wave1", "wave2", "wave2b", "wave3", "wave4"]]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_mape_vs_collapse_scatter.png")
    plt.close()
    print("  Saved 02_mape_vs_collapse_scatter.png")


def plot_3_loss_comparison(results):
    """Grouped bar chart comparing loss functions (wave 2 only)."""
    w2 = [r for r in results if r["wave"] in ("wave2", "wave2b")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Loss Function Comparison (Wave 2 + 2b)", fontsize=13, weight="bold")

    names = [r["name"].replace("w2_", "").replace("w2b_", "b_") for r in w2]
    x = np.arange(len(w2))

    # MAPE
    ax = axes[0]
    mapes = [r.get("mape", 0) or 0 for r in w2]
    colors = [COLORS["wave2"] if r["wave"] == "wave2" else COLORS["wave2b"] for r in w2]
    ax.barh(x, mapes, color=colors, edgecolor="white")
    ax.axvline(564.9, color=COLORS["baseline"], ls="--", lw=1.5, label="Baseline")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("MAPE")
    ax.legend(fontsize=8)

    # Collapse
    ax = axes[1]
    collapses = [r["collapse"] for r in w2]
    bar_colors = [COLORS["collapse_pos"] if c > 0.05 else
                  COLORS["collapse_neg"] if c < -0.05 else
                  COLORS["collapse_zero"] for c in collapses]
    ax.barh(x, collapses, color=bar_colors, edgecolor="white")
    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.axvline(-0.101, color=COLORS["baseline"], ls="--", lw=1, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Collapse (Pred - Acc)")
    ax.set_title("Collapse Metric")

    # AmpRatio
    ax = axes[2]
    ars = [r["amp_ratio"] for r in w2]
    ax.barh(x, ars, color=colors, edgecolor="white")
    ax.axvline(1.0, color="black", ls="-", lw=1.5, label="Perfect")
    ax.axvline(0.70, color=COLORS["baseline"], ls="--", lw=1, alpha=0.7, label="Baseline")
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Amplitude Ratio")
    ax.set_title("AmpRatio")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT_DIR}/03_loss_function_comparison.png")
    plt.close()
    print("  Saved 03_loss_function_comparison.png")


def plot_4_reasoning_module(results):
    """Comparison of reasoning module vs baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Reasoning Module (Wave 4) vs Baseline", fontsize=13, weight="bold")

    # Select key experiments for comparison
    compare = [r for r in results if r["name"] in [
        "w1_baseline_10ep", "w4_reason_frozen", "w4_reason_trainable",
        "w2_focal_g3", "w2_entropy_reg_a04"
    ]]
    names = [r["name"].replace("w1_", "").replace("w2_", "").replace("w4_", "") for r in compare]
    x = np.arange(len(compare))

    # MAPE + DA dual axis
    ax = axes[0]
    mapes = [r.get("mape", 0) or 0 for r in compare]
    das = [r.get("da", 0) or 0 for r in compare]
    colors = [COLORS[r["wave"]] for r in compare]

    bars = ax.bar(x - 0.2, mapes, 0.4, color=colors, alpha=0.8, label="MAPE (%)")
    ax2 = ax.twinx()
    ax2.plot(x, das, "ko-", markersize=8, linewidth=2, label="DA")
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax2.axhline(0.625, color=COLORS["baseline"], ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAPE (%)", color=COLORS["wave1"])
    ax2.set_ylabel("Directional Accuracy")
    ax.set_title("MAPE and DA")
    ax2.set_ylim(0.5, 0.70)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Token diversity + Collapse
    ax = axes[1]
    tokens = [r.get("unique_pred_tokens", 0) for r in compare]
    collapses = [r["collapse"] for r in compare]

    bars_tok = ax.bar(x - 0.2, tokens, 0.4, color=colors, alpha=0.8, label="Unique Tokens")
    ax3 = ax.twinx()
    ax3.plot(x, collapses, "rs-", markersize=8, linewidth=2, label="Collapse")
    ax3.axhline(0, color="black", ls="-", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Unique Predicted Tokens")
    ax3.set_ylabel("Collapse (Pred - Acc)")
    ax.set_title("Token Diversity and Collapse")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT_DIR}/04_reasoning_module_comparison.png")
    plt.close()
    print("  Saved 04_reasoning_module_comparison.png")


def plot_5_amplitude_heatmap(results):
    """Heatmap of key metrics across experiments."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle("Experiment Metrics Heatmap", fontsize=13, weight="bold")

    # Select metrics
    metric_names = ["MAPE", "DA", "Collapse", "AmpRatio", "Tokens"]
    names = [r["name"].replace("w1_", "").replace("w2_", "").replace("w2b_", "")
             .replace("w3_ft_", "ft_").replace("w4_", "") for r in results]

    # Build matrix (normalized for heatmap)
    matrix = np.zeros((len(results), len(metric_names)))
    for i, r in enumerate(results):
        matrix[i, 0] = r.get("mape", 0) or 0
        matrix[i, 1] = r.get("da", 0) or 0
        matrix[i, 2] = r["collapse"]
        matrix[i, 3] = r["amp_ratio"]
        matrix[i, 4] = r.get("unique_pred_tokens", 0)

    # Normalize each column for visualization
    norm_matrix = np.zeros_like(matrix)
    for j in range(len(metric_names)):
        col = matrix[:, j]
        if j == 0:  # MAPE: lower is better → invert
            norm_matrix[:, j] = 1 - (col - col.min()) / max(col.max() - col.min(), 1e-6)
        elif j == 2:  # Collapse: closer to 0 is better
            norm_matrix[:, j] = 1 - np.abs(col) / max(np.abs(col).max(), 1e-6)
        else:  # Higher is better
            norm_matrix[:, j] = (col - col.min()) / max(col.max() - col.min(), 1e-6)

    im = ax.imshow(norm_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # Annotate with actual values
    for i in range(len(results)):
        for j in range(len(metric_names)):
            val = matrix[i, j]
            if j == 0:
                text = f"{val:.0f}%"
            elif j in (1, 3):
                text = f"{val:.3f}"
            elif j == 2:
                text = f"{val:+.3f}"
            else:
                text = f"{int(val)}"
            color = "white" if norm_matrix[i, j] < 0.3 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Score (green=good, red=bad)", shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT_DIR}/05_metrics_heatmap.png")
    plt.close()
    print("  Saved 05_metrics_heatmap.png")


def plot_6_pareto_front(results):
    """Pareto front: MAPE vs |Collapse|."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Pareto Front: MAPE vs Collapse Deviation", fontsize=13, weight="bold")

    valid = [r for r in results if not r.get("zero_collapse")]
    mapes = [r.get("mape", 0) or 0 for r in valid]
    collapse_dev = [abs(r["collapse"]) for r in valid]

    # Find Pareto front
    pareto = []
    for i, r in enumerate(valid):
        dominated = False
        for j, r2 in enumerate(valid):
            if (r2.get("mape", 0) or 0) <= r.get("mape", 0) or 0 and abs(r2["collapse"]) <= abs(r["collapse"]) and i != j:
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    # Plot all points
    for r in valid:
        c = COLORS[r["wave"]]
        ax.scatter(r.get("mape", 0) or 0, abs(r["collapse"]), s=100, c=c,
                   alpha=0.6, edgecolors="white", linewidth=0.8)
        if r in pareto or r.get("mape", 0) or 0 < 560:
            ax.annotate(r["name"].replace("w1_", "").replace("w2_", "").replace("w2b_", "")
                       .replace("w3_ft_", "ft_").replace("w4_", ""),
                       (r.get("mape", 0) or 0, abs(r["collapse"])),
                       fontsize=7.5, ha="left", xytext=(5, 3), textcoords="offset points")

    # Draw Pareto front line
    pareto_sorted = sorted(pareto, key=lambda r: r.get("mape", 0) or 0)
    if len(pareto_sorted) > 1:
        px = [r.get("mape", 0) or 0 for r in pareto_sorted]
        py = [abs(r["collapse"]) for r in pareto_sorted]
        ax.plot(px, py, "k--", alpha=0.5, lw=1.5, label="Pareto Front")

    # Baseline reference
    ax.scatter(564.9, 0.101, s=200, c=COLORS["baseline"], marker="*", zorder=5, label="Baseline")

    ax.set_xlabel("MAPE (%) — lower is better")
    ax.set_ylabel("|Collapse| — closer to 0 is better")
    patches = [mpatches.Patch(color=COLORS[w], label=w.replace("wave", "Wave "))
               for w in ["wave1", "wave2", "wave2b", "wave3", "wave4"]]
    ax.legend(handles=patches + [plt.Line2D([0], [0], marker="*", color=COLORS["baseline"],
              ls="", markersize=12, label="Baseline")], fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_pareto_front.png")
    plt.close()
    print("  Saved 06_pareto_front.png")


def plot_7_wave_summary(results):
    """Per-wave summary boxplot."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Per-Wave Metric Distributions", fontsize=13, weight="bold")

    waves = ["wave1", "wave2", "wave2b", "wave3", "wave4"]
    wave_labels = ["Wave 1\n(HPO)", "Wave 2\n(Loss)", "Wave 2b\n(Combined)", "Wave 3\n(Fine-tune)", "Wave 4\n(Reasoning)"]

    for ax_idx, (metric, title, ylabel) in enumerate([
        ("mape", "MAPE", "MAPE (%)"),
        ("da", "Directional Accuracy", "DA"),
        ("collapse", "Collapse Metric", "Collapse"),
        ("amp_ratio", "Amplitude Ratio", "AmpRatio"),
    ]):
        ax = axes[ax_idx]
        data_by_wave = []
        for w in waves:
            if metric == "mape":
                vals = [r.get("mape", 0) or 0 for r in results if r["wave"] == w]
            elif metric == "da":
                vals = [r.get("da", 0) or 0 for r in results if r["wave"] == w]
            elif metric == "collapse":
                vals = [r["collapse"] for r in results if r["wave"] == w]
            else:
                vals = [r["amp_ratio"] for r in results if r["wave"] == w]
            data_by_wave.append(vals)

        bp = ax.boxplot(data_by_wave, patch_artist=True, labels=wave_labels, widths=0.6)
        for patch, w in zip(bp["boxes"], waves):
            patch.set_facecolor(COLORS[w])
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        if metric == "collapse":
            ax.axhline(0, color="black", ls="-", lw=1)
        if metric == "amp_ratio":
            ax.axhline(1.0, color="black", ls="-", lw=1)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{OUT_DIR}/07_wave_summary_boxplot.png")
    plt.close()
    print("  Saved 07_wave_summary_boxplot.png")


def plot_8_top5_radar(results):
    """Radar chart for top 5 experiments."""
    top5_names = ["w4_reason_frozen", "w1_wd0001", "w2_focal_g3",
                  "w2_entropy_reg_a04", "w1_baseline_10ep"]
    top5 = [r for r in results if r["name"] in top5_names]
    top5.sort(key=lambda r: top5_names.index(r["name"]))

    metrics = ["MAPE\n(inv)", "DA", "Collapse\n(1-|C|)", "AmpRatio\n(1-|AR-1|)", "Tokens"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Top 5 Experiments: Radar Comparison", fontsize=13, weight="bold", y=1.02)

    colors_top = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#7f8c8d"]

    for r, color in zip(top5, colors_top):
        mape = r.get("mape", 0) or 0
        da = r.get("da", 0) or 0
        collapse = r["collapse"]
        ar = r["amp_ratio"]
        tokens = r.get("unique_pred_tokens", 0)

        # Normalize to [0, 1] (higher = better)
        mape_norm = 1 - min(mape / 1500, 1)  # lower MAPE = better
        da_norm = da
        collapse_norm = 1 - min(abs(collapse) / 0.15, 1)  # closer to 0 = better
        ar_norm = 1 - min(abs(ar - 1) / 1.5, 1)  # closer to 1 = better
        tokens_norm = min(tokens / 50, 1)

        values = [mape_norm, da_norm, collapse_norm, ar_norm, tokens_norm]
        values += values[:1]

        label = r["name"].replace("w1_", "").replace("w2_", "").replace("w4_", "")
        ax.plot(angles, values, "o-", color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_top5_radar.png", bbox_inches="tight")
    plt.close()
    print("  Saved 08_top5_radar.png")


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading results...")
    results = load_results()
    results = [enrich(r) for r in results]
    print(f"  {len(results)} experiments loaded")

    print("\nGenerating charts...")
    plot_1_overview(results)
    plot_2_mape_vs_collapse(results)
    plot_3_loss_comparison(results)
    plot_4_reasoning_module(results)
    plot_5_amplitude_heatmap(results)
    plot_6_pareto_front(results)
    plot_7_wave_summary(results)
    plot_8_top5_radar(results)

    print(f"\nAll 8 charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
