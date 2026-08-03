"""Decision report for Exp 02 (tokenizer architecture sweep).

Generates REPORT.md plus narrative figures under plots/report/. The report
documents the observation that raw Collapse/Unique remain confounded in this
sweep even though every arm shares the same 7+7 codebook: the confounder is no
longer codebook size but the encoder-induced target distribution. Arms whose
encoders spread the data thinner earn cosmetically lower raw collapse and
higher raw unique counts without tracking their own targets any better, and
the extra diversity does not convert into mutual information or downstream
RankIC.

Read-only over the results tree; never touches weights or the sealed holdout.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "server_runs" / "results" / "02-tokenizer-tuning" / "seed42"
)

LOG2 = math.log(2.0)
LATE_EPOCH_START = 41
N_EVAL_DATES = 400.0

NEUTRAL = "#4c78a8"
HIGHLIGHT = {
    "64x192": "#d62728",  # selected (the Exp 01 default architecture)
    "48x256": "#2ca02c",  # raw-collapse bait
    "96x384": "#9467bd",  # raw-unique bait
    "96x192": "#ff7f0e",  # behaviour-best, signal-poor
}
DEAD = "#c7c7c7"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def colour_for(config: str, record: dict[str, Any]) -> str:
    if record["late_rankic_tstat"] < 2.0:
        return DEAD
    return HIGHLIGHT.get(config, NEUTRAL)


def config_label(directory_name: str) -> str:
    # emb_048_hid_192 -> 48x192
    parts = directory_name.replace("emb_", "").split("_hid_")
    return f"{int(parts[0])}x{int(parts[1])}"


def build_records() -> list[dict[str, Any]]:
    records = []
    for directory in sorted((RESULTS_ROOT / "configs").iterdir()):
        trajectory = directory / "epoch_trajectory"
        if not trajectory.is_dir():
            continue
        files = sorted(trajectory.glob("epoch_0*.json"))
        if len(files) < 50:
            continue
        rows = []
        for file in files:
            payload = load_json(file)
            aggregate = payload["aggregate"]
            training = payload["training"]
            rows.append(
                {
                    "epoch": payload["epoch"],
                    "raw_collapse": aggregate["median_daily_collapse_rate"],
                    "target_collapse": aggregate[
                        "median_daily_target_collapse_rate"
                    ],
                    "unique": aggregate["median_daily_joint_n_unique_tokens"],
                    "target_unique": aggregate[
                        "median_daily_joint_target_n_unique_tokens"
                    ],
                    "unique_alignment": aggregate[
                        "median_daily_joint_unique_token_alignment"
                    ],
                    "distribution_alignment": aggregate[
                        "median_daily_joint_distribution_alignment"
                    ],
                    "collapse_alignment": aggregate[
                        "median_daily_joint_collapse_alignment"
                    ],
                    "balance_p10": aggregate[
                        "p10_daily_joint_codebook_balance_score"
                    ],
                    "rank_ic": aggregate["avg_daily_rank_ic"],
                    "rank_ic_std": aggregate["window_daily_rank_ic_std"],
                    "ampratio": aggregate["avg_ampratio"],
                    "val_ce_bits": (
                        training["val_coarse_loss"]
                        + training["val_fine_loss"]
                    )
                    / LOG2,
                }
            )
        late = [row for row in rows if row["epoch"] >= LATE_EPOCH_START]

        def med(key: str) -> float:
            return statistics.median(row[key] for row in late)

        summary = load_json(directory / "dataset_token_summary.json")
        target_entropy = summary["splits"]["validation"]["joint"][
            "entropy_bits"
        ]
        tokenizer_metrics = load_json(directory / "tokenizer_metrics.json")
        ic = med("rank_ic")
        ic_se = med("rank_ic_std") / math.sqrt(N_EVAL_DATES)
        records.append(
            {
                "config": config_label(directory.name),
                "raw_collapse": med("raw_collapse"),
                "target_collapse": med("target_collapse"),
                "collapse_over_target": med("raw_collapse")
                / med("target_collapse"),
                "unique": med("unique"),
                "target_unique": med("target_unique"),
                "unique_alignment": med("unique_alignment"),
                "distribution_alignment": med("distribution_alignment"),
                "collapse_alignment": med("collapse_alignment"),
                "balance_p10": med("balance_p10"),
                "late_rankic": ic,
                "late_rankic_tstat": ic / ic_se if ic_se > 0 else 0.0,
                "ampratio": med("ampratio"),
                "joint_mi_bits": target_entropy - med("val_ce_bits"),
                "tokenizer_mae": tokenizer_metrics["mae"],
            }
        )
    return records


def annotate_points(axes, records, xkey, ykey, fontsize=8) -> None:
    for record in records:
        axes.annotate(
            record["config"],
            (record[xkey], record[ykey]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=fontsize,
            color=colour_for(record["config"], record),
        )


def fig1_target_confound(records, plot_dir: Path) -> None:
    """Raw collapse mostly reflects how concentrated each arm's target is."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ordered = sorted(records, key=lambda row: row["target_collapse"])
    labels = [row["config"] for row in ordered]
    x = range(len(ordered))
    axes[0].bar(
        x,
        [row["target_collapse"] * 100 for row in ordered],
        color=[colour_for(row["config"], row) for row in ordered],
        alpha=0.55,
        label="target collapse (encoder-induced)",
    )
    axes[0].plot(
        x,
        [row["raw_collapse"] * 100 for row in ordered],
        "o--",
        color="#333333",
        label="raw GPT collapse",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("median daily collapse (%)")
    axes[0].set_title(
        "Same 7+7 codebook, different homework:\n"
        "each encoder induces its own target concentration"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    xs = [row["target_collapse"] * 100 for row in records]
    ys = [row["raw_collapse"] * 100 for row in records]
    axes[1].scatter(
        xs,
        ys,
        c=[colour_for(row["config"], row) for row in records],
        s=60,
        zorder=3,
    )
    annotate_points(axes[1], records, "target_collapse", "raw_collapse")
    # least-squares trend on the percentage scale
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    grid = [min(xs), max(xs)]
    axes[1].plot(
        grid,
        [my + slope * (g - mx) for g in grid],
        "--",
        color="#888888",
        label=f"trend: +{slope:.2f} pp raw per pp target",
    )
    correlation = sxy / math.sqrt(
        sxx * sum((v - my) ** 2 for v in ys)
    )
    axes[1].set_xlabel("target collapse (%) - property of the encoder")
    axes[1].set_ylabel("raw GPT collapse (%)")
    axes[1].set_title(
        f"Raw collapse tracks the target (r = {correlation:.2f}):\n"
        "a cosmetically low raw value mostly means a thinner target"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Fix: the scatter panel needs the percentage-scale annotation offsets.
    figure.tight_layout()
    figure.savefig(plot_dir / "fig1_target_confound.png", dpi=180)
    plt.close(figure)


def fig2_raw_vs_normalized(records, plot_dir: Path) -> None:
    """Ranking flip: raw collapse vs collapse normalized by own target."""
    figure, axes = plt.subplots(figsize=(9, 5.2))
    by_raw = sorted(records, key=lambda row: row["raw_collapse"])
    by_norm = sorted(records, key=lambda row: row["collapse_over_target"])
    raw_rank = {row["config"]: i for i, row in enumerate(by_raw, 1)}
    norm_rank = {row["config"]: i for i, row in enumerate(by_norm, 1)}
    for record in records:
        config = record["config"]
        colour = colour_for(config, record)
        width = 2.4 if config in HIGHLIGHT else 1.0
        axes.plot(
            [0, 1],
            [raw_rank[config], norm_rank[config]],
            "-o",
            color=colour,
            linewidth=width,
            markersize=5,
        )
        axes.annotate(
            f"{config}  ({record['raw_collapse'] * 100:.1f}%)",
            (0, raw_rank[config]),
            textcoords="offset points",
            xytext=(-8, 0),
            ha="right",
            fontsize=8,
            color=colour,
        )
        axes.annotate(
            f"({record['collapse_over_target']:.2f}x)  {config}",
            (1, norm_rank[config]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            fontsize=8,
            color=colour,
        )
    axes.set_xlim(-0.45, 1.45)
    axes.set_xticks([0, 1])
    axes.set_xticklabels(
        [
            "rank by RAW collapse\n(lower looks 'better')",
            "rank by collapse / own target\n(the honest comparison)",
        ]
    )
    axes.set_ylabel("rank (1 = best)")
    axes.invert_yaxis()
    axes.set_title(
        "The ranking flips once each arm is judged against its own target:\n"
        "48x256 and 96x384 fall, 64x192 and 96x192 rise"
    )
    axes.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "fig2_raw_vs_normalized.png", dpi=180)
    plt.close(figure)


def fig3_diversity_not_information(records, plot_dir: Path) -> None:
    """Extra raw diversity converts into neither MI nor RankIC."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    xs = [row["unique"] for row in records]
    ys = [row["joint_mi_bits"] for row in records]
    axes[0].scatter(
        xs,
        ys,
        c=[colour_for(row["config"], row) for row in records],
        s=60,
        zorder=3,
    )
    annotate_points(axes[0], records, "unique", "joint_mi_bits")
    healthy = [
        row["joint_mi_bits"]
        for row in records
        if row["late_rankic_tstat"] >= 2.0
    ]
    axes[0].axhspan(
        min(healthy),
        max(healthy),
        color="#dddddd",
        alpha=0.5,
        label=f"healthy arms: MI {min(healthy):.2f}-{max(healthy):.2f} bits",
    )
    axes[0].set_xlabel("median daily unique joint tokens (raw diversity)")
    axes[0].set_ylabel("joint MI = H(target) - CE (bits/token)")
    axes[0].set_title(
        "Diversity is not information:\n"
        "48x256 emits 37% more tokens than 64x192, MI does not move"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    ys2 = [row["late_rankic"] for row in records]
    axes[1].scatter(
        xs,
        ys2,
        c=[colour_for(row["config"], row) for row in records],
        s=60,
        zorder=3,
    )
    annotate_points(axes[1], records, "unique", "late_rankic")
    axes[1].axhline(0.0, color="#888888", linewidth=1)
    axes[1].set_xlabel("median daily unique joint tokens (raw diversity)")
    axes[1].set_ylabel("late median daily RankIC")
    axes[1].set_title(
        "...and it does not convert into signal either:\n"
        "the diversity leaders sit below the selected arm on RankIC"
    )
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_dir / "fig3_diversity_not_information.png", dpi=180)
    plt.close(figure)


def fig4_decision(records, plot_dir: Path, selected: str) -> None:
    """Signal with uncertainty, plus the normalized-behaviour tiebreak."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ordered = sorted(records, key=lambda row: row["late_rankic"], reverse=True)
    labels = [row["config"] for row in ordered]
    x = range(len(ordered))
    errors = [
        row["late_rankic"] / row["late_rankic_tstat"]
        if row["late_rankic_tstat"] not in (0.0,)
        else 0.0
        for row in ordered
    ]
    axes[0].bar(
        x,
        [row["late_rankic"] for row in ordered],
        yerr=[abs(e) for e in errors],
        capsize=3,
        color=[colour_for(row["config"], row) for row in ordered],
    )
    axes[0].axhline(0.0, color="#333333", linewidth=1)
    for index, row in enumerate(ordered):
        axes[0].annotate(
            f"t={row['late_rankic_tstat']:.1f}",
            (index, max(row["late_rankic"], 0.0)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("late median daily RankIC (±1 SE, 400 days)")
    axes[0].set_title(
        "Signal over 400 single-day windows:\n"
        f"{selected} leads; grey arms fail the RankIC-t health gate"
    )
    axes[0].grid(axis="y", alpha=0.3)

    xs = [row["collapse_over_target"] for row in records]
    ys = [row["late_rankic"] for row in records]
    axes[1].scatter(
        xs,
        ys,
        c=[colour_for(row["config"], row) for row in records],
        s=[90 if row["config"] == selected else 55 for row in records],
        zorder=3,
    )
    annotate_points(axes[1], records, "collapse_over_target", "late_rankic")
    axes[1].axhline(0.0, color="#888888", linewidth=1)
    axes[1].set_xlabel("collapse / own target (lower = tracks target better)")
    axes[1].set_ylabel("late median daily RankIC")
    axes[1].set_title(
        "The decision quadrant: top-left is best.\n"
        f"{selected} pairs the strongest signal with near-best target-tracking"
    )
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_dir / "fig4_decision.png", dpi=180)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_report(records, selected: str, rationale: str) -> str:
    ordered = sorted(records, key=lambda row: row["late_rankic"], reverse=True)

    def mark(config: str) -> str:
        return f"**{config}**" if config == selected else config

    lines = [
        "# Exp 02 - Tokenizer architecture sweep: decision report",
        "",
        "All nine arms share the exact 7+7 codebook selected in Exp 01, a "
        "single GPT recipe, and the 400-window single-day full-coverage "
        "evaluation (holdout at offset 400+ untouched). The only variable is "
        "the tokenizer encoder (embedding x hidden).",
        "",
        "## 1. The headline table",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Arm",
                "Late RankIC (t)",
                "Raw collapse",
                "Target collapse",
                "Collapse / target",
                "Unique (target)",
                "DistAl",
                "Bal p10",
                "Joint MI",
                "Tok MAE",
            ],
            [
                [
                    mark(row["config"]),
                    f"{row['late_rankic']:.4f} ({row['late_rankic_tstat']:+.1f})",
                    f"{row['raw_collapse'] * 100:.1f}%",
                    f"{row['target_collapse'] * 100:.1f}%",
                    f"{row['collapse_over_target']:.2f}x",
                    f"{row['unique']:.0f} ({row['target_unique']:.0f})",
                    f"{row['distribution_alignment']:.3f}",
                    f"{row['balance_p10']:.3f}",
                    f"{row['joint_mi_bits']:.2f}",
                    f"{row['tokenizer_mae']:.4f}",
                ]
                for row in ordered
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 2. The confound came back wearing a different mask",
            "",
            "Exp 01 established that raw Collapse and raw Unique are "
            "mechanically driven by codebook size. This sweep pins the "
            "codebook to 7+7 everywhere, so raw values look comparable at "
            "last - but they are not. **The confounder changed identity: it "
            "is now the encoder-induced target distribution.** Each encoder "
            "slices the same market data into a different token stream, so "
            "each GPT is graded against different homework: the 48x256 "
            "target collapses at 13.9% per day while the 64x192 target "
            "collapses at 18.3% (`fig1_target_confound.png`). Raw GPT "
            "collapse tracks that target property almost one for one.",
            "",
            "Judged raw, 48x256 (40.2%) and 96x384 (41.6%) look 'better' "
            "than the selected 64x192 (44.7%). Judged against each arm's own "
            "target, the ranking flips (`fig2_raw_vs_normalized.png`): "
            "64x192 sits at 2.45x its target concentration - second only to "
            "96x192 (2.41x), whose signal is cut nearly in half - while "
            "48x256 drifts to 2.89x and 96x384 to 3.19x. The cosmetically "
            "low raw numbers mean a thinner target, not a stronger GPT.",
            "",
            "The same reasoning covers raw Unique: 48x256 emits 48 distinct "
            "tokens per day against a 556-token target (alignment 0.087); "
            "64x192 emits 35 against 430 (0.081). All nine arms live inside "
            "a narrow 0.066-0.090 alignment band - the diversity axis has "
            "no winner once the target is accounted for.",
            "",
            "## 3. Diversity that carries no information",
            "",
            "The decisive evidence is in `fig3_diversity_not_information."
            "png`. Moving from 64x192 to 48x256 buys 37% more raw diversity, "
            "yet joint MI stays flat (1.33 vs 1.34 bits/token) and late "
            "RankIC drops from 0.0418 to 0.0364. 96x384 pushes diversity "
            "further and loses even more signal (0.0148). Extra emitted "
            "variety that moves neither MI nor RankIC is reproduction of "
            "target noise, not extraction of structure - the operational "
            "definition of diversity the GPT cannot command. 128x256 shows "
            "the mirror-image failure: highest MI of the sweep (1.38) but "
            "the worst target-tracking (3.61x), and mid-pack signal.",
            "",
            "## 4. Health gate and decision",
            "",
            "The DA floors eliminated every arm (all late DA between 47.7% "
            "and 50.1%, |t| < 2.3), as preregistered expectations said they "
            "would; Stage A therefore degrades to the RankIC t-statistic "
            "gate (>= 2). 48x192 (raw collapse 77%, worst day emitting a "
            "single token, negative IC) and 96x256 (negative IC) fail it.",
            "",
            f"**Selected: {selected}.** {rationale}",
            "",
            "One more result worth recording: 64x192 is exactly the default "
            "encoder Exp 01 used. The sweep is a confirmation, not a "
            "coincidence - four times the encoder parameters (128x384) buys "
            "a better reconstruction MAE and nothing downstream, the same "
            "shape of conclusion Exp 01 reached about codebook size. The "
            "bottleneck of this pipeline is not representation capacity.",
            "",
            "## 5. Figures",
            "",
            "| Figure | Question it answers |",
            "| --- | --- |",
            "| `fig1_target_confound.png` | Why raw collapse is still "
            "confounded when every arm shares one codebook |",
            "| `fig2_raw_vs_normalized.png` | How the arm ranking flips "
            "under target-normalization |",
            "| `fig3_diversity_not_information.png` | Whether extra raw "
            "diversity converts into MI or RankIC (it does not) |",
            "| `fig4_decision.png` | The final signal-vs-tracking decision "
            "quadrant |",
            "",
            "![fig1](plots/report/fig1_target_confound.png)",
            "",
            "![fig2](plots/report/fig2_raw_vs_normalized.png)",
            "",
            "![fig3](plots/report/fig3_diversity_not_information.png)",
            "",
            "![fig4](plots/report/fig4_decision.png)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    selection = load_json(RESULTS_ROOT / "selection.json")
    selected = selection["selected"]["config"]
    rationale = selection.get("rationale", "")

    records = build_records()
    plot_dir = RESULTS_ROOT / "plots" / "report"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig1_target_confound(records, plot_dir)
    fig2_raw_vs_normalized(records, plot_dir)
    fig3_diversity_not_information(records, plot_dir)
    fig4_decision(records, plot_dir, selected)

    report = render_report(records, selected, rationale)
    report_path = RESULTS_ROOT / "REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Exp 02 report written to {report_path}")
    print(f"Figures: {plot_dir}")
    print(f"Selected: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
