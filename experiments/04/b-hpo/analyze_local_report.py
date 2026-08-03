"""Exp 04-B local analysis: aggregate downloaded trial results and render report plots.

Reads the downloaded results tree at
    server_runs/results/04b-hpo/seed42/
and writes
    server_runs/results/04b-hpo/seed42/combined_trial_summary.csv
    server_runs/results/04b-hpo/seed42/plots/*.png

Read-only against the trial data; no checkpoints are touched.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Kronos-R-Preview\server_runs\results\04b-hpo\seed42")
TRIALS = ROOT / "trials"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

LN2 = math.log(2.0)

# ---------------------------------------------------------------- load
leaderboard = json.loads((ROOT / "leaderboard.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "study_manifest.json").read_text(encoding="utf-8"))
rows = leaderboard["rows"]

planned = {p["tid"]: p for p in manifest["plan"]}
completed_tids = {r["tid"] for r in rows}
missing = [t for t in planned if t not in completed_tids]

HPARAMS = ["lr_muon", "lr", "dropout", "weight_decay", "label_smoothing",
           "fine_weight", "het_weight", "warmup_ratio"]

summaries = []
trajectories = {}
for r in rows:
    tid = r["tid"]
    tdir = TRIALS / tid
    csv = tdir / "epoch_trajectory" / "epoch_summary.csv"
    df = pd.read_csv(csv)
    trajectories[tid] = df

    token_summary = json.loads((tdir / "dataset_token_summary.json").read_text(encoding="utf-8"))
    h_coarse = token_summary["splits"]["validation"]["coarse"]["entropy_bits"]
    h_fine = token_summary["splits"]["validation"]["fine"]["entropy_bits"]

    med = df.median(numeric_only=True)
    s = {
        "tid": tid,
        "is_baseline": bool(r["is_baseline"]),
        "rank": r["rank"],
        # trajectory-median token quality (primary family)
        "coarse_balance": med["median_daily_codebook_balance_score"],
        "coarse_balance_p10": med["p10_daily_codebook_balance_score"],
        "coarse_support_f1": med["median_daily_token_support_f1"],
        "coarse_jsd": med["median_daily_token_jsd"],
        "coarse_eff_align": med["median_daily_effective_token_alignment"],
        "coarse_collapse": med["median_daily_collapse_rate"],
        "coarse_unique": med["median_daily_unique_tokens"],
        "fine_balance": med["median_daily_fine_codebook_balance_score"],
        "fine_support_f1": med["median_daily_fine_token_support_f1"],
        "fine_unique": med["median_daily_fine_n_unique_tokens"],
        "joint_balance": med["median_daily_joint_codebook_balance_score"],
        "joint_support_f1": med["median_daily_joint_token_support_f1"],
        "joint_unique": med["median_daily_joint_n_unique_tokens"],
        # guardrails
        "da": med["avg_da_per_date"],
        "rank_ic": med["avg_daily_rank_ic"],
        "mape": med["avg_mape"],
        "ampratio": med["avg_ampratio"],
        "ampratio_log_error": med["ampratio_log_error"],
        "val_loss_min": df["val_loss"].min(),
        "val_loss_median": med["val_loss"],
        # information captured per token, H(target) - CE, in bits
        "coarse_mi_bits": h_coarse - med["val_coarse_loss"] / LN2,
        "fine_mi_bits": h_fine - med["val_fine_loss"] / LN2,
        "best_da": r["best_da"],
        "best_da_epoch": r["best_da_epoch"],
    }
    for p in HPARAMS:
        s[p] = r["params"][p]
    summaries.append(s)

summary = pd.DataFrame(summaries).sort_values("coarse_balance", ascending=False).reset_index(drop=True)
summary.to_csv(ROOT / "combined_trial_summary.csv", index=False)

baseline = summary[summary["is_baseline"]].iloc[0]
top = summary.iloc[0]
top4 = summary.head(4)

print("=== completed:", len(summary), " missing:", missing)
print(summary[["tid", "coarse_balance", "coarse_support_f1", "coarse_jsd",
               "da", "rank_ic", "mape", "coarse_mi_bits"]].to_string(index=False))

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
BASE_C = "#d62728"   # baseline = red
TOP_C = "#2ca02c"    # best = green
MID_C = "#7f9ccb"    # others

def short(tid: str) -> str:
    return "baseline" if tid == "baseline" else tid.replace("trial_", "")[:8]

def bar_colors(df):
    best = df["coarse_balance"].max()
    return [BASE_C if b else (TOP_C if v == best else MID_C)
            for b, v in zip(df["is_baseline"], df["coarse_balance"])]

# ------------------------------------------------- 1. primary ranking bar
fig, ax = plt.subplots(figsize=(9, 6.2))
d = summary.iloc[::-1]
ax.barh([short(t) for t in d["tid"]], d["coarse_balance"], color=bar_colors(d))
for y, v in enumerate(d["coarse_balance"]):
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("median daily coarse codebook balance (full 50-epoch trajectory)")
ax.set_title("Exp 04-B HPO - primary metric ranking (17 completed trials)")
ax.set_xlim(0, max(d["coarse_balance"]) * 1.12)
fig.tight_layout()
fig.savefig(PLOTS / "primary_ranking.png")
plt.close(fig)

# ------------------------------------------------- 2. hyperparameter sensitivity
fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
log_x = {"lr_muon", "lr", "weight_decay"}
for ax, p in zip(axes.flat, HPARAMS):
    sub = summary.sort_values(p)
    x = sub[p].astype(float)
    ax.scatter(x, sub["coarse_balance"],
               c=[BASE_C if b else MID_C for b in sub["is_baseline"]], s=42, zorder=3)
    base_row = sub[sub["is_baseline"]]
    ax.axhline(baseline["coarse_balance"], color=BASE_C, ls="--", lw=1, alpha=0.6)
    # connect one-factor neighbours of the baseline recipe
    neigh = sub[np.isclose(sub[[q for q in HPARAMS if q != p]].astype(float).values,
                           baseline[[q for q in HPARAMS if q != p]].astype(float).values).all(axis=1)]
    neigh = neigh.sort_values(p)
    if len(neigh) > 1:
        ax.plot(neigh[p].astype(float), neigh["coarse_balance"], "-", color="#555555", lw=1.2, zorder=2)
    if p in log_x:
        ax.set_xscale("log")
        ticks = sorted(set(float(v) for v in x))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8)
        ax.minorticks_off()
    ax.set_xlabel(p)
    ax.set_ylabel("coarse balance" if p == HPARAMS[0] or p == "label_smoothing" else "")
    ax.set_title(p, fontsize=10)
fig.suptitle("Hyperparameter sensitivity (one-factor-at-a-time around baseline; dashed = baseline balance)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(PLOTS / "hparam_sensitivity.png")
plt.close(fig)

# ------------------------------------------------- 3. guardrail scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (gx, glabel) in zip(axes, [("rank_ic", "avg daily RankIC"), ("mape", "avg MAPE (%)")]):
    ax.scatter(summary[gx], summary["coarse_balance"], c=MID_C, s=46, zorder=3)
    for _, rr in summary.iterrows():
        if rr["is_baseline"]:
            ax.scatter(rr[gx], rr["coarse_balance"], c=BASE_C, s=90, marker="D", zorder=4)
            ax.annotate("baseline", (rr[gx], rr["coarse_balance"]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=BASE_C)
        elif rr.name < 3:
            ax.scatter(rr[gx], rr["coarse_balance"], facecolors="none", edgecolors=TOP_C, s=140, zorder=4)
            ax.annotate(short(rr["tid"]), (rr[gx], rr["coarse_balance"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=TOP_C)
    ax.set_xlabel(glabel)
    ax.set_ylabel("coarse balance")
    ax.set_title(f"primary vs guardrail: {glabel}")
fig.suptitle("Guardrail check - best token quality also clears downstream guardrails")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / "guardrail_scatter.png")
plt.close(fig)

# ------------------------------------------------- 4. epoch trajectories
show = list(top4["tid"]) + ["baseline"]
palette = ["#2ca02c", "#1f77b4", "#9467bd", "#8c564b", BASE_C]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for tid, c in zip(show, palette):
    df = trajectories[tid]
    lbl = short(tid) + (" (lr_muon=%g)" % summary.set_index("tid").loc[tid, "lr_muon"]
                        if tid != "baseline" else " (lr_muon=0.02)")
    ls = "--" if tid == "baseline" else "-"
    axes[0].plot(df["epoch"], df["val_loss"], ls, color=c, lw=1.4, label=lbl)
    axes[1].plot(df["epoch"], df["median_daily_codebook_balance_score"], ls, color=c, lw=1.4, label=lbl)
    axes[2].plot(df["epoch"], df["avg_daily_rank_ic"], ls, color=c, lw=1.4, label=lbl)
axes[0].set_title("val loss"); axes[0].set_xlabel("epoch")
axes[1].set_title("coarse codebook balance"); axes[1].set_xlabel("epoch")
axes[2].set_title("avg daily RankIC"); axes[2].set_xlabel("epoch")
axes[1].legend(fontsize=8, loc="lower right")
fig.suptitle("Epoch trajectories - top 4 trials vs baseline")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / "epoch_trajectories.png")
plt.close(fig)

# ------------------------------------------------- 5. token quality panel (levels)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sel = pd.concat([top4, summary[summary["is_baseline"]]]).drop_duplicates("tid")
sel = sel.sort_values("coarse_balance", ascending=False)
x = np.arange(len(sel))
w = 0.27
axes[0].bar(x - w, sel["coarse_balance"], w, label="coarse")
axes[0].bar(x, sel["fine_balance"], w, label="fine")
axes[0].bar(x + w, sel["joint_balance"], w, label="joint")
axes[0].set_xticks(x, [short(t) for t in sel["tid"]], rotation=20)
axes[0].set_title("codebook balance by level")
axes[0].legend()
axes[1].bar(x - w, sel["coarse_support_f1"], w, label="coarse")
axes[1].bar(x, sel["fine_support_f1"], w, label="fine")
axes[1].bar(x + w, sel["joint_support_f1"], w, label="joint")
axes[1].set_xticks(x, [short(t) for t in sel["tid"]], rotation=20)
axes[1].set_title("token support F1 by level")
axes[1].legend()
fig.suptitle("Token quality across codebook levels - top 4 trials vs baseline")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / "token_quality_panel.png")
plt.close(fig)

# ------------------------------------------------- 6. information bits
fig, ax = plt.subplots(figsize=(9, 5.6))
d = summary.iloc[::-1]
y = np.arange(len(d))
ax.barh(y - 0.19, d["coarse_mi_bits"], 0.38, label="coarse: H(target) - CE",
        color=[BASE_C if b else MID_C for b in d["is_baseline"]])
ax.barh(y + 0.19, d["fine_mi_bits"], 0.38, label="fine: H(target) - CE",
        color=[BASE_C if b else "#c2d3ea" for b in d["is_baseline"]], hatch="//", edgecolor="white")
ax.set_yticks(y, [short(t) for t in d["tid"]])
ax.set_xlabel("bits per token (higher = more target entropy captured)")
ax.set_title("Vocabulary-invariant learning evidence (validation marginals)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "info_bits.png")
plt.close(fig)

# ------------------------------------------------- 7. selection map: diversity vs likelihood
fig, ax = plt.subplots(figsize=(8.6, 6))
ax.scatter(summary["coarse_mi_bits"], summary["coarse_balance"],
           c=MID_C, s=52, zorder=3, label="trial")
sel_tid = top["tid"]
run_tid = summary.iloc[1]["tid"]
for tid, marker, color, dx, dy in [
        (sel_tid, "*", TOP_C, 8, 6), (run_tid, "s", "#1f77b4", 8, -12),
        ("baseline", "D", BASE_C, 8, -12)]:
    rr = summary[summary["tid"] == tid].iloc[0]
    ax.scatter(rr["coarse_mi_bits"], rr["coarse_balance"], marker=marker, c=color,
               s=200 if marker == "*" else 90, zorder=4,
               edgecolors="black" if marker == "*" else "none", linewidths=0.8)
    ax.annotate(f"{short(tid)}\n(lr_muon={rr['lr_muon']:g})",
                (rr["coarse_mi_bits"], rr["coarse_balance"]),
                textcoords="offset points", xytext=(dx, dy), fontsize=8, color=color)
ax.set_xlabel("likelihood evidence: coarse H(target) - CE (bits/token)")
ax.set_ylabel("diversity: median daily codebook balance")
ax.set_title("Selection map - token quality only (downstream metrics excluded by design)")
ax.text(0.02, 0.02,
        "top-2 are adjacent points on the lr_muon slope = robust region, not a lucky draw",
        transform=ax.transAxes, fontsize=8, color="#555555")
fig.tight_layout()
fig.savefig(PLOTS / "selection_map.png")
plt.close(fig)

# ------------------------------------------------- console digest for report
print("\n=== baseline vs top ===")
cols = ["coarse_balance", "coarse_balance_p10", "coarse_support_f1", "coarse_jsd",
        "coarse_eff_align", "fine_balance", "fine_support_f1", "joint_balance",
        "da", "rank_ic", "mape", "ampratio_log_error", "coarse_mi_bits", "fine_mi_bits"]
for c in cols:
    print(f"{c:22s} baseline={baseline[c]:.4f}  top={top[c]:.4f}  delta={top[c]-baseline[c]:+.4f}")
print("\nmissing planned trials:")
for t in missing:
    print(" ", t, planned[t]["params"])
print("\nlr_muon sweep:")
lm = summary.sort_values("lr_muon")[["lr_muon", "coarse_balance", "coarse_support_f1", "coarse_jsd", "rank_ic"]]
print(lm.drop_duplicates("lr_muon").to_string(index=False))
