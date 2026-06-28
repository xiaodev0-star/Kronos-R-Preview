# -*- coding: utf-8 -*-
"""BitSweep 实验绘图脚本（全量评估版本）

为 Exp-SweepBit.md 生成出版级可视化图表。数据源来自
`checkpoints/sweep/full_eval_seed42/metrics.json`（493 个交易日、~220 万条预测）。

Usage:
    python doc/gen_sweep_plots.py
"""
import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════
#  全局样式
# ═══════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#222222",
    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# 配色板
C_PRIMARY = "#2E5C8A"      # 主蓝
C_ACCENT  = "#C44536"      # 强调红（8+6）
C_OK      = "#3A9D5D"      # 健康绿
C_WARN    = "#E09F3E"      # 过渡黄
C_BAD     = "#9E2A2B"      # 失效红
C_NEUTRAL = "#7A7A7A"      # 中性灰
C_FILL    = "#EAF2F8"      # 浅蓝填充

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "checkpoints" / "sweep" / "full_eval_seed42" / "metrics.json"
OUT_DIR = ROOT / "doc" / "plt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════
with open(JSON_PATH, encoding="utf-8") as f:
    RAW = json.load(f)

CONFIGS = sorted(RAW.keys(), key=lambda k: (int(k.split("+")[0]), int(k.split("+")[1])))
RECO = "8+6"


def make_record(key):
    v = RAW[key]
    l1, l2 = map(int, key.split("+"))
    vocab = 2 ** (l1 + l2)
    unique = v["n_unique_tokens"]
    collapse = v["collapse_rate"] * 100.0
    util = unique / vocab * 100.0
    eff_bits = math.log2(unique)
    theo_bits = l1 + l2
    return {
        "config": key,
        "l1": l1,
        "l2": l2,
        "vocab": vocab,
        "unique": unique,
        "collapse": collapse,
        "util": util,
        "eff_bits": eff_bits,
        "theo_bits": theo_bits,
        "da": v["avg_da_per_date"],
        "da_std": v["da_std"],
        "da_above": v["avg_da_above_baseline"],
        "rank_ic": v.get("rank_ic", np.nan),
        "ampratio": v.get("ampratio", np.nan),
    }


DATA = [make_record(k) for k in CONFIGS]


def status(d):
    """分类：推荐 / 不满足约束 / 过度配置。"""
    if d["config"] == RECO:
        return "recommended"
    if d["unique"] < 64 or d["collapse"] > 30.0:
        return "fail"
    return "over"


def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out}")


# ═══════════════════════════════════════════════════════════════════
#  图 1：实验总览表
# ═══════════════════════════════════════════════════════════════════
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.axis("off")

    cols = ["Config", "L1+L2", "Vocab", "Unique", "Util%", "Collapse%", "Eff Bits", "DA%"]
    rows = [
        [
            d["config"],
            f'{d["l1"]}+{d["l2"]}',
            f'{d["vocab"]:,}',
            f'{d["unique"]}',
            f'{d["util"]:.3f}',
            f'{d["collapse"]:.2f}',
            f'{d["eff_bits"]:.2f}',
            f'{d["da"] * 100:.2f}',
        ]
        for d in DATA
    ]

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.75)

    # 表头
    for j in range(len(cols)):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")

    # 数据行：斑马纹 + 推荐高亮
    for i, d in enumerate(DATA):
        is_reco = d["config"] == RECO
        for j in range(len(cols)):
            cell = table[(i + 1, j)]
            cell.set_edgecolor("#e0e0e0")
            if is_reco:
                cell.set_facecolor("#fdecea")
                cell.set_text_props(weight="bold", color=C_ACCENT)
            elif i % 2 == 0:
                cell.set_facecolor("#fafbfc")
            else:
                cell.set_facecolor("white")

    # 数值列热力图（跳过推荐行）
    def heatmap(col_idx, values, reverse=False, cmin=None, cmax=None):
        vmin = cmin if cmin is not None else min(values)
        vmax = cmax if cmax is not None else max(values)
        for i, d in enumerate(DATA):
            if d["config"] == RECO:
                continue
            v = values[i]
            t = (v - vmin) / max(vmax - vmin, 1e-9)
            if reverse:
                t = 1 - t
            # 浅蓝 → 深蓝
            r = 0.95 - 0.28 * t
            g = 0.97 - 0.22 * t
            b = 1.00 - 0.10 * t
            table[(i + 1, col_idx)].set_facecolor((r, g, b))

    heatmap(4, [d["util"] for d in DATA])
    heatmap(5, [d["collapse"] for d in DATA], reverse=True)
    heatmap(6, [d["eff_bits"] for d in DATA])
    heatmap(7, [d["da"] for d in DATA])

    ax.set_title(
        "BitSweep 实验总览：10 组 BSQ 配置的码本使用指标",
        fontsize=14,
        pad=16,
        weight="bold",
    )
    save(fig, "summary_table.png")


# ═══════════════════════════════════════════════════════════════════
#  图 2：有效 Bits 饱和
# ═══════════════════════════════════════════════════════════════════
def plot_effective_bits():
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    theo = np.array([d["theo_bits"] for d in DATA])
    eff = np.array([d["eff_bits"] for d in DATA])
    coll = np.array([d["collapse"] for d in DATA])

    # 理想线
    ax.plot([11.5, 18.5], [11.5, 18.5], "--", color="#7f8c8d", lw=1.2,
            label="理想利用线 $y=x$", zorder=1)

    # 趋势线
    z = np.polyfit(theo, eff, 1)
    xs = np.linspace(11.5, 18.5, 200)
    ys = np.polyval(z, xs)
    ax.plot(xs, ys, "-", color=C_PRIMARY, lw=2.2,
            label=f"线性趋势（斜率 ${z[0]:.2f}$）", zorder=2)

    # 渐变填充：理想线与趋势线之间的信息利用带
    # 使用 contourf 在两条曲线之间生成平滑的颜色过渡
    xx, yy = np.meshgrid(
        np.linspace(11.5, 18.5, 400),
        np.linspace(5.5, 8.4, 400),
    )
    trend_grid = np.polyval(z, xx)
    ratio = (yy - trend_grid) / (xx - trend_grid)
    ratio = np.where((yy >= trend_grid) & (yy <= xx), ratio, np.nan)
    cmap_grad = mpl.colors.LinearSegmentedColormap.from_list(
        "usage", ["#3A9D5D", "#E09F3E", "#C44536"]
    )
    ax.contourf(xx, yy, ratio, levels=80, cmap=cmap_grad, alpha=0.22, zorder=0)

    # 散点：颜色映射 Collapse
    norm = mpl.colors.Normalize(vmin=coll.min(), vmax=coll.max())
    cmap = mpl.cm.RdYlGn_r
    for d in DATA:
        is_reco = d["config"] == RECO
        ax.scatter(
            d["theo_bits"], d["eff_bits"],
            s=360 if is_reco else 150,
            c=[cmap(norm(d["collapse"]))],
            edgecolors=C_ACCENT if is_reco else "white",
            linewidths=2.5 if is_reco else 1.2,
            marker="*" if is_reco else "o",
            zorder=5,
        )
        ax.annotate(
            d["config"],
            (d["theo_bits"], d["eff_bits"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10 if is_reco else 9,
            color=C_ACCENT if is_reco else "#2c3e50",
            weight="bold" if is_reco else "normal",
        )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("坍塌率 Collapse%", fontsize=10)

    # 顶部词表容量轴
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([12, 13, 14, 15, 16, 17, 18])
    ax2.set_xticklabels([f"${2**b:,}$" for b in [12, 13, 14, 15, 16, 17, 18]], fontsize=9)
    ax2.set_xlabel("联合词表大小 $V=2^{L_1+L_2}$", fontsize=10)

    ax.set_xlim(11.5, 18.5)
    ax.set_ylim(5.5, 8.4)
    ax.set_xlabel("理论码本容量 $L_1+L_2$ (bits)", fontsize=11)
    ax.set_ylabel("有效 Bits  =  $\\log_2(\\mathrm{Unique\\,Tokens})$", fontsize=11)
    ax.set_title(
        "有效 Bits 饱和：从 6+6 到 9+9 理论容量翻 64 倍，有效信息仅增 1.84 bits",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)

    ax.text(
        0.62, 0.12,
        f"理论 Bits  {theo.min():.0f} $\\rightarrow$ {theo.max():.0f}\n"
        f"有效 Bits  {eff.min():.2f} $\\rightarrow$ {eff.max():.2f}\n"
        f"趋势斜率  {z[0]:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    save(fig, "effective_bits_saturation.png")


# ═══════════════════════════════════════════════════════════════════
#  图 3：Unique vs Vocab 规模化效应
# ═══════════════════════════════════════════════════════════════════
def plot_unique_vs_vocab():
    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    vocabs = np.array([d["vocab"] for d in DATA])
    uniques = np.array([d["unique"] for d in DATA])

    ax.set_xscale("log")
    ax.set_yscale("log")

    # 理想线
    xs = np.logspace(3.5, 5.5, 100)
    ax.plot(xs, xs, "--", color="#95a5a6", lw=1.2,
            label="理想：$\\mathrm{Unique}=V$", zorder=1)

    # 幂律趋势
    logx = np.log10(vocabs)
    logy = np.log10(uniques)
    z = np.polyfit(logx, logy, 1)
    xt = np.logspace(3.6, 5.4, 100)
    yt = 10 ** np.polyval(z, np.log10(xt))
    ax.plot(xt, yt, "-", color=C_PRIMARY, lw=1.5, alpha=0.6,
            label=f"幂律趋势（指数 ${z[0]:.2f}$）", zorder=2)

    # 健康区间
    ax.axhspan(64, 128, color="#d5f5e3", alpha=0.4, zorder=0,
               label="健康 Unique 区间 $[64,128]$")

    # 散点
    for d in DATA:
        is_reco = d["config"] == RECO
        ax.scatter(
            d["vocab"], d["unique"],
            s=300 if is_reco else 130,
            c=C_ACCENT if is_reco else C_PRIMARY,
            edgecolors="white",
            linewidths=2.0,
            marker="*" if is_reco else "o",
            zorder=5,
        )
        ax.annotate(
            d["config"],
            (d["vocab"], d["unique"]),
            textcoords="offset points",
            xytext=(8, 7),
            fontsize=10 if is_reco else 9,
            color=C_ACCENT if is_reco else "#2c3e50",
            weight="bold" if is_reco else "normal",
        )

    # 趋势箭头
    ax.annotate(
        "", xy=(16384, 144), xytext=(4096, 61),
        arrowprops=dict(arrowstyle="->", color=C_OK, lw=2.0,
                        connectionstyle="arc3,rad=0.15"),
    )
    ax.text(
        7000, 82,
        "$V$ 翻 $4\\times$\nUnique 翻 $2.4\\times$",
        fontsize=9,
        color=C_OK,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_OK, alpha=0.9),
    )

    ax.annotate(
        "", xy=(262144, 218), xytext=(16384, 144),
        arrowprops=dict(arrowstyle="->", color=C_BAD, lw=2.0,
                        connectionstyle="arc3,rad=-0.15"),
    )
    ax.text(
        60000, 118,
        "$V$ 翻 $16\\times$\nUnique 仅 $+51\\%$",
        fontsize=9,
        color=C_BAD,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BAD, alpha=0.9),
    )

    # 分区注释
    ax.text(4500, 260, "码本过小\n表达受限", ha="center", fontsize=9,
            color=C_BAD, alpha=0.85, style="italic")
    ax.text(180000, 260, "码本过大\n死权重多", ha="center", fontsize=9,
            color="#b7950b", alpha=0.85, style="italic")

    ax.set_xlim(3000, 320000)
    ax.set_ylim(45, 320)
    ax.set_xlabel("理论码本容量 $V=2^{L_1+L_2}$（对数轴）", fontsize=11)
    ax.set_ylabel("GPT 实际激活的 Unique Tokens（对数轴）", fontsize=11)
    ax.set_title(
        "码本规模化效应：$V$ 从 4K 增至 262K，Unique 仅从 61 增至 218",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "unique_vs_vocab.png")


# ═══════════════════════════════════════════════════════════════════
#  图 4：Util% vs Collapse 健康象限
# ═══════════════════════════════════════════════════════════════════
def plot_health_quadrant():
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    utils = np.array([d["util"] for d in DATA])
    colls = np.array([d["collapse"] for d in DATA])
    uniques = np.array([d["unique"] for d in DATA])

    util_thr = 0.5
    coll_thr = 30.0

    # 分界线
    ax.axvline(util_thr, color="#7f8c8d", lw=1.0, ls=":", alpha=0.7)
    ax.axhline(coll_thr, color="#7f8c8d", lw=1.0, ls=":", alpha=0.7)

    # 象限底色
    ax.fill_between([util_thr, 2.0], 0, coll_thr, color="#d5f5e3", alpha=0.35, zorder=0)
    ax.fill_between([0.03, util_thr], coll_thr, 70, color="#fadbd8", alpha=0.35, zorder=0)
    ax.fill_between([0.03, util_thr], 0, coll_thr, color="#fef9e7", alpha=0.35, zorder=0)

    # 散点：按状态着色，大小映射 Unique
    color_map = {
        "recommended": C_ACCENT,
        "fail": C_BAD,
        "over": C_WARN,
    }
    for d in DATA:
        st = status(d)
        size = 160 + (d["unique"] - uniques.min()) / max(uniques.max() - uniques.min(), 1e-9) * 360
        ax.scatter(
            d["util"], d["collapse"],
            s=size,
            c=color_map[st],
            edgecolors="white",
            linewidths=1.5,
            marker="*" if st == "recommended" else "o",
            zorder=5,
        )
        ax.annotate(
            d["config"],
            (d["util"], d["collapse"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10 if st == "recommended" else 9,
            color=C_ACCENT if st == "recommended" else "#2c3e50",
            weight="bold" if st == "recommended" else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.04, 2.0)
    ax.set_ylim(12, 52)
    ax.set_xlabel("码本利用率 Util% = Unique / Vocab（对数轴）", fontsize=11)
    ax.set_ylabel("坍塌率 Collapse%（越低越好）", fontsize=11)
    ax.set_title("GPT 学习健康度：利用率 vs 坍塌率", fontsize=13, pad=12)

    # 象限标签
    ax.text(1.0, 16, "推荐区\n高利用 + 低坍塌", fontsize=9, color=C_OK, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_OK, alpha=0.9))
    ax.text(0.06, 48, "失效区\n低利用 + 高坍塌", fontsize=9, color=C_BAD, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_BAD, alpha=0.9))
    ax.text(0.06, 16, "过度配置区\n大词表 + 低利用", fontsize=9, color="#b7950b",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#b7950b", alpha=0.9))

    # 图例
    legend_elements = [
        Patch(facecolor=C_ACCENT, edgecolor="white", label="推荐 8+6"),
        Patch(facecolor=C_BAD, edgecolor="white", label="不满足约束"),
        Patch(facecolor=C_WARN, edgecolor="white", label="过度配置"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              title="状态", title_fontsize=9)

    # 气泡大小说明
    handles = [
        plt.scatter([], [], s=180, c="#bdc3c7", edgecolors="white"),
        plt.scatter([], [], s=360, c="#7f8c8d", edgecolors="white"),
        plt.scatter([], [], s=540, c="#2c3e50", edgecolors="white"),
    ]
    labels = ["Unique 小", "Unique 中", "Unique 大"]
    ax.legend(handles, labels, loc="lower left", fontsize=8,
              title="气泡大小", title_fontsize=8).set_zorder(10)

    save(fig, "utilization_vs_collapse.png")


# ═══════════════════════════════════════════════════════════════════
#  图 5：Util% 效率阶梯
# ═══════════════════════════════════════════════════════════════════
def plot_utilization_ladder():
    sorted_data = sorted(DATA, key=lambda d: d["util"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    y = np.arange(len(sorted_data))
    colors = []
    for d in sorted_data:
        st = status(d)
        if st == "recommended":
            colors.append(C_OK)
        elif st == "fail":
            colors.append(C_BAD)
        else:
            colors.append("#95a5a6")

    bars = ax.barh(
        y,
        [d["util"] for d in sorted_data],
        color=colors,
        edgecolor="white",
        height=0.62,
    )

    # 数值标签
    for i, d in enumerate(sorted_data):
        width = d["util"]
        label = f"{d['util']:.3f}%    U={d['unique']}, C={d['collapse']:.1f}%"
        ax.text(
            width + 0.02,
            i,
            label,
            va="center",
            fontsize=8.5,
            color="#2c3e50",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([d["config"] for d in sorted_data])
    ax.set_xlabel("码本利用率 Util%", fontsize=11)
    ax.set_title("码本利用率阶梯：8+6 在满足约束配置中效率最高", fontsize=13, pad=12)
    ax.set_xlim(0, 1.7)
    ax.invert_yaxis()

    # 经验阈值
    ax.axvline(0.5, color="#7f8c8d", lw=1.0, ls="--", alpha=0.7)
    ax.text(0.52, len(sorted_data) - 0.8, "Util = 0.5%", fontsize=8,
            color="#7f8c8d", va="top")

    legend_elements = [
        Patch(facecolor=C_OK, edgecolor="white", label="推荐 8+6"),
        Patch(facecolor=C_BAD, edgecolor="white", label="不满足约束"),
        Patch(facecolor="#95a5a6", edgecolor="white", label="过度配置"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    save(fig, "utilization_ladder.png")


# ═══════════════════════════════════════════════════════════════════
#  图 6：综合画像雷达图
# ═══════════════════════════════════════════════════════════════════
def plot_radar():
    """用四维雷达图展示候选配置的多维短板，不使用‘紧凑度’这种循环指标。"""
    configs = ["6+6", "7+7", "8+6", "8+7", "9+6", "9+9"]
    selected = [next(d for d in DATA if d["config"] == c) for c in configs]

    labels = ["Unique\n充足度", "Collapse\n健康度", "Util%\n效率", "信息\n密度"]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    util_max = max(d["util"] for d in selected)  # 6+6 的 1.489%
    info_densities = [d["eff_bits"] / d["theo_bits"] for d in selected]
    id_max = max(info_densities)

    def score(d):
        return [
            min(d["unique"] / 128.0, 1.0),
            max(0.0, 1.0 - d["collapse"] / 50.0),
            d["util"] / util_max,
            (d["eff_bits"] / d["theo_bits"]) / id_max,
        ]

    scores = [score(d) for d in selected]

    fig, ax = plt.subplots(figsize=(8.0, 7.6), subplot_kw=dict(polar=True))

    palette = ["#3498db", "#95a5a6", C_ACCENT, C_WARN, "#3A9D5D", "#bdc3c7"]
    linestyles = ["--", "--", "-", "-", "-", "-"]
    linewidths = [1.6, 1.6, 3.0, 1.8, 1.8, 1.6]

    for d, s, c, ls, lw in zip(selected, scores, palette, linestyles, linewidths):
        values = s + s[:1]
        ax.plot(
            angles,
            values,
            color=c,
            lw=lw,
            ls=ls,
            label=d["config"],
            zorder=5,
        )
        if d["config"] == RECO:
            ax.fill(angles, values, color=c, alpha=0.16, zorder=0)

    # 轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#888888", size=8)
    ax.set_title(
        "码本配置综合画像：不看‘紧凑度’，只看实际表达与健康度",
        fontsize=13,
        pad=24,
        weight="bold",
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.30, 1.10),
        fontsize=9,
        title="配置",
        title_fontsize=9,
    )

    # 短板诊断 + 最小维度得分
    diagnosis = (
        "短板诊断：\n"
        "• 6+6：Unique=61，表达容量不足\n"
        "• 7+7：Collapse=46.6%，输出严重坍塌\n"
        "• 8+7：Util=0.513%，词表利用率低\n"
        "• 9+6：Util=0.647%，利用率偏低\n"
        "• 9+9：Util=0.083%，有效信息极度稀疏"
    )
    fig.text(
        0.15,
        -0.02,
        diagnosis,
        fontsize=9,
        color="#2c3e50",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    # 图注
    fig.text(
        0.15,
        -0.19,
        "注：四轴均为中性效率/健康指标，已归一化到 [0,1]；Util% 效率以 6+6 的 Util=1.489% 为上限。",
        fontsize=8,
        color="#666666",
    )

    save(fig, "radar_comparison.png")


# ═══════════════════════════════════════════════════════════════════
#  图 7：同 Vocab 控制变量对比
# ═══════════════════════════════════════════════════════════════════
def plot_paired_comparison():
    pairs = [
        ("16,384", "7+7", "8+6"),
        ("32,768", "8+7", "9+6"),
        ("65,536", "8+8", "9+7"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))

    for ax, (label, a_name, b_name) in zip(axes, pairs):
        da = next(d for d in DATA if d["config"] == a_name)
        db = next(d for d in DATA if d["config"] == b_name)
        ax2 = ax.twinx()

        w = 0.35
        # Unique：左轴
        xu = np.array([0])
        ax.bar(xu - w / 2, [da["unique"]], w, label=a_name,
               color="#7fb3d5", edgecolor="white")
        ax.bar(xu + w / 2, [db["unique"]], w, label=b_name,
               color=C_ACCENT, edgecolor="white")

        # Util%：右轴
        xut = np.array([1])
        ax2.bar(xut - w / 2, [da["util"]], w, color="#7fb3d5",
                edgecolor="white", alpha=0.85)
        ax2.bar(xut + w / 2, [db["util"]], w, color=C_ACCENT,
                edgecolor="white", alpha=0.85)

        # 数值标签
        ax.text(0 - w / 2, da["unique"] + 6, f"{da['unique']}",
                ha="center", fontsize=9, color="#2c3e50")
        ax.text(0 + w / 2, db["unique"] + 6, f"{db['unique']}",
                ha="center", fontsize=9, color=C_ACCENT, weight="bold")

        ax2.text(1 - w / 2, da["util"] + 0.03, f"{da['util']:.3f}",
                 ha="center", fontsize=9, color="#2c3e50")
        ax2.text(1 + w / 2, db["util"] + 0.03, f"{db['util']:.3f}",
                 ha="center", fontsize=9, color=C_ACCENT, weight="bold")

        # Collapse / Eff Bits 信息盒
        info = (
            f"{a_name}:  C={da['collapse']:.1f}%,  E={da['eff_bits']:.2f}\n"
            f"{b_name}:  C={db['collapse']:.1f}%,  E={db['eff_bits']:.2f}"
        )
        ax.text(
            0.5, 0.96, info,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Unique", "Util%"], fontsize=10)
        ax.set_ylabel("Unique tokens", color="#2c3e50", fontsize=10)
        ax2.set_ylabel("Util%", color="#b36b00", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#b36b00")

        left_max = max(da["unique"], db["unique"])
        right_max = max(da["util"], db["util"])
        ax.set_ylim(0, left_max * 1.45)
        ax2.set_ylim(0, right_max * 1.55)
        ax.set_title(f"Vocab = {label}", fontsize=11.5)

        # 图例
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "同词表控制变量对比：L1/L2 分配结构对码本使用的影响",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    save(fig, "paired_comparison.png")


# ═══════════════════════════════════════════════════════════════════
#  图 7：DA 收敛
# ═══════════════════════════════════════════════════════════════════
def plot_da_convergence():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    x = np.arange(len(DATA))
    da_pct = np.array([d["da"] * 100 for d in DATA])
    std_pct = np.array([d["da_std"] * 100 for d in DATA])
    colors = [C_ACCENT if d["config"] == RECO else "#5dade2" for d in DATA]

    # 使用点 + 误差线，避免大误差棒压过柱状图
    ax.errorbar(
        x,
        da_pct,
        yerr=std_pct,
        fmt="o",
        markersize=9,
        color="#2c3e50",
        ecolor="#95a5a6",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
        label="日均 DA",
    )
    for xi, yi, c in zip(x, da_pct, colors):
        ax.scatter(xi, yi, s=120, c=c, edgecolors="white", linewidths=1.5, zorder=4)
    ax.axhline(50.0, color="#7f8c8d", lw=1.5, ls="--", label="随机基准 50%")

    ax.set_xticks(x)
    ax.set_xticklabels([d["config"] for d in DATA])
    ax.set_ylabel("日均 Directional Accuracy (%)", fontsize=11)
    ax.set_xlabel("(L1, L2) 配置", fontsize=11)
    ax.set_title(
        "DA 在随机基准附近收敛：码本大小不是方向预测杠杆",
        fontsize=13,
        pad=12,
    )
    ax.set_ylim(43, 55)

    # 极差标注
    ax.annotate(
        f"极差 = {da_pct.max() - da_pct.min():.2f} pp",
        xy=(0.97, 0.08),
        xycoords="axes fraction",
        fontsize=9,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
    )
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "da_convergence.png")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading data from: {JSON_PATH}")
    print(f"Configs ({len(CONFIGS)}): {CONFIGS}")
    print(f"Output dir: {OUT_DIR}\n")

    # 打印关键数值供报告核对
    da_pct = [d["da"] * 100 for d in DATA]
    print(f"DA range: {min(da_pct):.2f}% ~ {max(da_pct):.2f}%")
    print(f"Util max (8+6): {next(d['util'] for d in DATA if d['config']==RECO):.3f}%")
    z = np.polyfit([d["theo_bits"] for d in DATA], [d["eff_bits"] for d in DATA], 1)
    print(f"Eff-bits trend slope: {z[0]:.3f}\n")

    print("Generating 8 publication-quality plots...")
    plot_summary_table()
    plot_effective_bits()
    plot_unique_vs_vocab()
    plot_health_quadrant()
    plot_utilization_ladder()
    plot_radar()
    plot_paired_comparison()
    plot_da_convergence()
    print("\nDone. All plots saved to doc/plt/")
