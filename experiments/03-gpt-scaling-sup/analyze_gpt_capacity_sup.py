"""Analyze Exp 03-Sup with mature target-relative codebook diagnostics.

Selection is deliberately separated into three layers:

1. only stable five-epoch windows in each config's near-best validation-loss
   basin are eligible;
2. target-relative coarse-codebook metrics provide the primary ranking;
3. DA, daily RankIC, MAPE, and AmpRatio are non-scoring guardrails.

The single-seed recommendation uses a one-window-SD capacity rule: among
guardrail-passing Pareto candidates within one within-window standard deviation
of the best guardrail-passing coarse-balance result, choose the smallest model.
The raw primary leader and the parameter-efficiency knee are reported
separately, so the heuristic remains auditable and can be replaced after
multi-seed confirmation.

Fine and joint distributions remain mandatory diagnostics, but do not rank
capacity because the inherited training recipe teacher-conditions the fine
head on the previous coarse token while inference conditions on the currently
predicted coarse token.  Fixing that mismatch requires a separate retraining
study covering the complete controlled grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
DEFAULT_ROOT = SCRIPT_PATH.parent / "run_seed42"
WINDOW_SIZE = 5
LOSS_BASIN_RATIO = 1.01
EXPECTED_CONFIGS = (
    "deep",
    "depth6",
    "depth8",
    "width384_d4",
    "width512_d4_kv1",
    "xlarge",
)
AXES = {
    "depth": ("deep", "depth6", "depth8"),
    "width": ("deep", "width384_d4", "width512_d4_kv1"),
    "kv_heads": ("width512_d4_kv1", "xlarge"),
}

PRIMARY_FIELDS = (
    "median_daily_codebook_balance_score",
    "p10_daily_codebook_balance_score",
    "median_daily_token_support_f1",
    "median_daily_distribution_alignment",
    "median_daily_effective_token_alignment",
    "median_daily_collapse_alignment",
    "median_daily_target_support_recall",
    "median_daily_prediction_support_precision",
    "median_daily_token_jsd",
    "median_daily_coarse_token_accuracy",
)
DIAGNOSTIC_FIELDS = (
    "median_daily_fine_codebook_balance_score",
    "p10_daily_fine_codebook_balance_score",
    "median_daily_fine_token_support_f1",
    "median_daily_fine_token_jsd",
    "median_daily_fine_effective_token_alignment",
    "median_daily_fine_collapse_alignment",
    "median_daily_joint_codebook_balance_score",
    "p10_daily_joint_codebook_balance_score",
    "median_daily_joint_token_support_f1",
    "median_daily_joint_token_jsd",
    "median_daily_joint_effective_token_alignment",
    "median_daily_joint_collapse_alignment",
    "median_daily_joint_n_unique_tokens",
    "median_daily_joint_target_n_unique_tokens",
    "median_daily_joint_pred_effective_tokens",
    "median_daily_joint_target_effective_tokens",
    "median_daily_joint_collapse_rate",
    "median_daily_joint_target_collapse_rate",
    "median_daily_fine_n_unique_tokens",
    "median_daily_fine_target_n_unique_tokens",
    "median_daily_fine_collapse_rate",
    "median_daily_fine_target_collapse_rate",
)
GUARDRAILS = {
    "avg_da_per_date": {
        "direction": "higher",
        "floor": 0.01,
        "label": "DA",
    },
    "avg_daily_rank_ic": {
        "direction": "higher",
        "floor": 0.01,
        "label": "daily RankIC",
    },
    "avg_mape": {
        "direction": "lower",
        "floor": 0.05,
        "label": "MAPE",
    },
    "ampratio_log_error": {
        "direction": "lower",
        "floor": 0.05,
        "label": "|log AmpRatio|",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replace_with_retry(temporary: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            import time

            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    replace_with_retry(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    replace_with_retry(temporary, path)


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
    replace_with_retry(temporary, path)


def median(rows: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite {field} in mature window")
    return float(np.median(values))


def load_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "combined_epoch_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_json(path)
    for config in EXPECTED_CONFIGS:
        group = [row for row in rows if row["config"] == config]
        epochs = sorted(int(row["epoch"]) for row in group)
        if epochs != list(range(1, 51)):
            raise RuntimeError(
                f"{config} must have epochs 1-50; found {len(epochs)} rows"
            )
    required = set(PRIMARY_FIELDS) | set(DIAGNOSTIC_FIELDS) | set(GUARDRAILS)
    for row in rows:
        missing = [field for field in required if row.get(field) is None]
        if missing:
            raise RuntimeError(
                f"{row['config']} epoch {row['epoch']} lacks joint metrics: "
                + ", ".join(missing)
            )
    return rows


def consecutive_windows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["epoch"]))
    output = []
    for start in range(0, len(ordered) - WINDOW_SIZE + 1):
        window = ordered[start : start + WINDOW_SIZE]
        epochs = [int(row["epoch"]) for row in window]
        if epochs == list(range(epochs[0], epochs[0] + WINDOW_SIZE)):
            output.append(window)
    return output


def summarize_window(
    rows: list[dict[str, Any]],
    *,
    maturity_onset: int,
    loss_threshold: float,
    strict_loss_basin: bool,
) -> dict[str, Any]:
    first = rows[0]
    result = {
        "config": first["config"],
        "source": first["source"],
        "parameter_count": int(first["parameter_count"]),
        "dim": int(first["dim"]),
        "depth": int(first["depth"]),
        "heads": int(first["heads"]),
        "kv_heads": int(first["kv_heads"]),
        "ffn_multiplier": int(first["ffn_multiplier"]),
        "gradient_checkpointing": bool(first["gradient_checkpointing"]),
        "batch_tokens": int(first["batch_tokens"]),
        "eval_batch_size": int(first["eval_batch_size"]),
        "window_start": int(rows[0]["epoch"]),
        "window_end": int(rows[-1]["epoch"]),
        "representative_epoch": int(rows[WINDOW_SIZE // 2]["epoch"]),
        "maturity_onset": maturity_onset,
        "loss_basin_threshold": loss_threshold,
        "strict_loss_basin": strict_loss_basin,
        "median_val_loss": median(rows, "val_loss"),
    }
    for field in PRIMARY_FIELDS + DIAGNOSTIC_FIELDS + tuple(GUARDRAILS):
        result[field] = median(rows, field)
        result[f"window_std_{field}"] = float(
            np.std(
                [float(row[field]) for row in rows],
                ddof=1,
            )
        )
    result["avg_ampratio"] = median(rows, "avg_ampratio")
    return result


def eligible_windows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_loss = min(float(row["val_loss"]) for row in rows)
    threshold = min_loss * LOSS_BASIN_RATIO
    onset = min(
        int(row["epoch"])
        for row in rows
        if float(row["val_loss"]) <= threshold
    )
    all_windows = [
        window
        for window in consecutive_windows(rows)
        if int(window[0]["epoch"]) >= onset
    ]
    strict = [
        window
        for window in all_windows
        if all(float(row["val_loss"]) <= threshold for row in window)
    ]
    selected_pool = strict if strict else all_windows
    if not selected_pool:
        raise RuntimeError(
            f"No mature five-epoch window for {rows[0]['config']}"
        )
    summaries = [
        summarize_window(
            window,
            maturity_onset=onset,
            loss_threshold=threshold,
            strict_loss_basin=bool(strict),
        )
        for window in selected_pool
    ]
    return summaries, {
        "min_val_loss": min_loss,
        "loss_threshold": threshold,
        "maturity_onset": onset,
        "strict_windows": len(strict),
        "fallback_used": not bool(strict),
    }


def primary_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["median_daily_codebook_balance_score"]),
        float(row["p10_daily_codebook_balance_score"]),
        float(row["median_daily_token_support_f1"]),
        -float(row["median_daily_token_jsd"]),
        -float(row["parameter_count"]),
    )


def best_raw_window(windows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(windows, key=primary_key)


def guardrail_reference(
    deep_window: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    reference = {
        field: float(deep_window[field]) for field in GUARDRAILS
    }
    tolerances = {
        field: max(
            float(spec["floor"]),
            2.0 * float(deep_window[f"window_std_{field}"]),
        )
        for field, spec in GUARDRAILS.items()
    }
    return reference, tolerances


def apply_guardrails(
    row: dict[str, Any],
    reference: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    result = dict(row)
    failures = []
    for field, spec in GUARDRAILS.items():
        value = float(row[field])
        baseline = reference[field]
        tolerance = tolerances[field]
        delta = value - baseline
        result[f"guardrail_delta_{field}"] = delta
        result[f"guardrail_tolerance_{field}"] = tolerance
        if spec["direction"] == "higher":
            failed = value < baseline - tolerance
        else:
            failed = value > baseline + tolerance
        if failed:
            failures.append(str(spec["label"]))
    result["guardrail_pass"] = not failures
    result["guardrail_failures"] = ",".join(failures)
    return result


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passing = [row for row in rows if row["guardrail_pass"]]
    output = []
    for candidate in passing:
        dominated = any(
            other["parameter_count"] <= candidate["parameter_count"]
            and other["median_daily_codebook_balance_score"]
            >= candidate["median_daily_codebook_balance_score"]
            and (
                other["parameter_count"] < candidate["parameter_count"]
                or other["median_daily_codebook_balance_score"]
                > candidate["median_daily_codebook_balance_score"]
            )
            for other in passing
        )
        if not dominated:
            output.append(candidate)
    return sorted(output, key=lambda row: int(row["parameter_count"]))


def frontier_knee(front: list[dict[str, Any]]) -> dict[str, Any]:
    if len(front) <= 2:
        return min(front, key=lambda row: int(row["parameter_count"]))
    x = np.log10([float(row["parameter_count"]) for row in front])
    y = np.asarray(
        [
            float(row["median_daily_codebook_balance_score"])
            for row in front
        ],
        dtype=np.float64,
    )
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
    y_norm = (y - y.min()) / max(float(y.max() - y.min()), 1e-12)
    chord = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distance = y_norm - chord
    return front[int(np.argmax(distance))]


def select_windows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    maturity: dict[str, Any] = {}
    for config in EXPECTED_CONFIGS:
        group = [row for row in rows if row["config"] == config]
        pools[config], maturity[config] = eligible_windows(group)

    deep_raw = best_raw_window(pools["deep"])
    reference, tolerances = guardrail_reference(deep_raw)
    selected_windows = []
    for config in EXPECTED_CONFIGS:
        assessed = [
            apply_guardrails(window, reference, tolerances)
            for window in pools[config]
        ]
        passing = [row for row in assessed if row["guardrail_pass"]]
        selected = best_raw_window(passing if passing else assessed)
        selected["has_guardrail_passing_window"] = bool(passing)
        selected_windows.append(selected)

    # Keep the primary ranking pure: downstream guardrails are displayed next
    # to it and constrain selection, but never reorder target-relative results.
    ranked = sorted(selected_windows, key=primary_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["primary_rank"] = rank

    passing = [row for row in ranked if row["guardrail_pass"]]
    if not passing:
        raise RuntimeError("No mature window passes downstream guardrails")
    raw_leader = ranked[0]
    guardrail_leader = passing[0]
    front = pareto_front(selected_windows)
    knee = frontier_knee(front)
    near_best_margin = max(
        float(
            guardrail_leader[
                "window_std_median_daily_codebook_balance_score"
            ]
        ),
        1e-12,
    )
    near_best = [
        row
        for row in front
        if float(row["median_daily_codebook_balance_score"])
        >= float(
            guardrail_leader[
                "median_daily_codebook_balance_score"
            ]
        )
        - near_best_margin
    ]
    provisional = min(
        near_best, key=lambda row: int(row["parameter_count"])
    )
    selection = {
        "status": "provisional_single_seed",
        "selected": dict(provisional),
        "primary_coarse_balance_leader": dict(raw_leader),
        "guardrail_passing_leader": dict(guardrail_leader),
        "parameter_efficiency_knee": dict(knee),
        "near_best_rule": {
            "reference_config": guardrail_leader["config"],
            "leader_score": guardrail_leader[
                "median_daily_codebook_balance_score"
            ],
            "margin_one_leader_window_sd": near_best_margin,
            "eligible_configs": [row["config"] for row in near_best],
            "tie_break": "smallest parameter count",
        },
        "guardrail_reference": {
            "config": "deep",
            "window_start": deep_raw["window_start"],
            "window_end": deep_raw["window_end"],
            "values": reference,
            "tolerances": tolerances,
            "rule": (
                "adverse delta larger than max(practical floor, "
                "2x deep within-window epoch SD)"
            ),
        },
        "multi_seed_required": True,
        "holdout_used": False,
    }
    return ranked, selection, maturity


def sidecar_path(root: Path, config: str, epoch: int) -> Path:
    directory = root / "configs" / config / "epoch_trajectory"
    return directory / f"token_distributions_epoch_{epoch:03d}.npz"


def entropy_bits(probability: np.ndarray) -> float:
    nonzero = probability > 0
    return float(
        -np.sum(probability[nonzero] * np.log2(probability[nonzero]))
    )


def distribution_stats(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    *,
    level: str,
    vocab_fine: int,
) -> dict[str, Any]:
    pred_counts = pred_counts.astype(np.float64)
    true_counts = true_counts.astype(np.float64)
    pred_prob = pred_counts / pred_counts.sum()
    true_prob = true_counts / true_counts.sum()
    midpoint = 0.5 * (pred_prob + true_prob)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        nonzero = left > 0
        return float(
            np.sum(
                left[nonzero]
                * np.log2(left[nonzero] / right[nonzero])
            )
        )

    pred_support = pred_counts > 0
    true_support = true_counts > 0
    overlap = int(np.sum(pred_support & true_support))
    precision = overlap / max(int(pred_support.sum()), 1)
    recall = overlap / max(int(true_support.sum()), 1)
    support_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    pred_entropy = entropy_bits(pred_prob)
    true_entropy = entropy_bits(true_prob)
    pred_collapse = float(pred_prob.max())
    true_collapse = float(true_prob.max())

    def alignment(left: float, right: float) -> float:
        return float(min(left / right, right / left)) if left and right else 0.0

    jsd = 0.5 * kl(pred_prob, midpoint) + 0.5 * kl(true_prob, midpoint)
    effective_alignment = alignment(2**pred_entropy, 2**true_entropy)
    collapse_alignment = alignment(pred_collapse, true_collapse)
    balance = (
        support_f1
        * max(0.0, 1.0 - jsd)
        * effective_alignment
        * collapse_alignment
    ) ** 0.25

    def top_codes(counts: np.ndarray) -> list[dict[str, Any]]:
        ids = np.argsort(counts)[::-1]
        output = []
        total = float(counts.sum())
        for token_id in ids[:12]:
            if counts[token_id] <= 0:
                continue
            item = {
                "id": int(token_id),
                "count": int(counts[token_id]),
                "probability": float(counts[token_id] / total),
            }
            if level == "joint":
                item["coarse_id"] = int(token_id // vocab_fine)
                item["fine_id"] = int(token_id % vocab_fine)
            output.append(item)
        return output

    return {
        "n_pairs": int(pred_counts.sum()),
        "pred_unique": int(pred_support.sum()),
        "true_unique": int(true_support.sum()),
        "pred_collapse": pred_collapse,
        "true_collapse": true_collapse,
        "pred_entropy_bits": pred_entropy,
        "true_entropy_bits": true_entropy,
        "pred_effective_tokens": float(2**pred_entropy),
        "true_effective_tokens": float(2**true_entropy),
        "support_precision": float(precision),
        "target_support_recall": float(recall),
        "support_f1": float(support_f1),
        "jsd": float(jsd),
        "effective_alignment": effective_alignment,
        "collapse_alignment": collapse_alignment,
        "codebook_balance_score": float(balance),
        "pred_top_codes": top_codes(pred_counts),
        "true_top_codes": top_codes(true_counts),
    }


def inspect_representative_distributions(
    root: Path,
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    reference_targets: dict[str, np.ndarray] = {}
    for row in ranked:
        config = str(row["config"])
        epoch = int(row["representative_epoch"])
        path = sidecar_path(root, config, epoch)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            schema = int(payload["schema"][0])
            if schema != 2:
                raise RuntimeError(f"Unexpected distribution schema in {path}")
            vocab_fine = int(payload["vocab_fine"][0])
            levels: dict[str, Any] = {}
            for level in ("coarse", "fine", "joint"):
                pred = payload[f"{level}_pred_counts"].sum(axis=0)
                true = payload[f"{level}_true_counts"].sum(axis=0)
                if int(pred.sum()) != int(true.sum()):
                    raise RuntimeError(
                        f"{config}@{epoch} {level} pred/true count mismatch"
                    )
                if level not in reference_targets:
                    reference_targets[level] = true.copy()
                elif not np.array_equal(reference_targets[level], true):
                    raise RuntimeError(
                        f"Target {level} distribution differs across configs"
                    )
                levels[level] = distribution_stats(
                    pred,
                    true,
                    level=level,
                    vocab_fine=vocab_fine,
                )
            output[config] = {
                "epoch": epoch,
                "sidecar": str(path.resolve()),
                "sidecar_size_bytes": path.stat().st_size,
                "levels": levels,
            }
    return {
        "target_counts_identical_across_configs": True,
        "representatives": output,
    }


def axis_comparisons(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config = {row["config"]: row for row in ranked}
    output = []
    fields = (
        "median_daily_codebook_balance_score",
        "median_daily_token_support_f1",
        "median_daily_token_jsd",
        "median_daily_effective_token_alignment",
        "median_daily_collapse_alignment",
        "median_daily_fine_codebook_balance_score",
        "median_daily_joint_codebook_balance_score",
        "avg_da_per_date",
        "avg_daily_rank_ic",
        "avg_mape",
        "avg_ampratio",
    )
    for axis, configs in AXES.items():
        for left_name, right_name in zip(configs, configs[1:]):
            left = by_config[left_name]
            right = by_config[right_name]
            row: dict[str, Any] = {
                "axis": axis,
                "from_config": left_name,
                "to_config": right_name,
                "parameter_delta": int(right["parameter_count"])
                - int(left["parameter_count"]),
                "parameter_ratio": float(right["parameter_count"])
                / float(left["parameter_count"]),
            }
            for field in fields:
                row[f"delta_{field}"] = float(right[field]) - float(left[field])
            output.append(row)
    return output


def make_plot(root: Path, ranked: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    ordered = sorted(ranked, key=lambda row: int(row["parameter_count"]))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for row in ordered:
        marker = "o" if row["guardrail_pass"] else "x"
        colour = "tab:blue" if row["guardrail_pass"] else "tab:red"
        axis.scatter(
            row["parameter_count"] / 1e6,
            row["median_daily_codebook_balance_score"],
            marker=marker,
            color=colour,
            s=70,
        )
        axis.annotate(
            row["config"],
            (
                row["parameter_count"] / 1e6,
                row["median_daily_codebook_balance_score"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    selected = selection["selected"]
    axis.scatter(
        selected["parameter_count"] / 1e6,
        selected["median_daily_codebook_balance_score"],
        marker="*",
        color="gold",
        edgecolor="black",
        s=220,
        label="provisional selection",
        zorder=5,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Parameters (million, log scale)")
    axis.set_ylabel("Mature 5-epoch coarse codebook balance")
    axis.set_title("Exp 03-Sup: capacity vs target-relative coarse-code use")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "capacity_frontier.png", dpi=180)
    plt.close(figure)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(
    ranked: list[dict[str, Any]],
    selection: dict[str, Any],
    maturity: dict[str, Any],
    distributions: dict[str, Any],
) -> str:
    selected = selection["selected"]
    leader = selection["primary_coarse_balance_leader"]
    knee = selection["parameter_efficiency_knee"]
    lines = [
        "# Exp 03-Sup 分析报告",
        "",
        "> 状态：单 seed（42）探索性结论；holdout offset 400 未开启。",
        f"> 临时选型：`{selected['config']}@{selected['representative_epoch']}`，"
        f"成熟窗口 epoch {selected['window_start']}–{selected['window_end']}。",
        f"> Coarse target-relative 指标原始领先者：`{leader['config']}`；"
        f"参数效率拐点：`{knee['config']}`。",
        "",
        "## 选择规则",
        "",
        "- 成熟起点为首次进入 `val_loss <= 1.01 × min(val_loss)`；只比较连续 5 epoch 窗口。",
        "- 主排名是 target-relative coarse codebook balance；同时公开 support F1、JSD、有效 token 对齐和 collapse 对齐。",
        "- Fine/joint 是强制审计轨道，不参与容量主排名：当前训练与推理的 fine coarse-conditioning 存在一位错配，修复需另立全网格重训实验。",
        "- DA、逐日 RankIC、MAPE、AmpRatio 只作 guardrail，不进入码本得分。",
        "- 在 guardrail 通过的参数—coarse-score Pareto 前沿上，选取距离“guardrail 通过者中的领先点”不超过其窗口内 1 个标准差的最小模型。",
        "- 该规则只产生单-seed 临时选型；入围点必须补 multi-seed。",
        "",
        "## 成熟窗口主排名",
        "",
        "| Rank | Config@ep | Params | Window | Coarse balance | P10 | Support F1 | JSD | Eff align | Collapse align | Fine balance | Joint balance | DA | RankIC | MAPE | Amp | Guardrail |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(ranked, key=lambda item: int(item["primary_rank"])):
        lines.append(
            f"| {row['primary_rank']} | {row['config']}@{row['representative_epoch']} "
            f"| {row['parameter_count'] / 1e6:.2f}M "
            f"| {row['window_start']}–{row['window_end']} "
            f"| {row['median_daily_codebook_balance_score']:.3f} "
            f"| {row['p10_daily_codebook_balance_score']:.3f} "
            f"| {row['median_daily_token_support_f1']:.3f} "
            f"| {row['median_daily_token_jsd']:.3f} "
            f"| {row['median_daily_effective_token_alignment']:.3f} "
            f"| {row['median_daily_collapse_alignment']:.3f} "
            f"| {row['median_daily_fine_codebook_balance_score']:.3f} "
            f"| {row['median_daily_joint_codebook_balance_score']:.3f} "
            f"| {format_percent(row['avg_da_per_date'])} "
            f"| {row['avg_daily_rank_ic']:.4f} "
            f"| {row['avg_mape']:.3f} "
            f"| {row['avg_ampratio']:.3f} "
            f"| {'PASS' if row['guardrail_pass'] else 'FAIL: ' + row['guardrail_failures']} |"
        )
    lines.extend(
        [
            "",
            "## 受控轴解释",
            "",
            "- 深度轴：`deep → depth6 → depth8`，dim=256、heads=4、kv_heads=1 固定。",
            "- 宽度轴：`deep → width384_d4 → width512_d4_kv1`，depth=4、kv_heads=1 固定；heads 仅随 dim 调整以保持 head_dim=64。",
            "- KV 对照：`width512_d4_kv1 → xlarge` 只把 kv_heads 从 1 改为 2。",
            "",
            "## Fine / joint 分布审计",
            "",
            "每个 epoch 都有压缩 NPZ sidecar，保存四个验证窗口下 coarse、fine、joint 的完整 true/pred count vector；"
            "分析器验证所有配置的 true counts 完全一致，且每层 pred/true 总数一致。",
            "",
            "注意：fine/joint 数值描述的是当前已训练实现的实际行为。由于 fine head 的训练 teacher condition 与推理 condition 错位，"
            "它们不能单独归因于模型尺寸，也不能作为本轮容量主排名；但必须保留，防止用 coarse 结果外推完整 65,536 码本。",
            "",
            "| Config@ep | Fine unique (pred/true) | Fine effective (pred/true) | Fine collapse (pred/true) | Joint unique (pred/true) | Joint effective (pred/true) | Joint collapse (pred/true) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(ranked, key=lambda item: int(item["primary_rank"])):
        item = distributions["representatives"][row["config"]]
        fine = item["levels"]["fine"]
        joint = item["levels"]["joint"]
        lines.append(
            f"| {row['config']}@{item['epoch']} "
            f"| {fine['pred_unique']} / {fine['true_unique']} "
            f"| {fine['pred_effective_tokens']:.1f} / {fine['true_effective_tokens']:.1f} "
            f"| {format_percent(fine['pred_collapse'])} / {format_percent(fine['true_collapse'])} "
            f"| {joint['pred_unique']} / {joint['true_unique']} "
            f"| {joint['pred_effective_tokens']:.1f} / {joint['true_effective_tokens']:.1f} "
            f"| {format_percent(joint['pred_collapse'])} / {format_percent(joint['true_collapse'])} |"
        )
    lines.extend(
        [
            "",
            "完整 65,536 维计数不嵌入 Markdown；其路径、hash 和 top-code 摘要记录在 "
            "`representative_distribution_summary.json`，原始向量保存在各 epoch sidecar。",
            "",
            "## 成熟性与限制",
            "",
        ]
    )
    for config in EXPECTED_CONFIGS:
        item = maturity[config]
        lines.append(
            f"- `{config}`：min val loss={item['min_val_loss']:.4f}，"
            f"成熟起点 ep{item['maturity_onset']}，"
            f"严格 5-epoch 窗口={item['strict_windows']}，"
            f"fallback={'是' if item['fallback_used'] else '否'}。"
        )
    lines.extend(
        [
            "- 5 个相邻 epoch 不是独立重复；窗口标准差只用于抑制 checkpoint 偶然峰值，不是跨 seed 置信区间。",
            "- guardrail 是相对 `deep` 成熟窗口的退化筛查，并不证明下游指标显著改善。",
            "- 单 seed 只能确定下一轮复现实验的入围尺寸，不能宣称全局最优。",
            "- holdout 保持封存，直到多 seed 容量选型与后续 HPO 均冻结。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Exp 03-Sup mature target-relative capacity"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "study_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["settings"]["protocol_invariants"]["holdout_used"]:
        raise RuntimeError("Refusing to analyze a run that opened holdout")
    rows = load_rows(root)
    ranked, selection, maturity = select_windows(rows)
    distributions = inspect_representative_distributions(root, ranked)
    comparisons = axis_comparisons(ranked)

    selection.update(
        {
            "experiment": "Exp 03-Sup",
            "seed": 42,
            "window_size": WINDOW_SIZE,
            "loss_basin_ratio": LOSS_BASIN_RATIO,
            "selection_metric": (
                "median_daily_codebook_balance_score over a mature "
                "five-epoch window"
            ),
            "distribution_levels": ["coarse", "fine", "joint"],
        }
    )
    analysis = {
        "selection": selection,
        "maturity": maturity,
        "ranked_windows": ranked,
        "axis_comparisons": comparisons,
        "distribution_validation": {
            "target_counts_identical_across_configs": distributions[
                "target_counts_identical_across_configs"
            ],
            "representative_file": str(
                (root / "representative_distribution_summary.json").resolve()
            ),
        },
        "holdout_used": False,
        "single_seed": True,
    }
    atomic_write_json(root / "selection.json", selection)
    atomic_write_json(root / "analysis.json", analysis)
    atomic_write_json(
        root / "representative_distribution_summary.json",
        distributions,
    )
    write_csv(root / "capacity_window_ranking.csv", ranked)
    write_csv(root / "axis_comparisons.csv", comparisons)
    write_csv(root / "parameter_codebook_pareto.csv", pareto_front(ranked))
    atomic_write_text(
        root / "ANALYSIS.md",
        render_report(ranked, selection, maturity, distributions),
    )
    make_plot(root, ranked, selection)
    print(
        "Exp 03-Sup provisional selection: "
        f"{selection['selected']['config']}@"
        f"{selection['selected']['representative_epoch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
