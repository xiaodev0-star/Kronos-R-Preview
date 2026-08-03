"""Analyze the current Exp 02 tokenizer sweep without a composite score.

Measurement policy inherited from the Exp 01 review
--------------------------------------------------
Exp 01 initially read raw ``p90_daily_collapse_rate`` and raw
``median_daily_unique_tokens`` as if they were comparable across arms. They are
not when the codebook changes size, and they still drift inside Exp 02 because
different encoder capacities fill the fixed codebook to different degrees. This
module therefore judges behaviour with the alignment metrics that compare the
prediction against the *same-day target* distribution, keeps the raw counts only
as descriptive context, and adds two evidence groups Exp 01 lacked:

* per-window floors (``min_window_da``, ``min_window_daily_rank_ic``), which were
  the only fields that separated genuinely robust arms from arms whose four
  windows disagreed;
* predictive information in bits, ``H(target) - CE``, which is invariant to
  vocabulary size and shows directly whether extra codebook occupancy buys the
  GPT any learnable structure.

Nothing here is combined into a weighted score, and no selection is recorded
without an explicit reviewed ``--select_config``/``--rationale``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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

DEFAULT_WEIGHTS_ROOT, DEFAULT_ROOT = default_study_roots(
    "02-tokenizer-tuning", seed=42
)

QUALITY_OBJECTIVES = (
    ("avg_da_per_date", "max"),
    ("avg_daily_rank_ic", "max"),
    ("avg_mape", "min"),
)
# Capacity-normalized behaviour. Each alignment field is a symmetric ratio (or
# 1 - JSD) between the prediction and the same-day target distribution, so it
# cannot be inflated by simply having more codes available.
BEHAVIOUR_OBJECTIVES = (
    ("median_daily_collapse_alignment", "max"),
    ("median_daily_unique_token_alignment", "max"),
    ("ampratio_log_error", "min"),
)
# Worst-case evidence: an arm that only looks good on the pooled average is not
# a usable upstream dependency.
ROBUSTNESS_OBJECTIVES = (
    ("min_window_da", "max"),
    ("min_window_daily_rank_ic", "max"),
    ("p10_daily_codebook_balance_score", "max"),
)
JOINT_OBJECTIVES = QUALITY_OBJECTIVES + BEHAVIOUR_OBJECTIVES

# Raw behaviour counts are still reported, but only as descriptive context; they
# are never Pareto objectives and never enter the staged screen.
DESCRIPTIVE_BEHAVIOUR_FIELDS = (
    "p90_daily_collapse_rate",
    "worst_daily_collapse_rate",
    "median_daily_unique_tokens",
    "min_daily_unique_tokens",
)

# Preregistered Stage-A floors for the staged screen. Windows are single
# trading days, so a worst-window floor would degenerate into the single worst
# day out of ~400 - a bar every model fails. Stage A therefore gates on the
# level of the daily mean plus t-statistics of the daily series: the mean daily
# DA must clear the coin flip, and the mean daily RankIC must clear zero, each
# by at least two standard errors across the evaluated days. Both are invariant
# to how the days are grouped into windows. The floors stay deliberately loose:
# Exp 02 is an upstream architecture choice, not a deployment gate, and the
# preregistered health gate (collapse <= 0.35, unique >= 32) is expected to
# stay unmet at this stage.
STAGE_A_FLOORS = {
    "late_median_da": 0.50,
    "da_tstat_vs_coinflip": 2.0,
    "rankic_tstat_vs_zero": 2.0,
}

REQUIRED_ROW_FIELDS = (
    "avg_da_per_date",
    "avg_daily_rank_ic",
    "avg_mape",
    "n_dates",
    "min_window_da",
    "window_da_std",
    "min_window_daily_rank_ic",
    "window_daily_rank_ic_std",
    "median_daily_collapse_alignment",
    "median_daily_unique_token_alignment",
    "median_daily_effective_token_alignment",
    "median_daily_distribution_alignment",
    "median_daily_token_support_f1",
    "median_daily_codebook_balance_score",
    "p10_daily_codebook_balance_score",
    "median_daily_target_n_unique_tokens",
    "median_daily_target_effective_tokens",
    "median_daily_target_collapse_rate",
    "median_daily_pred_effective_tokens",
    "median_daily_collapse_rate",
    "tokenizer_coarse_utilization",
    "tokenizer_fine_utilization",
    "tokenizer_joint_utilization",
    "tokenizer_coarse_effective_codes",
    "tokenizer_fine_effective_codes",
)

LOG2 = math.log(2.0)


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


def finite(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def dominates(
    candidate: dict[str, Any],
    target: dict[str, Any],
    objectives: Iterable[tuple[str, str]],
) -> bool:
    strictly_better = False
    for key, direction in objectives:
        left = finite(candidate, key)
        right = finite(target, key)
        if left is None or right is None:
            return False
        if direction == "max":
            if left < right:
                return False
            strictly_better |= left > right
        else:
            if left > right:
                return False
            strictly_better |= left < right
    return strictly_better


def pareto_flags(
    rows: list[dict[str, Any]],
    objectives: tuple[tuple[str, str], ...],
) -> list[bool]:
    return [
        not any(
            other_index != index and dominates(other, row, objectives)
            for other_index, other in enumerate(rows)
        )
        for index, row in enumerate(rows)
    ]


def safe_spearman(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return {"rho": None, "pvalue": None, "n": len(left)}
    result = spearmanr(left, right)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if math.isfinite(rho) else None,
        "pvalue": pvalue if math.isfinite(pvalue) else None,
        "n": len(left),
    }


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
    os.replace(temporary, path)


def require_fields(rows: list[dict[str, Any]]) -> None:
    """Fail loudly instead of silently degrading the Pareto fronts.

    ``dominates`` treats a missing objective as "incomparable", so an absent
    alignment field would quietly turn every point into a Pareto point. The
    fields below are produced by the current evaluator and by the Exp 02 runner;
    if any is missing the bundle predates this analysis contract.
    """
    missing = sorted(
        {field for field in REQUIRED_ROW_FIELDS if field not in rows[0]}
    )
    if missing:
        raise RuntimeError(
            "combined_epoch_summary.json is missing capacity-normalized "
            f"fields required by this analysis: {missing}. Re-run the Exp 02 "
            "runner with the current evaluator so the alignment and per-layer "
            "codebook diagnostics are exported."
        )


def annotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = pareto_flags(rows, QUALITY_OBJECTIVES)
    behaviour = pareto_flags(rows, BEHAVIOUR_OBJECTIVES)
    robustness = pareto_flags(rows, ROBUSTNESS_OBJECTIVES)
    joint = pareto_flags(rows, JOINT_OBJECTIVES)
    return [
        {
            **row,
            "quality_pareto": quality[index],
            "behaviour_pareto": behaviour[index],
            "robustness_pareto": robustness[index],
            "joint_pareto": joint[index],
        }
        for index, row in enumerate(rows)
    ]


def arg_extreme(
    rows: list[dict[str, Any]], key: str, direction: str
) -> dict[str, Any]:
    valid = [row for row in rows if finite(row, key) is not None]
    function = max if direction == "max" else min
    return function(valid, key=lambda row: float(row[key]))


def config_directory_name(embedding_dim: int, hidden_dim: int) -> str:
    return f"emb_{int(embedding_dim):03d}_hid_{int(hidden_dim):03d}"


def target_entropy_bits(
    root: Path, embedding_dim: int, hidden_dim: int
) -> dict[str, float | None]:
    """Marginal validation entropy of the GPT targets, in bits.

    ``train_base.py`` writes the exact train/validation coarse/fine/joint token
    distributions next to the per-config history. Comparing the GPT's validation
    cross-entropy against these marginals gives the mutual information the model
    actually captured, which is the only prediction-quality statistic in this
    study that is invariant to how many codes the tokenizer exposes.
    """
    path = (
        root
        / "configs"
        / config_directory_name(embedding_dim, hidden_dim)
        / "dataset_token_summary.json"
    )
    empty: dict[str, float | None] = {
        "target_coarse_entropy_bits": None,
        "target_fine_entropy_bits": None,
        "target_joint_entropy_bits": None,
    }
    if not path.is_file():
        return empty
    try:
        validation = load_json(path)["splits"]["validation"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return empty
    return {
        "target_coarse_entropy_bits": float(
            validation["coarse"]["entropy_bits"]
        ),
        "target_fine_entropy_bits": float(validation["fine"]["entropy_bits"]),
        "target_joint_entropy_bits": float(
            validation["joint"]["entropy_bits"]
        ),
    }


def predictive_information(
    late: list[dict[str, Any]], entropy: dict[str, float | None]
) -> dict[str, float | None]:
    """Late-region CE in bits and the bits of target structure it explains."""

    def median_bits(key: str) -> float | None:
        values = [
            finite(row, key) for row in late if finite(row, key) is not None
        ]
        if not values:
            return None
        return float(np.median(values)) / LOG2

    coarse_ce = median_bits("val_coarse_loss")
    fine_ce = median_bits("val_fine_loss")
    coarse_h = entropy["target_coarse_entropy_bits"]
    fine_h = entropy["target_fine_entropy_bits"]
    joint_h = entropy["target_joint_entropy_bits"]
    coarse_mi = (
        coarse_h - coarse_ce
        if coarse_h is not None and coarse_ce is not None
        else None
    )
    fine_mi = (
        fine_h - fine_ce
        if fine_h is not None and fine_ce is not None
        else None
    )
    joint_mi = (
        joint_h - coarse_ce - fine_ce
        if joint_h is not None
        and coarse_ce is not None
        and fine_ce is not None
        else None
    )
    return {
        "late_coarse_ce_bits": coarse_ce,
        "late_fine_ce_bits": fine_ce,
        "late_coarse_mi_bits": coarse_mi,
        "late_fine_mi_bits": fine_mi,
        "late_joint_mi_bits": joint_mi,
        "late_coarse_mi_fraction": (
            coarse_mi / coarse_h
            if coarse_mi is not None and coarse_h
            else None
        ),
    }


def envelopes(
    rows: list[dict[str, Any]], root: Path
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in sorted({str(row["config"]) for row in rows}):
        group = sorted(
            [row for row in rows if row["config"] == config],
            key=lambda row: int(row["epoch"]),
        )
        late = group[-min(10, len(group)) :]
        best_da = arg_extreme(group, "avg_da_per_date", "max")
        best_rank = arg_extreme(group, "avg_daily_rank_ic", "max")
        best_mape = arg_extreme(group, "avg_mape", "min")
        best_collapse = arg_extreme(
            group, "p90_daily_collapse_rate", "min"
        )
        best_amp = arg_extreme(group, "ampratio_log_error", "min")
        best_balance = arg_extreme(
            group, "median_daily_codebook_balance_score", "max"
        )

        def late_median(key: str) -> float:
            return float(np.median([row[key] for row in late]))

        entropy = target_entropy_bits(
            root, group[0]["embedding_dim"], group[0]["hidden_dim"]
        )
        record = {
            "config": config,
            "embedding_dim": group[0]["embedding_dim"],
            "hidden_dim": group[0]["hidden_dim"],
            "n_epochs": len(group),
            "healthy_epochs": sum(bool(row.get("healthy")) for row in group),
            # --- tokenizer sufficiency -------------------------------------
            "tokenizer_mae": group[0]["tokenizer_mae"],
            "tokenizer_rmse": group[0]["tokenizer_rmse"],
            "tokenizer_coarse_unique": group[0]["tokenizer_coarse_unique"],
            "tokenizer_coarse_utilization": group[0][
                "tokenizer_coarse_utilization"
            ],
            "tokenizer_coarse_effective_codes": group[0][
                "tokenizer_coarse_effective_codes"
            ],
            "tokenizer_coarse_collapse": group[0][
                "tokenizer_coarse_collapse"
            ],
            "tokenizer_fine_unique": group[0]["tokenizer_fine_unique"],
            "tokenizer_fine_utilization": group[0][
                "tokenizer_fine_utilization"
            ],
            "tokenizer_fine_effective_codes": group[0][
                "tokenizer_fine_effective_codes"
            ],
            "tokenizer_joint_unique": group[0]["tokenizer_joint_unique"],
            "tokenizer_joint_entropy_bits": group[0][
                "tokenizer_joint_entropy_bits"
            ],
            "tokenizer_joint_utilization": group[0][
                "tokenizer_joint_utilization"
            ],
            "tokenizer_joint_collapse": group[0][
                "tokenizer_joint_collapse"
            ],
            # --- metric-specific extrema (not a winner) --------------------
            "best_da": best_da["avg_da_per_date"],
            "best_da_epoch": best_da["epoch"],
            "best_rankic": best_rank["avg_daily_rank_ic"],
            "best_rankic_epoch": best_rank["epoch"],
            "best_mape": best_mape["avg_mape"],
            "best_mape_epoch": best_mape["epoch"],
            "best_p90_collapse": best_collapse["p90_daily_collapse_rate"],
            "best_p90_collapse_epoch": best_collapse["epoch"],
            "best_ampratio": best_amp["avg_ampratio"],
            "best_ampratio_epoch": best_amp["epoch"],
            "best_codebook_balance": best_balance[
                "median_daily_codebook_balance_score"
            ],
            "best_codebook_balance_epoch": best_balance["epoch"],
            # --- mature, codebook-size-invariant quality -------------------
            "late_epoch_start": int(late[0]["epoch"]),
            "late_median_da": late_median("avg_da_per_date"),
            "late_median_rankic": late_median("avg_daily_rank_ic"),
            "late_median_mape": late_median("avg_mape"),
            "late_median_baseline_mape": late_median("avg_baseline_mape"),
            "late_median_ampratio": late_median("avg_ampratio"),
            # --- mature worst-case evidence --------------------------------
            "min_window_da": late_median("min_window_da"),
            "window_da_std": late_median("window_da_std"),
            "min_window_daily_rank_ic": late_median(
                "min_window_daily_rank_ic"
            ),
            "window_daily_rank_ic_std": late_median(
                "window_daily_rank_ic_std"
            ),
            "n_eval_dates": late_median("n_dates"),
            "min_window_ampratio": late_median("min_window_ampratio"),
            "max_window_ampratio": late_median("max_window_ampratio"),
            "p10_daily_codebook_balance_score": late_median(
                "p10_daily_codebook_balance_score"
            ),
            # --- capacity-normalized behaviour -----------------------------
            "late_median_collapse_alignment": late_median(
                "median_daily_collapse_alignment"
            ),
            "late_median_unique_token_alignment": late_median(
                "median_daily_unique_token_alignment"
            ),
            "late_median_effective_token_alignment": late_median(
                "median_daily_effective_token_alignment"
            ),
            "late_median_distribution_alignment": late_median(
                "median_daily_distribution_alignment"
            ),
            "late_median_token_support_f1": late_median(
                "median_daily_token_support_f1"
            ),
            "late_median_codebook_balance": late_median(
                "median_daily_codebook_balance_score"
            ),
            # --- same-day target references for the ratios above -----------
            "late_median_target_unique": late_median(
                "median_daily_target_n_unique_tokens"
            ),
            "late_median_target_effective": late_median(
                "median_daily_target_effective_tokens"
            ),
            "late_median_target_collapse": late_median(
                "median_daily_target_collapse_rate"
            ),
            # --- descriptive raw counts (never used as an objective) -------
            "late_median_unique": late_median("median_daily_unique_tokens"),
            "late_median_min_unique": late_median("min_daily_unique_tokens"),
            "late_median_pred_effective": late_median(
                "median_daily_pred_effective_tokens"
            ),
            "late_median_collapse": late_median("median_daily_collapse_rate"),
            "late_median_p90_collapse": late_median(
                "p90_daily_collapse_rate"
            ),
            "late_median_worst_collapse": late_median(
                "worst_daily_collapse_rate"
            ),
            # --- Pareto participation --------------------------------------
            "quality_pareto_epochs": sum(
                bool(row["quality_pareto"]) for row in group
            ),
            "behaviour_pareto_epochs": sum(
                bool(row["behaviour_pareto"]) for row in group
            ),
            "robustness_pareto_epochs": sum(
                bool(row["robustness_pareto"]) for row in group
            ),
            "joint_pareto_epochs": sum(
                bool(row["joint_pareto"]) for row in group
            ),
        }
        record.update(entropy)
        record.update(predictive_information(late, entropy))
        # t-statistics of the daily series, used by the Stage-A screen. With
        # single-day windows the per-window std IS the daily std, so the
        # standard error is std / sqrt(number of evaluated days).
        dates = max(float(record["n_eval_dates"]), 1.0)
        da_se = record["window_da_std"] / math.sqrt(dates)
        ic_se = record["window_daily_rank_ic_std"] / math.sqrt(dates)
        record["da_tstat_vs_coinflip"] = (
            (record["late_median_da"] - 0.5) / da_se if da_se > 0 else 0.0
        )
        record["rankic_tstat_vs_zero"] = (
            record["late_median_rankic"] / ic_se if ic_se > 0 else 0.0
        )
        output.append(record)
    return output


def staged_screen(summary: list[dict[str, Any]]) -> dict[str, Any]:
    """Document the ordered screen; never pick a winner automatically.

    Stage A keeps only arms whose daily-mean quality clears the preregistered,
    codebook-invariant floors with statistical margin (t >= 2 over the ~400
    evaluated days). Stage B ranks the survivors on capacity-normalized
    behaviour. Stage C reports tokenizer sufficiency. The result is an ordered
    shortlist for human review, not a score.
    """
    stage_a = []
    for row in summary:
        checks = {
            key: bool(float(row[key]) >= threshold)
            for key, threshold in STAGE_A_FLOORS.items()
        }
        stage_a.append(
            {
                "config": row["config"],
                "checks": checks,
                "passed": all(checks.values()),
                "late_median_da": row["late_median_da"],
                "da_tstat_vs_coinflip": row["da_tstat_vs_coinflip"],
                "late_median_rankic": row["late_median_rankic"],
                "rankic_tstat_vs_zero": row["rankic_tstat_vs_zero"],
                "late_median_mape": row["late_median_mape"],
                "min_window_da": row["min_window_da"],
                "min_window_daily_rank_ic": row["min_window_daily_rank_ic"],
            }
        )
    survivors = [item["config"] for item in stage_a if item["passed"]]
    # Degraded mode: when the DA floors eliminate every arm (the expected
    # outcome under strong market noise - DA hugs the coin flip everywhere),
    # Stage A loses discriminative power. The preregistered fallback gates
    # health on the RankIC t-statistic alone (the one return-space signal that
    # does materialise) and records that the degrade happened.
    degraded = not survivors
    if degraded:
        ic_floor = STAGE_A_FLOORS["rankic_tstat_vs_zero"]
        for item in stage_a:
            item["passed"] = bool(
                float(item["rankic_tstat_vs_zero"]) >= ic_floor
            )
        survivors = [item["config"] for item in stage_a if item["passed"]]
    by_config = {row["config"]: row for row in summary}

    def ranking(key: str, *, reverse: bool = True) -> list[dict[str, Any]]:
        ordered = sorted(
            survivors,
            key=lambda config: float(by_config[config][key]),
            reverse=reverse,
        )
        return [
            {
                "rank": index,
                "config": config,
                "value": float(by_config[config][key]),
            }
            for index, config in enumerate(ordered, start=1)
        ]

    return {
        "policy": (
            "Ordered screen, not a weighted score. Stage A applies "
            "codebook-invariant quality floors on every window; Stage B ranks "
            "survivors on capacity-normalized behaviour; Stage C reports "
            "tokenizer sufficiency. A reviewer still records the decision."
        ),
        "stage_a_floors": STAGE_A_FLOORS,
        "stage_a_degraded_to_rankic": degraded,
        "stage_a": stage_a,
        "stage_a_survivors": survivors,
        "stage_b_normalized_behaviour": {
            "p10_daily_codebook_balance_score": ranking(
                "p10_daily_codebook_balance_score"
            ),
            "late_median_collapse_alignment": ranking(
                "late_median_collapse_alignment"
            ),
            "late_median_unique_token_alignment": ranking(
                "late_median_unique_token_alignment"
            ),
            "late_median_distribution_alignment": ranking(
                "late_median_distribution_alignment"
            ),
        },
        "stage_c_tokenizer_sufficiency": {
            "tokenizer_mae": ranking("tokenizer_mae", reverse=False),
            "tokenizer_coarse_utilization": ranking(
                "tokenizer_coarse_utilization"
            ),
            "tokenizer_coarse_effective_codes": ranking(
                "tokenizer_coarse_effective_codes"
            ),
        },
    }


def correlations(
    rows: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = (
        "avg_da_per_date",
        "avg_daily_rank_ic",
        "avg_mape",
        "median_daily_collapse_alignment",
        "median_daily_unique_token_alignment",
        "median_daily_codebook_balance_score",
        "avg_ampratio",
        "p90_daily_collapse_rate",
        "median_daily_unique_tokens",
    )
    loss: dict[str, Any] = {}
    for metric in metrics:
        pairs = [
            (finite(row, "val_loss"), finite(row, metric)) for row in rows
        ]
        valid = [
            (float(left), float(right))
            for left, right in pairs
            if left is not None and right is not None
        ]
        loss[metric] = safe_spearman(
            [item[0] for item in valid],
            [item[1] for item in valid],
        )
    tokenizer_mae = [float(row["tokenizer_mae"]) for row in summary]
    coarse_utilization = [
        float(row["tokenizer_coarse_utilization"]) for row in summary
    ]
    downstream_metrics = (
        "late_median_da",
        "late_median_rankic",
        "late_median_mape",
        "min_window_da",
        "min_window_daily_rank_ic",
        "late_median_collapse_alignment",
        "late_median_unique_token_alignment",
        "late_median_distribution_alignment",
        "late_median_codebook_balance",
        "p10_daily_codebook_balance_score",
        "late_median_p90_collapse",
        "late_median_unique",
        "late_median_ampratio",
    )
    downstream = {
        metric: safe_spearman(
            tokenizer_mae, [float(row[metric]) for row in summary]
        )
        for metric in downstream_metrics
    }
    utilization = {
        metric: safe_spearman(
            coarse_utilization, [float(row[metric]) for row in summary]
        )
        for metric in downstream_metrics
    }
    information = {}
    available = [
        row
        for row in summary
        if finite(row, "late_coarse_mi_bits") is not None
    ]
    if len(available) >= 3:
        mi = [float(row["late_coarse_mi_bits"]) for row in available]
        for metric in ("late_median_da", "late_median_rankic"):
            information[metric] = safe_spearman(
                mi, [float(row[metric]) for row in available]
            )
    return {
        "val_loss_vs_epoch_metrics": loss,
        "tokenizer_mae_vs_late_downstream": downstream,
        "coarse_utilization_vs_late_downstream": utilization,
        "coarse_predictive_information_vs_late_quality": information,
    }


def plot_trajectories(
    plot_dir: Path, rows: list[dict[str, Any]]
) -> None:
    configs = sorted({str(row["config"]) for row in rows})
    colours = plt.get_cmap("tab10")
    quality = (
        ("avg_da_per_date", "Mean daily DA (%)", 100.0),
        ("avg_daily_rank_ic", "Mean daily RankIC", 1.0),
        ("avg_mape", "MAPE (%)", 1.0),
    )
    behaviour = (
        (
            "median_daily_collapse_alignment",
            "Collapse alignment vs target",
            1.0,
        ),
        (
            "median_daily_unique_token_alignment",
            "Unique-support alignment vs target",
            1.0,
        ),
        ("avg_ampratio", "AmpRatio", 1.0),
    )
    raw_behaviour = (
        ("p90_daily_collapse_rate", "P90 daily Collapse (%)", 100.0),
        ("median_daily_unique_tokens", "Median daily Unique", 1.0),
        (
            "median_daily_target_n_unique_tokens",
            "Median daily target Unique",
            1.0,
        ),
    )
    for filename, title, definitions in (
        ("quality_trajectories.png", "Prediction quality", quality),
        (
            "behaviour_trajectories.png",
            "Capacity-normalized prediction behaviour",
            behaviour,
        ),
        (
            "raw_behaviour_trajectories.png",
            "Raw behaviour counts (descriptive only)",
            raw_behaviour,
        ),
    ):
        figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
        for index, config in enumerate(configs):
            group = sorted(
                [row for row in rows if row["config"] == config],
                key=lambda row: int(row["epoch"]),
            )
            for axis, (metric, label, scale) in zip(axes, definitions):
                axis.plot(
                    [row["epoch"] for row in group],
                    [float(row[metric]) * scale for row in group],
                    label=config,
                    color=colours(index % 10),
                )
                axis.set_ylabel(label)
        if filename.startswith("quality"):
            axes[0].axhline(50, color="0.5", linestyle="--", linewidth=1)
        elif filename.startswith("raw_"):
            axes[0].axhline(35, color="0.5", linestyle="--", linewidth=1)
        else:
            axes[2].axhline(1, color="0.5", linestyle="--", linewidth=1)
        axes[-1].set_xlabel("GPT epoch")
        axes[0].legend(ncol=3, fontsize=8)
        figure.suptitle(f"Exp 02: {title}")
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)


def plot_tokenizer_grid(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    """Tokenizer-side grid: reconstruction plus how much codebook is filled.

    Exp 01's blind spot was that only joint-code aggregates were exported, so a
    badly under-filled coarse layer stayed invisible. The coarse/fine
    utilization panels make that failure mode explicit at this stage, where the
    encoder capacity is the variable under test.
    """
    embeddings = sorted({int(row["embedding_dim"]) for row in summary})
    hiddens = sorted({int(row["hidden_dim"]) for row in summary})
    shape = (len(hiddens), len(embeddings))
    panels = (
        ("tokenizer_mae", "Validation reconstruction MAE", 1.0, ".4f"),
        (
            "tokenizer_coarse_utilization",
            "Coarse-code utilization (%)",
            100.0,
            ".1f",
        ),
        (
            "tokenizer_fine_utilization",
            "Fine-code utilization (%)",
            100.0,
            ".1f",
        ),
        (
            "tokenizer_coarse_effective_codes",
            "Coarse effective codes (2^H)",
            1.0,
            ".1f",
        ),
    )
    matrices = {}
    for key, _, scale, _fmt in panels:
        matrix = np.full(shape, np.nan)
        for row in summary:
            i = hiddens.index(int(row["hidden_dim"]))
            j = embeddings.index(int(row["embedding_dim"]))
            matrix[i, j] = float(row[key]) * scale
        matrices[key] = matrix
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (key, title, _scale, fmt) in zip(axes.ravel(), panels):
        matrix = matrices[key]
        # Reconstruction error is better when small; utilization and effective
        # code counts are better when large.
        cmap = "viridis_r" if key == "tokenizer_mae" else "viridis"
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(embeddings)), embeddings)
        axis.set_yticks(range(len(hiddens)), hiddens)
        axis.set_xlabel("Embedding dimension")
        axis.set_ylabel("Hidden dimension")
        axis.set_title(title)
        midpoint = float(np.nanmedian(matrix))
        for i in range(shape[0]):
            for j in range(shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value):
                    # The grid is intentionally ragged: the capacity-extension
                    # tier does not cover every (embedding, hidden) pair.
                    axis.text(
                        j, i, "n/a", ha="center", va="center",
                        color="0.4", fontsize=8,
                    )
                    continue
                axis.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        "Exp 02 tokenizer-side evidence: reconstruction and codebook occupancy"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "tokenizer_grid.png", dpi=180)
    plt.close(figure)


def plot_late_dashboard(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    """Mature comparison on codebook-invariant quality and worst-case floors."""
    ordered = sorted(summary, key=lambda row: str(row["config"]))
    labels = [str(row["config"]) for row in ordered]
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0, False),
        ("late_median_rankic", "Late RankIC", 1.0, False),
        ("late_median_mape", "Late MAPE (%)", 1.0, True),
        ("min_window_da", "Worst-window DA (%)", 100.0, False),
        ("min_window_daily_rank_ic", "Worst-window RankIC", 1.0, False),
        ("late_median_ampratio", "Late AmpRatio", 1.0, None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (metric, title, scale, lower_better) in zip(
        axes.ravel(), definitions
    ):
        values = [float(row[metric]) * scale for row in ordered]
        bars = axis.bar(labels, values, color=plt.get_cmap("tab10").colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        if lower_better is None:
            axis.axhline(1, color="0.4", linestyle="--", linewidth=1)
            best_index = int(np.argmin(np.abs(np.asarray(values) - 1)))
        elif lower_better:
            best_index = int(np.argmin(values))
        else:
            best_index = int(np.argmax(values))
        bars[best_index].set_edgecolor("black")
        bars[best_index].set_linewidth(2)
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
        "Exp 02 mature-checkpoint comparison (last 10 epochs; no composite)"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "late_metric_dashboard.png", dpi=180)
    plt.close(figure)


def plot_normalized_behaviour(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    """Prediction behaviour read against the same-day target distribution.

    The paired bars in the first row show why raw counts mislead: the target's
    own diversity and collapse move with how much codebook each architecture
    fills, so only the ratio between prediction and target is comparable.
    """
    ordered = sorted(summary, key=lambda row: str(row["config"]))
    labels = [str(row["config"]) for row in ordered]
    positions = np.arange(len(ordered), dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    paired = (
        (
            axes[0, 0],
            "Daily unique tokens",
            "late_median_unique",
            "late_median_target_unique",
            1.0,
        ),
        (
            axes[0, 1],
            "Daily effective tokens (2^H)",
            "late_median_pred_effective",
            "late_median_target_effective",
            1.0,
        ),
        (
            axes[0, 2],
            "Daily top-1 share (%)",
            "late_median_collapse",
            "late_median_target_collapse",
            100.0,
        ),
    )
    for axis, title, predicted_key, target_key, scale in paired:
        axis.bar(
            positions - 0.2,
            [float(row[predicted_key]) * scale for row in ordered],
            width=0.4,
            label="prediction",
            color="#4c78a8",
        )
        axis.bar(
            positions + 0.2,
            [float(row[target_key]) * scale for row in ordered],
            width=0.4,
            label="target",
            color="#f58518",
        )
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_title(title)
        axis.legend(fontsize=8)

    normalized = (
        (
            axes[1, 0],
            "Collapse / unique / effective alignment",
            (
                ("late_median_collapse_alignment", "collapse"),
                ("late_median_unique_token_alignment", "unique"),
                ("late_median_effective_token_alignment", "effective"),
            ),
        ),
        (
            axes[1, 1],
            "Distribution alignment and support F1",
            (
                ("late_median_distribution_alignment", "1 - JSD"),
                ("late_median_token_support_f1", "support F1"),
            ),
        ),
        (
            axes[1, 2],
            "Codebook balance (median and worst decile)",
            (
                ("late_median_codebook_balance", "median day"),
                ("p10_daily_codebook_balance_score", "p10 day"),
            ),
        ),
    )
    for axis, title, series in normalized:
        width = 0.8 / len(series)
        for index, (key, label) in enumerate(series):
            offset = (index - (len(series) - 1) / 2) * width
            axis.bar(
                positions + offset,
                [float(row[key]) for row in ordered],
                width=width,
                label=label,
            )
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Exp 02 capacity-normalized behaviour: prediction versus same-day target"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "normalized_behaviour.png", dpi=180)
    plt.close(figure)


def plot_predictive_information(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    """Bits of target structure the GPT explains, which is vocabulary-invariant.

    Exp 01 showed this quantity saturates: a 64x larger joint vocabulary bought
    essentially no additional learnable information. At fixed bits, Exp 02 asks
    the sharper question of whether better codebook occupancy converts into more
    explained bits or merely into a longer unpredictable tail.
    """
    available = [
        row
        for row in summary
        if finite(row, "late_coarse_mi_bits") is not None
    ]
    if not available:
        return
    ordered = sorted(available, key=lambda row: str(row["config"]))
    labels = [str(row["config"]) for row in ordered]
    positions = np.arange(len(ordered), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(
        positions - 0.2,
        [float(row["target_coarse_entropy_bits"]) for row in ordered],
        width=0.4,
        label="H(coarse target)",
        color="#f58518",
    )
    axes[0].bar(
        positions + 0.2,
        [float(row["late_coarse_ce_bits"]) for row in ordered],
        width=0.4,
        label="validation CE",
        color="#4c78a8",
    )
    axes[0].set_xticks(positions, labels, rotation=35)
    axes[0].set_ylabel("bits / token")
    axes[0].set_title("Coarse target entropy vs cross-entropy")
    axes[0].legend(fontsize=8)

    for axis, keys, title in (
        (
            axes[1],
            (("late_coarse_mi_bits", "coarse"), ("late_joint_mi_bits", "joint")),
            "Explained information (bits / token)",
        ),
        (
            axes[2],
            (("late_coarse_mi_fraction", "coarse MI / H"),),
            "Explained fraction of coarse entropy",
        ),
    ):
        width = 0.8 / len(keys)
        for index, (key, label) in enumerate(keys):
            offset = (index - (len(keys) - 1) / 2) * width
            axis.bar(
                positions + offset,
                [
                    finite(row, key) if finite(row, key) is not None else 0.0
                    for row in ordered
                ],
                width=width,
                label=label,
            )
        axis.set_xticks(positions, labels, rotation=35)
        axis.set_title(title)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Exp 02 predictive information: vocabulary-invariant learning evidence"
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "predictive_information.png", dpi=180)
    plt.close(figure)


def plot_reversal(
    plot_dir: Path, summary: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    definitions = (
        ("late_median_da", "Late DA (%)", 100.0),
        ("late_median_rankic", "Late RankIC", 1.0),
        ("late_median_p90_collapse", "Late P90 Collapse (%)", 100.0),
    )
    for axis, (metric, label, scale) in zip(axes, definitions):
        for row in summary:
            x = float(row["tokenizer_mae"])
            y = float(row[metric]) * scale
            axis.scatter(x, y, s=55)
            axis.annotate(
                str(row["config"]),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Tokenizer reconstruction MAE")
        axis.set_ylabel(label)
    figure.suptitle("Does tokenizer reconstruction predict GPT behaviour?")
    figure.tight_layout()
    figure.savefig(plot_dir / "tokenizer_vs_downstream.png", dpi=180)
    plt.close(figure)


def plot_pareto(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Trade-off view on the normalized behaviour axis, not the raw count."""
    configs = sorted({str(row["config"]) for row in rows})
    colours = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(10, 7))
    for index, config in enumerate(configs):
        group = [row for row in rows if row["config"] == config]
        axis.scatter(
            [row["median_daily_collapse_alignment"] for row in group],
            [row["avg_da_per_date"] * 100 for row in group],
            s=18,
            alpha=0.5,
            label=config,
            color=colours(index % 10),
        )
    front = [row for row in rows if row["joint_pareto"]]
    axis.scatter(
        [row["median_daily_collapse_alignment"] for row in front],
        [row["avg_da_per_date"] * 100 for row in front],
        s=85,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="6D joint Pareto",
    )
    axis.set_xlabel(
        "Collapse alignment vs same-day target (1.0 = identical top-1 share)"
    )
    axis.set_ylabel("Mean daily DA (%)")
    axis.set_title("Exp 02 quality / normalized-behaviour trade-off")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "da_vs_collapse_pareto.png", dpi=180)
    plt.close(figure)


def format_optional(value: Any, spec: str) -> str:
    number = value if isinstance(value, (int, float)) else None
    if number is None or not math.isfinite(float(number)):
        return "n/a"
    return format(float(number), spec)


def render_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    correlation_payload: dict[str, Any],
    screen: dict[str, Any],
) -> str:
    bits_l1 = manifest["settings"]["bits_l1"]
    bits_l2 = manifest["settings"]["bits_l2"]
    lines = [
        "# Exp 02 tokenizer architecture rerun",
        "",
        f"- Status: **{manifest.get('status')}**",
        f"- Bits inherited from Exp 01: **{bits_l1}+{bits_l2}** "
        f"(coarse {2 ** int(bits_l1)}, fine {2 ** int(bits_l2)}, "
        f"joint {2 ** (int(bits_l1) + int(bits_l2)):,})",
        f"- Configurations: **{len(summary)}**",
        f"- Evaluated GPT checkpoints: **{len(rows)}**",
        f"- Health-passing checkpoints: **"
        f"{sum(bool(row.get('healthy')) for row in rows)}**",
        "",
        "The bit split is identical across arms, so DA, daily RankIC, and MAPE "
        "are directly comparable. Behaviour is not: each architecture fills the "
        "fixed codebook to a different degree, which moves the *target* "
        "distribution as well as the prediction. Behaviour is therefore judged "
        "by alignment against the same-day target, and the raw counts are kept "
        "only as context.",
        "",
        "## Group 1 - codebook-invariant quality (comparable across arms)",
        "",
        "| Config | Late DA | Worst-window DA | Late RankIC | "
        "Worst-window RankIC | Late MAPE | Baseline MAPE | Late amp "
        "[min, max] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['config']} | {row['late_median_da'] * 100:.2f}% | "
            f"{row['min_window_da'] * 100:.2f}% | "
            f"{row['late_median_rankic']:.4f} | "
            f"{row['min_window_daily_rank_ic']:.4f} | "
            f"{row['late_median_mape']:.3f} | "
            f"{row['late_median_baseline_mape']:.3f} | "
            f"{row['late_median_ampratio']:.3f} "
            f"[{row['min_window_ampratio']:.3f}, "
            f"{row['max_window_ampratio']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "`Baseline MAPE` is the no-change forecast. The per-date DA "
            "baseline used elsewhere is an oracle that already knows each "
            "day's majority direction, so a negative `da_above_baseline` is "
            "expected and is not a failure signal at this stage.",
            "",
            "## Group 2 - capacity-normalized behaviour",
            "",
            "| Config | Collapse align | Unique align | Effective align | "
            "1 - JSD | Support F1 | Balance (median) | Balance (p10) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['config']} | "
            f"{row['late_median_collapse_alignment']:.4f} | "
            f"{row['late_median_unique_token_alignment']:.4f} | "
            f"{row['late_median_effective_token_alignment']:.4f} | "
            f"{row['late_median_distribution_alignment']:.4f} | "
            f"{row['late_median_token_support_f1']:.4f} | "
            f"{row['late_median_codebook_balance']:.4f} | "
            f"{row['p10_daily_codebook_balance_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Config | Pred unique | Target unique | Pred eff | Target eff | "
            "Pred top-1 | Target top-1 | P90 collapse | Worst collapse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['config']} | {row['late_median_unique']:.1f} | "
            f"{row['late_median_target_unique']:.1f} | "
            f"{row['late_median_pred_effective']:.2f} | "
            f"{row['late_median_target_effective']:.2f} | "
            f"{row['late_median_collapse'] * 100:.1f}% | "
            f"{row['late_median_target_collapse'] * 100:.1f}% | "
            f"{row['late_median_p90_collapse'] * 100:.1f}% | "
            f"{row['late_median_worst_collapse'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Group 3 - tokenizer sufficiency and codebook occupancy",
            "",
            "| Config | Tok MAE | Tok RMSE | Coarse used | Coarse util | "
            "Coarse eff | Fine used | Fine util | Joint used | Joint util |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['config']} | {row['tokenizer_mae']:.4f} | "
            f"{row['tokenizer_rmse']:.4f} | "
            f"{int(row['tokenizer_coarse_unique'])} | "
            f"{row['tokenizer_coarse_utilization'] * 100:.1f}% | "
            f"{row['tokenizer_coarse_effective_codes']:.1f} | "
            f"{int(row['tokenizer_fine_unique'])} | "
            f"{row['tokenizer_fine_utilization'] * 100:.1f}% | "
            f"{int(row['tokenizer_joint_unique'])} | "
            f"{row['tokenizer_joint_utilization'] * 100:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Group 4 - predictive information (vocabulary-invariant)",
            "",
            "`MI = H(target) - CE`, in bits per token. Exp 01 found this "
            "quantity saturated near 1.3 bits across a 64x range of joint "
            "vocabularies, so it is the reference for deciding whether extra "
            "codebook occupancy is signal or tail noise.",
            "",
            "| Config | H coarse | CE coarse | MI coarse | MI / H | "
            "H fine | CE fine | MI fine | MI joint |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['config']} | "
            f"{format_optional(row['target_coarse_entropy_bits'], '.3f')} | "
            f"{format_optional(row['late_coarse_ce_bits'], '.3f')} | "
            f"{format_optional(row['late_coarse_mi_bits'], '.3f')} | "
            f"{format_optional(row['late_coarse_mi_fraction'], '.3f')} | "
            f"{format_optional(row['target_fine_entropy_bits'], '.3f')} | "
            f"{format_optional(row['late_fine_ce_bits'], '.3f')} | "
            f"{format_optional(row['late_fine_mi_bits'], '.3f')} | "
            f"{format_optional(row['late_joint_mi_bits'], '.3f')} |"
        )
    lines.extend(
        [
            "",
            "## Staged screen (ordered shortlist, not a score)",
            "",
            f"- Stage-A floors: `{screen['stage_a_floors']}`",
            "",
            "| Config | Late DA >= 50% | DA t-stat >= 2 | RankIC t-stat >= 2 | "
            "Stage A |",
            "|---|:--:|:--:|:--:|:--:|",
        ]
    )
    for item in screen["stage_a"]:
        checks = item["checks"]
        lines.append(
            f"| {item['config']} | "
            f"{'pass' if checks['late_median_da'] else 'fail'} "
            f"({item['late_median_da'] * 100:.2f}%) | "
            f"{'pass' if checks['da_tstat_vs_coinflip'] else 'fail'} "
            f"(t={item['da_tstat_vs_coinflip']:.1f}) | "
            f"{'pass' if checks['rankic_tstat_vs_zero'] else 'fail'} "
            f"(t={item['rankic_tstat_vs_zero']:.1f}) | "
            f"{'**pass**' if item['passed'] else 'fail'} |"
        )
    survivors = screen["stage_a_survivors"]
    lines.extend(
        [
            "",
            f"- Stage-A survivors: **{', '.join(survivors) or 'none'}**",
            "",
        ]
    )
    if survivors:
        lines.append(
            "Stage-B ranking of the survivors on capacity-normalized "
            "behaviour (independent rankings, deliberately not merged):"
        )
        lines.append("")
        for metric, ranked in screen["stage_b_normalized_behaviour"].items():
            order = " > ".join(
                f"{item['config']} ({item['value']:.4f})" for item in ranked
            )
            lines.append(f"- `{metric}`: {order}")
        lines.append("")
        lines.append("Stage-C tokenizer sufficiency among the survivors:")
        lines.append("")
        for metric, ranked in screen["stage_c_tokenizer_sufficiency"].items():
            order = " > ".join(
                f"{item['config']} ({item['value']:.4f})" for item in ranked
            )
            lines.append(f"- `{metric}`: {order}")
        lines.append("")
    else:
        lines.extend(
            [
                "No arm cleared Stage A. Do not promote a selection from "
                "Stage-B or Stage-C rankings alone; review the quality group "
                "and decide whether the floors or the recipe must change.",
                "",
            ]
        )
    lines.extend(
        [
            "## Figures",
            "",
            "- `plots/tokenizer_grid.png`: reconstruction and per-layer "
            "codebook occupancy.",
            "- `plots/quality_trajectories.png`: DA, RankIC, and MAPE by epoch.",
            "- `plots/behaviour_trajectories.png`: normalized alignment and "
            "amplitude by epoch.",
            "- `plots/raw_behaviour_trajectories.png`: raw counts, descriptive "
            "only.",
            "- `plots/normalized_behaviour.png`: prediction versus same-day "
            "target, plus the alignment and balance summary.",
            "- `plots/predictive_information.png`: bits of target structure "
            "explained.",
            "- `plots/late_metric_dashboard.png`: mature last-10-epoch "
            "comparison including worst-window floors.",
            "- `plots/tokenizer_vs_downstream.png`: reconstruction/downstream "
            "reversal.",
            "- `plots/da_vs_collapse_pareto.png`: non-composite trade-off.",
            "",
            "## Correlation audit",
            "",
            "Raw Spearman results are stored in `analysis.json`, including "
            "reconstruction MAE and coarse-code utilization against every late "
            "downstream metric. Correlation is descriptive and is not used as "
            "a selection score.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--weights_root", type=Path, default=DEFAULT_WEIGHTS_ROOT
    )
    parser.add_argument(
        "--select_config",
        default="",
        help="Optional reviewed tokenizer config, for example 64x192",
    )
    parser.add_argument(
        "--rationale",
        default="",
        help="Required human-reviewed rationale when recording a selection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "study_manifest.json")
    rows = load_json(root / "combined_epoch_summary.json")
    if not rows:
        raise RuntimeError("No Exp 02 rows to analyze")
    require_fields(rows)
    annotated = annotate(rows)
    summary = envelopes(annotated, root)
    screen = staged_screen(summary)
    selected_config = args.select_config.strip()
    rationale = args.rationale.strip()
    known_configs = {str(row["config"]) for row in summary}
    if selected_config and selected_config not in known_configs:
        raise ValueError(
            f"Unknown --select_config {selected_config!r}; "
            f"expected one of {sorted(known_configs)}"
        )
    if selected_config and not rationale:
        raise ValueError("--rationale is required with --select_config")
    correlation_payload = correlations(annotated, summary)
    front = [
        row
        for row in annotated
        if row["quality_pareto"]
        or row["behaviour_pareto"]
        or row["robustness_pareto"]
        or row["joint_pareto"]
    ]
    analysis = {
        "experiment": "Exp 02 Tokenizer architecture rerun",
        "status": manifest.get("status"),
        "n_configs": len(summary),
        "n_checkpoints": len(annotated),
        "n_healthy": sum(bool(row.get("healthy")) for row in annotated),
        "objectives": {
            "quality": [list(item) for item in QUALITY_OBJECTIVES],
            "behaviour": [list(item) for item in BEHAVIOUR_OBJECTIVES],
            "robustness": [list(item) for item in ROBUSTNESS_OBJECTIVES],
            "joint": [list(item) for item in JOINT_OBJECTIVES],
            "descriptive_only": list(DESCRIPTIVE_BEHAVIOUR_FIELDS),
        },
        "pareto_counts": {
            "quality": sum(bool(row["quality_pareto"]) for row in annotated),
            "behaviour": sum(
                bool(row["behaviour_pareto"]) for row in annotated
            ),
            "robustness": sum(
                bool(row["robustness_pareto"]) for row in annotated
            ),
            "joint": sum(bool(row["joint_pareto"]) for row in annotated),
        },
        "config_envelopes": summary,
        "staged_screen": screen,
        "correlations": correlation_payload,
        "selection_policy": manifest["settings"]["evaluation"][
            "selection_policy"
        ],
        "weighted_score_used": False,
        "holdout_used": bool(
            manifest["settings"]["evaluation"].get("holdout_used", False)
        ),
    }
    atomic_write_json(root / "analysis.json", analysis)
    write_csv(root / "config_envelopes.csv", summary)
    write_csv(root / "pareto_front.csv", front)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(plot_dir, annotated)
    plot_tokenizer_grid(plot_dir, summary)
    plot_late_dashboard(plot_dir, summary)
    plot_normalized_behaviour(plot_dir, summary)
    plot_predictive_information(plot_dir, summary)
    plot_reversal(plot_dir, summary)
    plot_pareto(plot_dir, annotated)
    atomic_write_text(
        root / "ANALYSIS.md",
        render_report(
            manifest, annotated, summary, correlation_payload, screen
        ),
    )
    if selected_config:
        chosen = next(
            row for row in summary if row["config"] == selected_config
        )
        weights_root = args.weights_root.resolve()
        tokenizer_path = (
            weights_root
            / "configs"
            / (
                f"emb_{int(chosen['embedding_dim']):03d}_"
                f"hid_{int(chosen['hidden_dim']):03d}"
            )
            / "tokenizer.pt"
        )
        selection = {
            "experiment": "Exp 02 Tokenizer architecture sweep",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected": {
                **chosen,
                "bits_l1": int(manifest["settings"]["bits_l1"]),
                "bits_l2": int(manifest["settings"]["bits_l2"]),
                "tokenizer_path": str(tokenizer_path.resolve()),
                "tokenizer_weight_relative_path": tokenizer_path.relative_to(
                    weights_root
                ).as_posix(),
            },
            "selection_rule": (
                "Human-reviewed non-composite judgment. Stage A: "
                "codebook-invariant quality (DA, daily RankIC, MAPE) with "
                "per-window floors. Stage B: capacity-normalized behaviour "
                "alignment against the same-day target, not raw Collapse or "
                "raw Unique counts. Stage C: tokenizer reconstruction and "
                "per-layer codebook occupancy."
            ),
            "staged_screen": {
                "stage_a_floors": screen["stage_a_floors"],
                "stage_a_survivors": screen["stage_a_survivors"],
                "selected_passed_stage_a": selected_config
                in screen["stage_a_survivors"],
            },
            "rationale": rationale,
            "human_review_recorded": True,
            "upstream_eligible": True,
            "holdout_used": False,
        }
        atomic_write_json(root / "selection.json", selection)
        if selected_config not in screen["stage_a_survivors"]:
            print(
                f"WARNING: {selected_config} did not clear the Stage-A "
                "quality floors; the recorded rationale must justify the "
                "override explicitly."
            )
    print(
        f"Analyzed {len(annotated)} checkpoints across {len(summary)} configs. "
        f"Stage-A survivors: "
        f"{', '.join(screen['stage_a_survivors']) or 'none'}. "
        f"Outputs: {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
