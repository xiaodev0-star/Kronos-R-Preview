"""PT-00D/§15 statistical comparison: paired circular moving-block bootstrap.

Pairing unit is the calendar date.  Cross-sectional stocks are NOT independent
time samples, so we never IID-bootstrap stock x date rows.

Procedure for a candidate-vs-reference contrast:
  1. intersect to the common date x stock_uid universe (and identical finite mask);
  2. compute the per-date metric on each side;
  3. per-date delta = candidate - reference;
  4. circular moving-block resample (block length L=5 main, L=10/20 sensitivity,
     10,000 replicates, fixed seed) over the contiguous date series;
  5. 95% percentile CI; significance = CI excludes 0 in the hypothesis direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap_utils import (  # noqa: E402
    circular_moving_block_bootstrap as _root_circular_moving_block_bootstrap,
)


def daily_metric_from_rows(rows, metric, dense_min=None):
    """Compute a per-date metric over per-row dicts.

    ``rows``: list of dicts with ``date_key`` and, depending on metric:
        - ``rank_ic``: needs ``rank_score`` and ``true_logret``
        - ``da``: needs ``direction_prob`` and ``true_logret``
        - ``mape``: needs ``pred_logret`` and ``true_logret``
        - ``rank_ic_mae``: needs ``pred_logret`` and ``true_logret`` (MAE on logret)
    Returns {date_key: value} restricted to dates with >= dense_min finite rows.
    """
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date_key"], []).append(r)
    out = {}
    for d, rs in by_date.items():
        rs = [r for r in rs if _finite(r)]
        if dense_min is not None and len(rs) < dense_min:
            continue
        if len(rs) < 2:
            continue
        if metric == "rank_ic":
            score = np.asarray([r["rank_score"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            if np.all(score == score[0]):
                out[d] = 0.0
            else:
                out[d] = spearmanr(score, tru)[0]
                if not np.isfinite(out[d]):
                    out[d] = 0.0
        elif metric == "da":
            prob = np.asarray([r["direction_prob"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean((prob > 0.5) == (tru > 0)))
        elif metric == "mape":
            pred = np.asarray([r["pred_logret"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean(np.abs(pred - tru)))
        elif metric == "mae":
            pred = np.asarray([r["pred_logret"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean(np.abs(pred - tru)))
        else:
            raise ValueError(f"unknown metric {metric}")
    return out


def _finite(r):
    if "true_logret" in r and not np.isfinite(r.get("true_logret")):
        return False
    for key in ("rank_score", "direction_prob", "pred_logret"):
        if key in r and r.get(key) is not None and not np.isfinite(r[key]):
            return False
    return True


def shared_universe(candidate_rows, reference_rows):
    """Intersect candidate/reference to common date x stock_uid set."""
    c = {(r["date_key"], r["stock_uid"]) for r in candidate_rows}
    r_ = {(r["date_key"], r["stock_uid"]) for r in reference_rows}
    common = c & r_
    c_by_key = {(r["date_key"], r["stock_uid"]): r for r in candidate_rows}
    r_by_key = {(r["date_key"], r["stock_uid"]): r for r in reference_rows}
    return [c_by_key[k] for k in common], [r_by_key[k] for k in common]


def circular_moving_block_bootstrap(
    deltas,
    block_length=5,
    n_replicates=10_000,
    seed=42,
):
    """Circular moving-block bootstrap over a contiguous date delta series.

    ``deltas`` must be ordered by date.  Samples contiguous circular blocks
    (start uniformly random, advance with wraparound).  Returns the replicate
    mean distribution.
    """
    # Canonical implementation lives in the root-level bootstrap_utils module
    # so every experiment package shares the exact same sampler.
    return _root_circular_moving_block_bootstrap(
        deltas,
        block_length=block_length,
        n_replicates=n_replicates,
        seed=seed,
    )


def paired_bootstrap_ci(
    candidate_rows,
    reference_rows,
    metric,
    dense_min=None,
    block_lengths=(5, 10, 20),
    n_replicates=10_000,
    seed=42,
    order_by_date=True,
):
    """Full paired comparison with moving-block CIs at several block lengths.

    Returns dict with point estimate (mean daily delta), per-block-length CI,
    directional significance, and block robustness.
    """
    cand, ref = shared_universe(candidate_rows, reference_rows)
    c_daily = daily_metric_from_rows(cand, metric, dense_min=dense_min)
    r_daily = daily_metric_from_rows(ref, metric, dense_min=dense_min)
    common_dates = sorted(set(c_daily) & set(r_daily))
    if len(common_dates) < 2:
        return {
            "metric": metric, "n_dates": len(common_dates),
            "point": None, "candidate_ci": None, "reference_ci": None,
            "error": "fewer than 2 common dense dates",
        }
    c_series = np.asarray([c_daily[d] for d in common_dates], dtype=float)
    r_series = np.asarray([r_daily[d] for d in common_dates], dtype=float)
    deltas = c_series - r_series
    point = float(deltas.mean())
    minimize = metric in ("mape", "mae")
    cis = {}
    for L in block_lengths:
        means = circular_moving_block_bootstrap(
            deltas, block_length=L, n_replicates=n_replicates, seed=seed)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        if minimize:
            signif = hi < 0.0
        else:
            signif = lo > 0.0
        cis[str(L)] = {
            "block_length": int(L), "ci_lower": lo, "ci_upper": hi,
            "significant_directional": bool(signif),
            "ci_excludes_zero": bool(lo <= 0.0 <= hi) is False,
        }
    block_robust = all(v["significant_directional"] for v in cis.values())
    return {
        "metric": metric,
        "n_dates": len(common_dates),
        "point": point,
        "minimize": minimize,
        "candidate_mean": float(c_series.mean()),
        "reference_mean": float(r_series.mean()),
        "block_cis": cis,
        "block_robust": block_robust,
        "offset_scope": "development_0_299_or_300_399_as_declared",
    }


def bootstrap_for_series(deltas, block_lengths=(5, 10, 20),
                         n_replicates=10_000, seed=42):
    """Convenience: CI for a raw per-date delta series (already computed)."""
    out = {}
    for L in block_lengths:
        means = circular_moving_block_bootstrap(
            deltas, block_length=L, n_replicates=n_replicates, seed=seed)
        out[str(L)] = {
            "block_length": int(L),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
        }
    return out
