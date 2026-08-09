"""PT-06: uncertainty, abstention (selective prediction) and output ensemble.

Uncertainty sources are model-implied posterior dispersion (aleatoric proxy):
  q1 = |E[r]| / sqrt(Var(r) + eps)
  q2 = max(P(up), 1 - P(up))
and (optionally) model-diversity between head outputs (epistemic proxy).

Abstention: within each date, keep the top-coverage fraction of stocks by a
pre-registered confidence, and report acted DA/RankIC at coverage
100/80/60/40/20% together with the baseline recomputed on the same acted
subset.  Thresholds are chosen on pre-cutoff calibration data only.

Ensemble: output-space averaging of J3 posterior median and any trained rank
head score; stacking weights fitted on the pre-cutoff calibration slice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json  # noqa: E402
from analyze_pt01 import daily_rank_ic  # noqa: E402

COVERAGES = (100, 80, 60, 40, 20)


def confidence_metrics(records):
    """Per-row confidence proxies from the exact posterior."""
    mean = records["post_mean"]
    std = records["post_std"]
    p_up = records["p_up"]
    q1 = np.abs(mean) / np.sqrt(np.maximum(std ** 2, 1e-12))
    q2 = np.maximum(p_up, 1.0 - p_up)
    return {"q1": q1, "q2": q2}


def abstain_curve(records, score_field, conf, dense_min=3634, max_cs=None):
    """Acted DA/RankIC vs coverage using the given confidence array.

    For each coverage, within each date keep the top-coverage fraction of stocks
    by ``conf`` and compute DA/RankIC on the acted subset.  The baseline
    (score_field) is recomputed on the identical acted subset.  The dense
    threshold scales with coverage (an acted subset is naturally smaller).
    """
    dates = np.asarray(records["date_key"])
    score = np.asarray(records[score_field])
    true = np.asarray(records["true_logret"])
    conf = np.asarray(conf)
    uniq, inv = np.unique(dates, return_inverse=True)
    max_cs = max_cs or int(max(np.bincount(inv)))
    out = {}
    for cov in COVERAGES:
        acted = np.zeros(len(dates), dtype=bool)
        for i in range(len(uniq)):
            m = inv == i
            idx = np.where(m)[0]
            if len(idx) < dense_min:
                continue
            k = max(1, int(round(len(idx) * cov / 100.0)))
            order = np.argsort(-conf[idx], kind="stable")
            acted[idx[order[:k]]] = True
        s_act = score[acted]
        t_act = true[acted]
        d_act = dates[acted]
        # dense threshold proportional to the acted subset size at this coverage
        acted_min = max(5, int(np.ceil(0.8 * max_cs * cov / 100.0)))
        ic, da, mae, cnt, dense = daily_rank_ic(s_act, t_act, d_act, acted_min)
        out[str(cov)] = {
            "coverage": cov / 100.0,
            "n_acted": int(acted.sum()),
            "acted_dense_min": acted_min,
            "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
            "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        }
    return out


def ensemble_outputs(records, scores, weights=None, name="ensemble"):
    """Weighted output-space ensemble over score columns (same rows).

    ``scores``: dict name -> array (per-row score, comparable scale).  If
    ``weights`` is None, equal weights after per-column standardization.
    """
    keys = list(scores.keys())
    arrs = np.stack([np.asarray(scores[k], dtype=np.float64) for k in keys])
    # standardize each member to unit std for scale-free combination
    std = arrs.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    z = (arrs - arrs.mean(axis=1, keepdims=True)) / std
    if weights is None:
        w = np.full(len(keys), 1.0 / len(keys))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    return z.T @ w


def run_abstention(pt01_records_path, dense_min=3634, out_path=None):
    rec = np.load(pt01_records_path, allow_pickle=True)
    conf = confidence_metrics(rec)
    result = {"schema_version": "pt06-v1", "dense_min": dense_min}
    for arm, field in (("J3_median", "post_median"), ("J2_mean", "post_mean"),
                       ("J4_p_up", "p_up"), ("J0_greedy", "greedy_return")):
        result[arm] = {
            "confidence_q1": abstain_curve(rec, field, conf["q1"], dense_min),
            "confidence_q2": abstain_curve(rec, field, conf["q2"], dense_min),
        }
    if out_path:
        write_json(out_path, result)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="server_runs/weights/06-posttrain/seed42/pt01_records.npz")
    ap.add_argument("--dense_min", type=int, default=3634)
    args = ap.parse_args()
    roots = resolve_roots()
    out = roots.results_root / "pt06_abstention.json"
    res = run_abstention(args.records, dense_min=args.dense_min, out_path=out)
    # print a compact summary
    for arm, v in res.items():
        if arm.startswith("J"):
            q1_100 = v["confidence_q1"].get("100", {}).get("avg_daily_rank_ic")
            q1_20 = v["confidence_q1"].get("20", {}).get("avg_daily_rank_ic")
            print(f"{arm}: RankIC @100%={q1_100:.5f} @20%={q1_20:.5f}")


if __name__ == "__main__":
    main()
