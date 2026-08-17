"""PT-01 analysis: per-arm per-date metrics + moving-block bootstrap contrasts.

Vectorized over numpy arrays (no per-row dicts).  Computes daily RankIC / DA /
MAE for each J arm, then paired circular moving-block bootstrap (L=5/10/20) for
the J1-J5 vs J0 contrasts.  Writes a summary JSON to results root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import resolve_roots, artifact_paths, write_json  # noqa: E402
from common import daily_rank_ic, _contrast  # noqa: E402

ARMS = {
    "J0_greedy": "greedy_return",
    "J1_joint_map": "map_return",
    "J2_posterior_mean": "post_mean",
    "J3_posterior_median": "post_median",
    "J4_p_up": "p_up",
}


def analyze(records_path, dense_min=3634, out_path=None):
    r = np.load(records_path, allow_pickle=True)
    dates = np.asarray(r["date_key"])
    true = np.asarray(r["true_logret"], dtype=np.float64)
    offset = np.asarray(r["offset"], dtype=np.int64)
    uniq_dates = np.unique(dates)
    per_arm = {}
    for arm, field in ARMS.items():
        score = np.asarray(r[field], dtype=np.float64)
        ic, da, mae, cnt, dense = daily_rank_ic(score, true, dates, dense_min)
        if arm == "J4_p_up":
            # direction-probability arm: DA uses the 0.5 threshold, not sign()
            da = _da_direction(score, true, dates, dense_min)
        per_arm[arm] = {
            "avg_daily_rank_ic": float(np.nanmean(ic[dense])),
            "avg_da_per_date": float(np.nanmean(da[dense])),
            "avg_mae": float(np.nanmean(mae[dense])),
            "n_dense_dates": int(dense.sum()),
            "daily_rank_ic": ic,
            "daily_da": da,
            "daily_mae": mae,
            "date_keys": uniq_dates,
        }

    # contrasts vs J0 (RankIC), full 400-day
    base_ic = per_arm["J0_greedy"]["daily_rank_ic"]
    base_mae = per_arm["J0_greedy"]["daily_mae"]
    contrasts = {}
    for arm in ("J1_joint_map", "J2_posterior_mean", "J3_posterior_median", "J4_p_up"):
        contrasts[arm] = _contrast(base_ic, per_arm[arm]["daily_rank_ic"],
                                   base_mae, per_arm[arm]["daily_mae"])

    # offset-scoped confirmation: development 0..299 vs confirmation 300..399
    scope = {}
    for arm in ("J3_posterior_median", "J2_posterior_mean", "J4_p_up"):
        scope[arm] = {}
        for name, mask in (("dev_0_299", offset < 300),
                           ("confirmation_300_399", (offset >= 300) & (offset < 400))):
            d = dates[mask]
            t = true[mask]
            s = np.asarray(r[ARMS[arm]], dtype=np.float64)[mask]
            ic, da, mae, cnt, dense = daily_rank_ic(s, t, d, dense_min)
            s0 = np.asarray(r["greedy_return"], dtype=np.float64)[mask]
            ic0, _, mae0, _, _ = daily_rank_ic(s0, t, d, dense_min)
            c = _contrast(ic0, ic, mae0, mae, n_replicates=5000)
            c["avg_daily_rank_ic"] = float(np.nanmean(ic[dense]))
            c["avg_j0_rank_ic"] = float(np.nanmean(ic0[dense]))
            scope[arm][name] = c

    summary = {
        "schema_version": "pt01-analysis-v2",
        "dense_min": dense_min,
        "n_rows": int(len(true)),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk not in
                     ("daily_rank_ic", "daily_da", "daily_mae", "date_keys")}
                 for k, v in per_arm.items()},
        "contrasts_vs_J0": contrasts,
        "offset_scopes": scope,
    }
    if out_path:
        write_json(out_path, summary)
    series = {arm: {"dates": per_arm[arm]["date_keys"].tolist(),
                    "rank_ic": np.nan_to_num(per_arm[arm]["daily_rank_ic"]).tolist(),
                    "da": np.nan_to_num(per_arm[arm]["daily_da"]).tolist()}
              for arm in ARMS}
    if out_path:
        write_json(Path(str(out_path).replace(".json", "_series.json")), series)
    return summary


def _da_direction(score, true, dates, dense_min):
    """Per-date DA for a direction-probability score (threshold 0.5)."""
    uniq, inv = np.unique(dates, return_inverse=True)
    n = len(uniq)
    da = np.full(n, np.nan)
    for i in range(n):
        m = inv == i
        if m.sum() < dense_min:
            continue
        da[i] = np.mean((score[m] > 0.5) == (true[m] > 0))
    return da


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=None)
    ap.add_argument("--dense_min", type=int, default=3634)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    roots = resolve_roots()
    paths = artifact_paths(roots=roots)
    records = args.records or str(paths["records"])
    out = args.out or (roots.results_root / "A-decode" / "analysis.json")
    s = analyze(records, dense_min=args.dense_min, out_path=Path(out))
    print("=== PT-01 arms ===")
    for arm, v in s["arms"].items():
        print(f"  {arm}: RankIC={v['avg_daily_rank_ic']:.5f} DA={v['avg_da_per_date']:.5f} MAE={v['avg_mae']:.5f}")
    print("=== contrasts vs J0 ===")
    for arm, v in s["contrasts_vs_J0"].items():
        print(f"  {arm}: delta={v['rank_ic_delta_vs_J0']:+.5f} robust={v['block_robust']} "
              f"CI5={v['block_cis']['5']['ci_lower']:.5f}..{v['block_cis']['5']['ci_upper']:.5f}")


if __name__ == "__main__":
    main()
