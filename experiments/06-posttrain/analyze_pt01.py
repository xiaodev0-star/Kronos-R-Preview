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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json  # noqa: E402
from compare_posttrain import circular_moving_block_bootstrap  # noqa: E402

ARMS = {
    "J0_greedy": "greedy_return",
    "J1_joint_map": "map_return",
    "J2_posterior_mean": "post_mean",
    "J3_posterior_median": "post_median",
    "J4_p_up": "p_up",
}


def daily_rank_ic(score, true, dates, dense_min):
    uniq, inv = np.unique(dates, return_inverse=True)
    n = len(uniq)
    ic = np.full(n, np.nan)
    da = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = inv == i
        c = int(m.sum())
        cnt[i] = c
        if c < dense_min:
            continue
        s = score[m]
        t = true[m]
        if c >= 2 and not np.all(s == s[0]):
            ic[i] = spearmanr(s, t)[0]
        else:
            ic[i] = 0.0
        da[i] = np.mean((np.sign(s) > 0) == (t > 0))
        mae[i] = np.mean(np.abs(s - t))
    dense = cnt >= dense_min
    return ic, da, mae, cnt, dense


def _contrast(base_ic, arm_ic, base_mae, arm_mae, n_replicates=10000):
    common = np.isfinite(base_ic) & np.isfinite(arm_ic)
    d_ic = arm_ic[common] - base_ic[common]
    d_mae = arm_mae[common] - base_mae[common]
    cis = {}
    for L in (5, 10, 20):
        b = circular_moving_block_bootstrap(d_ic, block_length=L,
                                            n_replicates=n_replicates, seed=42)
        lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        cis[str(L)] = {"ci_lower": lo, "ci_upper": hi, "significant": lo > 0.0}
    return {
        "rank_ic_delta_vs_J0": float(np.nanmean(d_ic)),
        "mae_delta_vs_J0": float(np.nanmean(d_mae)),
        "block_cis": cis,
        "block_robust": all(v["significant"] for v in cis.values()),
        "n_dates": int(common.sum()),
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
    ap.add_argument("--records", default="server_runs/weights/06-posttrain/seed42/pt01_records.npz")
    ap.add_argument("--dense_min", type=int, default=3634)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    roots = resolve_roots()
    out = args.out or (roots.results_root / "pt01_analysis.json")
    s = analyze(args.records, dense_min=args.dense_min, out_path=Path(out))
    print("=== PT-01 arms ===")
    for arm, v in s["arms"].items():
        print(f"  {arm}: RankIC={v['avg_daily_rank_ic']:.5f} DA={v['avg_da_per_date']:.5f} MAE={v['avg_mae']:.5f}")
    print("=== contrasts vs J0 ===")
    for arm, v in s["contrasts_vs_J0"].items():
        print(f"  {arm}: delta={v['rank_ic_delta_vs_J0']:+.5f} robust={v['block_robust']} "
              f"CI5={v['block_cis']['5']['ci_lower']:.5f}..{v['block_cis']['5']['ci_upper']:.5f}")


if __name__ == "__main__":
    main()
