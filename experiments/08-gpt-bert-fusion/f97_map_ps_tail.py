"""f97_map_ps_tail.py — tail blend: smooth-map conditional means x post_std
quantiles at the extreme ranks.

f95's map-tail (conditional means) wins the full+coverage metrics; f75's
post_std has the best-calibrated ACTED magnitudes (AR ~1.05).  Blend the two
tail sources at the distance extremes:
    tail_value(r) = (1 - b) * map(r) + b * q_ps(r)      (u > u_tail)
b fit on CALIB (min calib MAPE s.t. calib gates); tail fraction re-fit too.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, cand_path, write_json_ledger,
)
from f48_micro_scan import (  # noqa: E402
    daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse,
    top_frac,
)
from f51_adaptive_dir import date_boundary_qd, distance_quantile_mag  # noqa: E402
from f83_final_report import per_date_scale, daily_stats, coverage_stats  # noqa
from f55_smooth_mag import dist_rank_u, smooth_monotone_map, apply_map  # noqa: E402


def tail_blend_hybrid(dates, F, bulk, map_v, q_ps, boundary, tail, b):
    """bulk for the non-tail; (1-b)*map + b*q_ps for the tail rows."""
    out = np.where(np.isfinite(bulk), bulk, np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(bulk)
                      & np.isfinite(map_v) & np.isfinite(q_ps)
                      & np.isfinite(boundary))[0]
        if len(dm) < 4:
            continue
        key = np.abs(F[dm] - boundary[dm[0]])
        k_order = np.argsort(key, kind="stable")
        k = max(1, int(round(tail * len(dm))))
        for t in (k_order[:k], k_order[-k:]):
            idx = dm[t]
            out[idx] = (1.0 - b) * map_v[idx] + b * q_ps[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    f48 = np.load(wr / "f48_preds.npz", allow_pickle=True)
    F = f48["F"].astype(np.float64)
    BERT_E = f48["BERT_E"].astype(np.float64)
    pred_B = f48["pred_B"].astype(np.float64)
    true = f48["true_logret"].astype(np.float64)
    dates = np.asarray([str(d) for d in f48["date_key"]])
    quality = f48["quality"].astype(bool)
    dense = int(f48["dense_threshold"][0])
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    F0 = 0.2574
    lam = 0.25
    n = len(F)
    cand = np.load(cand_path("eval"), allow_pickle=True)
    med3 = np.load(wr / "f60_med_mag.npy")
    med3_s = per_date_scale(dates, med3, np.abs(pred_B))
    ps = cand["post_std"].astype(np.float64)
    ps_s = per_date_scale(dates, ps, np.abs(pred_B))
    dir_B, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B

    # calib pieces + map + q_ps
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    y_c = ccal["true_logret"].astype(np.float64)
    q_c = ccal["quality"].astype(bool)
    dc_c = np.asarray([str(d)[:10] for d in ccal["date_key"]])
    F_c = np.load(wr / "f52_calib_F.npy")
    BERT_E_c = np.load(wr / "f52_calib_BERTE.npy")
    pred_B_c = np.load(wr / "f52_calib_predB.npy")
    dense_c = max(5, int(0.8 * 574))
    ps_c = ccal["post_std"].astype(np.float64)
    ps_sc = per_date_scale(dc_c, ps_c, np.abs(pred_B_c))
    dir_B_c, bnd_B_c = date_boundary_qd(F_c, dc_c, BERT_E_c)
    bnd_s_c = (1.0 - lam) * F0 + lam * bnd_B_c
    u_c = dist_rank_u(dc_c, F_c, bnd_s_c)
    mv = np.isfinite(u_c) & np.isfinite(y_c) & q_c
    grid, gmon = smooth_monotone_map(u_c[mv], np.abs(y_c[mv]))
    map_c = apply_map(u_c, grid, gmon)
    q_ps_c = distance_quantile_mag(dc_c, F_c, ps_sc, bnd_s_c)
    u = dist_rank_u(dates, F, bnd_s)
    map_e = apply_map(u, grid, gmon)
    q_ps_e = distance_quantile_mag(dates, F, ps_s, bnd_s)

    res = {"schema": "f97-map-ps-tail-v1", "dense_threshold": dense, "lambda": lam}
    best = None
    for tail in (0.03, 0.05, 0.08):
        for b in (0.0, 0.3, 0.5, 0.7):
            h_c = tail_blend_hybrid(dc_c, F_c, np.abs(pred_B_c), map_c, q_ps_c,
                                    bnd_s_c, tail, b)
            mag_c = distance_quantile_mag(dc_c, F_c, h_c, bnd_s_c)
            pred_c = np.sign(F_c - bnd_s_c) * mag_c
            rec_c = {"date_key": dc_c, "true_logret": y_c, "quality": q_c}
            rec_c["__s__"] = pred_c
            st = {"mape": float(np.mean(list(daily_mape(rec_c, "__s__", dense_c).values()))),
                  "amp_ratio": ampratio(pred_c, y_c),
                  "collapse": token_collapse(pred_c, centers)}
            print(f"[f97] calib tail={tail:.2f} b={b:.1f}: MAPE={st['mape']:.4f} "
                  f"AR={st['amp_ratio']:.3f} Coll={st['collapse']:.3f}", flush=True)
            if 0.8 <= st["amp_ratio"] <= 1.2 and st["collapse"] <= 0.30:
                if best is None or st["mape"] < best[2]:
                    best = (tail, b, st["mape"])
    if best is None:
        best = (0.05, 0.0, 0.0)
    res["best_tail_b"] = list(best)
    print(f"[f97] calib-fit (tail, b) = {best[0]}, {best[1]}")

    h_e = tail_blend_hybrid(dates, F, med3_s, map_e, q_ps_e, bnd_s, best[0], best[1])
    mag = distance_quantile_mag(dates, F, h_e, bnd_s)
    pred = np.sign(F - bnd_s) * mag
    st = daily_stats(dates, true, pred, quality, dense, centers)
    st["coverage"] = {str(cv): coverage_stats(dates, true, quality, F, pred,
                                              dense, centers, cv)
                      for cv in (0.2, 0.1, 0.05)}
    res["eval"] = st
    print(f"[f97] EVAL tail={best[0]} b={best[1]}: RankIC={st['rank_ic']:.4f} "
          f"DA={st['da']:.4f} MAPE={st['mape']:.4f} AR={st['amp_ratio']:.3f} "
          f"Coll={st['collapse']:.3f}")
    print("[f97] coverage:",
          {k: (round(v["rank_ic"], 4), round(v["da"], 4), round(v["mape"], 4))
           for k, v in st["coverage"].items()})

    np.savez(wr / "f97_preds.npz",
             pred=pred,
             true_logret=true, date_key=dates, quality=quality,
             dense_threshold=np.array([dense]))

    out = rr / "f97_map_ps_tail.json"
    write_json_ledger(out, res, "f97_map_ps_tail")
    print(f"[f97] -> {out}")


if __name__ == "__main__":
    main()
