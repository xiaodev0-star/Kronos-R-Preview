"""f55_smooth_mag.py — conditional magnitude via monotone smoothing.

The champion (f54) assigns the date's MARGINAL |pred_B| quantiles by rank.
MAPE is limited because the marginal quantile is not the conditional mean.
This script fits a CALIB monotone map  u = per-date distance rank (0..1)
  -> E[|y| | u]   (bin means + isotonic + linear interpolation), then applies
it on eval.  Key properties:
  - outputs are conditional magnitudes (|y| >= 0 -> E[|y||u] >= 0; the mean
    over u is E[|y|] -> AmpRatio naturally ~ 0.9-1.0, modulo calib<->eval
    scale drift);
  - the interpolated map is continuous -> low token collapse;
  - pred = dir_s * map(u) stays monotone in F -> RankIC 0.0817 preserved.
  - a per-date scale correction (to the date's mean|pred_B|) is applied IF the
    calib-fitted AR drifts out of [0.8, 1.2] on eval (order-preserving).

Direction/boundary: calib-fit lambda from f54 (read from f54_preds.npz).
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

from _exp07 import (  # noqa: E402
    weights_root, results_root, cand_path, write_json_ledger,
)
from f48_micro_scan import (  # noqa: E402
    daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse,
    top_frac, block_masks,
)
from sklearn.isotonic import IsotonicRegression  # noqa: E402


def date_boundary_qd(F, dates, src):
    bnd = np.full(len(F), np.nan)
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(src))[0]
        if len(dm) < 5:
            continue
        qd = float(np.mean(src[dm] < 0))
        b = float(np.quantile(F[dm], qd))
        bnd[dm] = b
        out[dm] = np.sign(F[dm] - b)
    return out, bnd


def dist_rank_u(dates, F, boundary):
    """Per-date rank-percentile of |F - boundary| in [0, 1]."""
    u = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(boundary))[0]
        if len(dm) < 5:
            continue
        key = np.abs(F[dm] - boundary[dm[0]])
        r = np.argsort(np.argsort(key, kind="stable"), kind="stable") / max(len(dm) - 1, 1)
        u[dm] = r
    return u


def smooth_monotone_map(u, y, n_bins=80):
    """Bin means over u -> isotonic -> linear interpolation (continuous)."""
    m = np.isfinite(u) & np.isfinite(y)
    ub, yb = u[m], y[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    cents, vals = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mm = (ub >= lo) & (ub < hi)
        if mm.sum() >= 50:
            cents.append((lo + hi) / 2.0)
            vals.append(float(np.mean(yb[mm])))
    cents, vals = np.asarray(cents), np.asarray(vals)
    iso = IsotonicRegression(out_of_bounds="clip").fit(cents, vals)
    mon = iso.predict(cents)
    # enforce strict endpoint monotonicity via interpolation grid
    grid = np.linspace(0.0, 1.0, 401)
    gmon = iso.predict(grid)
    return grid, gmon


def apply_map(u, grid, gmon):
    return np.interp(u, grid, gmon)


def daily_stats(dates, true, score, quality, dense, centers):
    rec = {"date_key": dates, "true_logret": true, "quality": quality}
    rec["__s__"] = score
    return {
        "rank_ic": mean_ic(rec, "__s__", dense),
        "da": float(np.mean(list(daily_da(rec, "__s__", dense).values()))),
        "mape": float(np.mean(list(daily_mape(rec, "__s__", dense).values()))),
        "amp_ratio": ampratio(score, true),
        "collapse": token_collapse(score, centers),
    }


def per_date_scale(dates, pred, target_abs):
    """Scale pred per date to match mean|target| per date (order-preserving)."""
    out = np.full(len(pred), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(pred) & np.isfinite(target_abs))[0]
        if len(dm) < 2:
            continue
        s = np.nanmean(np.abs(target_abs[dm])) / max(np.nanmean(np.abs(pred[dm])), 1e-12)
        out[dm] = pred[dm] * s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    f54 = np.load(wr / "f54_preds.npz", allow_pickle=True)
    F = f54["F"].astype(np.float64)
    champ = f54["champ"].astype(np.float64)
    lam = float(f54["lambda_"][0])
    f48 = np.load(wr / "f48_preds.npz", allow_pickle=True)
    BERT_E = f48["BERT_E"].astype(np.float64)
    pred_B = f48["pred_B"].astype(np.float64)
    true = f48["true_logret"].astype(np.float64)
    dates = np.asarray([str(d) for d in f54["date_key"]])
    quality = f54["quality"].astype(bool)
    dense = int(f54["dense_threshold"][0])
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    F0 = 0.2574
    n = len(F)
    print(f"[f55] loaded; lambda={lam:.2f} n={n}", flush=True)

    res = {"schema": "f55-smooth-mag-v1", "dense_threshold": dense, "lambda": lam}

    # ---- calib: u -> |y| map ----
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    y_c = ccal["true_logret"].astype(np.float64)
    q_c = ccal["quality"].astype(bool)
    dc_c = np.asarray([str(d)[:10] for d in ccal["date_key"]])
    F_c = np.load(wr / "f52_calib_F.npy")
    BERT_E_c = np.load(wr / "f52_calib_BERTE.npy")
    dir_B_c, bnd_B_c = date_boundary_qd(F_c, dc_c, BERT_E_c)
    bnd_s_c = (1.0 - lam) * F0 + lam * bnd_B_c
    u_c = dist_rank_u(dc_c, F_c, bnd_s_c)
    mv = np.isfinite(u_c) & np.isfinite(y_c) & q_c
    grid, gmon = smooth_monotone_map(u_c[mv], np.abs(y_c[mv]))
    print(f"[f55] calib map: u={grid[0]:.2f}..{grid[-1]:.2f} "
          f"map={gmon[0]:.5f}..{gmon[-1]:.5f}", flush=True)

    # ---- eval: apply map ----
    dir_B, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    u = dist_rank_u(dates, F, bnd_s)
    mag = apply_map(u, grid, gmon)
    pred = np.sign(F - bnd_s) * mag
    st = daily_stats(dates, true, pred, quality, dense, centers)
    res["smooth_map"] = st
    print(f"[f55] smooth map: RankIC={st['rank_ic']:.4f} DA={st['da']:.4f} "
          f"MAPE={st['mape']:.4f} AR={st['amp_ratio']:.3f} Coll={st['collapse']:.3f}")

    # scaled version (if AR drifts)
    pred_s = per_date_scale(dates, pred, np.abs(pred_B))
    st_s = daily_stats(dates, true, pred_s, quality, dense, centers)
    res["smooth_map_scaled"] = st_s
    print(f"[f55] smooth map + per-date scale: RankIC={st_s['rank_ic']:.4f} "
          f"DA={st_s['da']:.4f} MAPE={st_s['mape']:.4f} AR={st_s['amp_ratio']:.3f} "
          f"Coll={st_s['collapse']:.3f}")

    # reference: champion
    st_c = daily_stats(dates, true, champ, quality, dense, centers)
    res["champion_ref"] = st_c
    print(f"[f55] champion ref: RankIC={st_c['rank_ic']:.4f} DA={st_c['da']:.4f} "
          f"MAPE={st_c['mape']:.4f} AR={st_c['amp_ratio']:.3f} Coll={st_c['collapse']:.3f}")

    np.savez(wr / "f55_preds.npz",
             pred_smooth=pred, pred_smooth_scaled=pred_s,
             true_logret=true, date_key=dates, quality=quality,
             dense_threshold=np.array([dense]))

    out = rr / "f55_smooth_mag.json"
    write_json_ledger(out, res, "f55_smooth_mag")
    print(f"[f55] -> {out}")


if __name__ == "__main__":
    main()
