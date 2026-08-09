"""f44_regularized_xgb.py — test whether a REGULARIZED XGBoost rank signal is
robust (strong on BOTH calib and eval), unlike the overfit xgb2.

xgb2 (300/4, 25 features) hit 0.0816 on eval but only 0.05-0.08 on calib/R2 —
a classic overfitting signature (it memorised fit-region momentum patterns that
happen to persist into the eval period).  Hypothesis: a REGULARIZED variant
(fewer trees, deeper min_child_weight, fewer most-robust features) captures the
cross-sectional relative-strength signal that generalises.

For each variant: train on fit, evaluate on R2-val / calib / eval.  The goal is
a variant whose calib RankIC is reasonably high (robust), THEN check the
FUSION = 0.5 z(F) + 0.5 z(xgb_var) on calib and eval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
BL = ROOT / "experiments" / "09-baselines"
for _p in (ROOT, SEVEN, EIGHT, BL, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(BL))
import data as bl_data
import common as bl_common


def rank_pct(dates, s):
    return rank_pct_per_date(dates, s)


def build_features(win_X, c1, dates, full=True):
    """Feature builder: full (25 cols) or robust-only (lag1/5/20 mom + vol rank)."""
    feats = []
    if full:
        for lag in (1, 2, 3, 5, 10, 20):
            if win_X.shape[1] >= lag:
                feats.append(win_X[:, -lag])
        for w in (5, 10, 20):
            if win_X.shape[1] >= w:
                feats.append(win_X[:, -w:].mean(axis=1))
                feats.append(win_X[:, -w:].std(axis=1))
        feats.append(np.abs(win_X[:, -1]))
        feats.append(np.sign(win_X[:, -1]))
        X = np.stack(feats, axis=1)
        X = np.concatenate([X, c1], axis=1)
        for col_idx in (0, 1, 2):
            X = np.concatenate([X, rank_pct(dates, win_X[:, -(col_idx + 1)])[:, None]], axis=1)
        for w in (5, 20):
            if win_X.shape[1] >= w:
                X = np.concatenate([X, rank_pct(dates, win_X[:, -w:].mean(axis=1))[:, None]], axis=1)
        if win_X.shape[1] >= 20:
            X = np.concatenate([X, rank_pct(dates, -win_X[:, -20:].std(axis=1))[:, None]], axis=1)
    else:
        # robust subset: cross-sectional ranks of lag1/5d/20d momentum, 20d vol, 20d return
        cols = [rank_pct(dates, win_X[:, -1]),
                rank_pct(dates, win_X[:, -5:].mean(axis=1)) if win_X.shape[1] >= 5 else rank_pct(dates, win_X[:, -1]),
                rank_pct(dates, win_X[:, -20:].mean(axis=1)) if win_X.shape[1] >= 20 else rank_pct(dates, win_X[:, -1]),
                rank_pct(dates, -win_X[:, -20:].std(axis=1)) if win_X.shape[1] >= 20 else rank_pct(dates, win_X[:, -1]),
                rank_pct(dates, win_X[:, -20:].sum(axis=1)) if win_X.shape[1] >= 20 else rank_pct(dates, win_X[:, -1])]
        X = np.stack(cols, axis=1)
        X = np.concatenate([X, c1], axis=1)
    return X


def date_groups(dates):
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    sizes = np.diff(bounds).astype(int)
    return order, sizes


def daily_ic(score, date_key, true, quality, dense):
    score = np.asarray(score, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(true) & np.asarray(quality).astype(bool)
    order, bounds, uniq = _split_points(date_key)
    s, t, v = score[order], true[order], valid[order]
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        m = v[lo:hi]
        c = int(m.sum())
        if c < dense:
            continue
        sv, tv = s[lo:hi][m], t[lo:hi][m]
        out[uniq[i]] = float(spearmanr(sv, tv)[0]) if (c >= 2 and not np.all(sv == sv[0])) else 0.0
    return out


def mean_ic(score, date_key, true, quality, dense):
    d = daily_ic(score, date_key, true, quality, dense)
    return float(np.mean(list(d.values()))) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"

    # ---- fit features (for training + R2-val) ----
    fitf = bl_data.load_fit_features()
    fits = bl_data.load_fit_sequences(window=32)
    fin = (np.isfinite(fits["X"]).all(axis=1) & np.isfinite(fitf["X"]).all(axis=1)
           & np.isfinite(fitf["y"]))
    dates_all = np.asarray([str(d)[:10] for d in fitf["date_key"]])
    y = fitf["y"][fin]
    dates = dates_all[fin]
    print("[f44] building fit features (full + robust)")
    Xr_full = build_features(fits["X"], fitf["X"], dates_all, full=True)[fin]
    Xr_rob = build_features(fits["X"], fitf["X"], dates_all, full=False)[fin]

    # ---- calib + eval features ----
    caf = np.load(sw / "calibration_cache.npz", allow_pickle=True)
    cas = bl_data.build_sequences("calib", 32, cache=True)
    ca_dates = np.asarray([str(d)[:10] for d in cas["date_key"]])
    ca_full = build_features(cas["X"], caf["c1_feats"], ca_dates, full=True)
    ca_rob = build_features(cas["X"], caf["c1_feats"], ca_dates, full=False)
    evf = bl_data.load_eval_features()
    evs = bl_data.load_eval_sequences(window=32)
    ev_meta = bl_common.load_eval_rows()
    ev_dates = np.asarray([str(d)[:10] for d in ev_meta["date_key"]])
    ev_full = build_features(evs["X"], evf["X"], ev_dates, full=True)
    ev_rob = build_features(evs["X"], evf["X"], ev_dates, full=False)

    import xgboost as xgb
    # ---- variants: (name, X_fit, n, depth, min_child, lr) ----
    variants = [
        ("xgb2_full_300_4", Xr_full, 300, 4, 8, 0.05),
        ("reg_full_150_3", Xr_full, 150, 3, 16, 0.03),
        ("reg_full_100_3", Xr_full, 100, 3, 24, 0.02),
        ("reg_robust_150_3", Xr_rob, 150, 3, 16, 0.03),
        ("reg_robust_100_3", Xr_rob, 100, 3, 24, 0.02),
    ]
    res = {"schema": "f44-regularized-xgb-v1", "seed": args.seed, "variants": {}}

    # R2-val mask
    r2_mask = (dates >= "2022-02-01") & (dates < "2023-02-01")
    r2_true = y[r2_mask]
    r2_dates = dates[r2_mask]
    ca_dense = max(5, int(0.8 * 574))
    ev_dense = ev_meta["dense_threshold"]
    r2_dense = max(5, int(0.8 * 400))

    for name, Xf, n, depth, mw, lr in variants:
        order, sizes = date_groups(dates)
        Xo, yo = Xf[order], y[order]
        keep = sizes >= 2
        vg = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
        model = xgb.XGBRanker(n_estimators=n, max_depth=depth, learning_rate=lr,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=mw,
                              reg_lambda=1.0, objective="rank:pairwise",
                              n_jobs=-1, seed=args.seed)
        model.fit(Xo[vg], yo[vg], group=sizes[keep])
        # eval on R2-val / calib / eval
        is_full = "full" in name
        Xe_r2 = (Xr_full if is_full else Xr_rob)[r2_mask]
        Xe_ca = ca_full if is_full else ca_rob
        Xe_ev = ev_full if is_full else ev_rob
        ic_r2 = mean_ic(model.predict(Xe_r2), r2_dates, r2_true,
                        np.isfinite(r2_true), r2_dense)
        ic_ca = mean_ic(model.predict(Xe_ca), ca_dates, cas["y"],
                        np.isfinite(cas["y"]), ca_dense)
        ic_ev = mean_ic(model.predict(Xe_ev), ev_dates, ev_meta["true_logret"],
                        ev_meta["quality"], ev_dense)
        res["variants"][name] = {"n": n, "depth": depth, "min_child": mw, "lr": lr,
                                 "r2val": ic_r2, "calib": ic_ca, "eval": ic_ev}
        print(f"[f44] {name}: r2={ic_r2:.4f} calib={ic_ca:.4f} eval={ic_ev:.4f}")

    out = rr / "f44_regularized_xgb.json"
    write_json_ledger(out, res, "f44_regularized_xgb", seed=args.seed)
    print(f"[f44] -> {out}")


if __name__ == "__main__":
    main()
