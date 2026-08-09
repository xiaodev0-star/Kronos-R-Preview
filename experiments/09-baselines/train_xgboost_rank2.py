"""train_xgboost_rank2.py — XGBoost rank + cross-sectional relative-strength features.

The rank:pairwise version reached 0.0742 (close to Exp08's 0.0817).  This adds
CROSS-SECTIONAL relative-strength features — the within-date rank-percentile of
the lag-1 return / 5d momentum / 20d momentum — since the target metric (daily
RankIC) is itself a cross-sectional ranking.  A few more tuning configs.

Usage:
    python train_xgboost_rank2.py --grid small
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common
import data
from common import load_eval_rows, full_metrics, daily_rank_ic, save_predictions, save_json, RESULTS

import xgboost as xgb


def rank_pct_per_date(dates, score):
    """Within-date rank-percentile in [0,1] (low=worst, high=best)."""
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    out = np.full(len(score), np.nan)
    s = score[order]
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blk = s[lo:hi]
        m = np.isfinite(blk)
        if m.sum() == 0:
            continue
        r = np.argsort(np.argsort(blk[m], kind="stable")).astype(float)
        r = r / max(1, len(r) - 1)
        tmp = np.full(blk.shape, np.nan)
        tmp[m] = r
        out[order[lo:hi]] = tmp
    return out


def build_features(win_X, c1, dates):
    """Rich time-series features + cross-sectional relative strength."""
    feats = []
    for lag in (1, 2, 3, 5, 10, 20):
        if win_X.shape[1] >= lag:
            feats.append(win_X[:, -lag])
    for w in (5, 10, 20):
        if win_X.shape[1] >= w:
            feats.append(win_X[:, -w:].mean(axis=1))
            feats.append(win_X[:, -w:].std(axis=1))
    feats.append(np.abs(win_X[:, -1]))
    feats.append(np.sign(win_X[:, -1]))
    X = np.stack(feats, axis=1)                       # time-series features
    X = np.concatenate([X, c1], axis=1)
    # cross-sectional relative strength (rank-percentile within date)
    for col_idx in (0, 1, 2):                          # lag-1, lag-2, lag-3 returns
        cs = rank_pct_per_date(dates, win_X[:, -(col_idx + 1)])
        X = np.concatenate([X, cs[:, None]], axis=1)
    # 5d / 20d momentum rank
    for w in (5, 20):
        if win_X.shape[1] >= w:
            cs = rank_pct_per_date(dates, win_X[:, -w:].mean(axis=1))
            X = np.concatenate([X, cs[:, None]], axis=1)
    # 20d vol rank (low vol = higher relative quality)
    if win_X.shape[1] >= 20:
        cs = rank_pct_per_date(dates, -win_X[:, -20:].std(axis=1))
        X = np.concatenate([X, cs[:, None]], axis=1)
    return X


def date_groups(dates):
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    sizes = np.diff(bounds).astype(int)
    return order, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["small", "big"], default="small")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fitf = data.load_fit_features()
    fits = data.load_fit_sequences(window=32)
    fin = (np.isfinite(fits["X"]).all(axis=1) & np.isfinite(fitf["X"]).all(axis=1)
           & np.isfinite(fitf["y"]))
    dates_all = np.asarray([str(d)[:10] for d in fitf["date_key"]])
    Xr = build_features(fits["X"], fitf["X"], dates_all)[fin]
    y = fitf["y"][fin]
    dates = dates_all[fin]
    print(f"[xr2] fit rows={len(y)} features={Xr.shape[1]}")

    tr_mask = dates < "2022-02-01"
    va_mask = (dates >= "2022-02-01") & (dates < "2023-02-01")
    grid = ([{"n": 300, "depth": 4, "lr": 0.05},
             {"n": 600, "depth": 6, "lr": 0.03},
             {"n": 1000, "depth": 6, "lr": 0.02}]
            if args.grid == "small" else
            [{"n": 300, "depth": 4, "lr": 0.05},
             {"n": 600, "depth": 6, "lr": 0.03},
             {"n": 1000, "depth": 7, "lr": 0.02},
             {"n": 1500, "depth": 8, "lr": 0.01},
             {"n": 800, "depth": 5, "lr": 0.04}])
    best = None
    for cfg in grid:
        order, sizes = date_groups(dates[tr_mask])
        Xt, yt = Xr[tr_mask][order], y[tr_mask][order]
        keep = sizes >= 2
        vg = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
        model = xgb.XGBRanker(
            n_estimators=cfg["n"], max_depth=cfg["depth"], learning_rate=cfg["lr"],
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_lambda=1.0, objective="rank:pairwise", n_jobs=-1, seed=args.seed)
        model.fit(Xt[vg], yt[vg], group=sizes[keep])
        vp = model.predict(Xr[va_mask])
        vr = {"date_key": dates[va_mask], "true_logret": y[va_mask],
              "quality": np.ones(va_mask.sum(), dtype=bool),
              "dense_threshold": max(5, int(0.8 * 400))}
        ic = float(np.mean(list(daily_rank_ic(vp, vr).values())))
        print(f"[xr2] tune {cfg} val RankIC={ic:.4f}")
        if best is None or ic > best[1]:
            best = (cfg, ic)
    cfg = best[0]
    print(f"[xr2] tuned -> {cfg} (val RankIC={best[1]:.4f})")

    order, sizes = date_groups(dates)
    Xo, yo = Xr[order], y[order]
    keep = sizes >= 2
    vg = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
    t0 = time.time()
    model = xgb.XGBRanker(
        n_estimators=cfg["n"], max_depth=cfg["depth"], learning_rate=cfg["lr"],
        subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
        reg_lambda=1.0, objective="rank:pairwise", n_jobs=-1, seed=args.seed)
    model.fit(Xo[vg], yo[vg], group=sizes[keep])
    print(f"[xr2] final fit in {time.time()-t0:.0f}s")

    evf = data.load_eval_features()
    evs = data.load_eval_sequences(window=32)
    ev_valid = np.isfinite(evs["X"]).all(axis=1) & evf["valid"]
    ev_dates = np.asarray([str(d)[:10] for d in load_eval_rows()["date_key"]])
    Xe = build_features(evs["X"], evf["X"], ev_dates)
    pred = model.predict(Xe).astype(np.float64)
    pred[~ev_valid] = np.nan

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[xr2] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("xgboost_rank2", pred, meta={
        "features": "rich+cs-relative-strength", "objective": "rank:pairwise",
        "group": "per-date", "n_trees": cfg["n"], "max_depth": cfg["depth"],
        "lr": cfg["lr"], "seed": args.seed})
    save_json(RESULTS / "xgboost_rank2_metrics.json", m)
    print(f"[xr2] saved -> {path}")


if __name__ == "__main__":
    main()
