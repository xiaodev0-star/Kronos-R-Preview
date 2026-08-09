"""train_xgboost_rank.py — XGBoost with RANK objective + richer features.

The original XGBoost baseline used plain regression on c1_feats (5 dims) →
predictions shrink toward zero → weak RankIC (0.022).  This version gives XGBoost
a fair shot at the daily-RankIC target:
  - RICHER features: multi-lag returns (1/2/3/5/10/20), 5/10/20-day momentum &
    volatility, |return|, combined with c1_feats.
  - RANK objective: XGBoost rank:pairwise with one query-group per DATE
    (labels = raw returns), directly optimizing within-date ranking.
  - More tuning rounds over the R2 validation slice.

Still bounded: a moderate grid (not exhaustive), and the eval slice 0..399 is
never used for fitting.

Usage:
    python train_xgboost_rank.py            # default grid
    python train_xgboost_rank.py --grid big # larger grid (slower)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common
import data
from common import load_eval_rows, full_metrics, daily_rank_ic, save_predictions, save_json, RESULTS

import xgboost as xgb


def build_rich_features(win_X, c1, valid=None):
    """[N, W] returns + [N,5] c1 -> [N, D] rich feature matrix."""
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
    X = np.stack(feats, axis=1)
    X = np.concatenate([X, c1], axis=1)
    if valid is not None:
        X = X[valid]
    return X


def date_groups(dates):
    """XGBoost rank groups: sizes of consecutive same-date blocks (dates sorted)."""
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

    # ---- fit rich features (aligned) ----
    fitf = data.load_fit_features()
    fits = data.load_fit_sequences(window=32)
    fin = (np.isfinite(fits["X"]).all(axis=1) & np.isfinite(fitf["X"]).all(axis=1)
           & np.isfinite(fitf["y"]))
    Xr = build_rich_features(fits["X"], fitf["X"], valid=fin)
    y = fitf["y"][fin]
    dates = np.asarray([str(d)[:10] for d in fitf["date_key"]])[fin]
    print(f"[xr] fit rows={len(y)} rich_features={Xr.shape[1]}")

    # ---- tuning on R2 val slice ----
    tr_mask = dates < "2022-02-01"
    va_mask = (dates >= "2022-02-01") & (dates < "2023-02-01")
    grid = ([{"n": 300, "depth": 4, "lr": 0.05},
             {"n": 600, "depth": 6, "lr": 0.03},
             {"n": 1000, "depth": 6, "lr": 0.02},
             {"n": 400, "depth": 5, "lr": 0.05}]
            if args.grid == "small" else
            [{"n": 300, "depth": 4, "lr": 0.05},
             {"n": 600, "depth": 6, "lr": 0.03},
             {"n": 1000, "depth": 7, "lr": 0.02},
             {"n": 1500, "depth": 8, "lr": 0.01},
             {"n": 500, "depth": 5, "lr": 0.05},
             {"n": 800, "depth": 4, "lr": 0.03}])
    best = None
    for cfg in grid:
        order, sizes = date_groups(dates[tr_mask])
        Xt = Xr[tr_mask][order]
        yt = y[tr_mask][order]
        # drop tiny groups (rank needs >=2 per group)
        keep = sizes >= 2
        valid_grp = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
        model = xgb.XGBRanker(
            n_estimators=cfg["n"], max_depth=cfg["depth"], learning_rate=cfg["lr"],
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_lambda=1.0, objective="rank:pairwise", n_jobs=-1, seed=args.seed)
        model.fit(Xt[valid_grp], yt[valid_grp], group=sizes[keep])
        vp = model.predict(Xr[va_mask])
        vr = {"date_key": dates[va_mask], "true_logret": y[va_mask],
              "quality": np.ones(va_mask.sum(), dtype=bool),
              "dense_threshold": max(5, int(0.8 * 400))}
        ic = float(np.mean(list(daily_rank_ic(vp, vr).values())))
        print(f"[xr] tune {cfg} val RankIC={ic:.4f}")
        if best is None or ic > best[1]:
            best = (cfg, ic)
    cfg = best[0]
    print(f"[xr] tuned -> {cfg} (val RankIC={best[1]:.4f})")

    # ---- final fit on all fit rows ----
    order, sizes = date_groups(dates)
    Xo, yo = Xr[order], y[order]
    keep = sizes >= 2
    valid_grp = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
    t0 = time.time()
    model = xgb.XGBRanker(
        n_estimators=cfg["n"], max_depth=cfg["depth"], learning_rate=cfg["lr"],
        subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
        reg_lambda=1.0, objective="rank:pairwise", n_jobs=-1, seed=args.seed)
    model.fit(Xo[valid_grp], yo[valid_grp], group=sizes[keep])
    print(f"[xr] final fit in {time.time()-t0:.0f}s")

    # ---- eval ----
    evf = data.load_eval_features()
    evs = data.load_eval_sequences(window=32)
    ev_valid = np.isfinite(evs["X"]).all(axis=1) & evf["valid"]
    Xe = build_rich_features(evs["X"], evf["X"], valid=None)
    pred = model.predict(Xe).astype(np.float64)
    pred[~ev_valid] = np.nan

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[xr] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("xgboost_rank", pred, meta={
        "features": "rich(14)+c1(5)", "objective": "rank:pairwise",
        "group": "per-date", "n_trees": cfg["n"], "max_depth": cfg["depth"],
        "lr": cfg["lr"], "seed": args.seed})
    save_json(RESULTS / "xgboost_rank_metrics.json", m)
    print(f"[xr] saved -> {path}")


if __name__ == "__main__":
    main()
