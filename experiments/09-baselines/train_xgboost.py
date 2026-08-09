"""train_xgboost.py — XGBoost baseline.

Trains an XGBoost regressor on the shared c1_feats (5-dim point-in-time
features: last-return / 5d momentum / 20d momentum / 20d vol / 20d volume)
to predict the next-day raw log return.  The predicted return is the row score
(the same monotone-scale contract as the Exp08 reference), and daily RankIC /
DA / MAPE are computed by the shared evaluator.

Maintainable / self-contained:
  - uses data.py + common.py (the shared layer)
  - predictions saved to outputs/xgboost_pred.npz
  - a separate run validates on a rolling slice before the final fit

Usage:
    python train_xgboost.py            # train + save predictions
    python train_xgboost.py --n-trees 300 --max-depth 4
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common
import data
from common import load_eval_rows, full_metrics, save_predictions, save_json, RESULTS

import xgboost as xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="small grid on R2 val slice")
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[xgb] loading fit features...")
    fit = data.load_fit_features()
    Xtr, ytr = fit["X"], fit["y"]
    # drop rows with non-finite features (should be none, but be safe)
    fin = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[fin], ytr[fin]
    print(f"[xgb] fit rows: {len(ytr)}  features: {Xtr.shape[1]}")

    dates = np.asarray([str(d)[:10] for d in fit["date_key"]])
    tr_mask = dates[fin] < "2022-02-01"
    va_mask = (dates[fin] >= "2022-02-01") & (dates[fin] < "2023-02-01")

    # ---- light tuning on the R2 val slice ----
    chosen = {"n_trees": args.n_trees, "max_depth": args.max_depth}
    if args.tune:
        grid = [{"n_trees": 200, "max_depth": 3},
                {"n_trees": 300, "max_depth": 4},
                {"n_trees": 500, "max_depth": 5}]
        best = None
        for cfg in grid:
            m_ = xgb.XGBRegressor(n_estimators=cfg["n_trees"], max_depth=cfg["max_depth"],
                                  learning_rate=args.lr, subsample=args.subsample,
                                  colsample_bytree=args.colsample,
                                  min_child_weight=args.min_child_weight,
                                  objective="reg:squarederror", n_jobs=-1, seed=args.seed)
            m_.fit(Xtr[tr_mask], ytr[tr_mask])
            vp = m_.predict(Xtr[va_mask])
            vr = {"date_key": fit["date_key"][fin][va_mask], "true_logret": ytr[va_mask],
                  "quality": np.ones(va_mask.sum(), dtype=bool),
                  "dense_threshold": max(5, int(0.8 * 400))}
            ic = full_metrics(vp, vr)["avg_daily_rank_ic"]
            print(f"[xgb] tune {cfg} val RankIC={ic:.4f}")
            if best is None or ic > best[1]:
                best = (cfg, ic)
        chosen.update(best[0])
        print(f"[xgb] tuned -> {chosen} (val RankIC={best[1]:.4f})")

    print(f"[xgb] final fit ({chosen['n_trees']} trees, depth {chosen['max_depth']})...")
    t0 = time.time()
    model = xgb.XGBRegressor(n_estimators=chosen["n_trees"], max_depth=chosen["max_depth"],
                             learning_rate=args.lr, subsample=args.subsample,
                             colsample_bytree=args.colsample,
                             min_child_weight=args.min_child_weight,
                             objective="reg:squarederror", n_jobs=-1, seed=args.seed)
    model.fit(Xtr, ytr)
    print(f"[xgb] trained in {time.time()-t0:.0f}s")

    # ---- eval predictions ----
    ev = data.load_eval_features()
    Xev = ev["X"]
    pred = model.predict(Xev).astype(np.float64)
    pred[~ev["valid"]] = np.nan

    # align + report
    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"], f"pred {len(pred)} != eval rows {rows['n_rows']}"
    m = full_metrics(pred, rows)
    print(f"[xgb] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("xgboost", pred, meta={
        "features": "c1_feats(5)",
        "objective": "reg:squarederror",
        "n_trees": chosen["n_trees"], "max_depth": chosen["max_depth"], "lr": args.lr,
        "seed": args.seed,
    })
    save_json(RESULTS / "xgboost_metrics.json", m)
    print(f"[xgb] saved -> {path}")


if __name__ == "__main__":
    main()
