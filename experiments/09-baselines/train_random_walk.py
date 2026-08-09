"""train_random_walk.py — random-walk baseline (zero-training).

Classic random-walk / no-change forecast: the best prediction for tomorrow is
today.  Score = the last VISIBLE daily log return before the target row
(feat[p-1, 0], i.e. the return of the most recent completed day).  No learned
parameters — it exists to anchor the comparison: any learned model must beat
"tomorrow ≈ today" to add value.

Score = X_eval[:, -1] from the W=32 return-window cache (window end is
feat[p-1, 0] by construction — see data.py).

Usage:
    python train_random_walk.py
"""
from __future__ import annotations

import numpy as np

import common
import data
from common import load_eval_rows, full_metrics, save_predictions, save_json, RESULTS


def main():
    print("[rw] random-walk baseline: pred[t+1] = return[t] (today)")
    ev = data.load_eval_sequences(window=32)
    Xev = ev["X"]
    pred = Xev[:, -1].astype(np.float64)     # last visible return
    # rows whose window was incomplete get NaN -> masked by the evaluator
    pred[np.isnan(Xev).any(axis=1)] = np.nan

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[rw] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("random_walk", pred, meta={
        "model": "random_walk", "rule": "pred[t+1]=return[t]", "trained": False})
    save_json(RESULTS / "random_walk_metrics.json", m)
    print(f"[rw] saved -> {path}")


if __name__ == "__main__":
    main()
