"""f32_robust_ensemble.py — robust aggregation of the 6-head ensemble.

The 6 BERT heads average via within-date rank-percentile mean.  Tests robust
aggregations (median / trimmed-mean / rank-of-heads) that down-weight outlier
heads per row, and whether they improve the fused F and the |F| abstention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def _daily_ic(rec, field, dense):
    score = np.asarray(rec[field], dtype=np.float64)
    true = np.asarray(rec["true_logret"], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(true) & np.asarray(rec["quality"]).astype(bool)
    order, bounds, uniq = _split_points(rec["date_key"])
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


def _exact_frac(rec, cf, cv, max_high=True):
    conf = np.asarray(rec[cf], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    acted = np.zeros(len(rec["stock_uid"]), dtype=bool)
    for d in np.unique(dates):
        dm = dates == d
        cvals = conf[dm]
        m = np.isfinite(cvals)
        if m.sum() == 0:
            continue
        keep_n = max(1, int(round(cv * m.sum())))
        ord_ = np.argsort(-cvals) if max_high else np.argsort(cvals)
        idx = np.where(dm)[0]
        acted[idx[ord_[:keep_n]]] = True
    return acted


def main():
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    R = np.stack(ranks, axis=1)          # [N, 6]
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)

    res = {"schema": "f32-robust-ensemble-v1", "dense_threshold": dense, "arms": {}}
    for name, agg in (
        ("mean", lambda r: r.mean(axis=1)),
        ("median", lambda r: np.median(r, axis=1)),
        ("trim10", lambda r: np.sort(r, axis=1)[:, 1:-1].mean(axis=1)),  # drop min+max
        ("minmax", lambda r: r.min(axis=1) + r.max(axis=1)),            # extremal heads
    ):
        ens = agg(R)
        rec[name] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
        rec[name + "_mag"] = np.abs(rec[name])
        ic_full = float(np.mean(list(_daily_ic(rec, name, dense).values())))
        # 20% abstention
        acted = _exact_frac(rec, name + "_mag", 0.2, True)
        ra = slice_rec(rec, acted)
        ic20 = float(np.mean(list(_daily_ic(ra, name, max(5, int(0.2 * dense))).values())))
        res["arms"][name] = {"full_ic": ic_full, "ic20": ic20}
        print(f"[f32] {name:7s} full_ic={ic_full:.4f} ic20={ic20:.4f}")

    out = rr / "f32_robust_ensemble.json"
    write_json_ledger(out, res, "f32_robust_ensemble")
    print(f"[f32] -> {out}")


if __name__ == "__main__":
    main()
