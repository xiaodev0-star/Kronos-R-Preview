"""f36_ensemble_xgb.py — output-space ensemble: Exp08 F + XGBoost rank2.

Exp08 F (0.5·z(BERT6-ens)+0.5·z(P6)) gives full RankIC 0.0817.
XGBoost rank2 (rank:pairwise + cross-sectional relative strength) gives 0.0816.
These are decorrelated signal sources (transformer representations vs
hand-crafted cross-sectional features).  An output-space ensemble may beat both
— the goal of this round (STRONGER than Exp08).

Tests:
  - decorrelation (per-date IC correlation) between F and xgboost_rank2
  - F_ens(w) = w·z(F) + (1-w)·z(xgb2) for w in {0.5, 0.6, 0.7}
  - full RankIC / DA + |score|-abstention coverage curve (20%)
"""
from __future__ import annotations

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
    weights_root, results_root, build_rec, cand_path, slice_rec,
    DailyIcCache, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from common import load_predictions  # noqa: E402


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


def _daily_da(rec, field, dense):
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
        out[uniq[i]] = float(np.mean((np.sign(s[lo:hi][m]) > 0) == (t[lo:hi][m] > 0)))
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

    # Exp08 F
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch_from(de["hidden"])
    ranks = []
    for s in range(45, 51):
        ck = torch_load(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt")
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch_no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)

    # XGBoost rank2
    xb = load_predictions("xgboost_rank2")
    assert xb is not None, "run train_xgboost_rank2.py first"
    rec["xgb2"] = np.asarray(xb["score"], dtype=np.float64)
    rec["z_xgb"] = z_per_date(dates, rec["xgb2"])

    # decorrelation (per-date IC)
    cc = DailyIcCache(rec, dense)
    icF = cc.series(rec["F"]); icX = cc.series(rec["z_xgb"])
    common_d = sorted(set(icF) & set(icX))
    ic_corr = float(np.corrcoef([icF[d] for d in common_d], [icX[d] for d in common_d])[0, 1])
    print(f"[f36] per-date IC correlation F vs xgb2: {ic_corr:.4f}")

    res = {"schema": "f36-ensemble-xgb-v1", "dense_threshold": dense,
           "ic_corr": ic_corr, "arms": {}}
    for w in (0.5, 0.6, 0.7):
        name = f"ENS_w{w:g}"
        rec[name] = w * z_per_date(dates, rec["F"]) + (1 - w) * rec["z_xgb"]
        rec[name + "_mag"] = np.abs(rec[name])
        ic_full = float(np.mean(list(_daily_ic(rec, name, dense).values())))
        da = float(np.mean(list(_daily_da(rec, name, dense).values())))
        acted = _exact_frac(rec, name + "_mag", 0.2, True)
        ra = slice_rec(rec, acted)
        ic20 = float(np.mean(list(_daily_ic(ra, name, max(5, int(0.2 * dense))).values())))
        res["arms"][name] = {"full_ic": ic_full, "da": da, "ic20": ic20}
        print(f"[f36] {name} full_ic={ic_full:.4f} da={da:.4f} ic20={ic20:.4f}")

    # reference numbers
    ref_F = float(np.mean(list(_daily_ic(rec, "F", dense).values())))
    ref_x = float(np.mean(list(_daily_ic(rec, "z_xgb", dense).values())))
    print(f"[f36] reference: F={ref_F:.4f} xgb2={ref_x:.4f}")

    out = rr / "f36_ensemble_xgb.json"
    write_json_ledger(out, res, "f36_ensemble_xgb")
    print(f"[f36] -> {out}")


# minimal torch aliases (keeps the file self-contained for the head apply)
def torch_from(a):
    import torch
    return torch.from_numpy(np.asarray(a).astype(np.float32))


def torch_load(p):
    import torch
    return torch.load(str(p), map_location="cpu", weights_only=False)


def torch_no_grad():
    import torch
    return torch.no_grad()


if __name__ == "__main__":
    main()
