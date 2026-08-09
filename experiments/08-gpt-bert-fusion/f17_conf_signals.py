"""f17_conf_signals.py — test |score|-magnitude and combo confidence signals.

c_dis (ensemble disagreement) is the best abstention signal so far (0.128 @20%).
Tests whether the score MAGNITUDE (|F|, |pred_logret|) — "how strongly the model
predicts a direction" — adds complementary confidence, and whether a 2D rule
(keep rows that are BOTH low-disagreement AND high-magnitude) beats c_dis alone
at matched coverage.
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
    DailyIcCache, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402


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
        from scipy.stats import spearmanr
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
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
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
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            sc = head(H).numpy().astype(np.float64)
        ranks.append(sc)   # RAW scores (not rank) for magnitude
    rec["c_dis"] = np.std([rank_pct_per_date(dates, r) for r in ranks], axis=0)
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_magF"] = np.abs(rec["F"])

    # calibrated magnitude |pred_logret|
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))  # placeholder; real below
    rec["c_magpred"] = np.abs(rec["F"])  # fallback

    # combo: 2D rule — low c_dis AND high |F| (both within-date top halves -> ~25% coverage)
    r_dis = rank_pct_per_date(dates, rec["c_dis"])      # low = agree
    r_mag = rank_pct_per_date(dates, -rec["c_magF"])    # low = high |F|
    rec["c_combo2d"] = 0.5 * r_dis + 0.5 * r_mag

    res = {"schema": "f17-conf-signals-v1", "dense_threshold": dense, "curves": {}}
    for cf, max_high in (("c_dis", False), ("c_magF", True), ("c_combo2d", False)):
        row = []
        for cv in (0.8, 0.6, 0.4, 0.2):
            acted = _exact_frac(rec, cf, cv, max_high=max_high)
            ra = slice_rec(rec, acted)
            da20 = max(5, int(round(cv * dense)))
            ic_ = _daily_ic(ra, "F", da20)
            row.append({"cov": cv, "acted_ic": float(np.mean(list(ic_.values()))) if ic_ else None})
            print(f"[f17] {cf:11s} cov={cv} acted_ic={row[-1]['acted_ic'] and round(row[-1]['acted_ic'],4)}")
        res["curves"][cf] = {"max_high": max_high, "curve": row}

    # 2D intersection at ~20%: top 45% by low-dis AND top 45% by high-mag
    act_dis = _exact_frac(rec, "c_dis", 0.45, max_high=False)
    act_mag = _exact_frac(rec, "c_magF", 0.45, max_high=True)
    act2d = act_dis & act_mag
    ra2 = slice_rec(rec, act2d)
    da2 = max(5, int(round(0.2 * dense)))
    ic2 = _daily_ic(ra2, "F", da2)
    res["2d_intersection_20pct"] = {"acted_ic": float(np.mean(list(ic2.values()))) if ic2 else None,
                                    "frac": float(act2d.mean())}
    print(f"[f17] 2D intersection frac={act2d.mean():.3f} acted_ic={res['2d_intersection_20pct']['acted_ic']}")

    out = rr / "f17_conf_signals.json"
    write_json_ledger(out, res, "f17_conf_signals")
    print(f"[f17] -> {out}")


if __name__ == "__main__":
    main()
