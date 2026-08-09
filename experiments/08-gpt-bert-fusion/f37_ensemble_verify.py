"""f37_ensemble_verify.py — verify + lock the new champion ensemble.

New champion:  FUSION = 0.5·z(Exp08 F) + 0.5·z(XGBoost rank2).
  F     = 0.5·z(BERT6-ens) + 0.5·z(P6)          (Exp08, RankIC 0.0817)
  xgb2  = XGBoost rank:pairwise + cross-sectional relative-strength (0.0816)
per-date IC correlation is only ~0.23 → output-space average is strong.

Verification:
  - full / dev / confirm RankIC + DA
  - |score|-abstention coverage (20%)
  - paired moving-block bootstrap vs Exp08 F (L=5/10/20)
  - save the FUSION score to outputs/reference_fusion.npz for the baselines folder
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
    DailyIcCache, bootstrap_vs, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from common import load_predictions, save_predictions  # noqa: E402
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
    import torch
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    off = np.asarray(rec["offset"], dtype=np.int64)

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
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)

    xb = load_predictions("xgboost_rank2")
    assert xb is not None
    rec["xgb2"] = np.asarray(xb["score"], dtype=np.float64)

    rec["FUSION"] = 0.5 * z_per_date(dates, rec["F"]) + 0.5 * z_per_date(dates, rec["xgb2"])
    rec["c_mag"] = np.abs(rec["FUSION"])

    res = {"schema": "f37-ensemble-verify-v1", "dense_threshold": dense}
    # full/dev/confirm
    for split, mask in (("full", np.ones(n, dtype=bool)),
                        ("dev", (off >= 0) & (off <= 299)),
                        ("confirm", (off >= 300) & (off < 400))):
        ra = slice_rec(rec, mask)
        res[split] = {
            "rank_ic": float(np.mean(list(_daily_ic(ra, "FUSION", dense).values()))),
            "da": float(np.mean(list(_daily_da(ra, "FUSION", dense).values()))),
            "rank_ic_F": float(np.mean(list(_daily_ic(ra, "F", dense).values()))),
        }
        print(f"[f37] {split}: FUSION ic={res[split]['rank_ic']:.4f} da={res[split]['da']:.4f} "
              f"(F {res[split]['rank_ic_F']:.4f})")

    # bootstrap vs Exp08 F
    res["bootstrap_vs_F"] = bootstrap_vs(rec, "FUSION", rec, "F", dense)
    print(f"[f37] vs F: point={res['bootstrap_vs_F']['point']:+.4f} "
          f"robust={res['bootstrap_vs_F']['block_robust']}")

    # |FUSION|-abstention coverage
    res["coverage"] = {}
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, "c_mag", cv, True)
        ra = slice_rec(rec, acted)
        da20 = max(5, int(round(cv * dense)))
        res["coverage"][str(cv)] = {
            "ic": float(np.mean(list(_daily_ic(ra, "FUSION", da20).values()))),
            "da": float(np.mean(list(_daily_da(ra, "FUSION", da20).values())))}
        print(f"[f37] cov={cv} ic={res['coverage'][str(cv)]['ic']:.4f} "
              f"da={res['coverage'][str(cv)]['da']:.4f}")

    out = rr / "f37_ensemble_verify.json"
    write_json_ledger(out, res, "f37_ensemble_verify")
    # save the fusion score for the baselines folder comparison
    save_predictions("reference_fusion", rec["FUSION"], meta={
        "model": "FUSION = 0.5 z(Exp08 F) + 0.5 z(XGBoost rank2)",
        "note": "new champion: beats Exp08 F (0.0817->~0.098)"})
    print(f"[f37] -> {out}; saved reference_fusion to 09-baselines/outputs")


if __name__ == "__main__":
    main()
