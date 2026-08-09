"""f28_final_variants.py — two final refinement probes.

A) Direction-split magnitude: calibrate iso(F -> logret) separately for
   positive-F and negative-F rows (asymmetric return behavior).
B) Composite confidence: |F| * (1 - norm c_dis) as the abstention score — keep
   rows that are BOTH high-conviction AND high-agreement.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402


def _daily_mape(rec, field, dense):
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
        out[uniq[i]] = float(np.mean(np.abs(np.exp(s[lo:hi][m] - t[lo:hi][m]) - 1.0)))
    return out


def _daily_ic(rec, field, dense):
    from scipy.stats import spearmanr
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
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])

    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    raw = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            sc = head(H).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, sc))
        raw.append(sc)
    ens = np.mean(ranks, axis=0)
    rec["c_dis"] = np.std(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])

    # ---- B) composite confidence ----
    rd = rank_pct_per_date(dates, rec["c_dis"])   # low = agree
    rec["c_comp"] = np.abs(rec["F"]) * (1.0 - rd)
    res = {"schema": "f28-final-variants-v1", "dense_threshold": dense,
           "abstention": {}, "magnitude": {}}
    for cf, max_high in (("c_mag", True), ("c_comp", True)):
        row = []
        for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
            acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, cf, cv, max_high)
            ra = slice_rec(rec, acted)
            da20 = max(5, int(round(cv * dense)))
            ic_ = _daily_ic(ra, "F", da20)
            row.append({"cov": cv, "ic": float(np.mean(list(ic_.values()))) if ic_ else None})
        res["abstention"][cf] = row
        print(f"[f28] {cf:8s}: " + " ".join(f"cov={r['cov']} ic={r['ic'] and round(r['ic'],4)}" for r in row))

    # ---- A) direction-split magnitude ----
    cc = np.load(cand_path("calib"), allow_pickle=True)
    de_c = np.load(wr / "bert_hidden_calib_w512_t2.npz", allow_pickle=True)
    Hc = torch.from_numpy(de_c["hidden"].astype(np.float32))
    ranks_c = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            ranks_c.append(rank_pct_per_date(cc["date_key"], head(Hc).numpy().astype(np.float64)))
    ens_c = np.mean(ranks_c, axis=0)
    gh = np.load(sw / "calibration_cache.npz", allow_pickle=True)
    ck6 = torch.load(str(sw / "head_P6_mlp_rank_spearman.pt"), map_location="cpu", weights_only=False)
    h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
    h6.load_state_dict(ck6["head_state"]); h6.eval()
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(gh["hidden"].astype(np.float32))).numpy().astype(np.float64)
    F_c = 0.5 * z_per_date(cc["date_key"], ens_c) + 0.5 * z_per_date(cc["date_key"], p6_c)
    y_c = cc["true_logret"].astype(np.float64)
    mv = np.isfinite(F_c) & np.isfinite(y_c) & cc["quality"].astype(bool)

    # single isotonic
    iso1 = IsotonicRegression(out_of_bounds="clip")
    iso1.fit(F_c[mv], y_c[mv])
    pred1 = np.full(n, np.nan)
    fin = np.isfinite(rec["F"])
    pred1[fin] = iso1.predict(rec["F"][fin])
    # direction-split
    iso_p = IsotonicRegression(out_of_bounds="clip")
    iso_n = IsotonicRegression(out_of_bounds="clip")
    mp = mv & (F_c >= 0); mn = mv & (F_c < 0)
    iso_p.fit(F_c[mp], y_c[mp]); iso_n.fit(F_c[mn], y_c[mn])
    pred2 = np.full(n, np.nan)
    Ff = rec["F"]
    pred2[fin & (Ff >= 0)] = iso_p.predict(Ff[fin & (Ff >= 0)])
    pred2[fin & (Ff < 0)] = iso_n.predict(Ff[fin & (Ff < 0)])
    for name, p in (("M_iso", pred1), ("M_dirsplit", pred2)):
        rec[name] = p
        mp_ = np.mean(list(_daily_mape(rec, name, dense).values()))
        res["magnitude"][name] = {"mape": float(mp_)}
        print(f"[f28] {name:12s} mape={mp_:.4f}")

    out = rr / "f28_final_variants.json"
    write_json_ledger(out, res, "f28_final_variants")
    print(f"[f28] -> {out}")


if __name__ == "__main__":
    main()
