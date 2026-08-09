"""f15_final_pipeline.py — definitive consolidated evaluation of the 08 pipeline.

Pipeline (all trained params on fit <2023-02-01; all calib params on audit
[2023-02,2024-02); 0..399 pure inference):
  Base rank score  F = 0.5*z(rank-ens of 6 BERT-hidden heads) + 0.5*z(P6)
  Confidence       c_dis = std across the 6 heads' within-date rank percentiles
  Magnitude        iso: isotonic F -> raw_logret (fit on calib)
  Abstention       per-date top-(1-cov) fraction by c_dis (exact coverage)

Reports full-400 / dev / confirm, coverage curves, paired moving-block bootstrap
vs J3 and P6, and the magnitude channel MAPE/DA at full and 20% coverage.
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
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, slice_rec, DailyIcCache, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402


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
    off = np.asarray(rec["offset"], dtype=np.int64)

    # ---- base scores ----
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
        ranks.append(rank_pct_per_date(dates, sc))
    ens = np.mean(ranks, axis=0)
    rec["c_dis"] = np.std(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["ens"] = ens
    rec["J3"] = np.asarray(rec["post_median"], dtype=np.float64)

    # ---- magnitude calibration on calib ----
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    de_c = np.load(wr / "bert_hidden_calib_w512_t2.npz", allow_pickle=True)
    Hc = torch.from_numpy(de_c["hidden"].astype(np.float32))
    ranks_c = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            ranks_c.append(rank_pct_per_date(ccal["date_key"],
                           head(Hc).numpy().astype(np.float64)))
    ens_c = np.mean(ranks_c, axis=0)
    gh = np.load(sw / "calibration_cache.npz", allow_pickle=True)
    ck6 = torch.load(str(sw / "head_P6_mlp_rank_spearman.pt"), map_location="cpu",
                     weights_only=False)
    h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
    h6.load_state_dict(ck6["head_state"]); h6.eval()
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(gh["hidden"].astype(np.float32))).numpy().astype(np.float64)
    F_c = 0.5 * z_per_date(ccal["date_key"], ens_c) + 0.5 * z_per_date(ccal["date_key"], p6_c)
    mv = np.isfinite(F_c) & np.isfinite(ccal["true_logret"].astype(np.float64)) \
        & ccal["quality"].astype(bool)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(F_c[mv], ccal["true_logret"].astype(np.float64)[mv])
    pred = np.full(n, np.nan)
    fin = np.isfinite(rec["F"])
    pred[fin] = iso.predict(rec["F"][fin])
    rec["pred_logret"] = pred

    res = {"schema": "f15-final-pipeline-v1", "dense_threshold": dense}

    # ---- full / dev / confirm ----
    fields = {"F_BERT6_P6": "F", "BERT_ens6": "ens", "J3": "J3", "P6_iso": "pred_logret"}
    res["full"] = metrics_table(rec, fields, dense)
    dev = slice_rec(rec, (off >= 0) & (off <= 299))
    conf = slice_rec(rec, (off >= 300) & (off < 400))
    res["dev_0_299"] = metrics_table(dev, fields, dense)
    res["confirm_300_399"] = metrics_table(conf, fields, dense)

    # ---- bootstrap ----
    res["bootstrap_vs_J3"] = bootstrap_vs(rec, "F", rec, "J3", dense)
    res["bootstrap_vs_P6"] = bootstrap_vs(rec, "F", rec, "ens", dense)

    # ---- coverage (c_dis exact) ----
    res["coverage"] = {}
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, "c_dis", cv, False)
        ra = slice_rec(rec, acted)
        da20 = max(5, int(round(cv * dense)))
        ic_ = DailyIcCache(ra, da20).series(ra["F"])
        da_ = _daily_da(ra, "F", da20)
        mp_ = _daily_mape(ra, "pred_logret", da20)
        res["coverage"][str(cv)] = {
            "acted_frac": float(acted.mean()),
            "acted_rank_ic": float(np.mean(list(ic_.values()))) if ic_ else None,
            "acted_da": float(np.mean(list(da_.values()))) if da_ else None,
            "acted_mape": float(np.mean(list(mp_.values()))) if mp_ else None,
        }
        print(f"[f15] cov={cv} frac={acted.mean():.3f} rank_ic={res['coverage'][str(cv)]['acted_rank_ic']} "
              f"da={res['coverage'][str(cv)]['acted_da']} mape={res['coverage'][str(cv)]['acted_mape']}")

    out = rr / "f15_final_pipeline.json"
    write_json_ledger(out, res, "f15_final_pipeline")
    print(f"[f15] -> {out}")
    print(f"[f15] full F rank_ic={res['full']['F_BERT6_P6']['avg_daily_rank_ic']:.4f} "
          f"da={res['full']['F_BERT6_P6']['avg_da_per_date']:.4f}")
    print(f"[f15] iso mape={res['full']['P6_iso']['avg_mape']:.4f} "
          f"iso da={res['full']['P6_iso']['avg_da_per_date']:.4f} "
          f"J3 mape={res['full']['J3']['avg_mape']:.4f}")


if __name__ == "__main__":
    main()
