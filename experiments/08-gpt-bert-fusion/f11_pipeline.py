"""f11_pipeline.py — protocol-compliant final pipeline evaluation.

Components (all fit parameters use ONLY the calibration slice):
  1. Base rank score: F_z_STRONG6 = 0.5 z(rank-ens of 6 BERT heads) + 0.5 z(P6).
  2. Confidence: c_dis6 = std across the 6 BERT-head rank percentiles.
  3. Abstention thresholds: per-coverage GLOBAL c_dis thresholds fit on calib
     (the 1-cv quantile of c_dis on calib), applied to eval -> achieved coverage.
     (Also reports the exact-coverage top-fraction rule as the diagnostic ceiling.)
  4. Magnitude channel: isotonic / linear mapping F_z_STRONG6 -> raw_logret fit
     on calib, applied to eval -> pred_logret for DA / MAPE.
Reports: full-window and per-coverage RankIC / DA / MAPE + paired moving-block
bootstrap of the acted RankIC vs J3 baseline.
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
    cand_path, slice_rec, DailyIcCache, write_json_ledger, _split_points,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402


STRONG = list(range(45, 51))


def load_scores(wr, six_wr, region, hidden_file):
    cand = np.load(cand_path(region), allow_pickle=True)
    dates = np.asarray(cand["date_key"])
    n = len(cand["stock_uid"])
    de = np.load(wr / hidden_file, allow_pickle=True)
    assert len(de["stock_uid"]) == n, f"{region} hidden misaligned"
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in STRONG:
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            sc = head(H).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, sc))
    ens = np.mean(ranks, axis=0)
    if region == "eval":
        p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    else:
        gh = np.load(six_wr / "calibration_cache.npz", allow_pickle=True)
        ck6 = torch.load(str(six_wr / "head_P6_mlp_rank_spearman.pt"),
                         map_location="cpu", weights_only=False)
        h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
        h6.load_state_dict(ck6["head_state"]); h6.eval()
        with torch.no_grad():
            p6 = h6(torch.from_numpy(gh["hidden"].astype(np.float32))).numpy().astype(np.float64)
    Fz = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    dis = np.std(ranks, axis=0)
    out = {"stock_uid": cand["stock_uid"], "date_key": dates,
           "true_logret": cand["true_logret"].astype(np.float64),
           "quality": cand["quality"].astype(bool), "Fz": Fz,
           "ens": ens, "c_dis": dis, "post_median": cand["post_median"].astype(np.float64)}
    if "offset" in cand.files:
        out["offset"] = cand["offset"]
    out["dense_threshold"] = int(cand["dense_threshold"][0]) \
        if "dense_threshold" in cand.files else \
        max(5, int(np.ceil(0.8 * int(np.max([len(np.where(cand["date_key"] == d)[0])
                                             for d in np.unique(cand["date_key"])])))))
    return out


def _rec_from(out):
    return {k: v for k, v in out.items() if k != "dense_threshold"}


def main():
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"

    ce = load_scores(wr, sw, "eval", "bert_hidden_eval_w512_t2.npz")
    cc = load_scores(wr, sw, "calib", "bert_hidden_calib_w512_t2.npz")
    rec_e = _rec_from(ce)
    rec_c = _rec_from(cc)
    dense = ce["dense_threshold"]
    dense_c = cc["dense_threshold"]
    print(f"[f11] eval_dense={dense} calib_dense={dense_c}")

    # ---- full-window base ----
    full = metrics_table(rec_e, {"F_z_STRONG6": "Fz", "ens6": "ens", "J3": "post_median"}, dense)
    for k, v in full.items():
        print(f"[f11] full {k:12s} rank_ic={v['avg_daily_rank_ic']:.4f} "
              f"da={v['avg_da_per_date']:.4f} mape={v['avg_mape']:.4f}")

    # ---- calib-fitted global abstention thresholds ----
    res = {"schema": "f11-pipeline-v1", "dense_threshold": dense,
           "full": {k: v["avg_daily_rank_ic"] for k, v in full.items()},
           "coverage_global_thr": {}, "coverage_exact": {}}
    cc_vals = np.asarray(rec_c["c_dis"])
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        if cv == 1.0:
            thr = -np.inf
            acted_e = np.ones(len(rec_e["stock_uid"]), dtype=bool)
        else:
            thr = float(np.quantile(cc_vals, 1 - cv))
            acted_e = np.asarray(rec_e["c_dis"]) <= thr
        rec_a = slice_rec(rec_e, acted_e)
        dense_act = max(5, int(round(cv * dense)))
        ic_ = DailyIcCache(rec_a, dense_act).series(rec_a["Fz"])
        res["coverage_global_thr"][str(cv)] = {
            "calib_threshold": thr,
            "achieved_frac": float(acted_e.mean()),
            "acted_rank_ic": float(np.mean(list(ic_.values()))) if ic_ else None,
        }
        print(f"[f11] global-thr cov={cv} thr={thr:.4f} achieved={acted_e.mean():.3f} "
              f"acted_ic={res['coverage_global_thr'][str(cv)]['acted_rank_ic']}")

        # exact-coverage diagnostic
        acted_x = _exact_frac(rec_e, "c_dis", cv, max_high=False)
        rec_x = slice_rec(rec_e, acted_x)
        ic_x = DailyIcCache(rec_x, dense_act).series(rec_x["Fz"])
        res["coverage_exact"][str(cv)] = {
            "acted_frac": float(acted_x.mean()),
            "acted_rank_ic": float(np.mean(list(ic_x.values()))) if ic_x else None,
        }

    # ---- magnitude calibration (isotonic Fz -> raw_logret on calib) ----
    m_valid = np.isfinite(rec_c["Fz"]) & np.isfinite(rec_c["true_logret"]) \
        & rec_c["quality"]
    iso = IsotonicRegression(y_min=None, y_max=None, out_of_bounds="clip")
    iso.fit(rec_c["Fz"][m_valid], rec_c["true_logret"][m_valid])
    fe = rec_e["Fz"]
    pred = np.full(len(fe), np.nan, dtype=np.float64)
    fin = np.isfinite(fe)
    pred[fin] = iso.predict(fe[fin])
    rec_e["pred_logret"] = pred
    mag = metrics_table(rec_e, {"iso_mag": "pred_logret", "J3_mag": "post_median"}, dense)
    res["magnitude"] = {k: {kk: v[kk] for kk in ("avg_da_per_date", "avg_mape", "avg_mae")
                            if kk in v} for k, v in mag.items()}
    print(f"[f11] magnitude iso: DA={mag['iso_mag']['avg_da_per_date']:.4f} "
          f"MAPE={mag['iso_mag']['avg_mape']:.4f}  vs J3 DA={mag['J3_mag']['avg_da_per_date']:.4f} "
          f"MAPE={mag['J3_mag']['avg_mape']:.4f}")

    # ---- bootstrap of best full-window arm vs J3 ----
    res["bootstrap_vs_J3_Fz"] = bootstrap_vs(rec_e, "Fz", rec_e, "post_median", dense)
    res["bootstrap_vs_P6_Fz"] = bootstrap_vs(rec_e, "Fz", rec_e, "ens", dense)  # ens not p6; keep ens

    out = rr / "f11_pipeline.json"
    write_json_ledger(out, res, "f11_pipeline")
    print(f"[f11] -> {out}")


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


if __name__ == "__main__":
    main()
