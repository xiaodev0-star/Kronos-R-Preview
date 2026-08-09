"""f5_calib_fit.py — Round 2: fit fusion weight + abstention thresholds on calib.

Protocol: every fitted parameter uses only the calibration slice
(audit_uids x [2023-02-01, 2024-02-01) = candidates_calib rows).  0..399 never
participates in fitting.

Two fits:
  1. BLEND weight w: score = w*rank(BERT_head) + (1-w)*rank(P6)  [or z-sum],
     w chosen on calib by maximizing avg daily RankIC over dense calib dates.
     Then evaluate that SAME w on eval 0..399 (paired bootstrap vs T5/P6).
     Also reports the calib-vs-eval optimal-weight drift (07 R2/R5 concern).
  2. ABSTENTION thresholds: for confidence signals (JS, entropy, std, |Pup-0.5|),
     per-coverage-target thresholds fit on calib, then applied to eval and the
     acted RankIC/DA reported with baseline on the same acted subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, DailyIcCache, slice_rec, write_json_ledger,
)
from f0_scores import compute_region, rank_pct_per_date  # noqa: E402


def _fit_blend_w(calib_rec, dense):
    """Choose w on calib maximizing daily RankIC of w*rank_bert+(1-w)*rank_p6."""
    cc = DailyIcCache(calib_rec, dense)
    best = None
    grid = {}
    for w in np.arange(0.0, 1.001, 0.05):
        w = float(round(w, 2))
        s = w * calib_rec["rank_bert"] + (1 - w) * calib_rec["rank_p6"]
        ic = np.mean(list(cc.series(s).values()))
        grid[w] = float(ic)
        if best is None or ic > best[1]:
            best = (w, ic)
    return best, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blend", "abstain", "all"], default="all")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    # ---- ensure score caches exist ----
    if not (wr / "scores_eval_fused.npz").exists():
        compute_region("eval", wr, ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42")
    if not (wr / "scores_calib_fused.npz").exists():
        compute_region("calib", wr, ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42")

    ce = np.load(wr / "scores_eval_fused.npz", allow_pickle=True)
    cc = np.load(wr / "scores_calib_fused.npz", allow_pickle=True)
    rec_e = {k: ce[k] for k in ce.files if ce[k].shape != (1,)}
    rec_c = {k: cc[k] for k in cc.files if cc[k].shape != (1,)}
    dense = int(ce["dense_threshold"][0])
    dense_c = int(cc["dense_threshold"][0])
    print(f"[f5] eval_dense={dense} calib_dense={dense_c}")

    res = {"schema": "f5-calib-fit-v1", "dense_threshold": dense}

    # ---- 1) blend weight ----
    if args.mode in ("blend", "all"):
        (w, ic_calib), grid = _fit_blend_w(rec_c, dense_c)
        res["blend"] = {"w_rank": w, "calib_grid_rankic": grid,
                        "calib_best_rankic": ic_calib}
        print(f"[f5] calib-fit w={w} calib_ic={ic_calib:.4f}")

        # evaluate this w on eval
        field = f"F_w{w:g}"
        rec_e[field] = w * rec_e["rank_bert"] + (1 - w) * rec_e["rank_p6"]
        res["blend"]["eval_rankic"] = float(
            metrics_table(rec_e, {field: field}, dense)[field]["avg_daily_rank_ic"])
        res["blend"]["eval_da"] = float(
            metrics_table(rec_e, {field: field}, dense)[field]["avg_da_per_date"])
        res["blend"]["bootstrap_vs_T5"] = bootstrap_vs(rec_e, field, rec_e, "bert_head", dense)
        res["blend"]["bootstrap_vs_P6"] = bootstrap_vs(rec_e, field, rec_e, "p6_score", dense)
        # eval-optimal w for drift diagnosis
        _, grid_e = _fit_blend_w(rec_e, dense)
        res["blend"]["eval_optimal_w"] = grid_e
        print(f"[f5] eval RankIC(w={w:g})={res['blend']['eval_rankic']:.4f}")

    # ---- 2) abstention thresholds ----
    if args.mode in ("abstain", "all"):
        # confidence signals on calib & eval
        for region, rec in (("calib", rec_c), ("eval", rec_e)):
            s = np.load(wr / f"scores_{region}_K8_w512_stride1.npz", allow_pickle=True)
            logp = s["logp_bert_full"]
            logpb = np.asarray(logp)
            ent = -np.sum(np.where(np.isfinite(logpb), np.exp(np.clip(logpb, -50, 50)) * logpb, 0.0),
                          axis=1)
            rec["c_entropy"] = ent
            if region == "eval":
                qg = np.load(wr / "gpt_q_eval_full128.npz", allow_pickle=True)["q"]
            else:
                qg = np.load(cand_path("calib"), allow_pickle=True)["gpt_q"]
            pb = np.exp(np.clip(np.asarray(logp), -50, 50))
            pb = pb / np.maximum(pb.sum(axis=1, keepdims=True), 1e-12)
            eps = 1e-12
            m = 0.5 * (pb + qg)
            js = 0.5 * (np.sum(pb * np.log((pb + eps) / (m + eps)), axis=1)
                        + np.sum(qg * np.log((qg + eps) / (m + eps)), axis=1))
            rec["c_js"] = js
            rec["c_std"] = np.asarray(rec["post_std"], dtype=np.float64)
            rec["c_pabs"] = np.abs(np.asarray(rec["p_up"], dtype=np.float64) - 0.5)

        # fit per-coverage thresholds on calib for each signal, apply to eval
        score_field = "F_zsum"
        res["abstention"] = {}
        for cf, max_high in (("c_js", False), ("c_entropy", False),
                             ("c_std", False), ("c_pabs", True)):
            row = {"signal": cf, "max_high": max_high, "coverage": {}}
            for cv in (0.8, 0.6, 0.4, 0.2):
                # calib: per-date keep top cv fraction by confidence
                thr_info = _per_date_threshold(rec_c, cf, cv, max_high=max_high)
                # apply same per-date keep rule on eval
                acted = _apply_rule(rec_e, cf, cv, max_high=max_high)
                if acted.sum() < 100:
                    continue
                rec_e_a = slice_rec(rec_e, acted)
                # coverage-scaled dense threshold: acted date must retain >= cv*dense_full
                dense_act = max(5, int(round(cv * dense)))
                aic = np.mean(list(DailyIcCache(rec_e_a, dense_act).series(rec_e_a[score_field]).values()))
                row["coverage"][str(cv)] = {
                    "calib_thr_median": float(np.nanmedian(thr_info["thr"])),
                    "eval_acted_frac": float(acted.mean()),
                    "eval_acted_rankic": float(aic),
                    "eval_acted_dense_threshold": dense_act,
                }
                print(f"[f5] {cf} cov={cv} acted_ic={aic:.4f}")
            res["abstention"][cf] = row

    out = rr / f"f5_calib_fit{('_' + args.tag) if args.tag else ''}.json"
    write_json_ledger(out, res, "f5_calib_fit", mode=args.mode, tag=args.tag)
    print(f"[f5] -> {out}")


def _per_date_threshold(rec, cf, cv, max_high=True):
    conf = np.asarray(rec[cf], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    thr = np.full(len(rec["stock_uid"]), np.nan)
    for d in np.unique(dates):
        dm = dates == d
        cvals = conf[dm]
        m = np.isfinite(cvals)
        if m.sum() == 0:
            continue
        keep_n = max(1, int(round(cv * m.sum())))
        ord_ = np.argsort(-cvals) if max_high else np.argsort(cvals)
        thr[dm] = cvals[ord_[keep_n - 1]] if keep_n <= m.sum() else cvals[ord_[-1]]
    return {"thr": thr}


def _apply_rule(rec, cf, cv, max_high=True):
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
