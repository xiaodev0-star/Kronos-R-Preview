"""07-C: Fusion experiments + R-series analysis.

Usage is intentionally documented in README.md; this file is the C-stage
entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# -- Path bootstrap --
ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent

for _p in (SEVEN, ROOT):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from scipy.stats import spearmanr  # noqa: E402

# -- Imports from common.py --
from common import (  # noqa: E402
    VOCAB_BASE, resolve_roots, write_json, append_trial, arm_metrics,
    weights_artifact, result_artifact, stage_weights, stage_results,
    posttrain_artifacts,
)

# -- Imports from common.py (merged improve_common) --
from common import (  # noqa: E402
    EPS, CENTERS_PATH, PT01_REC, P6_HEAD,
    cand_path, scores_path, gpt_q_path, weights_root, results_root,
    load_centers, softmax_rows, decode_coarse, poe_fused,
    build_rec, eval_dev_confirm, slice_rec,
    metrics_table, summarize, bootstrap_vs, calib_dense_threshold,
    daily_rank_ic_series, write_json_ledger,
    DailyIcCache, write_prediction_parquet,
    MODEL_VARIANTS, model_variant, PredictionParquetWriter,
    require_full_validation_coverage,
)

# -- Imports from experiment_io --
from experiment_io import assert_results_boundary  # noqa: E402

# -- Constants --
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ============================================================================
# Abstention constants (from 06-posttrain/b-frozen-heads/abstain.py)
# ============================================================================

COVERAGES = (100, 80, 60, 40, 20)


# ============================================================================
# R-series shared loads (from r_series.py)
# ============================================================================

SCORES_SUFFIX = ""   # set by --scores-suffix (e.g. scoring-aligned)


def _current_model_name(model_name=None):
    requested = model_name or SCORES_SUFFIX or "BERT"
    try:
        return model_variant(requested)["name"]
    except ValueError:
        return str(requested)


def _model_result_path(stem, model_name=None):
    return stage_results("C") / f"{stem}-{_current_model_name(model_name)}.json"


def _load_eval():
    cand = np.load(cand_path("eval"), allow_pickle=True)
    sc = np.load(scores_path("eval", suffix=SCORES_SUFFIX or "BERT"), allow_pickle=True)
    pt01 = np.load(PT01_REC, allow_pickle=True)
    return cand, sc, pt01


def _load_calib():
    cc = np.load(cand_path("calib"), allow_pickle=True)
    cs = np.load(scores_path("calib", suffix=SCORES_SUFFIX or "BERT"), allow_pickle=True)
    return cc, cs


def _p6_eval():
    return np.load(weights_artifact("p6-scores"))


def _attach_p6(rec, p6):
    if len(p6) == len(rec["stock_uid"]):
        rec["p6_score"] = p6.astype(np.float64)
    else:
        raise RuntimeError(f"p6 len {len(p6)} != rec {len(rec['stock_uid'])}")


# ============================================================================
# R1 -- BERT posterior decode trio (from r_series.py)
# ============================================================================

def r1(cand, sc, pt01, p6, centers, calib=False, out_json=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp = sc["logp_bert_full"]
    dec = decode_coarse(logp, centers, rec["p_mean0"], rec["p_std0"], log_space=True)
    rec["ebert_mean"] = dec["e_mean"]
    rec["ebert_median"] = dec["e_median"]
    rec["ebert_pup"] = dec["p_up_raw"]
    rec["ebert_pup_naive"] = dec["p_up_naive"]

    fields = {
        "E_BERT_mean": "ebert_mean",
        "E_BERT_median": "ebert_median",
        "P_BERT_up_raw": "ebert_pup",
        "P_BERT_up_naive": "ebert_pup_naive",
        "J3_median": "post_median",
        "J4_pup": "p_up",
        "J2_mean": "post_mean",
        "J0_greedy": "greedy_return",
        "P6": "p6_score",
    }
    dense = int(cand["dense_threshold"][0])
    refs = {"vs_J3": (rec, "post_median"),
            "vs_P6": (rec, "p6_score"),
            "vs_J2": (rec, "post_mean")}
    res = summarize(rec, fields, dense, refs=refs, label="R1_eval")
    if out_json:
        write_json_ledger(out_json, res, "r1")
    # calib diagnostics (no J-reference there; post fields are NaN in light cache).
    # Written after the eval result so a missing calib score cache (e.g. a
    # fine-tuned BERT that only rescored eval) never loses the eval summary.
    if calib:
        try:
            cc, cs = _load_calib()
            rec_c = build_rec(cc)
            dec_c = decode_coarse(cs["logp_bert_full"], centers,
                                  rec_c["p_mean0"], rec_c["p_std0"], log_space=True)
            rec_c["ebert_mean"] = dec_c["e_mean"]
            rec_c["ebert_median"] = dec_c["e_median"]
            rec_c["ebert_pup"] = dec_c["p_up_raw"]
            rec_c["ebert_pup_naive"] = dec_c["p_up_naive"]
            res["calib"] = metrics_table(rec_c, {
                "E_BERT_mean": "ebert_mean", "E_BERT_median": "ebert_median",
                "P_BERT_up_raw": "ebert_pup", "P_BERT_up_naive": "ebert_pup_naive",
            }, calib_dense_threshold())
            write_json_ledger(out_json, res, "r1_calib")
        except Exception as e:  # noqa: BLE001
            res["calib"] = {"error": str(e)}
    return res, rec


# ============================================================================
# R3 (pure-BERT select arms) -- CPU (from r_series.py)
# ============================================================================

def r3_bert(cand, sc, pt01, p6, centers, out_json=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp = sc["logp_bert_full"]
    topk = cand["topk_ids"].astype(np.int64)                     # [N, 8]
    rows = np.arange(len(rec["stock_uid"]))
    # BERT's log-p on GPT's candidate set
    lp_topk = np.take_along_axis(logp, topk, axis=1)             # [N, 8]
    w = softmax_rows(lp_topk)                                    # within-set weights
    best_k = np.argmax(lp_topk, axis=1)                          # BERT's pick
    cs = np.asarray(centers, dtype=np.float64)
    center_topk = cs[topk]                                       # [N, 8] normalized
    pstd = np.maximum(rec["p_std0"], EPS)
    # R3a: decoded center of BERT's single pick
    pick_center = center_topk[rows, best_k]
    rec["critic_pick_center"] = pick_center * pstd + rec["p_mean0"]
    # R3b: top-8 BERT-weighted expected return (decoded)
    weighted = np.sum(w * center_topk, axis=1)
    rec["critic_weighted_e"] = weighted * pstd + rec["p_mean0"]

    fields = {
        "critic_pick_center": "critic_pick_center",
        "critic_weighted_e": "critic_weighted_e",
        "J3_median": "post_median",
        "P6": "p6_score",
    }
    dense = int(cand["dense_threshold"][0])
    res = summarize(rec, fields, dense,
                    refs={"vs_J3": (rec, "post_median"),
                          "vs_P6": (rec, "p6_score")},
                    label="R3_bert")
    if out_json:
        write_json_ledger(out_json, res, "r3_bert")
    return res, rec


# ============================================================================
# R5 -- P(up) direction fusion (w fit on calib) (from r_series.py)
# ============================================================================

def r5(cand, sc, pt01, p6, centers, out_json=None, calib=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp = sc["logp_bert_full"]
    dec = decode_coarse(logp, centers, rec["p_mean0"], rec["p_std0"], log_space=True)
    pup_bert = dec["p_up_raw"]
    pup_gpt = np.asarray(rec["p_up"], dtype=np.float64)

    # calib-fit weight w over P_GPT(up) (decoded from full q) and P_BERT(up)
    fitted_w, curve = None, None
    if calib:
        cc, cs = _load_calib()
        rec_c = build_rec(cc)
        q_c = cc["gpt_q"].astype(np.float64)
        cs_centers = np.asarray(centers, dtype=np.float64)[None, :]
        thr_c = -rec_c["p_mean0"] / np.maximum(rec_c["p_std0"], EPS)
        pup_gpt_c = np.sum(q_c * (cs_centers > thr_c[:, None]), axis=1)
        dec_c = decode_coarse(cs["logp_bert_full"], centers,
                              rec_c["p_mean0"], rec_c["p_std0"], log_space=True)
        pup_bert_c = dec_c["p_up_raw"]
        dense_c = calib_dense_threshold()
        curve = {}
        best_w, best_ic = None, -1e9
        for wv in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            rec_c["f"] = wv * pup_gpt_c + (1.0 - wv) * pup_bert_c
            m = metrics_table(rec_c, {"fused_pup": "f"}, dense_c)["fused_pup"]
            ic = m["avg_daily_rank_ic"]
            curve[str(wv)] = {"w": wv, "calib_rank_ic": ic,
                              "calib_da": m["avg_da_per_date"]}
            if ic is not None and ic > best_ic:
                best_ic, best_w = ic, wv
        fitted_w = best_w
        print(f"[r5] calib-fitted w(P_GPT) = {fitted_w} (ic {best_ic:.4f})")

    rec["pup_avg"] = 0.5 * pup_gpt + 0.5 * pup_bert
    if fitted_w is not None:
        rec["pup_fit"] = fitted_w * pup_gpt + (1.0 - fitted_w) * pup_bert
    fields = {"P_GPT_up_J4": "p_up",
              "P_BERT_up": "ebert_pup",
              "P_up_avg_0.5": "pup_avg"}
    if fitted_w is not None:
        fields[f"P_up_fit_w{fitted_w:.1f}"] = "pup_fit"
    fields["P6"] = "p6_score"
    dense = int(cand["dense_threshold"][0])
    res = summarize(rec, fields, dense,
                    refs={"vs_J4": (rec, "p_up"),
                          "vs_P6": (rec, "p6_score")},
                    label="R5")
    res["calib_fitted_w"] = fitted_w
    res["calib_w_curve"] = curve
    if out_json:
        write_json_ledger(out_json, res, "r5", w=fitted_w)
    return res, rec


# ============================================================================
# R4 -- abstention curves (from r_series.py, abstain_curve inlined)
# ============================================================================

def _bert_entropy(logp, chunk=300_000):
    n = int(np.asarray(logp).shape[0])
    out = np.full(n, np.nan)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p = softmax_rows(logp[start:stop])
        fin = np.isfinite(p).all(axis=1)
        ent = -np.sum(np.where(fin[:, None], p, 0.0)
                      * np.log(np.maximum(np.where(fin[:, None], p, 1.0), EPS)), axis=1)
        out[start:stop] = np.where(fin, ent, np.nan)
    return out


def _f6_intersection(cand, logp):
    n = len(cand["stock_uid"])
    gk = np.argsort(-cand["topk_logq"], axis=1, kind="stable")[:, :8]
    bk = np.argsort(-logp, axis=1, kind="stable")[:, :8]
    inter = np.zeros(n, dtype=np.int64)
    for kk in range(8):
        inter += (bk[:, kk, None] == gk).any(axis=1)
    return inter.astype(np.float64)


def _abstain_curve(records, score_field, conf, dense_min=3634, max_cs=None):
    """Acted DA/RankIC vs coverage using the given confidence array.

    Inlined from 06-posttrain/b-frozen-heads/abstain.py.

    For each coverage, within each date keep the top-coverage fraction of stocks
    by ``conf`` and compute DA/RankIC on the acted subset.  The baseline
    (score_field) is recomputed on the identical acted subset.  The dense
    threshold scales with coverage (an acted subset is naturally smaller).
    """
    dates = np.asarray(records["date_key"])
    score = np.asarray(records[score_field])
    true = np.asarray(records["true_logret"])
    conf = np.asarray(conf)
    uniq, inv = np.unique(dates, return_inverse=True)
    max_cs = max_cs or int(max(np.bincount(inv)))
    out = {}
    for cov in COVERAGES:
        acted = np.zeros(len(dates), dtype=bool)
        for i in range(len(uniq)):
            m = inv == i
            idx = np.where(m)[0]
            if len(idx) < dense_min:
                continue
            k = max(1, int(round(len(idx) * cov / 100.0)))
            order = np.argsort(-conf[idx], kind="stable")
            acted[idx[order[:k]]] = True
        s_act = score[acted]
        t_act = true[acted]
        d_act = dates[acted]
        # dense threshold proportional to the acted subset size at this coverage
        acted_min = max(5, int(np.ceil(0.8 * max_cs * cov / 100.0)))
        ic, da, mae, cnt, dense = daily_rank_ic(s_act, t_act, d_act, acted_min)
        out[str(cov)] = {
            "coverage": cov / 100.0,
            "n_acted": int(acted.sum()),
            "acted_dense_min": acted_min,
            "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
            "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        }
    return out


def r4(cand, sc, pt01, p6, centers, out_json=None, gpt_q_eval=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp = sc["logp_bert_full"]
    # confidence features (higher = more confident)
    conf = {
        "bert_entropy_neg": -_bert_entropy(logp),      # lower entropy = higher conf
        "bert_margin": sc["bert_margin"],
        "f6_gpt_bert_intersection": _f6_intersection(cand, logp),
        "q1_gpt_06": np.abs(rec["post_mean"]) / np.sqrt(
            np.maximum(rec["post_std"] ** 2, 1e-12)),  # 06 baseline
    }
    if gpt_q_eval is not None:
        # JS(p_bert || q_gpt) -- divergence high = disagree = low confidence
        q_all = np.asarray(gpt_q_eval, dtype=np.float32)
        q_all = q_all / q_all.sum(axis=1, keepdims=True)
        n = int(np.asarray(logp).shape[0])
        js = np.full(n, np.nan)
        for start in range(0, n, 300_000):
            stop = min(start + 300_000, n)
            p = softmax_rows(logp[start:stop])
            fin = np.isfinite(p).all(axis=1)
            q = np.maximum(np.where(fin[:, None], q_all[start:stop], 1.0), EPS)
            q = q / q.sum(axis=1, keepdims=True)
            p0 = np.where(fin[:, None], p, 1.0)
            m = 0.5 * (p0 + q)
            js_chunk = 0.5 * (
                np.sum(p0 * np.log(np.maximum(p0, EPS) / np.maximum(m, EPS)), axis=1)
                + np.sum(q * np.log(q / np.maximum(m, EPS)), axis=1))
            js[start:stop] = np.where(fin, js_chunk, np.nan)
        conf["js_bert_gpt_neg"] = -js

    dense = int(cand["dense_threshold"][0])
    res = {"schema": "r4-abstention-v1", "dense_min": dense,
           "score_fields": {"J3": "post_median", "P6": "p6_score"}}
    for sf_name, sf in (("J3", "post_median"), ("P6", "p6_score")):
        res[sf_name] = {}
        for cname, cval in conf.items():
            curve = _abstain_curve(rec, sf, cval, dense_min=dense)
            covs = [float(curve[str(c)]["coverage"]) for c in (100, 80, 60, 40, 20)]
            ics = [curve[str(c)]["avg_daily_rank_ic"] for c in (100, 80, 60, 40, 20)]
            auc = float(np.trapz(ics, covs))  # integral over coverage
            res[sf_name][cname] = {"curve": curve, "auc_rank_ic": auc,
                                   "rankic_at_20": ics[-1]}
    if out_json:
        write_json_ledger(out_json, res, "r4")
    return res


# ============================================================================
# R2 -- full-128 PoE fusion (from r_series.py)
# ============================================================================

R2_LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load_gpt_q_eval():
    p = gpt_q_path("eval")
    if not p.exists():
        raise RuntimeError(
            f"{p} missing -- run build_gpt_full_q.py first (R2/R3poe need GPT full q)")
    return np.load(p, allow_pickle=True)["q"]


def r2(cand, sc, pt01, p6, centers, out_json=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp_e = sc["logp_bert_full"]
    q_e = _load_gpt_q_eval()

    # ---- lambda selection on the calibration slice (full-128 PoE) ----
    cc, cs = _load_calib()
    rec_c = build_rec(cc)
    q_c = cc["gpt_q"].astype(np.float64)
    logp_c = cs["logp_bert_full"]
    dense_c = calib_dense_threshold()
    lam_scores = {}
    best, best_ic = None, -1e9
    for lam in R2_LAM_GRID:
        pf = poe_fused(logp_c, q_c, lam)
        dec = decode_coarse(pf, centers, rec_c["p_mean0"], rec_c["p_std0"])
        rec_c["f"] = dec["e_mean"]
        m = metrics_table(rec_c, {"poe_e": "f"}, dense_c)["poe_e"]
        ic = m["avg_daily_rank_ic"]
        lam_scores[str(lam)] = {"lambda": lam, "calib_rank_ic": ic,
                                "calib_da": m["avg_da_per_date"]}
        if ic is not None and ic > best_ic:
            best_ic, best = ic, lam
    print(f"[r2] calib-chosen lambda = {best} (calib IC {best_ic:.4f})")

    # ---- apply chosen lambda (and neighbors) to eval ----
    evals = {}
    for lam in sorted({0.0, best, round(best - 0.1, 1), round(best + 0.1, 1)}):
        if lam < 0 or lam > 1:
            continue
        pf = poe_fused(logp_e, q_e, lam)
        dec = decode_coarse(pf, centers, rec["p_mean0"], rec["p_std0"])
        rec[f"poe_e_l{lam:.1f}"] = dec["e_mean"]
        rec[f"poe_med_l{lam:.1f}"] = dec["e_median"]
        rec[f"poe_pup_l{lam:.1f}"] = dec["p_up_raw"]
        evals[lam] = "poe_e_l{:.1f}".format(lam)

    fields = {}
    for lam, fld in evals.items():
        fields[f"PoE_e_lambda_{lam:.1f}"] = fld
    fields.update({"J3_median": "post_median", "J2_mean": "post_mean",
                   "J4_pup": "p_up", "P6": "p6_score"})
    dense = int(cand["dense_threshold"][0])
    res = summarize(rec, fields, dense,
                    refs={"vs_J3": (rec, "post_median"),
                          "vs_P6": (rec, "p6_score"),
                          "vs_J2": (rec, "post_mean")},
                    label="R2")
    res["calib_chosen_lambda"] = best
    res["calib_chosen_lambda_ic"] = best_ic
    res["calib_lambda_scores"] = lam_scores
    res["lambda_interior"] = bool(0.0 < best < 1.0)
    if out_json:
        write_json_ledger(out_json, res, "r2", lam=best)
    return res, rec


# ============================================================================
# R3 (PoE-select arms) -- needs gpt_q (from r_series.py)
# ============================================================================

def r3_poe(cand, sc, pt01, p6, centers, out_json=None, lam=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    q_e = _load_gpt_q_eval()
    if lam is None:
        lam = 0.5
    pf = poe_fused(sc["logp_bert_full"], q_e, lam)
    topk = cand["topk_ids"].astype(np.int64)
    rows = np.arange(len(rec["stock_uid"]))
    pf_topk = np.take_along_axis(pf, topk, axis=1)               # [N, 8]
    w = pf_topk / np.maximum(pf_topk.sum(axis=1, keepdims=True), EPS)
    best_k = np.argmax(pf_topk, axis=1)
    cs = np.asarray(centers, dtype=np.float64)
    center_topk = cs[topk]
    pstd = np.maximum(rec["p_std0"], EPS)
    rec["poe_pick_center"] = center_topk[rows, best_k] * pstd + rec["p_mean0"]
    rec["poe_weighted_e"] = np.sum(w * center_topk, axis=1) * pstd + rec["p_mean0"]

    fields = {
        f"poe_pick_center_l{lam:.1f}": "poe_pick_center",
        f"poe_weighted_e_l{lam:.1f}": "poe_weighted_e",
        "J3_median": "post_median", "P6": "p6_score",
    }
    dense = int(cand["dense_threshold"][0])
    res = summarize(rec, fields, dense,
                    refs={"vs_J3": (rec, "post_median"),
                          "vs_P6": (rec, "p6_score")},
                    label=f"R3_poe_lam{lam:.1f}")
    res["lambda"] = lam
    if out_json:
        write_json_ledger(out_json, res, "r3_poe", lam=lam)
    return res, rec


# ============================================================================
# R-series battery runner (from r_series.py)
# ============================================================================

def _run_guarded(name, fn):
    """Run one R arm; log and continue on failure so the battery survives."""
    try:
        result = fn()
        print(f"[battery] {name} OK", flush=True)
        return result
    except Exception as e:  # noqa: BLE001
        print(f"[battery] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        append_trial({"event": f"r_series_{name}", "status": "failed",
                      "error": str(e)[:500]})
        return None


# ============================================================================
# F-INT (B2): zero-training interpolation (from fuse_scores.py)
# ============================================================================

def fint_scores(cand, sc, lam):
    """Per-row F-INT score: max_k [(1-lam) logq + lam logp_bert] over candidates."""
    logq = cand["topk_logq"]
    lp = sc["logp_bert_topk"]
    if lp.shape[1] < logq.shape[1]:
        logq = logq[:, :lp.shape[1]]
    s = (1.0 - lam) * logq + lam * lp
    return np.nanmax(s, axis=1)


def select_lambda_calibration(calib_cand, calib_sc, dense_threshold=None):
    """Choose lambda on the CALIBRATION slice (audit_uids x [2023-02, 2024-02)).

    Plan 9.1/9.2: every fusion parameter is fit ONLY on the calibration slice;
    0..399 is pure inference and must never be used to fit lambda.  The calib
    slice has ~565 stocks/date (NOT the eval 3634), so its own dense threshold
    is derived from the calib cross-sections.
    """
    if dense_threshold is None:
        by_date = {}
        for d in calib_cand["date_key"]:
            by_date[str(d)] = by_date.get(str(d), 0) + 1
        max_cs = max(by_date.values())
        dense_threshold = max(5, int(np.ceil(0.8 * max_cs)))
    rec = {"date_key": calib_cand["date_key"],
           "true_logret": calib_cand["true_logret"].astype(np.float64),
           "quality": calib_cand["quality"].astype(bool)}
    best, best_ic = None, -1e9
    scores = {}
    for lam in LAMBDA_GRID:
        rec["fint"] = fint_scores(calib_cand, calib_sc, lam)
        m = arm_metrics(rec, "fint", dense_threshold)
        ic = m["avg_daily_rank_ic"]
        scores[str(lam)] = {"lambda": lam, "calib_avg_daily_rank_ic": ic,
                            "n_dense_dates": m["n_dense_dates"]}
        if ic is not None and ic > best_ic:
            best_ic, best = ic, lam
    return best, best_ic, scores


def run_fint(*, candidates, scores_eval, out_json=None,
             out_scores_npz=None, calib_candidates=None, calib_scores=None):
    cand = np.load(candidates, allow_pickle=True)
    sc = np.load(scores_eval, allow_pickle=True)
    if len(sc["stock_uid"]) != len(cand["stock_uid"]):
        raise RuntimeError("F-INT: scores_eval misaligned with candidates")
    dense_threshold = int(cand["dense_threshold"][0])

    if calib_candidates is not None and calib_scores is not None:
        # protocol-correct lambda selection on the calibration slice
        cc = np.load(calib_candidates, allow_pickle=True)
        cs = np.load(calib_scores, allow_pickle=True)
        if len(cs["stock_uid"]) != len(cc["stock_uid"]):
            raise RuntimeError("F-INT: calib BERT scores misaligned with calib candidates")
        # calib derives its own dense threshold (eval's 3634 would exclude all calib dates)
        best, best_ic, lam_scores = select_lambda_calibration(cc, cs, dense_threshold=None)
        fitted_on = "calibration_slice_audit_2023_02_2024_02"
    else:
        raise RuntimeError(
            "F-INT requires calibration-slice lambda selection (plan 9.1 forbids "
            "fitting lambda on 0..399).  Build calib candidates + BERT scores first: "
            "build_gpt_candidates --region calib && score_bert --region calib")

    rec = {"date_key": cand["date_key"], "stock_uid": cand["stock_uid"],
           "true_logret": cand["true_logret"].astype(np.float64),
           "quality": cand["quality"].astype(bool),
           "offset": cand["offset"].astype(np.int64)}
    rec["fint"] = fint_scores(cand, sc, best)
    full = arm_metrics(rec, "fint", dense_threshold)
    result = {
        "schema": "fint-v1",
        "chosen_lambda": best,
        "chosen_lambda_calib_rank_ic": best_ic,
        "lambda_scores": lam_scores,
        "chosen_lambda_full_400": {
            "avg_daily_rank_ic": full["avg_daily_rank_ic"],
            "avg_da_per_date": full["avg_da_per_date"],
            "n_dense_dates": full["n_dense_dates"],
        },
        "n_rows": len(cand["stock_uid"]),
        "fitted_on": fitted_on,
    }
    if out_json:
        write_json(out_json, result)
    if out_scores_npz is not None:
        out_scores_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_scores_npz,
                 stock_uid=cand["stock_uid"], date_key=cand["date_key"],
                 fint_score=rec["fint"],
                 chosen_lambda=np.array([best]))
    return result, rec, best


# ============================================================================
# F-STK (B3) / F-SEL (B4): stacking combiner (from fuse_scores.py)
# ============================================================================

FEATURE_NAMES = ["f1_post_median", "f2_p_up", "f3_post_std", "f4_p6",
                 "f5a_logp_bert_top1", "f5b_bert_gpt_rank", "f6_intersection",
                 "f7_electra", "f8_hist_cons"]


def stack_features(cand, sc, p6):
    """Build the stacking feature matrix for one region (calib or eval).

    Columns (FEATURE_NAMES):
      f1 post_median, f2 p_up, f3 post_std,
      f4 P6 rank score,
      f5a log p_BERT of BERT's best candidate,
      f5b the GPT-rank of BERT's chosen candidate (BERT-GPT consistency),
      f6 intersection size of GPT top-8 and BERT top-8 candidate sets,
      f7/f8 ELECTRA + hist-consistency (zeros until those stages exist).
    """
    n = len(cand["stock_uid"])
    lp = sc["logp_bert_topk"] if sc is not None else None
    lp_full = sc["logp_bert_full"] if sc is not None else None
    best_k = np.nanargmax(lp, axis=1) if lp is not None else np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    f5a = lp[rows, best_k] if lp is not None else np.zeros(n)
    f5b = np.zeros(n)
    f6 = np.zeros(n)
    if lp is not None:
        # GPT-rank of BERT's best candidate: argsort of GPT logq ascending
        gpt_order = np.argsort(np.argsort(-cand["topk_logq"], axis=1), axis=1)
        f5b = gpt_order[rows, best_k].astype(np.float64) / max(lp.shape[1], 1)
        # f6 = overlap of GPT top-8 (over GPT's candidate set) with BERT top-8
        # OVER THE FULL 128 vocab (logp_bert_full).  Per-row set intersection.
        gk = np.argsort(-cand["topk_logq"], axis=1, kind="stable")[:, :8]
        if lp_full is not None and lp_full.shape[1] == VOCAB_BASE:
            bk_full = np.argsort(-lp_full, axis=1, kind="stable")[:, :8]
            inter = np.zeros(n, dtype=np.int64)
            for kk in range(8):
                inter += (bk_full[:, kk, None] == gk).any(axis=1)
            f6 = inter.astype(np.float64)
    X = np.stack([cand["post_median"], cand["p_up"], cand["post_std"],
                  p6, f5a, f5b, f6,
                  np.zeros(n), np.zeros(n)], axis=-1).astype(np.float32)
    return X


def run_stack(*, candidates_calib, scores_calib, calib_p6, candidates_eval,
              scores_eval, eval_p6, out_scores_npz=None, out_json=None,
              hidden=64, epochs=20, seed=42):
    """Train the stack combiner on the calibration slice, apply to eval (B3)."""
    from common import MlpRankHead, soft_spearman_loss  # noqa: E402
    cc = np.load(candidates_calib, allow_pickle=True)
    cs = np.load(scores_calib, allow_pickle=True)
    if len(cs["stock_uid"]) != len(cc["stock_uid"]):
        raise RuntimeError("F-STK: calib BERT scores misaligned with calib candidates")
    calib_X = stack_features(cc, cs, calib_p6)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = MlpRankHead(dim=calib_X.shape[1], hidden=hidden, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    X = torch.from_numpy(calib_X)
    y = torch.from_numpy(cc["true_logret"]).float()
    dates = np.asarray(cc["date_key"])
    uniq = np.unique(dates)
    for ep in range(epochs):
        rng.shuffle(uniq)
        losses = []
        for d in uniq:
            m = dates == d
            if m.sum() < 30:
                continue
            xb, yb = X[m], y[m]
            ranks = ((yb.argsort().argsort() + 1) / yb.shape[0]).float()
            s = model(xb)
            loss = soft_spearman_loss(s, ranks, tau=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 5 == 0:
            print(f"[stk] epoch {ep} loss={np.mean(losses):.4f}")

    # apply to eval
    ce = np.load(candidates_eval, allow_pickle=True)
    se = np.load(scores_eval, allow_pickle=True)
    eval_X = stack_features(ce, se, eval_p6)
    with torch.no_grad():
        stack_score = model(torch.from_numpy(eval_X)).numpy()
    result = {"schema": "fstk-v1", "fitted_on": "calibration_slice",
              "n_epochs": epochs, "feature_names": FEATURE_NAMES,
              "n_calib_rows": len(cc["stock_uid"]), "n_eval_rows": len(eval_X)}
    if out_json:
        write_json(out_json, result)
    if out_scores_npz is not None:
        out_scores_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_scores_npz,
                 stock_uid=ce["stock_uid"], date_key=ce["date_key"],
                 stack_score=stack_score)
    return model, stack_score, result


# ============================================================================
# P6 scoring helper (inlined from eval_critic.py)
# ============================================================================

def load_p6_scores(head_path, hidden):
    """Evaluate the 06 P6 MLP rank head on hidden [N, dim] -> [N] scores."""
    import torch as _torch
    ck = _torch.load(str(head_path), map_location="cpu", weights_only=False)
    sd = ck["head_state"]
    # MlpRankHead(dim=256, hidden=64, dropout=0.0) -> Linear(0)/SiLU(1)/Identity(2)/Linear(3)
    net = _torch.nn.Sequential(
        _torch.nn.Linear(256, 64), _torch.nn.SiLU(), _torch.nn.Identity(),
        _torch.nn.Linear(64, 1))
    net.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
    net.eval()
    with _torch.no_grad():
        s = net(_torch.from_numpy(hidden).float()).squeeze(-1).numpy()
    return s


# ============================================================================
# Stage functions
# ============================================================================

def _stage_c1(model_name=None):
    """Run the R-series CPU arms."""
    print("=== R-series (all CPU arms) ===")
    centers = load_centers()
    rroot = stage_results("C")
    cand, sc, pt01 = _load_eval()
    require_full_validation_coverage(
        cand["offset"], cand["date_key"], label=f"{_current_model_name(model_name)} C"
    )
    p6 = _p6_eval()
    model_label = _current_model_name(model_name)
    r1_result = _run_guarded(
        "r1", lambda: r1(cand, sc, pt01, p6, centers, calib=True,
                          out_json=_model_result_path("r-series-r1", model_label)))
    _run_guarded("r3_bert", lambda: r3_bert(
        cand, sc, pt01, p6, centers,
        out_json=_model_result_path("r-series-r3-bert", model_label)))
    _run_guarded("r4", lambda: r4(cand, sc, pt01, p6, centers,
                                  out_json=_model_result_path("r-series-r4", model_label),
                                  gpt_q_eval=None))
    _run_guarded("r5", lambda: r5(cand, sc, pt01, p6, centers, calib=True,
                                  out_json=_model_result_path("r-series-r5", model_label)))
    append_trial({"event": "r_series_all_cpu", "status": "ok"})
    print("R-series complete -- JSON under", rroot)
    return r1_result


def _stage_c2():
    """Run full-vocabulary PoE fusion (needs the full GPT q cache)."""
    print("=== Full-vocabulary PoE fusion ===")
    centers = load_centers()
    rroot = stage_results("C")
    cand, sc, pt01 = _load_eval()
    p6 = _p6_eval()
    res, _ = r2(cand, sc, pt01, p6, centers,
                out_json=rroot / "r-series-r2.json")
    print(f"calib chosen lambda={res['calib_chosen_lambda']} interior="
          f"{res['lambda_interior']}")
    for k, v in res["calib_lambda_scores"].items():
        print(f"  lam={v['lambda']:.1f} calib_ic={v['calib_rank_ic']}")
    print("full:", json.dumps(res["full"], indent=1, default=str))


def _stage_c3():
    """Run BERT selection over the full-vocabulary PoE scores."""
    print("=== BERT PoE selection ===")
    centers = load_centers()
    rroot = stage_results("C")
    cand, sc, pt01 = _load_eval()
    p6 = _p6_eval()
    res, _ = r3_poe(cand, sc, pt01, p6, centers,
                    out_json=rroot / "r-series-r3-poe.json", lam=0.5)
    print(json.dumps(res["full"], indent=1, default=str))


def _stage_c4(model_name=None):
    """Run score-interpolation fusion with calibration-selected weight."""
    print("=== Score-interpolation fusion ===")
    roots_obj = resolve_roots(seed=42)
    candidates = weights_artifact("candidates-eval")
    model_label = _current_model_name(model_name)
    scores_eval = scores_path("eval", suffix=SCORES_SUFFIX or model_label)
    calib_cand = weights_artifact("candidates-calib")
    calib_sc = scores_path("calib", suffix=SCORES_SUFFIX or model_label)
    if not scores_eval.exists():
        raise RuntimeError(f"eval BERT scores missing: {scores_eval}")
    result, rec, best = run_fint(
        candidates=candidates, scores_eval=scores_eval,
        out_json=_model_result_path("fint", model_label),
        out_scores_npz=weights_artifact("fint-scores", model=model_label),
        calib_candidates=calib_cand if calib_cand.exists() else None,
        calib_scores=calib_sc if calib_sc.exists() else None)
    rec["fint_score"] = rec["fint"]
    result["model"] = model_label
    append_trial({"event": "fint", "chosen_lambda": best, "status": "ok"})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result, rec, best


def _stage_c5():
    """Run the calibration-trained stacking combiner."""
    print("=== Stacking fusion ===")
    roots_obj = resolve_roots(seed=42)
    candidates = weights_artifact("candidates-eval")
    scores_eval = weights_artifact("scores-eval")
    calib_cand = weights_artifact("candidates-calib")
    calib_sc = weights_artifact("scores-calib")
    if not (calib_cand.exists() and calib_sc.exists()):
        raise RuntimeError("F-STK needs calib candidates + BERT scores "
                           "(build_gpt_candidates --region calib && score_bert --region calib)")
    calib_p6 = load_p6_scores(
        posttrain_artifacts()["head_rank_mlp_spearman"],
        np.load(posttrain_artifacts()["calibration"],
                allow_pickle=True)["hidden"])
    eval_p6 = np.load(weights_artifact("p6-scores"))
    model, stack_score, result = run_stack(
        candidates_calib=calib_cand, scores_calib=calib_sc, calib_p6=calib_p6,
        candidates_eval=candidates, scores_eval=scores_eval, eval_p6=eval_p6,
        out_scores_npz=weights_artifact("stack-scores"),
        out_json=stage_results("C") / "stack.json")
    append_trial({"event": "fstk", "status": "ok",
                  "n_calib": result["n_calib_rows"]})
    print(json.dumps(result, indent=2, ensure_ascii=False))


FULL_PREDICTION_FIELDS = (
    "offset", "date_key", "stock_uid", "true_logret", "quality",
    "true_coarse_id", "gpt_top1_id", "bert_top1_id",
    "post_median", "p_up", "post_std", "p6_score", "bert_score",
    "bert_margin", "bert_decode_mean", "bert_decode_median", "bert_p_up",
    "rank_head_score", "fint_score", "stack_score",
)


def _combined_model_records(model_name, r1_records, fint_records, scores, candidates):
    """Join the per-model BERT decode, rank-head and fusion predictions."""
    n = len(r1_records["stock_uid"])
    if (len(scores["stock_uid"]) != n or len(candidates["stock_uid"]) != n
            or not np.array_equal(scores["stock_uid"], candidates["stock_uid"])
            or not np.array_equal(scores["date_key"], candidates["date_key"])):
        raise RuntimeError(f"{model_name}: score/candidate rows do not match R1 records")
    records = dict(r1_records)
    records["bert_score"] = np.nanmax(scores["logp_bert_topk"], axis=1).astype(np.float64)
    records["bert_margin"] = scores["bert_margin"].astype(np.float64)
    records["true_coarse_id"] = candidates["true_coarse_id"].astype(np.int16)
    records["gpt_top1_id"] = candidates["topk_ids"][:, 0].astype(np.int16)
    records["bert_top1_id"] = scores["bert_top1_id"].astype(np.int16)
    records["bert_decode_mean"] = records["ebert_mean"]
    records["bert_decode_median"] = records["ebert_median"]
    records["bert_p_up"] = records["ebert_pup"]
    records["fint_score"] = fint_records["fint"].astype(np.float64)

    rank_path = weights_artifact("rank-head-scores", model=model_name)
    if rank_path.exists():
        rank = np.load(rank_path, allow_pickle=True)
        if (not np.array_equal(rank["stock_uid"], candidates["stock_uid"])
                or not np.array_equal(rank["date_key"], candidates["date_key"])):
            raise RuntimeError(f"{model_name}: rank-head score rows are misaligned")
        records["rank_head_score"] = rank["rank_head_score"].astype(np.float64)

    require_full_validation_coverage(
        records["offset"], records["date_key"], label=f"{model_name} prediction table"
    )
    return records


def _stage_c_all():
    """Run R-series and score interpolation for every formal BERT variant."""
    prediction_path = results_root(seed=42) / "bert_critic_predictions.parquet"
    summary = {
        "schema": "bert-critic-model-evaluation-v2",
        "models": [],
        "prediction_records": str(prediction_path),
        "prediction_fields": ["model", *FULL_PREDICTION_FIELDS],
        "holdout_used": False,
    }
    global SCORES_SUFFIX
    with PredictionParquetWriter(
        prediction_path, FULL_PREDICTION_FIELDS, include_model=True
    ) as writer:
        for variant in MODEL_VARIANTS:
            model_name = variant["name"]
            SCORES_SUFFIX = variant["suffix"]
            print("=" * 70)
            print(f"[C] {model_name} — R-series and score interpolation")
            print("=" * 70)
            r1_payload = _stage_c1(model_name)
            fint_payload = _stage_c4(model_name)
            if r1_payload is None or fint_payload is None:
                raise RuntimeError(f"{model_name}: C analysis did not produce records")
            _, r1_records = r1_payload
            _, fint_records, chosen_lambda = fint_payload
            cand, scores, _ = _load_eval()
            records = _combined_model_records(
                model_name, r1_records, fint_records, scores, cand
            )
            writer.write(records, model=model_name)
            coverage = require_full_validation_coverage(
                records["offset"], records["date_key"], label=f"{model_name} C"
            )
            summary["models"].append({
                "name": model_name,
                "prediction_rows": int(len(records["stock_uid"])),
                "chosen_lambda": chosen_lambda,
                "validation_coverage": coverage,
            })
            del records, r1_records, fint_records
    summary["prediction_rows"] = int(writer.rows)
    write_json(results_root(seed=42) / "model-evaluation.json", summary)
    assert_results_boundary(resolve_roots(seed=42))
    print(f"[C] Combined prediction table: {prediction_path}")
    print(f"[C] Combined prediction rows: {writer.rows}")


def _stage_c_summary():
    """C-summary: Print summary of all R-series + F-series results."""
    rroot = stage_results("C")
    out = {}
    res_paths = {
        "r1": rroot / "r-series-r1.json",
        "r3_bert": rroot / "r-series-r3-bert.json",
        "r4": rroot / "r-series-r4.json",
        "r5": rroot / "r-series-r5.json",
        "r2": rroot / "r-series-r2.json",
        "r3_poe": rroot / "r-series-r3-poe.json",
        "fint": rroot / "fint.json",
        "fstk": rroot / "stack.json",
    }
    for name, path in res_paths.items():
        if Path(path).exists():
            d = json.load(open(path, encoding="utf-8"))
            out[name] = {"full": d.get("full"),
                         "dev": d.get("dev_0_299"),
                         "confirm": d.get("confirm_300_399"),
                         "bootstrap_vs": d.get("bootstrap_vs"),
                         "calib_chosen_lambda": d.get("calib_chosen_lambda"),
                         "calib_fitted_w": d.get("calib_fitted_w"),
                         "auc_rank_ic": d.get("auc_rank_ic"),
                         "chosen_lambda": d.get("chosen_lambda")}
    print(json.dumps(out, indent=1, default=str))


# ============================================================================
# CLI
# ============================================================================

DEBUG_STAGES = {
    "r-series": ("C: R-series analysis", _stage_c1),
    "poe-full": ("C: full-vocabulary PoE fusion", _stage_c2),
    "bert-select": ("C: BERT candidate selection", _stage_c3),
    "score-interpolation": ("C: score-interpolation fusion", _stage_c4),
    "stacking": ("C: stacking fusion", _stage_c5),
    "full": ("C: default analysis", _stage_c_all),
    "summary": ("C: summary", _stage_c_summary),
}


def main():
    if len(sys.argv) == 1:
        requested = "full"
        stage_args = []
    elif sys.argv[1] in {"-h", "--help"}:
        print("C_run.py 无参数时会完成默认的 R-series 与 score-interpolation 分析。")
        print("仅调试单个环节时可传入语义名称；完整流程名称为 full。")
        print("如需指定模型缓存，可附加 --scores-suffix <模型名称>。")
        return
    elif sys.argv[1] in {"--stage", "-s"}:
        if len(sys.argv) < 3:
            print("缺少调试阶段名称；C_run.py 默认运行完整 C 流程。")
            sys.exit(2)
        requested = sys.argv[2]
        stage_args = sys.argv[3:]
    elif sys.argv[1] == "--scores-suffix":
        requested = "full"
        stage_args = sys.argv[1:]
    else:
        requested = sys.argv[1]
        stage_args = sys.argv[2:]

    suffix_parser = argparse.ArgumentParser(add_help=False)
    suffix_parser.add_argument("--scores-suffix", type=str, default="")
    suffix_args, stage_args = suffix_parser.parse_known_args(stage_args)
    global SCORES_SUFFIX
    SCORES_SUFFIX = suffix_args.scores_suffix

    if requested in {"all", "C-all"}:
        requested = "full"
    if requested not in DEBUG_STAGES:
        print("C_run.py 默认运行完整 C 流程；未知的调试阶段：", requested)
        print("可用名称：r-series、poe-full、bert-select、score-interpolation、")
        print("stacking、full、summary")
        sys.exit(2)

    label, fn = DEBUG_STAGES[requested]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    sys.argv = [sys.argv[0], *stage_args]
    fn()


if __name__ == "__main__":
    main()
