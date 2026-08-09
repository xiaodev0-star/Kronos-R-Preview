"""r_series.py — plan §3: R-series correction experiments (zero-training).

The previous round ranked with token LIKELIHOODS (E-1).  These arms rank with
DECODED RETURNS from the full-128 coarse posterior, raw-space restored.

  R1  BERT posterior decode trio: E_BERT[r] / median_BERT / P_BERT(up)
      (= J2/J3/J4's BERT version).  Pure CPU, existing caches.
  R2  Full-128 PoE fusion: p_fused ∝ q_GPT^(1-lam) p_BERT^lam, decoded.  Needs
      gpt_q_eval_full128.npz (build_gpt_full_q.py).
  R3  critic picks a candidate among GPT's top-K, row score = that candidate's
      decoded return (R3a top-1 center, R3b top-K weighted E[r]).
  R4  likelihood/uncertainty features as ABSTENTION signals (never rank
      scores), vs the 06 q1 baseline.
  R5  P(up) direction fusion: w*P_GPT(up) + (1-w)*P_BERT(up), w fit on calib.

Statistics inherit 06: daily RankIC/DA/MAE, paired circular moving-block
bootstrap (L=5/10/20), dev 0..299 / confirm 300..399.  0..399 pure inference.

Usage:
    python r_series.py --mode all_cpu     # R1 + R3bert + R4 + R5 (one cache load)
    python r_series.py --mode r2          # PoE (after build_gpt_full_q)
    python r_series.py --mode r3poe       # PoE-select (after build_gpt_full_q)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    EPS, CENTERS_PATH, PT01_REC, P6_HEAD,
    cand_path, scores_path, gpt_q_path, weights_root, results_root,
    load_centers, softmax_rows, decode_coarse, poe_fused,
    build_rec, eval_dev_confirm, slice_rec,
    metrics_table, summarize, bootstrap_vs, calib_dense_threshold,
    daily_rank_ic_series, write_json_ledger,
)
from critic_common import append_trial  # noqa: E402


# ============================================================================
# Shared loads
# ============================================================================

SCORES_SUFFIX = ""   # set by main --scores-suffix (e.g. t1_w512 for fine-tuned eval)


def _load_eval():
    cand = np.load(cand_path("eval"), allow_pickle=True)
    sc = np.load(scores_path("eval", suffix=SCORES_SUFFIX), allow_pickle=True)
    pt01 = np.load(PT01_REC, allow_pickle=True)
    return cand, sc, pt01


def _load_calib():
    cc = np.load(cand_path("calib"), allow_pickle=True)
    cs = np.load(scores_path("calib", suffix=SCORES_SUFFIX), allow_pickle=True)
    return cc, cs


def _p6_eval():
    return np.load(weights_root() / "p6_eval_scores.npy")


def _attach_p6(rec, p6):
    if len(p6) == len(rec["stock_uid"]):
        rec["p6_score"] = p6.astype(np.float64)
    else:
        raise RuntimeError(f"p6 len {len(p6)} != rec {len(rec['stock_uid'])}")


# ============================================================================
# R1 — BERT posterior decode trio
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
# R3 (pure-BERT select arms) — CPU
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
# R5 — P(up) direction fusion (w fit on calib)
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
# R4 — abstention curves (likelihood features as "when to trust", never rank)
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
        # JS(p_bert || q_gpt) — divergence high = disagree = low confidence
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

    from abstention_posttrain import abstain_curve
    dense = int(cand["dense_threshold"][0])
    res = {"schema": "r4-abstention-v1", "dense_min": dense,
           "score_fields": {"J3": "post_median", "P6": "p6_score"}}
    for sf_name, sf in (("J3", "post_median"), ("P6", "p6_score")):
        res[sf_name] = {}
        for cname, cval in conf.items():
            curve = abstain_curve(rec, sf, cval, dense_min=dense)
            covs = [float(curve[str(c)]["coverage"]) for c in (100, 80, 60, 40, 20)]
            ics = [curve[str(c)]["avg_daily_rank_ic"] for c in (100, 80, 60, 40, 20)]
            auc = float(np.trapz(ics, covs))  # integral over coverage
            res[sf_name][cname] = {"curve": curve, "auc_rank_ic": auc,
                                   "rankic_at_20": ics[-1]}
    if out_json:
        write_json_ledger(out_json, res, "r4")
    return res


# ============================================================================
# R2 — full-128 PoE fusion (needs gpt_q_eval_full128.npz)
# ============================================================================

LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load_gpt_q_eval():
    p = gpt_q_path("eval")
    if not p.exists():
        raise RuntimeError(
            f"{p} missing — run build_gpt_full_q.py first (R2/R3poe need GPT full q)")
    return np.load(p, allow_pickle=True)["q"]


def r2(cand, sc, pt01, p6, centers, out_json=None):
    rec = build_rec(cand, pt01)
    _attach_p6(rec, p6)
    logp_e = sc["logp_bert_full"]
    q_e = _load_gpt_q_eval()

    # ---- λ selection on the calibration slice (full-128 PoE) ----
    cc, cs = _load_calib()
    rec_c = build_rec(cc)
    q_c = cc["gpt_q"].astype(np.float64)
    logp_c = cs["logp_bert_full"]
    dense_c = calib_dense_threshold()
    lam_scores = {}
    best, best_ic = None, -1e9
    for lam in LAM_GRID:
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

    # ---- apply chosen λ (and neighbors) to eval ----
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
# R3 (PoE-select arms) — needs gpt_q
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
# main
# ============================================================================

def _run_guarded(name, fn):
    """Run one R arm; log and continue on failure so the battery survives."""
    import traceback
    try:
        fn()
        print(f"[battery] {name} OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[battery] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        append_trial({"event": f"r_series_{name}", "status": "failed",
                      "error": str(e)[:500]})


def _make_summary(res_paths):
    out = {}
    for name, path in res_paths.items():
        if Path(path).exists():
            d = json.load(open(path, encoding="utf-8"))
            out[name] = {"full": d.get("full"),
                         "dev": d.get("dev_0_299"),
                         "confirm": d.get("confirm_300_399"),
                         "bootstrap_vs": d.get("bootstrap_vs"),
                         "calib_chosen_lambda": d.get("calib_chosen_lambda"),
                         "calib_fitted_w": d.get("calib_fitted_w"),
                         "auc_rank_ic": d.get("auc_rank_ic")}
    return out


def main():
    ap = argparse.ArgumentParser(description="R-series correction experiments")
    ap.add_argument("--mode", choices=["r1", "r3bert", "r4", "r5", "r2", "r3poe",
                                       "all_cpu", "all", "summary"], default="all_cpu")
    ap.add_argument("--scores-suffix", type=str, default="",
                    help="score-cache suffix (e.g. t1_w512) to evaluate a fine-tuned "
                         "BERT against, instead of the mlm_v1 scores")
    args = ap.parse_args()
    global SCORES_SUFFIX
    SCORES_SUFFIX = args.scores_suffix

    centers = load_centers()
    rroot = results_root()

    if args.mode == "r1":
        cand, sc, pt01 = _load_eval()
        res, _ = r1(cand, sc, pt01, _p6_eval(), centers, calib=True,
                    out_json=rroot / "r_series_r1.json")
        print(json.dumps(res["full"], indent=1, default=str))
        print("bootstrap vs J3:", json.dumps(res["bootstrap_vs"]["vs_J3"],
                                             indent=1, default=str))
    elif args.mode == "r3bert":
        cand, sc, pt01 = _load_eval()
        res, _ = r3_bert(cand, sc, pt01, _p6_eval(), centers,
                         out_json=rroot / "r_series_r3_bert.json")
        print(json.dumps(res["full"], indent=1, default=str))
    elif args.mode == "r4":
        cand, sc, pt01 = _load_eval()
        q_e = _load_gpt_q_eval() if gpt_q_path("eval").exists() else None
        res = r4(cand, sc, pt01, _p6_eval(), centers, out_json=rroot / "r_series_r4.json",
                 gpt_q_eval=q_e)
        for sf in ("J3", "P6"):
            print(f"--- {sf} abstention ---")
            for cname, v in res[sf].items():
                c20 = v["curve"]["20"]
                print(f"  {cname}: AUC_ic={v['auc_rank_ic']:.4f} "
                      f"IC@100={v['curve']['100']['avg_daily_rank_ic']:.4f} "
                      f"IC@20={v['curve']['20']['avg_daily_rank_ic']:.4f} "
                      f"(20cov dense_min={c20['acted_dense_min']})")
    elif args.mode == "r5":
        cand, sc, pt01 = _load_eval()
        res, _ = r5(cand, sc, pt01, _p6_eval(), centers, calib=True,
                    out_json=rroot / "r_series_r5.json")
        print(f"calib w={res['calib_fitted_w']}")
        print(json.dumps(res["full"], indent=1, default=str))
    elif args.mode == "r2":
        cand, sc, pt01 = _load_eval()
        res, _ = r2(cand, sc, pt01, _p6_eval(), centers,
                    out_json=rroot / "r_series_r2.json")
        print(f"calib chosen lambda={res['calib_chosen_lambda']} interior="
              f"{res['lambda_interior']}")
        for k, v in res["calib_lambda_scores"].items():
            print(f"  lam={v['lambda']:.1f} calib_ic={v['calib_rank_ic']}")
        print("full:", json.dumps(res["full"], indent=1, default=str))
    elif args.mode == "r3poe":
        cand, sc, pt01 = _load_eval()
        res, _ = r3_poe(cand, sc, pt01, _p6_eval(), centers,
                        out_json=rroot / "r_series_r3_poe.json", lam=0.5)
        print(json.dumps(res["full"], indent=1, default=str))
    elif args.mode == "all_cpu":
        cand, sc, pt01 = _load_eval()
        p6 = _p6_eval()
        _run_guarded("r1", lambda: r1(cand, sc, pt01, p6, centers, calib=True,
                                      out_json=rroot / "r_series_r1.json"))
        _run_guarded("r3_bert", lambda: r3_bert(
            cand, sc, pt01, p6, centers, out_json=rroot / "r_series_r3_bert.json"))
        _run_guarded("r4", lambda: r4(cand, sc, pt01, p6, centers,
                                      out_json=rroot / "r_series_r4.json",
                                      gpt_q_eval=None))
        _run_guarded("r5", lambda: r5(cand, sc, pt01, p6, centers, calib=True,
                                      out_json=rroot / "r_series_r5.json"))
        append_trial({"event": "r_series_all_cpu", "status": "ok"})
        print("all_cpu complete — JSON under", rroot)
    elif args.mode == "all":
        # full battery in one process: R1 + R3bert + R4(with JS) + R5 + R2 + R3poe
        cand, sc, pt01 = _load_eval()
        p6 = _p6_eval()
        _run_guarded("r1", lambda: r1(cand, sc, pt01, p6, centers, calib=True,
                                      out_json=rroot / "r_series_r1.json"))
        _run_guarded("r3_bert", lambda: r3_bert(
            cand, sc, pt01, p6, centers, out_json=rroot / "r_series_r3_bert.json"))
        q_e = _load_gpt_q_eval()
        _run_guarded("r4", lambda: r4(cand, sc, pt01, p6, centers,
                                      out_json=rroot / "r_series_r4.json",
                                      gpt_q_eval=q_e))
        _run_guarded("r5", lambda: r5(cand, sc, pt01, p6, centers, calib=True,
                                      out_json=rroot / "r_series_r5.json"))
        _run_guarded("r2", lambda: r2(cand, sc, pt01, p6, centers,
                                      out_json=rroot / "r_series_r2.json"))
        _run_guarded("r3_poe", lambda: r3_poe(
            cand, sc, pt01, p6, centers, out_json=rroot / "r_series_r3_poe.json", lam=0.5))
        append_trial({"event": "r_series_all", "status": "ok"})
        print("all complete — JSON under", rroot)
    elif args.mode == "summary":
        out = _make_summary({
            "r1": rroot / "r_series_r1.json", "r3_bert": rroot / "r_series_r3_bert.json",
            "r4": rroot / "r_series_r4.json", "r5": rroot / "r_series_r5.json",
            "r2": rroot / "r_series_r2.json", "r3_poe": rroot / "r_series_r3_poe.json",
        })
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
