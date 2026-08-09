"""fuse_scores.py — plan §9: F-INT / F-STK / F-SEL fusion.

Principles (§9.1): BERT/ELECTRA scores are NEVER used alone or as a GPT/P6
replacement — shared training errors mean fusion is where the value is.  Every
fusion parameter (lambda / stacking weights / acceptance threshold) is fit ONLY
on the calibration slice (audit_uids x [2023-02-01, 2024-02-01)); 0..399 is
never touched for fitting.

Arms:
    F-INT (B2)  score = max_k [(1-lambda) log q_GPT(c_k) + lambda log p_BERT(c_k)]
                lambda grid, chosen on dev 0..299 daily RankIC; magnitude stays
                J3 median.
    F-STK (B3)  stacking combiner over f1..f8 (see build_features), trained on
                the calibration slice, soft-Spearman or pairwise loss.
    F-SEL (B4)  F-STK + acceptance threshold tau: only rewrite the rank when the
                stacking top1 margin vs the reference top exceeds tau.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bert_data import VOCAB_BASE  # noqa: E402
from critic_common import resolve_roots, write_json, append_trial  # noqa: E402
from evaluate_posttrain import arm_metrics  # noqa: E402


# ============================================================================
# F-INT (B2): zero-training interpolation
# ============================================================================

LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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

    Plan §9.1/§9.2: every fusion parameter is fit ONLY on the calibration slice;
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
            "F-INT requires calibration-slice lambda selection (plan §9.1 forbids "
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
# F-STK (B3) / F-SEL (B4): stacking combiner over calibration-slice features
# ============================================================================
#
# Feature table (per stock-day row) is built by ``build_features`` from the
# calibration slice; the combiner is a small MLP trained with soft-Spearman on
# daily cross-sections (reusing posttrain_heads losses), then applied to the
# eval region.  f4 (P6) uses only out-of-fold predictions (P6 trained on fit_uids
# => its audit_uids predictions are naturally out-of-sample).  Fits ONLY on the
# calibration slice.

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
    from posttrain_heads import MlpRankHead, soft_spearman_loss
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT critic fusion (F-INT first)")
    ap.add_argument("--arm", choices=["fint", "stack", "all"], default="fint")
    args = ap.parse_args()
    roots = resolve_roots(seed=42)
    candidates = roots.weights_root / "candidates_eval_K8.npz"
    scores_eval = roots.weights_root / "scores_eval_K8_w512_stride1.npz"
    calib_cand = roots.weights_root / "candidates_calib_K8.npz"
    calib_sc = roots.weights_root / "scores_calib_K8_w512_stride1.npz"
    if args.arm in ("fint", "all"):
        if not scores_eval.exists():
            raise RuntimeError(f"eval BERT scores missing: {scores_eval}")
        result, rec, best = run_fint(
            candidates=candidates, scores_eval=scores_eval,
            out_json=roots.results_root / "fint_result.json",
            out_scores_npz=roots.weights_root / "fint_scores_eval.npz",
            calib_candidates=calib_cand if calib_cand.exists() else None,
            calib_scores=calib_sc if calib_sc.exists() else None)
        append_trial({"event": "fint", "chosen_lambda": best, "status": "ok"})
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.arm in ("stack", "all"):
        if not (calib_cand.exists() and calib_sc.exists()):
            raise RuntimeError("F-STK needs calib candidates + BERT scores "
                               "(build_gpt_candidates --region calib && score_bert --region calib)")
        from eval_critic import load_p6_scores
        calib_p6 = load_p6_scores(
            ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "head_P6_mlp_rank_spearman.pt",
            np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "calibration_cache.npz",
                    allow_pickle=True)["hidden"])
        eval_p6 = np.load(roots.weights_root / "p6_eval_scores.npy")
        model, stack_score, result = run_stack(
            candidates_calib=calib_cand, scores_calib=calib_sc, calib_p6=calib_p6,
            candidates_eval=candidates, scores_eval=scores_eval, eval_p6=eval_p6,
            out_scores_npz=roots.weights_root / "stack_scores_eval.npz",
            out_json=roots.results_root / "fstk_result.json")
        append_trial({"event": "fstk", "status": "ok",
                      "n_calib": result["n_calib_rows"]})
        print(json.dumps(result, indent=2, ensure_ascii=False))
