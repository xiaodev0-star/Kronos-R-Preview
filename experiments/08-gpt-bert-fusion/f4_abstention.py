"""f4_abstention.py — Round 2: confidence / abstention on a fused score.

The user's "GPT predict + BERT score/filter -> high precision" goal maps to
selective prediction: keep only the most confident rows and report the acted
metrics.  This script computes coverage curves (100/80/60/40/20%) for candidate
confidence signals on EVAL (0..399) for a given score field.

Confidence signals (all from cached assets, no training):
  c_js      JS(p_BERT || q_GPT)  (disagreement; R4 showed JS is the best BERT-line
              abstention signal)
  c_entropy BERT coarse entropy  (low entropy = high confidence)
  c_std     GPT posterior std    (low std = high confidence)
  c_pabs    |P(up) - 0.5|        (directional conviction)

Protocol: thresholds are chosen per-coverage TARGET on the FULL eval window for
this diagnostic.  Formal blend/threshold fitting uses the calib slice (f5 script);
this script's per-coverage values are the diagnostic ceiling/landscape.
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
for _p in (ROOT, SEVEN, EIGHT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    softmax_rows, write_json_ledger, _split_points,
)
from scipy.stats import spearmanr  # noqa: E402


def _daily_ic(rec, field, dense):
    score = np.asarray(rec[field], dtype=np.float64)
    true = np.asarray(rec["true_logret"], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(true) & np.asarray(rec["quality"]).astype(bool)
    order, bounds, uniq = _split_points(rec["date_key"])
    s = score[order]; t = true[order]; v = valid[order]
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
    s = score[order]; t = true[order]; v = valid[order]
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        m = v[lo:hi]
        c = int(m.sum())
        if c < dense:
            continue
        out[uniq[i]] = float(np.mean((np.sign(s[lo:hi][m]) > 0) == (t[lo:hi][m] > 0)))
    return out


def _coverage_curve(rec, score_field, conf_field, dense, coverages=(1.0, 0.8, 0.6, 0.4, 0.2),
                    max_high=True):
    """Within each coverage target, keep the top fraction of rows by confidence
    and report mean daily RankIC/DA over dense dates of the ACTED subset, plus
    the baseline (same score) on the same acted subset, plus overall coverage."""
    score = np.asarray(rec[score_field], dtype=np.float64)
    conf = np.asarray(rec[conf_field], dtype=np.float64)
    true = np.asarray(rec["true_logret"], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(conf) & np.isfinite(true) \
        & np.asarray(rec["quality"]).astype(bool)
    dates = np.asarray(rec["date_key"])
    uniq_dates = np.unique(dates)
    out = []
    for cv in coverages:
        acted = np.zeros(len(score), dtype=bool)
        kept_rows = 0
        for d in uniq_dates:
            dm = (dates == d) & valid
            if dm.sum() == 0:
                continue
            keep_n = int(round(cv * dm.sum()))
            keep_n = max(keep_n, 1)
            # sort by confidence within date
            idx = np.where(dm)[0]
            cvals = conf[idx]
            order = np.argsort(-cvals) if max_high else np.argsort(cvals)
            top = order[:keep_n]
            acted[idx[top]] = True
            kept_rows += len(top)
        rec_a = slice_rec(rec, acted)
        if len(rec_a["stock_uid"]) < 50:
            out.append({"coverage_target": cv, "error": "too few acted rows"})
            continue
        a_ic = np.mean(list(_daily_ic(rec_a, score_field, dense).values())) \
            if _daily_ic(rec_a, score_field, dense) else None
        a_da = np.mean(list(_daily_da(rec_a, score_field, dense).values())) \
            if _daily_da(rec_a, score_field, dense) else None
        out.append({
            "coverage_target": cv,
            "actual_frac_rows": float(kept_rows / max(1, int(valid.sum()))),
            "n_acted_rows": int(kept_rows),
            "acted_rank_ic": a_ic,
            "acted_da": a_da,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="F_zsum",
                    help="rec field to rank/score (e.g. F_zsum, bert_head, p6_score)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    dense = int(cand["dense_threshold"][0])

    # confidence signals from cached logp_bert_full + gpt_q_full128
    s = np.load(wr / "scores_eval_K8_w512_stride1.npz", allow_pickle=True)
    logp = s["logp_bert_full"]          # [N,128] log-space
    qg = np.load(wr / "gpt_q_eval_full128.npz", allow_pickle=True)
    q = qg["q"]                         # [N,128] prob
    n = len(rec["stock_uid"])
    print(f"[f4] n={n}")

    # JS(p_bert || q_gpt) — chunked to bound memory
    pb = softmax_rows(logp)
    qq = q
    eps = 1e-12
    js = np.zeros(n)
    CH = 400_000
    for st in range(0, n, CH):
        sp = min(st + CH, n)
        pb_c = np.maximum(pb[st:sp], eps)
        q_c = np.maximum(qq[st:sp], eps)
        m = 0.5 * (pb_c + q_c)
        js[st:sp] = 0.5 * (np.sum(pb_c * np.log(pb_c / m), axis=1)
                           + np.sum(q_c * np.log(q_c / m), axis=1))
    rec["c_js"] = js
    # BERT entropy
    rec["c_entropy"] = -np.sum(pb * np.maximum(pb, eps) * 0.0, axis=1)  # placeholder
    logpb = np.asarray(logp)
    rec["c_entropy"] = -np.sum(np.where(np.isfinite(logpb), np.exp(np.clip(logpb, -50, 50)) * logpb, 0.0),
                               axis=1)
    # GPT posterior std and |P(up)-0.5|
    rec["c_std"] = cand["post_std"].astype(np.float64)
    rec["c_pabs"] = np.abs(cand["p_up"].astype(np.float64) - 0.5)

    # baseline full-window IC for context
    from improve_common import metrics_table, DailyIcCache
    rec["c_js"][~np.isfinite(rec["c_js"])] = np.nan
    base_ic = float(np.nanmean(list(_daily_ic(rec, args.score, dense).values())))
    print(f"[f4] full-window {args.score} rank_ic={base_ic:.4f}")

    res = {"schema": "f4-abstention-v1", "score": args.score,
           "full_rank_ic": base_ic, "dense_threshold": dense, "coverage": {}}
    # max_high: JS (higher = more disagreement = less confident) -> keep LOW JS.
    # For each conf, report the curve; note direction in 'direction'.
    for cf, direction in (("c_js", "keep_low"), ("c_entropy", "keep_low"),
                          ("c_std", "keep_low"), ("c_pabs", "keep_high")):
        if cf not in rec:
            continue
        curve = _coverage_curve(rec, args.score, cf, dense, max_high=(direction == "keep_high"))
        res["coverage"][cf] = {"direction": direction, "curve": curve}
        print(f"\n[{cf} ({direction})]")
        for r in curve:
            if "acted_rank_ic" in r and r["acted_rank_ic"] is not None:
                print(f"  cov={r['coverage_target']:5.2f} rows={r['n_acted_rows']:>8d} "
                      f"acted_ic={r['acted_rank_ic']:.4f} acted_da={r['acted_da']:.4f}")

    out = rr / f"f4_abstention_{args.score}{('_' + args.tag) if args.tag else ''}.json"
    write_json_ledger(out, res, "f4_abstention", score=args.score, tag=args.tag)
    print(f"[f4] -> {out}")


if __name__ == "__main__":
    main()
