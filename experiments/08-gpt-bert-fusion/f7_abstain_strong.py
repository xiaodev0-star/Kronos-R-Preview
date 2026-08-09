"""f7_abstain_strong.py — abstention on the STRONG T5 ensemble (+P6 z-fusion).

Confidence signals on the fused score:
  c_pabs   |P(up)-0.5|  (GPT posterior directional conviction)     [aleatoric]
  c_js     JS(p_BERT||q_GPT) disagreement                          [two-model]
  c_dis    std across strong-head within-date rank percentiles     [epistemic]
  c_entropy BERT coarse entropy

Reports per-coverage (100/80/60/40/20%) acted RankIC/DA with baseline (same
score on the same acted subset).  Thresholds are per-date top-fraction rules
(no target leakage).  Also reports the full-window metrics for context.
"""
from __future__ import annotations

import argparse
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
    weights_root, results_root, metrics_table, build_rec, cand_path,
    slice_rec, DailyIcCache, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_field", default="F_z_STRONG")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])

    # ---- base scores ----
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["p6_score"] = p6
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    hidden = de["hidden"]
    rank_p6 = rank_pct_per_date(dates, p6)

    # load strong heads (3e-4:16): s45..s48
    from posttrain_heads import MlpRankHead  # noqa: E402
    import torch  # noqa: E402
    strong_specs = [
        ("s45", "head_BERT_mlp_rank_spearman_seed45_3e-4ep16.pt"),
        ("s46", "head_BERT_mlp_rank_spearman_seed46_3e-4ep16.pt"),
        ("s47", "head_BERT_mlp_rank_spearman_seed47_3e-4ep16.pt"),
        ("s48", "head_BERT_mlp_rank_spearman_seed48_3e-4ep16.pt"),
    ]
    ranks = []
    for name, fname in strong_specs:
        path = wr / fname
        if not path.exists():
            print(f"[f7] skip {name} (missing)")
            continue
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"])
        head.eval()
        with torch.no_grad():
            s = head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, s))
        rec[name] = s
        print(f"[f7] {name}: done")
    if not ranks:
        raise RuntimeError("no strong heads found")
    strong_ens = np.mean(ranks, axis=0)
    rec["T5_strong_ens"] = strong_ens
    rec["F_z_STRONG"] = 0.5 * z_per_date(dates, strong_ens) + 0.5 * z_per_date(dates, p6)
    rec["F_rs_STRONG"] = 0.5 * strong_ens + 0.5 * rank_p6
    rec["T5_dis"] = np.std(ranks, axis=0)          # ensemble disagreement

    # ---- confidence signals ----
    s_ = np.load(wr / "scores_eval_K8_w512_stride1.npz", allow_pickle=True)
    logp = np.asarray(s_["logp_bert_full"])
    logpb = logp
    ent = -np.sum(np.where(np.isfinite(logpb), np.exp(np.clip(logpb, -50, 50)) * logpb, 0.0),
                  axis=1)
    rec["c_entropy"] = ent
    qg = np.load(wr / "gpt_q_eval_full128.npz", allow_pickle=True)["q"]
    pb = np.exp(np.clip(logpb, -50, 50))
    pb = pb / np.maximum(pb.sum(axis=1, keepdims=True), 1e-12)
    eps = 1e-12
    m = 0.5 * (pb + qg)
    rec["c_js"] = 0.5 * (np.sum(pb * np.log((pb + eps) / (m + eps)), axis=1)
                         + np.sum(qg * np.log((qg + eps) / (m + eps)), axis=1))
    rec["c_pabs"] = np.abs(np.asarray(cand["p_up"], dtype=np.float64) - 0.5)
    rec["c_dis"] = rec["T5_dis"]

    fields = {"T5_strong_ens": "T5_strong_ens", "F_rs_STRONG": "F_rs_STRONG",
              "F_z_STRONG": "F_z_STRONG", "P6": "p6_score"}
    base = metrics_table(rec, fields, dense)
    for name in fields:
        print(f"[f7] full {name:16s} rank_ic={base[name]['avg_daily_rank_ic']:.4f} "
              f"da={base[name]['avg_da_per_date']:.4f}")

    # ---- coverage curves ----
    score_field = args.score_field
    if score_field not in rec:
        raise RuntimeError(f"score field {score_field} not in rec; choose from {list(rec.keys())}")
    res = {"schema": "f7-abstain-strong-v1", "score": score_field,
           "dense_threshold": dense,
           "full": {k: v["avg_daily_rank_ic"] for k, v in base.items()},
           "coverage": {}}
    for cf, max_high in (("c_pabs", True), ("c_js", False),
                         ("c_dis", False), ("c_entropy", False)):
        row = {"signal": cf, "max_high": max_high, "curve": []}
        for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
            if cv == 1.0:
                acted = np.ones(n, dtype=bool)
            else:
                acted = _apply_rule(rec, cf, cv, max_high=max_high)
            rec_a = slice_rec(rec, acted)
            dense_act = max(5, int(round(cv * dense)))
            ic_ = DailyIcCache(rec_a, dense_act).series(rec_a[score_field])
            aic = float(np.mean(list(ic_.values()))) if ic_ else None
            da_ = _daily_da(rec_a, score_field, dense_act)
            ada = float(np.mean(list(da_.values()))) if da_ else None
            row["curve"].append({"cov": cv, "acted_frac": float(acted.mean()),
                                 "acted_rank_ic": aic, "acted_da": ada})
            print(f"[f7] {cf:10s} cov={cv:4.2f} acted_ic={aic if aic is None else round(aic,4)} "
                  f"acted_da={ada if ada is None else round(ada,4)}")
        res["coverage"][cf] = row

    out = rr / "f7_abstain_strong.json"
    write_json_ledger(out, res, "f7_abstain_strong", score=score_field)
    print(f"[f7] -> {out}")


def _daily_da(rec, field, dense):
    from improve_common import _split_points
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
