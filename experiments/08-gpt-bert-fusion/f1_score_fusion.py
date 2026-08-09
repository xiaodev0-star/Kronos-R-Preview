"""f1_score_fusion.py — Round 1a: score-level fusion of the two best frozen-head arms.

The two strongest known row scores are:
  P6  = GPT causal-hidden MLP rank head  (eval RankIC ~0.0535, 06 finalist)
  T5  = BERT bidirectional-hidden MLP rank head (eval RankIC ~0.069-0.077, 07 milestone)

Both are frozen-backbone readouts on decorrelated representations (BERT vs GPT),
so output-space fusion is the cheapest, most direct realisation of the user's
"GPT predict + BERT score/filter" goal.  This script is PURE CPU / cached —
no training, no fitting on 0..399.

Arms (all zero-parameter unless stated):
  F-ranksum : within-date rank-percentile average (scale-free ensemble)
  F-zsum    : within-date z-score average
  F-w       : weight grid w*rank_BERT + (1-w)*rank_GPT  (diagnostic only;
              w fit on calib slice is reported separately as calib-fit diagnostic)
  F-ic-avg  : average the per-date IC ... (not used; keep simple)

Reported per arm: full-400 metrics, dev/confirm, paired moving-block bootstrap
vs T5-alone and vs P6-alone.
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

from improve_common import (
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, DailyIcCache, slice_rec, eval_dev_confirm, write_json_ledger,
)
from posttrain_heads import MlpRankHead  # noqa: E402
import torch  # noqa: E402


def _apply_head(head_path, hidden, dim=256, hidden_dim=64, dropout=0.1):
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=dim, hidden=hidden_dim, dropout=dropout,
                       loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)


def _rank_pct_per_date(rec, field):
    """Within-date rank percentile (0..1) for a score field."""
    score = np.asarray(rec[field], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    out = np.full(len(score), np.nan)
    s = score[order]
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blk = s[lo:hi]
        m = np.isfinite(blk)
        if m.sum() == 0:
            continue
        r = np.argsort(np.argsort(blk[m], kind="stable")).astype(float)
        r = r / max(1, len(r) - 1)
        tmp = np.full(blk.shape, np.nan)
        tmp[m] = r
        out[order[lo:hi]] = tmp
    return out


def _z_per_date(rec, field):
    score = np.asarray(rec[field], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    out = np.full(len(score), np.nan)
    s = score[order]
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blk = s[lo:hi]
        m = np.isfinite(blk)
        if m.sum() < 2:
            continue
        mu, sd_ = blk[m].mean(), blk[m].std()
        if sd_ == 0:
            continue
        tmp = np.full(blk.shape, np.nan)
        tmp[m] = (blk[m] - mu) / sd_
        out[order[lo:hi]] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None,
                    help="T5 BERT-head checkpoint (default weights_root/head_BERT_mlp_rank_spearman_seed42.pt)")
    ap.add_argument("--tag", type=str, default="t5")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])

    # ---- T5 BERT-head score ----
    head_path = Path(args.head) if args.head else wr / "head_BERT_mlp_rank_spearman_seed42.pt"
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    if len(de["stock_uid"]) != n:
        raise RuntimeError("bert eval hidden misaligned")
    rec["bert_head"] = _apply_head(head_path, de["hidden"])
    del de

    # ---- P6 GPT-head score ----
    p6 = np.load(wr / "p6_eval_scores.npy")
    if len(p6) != n:
        raise RuntimeError(f"p6 len {len(p6)} != {n}")
    rec["p6_score"] = p6.astype(np.float64)

    dense = int(cand["dense_threshold"][0])
    print(f"[f1] rows={n} dense={dense}")

    # ---- per-date decorrelation ----
    cc = DailyIcCache(rec, dense)
    ic_bert = cc.series(rec["bert_head"])
    ic_p6 = cc.series(rec["p6_score"])
    common = sorted(set(ic_bert) & set(ic_p6))
    import numpy as _np
    from scipy.stats import spearmanr as _sp
    # per-date score correlation (raw scores, pooled per date) on a sample
    rng = _np.random.RandomState(0)
    idx = rng.choice(n, min(n, 400_000), replace=False)
    bb = rec["bert_head"][idx]; pp = rec["p6_score"][idx]
    m = _np.isfinite(bb) & _np.isfinite(pp)
    rho = float(_sp(bb[m], pp[m])[0])
    ic_corr = float(_np.corrcoef([ic_bert[d] for d in common],
                                 [ic_p6[d] for d in common])[0, 1])
    ic_vals = _np.asarray([ic_bert[d] - ic_p6[d] for d in common])
    print(f"[f1] score-spearman(T5,P6)={rho:.4f}  per-date-IC-correl={ic_corr:.4f}")

    # ---- arms ----
    rec["rank_bert"] = _rank_pct_per_date(rec, "bert_head")
    rec["rank_p6"] = _rank_pct_per_date(rec, "p6_score")
    rec["z_bert"] = _z_per_date(rec, "bert_head")
    rec["z_p6"] = _z_per_date(rec, "p6_score")
    rec["F_ranksum"] = 0.5 * rec["rank_bert"] + 0.5 * rec["rank_p6"]
    rec["F_zsum"] = 0.5 * rec["z_bert"] + 0.5 * rec["z_p6"]

    fields = {"T5": "bert_head", "P6": "p6_score",
              "F_ranksum": "F_ranksum", "F_zsum": "F_zsum"}
    res = {"schema": "f1-score-fusion-v1", "n_rows": n,
           "rho_score": rho, "ic_corr": ic_corr,
           "dense_threshold": dense,
           "decorrelation": {"score_spearman": rho,
                             "per_date_ic_correlation": ic_corr}}

    # weight grid (diagnostic; no formal claim)
    grid = {}
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        field = f"F_w{w:g}"
        rec[field] = w * rec["rank_bert"] + (1 - w) * rec["rank_p6"]
        grid[f"w={w:g}"] = float(metrics_table(rec, {field: field}, dense)[field]["avg_daily_rank_ic"])
        fields[field] = field
    res["weight_grid"] = grid
    res["weight_grid_best"] = max(grid, key=grid.get)

    res["full"] = metrics_table(rec, fields, dense)
    dev, conf = eval_dev_confirm(rec)
    res["dev_0_299"] = metrics_table(dev, fields, dense)
    res["confirm_300_399"] = metrics_table(conf, fields, dense)

    # paired moving-block bootstrap vs T5-alone and vs P6-alone
    res["bootstrap_vs_T5"] = {}
    res["bootstrap_vs_P6"] = {}
    for name, field in fields.items():
        if field not in rec or field is None:
            continue
        res["bootstrap_vs_T5"][name] = bootstrap_vs(rec, field, rec, "bert_head", dense)
        res["bootstrap_vs_P6"][name] = bootstrap_vs(rec, field, rec, "p6_score", dense)

    out = rr / f"f1_score_fusion_{args.tag}.json"
    write_json_ledger(out, res, "f1_score_fusion", tag=args.tag)
    # console summary
    print("\n=== full-400 avg_daily_rank_ic ===")
    for name, field in fields.items():
        m = res["full"][name]
        print(f"  {name:12s} rank_ic={m['avg_daily_rank_ic']:.4f} da={m['avg_da_per_date']:.4f}")
    print("\n=== bootstrap vs T5 ===")
    for name in fields:
        b = res["bootstrap_vs_T5"][name]
        print(f"  {name:12s} vs_T5 {b.get('point', 0):+.4f} robust={b.get('block_robust')}")
    print("\n=== bootstrap vs P6 ===")
    for name in fields:
        b = res["bootstrap_vs_P6"][name]
        print(f"  {name:12s} vs_P6 {b.get('point', 0):+.4f} robust={b.get('block_robust')}")
    print("\nweight grid:", grid)
    print(f"[f1] -> {out}")


if __name__ == "__main__":
    main()
