"""f23_rankft_eval.py — evaluate the rank-aware fine-tuned BERT.

1. Cache the fine-tuned BERT's MASK-position hidden for the same subsets the
   probe used (fit first-150k, eval first-300k).
2. Train a fresh T5-recipe rank head on the fine-tuned fit hidden.
3. Apply to the fine-tuned eval hidden; report RankIC vs the frozen-T2 (0.0535)
   and frozen-T1 (0.0569) baselines on the same eval subset.

Usage:
  python f23_rankft_eval.py --bert checkpoints/bert_critic_rankft.pt --suffix rankft
"""
from __future__ import annotations

import argparse
import subprocess
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

from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import weights_root, results_root, cand_path, _split_points  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def _daily_ic(rec, field, dense):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default=str(ROOT / "checkpoints" / "bert_critic_rankft.pt"))
    ap.add_argument("--suffix", default="rankft")
    ap.add_argument("--fit_rows", type=int, default=150000)
    ap.add_argument("--eval_rows", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=80)
    args = ap.parse_args()

    wr = weights_root()
    rr = results_root()

    # 1) cache fine-tuned hidden for the subsets
    for region, n in (("fit", args.fit_rows), ("eval", args.eval_rows)):
        cmd = [sys.executable, str(SEVEN / "cache_bert_hidden.py"),
               "--region", region, "--bert", args.bert,
               "--max_rows", str(n), "--suffix", args.suffix]
        print(f"[f23] running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-600:])
        if r.returncode != 0:
            print("[f23] cache failed:", r.stderr[-600:])
            return

    # 2) train head on fine-tuned fit hidden
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    rows = {k: tc[k][:args.fit_rows] for k in ("stock_uid", "date_key", "true_logret")}
    ft = np.load(wr / f"bert_hidden_fit_w512_{args.suffix}.npz", allow_pickle=True)
    rows["hidden"] = ft["hidden"]
    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=3e-4, epochs=16, seed=args.seed)
    print(f"[f23] head trained, loss={hist['train_loss'][-1]:.4f}")

    # 3) apply to fine-tuned eval hidden
    cand = np.load(cand_path("eval"), allow_pickle=True)
    ne = args.eval_rows
    et = np.load(wr / f"bert_hidden_eval_w512_{args.suffix}.npz", allow_pickle=True)
    rec = {"date_key": cand["date_key"][:ne],
           "true_logret": cand["true_logret"][:ne].astype(np.float64),
           "quality": cand["quality"][:ne].astype(bool)}
    head.eval()
    with torch.no_grad():
        sc = head(torch.from_numpy(et["hidden"].astype(np.float32))).numpy().astype(np.float64)
    rec["score"] = sc
    dense_probe = max(5, int(round(0.8 * 750)))
    ic = _daily_ic(rec, "score", dense_probe)
    full_ic = float(np.mean(list(ic.values())))
    print(f"[f23] rankft BERT: eval full_ic={full_ic:.4f} (vs T2 frozen 0.0535 / T1 frozen 0.0569)")
    res = {"schema": "f23-rankft-eval-v1", "bert": args.bert,
           "full_ic": full_ic, "n_dates": len(ic),
           "baselines": {"T2_frozen": 0.0535, "T1_frozen": 0.0569}}
    out = rr / "f23_rankft_eval.json"
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[f23] -> {out}")


if __name__ == "__main__":
    main()
