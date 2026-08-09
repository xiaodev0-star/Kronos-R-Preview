"""apply_bert_head.py — apply a trained T5 BERT-head to the eval hidden cache.

Fast path: loads ``head_BERT_mlp_rank_spearman_seed42.pt`` (already trained) and
the eval hidden cache, computes the BERT-head RankIC on eval 0..399 with paired
moving-block bootstrap vs J3 and P6.  Avoids re-running the 1-2 h R0-R2 recipe
selection that ``train_bert_head.py --apply_eval`` redoes.
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
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, write_json_ledger,
)


def main():
    ap = argparse.ArgumentParser(description="Apply a trained T5 BERT-head to eval")
    ap.add_argument("--suffix", type=str, default="t2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wr = weights_root()

    head_path = wr / f"head_BERT_mlp_rank_spearman_seed{args.seed}.pt"
    if not head_path.exists():
        raise RuntimeError(f"head missing: {head_path}")
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    print(f"[apply] head <- {head_path}")

    sfx = f"_{args.suffix}" if args.suffix else ""
    eval_hidden = wr / f"bert_hidden_eval_w512{sfx}.npz"
    if not eval_hidden.exists():
        raise RuntimeError(f"eval hidden cache missing: {eval_hidden}")
    de = np.load(eval_hidden, allow_pickle=True)
    cand = np.load(cand_path("eval"), allow_pickle=True)
    n = len(cand["stock_uid"])
    if len(de["stock_uid"]) != n:
        raise RuntimeError(f"eval hidden rows {len(de['stock_uid'])} != candidates {n}")
    print(f"[apply] eval hidden rows {n}")

    H = torch.from_numpy(de["hidden"].astype(np.float32))
    with torch.no_grad():
        score = head(H).numpy()
    rec = build_rec(cand)
    rec["bert_head"] = score.astype(np.float64)
    p6 = np.load(wr / "p6_eval_scores.npy")
    if len(p6) != n:
        raise RuntimeError(f"P6 len {len(p6)} != candidates {n}")
    rec["p6_score"] = p6.astype(np.float64)
    dense = int(cand["dense_threshold"][0])

    res = {
        "schema": "t5-head-eval-v1", "seed": args.seed, "suffix": args.suffix,
        "head": str(head_path), "recipe": ck.get("recipe"),
        "eval_metrics": metrics_table(rec, {
            "BERT_head": "bert_head", "J3_median": "post_median",
            "J2_mean": "post_mean", "J4_pup": "p_up", "P6": "p6_score"},
            dense),
        "bootstrap_vs_J3": bootstrap_vs(rec, "bert_head", rec, "post_median", dense),
        "bootstrap_vs_P6": bootstrap_vs(rec, "bert_head", rec, "p6_score", dense),
    }
    out = results_root() / f"t5_head_eval_seed{args.seed}{sfx}.json"
    write_json_ledger(out, res, "t5_head_apply", seed=args.seed)
    print(json.dumps({
        "BERT_head": res["eval_metrics"]["BERT_head"],
        "vs_J3": res["bootstrap_vs_J3"],
        "vs_P6": res["bootstrap_vs_P6"],
    }, indent=1, default=str))
    print(f"[apply] -> {out}")


if __name__ == "__main__":
    main()
