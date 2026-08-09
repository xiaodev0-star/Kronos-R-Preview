"""f6_ensemble_eval.py — Round 3: evaluate all trained T5 heads + ensembles + P6 fusion.

Applies every BERT-hidden rank head to the eval hidden cache, builds
  - per-head scores
  - strong ensemble (3e-4/16 recipe, seeds 45..48)
  - all-recipe ensemble (every head)
  - fused with P6 at rank-sum / z-sum
and reports full-400 + dev/confirm + paired moving-block bootstrap vs the
current best single reference (T5 s45) and P6.
"""
from __future__ import annotations

import argparse
import json
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
    cand_path, slice_rec, write_json_ledger,
)
from posttrain_heads import MlpRankHead  # noqa: E402
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


def apply_head(head_path, hidden, dim=256, hidden_dim=64, dropout=0.1):
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=dim, hidden=hidden_dim, dropout=dropout, loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)


HEAD_SPECS = [
    ("s42_1e-3:12", "head_BERT_mlp_rank_spearman_seed42.pt", {"dropout": 0.1}),
    ("s43_1e-3:12", "head_BERT_mlp_rank_spearman_seed43_1e-3ep12.pt", {"dropout": 0.1}),
    ("s44_1e-3:12", "head_BERT_mlp_rank_spearman_seed44_1e-3ep12.pt", {"dropout": 0.1}),
    ("s45_3e-4:16", "head_BERT_mlp_rank_spearman_seed45_3e-4ep16.pt", {"dropout": 0.1}),
    ("s46_3e-4:16", "head_BERT_mlp_rank_spearman_seed46_3e-4ep16.pt", {"dropout": 0.1}),
    ("s47_3e-4:16", "head_BERT_mlp_rank_spearman_seed47_3e-4ep16.pt", {"dropout": 0.1}),
    ("s48_3e-4:16", "head_BERT_mlp_rank_spearman_seed48_3e-4ep16.pt", {"dropout": 0.1}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=7, help="use first N heads")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    assert len(de["stock_uid"]) == n
    hidden = de["hidden"]
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])

    scores = {}
    ranks = {}
    specs = HEAD_SPECS[:args.heads]
    for name, fname, kw in specs:
        path = wr / fname
        if not path.exists():
            print(f"[f6] skip {name} (missing {path})")
            continue
        s = apply_head(path, hidden, **kw)
        scores[name] = s
        ranks[name] = rank_pct_per_date(dates, s)
        print(f"[f6] {name}: done")

    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["p6_score"] = p6
    rank_p6 = rank_pct_per_date(dates, p6)

    fields = {"P6": "p6_score"}
    for name in scores:
        rec[name] = scores[name]
        fields[name] = name
        rec["rank_" + name] = ranks[name]

    # strong ensemble = 3e-4/16 heads
    strong = [k for k in ranks if "3e-4:16" in k]
    if strong:
        ens = np.mean([ranks[k] for k in strong], axis=0)
        rec["T5_strong_ens"] = ens
        fields["T5_strong_ens"] = "T5_strong_ens"
    all_ens = np.mean([ranks[k] for k in ranks], axis=0) if ranks else None
    if all_ens is not None:
        rec["T5_all_ens"] = all_ens
        fields["T5_all_ens"] = "T5_all_ens"

    # fusion with P6
    for tag, ens_field in (("STRONG", "T5_strong_ens"), ("ALL", "T5_all_ens")):
        if ens_field not in rec:
            continue
        rec[f"F_rs_{tag}"] = 0.5 * rec[ens_field] + 0.5 * rank_p6
        fields[f"F_rs_{tag}"] = f"F_rs_{tag}"
        rec[f"F_z_{tag}"] = 0.5 * z_per_date(dates, rec[ens_field]) \
            + 0.5 * z_per_date(dates, p6)
        fields[f"F_z_{tag}"] = f"F_z_{tag}"

    res = {"schema": "f6-ensemble-eval-v1", "heads": [s[0] for s in specs],
           "n_rows": n, "dense_threshold": dense}
    res["full"] = metrics_table(rec, fields, dense)
    res["dev_0_299"] = metrics_table(slice_rec(rec, (np.asarray(rec["offset"]) >= 0) & (np.asarray(rec["offset"]) <= 299)), fields, dense)
    res["confirm_300_399"] = metrics_table(slice_rec(rec, (np.asarray(rec["offset"]) >= 300) & (np.asarray(rec["offset"]) < 400)), fields, dense)

    # bootstrap vs best single (s45) and vs P6
    ref_best = "s45_3e-4:16" if "s45_3e-4:16" in rec else list(scores)[0]
    res["bootstrap_vs_best"] = {}
    res["bootstrap_vs_P6"] = {}
    for name, field in fields.items():
        if field not in rec:
            continue
        if field != ref_best:
            res["bootstrap_vs_best"][name] = bootstrap_vs(rec, field, rec, ref_best, dense)
        res["bootstrap_vs_P6"][name] = bootstrap_vs(rec, field, rec, "p6_score", dense)

    out = rr / "f6_ensemble_eval.json"
    write_json_ledger(out, res, "f6_ensemble_eval", n_heads=args.heads)

    print("\n=== full-400 avg_daily_rank_ic ===")
    for name, field in fields.items():
        m = res["full"][name]
        print(f"  {name:22s} rank_ic={m['avg_daily_rank_ic']:.4f} da={m['avg_da_per_date']:.4f}")
    print("\n=== dev / confirm ===")
    for name in fields:
        d = res["dev_0_299"][name]["avg_daily_rank_ic"]
        c = res["confirm_300_399"][name]["avg_daily_rank_ic"]
        print(f"  {name:22s} dev={d:.4f} conf={c:.4f}")
    print("\n=== vs best single (T5 s45) ===")
    for name in fields:
        b = res["bootstrap_vs_best"].get(name)
        if b:
            print(f"  {name:22s} point={b.get('point', 0):+.4f} robust={b.get('block_robust')}")
    print(f"[f6] -> {out}")


if __name__ == "__main__":
    main()
