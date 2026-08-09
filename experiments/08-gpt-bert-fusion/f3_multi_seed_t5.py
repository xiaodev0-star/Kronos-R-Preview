"""f3_multi_seed_t5.py — Round 1c: multi-seed / multi-recipe BERT-head ensemble.

The T5 head is strong (0.069-0.077) but recipe-selection-fragile (R0-R2 rolling
noise flips the near-optimal recipe; docs IMPROVEMENT_FINAL §0-5).  A single
head is seed/recipe luck.  This trains several BERT-hidden rank heads (varying
optimizer seed AND the two best-known recipes), then produces:

  - per-head eval scores + metrics
  - an ENSEMBLE score = within-date mean of per-head rank-percentiles
    (scale-free output-space averaging; zero fitted parameters on 0..399)

Each head trains on the fit region with a FIXED (locked) recipe — no R0-R2
selection here (that was done once in the T5 study; recipe locking is the
protocol-correct way to add replication seeds).

Usage:
  python f3_multi_seed_t5.py --combos '[["42","3e-4:16"],["43","1e-3:12"],["44","1e-3:12"]]'
  combos entries: seed_str, recipe_key with keys "1e-3:12" or "3e-4:16" (lr:epochs)
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
SIX = ROOT / "experiments" / "06-posttrain"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, SIX, EIGHT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from posttrain_heads import MlpRankHead  # noqa: E402
from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, DailyIcCache, _split_points, write_json_ledger,
)

RECIPES = {
    "1e-3:12": {"lr": 1e-3, "epochs": 12, "dropout": 0.1},
    "3e-4:16": {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
}


def _rows_from_hidden(hidden_path):
    d = np.load(hidden_path, allow_pickle=True)
    rows = {k: d[k] for k in d.files}
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    if len(tc["stock_uid"]) == len(rows["stock_uid"]):
        for k in ("true_logret", "date_key", "stock_uid"):
            rows[k] = tc[k]
    else:
        raise RuntimeError("fit hidden cache rows != training_cache rows")
    return rows


def _rank_pct_per_date(dates, score):
    score = np.asarray(score, dtype=np.float64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", required=True,
                    help='JSON list of [seed, recipe_key] e.g. \'[["43","1e-3:12"],["44","1e-3:12"]]\'')
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    combos = json.loads(args.combos)
    wr = weights_root()
    rr = results_root()

    rows = _rows_from_hidden(wr / "bert_hidden_fit_w512_t2.npz")
    fit_idx = final_fit_split(rows)
    print(f"[f3] fit rows={len(fit_idx)}")

    heads = {}
    for seed, rkey in combos:
        recipe = RECIPES[rkey]
        head = MlpRankHead(dim=256, hidden=64, dropout=recipe["dropout"],
                           loss="soft_spearman")
        _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                      lr=recipe["lr"], epochs=recipe["epochs"],
                                      seed=int(seed))
        out_head = wr / f"head_BERT_mlp_rank_spearman_seed{seed}_{rkey.replace(':','ep')}.pt"
        torch.save({"head_state": head.state_dict(), "recipe": recipe,
                    "seed": int(seed), "train_history": hist,
                    "schema": "f3-t5-multiseed-v1"}, out_head)
        heads[(seed, rkey)] = head
        print(f"[f3] saved {out_head} final_train_loss={hist['train_loss'][-1]:.4f}")

    if args.eval:
        cand = np.load(cand_path("eval"), allow_pickle=True)
        de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
        n = len(cand["stock_uid"])
        assert len(de["stock_uid"]) == n
        H = torch.from_numpy(de["hidden"].astype(np.float32))
        rec = build_rec(cand)
        dates = np.asarray(rec["date_key"])
        fields = {}
        ens_rank = np.zeros(n)
        for (seed, rkey), head in heads.items():
            head.eval()
            with torch.no_grad():
                s = head(H).numpy().astype(np.float64)
            rec[f"t5_s{seed}_{rkey}"] = s
            fields[f"T5_s{seed}_{rkey.replace(':','_')}"] = f"t5_s{seed}_{rkey}"
            ens_rank += _rank_pct_per_date(dates, s)
        ens_rank /= len(heads)
        rec["t5_ens"] = ens_rank
        fields["T5_ENS"] = "t5_ens"
        dense = int(cand["dense_threshold"][0])
        res = {"schema": "f3-multiseed-t5-v1", "combos": combos,
               "n_rows": n, "dense_threshold": dense}
        res["full"] = metrics_table(rec, fields, dense)
        res["bootstrap_vs_T5s42"] = bootstrap_vs(rec, "t5_ens", rec, "t5_s42_1e-3:12", dense) \
            if "t5_s42_1e-3:12" in rec else None
        res["bootstrap_vs_P6"] = bootstrap_vs(rec, "t5_ens", rec, "p6_score", dense) \
            if "p6_score" in rec else None
        dev_off = np.asarray(rec["offset"] if "offset" in rec else cand["offset"],
                             dtype=np.int64)
        # dev/confirm via candidates offset
        from improve_common import slice_rec
        dev = slice_rec(rec, (dev_off >= 0) & (dev_off <= 299))
        conf = slice_rec(rec, (dev_off >= 300) & (dev_off < 400))
        res["dev_0_299"] = metrics_table(dev, fields, dense)
        res["confirm_300_399"] = metrics_table(conf, fields, dense)
        out = rr / "f3_multi_seed_t5.json"
        write_json_ledger(out, res, "f3_multiseed_t5")
        print("\n=== full-400 ===")
        for name, field in fields.items():
            m = res["full"][name]
            print(f"  {name:12s} rank_ic={m['avg_daily_rank_ic']:.4f}")
        print("=== dev / confirm (ENS) ===")
        print(f"  dev   {res['dev_0_299']['T5_ENS']['avg_daily_rank_ic']:.4f}")
        print(f"  conf  {res['confirm_300_399']['T5_ENS']['avg_daily_rank_ic']:.4f}")
        if res.get("bootstrap_vs_T5s42"):
            b = res["bootstrap_vs_T5s42"]
            print(f"  ENS vs s42 point {b.get('point',0):+.4f} robust={b.get('block_robust')}")
        print(f"[f3] -> {out}")


if __name__ == "__main__":
    main()
