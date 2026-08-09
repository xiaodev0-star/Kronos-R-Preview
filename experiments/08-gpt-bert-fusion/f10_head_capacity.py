"""f10_head_capacity.py — test head-capacity variants on BERT hidden (base score).

Variant heads (all frozen BERT hidden -> rank head, soft-Spearman, R2-only
recipe selection to bound runtime, final fit on <2023-02-01):
  W64   MlpRankHead hidden=64   (current T5 recipe baseline)
  W128  MlpRankHead hidden=128
  DEEP  MlpRankHead3 hidden=128->64 (3 layers)
Evaluated on eval: base score + c_dis abstention at 20%.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
SIX = ROOT / "experiments" / "06-posttrain"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, SIX, EIGHT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_heads import fold_split, final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, build_rec, cand_path,
    slice_rec, DailyIcCache, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


class MlpRankHead3(nn.Module):
    def __init__(self, dim=256, hidden=128, dropout=0.1, loss="soft_spearman"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1))
        self.loss = loss

    def forward(self, h):
        return self.net(h).squeeze(-1)


def _rows_from_hidden(hidden_path):
    d = np.load(hidden_path, allow_pickle=True)
    rows = {k: d[k] for k in d.files}
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    for k in ("true_logret", "date_key", "stock_uid"):
        rows[k] = tc[k]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["W64", "W128", "DEEP"], default="W128")
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    rows = _rows_from_hidden(wr / "bert_hidden_fit_w512_t2.npz")
    fit_idx, val_idx = fold_split(rows, "R2")
    print(f"[f10] variant={args.variant} R2 fit={len(fit_idx)} val={len(val_idx)}")

    if args.variant == "W64":
        factory = lambda: MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        recipe = {"lr": 3e-4, "epochs": 16, "dropout": 0.1}
    elif args.variant == "W128":
        factory = lambda: MlpRankHead(dim=256, hidden=128, dropout=0.1, loss="soft_spearman")
        recipe = {"lr": 3e-4, "epochs": 16, "dropout": 0.1}
    else:
        factory = lambda: MlpRankHead3(dim=256, hidden=128, dropout=0.1)
        recipe = {"lr": 3e-4, "epochs": 16, "dropout": 0.1}

    # quick R2 sanity (1 config, few epochs) — diagnostic only
    h, hist = train_rank_per_date(factory(), rows, fit_idx, val_idx, "soft_spearman",
                                  lr=recipe["lr"], epochs=4, seed=args.seed)
    print(f"[f10] R2 sanity val_loss={hist['val_loss'][-1]:.4f}")

    # final fit
    fit_all = final_fit_split(rows)
    head = factory()
    _, hist = train_rank_per_date(head, rows, fit_all, fit_all, "soft_spearman",
                                  lr=recipe["lr"], epochs=recipe["epochs"], seed=args.seed)
    out_head = wr / f"head_BERT_{args.variant}_rank_seed{args.seed}.pt"
    torch.save({"head_state": head.state_dict(), "recipe": recipe, "seed": args.seed,
                "variant": args.variant}, out_head)
    print(f"[f10] saved {out_head}")

    # eval apply
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    head.eval()
    with torch.no_grad():
        s = head(torch.from_numpy(de["hidden"].astype(np.float32))).numpy().astype(np.float64)
    rec["score"] = s
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F_z"] = 0.5 * z_per_date(dates, s) + 0.5 * z_per_date(dates, p6)

    # c_dis from 6 strong heads
    strong_specs = [f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt" for s in range(45, 51)]
    ranks = []
    for fname in strong_specs:
        ck = torch.load(str(wr / fname), map_location="cpu", weights_only=False)
        hh = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        hh.load_state_dict(ck["head_state"]); hh.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, hh(torch.from_numpy(de["hidden"].astype(np.float32))).numpy().astype(np.float64)))
    rec["c_dis"] = np.std(ranks, axis=0)

    fields = {"score": "score", "F_z": "F_z"}
    full = metrics_table(rec, fields, dense)
    print(f"[f10] full score rank_ic={full['score']['avg_daily_rank_ic']:.4f} "
          f"F_z rank_ic={full['F_z']['avg_daily_rank_ic']:.4f}")

    # coverage at 20% with c_dis
    acted = _apply_rule(rec, "c_dis", 0.2, max_high=False)
    rec_a = slice_rec(rec, acted)
    da_act = max(5, int(round(0.2 * dense)))
    ic20 = np.mean(list(DailyIcCache(rec_a, da_act).series(rec_a["F_z"]).values()))
    print(f"[f10] F_z + c_dis @20% acted_ic={ic20:.4f}")

    res = {"schema": "f10-head-capacity-v1", "variant": args.variant, "seed": args.seed,
           "recipe": recipe, "full": {k: v["avg_daily_rank_ic"] for k, v in full.items()},
           "F_z_cdis20": float(ic20)}
    out = rr / f"f10_{args.variant}.json"
    write_json_ledger(out, res, "f10_head_capacity", variant=args.variant)


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
