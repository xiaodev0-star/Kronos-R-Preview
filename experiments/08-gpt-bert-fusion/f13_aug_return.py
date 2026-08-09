"""f13_aug_return.py — augmented BERT head + BERT-hidden return head.

A) Rank head on [BERT hidden (256) + c1 point-in-time features (5)] = 261-dim.
   Does explicit momentum/vol signal add to BERT hidden for ranking?
B) MlpReturnHead (Huber) on BERT hidden -> raw-logret magnitude channel.
   Does a direct return head on BERT hidden beat isotonic-calibrated rank score
   for MAPE / DA?

Fit data: BERT fit hidden + training_cache c1_feats/true_logret (aligned).
Eval: BERT eval hidden + 06 eval_c1_feats.npz.
"""
from __future__ import annotations

import argparse
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

from train_heads import final_fit_split, fold_split, train_rank_per_date, train_one  # noqa: E402
from posttrain_heads import MlpRankHead, MlpReturnHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, build_rec, cand_path,
    write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


def _fit_rows(wr):
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    b = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    assert np.array_equal(tc["stock_uid"], b["stock_uid"])
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret", "c1_feats")}
    rows["hidden"] = np.concatenate([b["hidden"], tc["c1_feats"]], axis=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=54)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    rows = _fit_rows(wr)
    print(f"[f13] aug fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")
    fit_idx = final_fit_split(rows)

    # A) augmented rank head (quick R2 check + final fit)
    fit_r2, val_r2 = fold_split(rows, "R2")
    h0, hist0 = train_rank_per_date(
        MlpRankHead(dim=rows["hidden"].shape[1], hidden=64, dropout=0.1,
                    loss="soft_spearman"),
        rows, fit_r2, val_r2, "soft_spearman", lr=3e-4, epochs=4, seed=args.seed)
    print(f"[f13] aug rank R2 sanity val={hist0['val_loss'][-1]:.4f}")
    head_r = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64, dropout=0.1,
                         loss="soft_spearman")
    _, _ = train_rank_per_date(head_r, rows, fit_idx, fit_idx, "soft_spearman",
                               lr=3e-4, epochs=16, seed=args.seed)
    torch.save({"head_state": head_r.state_dict(),
                "recipe": {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
                "seed": args.seed, "kind": "aug_rank"}, wr / f"head_BERT_aug_rank_seed{args.seed}.pt")

    # B) return head on BERT hidden only
    rows2 = {k: rows[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows2["hidden"] = rows["hidden"][:, :256]
    # quick R2
    h0b, _ = train_one(MlpReturnHead(dim=256, hidden=64, dropout=0.1),
                       rows2, fit_r2, val_r2, "huber", lr=3e-4, epochs=4,
                       batch_size=4096, seed=args.seed)
    head_m = MlpReturnHead(dim=256, hidden=64, dropout=0.1)
    _, _ = train_one(head_m, rows2, fit_idx, fit_idx, "huber", lr=3e-4,
                     epochs=8, batch_size=4096, seed=args.seed)
    torch.save({"head_state": head_m.state_dict(),
                "recipe": {"lr": 3e-4, "epochs": 8, "dropout": 0.1},
                "seed": args.seed, "kind": "return"}, wr / f"head_BERT_return_seed{args.seed}.pt")
    print(f"[f13] trained aug rank + return heads")

    # ---- eval apply ----
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    c1 = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "eval_c1_feats.npz", allow_pickle=True)
    if len(c1["c1_feats"]) != n:
        raise RuntimeError("eval c1_feats misaligned")
    H_aug = torch.from_numpy(np.concatenate([de["hidden"], c1["c1_feats"]], axis=1).astype(np.float32))

    head_r.eval(); head_m.eval()
    with torch.no_grad():
        rec["aug_rank"] = head_r(H_aug).numpy().astype(np.float64)
        rec["ret_head"] = head_m(H).numpy().astype(np.float64)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["Fz_aug"] = 0.5 * z_per_date(dates, rec["aug_rank"]) + 0.5 * z_per_date(dates, p6)
    fields = {"aug_rank": "aug_rank", "Fz_aug": "Fz_aug", "ret_head": "ret_head",
              "J3": "post_median"}
    full = metrics_table(rec, fields, dense)
    for k, v in full.items():
        print(f"[f13] full {k:10s} rank_ic={v['avg_daily_rank_ic']:.4f} "
              f"da={v['avg_da_per_date']:.4f} mape={v['avg_mape']:.4f} mae={v['avg_mae']:.4f}")

    res = {"schema": "f13-aug-return-v1", "seed": args.seed,
           "full": {k: {kk: v[kk] for kk in ("avg_daily_rank_ic", "avg_da_per_date",
                                             "avg_mape", "avg_mae")} for k, v in full.items()}}
    out = rr / "f13_aug_return.json"
    write_json_ledger(out, res, "f13_aug_return", seed=args.seed)


if __name__ == "__main__":
    main()
