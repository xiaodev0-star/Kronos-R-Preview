"""f19_gpt_head_strong.py — retrain the GPT-hidden rank head with the T5 recipe.

The 06 P6 head used lr 3e-4/6ep (older recipe).  The BERT heads use 3e-4/16ep.
Train P6-style heads on GPT hidden with the STRONGER recipe (multi-seed), and
check whether a stronger GPT head improves the BERT+GPT fusion.
"""
from __future__ import annotations

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

from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, build_rec, cand_path,
    write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


def main():
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = tc["hidden"]
    fit_idx = final_fit_split(rows)
    print(f"[f19] GPT fit rows={len(fit_idx)}")

    # train 3 GPT heads, T5 recipe
    heads = {}
    for s in (60, 61, 62):
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                      lr=3e-4, epochs=16, seed=s)
        torch.save({"head_state": head.state_dict(),
                    "recipe": {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
                    "seed": s}, wr / f"head_GPT_strong_seed{s}.pt")
        heads[s] = head
        print(f"[f19] trained GPT strong head s{s} loss={hist['train_loss'][-1]:.4f}")

    # eval
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    gh = np.load(sw / "hidden_cache.npz", allow_pickle=True)
    H = torch.from_numpy(gh["hidden"].astype(np.float32))
    g_ranks = []
    for s, head in heads.items():
        head.eval()
        with torch.no_grad():
            sc = head(H).numpy().astype(np.float64)
        rec[f"gpt_s{s}"] = sc
        g_ranks.append(rank_pct_per_date(dates, sc))
    ens_g = np.mean(g_ranks, axis=0)
    rec["gpt_ens"] = ens_g

    # BERT ens (6 heads)
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    Hb = torch.from_numpy(de["hidden"].astype(np.float32))
    b_ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        hb = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        hb.load_state_dict(ck["head_state"]); hb.eval()
        with torch.no_grad():
            b_ranks.append(rank_pct_per_date(dates, hb(Hb).numpy().astype(np.float64)))
    ens_b = np.mean(b_ranks, axis=0)
    rec["bert_ens"] = ens_b

    rec["F_gptstrong"] = 0.5 * z_per_date(dates, ens_b) + 0.5 * z_per_date(dates, ens_g)
    rec["F_oldp6"] = 0.5 * z_per_date(dates, ens_b) + 0.5 * z_per_date(dates,
                      np.load(wr / "p6_eval_scores.npy").astype(np.float64))
    fields = {"gpt_ens": "gpt_ens", "bert_ens": "bert_ens",
              "F_gptstrong": "F_gptstrong", "F_oldp6": "F_oldp6"}
    full = metrics_table(rec, fields, dense)
    for k, v in full.items():
        print(f"[f19] full {k:12s} rank_ic={v['avg_daily_rank_ic']:.4f} "
              f"da={v['avg_da_per_date']:.4f}")

    res = {"schema": "f19-gpt-head-strong-v1",
           "full": {k: v["avg_daily_rank_ic"] for k, v in full.items()}}
    out = rr / "f19_gpt_head_strong.json"
    write_json_ledger(out, res, "f19_gpt_head_strong")
    print(f"[f19] -> {out}")


if __name__ == "__main__":
    main()
