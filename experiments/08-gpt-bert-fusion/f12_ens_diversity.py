"""f12_ens_diversity.py — add diversity to the BERT-head ensemble.

Trains:
  - 2 more soft-Spearman heads (seeds 51, 52, 3e-4/16)
  - 1 pairwise-loss head (P7-style, seeds 53, 3e-4/16)  [different gradient
    structure -> better c_dis signal]
Then builds an 9-head ensemble (or 8 without pairwise), tests:
  - base score variants (ens, F_z with P6, F_z with P6+J4)
  - c_dis abstention at 20% (exact coverage)
Saves new head artifacts.
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

from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, build_rec, cand_path,
    slice_rec, DailyIcCache, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402


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
    ap.add_argument("--train_extra", action="store_true")
    ap.add_argument("--apply_only", action="store_true")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    rows = _rows_from_hidden(wr / "bert_hidden_fit_w512_t2.npz")
    fit_idx = final_fit_split(rows)

    if args.train_extra:
        # 2 more soft-Spearman
        for s in (51, 52):
            head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
            _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                          lr=3e-4, epochs=16, seed=s)
            torch.save({"head_state": head.state_dict(),
                        "recipe": {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
                        "seed": s, "loss": "soft_spearman"},
                       wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt")
            print(f"[f12] trained ss seed {s}")
        # pairwise head
        for s in (53,):
            head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="pairwise")
            _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "pairwise",
                                          lr=3e-4, epochs=16, seed=s)
            torch.save({"head_state": head.state_dict(),
                        "recipe": {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
                        "seed": s, "loss": "pairwise"},
                       wr / f"head_BERT_pairwise_seed{s}_3e-4ep16.pt")
            print(f"[f12] trained pairwise seed {s}")

    if args.apply_only or not args.train_extra:
        _apply(wr, rr)


def _apply(wr, rr):
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)

    def load(fname, loss="soft_spearman", drop=0.1):
        ck = torch.load(str(wr / fname), map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=drop, loss=loss)
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            return head(H).numpy().astype(np.float64)

    ss = [f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt" for s in range(45, 53)]
    ss_ranks = [rank_pct_per_date(dates, load(f)) for f in ss if (wr / f).exists()]
    pw_path = wr / "head_BERT_pairwise_seed53_3e-4ep16.pt"
    all_ranks = list(ss_ranks)
    if pw_path.exists():
        all_ranks.append(rank_pct_per_date(dates, load(pw_path.name, loss="pairwise")))

    ens_ss = np.mean(ss_ranks, axis=0)
    ens_all = np.mean(all_ranks, axis=0)
    rec["ens_ss"] = ens_ss
    rec["ens_all"] = ens_all
    rec["F_z_ss"] = 0.5 * z_per_date(dates, ens_ss) + 0.5 * z_per_date(dates, p6)
    rec["F_z_all"] = 0.5 * z_per_date(dates, ens_all) + 0.5 * z_per_date(dates, p6)
    # 3-way with J4
    j4 = np.asarray(cand["p_up"], dtype=np.float64)
    rec["F_z3"] = (0.4 * z_per_date(dates, ens_all) + 0.3 * z_per_date(dates, p6)
                   + 0.3 * z_per_date(dates, j4))
    # c_dis
    rec["c_dis_ss"] = np.std(ss_ranks, axis=0)
    rec["c_dis_all"] = np.std(all_ranks, axis=0)

    fields = {"ens_ss": "ens_ss", "ens_all": "ens_all", "F_z_ss": "F_z_ss",
              "F_z_all": "F_z_all", "F_z3": "F_z3"}
    full = metrics_table(rec, fields, dense)
    for k, v in full.items():
        print(f"[f12] full {k:10s} rank_ic={v['avg_daily_rank_ic']:.4f} "
              f"da={v['avg_da_per_date']:.4f}")

    res = {"schema": "f12-ens-diversity-v1", "dense_threshold": dense,
           "full": {k: v["avg_daily_rank_ic"] for k, v in full.items()},
           "cdis20": {}}
    for score_f in ("F_z_ss", "F_z_all", "F_z3"):
        for cf in ("c_dis_ss", "c_dis_all"):
            acted = _exact_frac(rec, cf, 0.2, max_high=False)
            rec_a = slice_rec(rec, acted)
            da20 = max(5, int(round(0.2 * dense)))
            ic20 = np.mean(list(DailyIcCache(rec_a, da20).series(rec_a[score_f]).values()))
            res["cdis20"][f"{score_f}+{cf}"] = float(ic20)
            print(f"[f12] {score_f:10s} + {cf:10s} @20% acted_ic={ic20:.4f}")

    out = rr / "f12_ens_diversity.json"
    write_json_ledger(out, res, "f12_ens_diversity")


def _exact_frac(rec, cf, cv, max_high=True):
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
