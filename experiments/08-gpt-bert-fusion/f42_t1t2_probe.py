"""f42_t1t2_probe.py — quick subset test: does T1+T2 BERT ensemble help?

Uses the EXISTING T1 probe caches (150k fit / 300k eval) to answer the
T1+T2 representation-ensemble question fast, before investing in full caches.
  - train 3 T1 heads on the 150k T1-probe fit hidden
  - apply the 6 existing T2 heads to the first-300k eval hidden
  - compare F12 = 0.5 z(0.5 T1-ens + 0.5 T2-ens) + 0.5 z(P6) vs F on the same
    300k eval subset.
The absolute IC is noisy at this subset size; the RELATIVE delta is the signal.
"""
from __future__ import annotations

import argparse
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
from improve_common import (  # noqa: E402
    weights_root, results_root, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def daily_ic(rec, field, dense):
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
    ap.add_argument("--seed", type=int, default=170)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"

    # ---- train T1 heads on the 150k probe fit hidden ----
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    b1 = np.load(wr / "bert_hidden_fit_w512_t1_probe.npz", allow_pickle=True)
    nf = len(b1["stock_uid"])
    rows = {k: tc[k][:nf] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = b1["hidden"]
    fit_idx = final_fit_split(rows)
    print(f"[f42] T1 probe fit rows={len(fit_idx)}")
    t1_heads = []
    for s in (args.seed, args.seed + 1, args.seed + 2):
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                      lr=3e-4, epochs=16, seed=s)
        t1_heads.append(head)
        print(f"[f42] T1 head s{s} loss={hist['train_loss'][-1]:.4f}")

    # ---- eval subset (first 300k) ----
    cand = np.load(wr / "candidates_eval_K8.npz", allow_pickle=True)
    NE = 300000
    e1 = np.load(wr / "bert_hidden_eval_w512_t1_probe.npz", allow_pickle=True)["hidden"]
    e2 = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True,
                 mmap_mode="r")["hidden"][:NE]
    H1 = torch.from_numpy(e1.astype(np.float32))
    H2 = torch.from_numpy(e2.astype(np.float32))
    dates = np.asarray([str(d)[:10] for d in cand["date_key"]][:NE])
    rec = {"date_key": dates, "true_logret": cand["true_logret"][:NE].astype(np.float64),
           "quality": cand["quality"][:NE].astype(bool)}
    dense = max(5, int(0.8 * 750))

    # T1-ens
    r1 = []
    for h in t1_heads:
        h.eval()
        with torch.no_grad():
            r1.append(rank_pct_per_date(dates, h(H1).numpy().astype(np.float64)))
    ens1 = np.mean(r1, axis=0)
    # T2-ens (existing heads)
    r2 = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            r2.append(rank_pct_per_date(dates, h(H2).numpy().astype(np.float64)))
    ens2 = np.mean(r2, axis=0)
    ens12 = 0.5 * ens1 + 0.5 * ens2
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)[:NE]
    F = 0.5 * z_per_date(dates, ens2) + 0.5 * z_per_date(dates, p6)
    F12 = 0.5 * z_per_date(dates, ens12) + 0.5 * z_per_date(dates, p6)

    res = {"schema": "f42-t1t2-probe-v1", "seed": args.seed,
           "n_eval_subset": NE, "dense": dense}
    for name, s_ in (("ens2", ens2), ("ens12", ens12), ("F", F), ("F12", F12)):
        rec["s"] = s_
        res[name] = float(np.mean(list(daily_ic(rec, "s", dense).values())))
        print(f"[f42] {name}: {res[name]:.4f}")
    print(f"[f42] F12-F delta = {res['F12']-res['F']:+.4f}")

    out = rr / f"f42_t1t2_probe_seed{args.seed}.json"
    write_json_ledger(out, res, "f42_t1t2_probe", seed=args.seed)
    print(f"[f42] -> {out}")


if __name__ == "__main__":
    main()
