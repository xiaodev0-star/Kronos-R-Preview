"""f34_direction.py — BERT-hidden direction head for the DA channel.

The pipeline's DA comes from the fused F sign / iso magnitude (~0.516-0.518).
Tests a dedicated BCE direction head on BERT hidden (and on the fused features)
for higher DA.  Also tests combining the direction head's probability with the
rank score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_heads import final_fit_split  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


class DirHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


def _daily_da(rec, field, dense):
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
        out[uniq[i]] = float(np.mean((np.sign(s[lo:hi][m]) > 0) == (t[lo:hi][m] > 0)))
    return out


def main():
    wr = weights_root()
    rr = results_root()
    d = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    rows = {"hidden": d["hidden"], "y": tc["true_logret"].astype(np.float32),
            "date_key": tc["date_key"], "stock_uid": tc["stock_uid"]}
    fit_idx = final_fit_split(rows)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = DirHead(dim=256, hidden=64, dropout=0.1).to(dev)
    H = torch.from_numpy(rows["hidden"][fit_idx].astype(np.float32)).to(dev)
    Y = torch.from_numpy((rows["y"][fit_idx] > 0).astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.0)
    rng = np.random.RandomState(130)
    for ep in range(12):
        head.train()
        perm = rng.permutation(len(fit_idx))
        tot = 0; ns = 0
        for s in range(0, len(fit_idx), 4096):
            ids = perm[s:s + 4096]
            if len(ids) < 2:
                continue
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(head(H[ids]), Y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += loss.item(); ns += 1
        if ep in (5, 11):
            print(f"[f34] ep{ep} loss={tot/max(1,ns):.4f}")
    torch.save({"head_state": head.state_dict()}, wr / "head_BERT_dir_seed130.pt")

    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    He = torch.from_numpy(de["hidden"].astype(np.float32)).to(dev)
    head.eval()
    with torch.no_grad():
        p_up = torch.sigmoid(head(He)).cpu().numpy().astype(np.float64)
    rec["dir_prob"] = p_up

    # F score for comparison
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval(); h = h.to(dev)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(He).cpu().numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["Fsign"] = np.sign(rec["F"])

    res = {"schema": "f34-direction-v1", "dense_threshold": dense}
    for name, field in (("Fsign", "Fsign"), ("dir_head", "dir_prob"), ("J4_pup", "p_up")):
        da = float(np.mean(list(_daily_da(rec, field, dense).values())))
        res[name] = {"da": da}
        print(f"[f34] {name:10s} da={da:.4f}")

    out = rr / "f34_direction.json"
    write_json_ledger(out, res, "f34_direction")
    print(f"[f34] -> {out}")


if __name__ == "__main__":
    main()
