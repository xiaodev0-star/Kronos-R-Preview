"""f26_rankft_compare.py — rank-aware BERT vs frozen T2, matched subsets.

Compares a fresh T5-recipe rank head on:
  (A) fine-tuned BERT hidden (rankft, fit 150k / eval 100k)
  (B) frozen T2 BERT hidden (same rows)
The relative eval RankIC on the same eval subset isolates the representation effect.
"""
from __future__ import annotations

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
from improve_common import weights_root, results_root, cand_path, _split_points, write_json_ledger  # noqa: E402
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
    wr = weights_root()
    rr = results_root()
    NE = 100000
    NF = 150000
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    meta = {k: tc[k][:NF] for k in ("stock_uid", "date_key", "true_logret")}
    cand = np.load(cand_path("eval"), allow_pickle=True)
    erec = {"date_key": cand["date_key"][:NE],
            "true_logret": cand["true_logret"][:NE].astype(np.float64),
            "quality": cand["quality"][:NE].astype(bool)}
    dense_probe = max(5, int(round(0.8 * 250)))

    res = {}
    for tag, fit_path, eval_path in (
        ("rankft", "bert_hidden_fit_w512_rankft.npz", "bert_hidden_eval_w512_rankft100k.npz"),
        ("T2", "bert_hidden_fit_w512_t2.npz", "bert_hidden_eval_w512_t2.npz")):
        rows = dict(meta)
        rows["hidden"] = np.load(wr / fit_path, allow_pickle=True)["hidden"]
        fit_idx = final_fit_split(rows)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                      lr=3e-4, epochs=16, seed=100)
        he = np.load(wr / eval_path, allow_pickle=True)["hidden"][:NE]
        head.eval()
        with torch.no_grad():
            sc = head(torch.from_numpy(he.astype(np.float32))).numpy().astype(np.float64)
        erec["score"] = sc
        ic = _daily_ic(erec, "score", dense_probe)
        res[tag] = {"full_ic": float(np.mean(list(ic.values()))), "n_dates": len(ic)}
        print(f"[f26] {tag}: full_ic={res[tag]['full_ic']:.4f} n_dates={len(ic)}")

    out = rr / "f26_rankft_compare.json"
    write_json_ledger(out, res, "f26_rankft_compare")
    print(f"[f26] -> {out}")


if __name__ == "__main__":
    main()
