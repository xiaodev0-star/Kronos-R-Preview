"""f20_t1_t2_compare.py — compare T1-w512 vs T2 BERT hidden for the rank head.

Trains identical T5-style heads on the SAME fit rows (first 150k) using T1
hidden vs T2 hidden, evaluates on the SAME eval rows (first 300k).  The
relative RankIC difference isolates the BERT-checkpoint effect on ranking.
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
    weights_root, results_root, build_rec, cand_path, _split_points,
    write_json_ledger,
)


def _daily_ic(rec, field, dense):
    from scipy.stats import spearmanr
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
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"

    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    fit_meta = {k: tc[k][:150000] for k in ("stock_uid", "date_key", "true_logret")}
    f1 = np.load(wr / "bert_hidden_fit_w512_t1_probe.npz", allow_pickle=True)
    f2 = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    assert len(f1["stock_uid"]) == 150000 and len(f2["stock_uid"]) >= 150000

    res = {}
    for tag, hid in (("T1", f1["hidden"]), ("T2", f2["hidden"][:150000])):
        rows = dict(fit_meta)
        rows["hidden"] = hid
        fit_idx = final_fit_split(rows)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                      lr=3e-4, epochs=16, seed=70)
        # eval on same eval subset
        cand = np.load(cand_path("eval"), allow_pickle=True)
        e1 = np.load(wr / "bert_hidden_eval_w512_t1_probe.npz", allow_pickle=True)
        e2 = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
        H_eval = e1["hidden"] if tag == "T1" else e2["hidden"][:300000]
        n_eval = len(e1["stock_uid"])
        rec = {"date_key": cand["date_key"][:n_eval],
               "true_logret": cand["true_logret"][:n_eval].astype(np.float64),
               "quality": cand["quality"][:n_eval].astype(bool)}
        head.eval()
        with torch.no_grad():
            sc = head(torch.from_numpy(H_eval.astype(np.float32))).numpy().astype(np.float64)
        rec["score"] = sc
        # eval dense: ~750 stocks/date
        dense_probe = max(5, int(round(0.8 * 750)))
        ic = _daily_ic(rec, "score", dense_probe)
        full_ic = float(np.mean(list(ic.values())))
        dev_ic = float(np.mean([v for k, v in ic.items() if k < "2025-01-01"])) if ic else None
        conf_ic = float(np.mean([v for k, v in ic.items() if k >= "2025-01-01"])) if ic else None
        res[tag] = {"full_ic": full_ic, "dev_ic": dev_ic, "conf_ic": conf_ic,
                    "n_dates": len(ic), "train_loss": hist["train_loss"][-1]}
        print(f"[f20] {tag}: full_ic={full_ic:.4f} dev_ic={dev_ic} conf_ic={conf_ic} "
              f"n_dates={len(ic)}")

    out = rr / "f20_t1_t2_compare.json"
    write_json_ledger(out, res, "f20_t1_t2_compare")
    print(f"[f20] -> {out}")


if __name__ == "__main__":
    main()
