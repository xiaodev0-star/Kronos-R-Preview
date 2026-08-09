"""f22_coverage_bootstrap.py — paired moving-block bootstrap on the |F|-acted subset.

Protocol: candidate (F) vs reference (J3) must be compared on the SAME acted
subset.  For each coverage target, applies the |F| keep-rule, slices BOTH F and
J3 to the acted rows, and runs the paired daily-IC moving-block bootstrap.
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

from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    bootstrap_vs, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402


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


def main():
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            ranks.append(head(H).numpy().astype(np.float64))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])

    res = {"schema": "f22-coverage-bootstrap-v1", "dense_threshold": dense,
           "coverage": {}}
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(len(rec["stock_uid"]), dtype=bool) if cv == 1.0 else \
            _exact_frac(rec, "c_mag", cv, max_high=True)
        rec_a = slice_rec(rec, acted)
        dense_a = max(5, int(round(cv * dense)))
        bs = bootstrap_vs(rec_a, "F", rec_a, "post_median", dense_a)
        res["coverage"][str(cv)] = bs
        print(f"[f22] cov={cv} vs_J3 point={bs.get('point')} robust={bs.get('block_robust')} "
              f"n_dates={bs.get('n_dates')}")

    out = rr / "f22_coverage_bootstrap.json"
    write_json_ledger(out, res, "f22_coverage_bootstrap")
    print(f"[f22] -> {out}")


if __name__ == "__main__":
    main()
