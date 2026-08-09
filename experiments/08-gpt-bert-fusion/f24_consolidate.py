"""f24_consolidate.py — definitive consolidation of the locked 08 pipeline.

Produces the final summary JSON + prints the headline table.  Runs the full
coverage curve (|F| abstention) with dev/confirm and the paired moving-block
bootstrap of F vs J3 on the full and 20%-coverage subsets.
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
    bootstrap_vs, DailyIcCache, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402


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
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    off = np.asarray(rec["offset"], dtype=np.int64)

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
    rec["J3"] = rec["post_median"]

    res = {"schema": "f24-consolidate-v1", "dense_threshold": dense,
           "coverage": {}}
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, "c_mag", cv, True)
        ra = slice_rec(rec, acted)
        da20 = max(5, int(round(cv * dense)))
        ic_ = _daily_ic(ra, "F", da20)
        da_ = _daily_da(ra, "F", da20)
        bs = bootstrap_vs(ra, "F", ra, "J3", da20)
        res["coverage"][str(cv)] = {
            "acted_frac": float(acted.mean()),
            "rank_ic": float(np.mean(list(ic_.values()))) if ic_ else None,
            "da": float(np.mean(list(da_.values()))) if da_ else None,
            "bootstrap_vs_J3": bs,
        }
        print(f"[f24] cov={cv} ic={res['coverage'][str(cv)]['rank_ic'] and round(res['coverage'][str(cv)]['rank_ic'],4)} "
              f"da={res['coverage'][str(cv)]['da'] and round(res['coverage'][str(cv)]['da'],4)} "
              f"vsJ3={bs.get('point') and round(bs['point'],4)} robust={bs.get('block_robust')}")

    out = rr / "f24_consolidate.json"
    write_json_ledger(out, res, "f24_consolidate")
    print(f"[f24] -> {out}")


if __name__ == "__main__":
    main()
