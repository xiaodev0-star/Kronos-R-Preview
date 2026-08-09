"""f25_seed_confirm.py — full-pipeline seed confirmation.

Builds the full pipeline (F = 0.5 z(ens) + 0.5 z(P6); |F| abstention) using an
ARBITRARY seed set of BERT heads, and reports the coverage curve.  Compares the
development ensemble (seeds 45-50) against the confirmation ensemble
(seeds 91-93) to verify pipeline seed robustness.
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

from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    DailyIcCache, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="45,46,47,48,49,50",
                    help="comma list of head seed numbers")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in seeds:
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            ranks.append(head(H).numpy().astype(np.float64))
    ens = np.mean([rank_pct_per_date(dates, r) for r in ranks], axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])

    res = {"schema": "f25-seed-confirm-v1", "seeds": seeds, "tag": args.tag,
           "coverage": {}}
    print(f"[f25] seeds={seeds}")
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, "c_mag", cv, True)
        ra = slice_rec(rec, acted)
        da20 = max(5, int(round(cv * dense)))
        ic_ = _daily_ic(ra, "F", da20)
        res["coverage"][str(cv)] = {"acted_ic": float(np.mean(list(ic_.values()))) if ic_ else None}
        print(f"[f25] cov={cv} ic={res['coverage'][str(cv)]['acted_ic'] and round(res['coverage'][str(cv)]['acted_ic'],4)}")

    out = rr / f"f25_seed_{args.tag}.json"
    write_json_ledger(out, res, "f25_seed_confirm", tag=args.tag)
    print(f"[f25] -> {out}")


if __name__ == "__main__":
    main()
