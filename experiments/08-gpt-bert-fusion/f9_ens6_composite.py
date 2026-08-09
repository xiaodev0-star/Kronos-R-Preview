"""f9_ens6_composite.py — 6-head strong ensemble + composite confidence.

Uses all six 3e-4/16 heads (s45..s50).  Tests:
  - base: F_z_STRONG6 = 0.5 z(ens6) + 0.5 z(P6)
  - c_dis6 : std across 6 heads
  - c_comp : within-date rank-avg of (c_dis, c_pabs) — agreement AND conviction
Coverage curves on the base score.
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
    weights_root, results_root, metrics_table, build_rec, cand_path,
    slice_rec, DailyIcCache, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from f7_abstain_strong import _apply_rule, _daily_da  # noqa: E402


def main():
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])

    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["p6_score"] = p6
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    hidden = de["hidden"]

    heads = [f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt" for s in range(45, 51)]
    ranks = []
    for fname in heads:
        ck = torch.load(str(wr / fname), map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            s = head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, s))
    ens6 = np.mean(ranks, axis=0)
    rec["ens6"] = ens6
    rec["F_z_STRONG6"] = 0.5 * z_per_date(dates, ens6) + 0.5 * z_per_date(dates, p6)
    rec["F_rs_STRONG6"] = 0.5 * ens6 + 0.5 * rank_pct_per_date(dates, p6)
    rec["c_dis6"] = np.std(ranks, axis=0)

    # c_pabs
    rec["c_pabs"] = np.abs(np.asarray(cand["p_up"], dtype=np.float64) - 0.5)
    # composite: within-date rank of c_dis + rank of (1 - pabs-soft) ... both "keep low" style
    rd = rank_pct_per_date(dates, rec["c_dis6"])       # low = agree
    rp = rank_pct_per_date(dates, -rec["c_pabs"])      # low = high conviction
    rec["c_comp"] = 0.5 * rd + 0.5 * rp                # keep LOW composite

    fields = {"F_z_STRONG6": "F_z_STRONG6", "F_rs_STRONG6": "F_rs_STRONG6",
              "ens6": "ens6", "P6": "p6_score"}
    base = metrics_table(rec, fields, dense)
    for k, v in base.items():
        print(f"[f9] full {k:16s} rank_ic={v['avg_daily_rank_ic']:.4f} da={v['avg_da_per_date']:.4f}")

    res = {"schema": "f9-ens6-composite-v1", "dense_threshold": dense,
           "full": {k: v["avg_daily_rank_ic"] for k, v in base.items()},
           "coverage": {}}
    for score_field in ("F_z_STRONG6", "F_rs_STRONG6"):
        res["coverage"][score_field] = {}
        for cf, max_high in (("c_dis6", False), ("c_pabs", True), ("c_comp", False)):
            curve = []
            for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
                acted = np.ones(n, dtype=bool) if cv == 1.0 else \
                    _apply_rule(rec, cf, cv, max_high=max_high)
                rec_a = slice_rec(rec, acted)
                da_act = max(5, int(round(cv * dense)))
                ic_ = DailyIcCache(rec_a, da_act).series(rec_a[score_field])
                da_ = _daily_da(rec_a, score_field, da_act)
                curve.append({"cov": cv,
                              "acted_rank_ic": float(np.mean(list(ic_.values()))) if ic_ else None,
                              "acted_da": float(np.mean(list(da_.values()))) if da_ else None})
            res["coverage"][score_field][cf] = {"max_high": max_high, "curve": curve}
            print(f"[f9] {score_field:14s} {cf:8s}: "
                  + " ".join(f"cov={c['cov']:.2f} ic={c['acted_rank_ic'] and round(c['acted_rank_ic'],4)}" for c in curve))

    out = rr / "f9_ens6_composite.json"
    write_json_ledger(out, res, "f9_ens6_composite")
    print(f"[f9] -> {out}")


if __name__ == "__main__":
    main()
