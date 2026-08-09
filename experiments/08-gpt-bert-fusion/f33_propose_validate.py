"""f33_propose_validate.py — GPT-propose / BERT-validate re-ranking.

The user's framing: GPT proposes candidates, BERT validates.  Tests a two-stage
structure:
  Stage 1: per date, GPT (P6) proposes the top-N stocks.
  Stage 2: within the proposed top-N, re-rank by the BERT-head ensemble score.
Compares the selective RankIC of the re-ranked top-N vs the |F|-selected top-N
(the current pipeline) at matched coverage.  Also compares the overall F score
to a propose-validate-combined score.
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
    _split_points, write_json_ledger,
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


def _select(rec, field, frac, top=True):
    """Per-date top/bottom frac by a field; returns a rec."""
    score = np.asarray(rec[field], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    acted = np.zeros(len(rec["stock_uid"]), dtype=bool)
    for d in np.unique(dates):
        dm = dates == d
        cvals = score[dm]
        m = np.isfinite(cvals)
        if m.sum() == 0:
            continue
        keep_n = max(1, int(round(frac * m.sum())))
        ord_ = np.argsort(-cvals) if top else np.argsort(cvals)
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

    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["p6"] = p6
    rec["ens"] = ens
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])
    rec["rank_ens"] = ens
    rec["rank_p6"] = rank_pct_per_date(dates, p6)

    res = {"schema": "f33-propose-validate-v1", "dense_threshold": dense, "arms": {}}
    for frac in (0.2, 0.1):
        # propose: top frac by P6; validate: re-rank within by BERT ens
        prop = _select(rec, "p6", frac, top=True)
        rec_a = slice_rec(rec, prop)
        ic_prop_bert = float(np.mean(list(_daily_ic(rec_a, "ens", max(5, int(frac * dense))).values())))
        ic_prop_fused = float(np.mean(list(_daily_ic(rec_a, "F", max(5, int(frac * dense))).values())))
        # |F| top-frac (current pipeline)
        sel_f = _select(rec, "c_mag", frac, top=True)
        rec_f = slice_rec(rec, sel_f)
        ic_f = float(np.mean(list(_daily_ic(rec_f, "F", max(5, int(frac * dense))).values())))
        res["arms"][f"top{int(frac*100)}"] = {
            "propose_p6_rerank_bert": ic_prop_bert,
            "propose_p6_fused": ic_prop_fused,
            "F_mag_select": ic_f,
        }
        print(f"[f33] top{int(frac*100)}: rerank_bert={ic_prop_bert:.4f} fused_in_prop={ic_prop_fused:.4f} "
              f"F_mag_select={ic_f:.4f}")

    out = rr / "f33_propose_validate.json"
    write_json_ledger(out, res, "f33_propose_validate")
    print(f"[f33] -> {out}")


if __name__ == "__main__":
    main()
