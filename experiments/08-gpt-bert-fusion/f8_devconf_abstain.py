"""f8_devconf_abstain.py — dev/confirm robustness of the c_dis abstention curve.

Checks that the ensemble-disagreement abstention gain on F_z_STRONG is not a
recent-window artifact: recompute coverage curves on dev (0..299) and confirm
(300..399) separately.
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
    DailyIcCache, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from f7_abstain_strong import _apply_rule, _daily_da  # noqa: E402


def _curve(rec, score_field, cf, max_high, dense):
    n = len(rec["stock_uid"])
    rows = []
    for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
        acted = np.ones(n, dtype=bool) if cv == 1.0 else _apply_rule(rec, cf, cv, max_high)
        rec_a = slice_rec(rec, acted)
        dense_act = max(5, int(round(cv * dense)))
        ic_ = DailyIcCache(rec_a, dense_act).series(rec_a[score_field])
        da_ = _daily_da(rec_a, score_field, dense_act)
        rows.append({"cov": cv,
                     "acted_rank_ic": float(np.mean(list(ic_.values()))) if ic_ else None,
                     "acted_da": float(np.mean(list(da_.values()))) if da_ else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_field", default="F_z_STRONG")
    ap.add_argument("--signal", default="c_dis")
    ap.add_argument("--max_high", action="store_true", default=False)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])

    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    hidden = de["hidden"]
    strong_specs = [f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt" for s in (45, 46, 47, 48)]
    ranks = []
    for fname in strong_specs:
        ck = torch.load(str(wr / fname), map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            s = head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, s))
    strong_ens = np.mean(ranks, axis=0)
    rec["F_z_STRONG"] = 0.5 * z_per_date(dates, strong_ens) \
        + 0.5 * z_per_date(dates, p6)
    rec[args.signal] = np.std(ranks, axis=0)

    s_ = np.load(wr / "scores_eval_K8_w512_stride1.npz", allow_pickle=True)
    logpb = np.asarray(s_["logp_bert_full"])
    if args.signal == "c_js":
        qg = np.load(wr / "gpt_q_eval_full128.npz", allow_pickle=True)["q"]
        pb = np.exp(np.clip(logpb, -50, 50)); pb = pb / pb.sum(1, keepdims=True)
        eps = 1e-12; m = 0.5 * (pb + qg)
        rec["c_js"] = 0.5 * (np.sum(pb * np.log((pb + eps) / (m + eps)), 1)
                             + np.sum(qg * np.log((qg + eps) / (m + eps)), 1))
    elif args.signal == "c_pabs":
        rec["c_pabs"] = np.abs(np.asarray(cand["p_up"], dtype=np.float64) - 0.5)

    off = np.asarray(rec["offset"], dtype=np.int64)
    dev = slice_rec(rec, (off >= 0) & (off <= 299))
    conf = slice_rec(rec, (off >= 300) & (off < 400))
    max_high = True if args.signal in ("c_pabs",) else args.max_high

    res = {"schema": "f8-devconf-abstain-v1", "score": args.score_field,
           "signal": args.signal, "max_high": max_high,
           "dev": _curve(dev, args.score_field, args.signal, max_high, dense),
           "confirm": _curve(conf, args.score_field, args.signal, max_high, dense),
           "full": _curve(rec, args.score_field, args.signal, max_high, dense)}
    print(f"\n=== DEV 0..299 ({args.signal}) ===")
    for r in res["dev"]:
        print(f"  cov={r['cov']:4.2f} acted_ic={r['acted_rank_ic']} acted_da={r['acted_da']}")
    print(f"=== CONFIRM 300..399 ===")
    for r in res["confirm"]:
        print(f"  cov={r['cov']:4.2f} acted_ic={r['acted_rank_ic']} acted_da={r['acted_da']}")

    out = rr / f"f8_devconf_{args.signal}.json"
    write_json_ledger(out, res, "f8_devconf_abstain", signal=args.signal)
    print(f"[f8] -> {out}")


if __name__ == "__main__":
    main()
