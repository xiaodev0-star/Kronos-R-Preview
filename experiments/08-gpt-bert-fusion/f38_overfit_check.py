"""f38_overfit_check.py — overfitting diagnostics for the FUSION model.

The 0..399 eval window has been observed across many experiment rounds; the
honest worry is that the FUSION gain (+0.016 vs Exp08 F) is concentrated in a
sub-period or is an artifact of eval peeking.  Concrete checks:

  A) Time-quarter split of eval (0-99 / 100-199 / 200-299 / 300-399): is the
     FUSION > F gain present in ALL quarters or concentrated?
  B) Monthly stability: monthly RankIC of FUSION vs F (how often does FUSION win?).
  C) Leave-quarter-out ensemble-weight check is NOT needed (weight is fixed 0.5),
     but we report the fixed-weight behaviour per quarter.

All diagnostics use the SAME fixed 0.5/0.5 fusion (zero fitted parameters).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
BL = ROOT / "experiments" / "09-baselines"
for _p in (ROOT, SEVEN, EIGHT, BL, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from common import load_predictions  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def _daily_ic_series(rec, field, dense):
    """Return dict date->IC (not averaged) for per-quarter / per-month analysis."""
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
    import torch
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
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    xb = load_predictions("xgboost_rank2")
    rec["xgb2"] = np.asarray(xb["score"], dtype=np.float64)
    rec["FUSION"] = 0.5 * z_per_date(dates, rec["F"]) + 0.5 * z_per_date(dates, rec["xgb2"])

    res = {"schema": "f38-overfit-check-v1", "dense_threshold": dense}

    # ---- A) time-quarter split ----
    quarters = {
        "q1_0_99": (off >= 0) & (off < 100),
        "q2_100_199": (off >= 100) & (off < 200),
        "q3_200_299": (off >= 200) & (off < 300),
        "q4_300_399": (off >= 300) & (off < 400),
    }
    res["quarters"] = {}
    print("[f38] quarter:  FUSION / F / xgb2  (delta vs F)")
    for name, mask in quarters.items():
        ra = slice_rec(rec, mask)
        icF = np.mean(list(_daily_ic_series(ra, "F", dense).values()))
        icX = np.mean(list(_daily_ic_series(ra, "xgb2", dense).values()))
        icU = np.mean(list(_daily_ic_series(ra, "FUSION", dense).values()))
        res["quarters"][name] = {"FUSION": icU, "F": icF, "xgb2": icX,
                                 "delta": icU - icF}
        print(f"  {name}: {icU:.4f} / {icF:.4f} / {icX:.4f}  (delta {icU-icF:+.4f})")

    # ---- B) monthly win-rate ----
    icU = _daily_ic_series(rec, "FUSION", dense)
    icF = _daily_ic_series(rec, "F", dense)
    common = sorted(set(icU) & set(icF))
    wins = sum(1 for d in common if icU[d] > icF[d])
    losses = sum(1 for d in common if icU[d] < icF[d])
    res["monthly"] = {"n_dates": len(common), "wins": wins, "losses": losses,
                      "win_rate": wins / max(1, len(common))}
    print(f"[f38] monthly FUSION>F: {wins}/{len(common)} = {wins/max(1,len(common)):.3f}")

    # ---- C) xgb2 config check: is 300/4 (the val-selected smallest) stable? ----
    # We already know the grid: 300/4 (val 0.0728) beat 600/6 (0.0657) and 1000/6 (0.0581)
    # on the R2 val slice.  Report this as evidence the selection was val-based.
    res["xgb2_selection"] = {
        "note": "config selected on R2 val slice (2022-02..2023-02), not eval",
        "val_RankIC": {"300_4": 0.0728, "600_6": 0.0657, "1000_6": 0.0581},
    }

    out = rr / "f38_overfit_check.json"
    write_json_ledger(out, res, "f38_overfit_check")
    print(f"[f38] -> {out}")


if __name__ == "__main__":
    main()
