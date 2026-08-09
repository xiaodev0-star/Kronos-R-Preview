"""f27_magnitude.py — magnitude-channel refinement (MAPE / DA).

Compares magnitude mappings fit on calib, applied to eval:
  M0  J3 posterior median (frozen baseline)
  M1  isotonic(F -> logret)                 (current pipeline)
  M2  isotonic(linear-combo [F, J3, E_BERT, P_up] -> logret)
  M3  linear regression on [F, J3, E_BERT, P_up] -> logret
Reports MAPE / DA / MAE.  All fit params use only the audit slice.
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
    weights_root, results_root, build_rec, cand_path, softmax_rows,
    decode_coarse, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def _daily_mape(rec, field, dense):
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
        out[uniq[i]] = float(np.mean(np.abs(np.exp(s[lo:hi][m] - t[lo:hi][m]) - 1.0)))
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


def _head_scores(wr, region, hidden_file, seeds):
    cand = np.load(cand_path(region), allow_pickle=True)
    dates = np.asarray(cand["date_key"])
    de = np.load(wr / hidden_file, allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in seeds:
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, head(H).numpy().astype(np.float64)))
    return cand, dates, np.mean(ranks, axis=0)


def main():
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    SEEDS = list(range(45, 51))

    # ---- eval features ----
    ce, de, ens_e = _head_scores(wr, "eval", "bert_hidden_eval_w512_t2.npz", SEEDS)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    F_e = 0.5 * z_per_date(de, ens_e) + 0.5 * z_per_date(de, p6)
    rec_e = {"date_key": ce["date_key"], "true_logret": ce["true_logret"].astype(np.float64),
             "quality": ce["quality"].astype(bool)}
    se = np.load(wr / "scores_eval_K8_w512_stride1.npz", allow_pickle=True)
    pb_e = softmax_rows(se["logp_bert_full"])
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    dec_e = decode_coarse(pb_e, centers, ce["p_mean0"], ce["p_std0"], log_space=False)
    rec_e["J3"] = ce["post_median"].astype(np.float64)
    rec_e["E_BERT"] = dec_e["e_median"]
    rec_e["F"] = F_e

    # ---- calib features ----
    cc, dc, ens_c = _head_scores(wr, "calib", "bert_hidden_calib_w512_t2.npz", SEEDS)
    gh = np.load(sw / "calibration_cache.npz", allow_pickle=True)
    ck6 = torch.load(str(sw / "head_P6_mlp_rank_spearman.pt"), map_location="cpu", weights_only=False)
    h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
    h6.load_state_dict(ck6["head_state"]); h6.eval()
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(gh["hidden"].astype(np.float32))).numpy().astype(np.float64)
    F_c = 0.5 * z_per_date(dc, ens_c) + 0.5 * z_per_date(dc, p6_c)
    sc = np.load(wr / "scores_calib_K8_w512_stride1.npz", allow_pickle=True)
    pb_c = softmax_rows(sc["logp_bert_full"])
    dec_c = decode_coarse(pb_c, centers, cc["p_mean0"], cc["p_std0"], log_space=False)
    y_c = cc["true_logret"].astype(np.float64)
    mval = np.isfinite(F_c) & np.isfinite(y_c) & cc["quality"].astype(bool)

    res = {"schema": "f27-magnitude-v1", "dense_threshold": int(ce["dense_threshold"][0]),
           "arms": {}}
    dense = int(ce["dense_threshold"][0])

    def eval_arm(name, pred_e):
        rec_e[name] = pred_e
        mp = np.mean(list(_daily_mape(rec_e, name, dense).values()))
        da = np.mean(list(_daily_da(rec_e, name, dense).values()))
        res["arms"][name] = {"mape": float(mp), "da": float(da)}
        print(f"[f27] {name:6s} mape={mp:.4f} da={da:.4f}")

    eval_arm("M0_J3", rec_e["J3"])
    # M1 isotonic F
    iso1 = IsotonicRegression(out_of_bounds="clip")
    iso1.fit(F_c[mval], y_c[mval])
    F_e_fin = np.where(np.isfinite(F_e), F_e, np.nanmedian(F_c))
    eval_arm("M1_isoF", iso1.predict(F_e_fin))

    # M2 isotonic on linear combo
    X_c = np.stack([F_c, cc["post_median"].astype(np.float64),
                    dec_c["e_median"], cc["p_up"].astype(np.float64)], axis=1)
    X_e = np.stack([F_e, rec_e["J3"], rec_e["E_BERT"],
                    ce["p_up"].astype(np.float64)], axis=1)
    X_e_fin = np.where(np.isfinite(X_e), X_e, 0.0)
    lin = LinearRegression()
    lin.fit(X_c[mval], y_c[mval])
    combo_c = lin.predict(X_c)
    combo_e = lin.predict(X_e_fin)
    iso2 = IsotonicRegression(out_of_bounds="clip")
    iso2.fit(combo_c[mval], y_c[mval])
    eval_arm("M2_isocombo", iso2.predict(combo_e))
    eval_arm("M3_lin", combo_e)

    out = rr / "f27_magnitude.json"
    write_json_ledger(out, res, "f27_magnitude")
    print(f"[f27] -> {out}")


if __name__ == "__main__":
    main()
