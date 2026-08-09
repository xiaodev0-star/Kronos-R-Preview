"""f39_fusion_generalize.py — pre-cutoff generalization test of the FUSION.

Addresses the overfitting concern WITHOUT touching the sealed holdout and
WITHOUT fitting on the peeked 0..399 eval window:

  (1) Retrain the XGBoost rank2 model (val-selected 300/4 config), SAVE it.
  (2) Build F (=0.5 z(BERT6-ens)+0.5 z(P6)) and xgb2 predictions on three
      regions: R2-val (fit rows 2022-02..2023-02), calib
      (audit_uids 2023-02..2024-02), and eval (0..399).
  (3) Fit the FUSION weight w on CALIB only (maximize daily RankIC there).
  (4) Apply w to eval and report F / xgb2 / FUSION(w) on R2-val, calib, eval.

If FUSION(w) improves over F on calib AND eval with a calib-fit weight, the
+0.016 eval gain is more likely a real (decorrelated-signal) effect than pure
eval peeking.  If it does NOT improve on calib, that is honest evidence the
gain is partly eval-specific.
"""
from __future__ import annotations

import argparse
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
    weights_root, results_root, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(BL))
import common as bl_common
import data as bl_data


def daily_rank_ic(score, date_key, true, quality, dense):
    score = np.asarray(score, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(true) & np.asarray(quality).astype(bool)
    order, bounds, uniq = _split_points(date_key)
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


def mean_ic(score, date_key, true, quality, dense):
    d = daily_rank_ic(score, date_key, true, quality, dense)
    return float(np.mean(list(d.values()))) if d else None


def rank_pct(dates, s):
    return rank_pct_per_date(dates, s)


def build_rich_features(win_X, c1, dates):
    """Same feature builder as train_xgboost_rank2 (rich + cross-sectional)."""
    feats = []
    for lag in (1, 2, 3, 5, 10, 20):
        if win_X.shape[1] >= lag:
            feats.append(win_X[:, -lag])
    for w in (5, 10, 20):
        if win_X.shape[1] >= w:
            feats.append(win_X[:, -w:].mean(axis=1))
            feats.append(win_X[:, -w:].std(axis=1))
    feats.append(np.abs(win_X[:, -1]))
    feats.append(np.sign(win_X[:, -1]))
    X = np.stack(feats, axis=1)
    X = np.concatenate([X, c1], axis=1)
    for col_idx in (0, 1, 2):
        X = np.concatenate([X, rank_pct(dates, win_X[:, -(col_idx + 1)])[:, None]], axis=1)
    for w in (5, 20):
        if win_X.shape[1] >= w:
            X = np.concatenate([X, rank_pct(dates, win_X[:, -w:].mean(axis=1))[:, None]], axis=1)
    if win_X.shape[1] >= 20:
        X = np.concatenate([X, rank_pct(dates, -win_X[:, -20:].std(axis=1))[:, None]], axis=1)
    return X


def date_groups(dates):
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    sizes = np.diff(bounds).astype(int)
    return order, sizes


def bert_F_scores(wr, hidden_path, dates, p6_score):
    """BERT6-ens + P6 z-fusion (Exp08 F) for a hidden cache."""
    import torch
    de = np.load(hidden_path, allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    F = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6_score)
    return F, ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain xgb2 (else load)")
    ap.add_argument("--model", default=str(BL / "checkpoints" / "xgb_rank2.json"))
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    import torch
    import xgboost as xgb

    # ---- 1) features for fit (R2-val mask) ----
    fitf = bl_data.load_fit_features()
    fits = bl_data.load_fit_sequences(window=32)
    fin = (np.isfinite(fits["X"]).all(axis=1) & np.isfinite(fitf["X"]).all(axis=1)
           & np.isfinite(fitf["y"]))
    dates_all = np.asarray([str(d)[:10] for d in fitf["date_key"]])
    Xr = build_rich_features(fits["X"], fitf["X"], dates_all)[fin]
    y = fitf["y"][fin]
    dates = dates_all[fin]
    print(f"[f39] fit features {Xr.shape}")

    # ---- 2) train or load xgb2 ----
    if args.retrain or not Path(args.model).exists():
        order, sizes = date_groups(dates)
        Xo, yo = Xr[order], y[order]
        keep = sizes >= 2
        vg = np.concatenate([np.repeat(k, s) for k, s in zip(keep, sizes)])
        print("[f39] training xgb2 (300/4)...")
        model = xgb.XGBRanker(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
                              reg_lambda=1.0, objective="rank:pairwise",
                              n_jobs=-1, seed=42)
        model.fit(Xo[vg], yo[vg], group=sizes[keep])
        model.save_model(args.model)
        print(f"[f39] xgb2 saved -> {args.model}")
    else:
        model = xgb.XGBRanker()
        model.load_model(args.model)
        print("[f39] loaded xgb2")

    # ---- 3) apply to regions ----
    def apply_region(win, c1, dates_r):
        Xe = build_rich_features(win, c1, dates_r)
        return model.predict(Xe).astype(np.float64)

    # R2-val
    va = (dates >= "2022-02-01") & (dates < "2023-02-01")
    xgb_r2 = apply_region(fits["X"][fin][va], fitf["X"][fin][va], dates[va])
    y_r2 = y[va]
    # eval
    evf = bl_data.load_eval_features()
    evs = bl_data.load_eval_sequences(window=32)
    ev_dates = np.asarray([str(d)[:10] for d in bl_common.load_eval_rows()["date_key"]])
    xgb_ev = apply_region(evs["X"], evf["X"], ev_dates)
    ev_meta = bl_common.load_eval_rows()
    # calib
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    caf = np.load(sw / "calibration_cache.npz", allow_pickle=True)
    cas = bl_data.build_sequences("calib", 32, cache=True)
    ca_dates = np.asarray([str(d)[:10] for d in cas["date_key"]])
    if "c1_feats" in caf.files:
        ca_c1 = caf["c1_feats"]
    else:
        raise RuntimeError("calibration_cache has no c1_feats; build calib c1 separately")
    xgb_ca = apply_region(cas["X"], ca_c1, ca_dates)

    # ---- 4) F scores on regions ----
    # eval F (reuse the reference_fusion components? recompute)
    p6_ev = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    F_ev, ens_ev = bert_F_scores(wr, wr / "bert_hidden_eval_w512_t2.npz", ev_dates, p6_ev)
    # calib F
    gh_ca = np.load(sw / "calibration_cache.npz", allow_pickle=True)["hidden"]
    ck6 = torch.load(str(sw / "head_P6_mlp_rank_spearman.pt"),
                     map_location="cpu", weights_only=False)
    h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
    h6.load_state_dict(ck6["head_state"]); h6.eval()
    with torch.no_grad():
        p6_ca = h6(torch.from_numpy(gh_ca.astype(np.float32))).numpy().astype(np.float64)
    F_ca, _ = bert_F_scores(wr, wr / "bert_hidden_calib_w512_t2.npz", ca_dates, p6_ca)
    # R2-val F
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    gh_r2 = tc["hidden"][fin][va]
    with torch.no_grad():
        p6_r2 = h6(torch.from_numpy(gh_r2.astype(np.float32))).numpy().astype(np.float64)
    dates_r2 = dates[va]
    bh_r2 = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)["hidden"][fin][va]
    # BERT-ens for R2
    Hr = torch.from_numpy(bh_r2.astype(np.float32))
    ranks_r2 = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks_r2.append(rank_pct(dates_r2, h(Hr).numpy().astype(np.float64)))
    ens_r2 = np.mean(ranks_r2, axis=0)
    F_r2 = 0.5 * z_per_date(dates_r2, ens_r2) + 0.5 * z_per_date(dates_r2, p6_r2)

    # ---- 5) fit fusion weight w on CALIB ----
    ca_dense = max(5, int(0.8 * 574))
    zF_ca = z_per_date(ca_dates, F_ca)
    zX_ca = z_per_date(ca_dates, xgb_ca)
    ca_qual = np.isfinite(cas["y"])
    best = None
    grid = {}
    for w in np.arange(0.0, 1.001, 0.05):
        w = round(float(w), 2)
        fus = w * zF_ca + (1 - w) * zX_ca
        ic = mean_ic(fus, ca_dates, cas["y"], ca_qual, ca_dense)
        grid[w] = ic
        if best is None or ic > best[1]:
            best = (w, ic)
    w_fit = best[0]
    print(f"[f39] calib-fit w={w_fit} (calib RankIC={best[1]:.4f})")
    print(f"[f39] calib grid (sample): "
          + " ".join(f"{k}:{v:.4f}" for k, v in sorted(grid.items())[::5]))

    # ---- 6) apply to eval + report all regions ----
    ev_dense = ev_meta["dense_threshold"]
    ev_qual = ev_meta["quality"]
    ev_true = ev_meta["true_logret"]
    FUS_ev = w_fit * z_per_date(ev_dates, F_ev) + (1 - w_fit) * z_per_date(ev_dates, xgb_ev)
    res = {"schema": "f39-fusion-generalize-v1", "w_calib_fit": w_fit,
           "calib_grid": grid}
    regions = {
        "R2_val": (F_r2, xgb_r2, y_r2, np.isfinite(y_r2), max(5, int(0.8 * 400)), dates_r2),
        "calib": (F_ca, xgb_ca, cas["y"], ca_qual, ca_dense, ca_dates),
        "eval": (F_ev, xgb_ev, ev_true, ev_qual, ev_dense, ev_dates),
    }
    for name, (F_, X_, y_, q_, dn, dts) in regions.items():
        zF = z_per_date(dts, F_)
        zX = z_per_date(dts, X_)
        fus = w_fit * zF + (1 - w_fit) * zX
        res[name] = {
            "F": mean_ic(F_, dts, y_, q_, dn),
            "xgb2": mean_ic(X_, dts, y_, q_, dn),
            "FUSION(w_fit)": mean_ic(fus, dts, y_, q_, dn),
        }
        print(f"[f39] {name}: F={res[name]['F']:.4f} xgb2={res[name]['xgb2']:.4f} "
              f"FUSION={res[name]['FUSION(w_fit)']:.4f}")

    out = rr / "f39_fusion_generalize.json"
    write_json_ledger(out, res, "f39_fusion_generalize")
    print(f"[f39] -> {out}")


if __name__ == "__main__":
    main()
