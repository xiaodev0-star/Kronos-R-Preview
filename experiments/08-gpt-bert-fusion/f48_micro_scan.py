"""f48_micro_scan.py — Round-2 (micro prediction) Phase 0+1.

Goal (18h round): improve MICRO prediction (per-stock magnitude) starting from
BERT + isotonic, under HARD GATES (RankIC>=0.075, AmpRatio in [0.8,1.2],
Collapse<0.3), optimising MAPE (down) and DA (up).  GPT backbone is frozen.

Phase 0 — the first BERT-centric measurement: micro stats of BERT_E (BERT
posterior mean) itself, vs GPT sample (f46 pred_B), isotonic (pred_A),
f46 hybrid (pred_H), and two cheap cand fields (greedy_return, post_median).

Phase 1 — magnitude-source scan under F ordering (per-date quantile remap,
f46 scheme): sources {GPT sample (reference), BERT_E, greedy_return,
post_median}.  Any monotone remap preserves RankIC exactly, so the different
iators are MAPE / DA / AmpRatio / Collapse.

Phase 2 preview — direction/magnitude decoupling: direction from isotonic
sign (calibrated boundary on calib), magnitude = per-date quantile of |src|.

Every metric is also reported per quarterly block (robustness discipline).
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

from _exp07 import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, _split_points,
    softmax_rows, decode_coarse, write_json_ledger, stage_weights,
    weights_artifact, posttrain_artifacts,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from _exp07 import MlpRankHead  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


# --------------------------------------------------------------------------
# metrics (project definitions; dense threshold per date)
# --------------------------------------------------------------------------
def daily_ic(rec, field, dense):
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


def daily_da(rec, field, dense):
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


def daily_mape(rec, field, dense):
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


def mean_ic(rec, field, dense):
    d = daily_ic(rec, field, dense)
    return float(np.mean(list(d.values()))) if d else None


def ampratio(pred, true):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    return float(np.mean(np.abs(p[m])) / max(np.mean(np.abs(t[m])), 1e-12))


def token_collapse(pred, centers):
    p = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(p)
    idx = np.argmin(np.abs(centers[None, :] - p[:, None]), axis=1)
    counts = np.bincount(idx[m], minlength=len(centers))
    return float(counts.max() / max(1, counts.sum()))


def top_frac(rec, cf, cv, max_high=True):
    """Select top/bottom cv rows per date by |confidence| (abs — the caller
    passes a signed score like F; |F| is the confidence, per Exp08)."""
    conf = np.abs(np.asarray(rec[cf], dtype=np.float64))
    dates = np.asarray(rec["date_key"])
    acted = np.zeros(len(rec["stock_uid"]), dtype=bool)
    for d in np.unique(dates):
        dm = (dates == d) & np.isfinite(conf)
        if dm.sum() == 0:
            continue
        keep = max(1, int(round(cv * dm.sum())))
        idx = np.where(dm)[0]
        order = np.argsort(-conf[idx]) if max_high else np.argsort(conf[idx])
        acted[idx[order[:keep]]] = True
    return acted


def block_masks(dates, k=4):
    """k equal-count blocks over unique sorted dates."""
    uniq = np.unique(dates)
    edges = np.array_split(uniq, k)
    masks = []
    for ed in edges:
        masks.append(np.isin(dates, ed))
    return masks


# --------------------------------------------------------------------------
def quantile_remap_signed(dates, F, src):
    """Per date: assign src values sorted ascending to F-sorted rows (f46)."""
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(src))[0]
        if len(dm) < 2:
            continue
        f_order = np.argsort(F[dm], kind="stable")
        s_order = np.argsort(src[dm], kind="stable")
        assigned = np.empty(len(dm))
        assigned[f_order] = src[dm][s_order]
        out[dm] = assigned
    return out


def quantile_remap_abs(dates, F, src_abs, dirn, F0=0.0):
    """Per date: |mag| from src_abs aligned by DISTANCE FROM THE SIGN BOUNDARY
    (smallest |mag| at the boundary, largest at the extremes), sign from dirn.

    This keeps the result strictly monotone in F: for F < F0 the value is
    -m(dist) (decreasing), for F > F0 it is +m(dist) (increasing), with
    m(.) increasing in |F - F0|.  (The naive F-aligned assignment puts the
    largest |mag| at the sign flip and inverts the positive side.)"""
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(src_abs)
                      & np.isfinite(dirn) & (dirn != 0))[0]
        if len(dm) < 2:
            continue
        key = np.abs(F[dm] - F0)                    # distance from boundary
        k_order = np.argsort(key, kind="stable")    # center -> extreme
        m_order = np.argsort(src_abs[dm], kind="stable")
        assigned = np.empty(len(dm))
        assigned[k_order] = src_abs[dm][m_order]    # smallest |mag| at center
        out[dm] = np.sign(dirn[dm]) * assigned
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wr = stage_weights("B")
    rr = results_root()
    sw = posttrain_artifacts()

    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    true = np.asarray(rec["true_logret"], dtype=np.float64)
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    print(f"[f48] n={n} dates={len(np.unique(dates))} dense={dense}", flush=True)

    # ---- F (ranking source, unchanged from Exp08) ----
    de = np.load(weights_artifact("hidden-eval"), allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in (42,):
        ck = torch.load(str(weights_artifact("bert-head", seed=s)),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
        print(f"[f48] ens seed{s} done", flush=True)
    ens = np.mean(ranks, axis=0)
    p6 = np.load(weights_artifact("p6-scores")).astype(np.float64)
    F = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["F"] = F

    # ---- BERT_E (BERT posterior mean, from cached coarse posterior) ----
    sb = np.load(weights_artifact("scores-eval"), allow_pickle=True)
    pb = softmax_rows(np.asarray(sb["logp_bert_full"]))
    dec = decode_coarse(pb, centers, cand["p_mean0"], cand["p_std0"], log_space=False)
    BERT_E = dec["e_mean"]
    rec["BERT_E"] = BERT_E
    print(f"[f48] BERT_E built: mean|.|={np.nanmean(np.abs(BERT_E)):.5f} "
          f"std={np.nanstd(BERT_E):.5f}", flush=True)

    # ---- cached baselines from f46 ----
    f46 = np.load(wr / "f46_preds.npz", allow_pickle=True)
    pred_A = f46["pred_A"].astype(np.float64)     # isotonic F->logret
    pred_B = f46["pred_B"].astype(np.float64)     # GPT joint-sample magnitudes
    pred_H = f46["pred_H"].astype(np.float64)     # f46 hybrid (GPT mag under F)
    greedy = cand["greedy_return"].astype(np.float64)
    pmed = cand["post_median"].astype(np.float64)
    rec.update({"pred_A": pred_A, "pred_B": pred_B, "pred_H": pred_H,
                "greedy": greedy, "pmed": pmed})

    # ---- calib isotonic (sign boundary) ----
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    de_c = np.load(weights_artifact("hidden-calib"), allow_pickle=True)
    Hc = torch.from_numpy(de_c["hidden"].astype(np.float32))
    ranks_c = []
    for s in (42,):
        ck = torch.load(str(weights_artifact("bert-head", seed=s)),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks_c.append(rank_pct_per_date(ccal["date_key"],
                                             h(Hc).numpy().astype(np.float64)))
    ens_c = np.mean(ranks_c, axis=0)
    gh_c = np.load(sw["calibration"], allow_pickle=True)["hidden"]
    ck6 = torch.load(str(sw["head_rank_mlp_spearman"]), map_location="cpu",
                     weights_only=False)
    h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
    h6.load_state_dict(ck6["head_state"]); h6.eval()
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(gh_c.astype(np.float32))).numpy().astype(np.float64)
    F_c = 0.5 * z_per_date(ccal["date_key"], ens_c) + 0.5 * z_per_date(ccal["date_key"], p6_c)
    y_c = ccal["true_logret"].astype(np.float64)
    mv = np.isfinite(F_c) & np.isfinite(y_c) & ccal["quality"].astype(bool)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(F_c[mv], y_c[mv])
    dir_iso = np.full(n, np.nan)
    finF = np.isfinite(F)
    dir_iso[finF] = np.sign(iso.predict(F[finF]))
    rec["dir_iso"] = dir_iso
    print(f"[f48] iso boundary: P(iso>0)={np.nanmean(dir_iso > 0):.3f} (F sign "
          f"{np.mean(np.sign(F) > 0):.3f})", flush=True)

    # ------------------------------------------------------------------
    # Phase 1: signed quantile remaps (f46 scheme) — magnitude source scan
    # ------------------------------------------------------------------
    rec["H_gpt"] = quantile_remap_signed(dates, F, pred_B)     # reference
    rec["H_bert"] = quantile_remap_signed(dates, F, BERT_E)    # BERT posterior mean
    rec["H_greedy"] = quantile_remap_signed(dates, F, greedy)  # GPT greedy decode
    rec["H_pmed"] = quantile_remap_signed(dates, F, pmed)      # GPT posterior median

    # ------------------------------------------------------------------
    # Phase 2 preview: direction/magnitude decoupling
    # ------------------------------------------------------------------
    rec["I_bert"] = quantile_remap_abs(dates, F, np.abs(BERT_E), dir_iso)
    rec["I_gpt"] = quantile_remap_abs(dates, F, np.abs(pred_B), dir_iso)
    rec["I_Fsign"] = quantile_remap_abs(dates, F, np.abs(BERT_E), np.sign(F))

    # ------------------------------------------------------------------
    # reports
    # ------------------------------------------------------------------
    res = {"schema": "f48-micro-scan-v1", "dense_threshold": dense, "n": n}
    fields = ["BERT_E", "pred_A", "pred_B", "pred_H", "greedy", "pmed",
              "H_gpt", "H_bert", "H_greedy", "H_pmed", "I_bert", "I_gpt", "I_Fsign"]
    blk_masks = block_masks(dates, 4)
    print("\n=== f48 metric table ===")
    for fld in fields:
        row = {"rank_ic": mean_ic(rec, fld, dense),
               "da": float(np.mean(list(daily_da(rec, fld, dense).values()))),
               "mape": float(np.mean(list(daily_mape(rec, fld, dense).values()))),
               "amp_ratio": ampratio(rec[fld], true),
               "collapse": token_collapse(rec[fld], centers),
               "mean_abs_pred": float(np.nanmean(np.abs(rec[fld]))),
               "std_pred": float(np.nanstd(rec[fld]))}
        blk = []
        for bm in blk_masks:
            sub = {k: np.asarray(rec[k])[bm] for k in ("date_key", "true_logret", "quality")}
            sub[fld] = np.asarray(rec[fld])[bm]
            blk.append({
                "rank_ic": mean_ic(sub, fld, dense),
                "da": float(np.mean(list(daily_da(sub, fld, dense).values()))),
                "mape": float(np.mean(list(daily_mape(sub, fld, dense).values()))),
                "amp_ratio": ampratio(sub[fld], sub["true_logret"]),
                "collapse": token_collapse(sub[fld], centers)})
        row["blocks"] = blk
        res[fld] = row
        print(f"  {fld:9s} RankIC={row['rank_ic']:.4f} DA={row['da']:.4f} "
              f"MAPE={row['mape']:.4f} AmpRatio={row['amp_ratio']:.3f} "
              f"Collapse={row['collapse']:.3f}")

    # ---- acted (top-20% |F|) RankIC of headline candidates ----
    acted = top_frac(rec, "F", 0.2)
    ra = {k: v[acted] for k, v in rec.items()}
    ra["stock_uid"] = rec["stock_uid"][acted]
    da20 = max(5, int(round(0.2 * dense)))
    res["acted_top20"] = {f: mean_ic(ra, f, da20)
                          for f in ("pred_H", "H_gpt", "H_bert", "I_bert", "I_gpt", "F")}
    print("  acted top-20% RankIC:", {k: round(v, 4) for k, v in res["acted_top20"].items()})

    # ---- save predictions for reuse ----
    np.savez(wr / "f48_preds.npz",
             BERT_E=BERT_E, pred_A=pred_A, pred_B=pred_B, pred_H=pred_H,
             H_gpt=rec["H_gpt"], H_bert=rec["H_bert"], H_greedy=rec["H_greedy"],
             H_pmed=rec["H_pmed"], I_bert=rec["I_bert"], I_gpt=rec["I_gpt"],
             I_Fsign=rec["I_Fsign"], F=F, dir_iso=dir_iso,
             true_logret=true, date_key=dates, quality=rec["quality"],
             dense_threshold=np.array([dense]))

    out = rr / "f48_micro_scan.json"
    write_json_ledger(out, res, "f48_micro_scan")
    print(f"[f48] -> {out}")


if __name__ == "__main__":
    main()
