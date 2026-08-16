"""f83_final_report.py — definitive full report of the final champion (f82).

Rebuilds the f82 champion (med3 bulk + post_std tail 2%, lambda 0.25) from
cached assets and reports: full metrics + quarterly blocks + coverage + monthly
win rates vs f46 hybrid / isotonic.  Deterministic (no sampling at report
time — the median magnitudes are precomputed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    weights_root, results_root, cand_path, write_json_ledger,
)
from f48_micro_scan import (  # noqa: E402
    daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse,
    top_frac, block_masks,
)
from f51_adaptive_dir import date_boundary_qd, distance_quantile_mag  # noqa: E402


def daily_stats(dates, true, score, quality, dense, centers):
    rec = {"date_key": dates, "true_logret": true, "quality": quality}
    rec["__s__"] = score
    return {
        "rank_ic": mean_ic(rec, "__s__", dense),
        "da": float(np.mean(list(daily_da(rec, "__s__", dense).values()))),
        "mape": float(np.mean(list(daily_mape(rec, "__s__", dense).values()))),
        "amp_ratio": ampratio(score, true),
        "collapse": token_collapse(score, centers),
    }


def coverage_stats(dates, true, quality, F, pred, dense, centers, cv):
    rec = {"date_key": dates, "true_logret": true, "quality": quality,
           "stock_uid": np.zeros(len(F), dtype=object), "F": F}
    acted = top_frac(rec, "F", cv)
    ra = {"date_key": dates[acted], "true_logret": true[acted],
          "quality": quality[acted], "__s__": pred[acted]}
    dc = max(5, int(round(cv * dense)))
    return {
        "rank_ic": mean_ic(ra, "__s__", dc),
        "da": float(np.mean(list(daily_da(ra, "__s__", dc).values()))),
        "mape": float(np.mean(list(daily_mape(ra, "__s__", dc).values()))),
    }


def monthly_winrate(dates, true, quality, dense, cand_a, cand_b):
    months = np.array([str(d)[:7] for d in dates])
    wins = {"da": 0, "mape": 0}
    tot = {"da": 0, "mape": 0}
    dm = max(5, int(round(0.8 * dense)))
    for m in np.unique(months):
        mm = months == m
        vals = {}
        for name, s in (("a", cand_a[mm]), ("b", cand_b[mm])):
            ra = {"date_key": dates[mm], "true_logret": true[mm],
                  "quality": quality[mm], "__s__": s}
            da = daily_da(ra, "__s__", dm)
            mape = daily_mape(ra, "__s__", dm)
            if len(da) and len(mape):
                vals[name] = (float(np.mean(list(da.values()))),
                              float(np.mean(list(mape.values()))))
        if "a" not in vals or "b" not in vals:
            continue
        for i, k in enumerate(("da", "mape")):
            tot[k] += 1
            wins[k] += (vals["a"][i] > vals["b"][i]) if k == "da" \
                else (vals["a"][i] < vals["b"][i])
    return {k: (wins[k] / tot[k] if tot[k] else None) for k in ("da", "mape")}


def per_date_scale(dates, src, target_abs):
    out = np.full(len(src), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(src) & np.isfinite(target_abs))[0]
        if len(dm) < 2:
            continue
        s = np.nanmean(np.abs(target_abs[dm])) / max(np.nanmean(np.abs(src[dm])), 1e-12)
        out[dm] = src[dm] * s
    return out


def hybrid(dates, F, bulk, tail_src, boundary, tail):
    out = np.where(np.isfinite(bulk), bulk, np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(bulk)
                      & np.isfinite(tail_src) & np.isfinite(boundary))[0]
        if len(dm) < 4:
            continue
        key = np.abs(F[dm] - boundary[dm[0]])
        k_order = np.argsort(key, kind="stable")
        k = max(1, int(round(tail * len(dm))))
        for t in (k_order[:k], k_order[-k:]):
            out[dm[t]] = tail_src[dm[t]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    f48 = np.load(wr / "f48_preds.npz", allow_pickle=True)
    F = f48["F"].astype(np.float64)
    BERT_E = f48["BERT_E"].astype(np.float64)
    pred_B = f48["pred_B"].astype(np.float64)
    pred_H = f48["pred_H"].astype(np.float64)
    pred_A = f48["pred_A"].astype(np.float64)
    true = f48["true_logret"].astype(np.float64)
    dates = np.asarray([str(d) for d in f48["date_key"]])
    quality = f48["quality"].astype(bool)
    dense = int(f48["dense_threshold"][0])
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    F0 = 0.2574
    lam = 0.25
    tail = 0.02
    n = len(F)
    cand = np.load(cand_path("eval"), allow_pickle=True)

    med3 = np.load(wr / "f60_med_mag.npy")
    med3_s = per_date_scale(dates, med3, np.abs(pred_B))
    ps = cand["post_std"].astype(np.float64)
    ps_s = per_date_scale(dates, ps, np.abs(pred_B))
    dir_B, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    h = hybrid(dates, F, med3_s, ps_s, bnd_s, tail)
    mag = distance_quantile_mag(dates, F, h, bnd_s)
    champ = np.sign(F - bnd_s) * mag

    st = daily_stats(dates, true, champ, quality, dense, centers)
    st["blocks"] = [daily_stats(dates[bm], true[bm], champ[bm], quality[bm],
                                dense, centers) for bm in block_masks(dates, 4)]
    st["coverage"] = {str(cv): coverage_stats(dates, true, quality, F, champ,
                                              dense, centers, cv)
                      for cv in (0.2, 0.1, 0.05)}
    st["monthly_win_vs_f46"] = monthly_winrate(dates, true, quality, dense,
                                               champ, pred_H)
    st["monthly_win_vs_iso"] = monthly_winrate(dates, true, quality, dense,
                                               champ, pred_A)
    res = {"schema": "f83-final-champion-v1", "dense_threshold": dense, "n": n,
           "lambda": lam, "tail": tail, "champion": st}
    print(f"[f83] FINAL champion: RankIC={st['rank_ic']:.4f} DA={st['da']:.4f} "
          f"MAPE={st['mape']:.4f} AR={st['amp_ratio']:.3f} Coll={st['collapse']:.3f}")
    print("[f83] blocks DA/MAPE/Coll:",
          [(round(b["da"], 4), round(b["mape"], 4), round(b["collapse"], 3))
           for b in st["blocks"]])
    print("[f83] coverage:",
          {k: (round(v["rank_ic"], 4), round(v["da"], 4), round(v["mape"], 4))
           for k, v in st["coverage"].items()})
    print(f"[f83] monthly win vs f46: {st['monthly_win_vs_f46']}")
    print(f"[f83] monthly win vs iso: {st['monthly_win_vs_iso']}")

    np.savez(wr / "f83_preds.npz",
             champ=champ, dir_s=np.sign(F - bnd_s), mag=mag, bnd_s=bnd_s,
             true_logret=true, date_key=dates, quality=quality,
             dense_threshold=np.array([dense]))

    out = rr / "f83_final_champion.json"
    write_json_ledger(out, res, "f83_final_champion")
    print(f"[f83] -> {out}")


if __name__ == "__main__":
    main()
