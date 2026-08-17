"""f51_adaptive_dir.py — Round-2 Phase 3b: market-adaptive direction boundary.

f50 findings: calib-fitted q boundary does NOT transfer (calib DA rises with q
to 0.615 at q=0.6, eval DA of dir_q(0.6)=0.5178 < dir_iso 0.5181 — the known
calib<->eval drift).  Meanwhile f48's H_bert hit eval DA 0.5201 with a
ZERO-PARAMETER per-date boundary: sign flips at the rank where BERT_E's signed
values cross zero, i.e. at the date's q_d = P(BERT_E<0) quantile of F — a
market-adaptive boundary (down days -> high boundary -> fewer up calls).

Exp A — eval DA of the full q grid (drift diagnostic).
Exp B — dir_B: per-date boundary at q_d = P(BERT_E<0 | date):
          sign(F - quantile_qd(F, date)).  Zero parameters.
Exp C — combine dir_B with the I-family magnitudes (distance-from-boundary
        quantiles of |GPT sample|) -> full metric table + blocks + acted.
Exp D — block robustness of H_bert's DA (0.5201) vs dir_iso (0.5181).
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

from _exp07 import (  # noqa: E402
    weights_root, results_root, write_json_ledger,
)
from f48_micro_scan import (  # noqa: E402
    daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse,
    top_frac, block_masks,
)


def date_quantile_dir(F, dates, q):
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F))[0]
        if len(dm) < 5:
            continue
        out[dm] = np.sign(F[dm] - np.quantile(F[dm], q))
    return out


def date_boundary_qd(F, dates, src):
    """Per-date boundary VALUE at q_d = P(src<0 | date) (for magnitude
    alignment), plus the direction sign(F - boundary)."""
    bnd = np.full(len(F), np.nan)
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(src))[0]
        if len(dm) < 5:
            continue
        qd = float(np.mean(src[dm] < 0))
        b = float(np.quantile(F[dm], qd))
        bnd[dm] = b
        out[dm] = np.sign(F[dm] - b)
    return out, bnd


def distance_quantile_mag(dates, F, src_abs, boundary):
    """|mag| quantiles aligned by distance from the (per-date) boundary."""
    out = np.full(len(F), np.nan)
    for d in np.unique(dates):
        dm = np.where((dates == d) & np.isfinite(F) & np.isfinite(src_abs)
                      & np.isfinite(boundary))[0]
        if len(dm) < 2:
            continue
        key = np.abs(F[dm] - boundary[dm[0]])
        k_order = np.argsort(key, kind="stable")
        m_order = np.argsort(src_abs[dm], kind="stable")
        assigned = np.empty(len(dm))
        assigned[k_order] = src_abs[dm][m_order]
        out[dm] = assigned
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    f48 = np.load(wr / "f48_preds.npz", allow_pickle=True)
    F = f48["F"].astype(np.float64)
    dir_iso = f48["dir_iso"].astype(np.float64)
    BERT_E = f48["BERT_E"].astype(np.float64)
    pred_B = f48["pred_B"].astype(np.float64)
    pred_H = f48["pred_H"].astype(np.float64)
    H_bert = f48["H_bert"].astype(np.float64)
    true = f48["true_logret"].astype(np.float64)
    dates = np.asarray([str(d) for d in f48["date_key"]])
    quality = f48["quality"].astype(bool)
    dense = int(f48["dense_threshold"][0])
    n = len(F)
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    print(f"[f51] loaded f48_preds n={n}", flush=True)

    res = {"schema": "f51-adaptive-dir-v1", "dense_threshold": dense, "n": n,
           "exp_a": {}, "exp_b": {}, "exp_c": {}, "exp_d": {}}

    # ---- Exp A: full q grid on eval (drift diagnostic) ----
    rec_e = {"date_key": dates, "true_logret": true, "quality": quality}
    print("[f51] ExpA q-grid eval DA:")
    for q in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        rec_e["__s__"] = date_quantile_dir(F, dates, q)
        da = float(np.mean(list(daily_da(rec_e, "__s__", dense).values())))
        res["exp_a"][str(q)] = da
        print(f"  q={q:.2f} eval DA={da:.4f}")

    # ---- Exp B: market-adaptive boundary from BERT_E ----
    dir_B, bnd_B = date_boundary_qd(F, dates, BERT_E)
    rec_e["__s__"] = dir_B
    da_B = float(np.mean(list(daily_da(rec_e, "__s__", dense).values())))
    res["exp_b"] = {"dir_B_da": da_B}
    print(f"[f51] ExpB dir_B (qd=P(BERT_E<0)) eval DA={da_B:.4f}")

    # ---- Exp C: dir_B x I-family magnitudes ----
    F0 = 0.2574  # iso sign boundary (f49/f50)
    bnd_global = np.where(np.isfinite(dir_B), F * 0 + F0, np.nan)  # global
    mag = distance_quantile_mag(dates, F, np.abs(pred_B), bnd_global)
    pred_C = dir_B * mag
    st = daily_stats(dates, true, pred_C, quality, dense, centers)
    res["exp_c"]["dirB_x_igpt_mag"] = st
    print(f"[f51] ExpC dir_B x I-gpt mag: RankIC={st['rank_ic']:.4f} DA={st['da']:.4f} "
          f"MAPE={st['mape']:.4f} AR={st['amp_ratio']:.3f} Coll={st['collapse']:.3f}")
    # per-date q_d boundary for BOTH direction and magnitude (aligned)
    mag2 = distance_quantile_mag(dates, F, np.abs(pred_B), bnd_B)
    pred_C2 = dir_B * mag2
    st2 = daily_stats(dates, true, pred_C2, quality, dense, centers)
    res["exp_c"]["dirB_x_igpt_mag_perdate_bnd"] = st2
    print(f"[f51] ExpC (per-date bnd): RankIC={st2['rank_ic']:.4f} DA={st2['da']:.4f} "
          f"MAPE={st2['mape']:.4f} AR={st2['amp_ratio']:.3f} Coll={st2['collapse']:.3f}")

    # ---- Exp D: block DA of H_bert vs dir_iso (robustness of 0.5201) ----
    blk_masks = block_masks(dates, 4)
    for name, s in (("H_bert", H_bert), ("dir_iso", dir_iso), ("pred_C2", pred_C2)):
        blk_da, blk_ic = [], []
        for bm in blk_masks:
            rec_b = {"date_key": dates[bm], "true_logret": true[bm],
                     "quality": quality[bm], "__s__": s[bm]}
            blk_da.append(float(np.mean(list(daily_da(rec_b, "__s__", dense).values()))))
            blk_ic.append(mean_ic(rec_b, "__s__", dense))
        res["exp_d"][name] = {"blk_da": blk_da, "blk_ic": blk_ic}
        print(f"[f51] ExpD {name:9s} block DA={[round(b, 4) for b in blk_da]} "
              f"block IC={[round(b, 4) for b in blk_ic]}")

    # ---- acted top-20% for the champion candidates ----
    rec_all = {"date_key": dates, "true_logret": true, "quality": quality,
               "stock_uid": np.zeros(n, dtype=object),
               "F": F, "pred_C": pred_C, "pred_C2": pred_C2, "H_bert": H_bert}
    acted = top_frac(rec_all, "F", 0.2)
    ra = {k: v[acted] for k, v in rec_all.items()}
    da20 = max(5, int(round(0.2 * dense)))
    res["acted_top20"] = {k: mean_ic(ra, k, da20) for k in ("pred_C", "pred_C2", "H_bert")}
    print("[f51] acted top-20% RankIC:", {k: round(v, 4) for k, v in res["acted_top20"].items()})

    np.savez(wr / "f51_preds.npz",
             dir_B=dir_B, pred_C=pred_C, pred_C2=pred_C2,
             true_logret=true, date_key=dates, quality=quality,
             dense_threshold=np.array([dense]))

    out = rr / "f51_adaptive_dir.json"
    write_json_ledger(out, res, "f51_adaptive_dir")
    print(f"[f51] -> {out}")


if __name__ == "__main__":
    main()
