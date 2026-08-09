"""f41_t1_t2_ensemble.py — T1 + T2 BERT representation ensemble (calib-validated).

The 6 existing heads read the T2 BERT hidden (GPT-noise-denoised).  A SECOND
BERT checkpoint (T1, pre-denoising) gives a decorrelated representation.  This
arm trains 6 fresh heads on the T1 hidden and builds a 12-head representation
ensemble, fused with P6 exactly like Exp08:

  ens12 = 0.5·rank-ens(T1 heads) + 0.5·rank-ens(T2 heads)
  F12   = 0.5·z(ens12) + 0.5·z(P6)

DISCIPLINE: validated on the CALIB slice FIRST (audit_uids x 2023-02..2024-02,
out-of-sample for all heads).  Only if F12 improves over F on BOTH calib and
eval is it adopted (this is what caught the FUSION overfit in f38/f39).
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

from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, _split_points,
    write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


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


def mean_ic(rec, field, dense):
    d = daily_ic(rec, field, dense)
    return float(np.mean(list(d.values()))) if d else None


def region_meta(cand):
    return {"date_key": np.asarray([str(d)[:10] for d in cand["date_key"]]),
            "true_logret": cand["true_logret"].astype(np.float64),
            "quality": cand["quality"].astype(bool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="60,61,62,63,64,65")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    wr = weights_root()
    rr = results_root()

    # ---- 1) train T1 heads (if not skipped) ----
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    b1 = np.load(wr / "bert_hidden_fit_w512_t1_full.npz", allow_pickle=True)
    assert np.array_equal(tc["stock_uid"], b1["stock_uid"])
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = b1["hidden"]
    fit_idx = final_fit_split(rows)
    print(f"[f41] T1 fit rows={len(fit_idx)}")

    if not args.skip_train:
        for s in seeds:
            head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
            _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                          lr=3e-4, epochs=16, seed=s)
            torch.save({"head_state": head.state_dict(), "seed": s,
                        "bert": "t1"},
                       wr / f"head_T1_mlp_rank_spearman_seed{s}_3e-4ep16.pt")
            print(f"[f41] T1 head s{s} trained (loss {hist['train_loss'][-1]:.4f})")

    # ---- 2) evaluate on calib FIRST, then eval ----
    p6_ev = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    res = {"schema": "f41-t1-t2-ensemble-v1", "seeds": seeds}
    for region, hf_t2, hf_t1, p6 in (
        ("calib", wr / "bert_hidden_calib_w512_t2.npz",
         wr / "bert_hidden_calib_w512_t1_full.npz", None),
        ("eval", wr / "bert_hidden_eval_w512_t2.npz",
         wr / "bert_hidden_eval_w512_t1_full.npz", p6_ev)):
        cand = np.load(cand_path(region), allow_pickle=True)
        meta = region_meta(cand)
        dense = int(cand["dense_threshold"][0]) if "dense_threshold" in cand.files \
            else max(5, int(0.8 * int(np.max([len(np.where(meta["date_key"] == d)[0])
                                              for d in np.unique(meta["date_key"])]))))
        d2 = np.load(hf_t2, allow_pickle=True)
        d1 = np.load(hf_t1, allow_pickle=True)
        H2 = torch.from_numpy(d2["hidden"].astype(np.float32))
        H1 = torch.from_numpy(d1["hidden"].astype(np.float32))
        # T2 heads (existing)
        r2 = []
        for s in range(45, 51):
            ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                            map_location="cpu", weights_only=False)
            h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
            h.load_state_dict(ck["head_state"]); h.eval()
            with torch.no_grad():
                r2.append(rank_pct_per_date(meta["date_key"], h(H2).numpy().astype(np.float64)))
        # T1 heads (new)
        r1 = []
        for s in seeds:
            ck = torch.load(str(wr / f"head_T1_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                            map_location="cpu", weights_only=False)
            h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
            h.load_state_dict(ck["head_state"]); h.eval()
            with torch.no_grad():
                r1.append(rank_pct_per_date(meta["date_key"], h(H1).numpy().astype(np.float64)))
        ens2 = np.mean(r2, axis=0)
        ens1 = np.mean(r1, axis=0)
        ens12 = 0.5 * ens2 + 0.5 * ens1
        # F12 and F (Exp08) on this region
        if region == "eval":
            p6_r = p6
            F = 0.5 * z_per_date(meta["date_key"], ens2) + 0.5 * z_per_date(meta["date_key"], p6_r)
            F12 = 0.5 * z_per_date(meta["date_key"], ens12) + 0.5 * z_per_date(meta["date_key"], p6_r)
        else:
            # P6 on calib
            gh = np.load(sw / "calibration_cache.npz", allow_pickle=True)["hidden"]
            ck6 = torch.load(str(sw / "head_P6_mlp_rank_spearman.pt"),
                             map_location="cpu", weights_only=False)
            h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
            h6.load_state_dict(ck6["head_state"]); h6.eval()
            with torch.no_grad():
                p6_r = h6(torch.from_numpy(gh.astype(np.float32))).numpy().astype(np.float64)
            F = 0.5 * z_per_date(meta["date_key"], ens2) + 0.5 * z_per_date(meta["date_key"], p6_r)
            F12 = 0.5 * z_per_date(meta["date_key"], ens12) + 0.5 * z_per_date(meta["date_key"], p6_r)
        rec = dict(meta)
        rec["F"] = F; rec["F12"] = F12; rec["ens12"] = ens12; rec["ens2"] = ens2
        res[region] = {
            "F": mean_ic(rec, "F", dense),
            "F12": mean_ic(rec, "F12", dense),
            "ens12": mean_ic(rec, "ens12", dense),
            "ens2": mean_ic(rec, "ens2", dense),
            "delta_F12_vs_F": mean_ic(rec, "F12", dense) - mean_ic(rec, "F", dense),
            "dense": dense,
        }
        print(f"[f41] {region}: F={res[region]['F']:.4f} F12={res[region]['F12']:.4f} "
              f"(ens2 {res[region]['ens2']:.4f} ens12 {res[region]['ens12']:.4f}) "
              f"delta={res[region]['delta_F12_vs_F']:+.4f}")

    out = rr / "f41_t1_t2_ensemble.json"
    write_json_ledger(out, res, "f41_t1_t2_ensemble")
    print(f"[f41] -> {out}")


if __name__ == "__main__":
    main()
