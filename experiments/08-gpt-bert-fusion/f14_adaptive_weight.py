"""f14_adaptive_weight.py — per-row adaptive GPT/BERT fusion weight.

Instead of a fixed 0.5/0.5 blend, weight = smooth function of BERT ensemble
disagreement: trust BERT more where the BERT heads agree (low c_dis), trust GPT
(P6) more where BERT disagrees.  Zero fitted parameters on eval — the weight
function is a pre-registered monotone map from c_dis to w in [0.6, 0.2].

score_i = w_i * z(BERT_ens) + (1 - w_i) * z(P6)
w_i = 0.6 - 0.4 * clip(c_dis_i / tau, 0, 1)      # tau = calib median c_dis

tau is fit on the calibration slice (median c_dis there).  Reports full-window
and coverage curves vs the fixed-blend baseline.
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


def _load_scores(wr, region, hidden_file):
    cand = np.load(cand_path(region), allow_pickle=True)
    dates = np.asarray(cand["date_key"])
    n = len(cand["stock_uid"])
    de = np.load(wr / hidden_file, allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        head.load_state_dict(ck["head_state"]); head.eval()
        with torch.no_grad():
            sc = head(H).numpy().astype(np.float64)
        ranks.append(rank_pct_per_date(dates, sc))
    ens = np.mean(ranks, axis=0)
    dis = np.std(ranks, axis=0)
    if region == "eval":
        p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    else:
        gh = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                     / "calibration_cache.npz", allow_pickle=True)
        ck6 = torch.load(str(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                             / "head_P6_mlp_rank_spearman.pt"),
                         map_location="cpu", weights_only=False)
        h6 = MlpRankHead(dim=256, hidden=64, dropout=0.0, loss="soft_spearman")
        h6.load_state_dict(ck6["head_state"]); h6.eval()
        with torch.no_grad():
            p6 = h6(torch.from_numpy(gh["hidden"].astype(np.float32))).numpy().astype(np.float64)
    out = {"date_key": dates, "true_logret": cand["true_logret"].astype(np.float64),
           "quality": cand["quality"].astype(bool), "ens": ens, "c_dis": dis,
           "p6": p6}
    return out, dates


def _exact_frac(rec, cf, cv, max_high=True):
    conf = np.asarray(rec[cf], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    acted = np.zeros(len(rec["stock_uid"]), dtype=bool)
    for d in np.unique(dates):
        dm = dates == d
        cvals = conf[dm]
        m = np.isfinite(cvals)
        if m.sum() == 0:
            continue
        keep_n = max(1, int(round(cv * m.sum())))
        ord_ = np.argsort(-cvals) if max_high else np.argsort(cvals)
        idx = np.where(dm)[0]
        acted[idx[ord_[:keep_n]]] = True
    return acted


def main():
    wr = weights_root()
    rr = results_root()
    ce, dates_e = _load_scores(wr, "eval", "bert_hidden_eval_w512_t2.npz")
    cc, dates_c = _load_scores(wr, "calib", "bert_hidden_calib_w512_t2.npz")
    cand = np.load(cand_path("eval"), allow_pickle=True)
    dense = int(cand["dense_threshold"][0])
    n = len(ce["true_logret"])

    rec = {"date_key": ce["date_key"], "true_logret": ce["true_logret"],
           "quality": ce["quality"]}
    rec["ens"] = ce["ens"]
    rec["c_dis"] = ce["c_dis"]
    rec["z_ens"] = z_per_date(dates_e, ce["ens"])
    rec["z_p6"] = z_per_date(dates_e, ce["p6"])
    rec["F_fixed"] = 0.5 * rec["z_ens"] + 0.5 * rec["z_p6"]

    # tau from calib median c_dis
    tau = float(np.nanmedian(cc["c_dis"]))
    w = 0.6 - 0.4 * np.clip(rec["c_dis"] / max(tau, 1e-9), 0, 1)
    rec["F_adapt"] = w * rec["z_ens"] + (1 - w) * rec["z_p6"]
    print(f"[f14] calib tau={tau:.4f}")

    fields = {"F_fixed": "F_fixed", "F_adapt": "F_adapt", "ens": "ens"}
    full = metrics_table(rec, fields, dense)
    for k, v in full.items():
        print(f"[f14] full {k:9s} rank_ic={v['avg_daily_rank_ic']:.4f} "
              f"da={v['avg_da_per_date']:.4f}")

    res = {"schema": "f14-adaptive-weight-v1", "tau": tau,
           "full": {k: v["avg_daily_rank_ic"] for k, v in full.items()}}
    # coverage with c_dis on both
    for sf in ("F_fixed", "F_adapt"):
        row = []
        for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
            acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, "c_dis", cv, False)
            ra = slice_rec(rec, acted)
            da20 = max(5, int(round(cv * dense)))
            ic_ = DailyIcCache(ra, da20).series(ra[sf])
            row.append({"cov": cv, "acted_ic": float(np.mean(list(ic_.values()))) if ic_ else None})
            print(f"[f14] {sf} cov={cv} acted_ic={row[-1]['acted_ic'] and round(row[-1]['acted_ic'],4)}")
        res[f"coverage_{sf}"] = row

    out = rr / "f14_adaptive_weight.json"
    write_json_ledger(out, res, "f14_adaptive_weight")
    print(f"[f14] -> {out}")


if __name__ == "__main__":
    main()
