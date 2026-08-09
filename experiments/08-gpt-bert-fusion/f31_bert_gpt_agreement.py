"""f31_bert_gpt_agreement.py — BERT-posterior vs GPT-posterior direction agreement.

Genuinely-new consistency filter: keep rows where BERT's decoded posterior median
SIGN agrees with GPT's P(up) direction (both models' token-level posteriors point
the same way).  Tests this as an abstention signal on the fused F score, vs the
|F| baseline.  Pure CPU (cached logp_bert_full + candidates posterior stats).
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
    softmax_rows, decode_coarse, _split_points, write_json_ledger,
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
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])

    # BERT decoded median from logp_bert_full
    s_ = np.load(wr / "scores_eval_K8_w512_stride1.npz", allow_pickle=True)
    pb = softmax_rows(s_["logp_bert_full"])
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    dec = decode_coarse(pb, centers, cand["p_mean0"], cand["p_std0"], log_space=False)
    rec["bert_median_sign"] = np.sign(dec["e_median"])
    rec["gpt_pup_dir"] = (cand["p_up"].astype(np.float64) >= 0.5).astype(float)
    agree = (rec["bert_median_sign"] == rec["gpt_pup_dir"]).astype(bool)
    rec["c_agr"] = np.where(agree, 1.0, 0.0)

    res = {"schema": "f31-bert-gpt-agreement-v1", "dense_threshold": dense}
    print(f"[f31] direction-agreement frac: {agree.mean():.3f}")
    res["agree_frac"] = float(agree.mean())

    # agree-only as a filter
    for name, mask in (("all", np.ones(n, dtype=bool)), ("agree", agree), ("disagree", ~agree)):
        ra = slice_rec(rec, mask)
        ic_ = _daily_ic(ra, "F", dense)
        res[name] = {"ic": float(np.mean(list(ic_.values()))) if ic_ else None,
                     "n": int(mask.sum())}
        print(f"[f31] {name:10s} ic={res[name]['ic'] and round(res[name]['ic'],4)} n={mask.sum()}")

    # combine agree with |F| coverage
    res["combo"] = {}
    for cv in (0.8, 0.6, 0.4, 0.2):
        cmag = _exact_frac(rec, "c_mag", cv, True)
        mask = agree & cmag
        ra = slice_rec(rec, mask)
        da20 = max(5, int(round(cv * dense)))
        ic_ = _daily_ic(ra, "F", da20)
        res["combo"][str(cv)] = {"ic": float(np.mean(list(ic_.values()))) if ic_ else None,
                                 "frac": float(mask.mean())}
        print(f"[f31] agree & |F|top{cv}: ic={res['combo'][str(cv)]['ic'] and round(res['combo'][str(cv)]['ic'],4)} "
              f"frac={mask.mean():.3f}")

    out = rr / "f31_bert_gpt_agreement.json"
    write_json_ledger(out, res, "f31_bert_gpt_agreement")
    print(f"[f31] -> {out}")


if __name__ == "__main__":
    main()
