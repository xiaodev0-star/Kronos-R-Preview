"""f40_bert_cs_head.py — BERT-hidden rank head + cross-sectional features (TRAINED).

The output-ensemble FUSION (f36/f37) was shown to be eval-peeked (f38/f39:
calib-fit weight collapses to w=1.0).  This arm instead FEEDS cross-sectional
relative-strength features INTO the BERT-hidden rank head during FIT training,
so the head learns (from fit, not eval) how to use them.

Input: [BERT hidden (256) ∥ CS features (k)]  -> soft-Spearman rank head.
CS features (per-date rank-percentile, target-aligned):
  r1/r5/r20 momentum rank, 20d vol rank (low=better), 20d return rank.
Validated FIRST on the CALIB slice (audit_uids x 2023-02..2024-02, out-of-sample
for the heads) — the same discipline that caught the FUSION overfit.  Only if
the head improves RankIC on BOTH calib and eval vs the plain BERT ensemble is it
adopted.
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
BL = ROOT / "experiments" / "09-baselines"
for _p in (ROOT, SEVEN, EIGHT, BL, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_heads import final_fit_split, fold_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, _split_points,
    write_json_ledger,
)
from f0_scores import rank_pct_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(BL))
import data as bl_data


def cs_features(win_X, dates):
    """Cross-sectional relative-strength features (per-date rank-percentile)."""
    cols = []
    for lag in (1, 5, 20):
        if win_X.shape[1] >= lag:
            cols.append(rank_pct_per_date(dates, win_X[:, -lag]))
    for w in (5, 20):
        if win_X.shape[1] >= w:
            cols.append(rank_pct_per_date(dates, win_X[:, -w:].mean(axis=1)))
    if win_X.shape[1] >= 20:
        cols.append(rank_pct_per_date(dates, -win_X[:, -20:].std(axis=1)))
    cols.append(rank_pct_per_date(dates, win_X[:, -20:].sum(axis=1)))
    return np.stack(cols, axis=1).astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    # ---- fit: BERT hidden + CS features ----
    fits = bl_data.load_fit_sequences(window=32)
    b = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    fin = np.isfinite(fits["X"]).all(axis=1)
    fit_dates = np.asarray([str(d)[:10] for d in tc["date_key"]])
    CS = cs_features(fits["X"], fit_dates)
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = np.concatenate([b["hidden"], CS], axis=1)[fin]
    for k in rows:
        if isinstance(rows[k], np.ndarray) and rows[k].shape[0] != len(fin):
            pass
    rows["stock_uid"] = tc["stock_uid"][fin]
    rows["date_key"] = tc["date_key"][fin]
    rows["true_logret"] = tc["true_logret"][fin]
    print(f"[f40] fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")

    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64, dropout=0.1,
                       loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=3e-4, epochs=args.epochs, seed=args.seed)
    torch.save({"head_state": head.state_dict(), "seed": args.seed},
               wr / f"head_BERT_cs_rank_seed{args.seed}.pt")
    print(f"[f40] head trained (loss {hist['train_loss'][-1]:.4f})")

    # ---- evaluate on eval AND calib ----
    res = {"schema": "f40-bert-cs-head-v1", "seed": args.seed, "epochs": args.epochs}
    for region, hidden_f, seq_f in (
        ("eval", wr / "bert_hidden_eval_w512_t2.npz", bl_data.load_eval_sequences(32)),
        ("calib", wr / "bert_hidden_calib_w512_t2.npz", bl_data.build_sequences("calib", 32))):
        cand = np.load(cand_path(region), allow_pickle=True)
        de = np.load(hidden_f, allow_pickle=True)
        dates = np.asarray([str(d)[:10] for d in cand["date_key"]])
        CS_r = cs_features(seq_f["X"], dates)
        Xe = np.concatenate([de["hidden"], CS_r], axis=1)
        head.eval()
        with torch.no_grad():
            score = head(torch.from_numpy(Xe.astype(np.float32))).numpy().astype(np.float64)
        # plain BERT-ens reference on this region
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
        rec = {"date_key": dates, "true_logret": cand["true_logret"].astype(np.float64),
               "quality": cand["quality"].astype(bool)}
        dense = max(5, int(0.8 * int(np.max([len(np.where(dates == d)[0])
                                             for d in np.unique(dates)]))))
        ic_cs = float(np.mean(list(daily_ic({**rec, "s": score}, "s", dense).values())))
        ic_ens = float(np.mean(list(daily_ic({**rec, "s": ens}, "s", dense).values())))
        res[region] = {"CS_head": ic_cs, "BERT_ens": ic_ens, "delta": ic_cs - ic_ens,
                       "dense": dense}
        print(f"[f40] {region}: CS_head={ic_cs:.4f} BERT_ens={ic_ens:.4f} "
              f"delta={ic_cs-ic_ens:+.4f}")

    out = rr / f"f40_bert_cs_head_seed{args.seed}.json"
    write_json_ledger(out, res, "f40_bert_cs_head", seed=args.seed)
    print(f"[f40] -> {out}")


if __name__ == "__main__":
    main()
