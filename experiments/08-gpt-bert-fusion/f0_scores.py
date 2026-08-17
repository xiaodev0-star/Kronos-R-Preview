"""f0_scores.py — shared score computation for the 08 fusion round.

Computes and persists the base + fused row scores on a region, so downstream
scripts (abstention, blend fitting) share one source of truth.

Scores (all saved to <weights_root>/scores_{region}_fused.npz):
  bert_head   T5 BERT-hidden rank head score
  p6_score    P6 GPT-hidden rank head score (eval: cached npy; calib: apply head)
  rank_bert / rank_p6 / z_bert / z_p6   within-date rank-percentile / z-score
  F_ranksum   (rank_bert + rank_p6)/2
  F_zsum      (z_bert + z_p6)/2
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
    cand_path, stage_weights, weights_artifact, posttrain_artifacts,
)
from _exp07 import MlpRankHead  # noqa: E402


def apply_head(head_path, hidden, dim=256, hidden_dim=64, dropout=0.1):
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=dim, hidden=hidden_dim, dropout=dropout,
                       loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(hidden.astype(np.float32))).numpy().astype(np.float64)


def rank_pct_per_date(dates, score):
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    out = np.full(len(score), np.nan)
    s = score[order]
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blk = s[lo:hi]
        m = np.isfinite(blk)
        if m.sum() == 0:
            continue
        r = np.argsort(np.argsort(blk[m], kind="stable")).astype(float)
        r = r / max(1, len(r) - 1)
        tmp = np.full(blk.shape, np.nan)
        tmp[m] = r
        out[order[lo:hi]] = tmp
    return out


def z_per_date(dates, score):
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    out = np.full(len(score), np.nan)
    s = score[order]
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blk = s[lo:hi]
        m = np.isfinite(blk)
        if m.sum() < 2:
            continue
        mu, sdv = blk[m].mean(), blk[m].std()
        if sdv == 0:
            continue
        tmp = np.full(blk.shape, np.nan)
        tmp[m] = (blk[m] - mu) / sdv
        out[order[lo:hi]] = tmp
    return out


def compute_region(region, wr=None, six_artifacts=None):
    wr = Path(wr) if wr is not None else stage_weights("B")
    six_artifacts = six_artifacts or posttrain_artifacts()
    cand = np.load(cand_path(region), allow_pickle=True)
    dates = np.asarray(cand["date_key"])
    n = len(cand["stock_uid"])

    out = {}
    # ---- BERT hidden -> T5 head score ----
    hidden_key = "hidden-eval" if region == "eval" else "hidden-calib"
    bh = np.load(weights_artifact(hidden_key), allow_pickle=True)
    assert len(bh["stock_uid"]) == n
    out["bert_head"] = apply_head(weights_artifact("bert-head"),
                                  bh["hidden"])
    # ---- GPT hidden -> P6 score ----
    if region == "eval":
        p6 = np.load(weights_artifact("p6-scores"))
        out["p6_score"] = p6.astype(np.float64)
    else:
        gh = np.load(six_artifacts["calibration"], allow_pickle=True)
        assert len(gh["stock_uid"]) == n
        out["p6_score"] = apply_head(six_artifacts["head_rank_mlp_spearman"],
                                     gh["hidden"])

    # ---- fused arms ----
    out["rank_bert"] = rank_pct_per_date(dates, out["bert_head"])
    out["rank_p6"] = rank_pct_per_date(dates, out["p6_score"])
    out["z_bert"] = z_per_date(dates, out["bert_head"])
    out["z_p6"] = z_per_date(dates, out["p6_score"])
    out["F_ranksum"] = 0.5 * out["rank_bert"] + 0.5 * out["rank_p6"]
    out["F_zsum"] = 0.5 * out["z_bert"] + 0.5 * out["z_p6"]
    out["stock_uid"] = cand["stock_uid"]
    out["date_key"] = cand["date_key"]
    out["true_logret"] = cand["true_logret"].astype(np.float64)
    out["quality"] = cand["quality"].astype(bool)
    out["post_median"] = cand["post_median"].astype(np.float64)
    out["p_up"] = cand["p_up"].astype(np.float64)
    out["post_std"] = cand["post_std"].astype(np.float64)
    if "offset" in cand.files:
        out["offset"] = cand["offset"]
    if "dense_threshold" in cand.files:
        out["dense_threshold"] = cand["dense_threshold"]
    else:
        # calib region: same 0.8*max-cross-section rule used elsewhere
        _, counts = np.unique(dates, return_counts=True)
        out["dense_threshold"] = np.array([max(5, int(np.ceil(0.8 * int(counts.max()))))])

    path = stage_weights("C") / f"scores-{region}-fused.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()})
    print(f"[f0] wrote {path} rows={n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["eval", "calib"], required=True)
    args = ap.parse_args()
    compute_region(args.region)


if __name__ == "__main__":
    main()
