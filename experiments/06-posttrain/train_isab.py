"""PT-04 ISAB pilot: train S3 (fixed forward) and evaluate date-aware.

The ISAB treats one date's full cross-section as the set, so scoring on the
eval window must group rows by date (one set per scoring call).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json, append_trial  # noqa: E402
from posttrain_heads import ISABSetTransformer  # noqa: E402
from train_heads import load_rows, fold_split, final_fit_split, \
    train_rank_per_date  # noqa: E402
from analyze_pt01 import daily_rank_ic, _contrast  # noqa: E402


def score_isab_date_aware(head, hidden, dates, device, min_stocks=30):
    """Score per-date sets; returns per-row score (NaN for dropped dates)."""
    uniq, inv = np.unique(dates, return_inverse=True)
    score = np.full(hidden.shape[0], np.nan, dtype=np.float32)
    head = head.to(device).eval()
    with torch.no_grad():
        for i in range(len(uniq)):
            m = inv == i
            if m.sum() < min_stocks:
                continue
            hb = torch.from_numpy(hidden[m]).to(device)  # [B, latent] (one set)
            score[m] = head(hb).cpu().numpy()
    return score


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_cache",
                    default="server_runs/weights/06-posttrain/seed42/training_cache.npz")
    ap.add_argument("--eval_hidden",
                    default="server_runs/weights/06-posttrain/seed42/hidden_cache.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = load_rows(args.training_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # R2 recipe selection
    fit_idx, val_idx = fold_split(rows, "R2")
    best = None
    for lr in (3e-4, 1e-3):
        head = ISABSetTransformer().to(device)
        h, hist = train_rank_per_date(head, rows, fit_idx, val_idx, "pairwise",
                                      lr=lr, epochs=3, seed=args.seed)
        vloss = hist["val_loss"][-1] if hist.get("val_loss") else hist["train_loss"][-1]
        if best is None or vloss < best[0]:
            best = (vloss, {"lr": lr})
    recipe = {"lr": best[1]["lr"], "epochs": 4, "batch_size": 0}
    fit_all = final_fit_split(rows)
    head = ISABSetTransformer().to(device)
    h, _ = train_rank_per_date(head, rows, fit_all, fit_all, "pairwise",
                               lr=recipe["lr"], epochs=recipe["epochs"], seed=args.seed)
    weights_root = resolve_roots().weights_root
    art = weights_root / "head_S3_isab.pt"
    torch.save({"head_state": head.state_dict(), "arm": "S3_isab",
                "recipe": recipe, "seed": args.seed}, art)
    print(f"[isab] trained {art} (recipe {recipe})")
    append_trial({"stage": "pt04", "arm": "S3_isab", "recipe": recipe})

    # date-aware eval
    d = np.load(args.eval_hidden, allow_pickle=True)
    hidden = d["hidden"]
    dates = np.asarray(d["date_key"])
    true = d["true_logret"].astype(np.float64)
    offset = np.asarray(d["offset"])
    pt01 = np.load(str(Path(args.eval_hidden).with_name("pt01_records.npz")),
                   allow_pickle=True)
    score = score_isab_date_aware(head, hidden, dates, device)
    j0 = pt01["greedy_return"]
    results = {}
    for name, mask in (("full400", np.ones(len(dates), bool)),
                       ("dev_0_299", offset < 300),
                       ("confirmation_300_399", (offset >= 300) & (offset < 400))):
        ic, da, mae, cnt, dense = daily_rank_ic(score[mask], true[mask], dates[mask], 3634)
        ic0, _, _, _, _ = daily_rank_ic(j0[mask], true[mask], dates[mask], 3634)
        c = _contrast(ic0, ic, np.zeros_like(ic0), np.zeros_like(ic))
        results[name] = {"rank_ic": float(np.nanmean(ic[dense])),
                         "delta_vs_J0": c["rank_ic_delta_vs_J0"],
                         "block_robust": c["block_robust"]}
        print(f"[isab] {name}: RankIC={results[name]['rank_ic']:.5f} "
              f"deltaJ0={c['rank_ic_delta_vs_J0']:+.5f} robust={c['block_robust']}")
    out = resolve_roots().results_root / "pt04_isab_result.json"
    write_json(out, {"schema_version": "pt04-isab-v1", "recipe": recipe,
                     "results": results})
    print(f"[isab] wrote {out}")


if __name__ == "__main__":
    main()
