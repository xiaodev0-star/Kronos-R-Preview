"""train_bert_head.py — plan §4 T5: BERT-hidden rank head ("P6 on BERT").

Frozen BERT MASK-position hidden [N, 256] (from cache_bert_hidden.py) -> a
MlpRankHead trained with soft-Spearman per date, exactly the 06 P6 recipe
(R0-R2 rolling selection, final fit on <2023-02-01, no early stop on 0..399).

Acceptance (plan §4 T5): BERT-head alone RankIC >= J3 (0.0398) and stacking
with P6 >= +0.010 vs P6.  Trained components need seed 43/44 confirmation.

Usage:
    python experiments/07-bert-critic/train_bert_head.py --seed 42
    python experiments/07-bert-critic/train_bert_head.py --seed 42 --apply_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from posttrain_heads import MlpRankHead  # noqa: E402
from train_heads import fold_split, final_fit_split, train_rank_per_date  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, append_trial,
)

# fixed recipe candidate grid (06 P6-style; R0-R2 roll selects one)
HP_GRID = [
    {"lr": 1e-3, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
    {"lr": 1e-3, "epochs": 12, "dropout": 0.1},
]


def _rows_from_hidden(hidden_path):
    d = np.load(hidden_path, allow_pickle=True)
    rows = {k: d[k] for k in d.files}
    # the fit hidden cache is built from the 06 training_cache rows in order;
    # join the target / normalization fields the head trainer needs.
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    if len(tc["stock_uid"]) == len(rows["stock_uid"]):
        for k in ("true_logret", "p_mean0", "p_std0", "quality"):
            if k in tc.files:
                rows[k] = tc[k]
    else:
        raise RuntimeError(
            f"fit hidden cache rows ({len(rows['stock_uid'])}) != training_cache "
            f"({len(tc['stock_uid'])}) — cannot join true_logret")
    return rows


def _head_fingerprint():
    pass


def select_recipe(rows, loss_kind="soft_spearman", seed=42):
    """R0-R2 rolling: pick the hp minimizing mean fold val rank loss."""
    results = []
    for hp in HP_GRID:
        fold_losses = []
        for fold in ("R0", "R1", "R2"):
            fit_idx, val_idx = fold_split(rows, fold)
            head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                               dropout=hp["dropout"], loss=loss_kind)
            _, hist = train_rank_per_date(head, rows, fit_idx, val_idx,
                                          loss_kind, lr=hp["lr"],
                                          epochs=hp["epochs"], seed=seed)
            fold_losses.append(hist["val_loss"][-1])
        results.append({"hp": hp, "mean_fold_val_loss": float(np.mean(fold_losses)),
                        "fold_val_loss": dict(zip(("R0", "R1", "R2"), fold_losses))})
        print(f"[t5] hp={hp} mean_fold_val_loss={np.mean(fold_losses):.4f}", flush=True)
    results.sort(key=lambda r: r["mean_fold_val_loss"])
    return results[0]


def main():
    ap = argparse.ArgumentParser(description="T5 BERT-hidden rank head")
    ap.add_argument("--hidden_fit", default=None)
    ap.add_argument("--suffix", type=str, default="",
                    help="hidden-cache suffix (e.g. t1_w512) matching cache_bert_hidden --suffix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply_eval", action="store_true",
                    help="apply to eval hidden cache if present")
    args = ap.parse_args()

    wr = weights_root()
    if args.hidden_fit:
        hidden_fit = Path(args.hidden_fit)
    else:
        sfx = f"_{args.suffix}" if args.suffix else ""
        hidden_fit = wr / f"bert_hidden_fit_w512{sfx}.npz"
    if not hidden_fit.exists():
        raise RuntimeError(f"fit hidden cache missing: {hidden_fit} "
                           "(run cache_bert_hidden.py --region fit first)")
    rows = _rows_from_hidden(hidden_fit)
    print(f"[t5] fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")

    best = select_recipe(rows, seed=args.seed)
    print(f"[t5] selected recipe: {best['hp']}")

    # final fit on all pre-cutoff fit rows (no early stop)
    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                       dropout=best["hp"]["dropout"], loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=best["hp"]["lr"], epochs=best["hp"]["epochs"],
                                  seed=args.seed)
    out_head = wr / f"head_BERT_mlp_rank_spearman_seed{args.seed}.pt"
    torch.save({"head_state": head.state_dict(),
                "recipe": best["hp"], "seed": args.seed,
                "fold_selection": best,
                "train_history": hist,
                "hidden_cache": str(hidden_fit),
                "schema": "t5-bert-head-v1"}, out_head)
    print(f"[t5] saved {out_head}")

    result = {"schema": "t5-head-v1", "seed": args.seed,
              "recipe": best["hp"], "fold_selection": best,
              "final_train_loss": hist["train_loss"][-1],
              "n_fit_rows": int(len(fit_idx))}
    # ---- eval apply (offsets 0..399), if eval hidden cache present ----
    eval_hidden = wr / f"bert_hidden_eval_w512{('_' + args.suffix) if args.suffix else ''}.npz"
    if args.apply_eval and eval_hidden.exists():
        cand = np.load(cand_path("eval"), allow_pickle=True)
        de = np.load(eval_hidden, allow_pickle=True)
        n = len(cand["stock_uid"])
        if len(de["stock_uid"]) != n:
            raise RuntimeError("eval hidden cache misaligned with candidates")
        H = torch.from_numpy(de["hidden"].astype(np.float32))
        head.eval()
        with torch.no_grad():
            score = head(H).numpy()
        rec = build_rec(cand)
        rec["bert_head"] = score.astype(np.float64)
        p6 = np.load(weights_root() / "p6_eval_scores.npy")
        if len(p6) != n:
            raise RuntimeError(f"P6 len {len(p6)} != candidates {n}")
        rec["p6_score"] = p6.astype(np.float64)
        dense = int(cand["dense_threshold"][0])
        result["eval_metrics"] = metrics_table(rec, {
            "BERT_head": "bert_head", "J3_median": "post_median", "P6": "p6_score"},
            dense)
        result["bootstrap_vs_J3"] = bootstrap_vs(rec, "bert_head", rec, "post_median", dense)
        result["bootstrap_vs_P6"] = bootstrap_vs(rec, "bert_head", rec, "p6_score", dense)
        print(json.dumps({"eval_metrics": result["eval_metrics"],
                          "vs_J3": result["bootstrap_vs_J3"],
                          "vs_P6": result["bootstrap_vs_P6"]}, indent=1, default=str))
    json_path = results_root() / f"t5_head_seed{args.seed}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    append_trial({"event": "t5_head_train", "seed": args.seed, "status": "ok"})
    print(f"[t5] result -> {json_path}")


if __name__ == "__main__":
    main()
