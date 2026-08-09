"""f2_concat_head.py — Round 1b: representation-level fusion rank head.

Train an MLP rank head (soft-Spearman, per-date) on the CONCATENATED hidden of
the two decorrelated frozen backbones:
    GPT  causal-hidden    [N, 256]  (06 training_cache.npz / hidden_cache.npz)
    BERT bidirectional    [N, 256]  (07 bert_hidden_{fit,eval}_w512_t2.npz)
=> [N, 512] representation -> MlpRankHead -> per-date soft-Spearman.

This is a *learned* fusion: unlike the score-level blend (F_ranksum), the head
learns how much to trust each representation per-stock.  Protocol identical to
T5/P6: R0-R2 rolling recipe selection, final fit on <2023-02-01, no early stop,
0..399 pure inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
SIX = ROOT / "experiments" / "06-posttrain"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, SIX, EIGHT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from posttrain_heads import MlpRankHead  # noqa: E402
from train_heads import fold_split, final_fit_split, train_rank_per_date  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, bootstrap_vs, build_rec,
    cand_path, write_json_ledger, append_trial,
)

HP_GRID = [
    {"lr": 1e-3, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
    {"lr": 1e-3, "epochs": 12, "dropout": 0.1},
]


def six_wr():
    return ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"


def build_fit_rows(wr):
    sw = six_wr()
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    b = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    assert np.array_equal(tc["stock_uid"], b["stock_uid"]), "fit uid mismatch"
    assert np.array_equal(tc["date_key"], b["date_key"]), "fit date mismatch"
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = np.concatenate([tc["hidden"], b["hidden"]], axis=1)
    return rows


def select_recipe(rows, seed=42):
    results = []
    for hp in HP_GRID:
        losses = []
        for fold in ("R0", "R1", "R2"):
            fit_idx, val_idx = fold_split(rows, fold)
            if len(fit_idx) < 50 or len(val_idx) < 20:
                continue
            head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                               dropout=hp["dropout"], loss="soft_spearman")
            _, hist = train_rank_per_date(head, rows, fit_idx, val_idx,
                                          "soft_spearman", lr=hp["lr"],
                                          epochs=hp["epochs"], seed=seed)
            losses.append(hist["val_loss"][-1])
        mean_vl = float(np.mean(losses)) if losses else None
        results.append({"hp": hp, "mean_fold_val_loss": mean_vl,
                        "fold_val_loss": losses})
        print(f"[f2] hp={hp} mean_fold_val_loss={mean_vl:.4f}", flush=True)
    results = [r for r in results if r["mean_fold_val_loss"] is not None]
    results.sort(key=lambda r: r["mean_fold_val_loss"])
    return results[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply_eval", action="store_true")
    ap.add_argument("--skip_selection", action="store_true",
                    help="skip R0-R2, use locked recipe below")
    ap.add_argument("--locked_recipe", type=str, default="",
                    help="json dict override e.g. '{\"lr\":0.0003,\"epochs\":16,\"dropout\":0.1}'")
    ap.add_argument("--tag", type=str, default="concat")
    ap.add_argument("--apply_only", action="store_true",
                    help="skip training; load existing head and only run eval apply")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    if args.apply_only:
        _apply_existing(args, wr, rr)
        return

    rows = build_fit_rows(wr)
    print(f"[f2] concat fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")

    if args.skip_selection:
        import json as _j
        recipe = _j.loads(args.locked_recipe) if args.locked_recipe else \
            {"lr": 3e-4, "epochs": 16, "dropout": 0.1}
        best = {"hp": recipe, "mean_fold_val_loss": None, "fold_val_loss": []}
        print(f"[f2] locked recipe (no R0-R2): {recipe}")
    else:
        best = select_recipe(rows, seed=args.seed)
        print(f"[f2] selected recipe: {best['hp']}")

    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                       dropout=best["hp"]["dropout"], loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=best["hp"]["lr"], epochs=best["hp"]["epochs"],
                                  seed=args.seed)
    out_head = wr / f"head_{args.tag}_mlp_rank_spearman_seed{args.seed}.pt"
    torch.save({"head_state": head.state_dict(), "recipe": best["hp"],
                "seed": args.seed, "fold_selection": best,
                "train_history": hist, "schema": "f2-concat-head-v1"},
               out_head)
    print(f"[f2] saved {out_head}")

    result = {"schema": "f2-concat-head-v1", "seed": args.seed,
              "recipe": best["hp"], "fold_selection": best,
              "final_train_loss": hist["train_loss"][-1],
              "n_fit_rows": int(len(fit_idx))}

    if args.apply_eval:
        cand = np.load(cand_path("eval"), allow_pickle=True)
        sw = six_wr()
        g = np.load(sw / "hidden_cache.npz", allow_pickle=True)
        be = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
        n = len(cand["stock_uid"])
        assert len(g["stock_uid"]) == n and len(be["stock_uid"]) == n
        H = torch.from_numpy(
            np.concatenate([g["hidden"], be["hidden"]], axis=1).astype(np.float32))
        head.eval()
        with torch.no_grad():
            score = head(H).numpy()
        rec = build_rec(cand)
        rec["concat_head"] = score.astype(np.float64)
        p6 = np.load(wr / "p6_eval_scores.npy")
        rec["p6_score"] = p6.astype(np.float64)
        head_bert = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        ck = torch.load(str(wr / "head_BERT_mlp_rank_spearman_seed42.pt"),
                        map_location="cpu", weights_only=False)
        head_bert.load_state_dict(ck["head_state"])
        head_bert.eval()
        with torch.no_grad():
            rec["bert_head"] = head_bert(
                torch.from_numpy(be["hidden"].astype(np.float32))).numpy()
        del H, g, be
        dense = int(cand["dense_threshold"][0])
        fields = {"CONCAT": "concat_head", "T5": "bert_head", "P6": "p6_score",
                  "J3_median": "post_median"}
        result["eval_metrics"] = metrics_table(rec, fields, dense)
        result["bootstrap_vs_T5"] = bootstrap_vs(rec, "concat_head", rec, "bert_head", dense)
        result["bootstrap_vs_P6"] = bootstrap_vs(rec, "concat_head", rec, "p6_score", dense)
        result["bootstrap_vs_J3"] = bootstrap_vs(rec, "concat_head", rec, "post_median", dense)
        print(json.dumps({"eval_metrics": result["eval_metrics"],
                          "vs_T5": result["bootstrap_vs_T5"],
                          "vs_P6": result["bootstrap_vs_P6"]}, indent=1, default=str))

    out = rr / f"f2_concat_head_seed{args.seed}.json"
    write_json_ledger(out, result, "f2_concat_head", seed=args.seed, tag=args.tag)
    print(f"[f2] result -> {out}")


def _apply_existing(args, wr, rr):
    """Apply a previously trained concat head to eval (no training)."""
    head_path = wr / f"head_{args.tag}_mlp_rank_spearman_seed{args.seed}.pt"
    if not head_path.exists():
        raise RuntimeError(f"head missing: {head_path}")
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=512, hidden=64, dropout=0.0, loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    print(f"[f2-apply] head <- {head_path} recipe={ck.get('recipe')}")

    cand = np.load(cand_path("eval"), allow_pickle=True)
    sw = six_wr()
    g = np.load(sw / "hidden_cache.npz", allow_pickle=True)
    be = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    n = len(cand["stock_uid"])
    assert len(g["stock_uid"]) == n and len(be["stock_uid"]) == n
    H = torch.from_numpy(
        np.concatenate([g["hidden"], be["hidden"]], axis=1).astype(np.float32))
    with torch.no_grad():
        score = head(H).numpy()
    rec = build_rec(cand)
    rec["concat_head"] = score.astype(np.float64)
    p6 = np.load(wr / "p6_eval_scores.npy")
    rec["p6_score"] = p6.astype(np.float64)
    head_bert = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
    ck_bert = torch.load(str(wr / "head_BERT_mlp_rank_spearman_seed42.pt"),
                         map_location="cpu", weights_only=False)
    head_bert.load_state_dict(ck_bert["head_state"])
    head_bert.eval()
    with torch.no_grad():
        rec["bert_head"] = head_bert(
            torch.from_numpy(be["hidden"].astype(np.float32))).numpy()
    del H, g, be
    dense = int(cand["dense_threshold"][0])
    fields = {"CONCAT": "concat_head", "T5": "bert_head", "P6": "p6_score",
              "J3_median": "post_median"}
    result = {"schema": "f2-concat-head-apply-v1", "seed": args.seed,
              "recipe": ck.get("recipe")}
    result["eval_metrics"] = metrics_table(rec, fields, dense)
    result["bootstrap_vs_T5"] = bootstrap_vs(rec, "concat_head", rec, "bert_head", dense)
    result["bootstrap_vs_P6"] = bootstrap_vs(rec, "concat_head", rec, "p6_score", dense)
    result["bootstrap_vs_J3"] = bootstrap_vs(rec, "concat_head", rec, "post_median", dense)
    out = rr / f"f2_concat_head_seed{args.seed}_apply.json"
    write_json_ledger(out, result, "f2_concat_head_apply", seed=args.seed, tag=args.tag)
    print(json.dumps({"eval_metrics": result["eval_metrics"],
                      "vs_T5": result["bootstrap_vs_T5"],
                      "vs_P6": result["bootstrap_vs_P6"]}, indent=1, default=str))
    print(f"[f2-apply] -> {out}")


if __name__ == "__main__":
    main()
