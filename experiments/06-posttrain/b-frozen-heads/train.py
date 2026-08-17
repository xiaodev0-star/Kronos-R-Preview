"""PT-03/PT-04/PT-05: training frozen-backbone independent heads.

Loads the pre-cutoff training cache (fit_uids), runs the R0-R2 rolling
validation to select a fixed-step recipe (no early-stop on 0..399), then trains
the final head on <2023-02-01 without early stopping.  Trained heads are saved
as lightweight artifacts (weights root) that never overwrite the base
checkpoint; the base coarse/fine logits are untouched (guardrail test_08).

Loss reductions are invariant to microbatch packing; rank losses see exactly one
date per batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import resolve_roots, artifact_paths, write_json  # noqa: E402
from common import DailyCrossSectionLoader  # noqa: E402
from common import (  # noqa: E402
    LinearReturnHead, MlpReturnHead, LinearDirectionHead, MlpDirectionHead,
    LinearRankHead, MlpRankHead, IndependentMLP, DeepSetsHead, ISABSetTransformer,
    C1FeatureHead, C2PosteriorHead,
    soft_spearman_loss, pairwise_logistic_loss, huber_loss,
    load_rows, fold_split, final_fit_split, select_recipe, final_fit,
    train_rank_per_date,
)


def run_pt03(training_cache, calibration_cache, eval_hidden, roots=None, seed=42):
    """Run the PT-03 probe matrix; save head artifacts + selection JSON."""
    rows = load_rows(training_cache)
    roots = roots or resolve_roots(seed=seed)
    paths = artifact_paths(roots=roots)
    results_root = roots.results_root / "B-heads"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def hp_grid(head_kw, lrs, epochs=6, batch=4096):
        out = []
        for lr in lrs:
            out.append({"head": head_kw, "lr": lr, "epochs": epochs,
                        "batch_size": batch})
        return out

    arms = [
        ("return-linear", lambda: LinearReturnHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
        ("return-mlp", lambda: MlpReturnHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
        ("direction-linear", lambda: LinearDirectionHead(), "bce",
         hp_grid({}, [3e-4, 1e-3])),
        ("direction-mlp", lambda: MlpDirectionHead(), "bce",
         hp_grid({}, [3e-4, 1e-3])),
        ("rank-linear-pairwise", lambda: LinearRankHead(loss="pairwise"), "pairwise",
         hp_grid({}, [3e-4, 1e-3])),
        ("rank-mlp-spearman", lambda: MlpRankHead(loss="soft_spearman"), "soft_spearman",
         hp_grid({}, [3e-4, 1e-3])),
        ("rank-mlp-pairwise", lambda: MlpRankHead(loss="pairwise"), "pairwise",
         hp_grid({}, [3e-4, 1e-3])),
        # C1 control: same-capacity head on point-in-time features only
        ("feature-only", lambda: C1FeatureHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
    ]
    head_paths = {
        "return-linear": paths["head_return_linear"],
        "return-mlp": paths["head_return_mlp"],
        "direction-linear": paths["head_direction_linear"],
        "direction-mlp": paths["head_direction_mlp"],
        "rank-linear-pairwise": paths["head_rank_linear_pairwise"],
        "rank-mlp-spearman": paths["head_rank_mlp_spearman"],
        "rank-mlp-pairwise": paths["head_rank_mlp_pairwise"],
        "feature-only": paths["head_feature_only"],
    }
    paths["head_dir"].mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, factory, loss_kind, grid in arms:
        print(f"[pt03] selecting recipe for {name}")
        recipe = select_recipe(factory, rows, loss_kind, grid, seed=seed)
        head = final_fit(factory, rows, recipe, loss_kind, seed=seed)
        path = head_paths[name]
        torch.save({"head_state": head.state_dict(),
                    "arm": name, "recipe": recipe, "seed": seed}, path)
        summary[name] = {"recipe": recipe, "artifact": str(path)}
    write_json(results_root / "pt03_recipe_selection.json", summary)
    print("[pt03] done")
    return summary


# ============================================================================
# PT-04: cross-sectional set heads (Independent MLP / DeepSets / ISAB pilot)
# ============================================================================

def build_cross_section_records(rows, date_filter=None):
    """Convert the training cache arrays into per-row records for the loader."""
    recs = []
    dates = np.asarray(rows["date_key"])
    for i in range(len(dates)):
        d = str(dates[i])[:10]
        if date_filter is not None and not date_filter(d):
            continue
        recs.append({"date_key": d, "stock_uid": str(rows["stock_uid"][i]),
                     "hidden": rows["hidden"][i],
                     "true_logret": float(rows["true_logret"][i])})
    return recs


def run_pt04(training_cache, eval_hidden, roots=None, seed=42):
    """PT-04: Independent MLP + DeepSets (controls); ISAB pilot only if DeepSets
    shows increment.  Uses the faster pairwise rank loss and a compact R2-based
    recipe selection to bound runtime on the small GPU."""
    rows = load_rows(training_cache)
    roots = roots or resolve_roots(seed=seed)
    paths = artifact_paths(roots=roots)
    results_root = roots.results_root / "B-heads"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recipe selection on R2 only (fit <2022-02-01, val [2022-02-01, 2023-02-01))
    fit_idx, val_idx = fold_split(rows, "R2")
    summary = {}
    for name, factory, path_key in (("independent-mlp", IndependentMLP, "head_independent_mlp"),
                                    ("deepsets", DeepSetsHead, "head_deepsets"),
                                    ("isab", ISABSetTransformer, "head_isab")):
        print(f"[pt04] selecting recipe for {name} (R2)")
        best = None
        for lr in (3e-4, 1e-3):
            head = factory().to(device)
            h, hist = train_rank_per_date(head, rows, fit_idx, val_idx, "pairwise",
                                          lr=lr, epochs=3, seed=seed)
            vloss = hist["val_loss"][-1] if hist.get("val_loss") else hist["train_loss"][-1]
            if best is None or vloss < best[0]:
                best = (vloss, {"lr": lr})
        if best is None:
            continue
        recipe = {"lr": best[1]["lr"], "epochs": 4, "batch_size": 0}
        fit_idx_all = final_fit_split(rows)
        head = factory().to(device)
        h, _ = train_rank_per_date(head, rows, fit_idx_all, fit_idx_all, "pairwise",
                                   lr=recipe["lr"], epochs=recipe["epochs"], seed=seed)
        path = paths[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head_state": head.state_dict(), "arm": name,
                    "recipe": recipe, "seed": seed}, path)
        summary[name] = {"recipe": recipe, "artifact": str(path),
                         "selected": best[1]}
    write_json(results_root / "pt04_recipe_selection.json", summary)
    print("[pt04] done")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run PT-03 frozen probe matrix")
    ap.add_argument("--training_cache", default=None)
    ap.add_argument("--calibration_cache", default=None)
    ap.add_argument("--eval_hidden", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["pt03", "pt04"], default="pt03")
    args = ap.parse_args()
    paths = artifact_paths(seed=args.seed)
    training = args.training_cache or str(paths["training"])
    calibration = args.calibration_cache or str(paths["calibration"])
    hidden = args.eval_hidden or str(paths["hidden"])
    if args.stage == "pt04":
        run_pt04(training, hidden, seed=args.seed)
    else:
        run_pt03(training, calibration, hidden, seed=args.seed)


if __name__ == "__main__":
    main()
