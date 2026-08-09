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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json, append_trial  # noqa: E402
from posttrain_data import DailyCrossSectionLoader  # noqa: E402
from posttrain_heads import (  # noqa: E402
    LinearReturnHead, MlpReturnHead, LinearDirectionHead, MlpDirectionHead,
    LinearRankHead, MlpRankHead, IndependentMLP, DeepSetsHead, ISABSetTransformer,
    C1FeatureHead, C2PosteriorHead,
    soft_spearman_loss, pairwise_logistic_loss, huber_loss,
)

FOLDS = {
    "R0": (None, "2020-02-01", "2020-02-01", "2021-02-01"),
    "R1": (None, "2021-02-01", "2021-02-01", "2022-02-01"),
    "R2": (None, "2022-02-01", "2022-02-01", "2023-02-01"),
}
FINAL_FIT_STOP = "2023-02-01"


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def in_interval(d, start, stop):
    if start is not None and d < start:
        return False
    if stop is not None and d >= stop:
        return False
    return True


def fold_split(rows, fold):
    start, fstop, vstart, vstop = FOLDS[fold]
    fit_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], start, fstop)]
    val_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], vstart, vstop)]
    return np.asarray(fit_idx), np.asarray(val_idx)


def final_fit_split(rows):
    idx = [i for i, d in enumerate(rows["date_key"])
           if str(d)[:10] < FINAL_FIT_STOP]
    return np.asarray(idx)


def bce_direction(head_out, y):
    return nn.functional.binary_cross_entropy(
        head_out, y, reduction="mean")


def compute_head_loss(head, h, feats, y, loss_kind, tau=1.0, dead_zone=None):
    if isinstance(head, (C1FeatureHead,)):
        out = head(feats)
    elif isinstance(head, (C2PosteriorHead,)):
        out = head(h)
    else:
        out = head(h)
    if loss_kind in ("huber", "mse", "reg"):
        return huber_loss(out, y, delta=1.0)
    if loss_kind in ("bce", "direction"):
        return bce_direction(out, y)
    if loss_kind in ("soft_spearman",):
        ranks = np.argsort(np.argsort(y.detach().cpu().numpy())).astype(np.float32)
        t = torch.from_numpy(ranks).to(y.device)
        return soft_spearman_loss(out, t, tau=tau)
    if loss_kind in ("pairwise",):
        return pairwise_logistic_loss(out, y, tau=tau, dead_zone=dead_zone)
    raise ValueError(loss_kind)


def train_one(head, rows, fit_idx, val_idx, loss_kind, *, lr, epochs, batch_size,
              seed=42, tau=1.0, dead_zone=None, val_every=1, max_grad_norm=1.0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    device = next(head.parameters()).device
    H = torch.from_numpy(rows["hidden"]).to(device)
    Y = torch.from_numpy(rows["true_logret"].astype(np.float32)).to(device)
    if loss_kind in ("bce", "direction"):
        Y = (Y > 0.0).float()
    feats = torch.from_numpy(rows["c1_feats"].astype(np.float32)).to(device) \
        if "c1_feats" in rows else None
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.RandomState(seed)
    n_fit = len(fit_idx)
    best_val = None
    for ep in range(epochs):
        perm = rng.permutation(n_fit)
        head.train()
        ep_loss = 0.0
        steps = 0
        for s in range(0, n_fit, batch_size):
            ids = fit_idx[perm[s:s + batch_size]]
            if len(ids) < 2:
                continue
            hb = H[ids]
            yb = Y[ids]
            fb = feats[ids] if feats is not None else None
            opt.zero_grad()
            loss = compute_head_loss(head, hb, fb, yb, loss_kind, tau=tau,
                                     dead_zone=dead_zone)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            opt.step()
            ep_loss += loss.item()
            steps += 1
        ep_loss /= max(1, steps)
        history["train_loss"].append(ep_loss)
        if (ep + 1) % val_every == 0:
            head.eval()
            with torch.no_grad():
                vloss = 0.0
                vsteps = 0
                for s in range(0, len(val_idx), batch_size):
                    ids = val_idx[s:s + batch_size]
                    if len(ids) < 2:
                        continue
                    vloss += compute_head_loss(
                        head, H[ids], feats[ids] if feats is not None else None,
                        Y[ids], loss_kind, tau=tau, dead_zone=dead_zone).item()
                    vsteps += 1
                vloss /= max(1, vsteps)
            history["val_loss"].append(vloss)
            if best_val is None or vloss < best_val:
                best_val = vloss
    return head, history


def _build_cross_section_recs(rows, idx):
    recs = []
    for i in idx:
        recs.append({"date_key": str(rows["date_key"][i])[:10],
                     "stock_uid": str(rows["stock_uid"][i]),
                     "hidden": rows["hidden"][i],
                     "true_logret": float(rows["true_logret"][i]),
                     "raw_logret": float(rows["true_logret"][i])})
    return recs


def _eval_rank_loss(head, rows, val_idx, loss_kind, tau=1.0, dead_zone=None):
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, val_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=0, shuffle_dates=False)
    head.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            total += compute_head_loss(head, hb, None, yb, loss_kind, tau=tau,
                                       dead_zone=dead_zone).item()
            steps += 1
    return total / max(1, steps)


def train_rank_per_date(head, rows, fit_idx, val_idx, loss_kind, *, lr, epochs,
                        seed=42, tau=1.0, dead_zone=None):
    """Rank heads train one date cross-section per step (soft-Spearman/pairwise)."""
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, fit_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=seed, shuffle_dates=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        head.train()
        total = 0.0
        steps = 0
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            opt.zero_grad()
            loss = compute_head_loss(head, hb, None, yb, loss_kind, tau=tau,
                                     dead_zone=dead_zone)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        history["train_loss"].append(total / max(1, steps))
        history["val_loss"].append(_eval_rank_loss(
            head, rows, val_idx, loss_kind, tau=tau, dead_zone=dead_zone))
    return head, history


def select_recipe(head_factory, rows, loss_kind, hyper_grid, seed=42, **loss_kw):
    """R0-R2 rolling selection of a fixed-step recipe (lr/dropout/epochs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for hp in hyper_grid:
        val_scores = []
        for fold in ("R0", "R1", "R2"):
            fit_idx, val_idx = fold_split(rows, fold)
            if len(fit_idx) < 50 or len(val_idx) < 20:
                val_scores.append(None)
                continue
            head = head_factory(**hp["head"]).to(device)
            if loss_kind in ("soft_spearman", "pairwise"):
                h, hist = train_rank_per_date(
                    head, rows, fit_idx, val_idx, loss_kind, lr=hp["lr"],
                    epochs=hp["epochs"], seed=seed, **loss_kw)
            else:
                h, hist = train_one(
                    head, rows, fit_idx, val_idx, loss_kind, lr=hp["lr"],
                    epochs=hp["epochs"], batch_size=hp["batch_size"], seed=seed,
                    **loss_kw)
            val_scores.append(hist["val_loss"][-1] if hist.get("val_loss") else None)
        valid = [v for v in val_scores if v is not None]
        results.append({"hp": hp, "fold_val_loss": dict(zip(("R0", "R1", "R2"),
                                                            val_scores)),
                        "mean_val_loss": float(np.mean(valid)) if valid else None})
    best = min([r for r in results if r["mean_val_loss"] is not None],
               key=lambda r: r["mean_val_loss"])
    return best


def final_fit(head_factory, rows, recipe, loss_kind, seed=42, **loss_kw):
    """Final head fit on all <2023-02-01 with the locked recipe (no early stop)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_idx = final_fit_split(rows)
    hp = recipe["hp"]
    head = head_factory(**hp["head"]).to(device)
    if loss_kind in ("soft_spearman", "pairwise"):
        h, hist = train_rank_per_date(
            head, rows, fit_idx, fit_idx, loss_kind, lr=hp["lr"],
            epochs=hp["epochs"], seed=seed, **loss_kw)
    else:
        h, hist = train_one(
            head, rows, fit_idx, fit_idx, loss_kind, lr=hp["lr"],
            epochs=hp["epochs"], batch_size=hp["batch_size"], seed=seed, **loss_kw)
    return h


def run_pt03(training_cache, calibration_cache, eval_hidden, roots=None, seed=42):
    """Run the PT-03 probe matrix; save head artifacts + selection JSON."""
    rows = load_rows(training_cache)
    weights_root = (roots.weights_root if roots else resolve_roots().weights_root)
    results_root = (roots.results_root if roots else resolve_roots().results_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def hp_grid(head_kw, lrs, epochs=6, batch=4096):
        out = []
        for lr in lrs:
            out.append({"head": head_kw, "lr": lr, "epochs": epochs,
                        "batch_size": batch})
        return out

    arms = [
        ("P1_linear_return", lambda: LinearReturnHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
        ("P2_mlp_return", lambda: MlpReturnHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
        ("P3_linear_direction", lambda: LinearDirectionHead(), "bce",
         hp_grid({}, [3e-4, 1e-3])),
        ("P4_mlp_direction", lambda: MlpDirectionHead(), "bce",
         hp_grid({}, [3e-4, 1e-3])),
        ("P5_linear_rank_pairwise", lambda: LinearRankHead(loss="pairwise"), "pairwise",
         hp_grid({}, [3e-4, 1e-3])),
        ("P6_mlp_rank_spearman", lambda: MlpRankHead(loss="soft_spearman"), "soft_spearman",
         hp_grid({}, [3e-4, 1e-3])),
        ("P7_mlp_rank_pairwise", lambda: MlpRankHead(loss="pairwise"), "pairwise",
         hp_grid({}, [3e-4, 1e-3])),
        # C1 control: same-capacity head on point-in-time features only
        ("C1_feature_only", lambda: C1FeatureHead(), "huber",
         hp_grid({}, [3e-4, 1e-3])),
    ]
    summary = {}
    for name, factory, loss_kind, grid in arms:
        print(f"[pt03] selecting recipe for {name}")
        recipe = select_recipe(factory, rows, loss_kind, grid, seed=seed)
        head = final_fit(factory, rows, recipe, loss_kind, seed=seed)
        path = weights_root / f"head_{name}.pt"
        torch.save({"head_state": head.state_dict(),
                    "arm": name, "recipe": recipe, "seed": seed}, path)
        summary[name] = {"recipe": recipe, "artifact": str(path)}
        append_trial({"stage": "pt03", "arm": name, "recipe": recipe})
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
    weights_root = (roots.weights_root if roots else resolve_roots().weights_root)
    results_root = (roots.results_root if roots else resolve_roots().results_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recipe selection on R2 only (fit <2022-02-01, val [2022-02-01, 2023-02-01))
    fit_idx, val_idx = fold_split(rows, "R2")
    summary = {}
    for name, factory in (("S1_independent_mlp", IndependentMLP),
                          ("S2_deepsets", DeepSetsHead),
                          ("S3_isab", ISABSetTransformer)):
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
        path = weights_root / f"head_{name}.pt"
        torch.save({"head_state": head.state_dict(), "arm": name,
                    "recipe": recipe, "seed": seed}, path)
        summary[name] = {"recipe": recipe, "artifact": str(path),
                         "selected": best[1]}
        append_trial({"stage": "pt04", "arm": name, "recipe": recipe})
    write_json(results_root / "pt04_recipe_selection.json", summary)
    print("[pt04] done")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run PT-03 frozen probe matrix")
    ap.add_argument("--training_cache", default="server_runs/weights/06-posttrain/seed42/training_cache.npz")
    ap.add_argument("--calibration_cache", default="server_runs/weights/06-posttrain/seed42/calibration_cache.npz")
    ap.add_argument("--eval_hidden", default="server_runs/weights/06-posttrain/seed42/hidden_cache.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["pt03", "pt04"], default="pt03")
    args = ap.parse_args()
    if args.stage == "pt04":
        run_pt04(args.training_cache, args.eval_hidden, seed=args.seed)
    else:
        run_pt03(args.training_cache, args.calibration_cache, args.eval_hidden, seed=args.seed)


if __name__ == "__main__":
    main()
