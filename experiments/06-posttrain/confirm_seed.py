"""PT-90: seed43/44 confirmation for the leading trained head (locked recipe).

Per protocol §13.4: only the unique finalist (or up to two trained candidates
before lock) is advanced to seeds 43/44 with the EXACT locked recipe; both
confirmation seeds must be same-direction as the main hypothesis before
``ready_for_holdout=true``.  The seed is the only changed variable.

Usage:
    python confirm_seed.py --arm P5_linear_rank_pairwise --seeds 43 44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json, append_trial  # noqa: E402
from posttrain_heads import LinearRankHead, MlpRankHead, LinearReturnHead, \
    MlpReturnHead, LinearDirectionHead, MlpDirectionHead  # noqa: E402
from train_heads import load_rows, train_rank_per_date, final_fit_split  # noqa: E402
from analyze_pt01 import daily_rank_ic, _contrast  # noqa: E402

FACTORIES = {
    "P1_linear_return": (LinearReturnHead, "huber"),
    "P2_mlp_return": (MlpReturnHead, "huber"),
    "P3_linear_direction": (LinearDirectionHead, "bce"),
    "P4_mlp_direction": (MlpDirectionHead, "bce"),
    "P5_linear_rank_pairwise": (LinearRankHead, "pairwise"),
    "P6_mlp_rank_spearman": (MlpRankHead, "soft_spearman"),
    "P7_mlp_rank_pairwise": (MlpRankHead, "pairwise"),
}


def confirm(arm, seeds, training_cache, eval_hidden, recipe, dense_min=3634,
            out_path=None):
    rows = load_rows(training_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(eval_hidden, allow_pickle=True)
    hidden = torch.from_numpy(d["hidden"]).to(device)
    dates = np.asarray(d["date_key"])
    true = d["true_logret"].astype(np.float64)
    offset = np.asarray(d["offset"])
    pt01 = np.load(str(Path(eval_hidden).with_name("pt01_records.npz")),
                   allow_pickle=True)
    base_ic, _, _, _, _ = daily_rank_ic(pt01["greedy_return"], true, dates, dense_min)
    j3_ic, _, _, _, _ = daily_rank_ic(pt01["post_median"], true, dates, dense_min)

    factory, loss_kind = FACTORIES[arm]
    hp = recipe["hp"]
    results = {}
    for seed in seeds:
        fit_idx = final_fit_split(rows)
        head = factory().to(device)
        h, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, loss_kind,
                                      lr=hp["lr"], epochs=hp["epochs"], seed=seed)
        h.eval()
        with torch.no_grad():
            score = h(hidden).squeeze(-1).cpu().numpy()
        ic, da, mae, cnt, dense = daily_rank_ic(score, true, dates, dense_min)
        c_j0 = _contrast(base_ic, ic, np.zeros_like(base_ic), np.zeros_like(ic))
        c_j3 = _contrast(j3_ic, ic, np.zeros_like(j3_ic), np.zeros_like(ic))
        results[f"seed{seed}"] = {
            "avg_daily_rank_ic": float(np.nanmean(ic[dense])),
            "avg_da_per_date": float(np.nanmean(da[dense])),
            "contrast_vs_J0": c_j0,
            "contrast_vs_J3": c_j3,
        }
        print(f"[confirm] seed{seed}: RankIC={results[f'seed{seed}']['avg_daily_rank_ic']:.5f} "
              f"vsJ0={c_j0['rank_ic_delta_vs_J0']:+.5f} vsJ3={c_j3['rank_ic_delta_vs_J0']:+.5f}")
        append_trial({"stage": "pt90_confirm", "arm": arm, "seed": seed,
                      "recipe": recipe, "rank_ic": results[f"seed{seed}"]["avg_daily_rank_ic"]})
    summary = {"schema_version": "pt90-seed-confirm-v1", "arm": arm,
               "seeds": seeds, "recipe": recipe, "results": results,
               "posttrain_three_seed_consistent": None}
    if out_path:
        write_json(out_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="P5_linear_rank_pairwise")
    ap.add_argument("--seeds", default="43,44")
    ap.add_argument("--training_cache",
                    default="server_runs/weights/06-posttrain/seed42/training_cache.npz")
    ap.add_argument("--eval_hidden",
                    default="server_runs/weights/06-posttrain/seed42/hidden_cache.npz")
    ap.add_argument("--recipe", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    # locked recipe for the seed42 winner comes from the trial ledger
    import json
    ledger = [json.loads(l) for l in open(
        "experiments/06-posttrain/trial_ledger.jsonl", encoding="utf-8")
        if l.strip()]
    recipe = None
    if args.recipe:
        recipe = json.loads(args.recipe)
    else:
        for row in reversed(ledger):
            if row.get("stage") == "pt03" and row.get("arm") == args.arm:
                recipe = row["recipe"]
                break
    if recipe is None:
        raise SystemExit(f"no locked recipe found for {args.arm}")
    roots = resolve_roots()
    out = roots.results_root / f"pt90_confirm_{args.arm}.json"
    confirm(args.arm, seeds, args.training_cache, args.eval_hidden, recipe,
            out_path=out)


if __name__ == "__main__":
    main()
