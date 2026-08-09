"""PT-03/PT-04: evaluate trained frozen-backbone heads on the 400-day window.

Loads trained head artifacts, scores the eval-region hidden cache rows, and
computes per-date RankIC / DA / MAE, plus moving-block bootstrap contrasts vs
the J0 greedy reference.  Heads are read-only on the frozen backbone, so the
base token logits are untouched (guardrail).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json  # noqa: E402
from posttrain_heads import (  # noqa: E402
    LinearReturnHead, MlpReturnHead, LinearDirectionHead, MlpDirectionHead,
    LinearRankHead, MlpRankHead, IndependentMLP, DeepSetsHead, ISABSetTransformer,
    C1FeatureHead,
)
from analyze_pt01 import daily_rank_ic, _contrast  # noqa: E402

HEAD_TYPES = {
    "P1_linear_return": LinearReturnHead,
    "P2_mlp_return": MlpReturnHead,
    "P3_linear_direction": LinearDirectionHead,
    "P4_mlp_direction": MlpDirectionHead,
    "P5_linear_rank_pairwise": LinearRankHead,
    "P6_mlp_rank_spearman": MlpRankHead,
    "P7_mlp_rank_pairwise": MlpRankHead,
    "S1_independent_mlp": IndependentMLP,
    "S2_deepsets": DeepSetsHead,
    "S3_isab": ISABSetTransformer,
}


def score_head(head, hidden, head_kind, device, batch=4096):
    """Score eval hidden rows; returns (score, direction_prob or None)."""
    out = np.empty(hidden.shape[0], dtype=np.float32)
    direction = None
    head = head.to(device).eval()
    with torch.no_grad():
        for s in range(0, hidden.shape[0], batch):
            hb = hidden[s:s + batch].to(device)
            if "direction" in head_kind:
                prob = head(hb).cpu().numpy()
                out[s:s + batch] = prob
                direction = out
            else:
                out[s:s + batch] = head(hb).cpu().numpy()
    return out, direction


def run_heads_eval(head_dir, eval_hidden_path, eval_c1_path=None, dense_min=3634,
                   out_path=None):
    hidden_data = np.load(eval_hidden_path, allow_pickle=True)
    hidden = torch.from_numpy(hidden_data["hidden"])
    dates = np.asarray(hidden_data["date_key"])
    true = np.asarray(hidden_data["true_logret"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_c1 = np.load(eval_c1_path, allow_pickle=True)["c1_feats"] if eval_c1_path else None

    # J0 reference (greedy) from the verified pt01 records
    pt01_path = Path(eval_hidden_path).with_name("pt01_records.npz")
    pt01 = np.load(pt01_path, allow_pickle=True) if pt01_path.exists() else None
    if pt01 is None:
        raise FileNotFoundError(f"need {pt01_path} for the J0 greedy reference")
    base_ic, _, base_mae, _, _ = daily_rank_ic(
        pt01["greedy_return"], true, dates, dense_min)

    results = {}
    for art in sorted(Path(head_dir).glob("head_*.pt")):
        name = art.stem[len("head_"):]
        factory = HEAD_TYPES.get(name)
        if factory is None:
            continue
        head = factory()
        payload = torch.load(art, map_location="cpu", weights_only=False)
        head.load_state_dict(payload["head_state"])
        head = head.to(device).eval()
        with torch.no_grad():
            if name.startswith("C1"):
                if eval_c1 is None:
                    print(f"[eval_heads] skip {name}: no eval c1 features")
                    continue
                score = np.empty(hidden.shape[0], dtype=np.float32)
                for s in range(0, hidden.shape[0], 8192):
                    e = min(s + 8192, hidden.shape[0])
                    fb = torch.from_numpy(eval_c1[s:e]).to(device)
                    score[s:e] = head(fb).cpu().numpy()
            else:
                score, _ = score_head(head, hidden, name, device)
        ic, da, mae, cnt, dense = daily_rank_ic(score, true, dates, dense_min)
        if "direction" in name:
            da = _da_from_series(score, true, dates, dense_min)
        c = _contrast(base_ic, ic, base_mae, mae)
        results[name] = {
            "avg_daily_rank_ic": float(np.nanmean(ic[dense])),
            "avg_da_per_date": float(np.nanmean(da[dense])),
            "avg_mae": float(np.nanmean(mae[dense])),
            "contrast_vs_J0": c,
            "artifact": str(art),
        }
        print(f"[eval_heads] {name}: RankIC={results[name]['avg_daily_rank_ic']:.5f} "
              f"delta={c['rank_ic_delta_vs_J0']:+.5f} robust={c['block_robust']}")
    summary = {"schema_version": "head-eval-v1", "dense_min": dense_min,
               "heads": results}
    if out_path:
        write_json(out_path, summary)
    return summary


def _da_from_series(score, true, dates, dense_min):
    uniq = np.unique(dates)
    da = np.full(len(uniq), np.nan)
    for i, d in enumerate(uniq):
        m = dates == d
        if m.sum() < dense_min:
            continue
        da[i] = np.mean((score[m] > 0.5) == (true[m] > 0))
    return da


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained PostTrain heads")
    ap.add_argument("--head_dir", default="server_runs/weights/06-posttrain/seed42")
    ap.add_argument("--eval_hidden", default="server_runs/weights/06-posttrain/seed42/hidden_cache.npz")
    ap.add_argument("--eval_c1", default=None)
    ap.add_argument("--dense_min", type=int, default=3634)
    args = ap.parse_args()
    roots = resolve_roots()
    out = roots.results_root / "pt03_pt04_head_eval.json"
    run_heads_eval(args.head_dir, args.eval_hidden, eval_c1_path=args.eval_c1,
                   dense_min=args.dense_min, out_path=out)


if __name__ == "__main__":
    main()
