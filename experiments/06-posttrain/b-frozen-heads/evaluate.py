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
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import resolve_roots, artifact_paths, write_json  # noqa: E402
from common import (  # noqa: E402
    LinearReturnHead, MlpReturnHead, LinearDirectionHead, MlpDirectionHead,
    LinearRankHead, MlpRankHead, IndependentMLP, DeepSetsHead, ISABSetTransformer,
    C1FeatureHead,
)
from common import daily_rank_ic, _contrast  # noqa: E402

HEAD_TYPES = {
    "return-linear": LinearReturnHead,
    "return-mlp": MlpReturnHead,
    "direction-linear": LinearDirectionHead,
    "direction-mlp": MlpDirectionHead,
    "rank-linear-pairwise": LinearRankHead,
    "rank-mlp-spearman": MlpRankHead,
    "rank-mlp-pairwise": MlpRankHead,
    "feature-only": C1FeatureHead,
    "independent-mlp": IndependentMLP,
    "deepsets": DeepSetsHead,
    "isab": ISABSetTransformer,
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
                   out_path=None, preds_path=None):
    hidden_data = np.load(eval_hidden_path, allow_pickle=True)
    hidden = torch.from_numpy(hidden_data["hidden"])
    dates = np.asarray(hidden_data["date_key"])
    symbols = np.asarray(hidden_data["symbol"])
    true = np.asarray(hidden_data["true_logret"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_c1 = np.load(eval_c1_path, allow_pickle=True)["c1_feats"] if eval_c1_path else None

    # J0 reference (greedy) from the verified pt01 records
    pt01_path = artifact_paths()["records"]
    pt01 = np.load(pt01_path, allow_pickle=True) if pt01_path.exists() else None
    if pt01 is None:
        raise FileNotFoundError(f"need {pt01_path} for the J0 greedy reference")
    base_ic, _, base_mae, _, _ = daily_rank_ic(
        pt01["greedy_return"], true, dates, dense_min)

    results = {}
    pred_parts = []  # (name, per-stock score)
    for art in sorted(Path(head_dir).glob("*.pt")):
        name = art.stem
        factory = HEAD_TYPES.get(name)
        if factory is None:
            continue
        head = factory()
        payload = torch.load(art, map_location="cpu", weights_only=False)
        head.load_state_dict(payload["head_state"])
        head = head.to(device).eval()
        with torch.no_grad():
            if name == "feature-only":
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
        pred_parts.append((name, score.astype(np.float32)))
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

    if preds_path and pred_parts:
        _write_head_parquet(pred_parts, dates, symbols, true, Path(preds_path))
    return summary


def _write_head_parquet(pred_parts, dates, symbols, true, path):
    """Write per-stock per-head predictions as one long Parquet table."""
    names = np.concatenate([np.full(len(s), n, dtype=object) for n, s in pred_parts])
    scores = np.concatenate([s for _, s in pred_parts])
    n_heads = len(pred_parts)
    table = pa.Table.from_arrays([
        pa.array(names, type=pa.string()),
        pa.array(np.tile(dates, n_heads), type=pa.string()),
        pa.array(np.tile(symbols, n_heads), type=pa.string()),
        pa.array(scores, type=pa.float32()),
        pa.array(np.tile(true.astype(np.float32), n_heads), type=pa.float32()),
    ], names=["head", "date", "symbol", "score", "true_logret"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"[eval_heads] wrote {path} ({table.num_rows/1e6:.1f}M rows)", flush=True)


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
    ap.add_argument("--head_dir", default=None)
    ap.add_argument("--eval_hidden", default=None)
    ap.add_argument("--eval_c1", default=None)
    ap.add_argument("--dense_min", type=int, default=3634)
    ap.add_argument("--preds", default=None,
                    help="Parquet path for per-stock per-head predictions")
    args = ap.parse_args()
    roots = resolve_roots()
    paths = artifact_paths(roots=roots)
    head_dir = args.head_dir or str(paths["head_dir"])
    eval_hidden = args.eval_hidden or str(paths["hidden"])
    eval_c1 = args.eval_c1 or str(paths["eval_features"])
    out = roots.results_root / "B-heads" / "evaluation.json"
    preds = (Path(args.preds) if args.preds else
             roots.results_root / "B-heads" / "head-predictions.parquet")
    run_heads_eval(head_dir, eval_hidden, eval_c1_path=eval_c1,
                   dense_min=args.dense_min, out_path=out, preds_path=preds)


if __name__ == "__main__":
    main()
