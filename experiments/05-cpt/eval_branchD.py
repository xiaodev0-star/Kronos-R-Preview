"""Evaluate Branch D (MTP) checkpoints under the 400-window protocol.

Builds a trajectory trial dir for a Branch D checkpoint (or its per-epoch
snapshots) and runs the formal 400-window evaluation against the CPT baseline
(exp04b_8ceb_ep100.pt).  The evaluator reads only the main coarse head, so the
MTP future heads do not affect the main-head metrics; the MTP consistency
filter is evaluated separately by d_consistency_filter.py.

Usage:
    python experiments/05-cpt/eval_branchD.py \
        --ckpt checkpoints/branchD_mtp.pt \
        --tag mtp \
        --epochs 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

EVAL_SCRIPT = ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
REF_OVERRIDE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "override.json"
CPT_CACHE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "cache"


def build_index(ckpt: Path, tag: str, output_dir: Path, epochs: list[int]) -> Path:
    """Create a model_checkpoints.json for a Branch D run (or single ckpt)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REF_OVERRIDE, output_dir / "override.json")
    entries = []
    for ep in epochs:
        p = ckpt if ep == 0 else ckpt.parent / f"{ckpt.stem}_ep{ep}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        train_loss = payload.get("train_loss")
        val_loss = payload.get("val_loss")
        entries.append({
            "epoch": ep if ep > 0 else 1,
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
            "train_loss": train_loss if isinstance(train_loss, (int, float)) else 3.0,
            "val_loss": val_loss if isinstance(val_loss, (int, float)) else 3.6,
            "learning_rate": 0.0,
            "learning_rate_adam": 0.0,
            "optimizer_steps_this_epoch": 0,
            "global_step": 0,
            "best_so_far": False,
        })
    index = {
        "tag": tag,
        "save_path": str(ckpt.resolve()),
        "updated_epoch": max((e["epoch"] for e in entries), default=0),
        "checkpoints": entries,
    }
    index_path = output_dir / "model_checkpoints.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--epochs", type=str, default="",
                        help="Comma-separated per-epoch snapshots to include; "
                             "empty = evaluate the final checkpoint only")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--output_root", type=Path, default=None)
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    epochs = [0]
    if args.epochs.strip():
        epochs = [int(v) for v in args.epochs.split(",") if v.strip()]

    output_root = args.output_root or (
        ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
    )
    trial_dir = output_root / f"branchD_{args.tag}"
    build_index(args.ckpt, args.tag, trial_dir, epochs)

    eval_epochs = ",".join(str(e if e > 0 else 1) for e in epochs)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--trial_dir", str(trial_dir),
        "--tokenizer", str(args.tokenizer),
        "--output_dir", str(trial_dir / "epoch_trajectory"),
        "--epochs", eval_epochs,
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(CPT_CACHE),
        "--no_reference_check",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
