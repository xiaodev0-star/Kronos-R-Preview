"""Evaluate Branch C checkpoints under the 400-window protocol.

Builds a trajectory trial dir for a Branch C checkpoint (or its per-epoch
snapshots) and runs the formal 400-window evaluation, then (optionally) emits
near (offset 300-399) / far (offset 0-299) paired-bootstrap summaries against a
CPT reference epoch JSON using bootstrap_compare.py --offset_min/--offset_max.

Usage:
    python experiments/05-cpt/eval_branchC.py \
        --ckpt checkpoints/branchC_combined.pt \
        --tag branchC_combined \
        --epochs 1,2,3,4,5,6 \
        --reference server_runs/results/04b-cpt/seed42/trials/local_cpt/epoch_trajectory/epoch_100.json
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
BOOTSTRAP_SCRIPT = ROOT / "experiments" / "05-cpt" / "bootstrap_compare.py"
REF_OVERRIDE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "override.json"
CPT_CACHE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "cache"
DEFAULT_BOOTSTRAP_KEY = "median_daily_codebook_balance_score"


def build_index(ckpt: Path, tag: str, output_dir: Path, epochs: list[int]) -> Path:
    """Create a model_checkpoints.json for a Branch C run (or single ckpt)."""
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


def bootstrap_summary(candidate_json: Path, reference_json: Path, key: str,
                      near: bool, label: str) -> int:
    """Run bootstrap_compare.py over the near (300-399) or far (0-299) slice."""
    cmd = [
        sys.executable, str(BOOTSTRAP_SCRIPT),
        "--candidate", str(candidate_json),
        "--reference", str(reference_json),
        "--key", key,
        "--bootstrap", "2000",
        "--seed", "42",
    ]
    if near:
        cmd += ["--offset_min", "300"]
        print(f"\n=== [{label}] NEAR (offset 300-399) ===", flush=True)
    else:
        cmd += ["--offset_max", "299"]
        print(f"\n=== [{label}] FAR (offset 0-299) ===", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


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
    parser.add_argument("--reference", type=Path, default=None,
                        help="CPT baseline epoch JSON; when given, also emit "
                             "near/far bootstrap summaries per epoch")
    parser.add_argument("--key", type=str, default=DEFAULT_BOOTSTRAP_KEY,
                        help="bootstrap metric key for the near/far summaries")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    epochs = [0]
    if args.epochs.strip():
        epochs = [int(v) for v in args.epochs.split(",") if v.strip()]

    output_root = args.output_root or (
        ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
    )
    trial_dir = output_root / f"branchC_{args.tag}"
    build_index(args.ckpt, args.tag, trial_dir, epochs)

    eval_epochs = ",".join(str(e if e > 0 else 1) for e in epochs)
    output_dir = trial_dir / "epoch_trajectory"
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--trial_dir", str(trial_dir),
        "--tokenizer", str(args.tokenizer),
        "--output_dir", str(output_dir),
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
    if result.returncode != 0:
        return result.returncode

    if args.reference is None:
        return 0
    if not args.reference.exists():
        raise FileNotFoundError(f"reference not found: {args.reference}")
    for ep in epochs:
        indexed_epoch = ep if ep > 0 else 1  # build_index maps final ckpt to epoch 1
        epoch_file = output_dir / f"epoch_{indexed_epoch:03d}.json"
        if not epoch_file.exists():
            print(f"[warn] missing epoch JSON: {epoch_file}", flush=True)
            continue
        label = f"ep{ep:03d}"
        for near in (True, False):
            rc = bootstrap_summary(epoch_file, args.reference, args.key,
                                   near=near, label=label)
            if rc != 0:
                print(f"[warn] bootstrap failed (rc={rc}) for {label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
