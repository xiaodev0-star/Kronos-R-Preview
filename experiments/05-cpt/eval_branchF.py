"""Evaluate Branch F (LoRA-per-regime) checkpoints under the 400-window protocol.

Branch F attaches one frozen-backbone LoRA adapter per market regime (trailing
20-day realized-vol terciles).  This runner drives the formal per-regime
evaluation:

  (a) each Branch F checkpoint (--ckpt / --epochs) through
      ``evaluate_epoch_trajectory.py`` with ``--lora`` + regime tagging;
  (b) optionally re-runs the dense CPT baseline (--baseline_ckpt) with the same
      regime tagging (NO ``--lora``) so candidate and reference both carry
      per-regime slices from a shared prepared-input cache;
  (c) optionally runs ``bootstrap_regime_compare.py`` for every regime x the
      headline metric keys (balance / DA / RankIC / MAPE / AmpRatio).

Usage:
    python experiments/05-cpt/eval_branchF.py \
        --ckpt checkpoints/branchF_r8.pt \
        --tag branchF_r8 \
        --epochs 2,3,4 \
        --also_baseline \
        --baseline_ckpt checkpoints/exp04b_8ceb_ep100.pt \
        --bootstrap
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
BOOTSTRAP_SCRIPT = ROOT / "experiments" / "05-cpt" / "bootstrap_regime_compare.py"
REF_OVERRIDE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "override.json"
CPT_CACHE = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "local_cpt" / "cache"
DEFAULT_BOOTSTRAP_KEYS = (
    "median_daily_codebook_balance_score",
    "avg_da_per_date",
    "avg_daily_rank_ic",
    "mape",
    "ampratio",
)


def build_index(ckpt: Path, tag: str, output_dir: Path, epochs: list[int]) -> Path:
    """Create a model_checkpoints.json for a Branch F run (or single ckpt)."""
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
        lora_r = payload.get("config", {}).get("lora_r", payload.get("lora_r"))
        lora_alpha = payload.get("config", {}).get(
            "lora_alpha", payload.get("lora_alpha")
        )
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
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
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


def run_evaluator(trial_dir: Path, tokenizer: Path, eval_epochs: str,
                  output_dir: Path, regime_window: int, regime_quantiles: str,
                  lora: bool, prepared_cache_dir: Path) -> int:
    """Invoke evaluate_epoch_trajectory.py for one trial dir."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--trial_dir", str(trial_dir),
        "--tokenizer", str(tokenizer),
        "--output_dir", str(output_dir),
        "--epochs", eval_epochs,
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(prepared_cache_dir),
        "--no_reference_check",
    ]
    if lora:
        cmd.append("--lora")
    if regime_window > 0:
        cmd += ["--regime_window", str(regime_window),
                "--regime_quantiles", regime_quantiles]
    print(f"Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def run_regime_bootstrap(epoch_file: Path, reference_file: Path,
                         regime: str, keys: tuple[str, ...]) -> int:
    """Run bootstrap_regime_compare.py for one regime over headline keys."""
    rc = 0
    for key in keys:
        cmd = [
            sys.executable, str(BOOTSTRAP_SCRIPT),
            "--candidate", str(epoch_file),
            "--reference", str(reference_file),
            "--key", key,
            "--regime", regime,
            "--bootstrap", "2000",
            "--seed", "42",
        ]
        print(f"\n=== regime {regime} key={key} ===", flush=True)
        print(f"Running: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd).returncode
        if code != 0:
            rc = code
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Branch F checkpoint (or the path whose _epN.pt "
                             "snapshots carry the epochs).")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--epochs", type=str, default="",
                        help="Comma-separated per-epoch snapshots; empty = final "
                             "checkpoint only.")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--regime_window", type=int, default=20)
    parser.add_argument("--regime_quantiles", type=str, default="0.333,0.667")
    parser.add_argument("--prepared_cache_dir", type=Path, default=CPT_CACHE)
    parser.add_argument("--also_baseline", action="store_true",
                        help="Re-run the dense baseline with regime tagging for "
                             "per-regime reference slices.")
    parser.add_argument("--baseline_ckpt", type=Path,
                        default=ROOT / "checkpoints" / "exp04b_8ceb_ep100.pt")
    parser.add_argument("--baseline_tag", type=str, default="dense_ref")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap_regime_compare for each regime after "
                             "evaluation (requires --baseline_ckpt output).")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    if args.regime_window <= 0:
        raise ValueError("Branch F evaluation requires --regime_window > 0")

    epochs = [0]
    if args.epochs.strip():
        epochs = [int(v) for v in args.epochs.split(",") if v.strip()]

    output_root = args.output_root or (
        ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
    )

    # --- (a) Branch F candidate with --lora + regime tagging ---
    trial_dir = output_root / f"branchF_{args.tag}"
    build_index(args.ckpt, args.tag, trial_dir, epochs)
    eval_epochs = ",".join(str(e if e > 0 else 1) for e in epochs)
    rc = run_evaluator(
        trial_dir, args.tokenizer, eval_epochs,
        trial_dir / "epoch_trajectory",
        args.regime_window, args.regime_quantiles, lora=True,
        prepared_cache_dir=args.prepared_cache_dir,
    )
    if rc != 0:
        return rc

    # --- (b) dense baseline re-run with regime tagging (shared cache) ---
    baseline_dir = None
    if args.also_baseline:
        if not args.baseline_ckpt.exists():
            raise FileNotFoundError(args.baseline_ckpt)
        baseline_dir = output_root / f"branchF_{args.baseline_tag}"
        build_index(args.baseline_ckpt, args.baseline_tag, baseline_dir, [0])
        rc = run_evaluator(
            baseline_dir, args.tokenizer, "1",
            baseline_dir / "epoch_trajectory",
            args.regime_window, args.regime_quantiles, lora=False,
            prepared_cache_dir=args.prepared_cache_dir,
        )
        if rc != 0:
            return rc

    # --- (c) per-regime paired bootstrap candidate vs dense baseline ---
    if args.bootstrap and baseline_dir is not None:
        reference_file = baseline_dir / "epoch_trajectory" / "epoch_001.json"
        if not reference_file.exists():
            raise FileNotFoundError(reference_file)
        for ep in epochs:
            indexed_epoch = ep if ep > 0 else 1
            epoch_file = trial_dir / "epoch_trajectory" / f"epoch_{indexed_epoch:03d}.json"
            if not epoch_file.exists():
                print(f"[warn] missing epoch JSON: {epoch_file}", flush=True)
                continue
            for regime in ("0", "1", "2"):
                code = run_regime_bootstrap(
                    epoch_file, reference_file, regime, DEFAULT_BOOTSTRAP_KEYS)
                if code != 0:
                    print(f"[warn] regime bootstrap failed (rc={code}) "
                          f"for ep{ep:03d} regime {regime}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
