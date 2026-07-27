"""Evaluate one trained BSQ tokenizer on the held-out tokenizer features.

This is intentionally separate from the sweep orchestrator so each bit
configuration is loaded in a clean Python process.  It reports reconstruction
quality and code usage without touching GPT/test-window data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def distribution_metrics(counts: np.ndarray) -> dict[str, Any]:
    counts = counts.astype(np.int64, copy=False)
    total = int(counts.sum())
    used = counts[counts > 0]
    if total <= 0:
        return {
            "n_unique": 0,
            "collapse_rate": 0.0,
            "entropy_bits": 0.0,
            "effective_codes": 0.0,
        }
    probabilities = used.astype(np.float64) / total
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "n_unique": int(len(used)),
        "collapse_rate": float(used.max() / total),
        "entropy_bits": entropy_bits,
        "effective_codes": float(2.0 ** entropy_bits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenizer reconstruction and code-usage diagnostics"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_path = args.tokenizer.resolve()
    features_path = args.features.resolve()
    output_path = args.output.resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if not features_path.is_file():
        raise FileNotFoundError(features_path)
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from config import set_global_seed
    from model import load_tokenizer

    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(str(tokenizer_path), device)
    feature_archive = np.load(features_path, mmap_mode="r")
    features = feature_archive["features"]
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(f"Expected [N, 4] features, got {features.shape}")

    coarse_vocab = int(tokenizer.vocab_coarse)
    fine_vocab = int(tokenizer.bsq_fine.vocab_size)
    coarse_counts = np.zeros(coarse_vocab, dtype=np.int64)
    fine_counts = np.zeros(fine_vocab, dtype=np.int64)
    joint_counts = np.zeros(coarse_vocab * fine_vocab, dtype=np.int64)
    absolute_error = np.zeros(4, dtype=np.float64)
    squared_error = np.zeros(4, dtype=np.float64)
    n_rows = 0

    with torch.inference_mode():
        for start in range(0, len(features), args.chunk_size):
            batch_np = np.asarray(
                features[start : start + args.chunk_size], dtype=np.float32
            )
            batch = torch.from_numpy(batch_np).to(device)
            indices = tokenizer.encode_all(batch)
            if indices.ndim == 2:
                reconstruction = tokenizer.decode_all(
                    indices.unsqueeze(1)
                )[:, 0, :]
            else:
                reconstruction = tokenizer.decode_all(indices)
            error = reconstruction.float() - batch
            absolute_error += error.abs().sum(dim=0).cpu().numpy()
            squared_error += error.square().sum(dim=0).cpu().numpy()

            ids = indices.reshape(-1, 2).cpu().numpy().astype(
                np.int64, copy=False
            )
            coarse = ids[:, 0]
            fine = ids[:, 1]
            joint = coarse * fine_vocab + fine
            coarse_counts += np.bincount(
                coarse, minlength=coarse_vocab
            )[:coarse_vocab]
            fine_counts += np.bincount(
                fine, minlength=fine_vocab
            )[:fine_vocab]
            joint_counts += np.bincount(
                joint, minlength=coarse_vocab * fine_vocab
            )[: coarse_vocab * fine_vocab]
            n_rows += len(batch_np)

    checkpoint = torch.load(
        tokenizer_path, map_location="cpu", weights_only=False
    )
    per_feature_mae = absolute_error / max(n_rows, 1)
    per_feature_rmse = np.sqrt(squared_error / max(n_rows, 1))
    coarse_metrics = distribution_metrics(coarse_counts)
    fine_metrics = distribution_metrics(fine_counts)
    joint_metrics = distribution_metrics(joint_counts)
    coarse_metrics["utilization"] = coarse_metrics["n_unique"] / coarse_vocab
    fine_metrics["utilization"] = fine_metrics["n_unique"] / fine_vocab
    joint_metrics["utilization"] = (
        joint_metrics["n_unique"] / (coarse_vocab * fine_vocab)
    )

    payload = {
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "features": str(features_path),
        "features_sha256": file_sha256(features_path),
        "n_rows": n_rows,
        "bits_l1": int(tokenizer.bits_l1),
        "bits_l2": int(tokenizer.bits_l2),
        "coarse_vocab": coarse_vocab,
        "fine_vocab": fine_vocab,
        "joint_vocab": coarse_vocab * fine_vocab,
        "checkpoint_best_val_loss": float(
            checkpoint.get("best_val_loss", math.nan)
        ),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", -1)) + 1,
        "mae": float(per_feature_mae.mean()),
        "rmse": float(np.sqrt(squared_error.sum() / max(n_rows * 4, 1))),
        "per_feature_mae": {
            name: float(value)
            for name, value in zip(
                ("log_ret", "log_high", "log_low", "log_open"),
                per_feature_mae,
            )
        },
        "per_feature_rmse": {
            name: float(value)
            for name, value in zip(
                ("log_ret", "log_high", "log_low", "log_open"),
                per_feature_rmse,
            )
        },
        "coarse_codes": coarse_metrics,
        "fine_codes": fine_metrics,
        "joint_codes": joint_metrics,
    }
    atomic_write_json(output_path, payload)
    print(
        f"Tokenizer {tokenizer.bits_l1}+{tokenizer.bits_l2}: "
        f"MAE={payload['mae']:.6f}, "
        f"joint_unique={joint_metrics['n_unique']}, "
        f"joint_entropy={joint_metrics['entropy_bits']:.3f} bits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
