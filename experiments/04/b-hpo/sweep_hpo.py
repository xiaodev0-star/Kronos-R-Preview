r"""Exp 04-B: one-stage, full-data GPT hyperparameter search (Muon recipe).

The study deliberately uses the final training distribution for every trial.
There is no small-stock screening phase.  The baseline is always trial 1, the
search seed and training seed are fixed at 42, and the final holdout is only
opened by the explicit ``--holdout`` command after all planned trials finish.

The 2026-07-28 revision follows the Exp 04-A data-driven selection: the fixed
optimizer is Muon (2D weights) + AdamW (1D/embeddings) and the primary search
axis is ``lr_muon``. The default epoch budget is 50 so each trial's full
trajectory feeds the all-epoch token-quality ranking (matching Exp 03/04-A);
the cosine schedule rescales with the epoch budget, so a 50-epoch run anneals
completely.

Normal search (reads reviewed Exp 02/03/04-A selections from server_runs/):
    python experiments/04/b-hpo/sweep_hpo.py

Server run without server_runs/ (portable bundle carries the tokenizer,
the Exp 03 architecture, and the Exp 04-A muon selection; output still goes
to KRONOS_WEIGHTS_ROOT/KRONOS_RESULTS_ROOT, defaulting to server_runs/):
    python experiments/04/b-hpo/sweep_hpo.py --portable

Temporary end-to-end smoke test:
    python experiments/04/b-hpo/sweep_hpo.py --smoke
    python experiments/04/b-hpo/sweep_hpo.py --smoke --portable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
EXP04_DIR = EXP_DIR.parent
if str(EXP04_DIR) not in sys.path:
    sys.path.insert(0, str(EXP04_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ablation_common as AB_COMMON
from experiment_io import (
    StudyLayout,
    default_study_roots,
    runtime_environment,
    write_download_manifest,
)

TRAIN_SCRIPT = ROOT / "train_base.py"
EVAL_SCRIPT = ROOT / "eval.py"
TRAJECTORY_SCRIPT = EXP_DIR / "evaluate_epoch_trajectory.py"
DEFAULT_WEIGHTS_ROOT, DEFAULT_RESULTS_ROOT = default_study_roots(
    "04b-hpo", seed=42
)
_, EXP03_RESULTS_ROOT = default_study_roots("03-gpt-scaling", seed=42)
_, EXP04A_RESULTS_ROOT = default_study_roots(
    "04a-optimizer-ablation", seed=42
)
DEFAULT_ARCH_SELECTION = EXP03_RESULTS_ROOT / "selection.json"
EXP04A_SELECTION = EXP04A_RESULTS_ROOT / "selection.json"
SEED = 42
HOLDOUT_OFFSET = 400
EVAL_DAYS = 1
# Full-coverage, single-day-resolution screening: tile the pre-holdout region
# [0, HOLDOUT_OFFSET) with contiguous EVAL_DAYS-day windows (offsets
# 0, 1, ..., 399) so every pre-holdout trading day is scored as its own window.
DEFAULT_VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, EVAL_DAYS))
# The final sealed holdout is a separate, once-only gate at offset
# HOLDOUT_OFFSET and is intentionally a single fixed window.
HOLDOUT_DAYS = 80
DEFAULT_TIME_BUDGET_HOURS = 10.0
REFERENCE_EPOCHS = 50
# Measured Exp 04-A deep-architecture cost (2026-07-28): training ~96 s/epoch
# and trajectory evaluation ~40 s/epoch, normalized to the 50-epoch reference.
ESTIMATED_TRAIN_MINUTES = 82.0
ESTIMATED_VALIDATION_MINUTES = 35.0
BUDGET_RESERVE_MINUTES = 30.0
IMPLEMENTATION_FILES = (
    "experiment_io.py",
    "train_base.py",
    "eval.py",
    "eval_helpers.py",
    "config.py",
    "data_processor.py",
    "model/layers.py",
    "model/kronos_preview.py",
    "model/tokenizer.py",
    "experiments/04/b-hpo/sweep_hpo.py",
    "experiments/04/b-hpo/evaluate_epoch_trajectory.py",
    "experiments/04/ablation_common.py",
)

# Exp 04-A selected Muon (CE was fixed without a separate loss ablation).  B
# uses a deterministic, interpretable sequence of full-data trials rather than
# a confounded random draw from a large grid.  lr_muon is the primary axis; the
# AdamW-side lr only covers embeddings/1D parameters under Muon.
SEARCH_SPACE: dict[str, list[float]] = {
    "lr_muon": [0.005, 0.01, 0.02, 0.03, 0.04],
    "lr": [1e-4, 3e-4, 6e-4],
    "dropout": [0.0, 0.05, 0.10, 0.15, 0.20],
    "weight_decay": [0.001, 0.01, 0.05, 0.10],
    "fine_weight": [0.0, 0.10, 0.20, 0.30, 0.50],
    "het_weight": [0.0, 0.05, 0.10, 0.20],
    "warmup_ratio": [0.02, 0.05, 0.10],
    "label_smoothing": [0.0, 0.03, 0.05, 0.10],
}

BASELINE: dict[str, float] = {
    "lr": 3e-4,
    "lr_muon": 0.02,
    "dropout": 0.10,
    "weight_decay": 0.01,
    "fine_weight": 0.30,
    "het_weight": 0.10,
    "warmup_ratio": 0.05,
    "label_smoothing": 0.0,
}


def _candidate(**overrides: float) -> dict[str, float]:
    params = dict(BASELINE)
    params.update(overrides)
    return params


# Prefixes are meaningful: the default budget covers the lr_muon screen, the
# AdamW-side lr probes, then the regularization probes at the baseline
# lr_muon.  Every trial remains full-data/full-seq.
CANDIDATE_PLAN: tuple[dict[str, float], ...] = (
    dict(BASELINE),
    _candidate(lr_muon=0.01),
    _candidate(lr_muon=0.04),
    _candidate(lr_muon=0.005),
    _candidate(lr_muon=0.03),
    _candidate(lr=1e-4),
    _candidate(lr=6e-4),
    _candidate(dropout=0.05),
    _candidate(dropout=0.20),
    _candidate(weight_decay=0.05),
    _candidate(label_smoothing=0.05),
    _candidate(fine_weight=0.10),
    _candidate(het_weight=0.0),
    _candidate(warmup_ratio=0.10),
    _candidate(dropout=0.0),
    _candidate(weight_decay=0.001),
    # Extended probes (trials 17-20) for the server run at epochs=50: each
    # pushes one regularizer further from the baseline so the full-trajectory
    # token-quality ranking sees the edges of the grid, not just its centre.
    _candidate(label_smoothing=0.10),
    _candidate(warmup_ratio=0.02),
    _candidate(fine_weight=0.50),
    _candidate(het_weight=0.20),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_runtime() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Exp 04-B requires Python >=3.10, got {sys.version.split()[0]} "
            f"at {sys.executable}"
        )
    if (
        not TRAIN_SCRIPT.is_file()
        or not EVAL_SCRIPT.is_file()
        or not TRAJECTORY_SCRIPT.is_file()
    ):
        raise FileNotFoundError(f"Could not resolve repository root from {__file__}")


def parse_offsets(text: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offsets must be comma-separated integers") from exc
    if not offsets or any(offset < 0 for offset in offsets):
        raise argparse.ArgumentTypeError("offsets must contain non-negative integers")
    if len(set(offsets)) != len(offsets):
        raise argparse.ArgumentTypeError("offsets must be unique")
    return offsets


def estimated_trial_minutes(epochs: int) -> float:
    train_minutes = ESTIMATED_TRAIN_MINUTES * epochs / REFERENCE_EPOCHS
    validation_minutes = ESTIMATED_VALIDATION_MINUTES * epochs / REFERENCE_EPOCHS
    return train_minutes + validation_minutes


def recommended_trial_count(time_budget_hours: float, epochs: int = 50) -> int:
    available_minutes = time_budget_hours * 60.0 - BUDGET_RESERVE_MINUTES
    return min(
        len(CANDIDATE_PLAN),
        max(1, int(available_minutes // estimated_trial_minutes(epochs))),
    )


def trial_id(params: dict[str, float]) -> str:
    if params == BASELINE:
        return "baseline"
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return "trial_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def build_trial_plan(n_trials: int) -> list[dict[str, Any]]:
    if not 1 <= n_trials <= len(CANDIDATE_PLAN):
        raise ValueError(
            f"n_trials must be in [1, {len(CANDIDATE_PLAN)}], got {n_trials}"
        )
    selected = [dict(params) for params in CANDIDATE_PLAN[:n_trials]]

    return [
        {
            "order": index + 1,
            "tid": trial_id(params),
            "is_baseline": params == BASELINE,
            "params": params,
        }
        for index, params in enumerate(selected)
    ]


def study_settings(
    *,
    n_trials: int,
    epochs: int,
    tokenizer: dict[str, Any],
    architecture: dict[str, Any],
    upstream_selections: dict[str, Any],
    validation_offsets: tuple[int, ...],
    eval_days: int,
    batch_tokens: int,
    accumulation_steps: int,
    eval_batch_size: int,
    max_collapse_rate: float,
    min_unique_tokens: int,
    time_budget_hours: float | None,
    trial_count_source: str,
    smoke: bool = False,
) -> dict[str, Any]:
    train_minutes = ESTIMATED_TRAIN_MINUTES * epochs / REFERENCE_EPOCHS
    validation_minutes = ESTIMATED_VALIDATION_MINUTES * epochs / REFERENCE_EPOCHS
    trial_minutes = train_minutes + validation_minutes
    effective_batch_tokens = (
        2048 if smoke else batch_tokens or int(architecture["batch_tokens"])
    )
    effective_accumulation = 8 if smoke else accumulation_steps
    effective_eval_batch_size = (
        1 if smoke else eval_batch_size or int(architecture["eval_batch_size"])
    )
    return {
        "seed": SEED,
        "n_trials": n_trials,
        "epochs": epochs,
        "tokenizer": tokenizer,
        "architecture": architecture,
        "upstream_selections": upstream_selections,
        "dataset": AB_COMMON.PIPE.dataset_signature(),
        "implementation_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in IMPLEMENTATION_FILES
        },
        "loss": "ce",
        "optimizer": "muon",
        "max_stocks": 24 if smoke else 0,
        "max_seq_len": 256 if smoke else 0,
        "batch_tokens": effective_batch_tokens,
        "accumulation_steps": effective_accumulation,
        "constant_accumulation": True,
        "controlled_loader_seed": SEED,
        "exact_accumulation_boundaries": True,
        "eval_n_stocks": 6 if smoke else 0,
        "eval_sample_strategy": "shortest" if smoke else "random",
        "eval_days": 2 if smoke else eval_days,
        "validation_offsets": [3] if smoke else list(validation_offsets),
        "eval_batch_size": effective_eval_batch_size,
        "max_collapse_rate": 1.0 if smoke else max_collapse_rate,
        "min_unique_tokens": 1 if smoke else min_unique_tokens,
        "holdout_offset": HOLDOUT_OFFSET,
        "holdout_days": HOLDOUT_DAYS,
        "time_budget_hours": time_budget_hours,
        "trial_count_source": trial_count_source,
        "runtime_estimate": {
            "reference_epochs": REFERENCE_EPOCHS,
            "train_minutes_per_trial": train_minutes,
            "validation_minutes_per_trial": validation_minutes,
            "total_minutes_per_trial": trial_minutes,
            "fixed_reserve_minutes": BUDGET_RESERVE_MINUTES,
            "estimated_total_minutes": (
                n_trials * trial_minutes + BUDGET_RESERVE_MINUTES
            ),
        },
        "retain_epoch_checkpoints": True,
        "evaluate_every_epoch": True,
        "mature_window_epochs": epochs,
        "mature_window_policy": (
            "Full epoch trajectory as a single window; the maturity onset and "
            "1%-of-best validation-loss basin are recorded for audit but no "
            "longer restrict which epochs enter the ranking."
        ),
        "smoke": smoke,
    }


def study_fingerprint(settings: dict[str, Any], plan: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {"settings": settings, "plan": plan},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_manifest(
    results_root: Path,
    settings: dict[str, Any],
    plan: list[dict[str, Any]],
    layout: StudyLayout,
) -> dict[str, Any]:
    fingerprint = study_fingerprint(settings, plan)
    manifest_path = results_root / "study_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("study_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{manifest_path} belongs to a different study definition. "
                "Archive that study before changing its budget or search plan."
            )
        return manifest

    manifest = {
        "experiment": "Exp 04-B",
        "design": "one-stage full-data HPO",
        "status": "planned",
        "created_at_utc": utc_now(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "study_fingerprint": fingerprint,
        "settings": settings,
        "search_space": SEARCH_SPACE,
        "baseline": BASELINE,
        "plan": plan,
        "artifact_layout": layout.metadata(),
        "runtime_environment": runtime_environment(ROOT),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def trial_paths(
    results_root: Path,
    tid: str,
    weights_root: Path | None = None,
) -> dict[str, Path]:
    if weights_root is None:
        manifest_path = results_root / "study_manifest.json"
        manifest = load_json(manifest_path)
        weights_root = Path(manifest["artifact_layout"]["weights_root"])
    results = results_root / "trials" / tid
    weights = weights_root / "trials" / tid
    return {
        "directory": results,
        "weights_directory": weights,
        "params": results / "params.json",
        "override": results / "override.json",
        "model": weights / "model.pt",
        "resume": weights / "model.pt.ckpt",
        "checkpoint_index": results / "model_checkpoints.json",
        "result": results / "result.json",
        "history": results / f"history_exp04b_{tid}.json",
        "trajectory": results / "epoch_trajectory",
        "gpt_log": results / "gpt.log",
        "trajectory_log": results / "trajectory.log",
        "prepared_cache": weights_root / "shared" / "prepared_eval",
    }


def runtime_override(
    params: dict[str, float],
    *,
    settings: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    tokenizer = settings["tokenizer"]
    architecture = settings["architecture"]
    return {
        "DataConfig": {
            "random_seed": SEED,
            "max_stocks": settings["max_stocks"],
        },
        "TokenizerConfig": {
            "hidden_dim": tokenizer["hidden_dim"],
            "embedding_dim": tokenizer["embedding_dim"],
            "bits_l1": tokenizer["bits_l1"],
            "bits_l2": tokenizer["bits_l2"],
            "bits_per_quantizer": 0,
            "random_seed": SEED,
        },
        "ModelConfig": {
            "dim": architecture["dim"],
            "depth": architecture["depth"],
            "heads": architecture["heads"],
            "num_kv_heads": architecture["kv_heads"],
            "ffn_multiplier": architecture["ffn_multiplier"],
        },
        "TrainingConfig": {
            "random_seed": SEED,
            "warmup_ratio": params["warmup_ratio"],
            "batch_size": 1,
            "accumulation_steps": settings["accumulation_steps"],
            "use_gradient_checkpointing": architecture[
                "gradient_checkpointing"
            ],
            "save_dir": str(cache_root.resolve()),
        },
    }


def build_train_command(
    params: dict[str, float],
    paths: dict[str, Path],
    settings: dict[str, Any],
    tid: str,
) -> list[str]:
    architecture = settings["architecture"]
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--save_path", str(paths["model"]),
        "--tokenizer_path", str(settings["tokenizer"]["path"]),
        "--epochs", str(settings["epochs"]),
        "--tag", f"exp04b_{tid}",
        "--loss", "ce",
        "--gamma", "0",
        "--optimizer", "muon",
        "--lr", str(params["lr"]),
        "--lr_muon", str(params["lr_muon"]),
        "--dropout", str(params["dropout"]),
        "--weight_decay", str(params["weight_decay"]),
        "--fine_weight", str(params["fine_weight"]),
        "--heteroscedastic",
        "--het_weight", str(params["het_weight"]),
        "--label_smoothing", str(params["label_smoothing"]),
        "--entropy_alpha", "0",
        "--max_stocks", str(settings["max_stocks"]),
        "--max_seq_len", str(settings["max_seq_len"]),
        "--batch_tokens", str(settings["batch_tokens"]),
        "--batch_cap", "64",
        "--early_stop_patience", "0",
        "--history_per_epoch",
        "--metrics_dir", str(paths["directory"]),
        "--constant_accumulation",
        "--controlled_loader_seed",
        str(settings["controlled_loader_seed"]),
        "--exact_accumulation_boundaries",
        "--dim", str(architecture["dim"]),
        "--depth", str(architecture["depth"]),
        "--heads", str(architecture["heads"]),
        "--num_kv_heads", str(architecture["kv_heads"]),
        "--ffn_multiplier", str(architecture["ffn_multiplier"]),
    ]
    if architecture["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    return command


def build_eval_command(
    *,
    model: Path,
    tokenizer: Path,
    output: Path,
    n_stocks: int,
    n_days: int,
    start_offset: int,
    batch_size: int,
    sample_strategy: str,
) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "windowed",
        "--gpt_ckpt", str(model),
        "--tokenizer", str(tokenizer),
        "--output", str(output),
        "--seed", str(SEED),
        "--n_stocks", str(n_stocks),
        "--n_days", str(n_days),
        "--start_offset", str(start_offset),
        "--batch_size", str(batch_size),
        "--sample_strategy", sample_strategy,
        "--include_per_date",
    ]


def run_command(
    command: list[str],
    env: dict[str, str],
    *,
    log_path: Path | None = None,
    label: str = "Exp 04-B subprocess",
) -> None:
    if log_path is not None:
        AB_COMMON.PIPE.run_command(
            command,
            env=env,
            log_path=log_path,
            label=label,
        )
        return
    rendered = (
        subprocess.list2cmdline(command)
        if os.name == "nt"
        else shlex.join(command)
    )
    print("\n$ " + rendered, flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{rendered}"
        )


def checkpoint_metadata(model_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return {
        "completed": bool(checkpoint.get("completed", False)),
        "best_val_loss": float(checkpoint.get("val_loss", math.nan)),
        "best_epoch": int(checkpoint.get("epoch", -1)) + 1,
    }


def cached_eval_matches(
    payload: dict[str, Any],
    *,
    n_stocks: int,
    n_days: int,
    start_offset: int,
    sample_strategy: str,
) -> bool:
    return (
        payload.get("mode") == "windowed"
        and payload.get("seed") == SEED
        and payload.get("n_stocks_requested") == n_stocks
        and payload.get("n_days_requested") == n_days
        and payload.get("start_offset") == start_offset
        and payload.get("sample_strategy") == sample_strategy
        and payload.get("n_predictions", 0) > 0
        and payload.get("n_dates", 0) > 0
        and isinstance(payload.get("per_date"), dict)
        and "max_daily_collapse_rate" in payload
        and "min_daily_unique_tokens" in payload
        and payload.get("min_date_coverage_ratio") == 0.8
    )


def evaluate_window(
    *,
    model: Path,
    tokenizer: Path,
    output: Path,
    override_path: Path,
    n_stocks: int,
    n_days: int,
    start_offset: int,
    batch_size: int,
    sample_strategy: str,
) -> dict[str, Any]:
    if output.exists():
        cached = load_json(output)
        if cached_eval_matches(
            cached,
            n_stocks=n_stocks,
            n_days=n_days,
            start_offset=start_offset,
            sample_strategy=sample_strategy,
        ):
            print(f"  Reusing {output.name}")
            return cached

    env = os.environ.copy()
    env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(override_path)
    command = build_eval_command(
        model=model,
        tokenizer=tokenizer,
        output=output,
        n_stocks=n_stocks,
        n_days=n_days,
        start_offset=start_offset,
        batch_size=batch_size,
        sample_strategy=sample_strategy,
    )
    run_command(command, env)
    payload = load_json(output)
    if not cached_eval_matches(
        payload,
        n_stocks=n_stocks,
        n_days=n_days,
        start_offset=start_offset,
        sample_strategy=sample_strategy,
    ):
        raise RuntimeError(f"Evaluation output is incomplete: {output}")
    return payload


def aggregate_windows(
    windows: list[dict[str, Any]],
    *,
    max_collapse_rate: float,
    min_unique_tokens: int,
) -> dict[str, Any]:
    if not windows:
        raise ValueError("No validation windows were evaluated")
    weights = [max(int(window.get("n_dates", 0)), 0) for window in windows]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise RuntimeError("Validation produced zero usable dates")

    def weighted_mean(field: str) -> float:
        return float(
            sum(float(window.get(field, 0.0)) * weight
                for window, weight in zip(windows, weights))
            / total_weight
        )

    collapse = max(
        float(window.get("max_daily_collapse_rate",
                         window.get("collapse_rate", 1.0)))
        for window in windows
    )
    unique = min(
        int(window.get("min_daily_unique_tokens",
                       window.get("n_unique_tokens", 0)))
        for window in windows
    )
    amp_ratio = weighted_mean("ampratio")
    amp_log_error = abs(math.log(amp_ratio)) if amp_ratio > 0 else None
    healthy = (
        all(int(window.get("n_predictions", 0)) > 0 for window in windows)
        and collapse <= max_collapse_rate
        and unique >= min_unique_tokens
    )
    return {
        "healthy": healthy,
        "avg_da_per_date": weighted_mean("avg_da_per_date"),
        "avg_da_above_baseline": weighted_mean("avg_da_above_baseline"),
        "avg_daily_rank_ic": weighted_mean("avg_daily_rank_ic"),
        "pooled_rank_ic": weighted_mean("rank_ic"),
        "avg_ampratio": amp_ratio,
        "ampratio_log_error": amp_log_error,
        "avg_mape": weighted_mean("mape"),
        "avg_baseline_mape": weighted_mean("baseline_mape"),
        "worst_collapse_rate": collapse,
        "min_unique_tokens": unique,
        "n_windows": len(windows),
        "n_dates": sum(int(window.get("n_dates", 0)) for window in windows),
        "n_predictions": sum(int(window.get("n_predictions", 0)) for window in windows),
        "health_gate": {
            "max_collapse_rate": max_collapse_rate,
            "min_unique_tokens": min_unique_tokens,
        },
    }


def run_epoch_trajectory(
    *,
    paths: dict[str, Path],
    settings: dict[str, Any],
    env: dict[str, str],
    tid: str,
) -> list[dict[str, Any]]:
    if not AB_COMMON.PIPE.trajectory_is_complete(paths, settings["epochs"]):
        command = [
            sys.executable,
            str(TRAJECTORY_SCRIPT),
            "--trial_dir",
            str(paths["directory"]),
            "--tokenizer",
            str(settings["tokenizer"]["path"]),
            "--output_dir",
            str(paths["trajectory"]),
            "--prepared_cache_dir",
            str(paths["prepared_cache"].resolve()),
            "--epochs",
            f"1-{settings['epochs']}",
            "--offsets",
            ",".join(str(value) for value in settings["validation_offsets"]),
            "--n_days",
            str(settings["eval_days"]),
            "--n_stocks",
            str(settings["eval_n_stocks"]),
            "--batch_size",
            str(settings["eval_batch_size"]),
            "--seed",
            str(SEED),
            "--sample_strategy",
            settings["eval_sample_strategy"],
            "--max_collapse_rate",
            str(settings["max_collapse_rate"]),
            "--min_unique_tokens",
            str(settings["min_unique_tokens"]),
            "--no_reference_check",
            "--experiment_label",
            f"Exp 04-B HPO · {tid}",
        ]
        AB_COMMON.PIPE.run_command(
            command,
            env=env,
            log_path=paths["trajectory_log"],
            label=f"Exp 04-B {tid} epoch trajectory",
        )
    if not AB_COMMON.PIPE.trajectory_is_complete(paths, settings["epochs"]):
        raise RuntimeError(f"Trajectory completion check failed for {tid}")
    summary_path = paths["trajectory"] / "epoch_summary.json"
    rows = load_json(summary_path)
    if len(rows) != settings["epochs"]:
        raise RuntimeError(
            f"Expected {settings['epochs']} trajectory rows for {tid}, "
            f"found {len(rows)}"
        )
    return sorted(rows, key=lambda row: int(row["epoch"]))


def aggregate_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No trajectory rows")
    # select_mature_window now returns the full epoch trajectory as a single
    # window, so the aggregates below summarize the whole training run instead
    # of a narrow mature plateau. Output field names still carry the legacy
    # "mature_" prefix to avoid breaking downstream consumers, but the values
    # are full-trajectory statistics.
    window_rows, window_metadata = AB_COMMON.select_mature_window(rows)
    epoch_count = len(window_rows)

    def median(field: str) -> float:
        values = sorted(float(row[field]) for row in window_rows)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    def optional_median(field: str) -> float | None:
        values = sorted(
            float(row[field])
            for row in window_rows
            if row.get(field) is not None
        )
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    best_da = max(rows, key=lambda row: float(row["avg_da_per_date"]))
    best_rankic = max(rows, key=lambda row: float(row["avg_daily_rank_ic"]))
    best_mape = min(rows, key=lambda row: float(row["avg_mape"]))
    healthy_epochs = sum(bool(row["healthy"]) for row in window_rows)
    amp_ratio = median("avg_ampratio")
    return {
        # A legal HPO candidate must remain healthy across the full trajectory;
        # a one-epoch pass is not enough to open the holdout.
        "healthy": healthy_epochs == epoch_count,
        "healthy_epochs_in_mature_window": healthy_epochs,
        "mature_window_epochs": window_metadata["window_epochs"],
        "representative_epoch": int(window_rows[len(window_rows) // 2]["epoch"]),
        "maturity_onset_epoch": window_metadata["maturity_onset_epoch"],
        "minimum_val_loss": window_metadata["minimum_val_loss"],
        "near_best_loss_ceiling": window_metadata[
            "near_best_loss_ceiling"
        ],
        "avg_da_per_date": median("avg_da_per_date"),
        "avg_da_above_baseline": median("avg_da_above_baseline"),
        "avg_daily_rank_ic": median("avg_daily_rank_ic"),
        "avg_mape": median("avg_mape"),
        "avg_baseline_mape": median("avg_baseline_mape"),
        "avg_ampratio": amp_ratio,
        "ampratio_log_error": abs(math.log(amp_ratio)) if amp_ratio > 0 else None,
        "median_daily_collapse_rate": median("median_daily_collapse_rate"),
        "p90_daily_collapse_rate": median("p90_daily_collapse_rate"),
        "worst_collapse_rate": median("worst_daily_collapse_rate"),
        "median_daily_unique_tokens": median("median_daily_unique_tokens"),
        "min_unique_tokens": int(round(median("min_daily_unique_tokens"))),
        "median_daily_codebook_balance_score": optional_median(
            "median_daily_codebook_balance_score"
        ),
        "p10_daily_codebook_balance_score": optional_median(
            "p10_daily_codebook_balance_score"
        ),
        "median_daily_token_support_f1": optional_median(
            "median_daily_token_support_f1"
        ),
        "median_daily_target_support_recall": optional_median(
            "median_daily_target_support_recall"
        ),
        "median_daily_effective_token_alignment": optional_median(
            "median_daily_effective_token_alignment"
        ),
        "median_daily_token_jsd": optional_median(
            "median_daily_token_jsd"
        ),
        "best_da": float(best_da["avg_da_per_date"]),
        "best_da_epoch": int(best_da["epoch"]),
        "best_rankic": float(best_rankic["avg_daily_rank_ic"]),
        "best_rankic_epoch": int(best_rankic["epoch"]),
        "best_mape": float(best_mape["avg_mape"]),
        "best_mape_epoch": int(best_mape["epoch"]),
        "final_epoch": int(rows[-1]["epoch"]),
        "n_evaluated_epochs": len(rows),
    }


def result_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    # Token-balance-dominant ordering, aligned with Exp 03's selection
    # philosophy. Downstream DA/RankIC/MAPE/AmpRatio are demoted to
    # guardrails inspected by the caller, not tie-breakers here.
    aggregate = result["validation"]["aggregate"]
    codebook_balance = aggregate.get("median_daily_codebook_balance_score")
    p10_balance = aggregate.get("p10_daily_codebook_balance_score")
    support_f1 = aggregate.get("median_daily_token_support_f1")
    token_jsd = aggregate.get("median_daily_token_jsd")
    effective_alignment = aggregate.get("median_daily_effective_token_alignment")
    collapse = aggregate.get("median_daily_collapse_rate")
    unique = aggregate.get("median_daily_unique_tokens")
    return (
        0 if aggregate.get("healthy", False) else 1,  # health gate retained
        -float(codebook_balance) if codebook_balance is not None else math.inf,
        -float(p10_balance) if p10_balance is not None else math.inf,
        -float(support_f1) if support_f1 is not None else math.inf,
        float(token_jsd) if token_jsd is not None else math.inf,
        -float(effective_alignment) if effective_alignment is not None else -math.inf,
        float(collapse) if collapse is not None else math.inf,
        -float(unique) if unique is not None else -math.inf,
        result["tid"],
    )


def load_completed_results(output_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((output_root / "trials").glob("*/result.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            results.append(payload)
    return results


def write_leaderboard(output_root: Path) -> dict[str, Any]:
    results = sorted(load_completed_results(output_root), key=result_sort_key)
    baseline = next((item for item in results if item.get("is_baseline")), None)
    baseline_da = (
        baseline["validation"]["aggregate"]["avg_da_per_date"]
        if baseline is not None else None
    )
    rows = []
    for rank, result in enumerate(results, start=1):
        aggregate = result["validation"]["aggregate"]
        rows.append({
            "rank": rank,
            "tid": result["tid"],
            "is_baseline": result["is_baseline"],
            "params": result["params"],
            "healthy": aggregate["healthy"],
            "avg_da_per_date": aggregate["avg_da_per_date"],
            "delta_da_vs_baseline": (
                aggregate["avg_da_per_date"] - baseline_da
                if baseline_da is not None else None
            ),
            "avg_daily_rank_ic": aggregate["avg_daily_rank_ic"],
            "avg_mape": aggregate["avg_mape"],
            "avg_ampratio": aggregate["avg_ampratio"],
            "p90_daily_collapse_rate": aggregate[
                "p90_daily_collapse_rate"
            ],
            "worst_collapse_rate": aggregate["worst_collapse_rate"],
            "min_unique_tokens": aggregate["min_unique_tokens"],
            "median_daily_codebook_balance_score": aggregate.get(
                "median_daily_codebook_balance_score"
            ),
            "median_daily_target_support_recall": aggregate.get(
                "median_daily_target_support_recall"
            ),
            "median_daily_effective_token_alignment": aggregate.get(
                "median_daily_effective_token_alignment"
            ),
            "median_daily_token_jsd": aggregate.get(
                "median_daily_token_jsd"
            ),
            "healthy_epochs_in_mature_window": aggregate[
                "healthy_epochs_in_mature_window"
            ],
            "mature_window_epochs": aggregate["mature_window_epochs"],
            "maturity_onset_epoch": aggregate["maturity_onset_epoch"],
            "representative_epoch": aggregate["representative_epoch"],
            "best_da": aggregate["best_da"],
            "best_da_epoch": aggregate["best_da_epoch"],
            "best_val_loss": result["training"]["best_val_loss"],
            "model_path": result["training"]["model_path"],
        })
    leaderboard = {
        "updated_at_utc": utc_now(),
        "selection_rule": [
            "No weighted score. Each trial is summarized over its full epoch trajectory.",
            "Primary ranking is by target-relative coarse codebook balance (median, p10), token support F1, token JSD, effective token alignment, collapse, and unique tokens.",
            "Downstream DA/RankIC/MAPE/AmpRatio serve only as guardrails.",
            "Baseline (Exp 04-A selected optimizer) must not be dominated.",
        ],
        "n_completed": len(rows),
        "rows": rows,
    }
    atomic_write_json(output_root / "leaderboard.json", leaderboard)

    if rows:
        print("\nLeaderboard")
        print("rank  trial             ok      DA      dBase   dailyIC   collapse  unique")
        for row in rows:
            delta = row["delta_da_vs_baseline"]
            print(
                f"{row['rank']:>4}  {row['tid']:<16} "
                f"{str(row['healthy']):<5} "
                f"{row['avg_da_per_date'] * 100:>7.2f}% "
                f"{(delta or 0.0) * 100:>+7.2f}pp "
                f"{row['avg_daily_rank_ic']:>8.4f} "
                f"{row['worst_collapse_rate'] * 100:>8.2f}% "
                f"{row['min_unique_tokens']:>7}"
            )
    return leaderboard


def run_trial(
    *,
    trial: dict[str, Any],
    layout: StudyLayout,
    settings: dict[str, Any],
    study_id: str,
    cache_root: Path,
) -> dict[str, Any]:
    tid = trial["tid"]
    params = trial["params"]
    paths = trial_paths(
        layout.results_root, tid, layout.weights_root
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["weights_directory"].mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["params"], params)
    atomic_write_json(
        paths["override"],
        runtime_override(params, settings=settings, cache_root=cache_root),
    )

    if paths["result"].exists():
        existing = load_json(paths["result"])
        if (
            existing.get("status") == "completed"
            and existing.get("study_fingerprint") == study_id
        ):
            print(f"  Reusing completed result for {tid}")
            return existing

    started = time.monotonic()
    try:
        env = os.environ.copy()
        env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(paths["override"])
        metadata: dict[str, Any] | None = None
        if paths["model"].exists():
            metadata = checkpoint_metadata(paths["model"])

        if metadata is None or not metadata["completed"]:
            run_command(
                build_train_command(params, paths, settings, tid),
                env,
                log_path=paths["gpt_log"],
                label=f"Exp 04-B {tid} GPT",
            )
            if not paths["model"].exists():
                raise RuntimeError(f"Training did not produce {paths['model']}")
            metadata = checkpoint_metadata(paths["model"])
        else:
            print(f"  Reusing completed model for {tid}")

        if not paths["checkpoint_index"].exists():
            raise RuntimeError(
                f"Per-epoch checkpoint index is missing: {paths['checkpoint_index']}"
            )
        checkpoint_index = load_json(paths["checkpoint_index"])
        checkpoint_entries = checkpoint_index.get("checkpoints", [])
        epoch_checkpoint_count = len(checkpoint_entries)
        if epoch_checkpoint_count != settings["epochs"]:
            raise RuntimeError(
                f"Expected {settings['epochs']} epoch checkpoints for {tid}, "
                f"found {epoch_checkpoint_count}"
            )
        indexed_epochs = [int(item.get("epoch", 0)) for item in checkpoint_entries]
        if indexed_epochs != list(range(1, settings["epochs"] + 1)):
            raise RuntimeError(
                f"Checkpoint index has non-contiguous epochs for {tid}: "
                f"{indexed_epochs}"
            )
        missing_checkpoints = [
            item.get("path", "") for item in checkpoint_entries
            if not Path(item.get("path", "")).is_file()
        ]
        if missing_checkpoints:
            raise RuntimeError(
                f"Checkpoint index contains missing files for {tid}: "
                f"{missing_checkpoints}"
            )

        trajectory_rows = run_epoch_trajectory(
            paths=paths,
            settings=settings,
            env=env,
            tid=tid,
        )
        aggregate = aggregate_trajectory(trajectory_rows)
        final_model_path = Path(checkpoint_entries[-1]["path"]).resolve()
        representative_entry = next(
            item
            for item in checkpoint_entries
            if int(item["epoch"]) == int(aggregate["representative_epoch"])
        )
        representative_model_path = Path(
            representative_entry["path"]
        ).resolve()
        result = {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "study_fingerprint": study_id,
            "order": trial["order"],
            "tid": tid,
            "is_baseline": trial["is_baseline"],
            "seed": SEED,
            "params": params,
            "fixed": {
                "loss": "ce",
                "optimizer": "muon",
                "max_stocks": settings["max_stocks"],
                "max_seq_len": settings["max_seq_len"],
                "epochs": settings["epochs"],
                "early_stop_patience": 0,
                "constant_accumulation": True,
                "controlled_loader_seed": settings[
                    "controlled_loader_seed"
                ],
                "exact_accumulation_boundaries": settings[
                    "exact_accumulation_boundaries"
                ],
                "architecture": settings["architecture"]["config"],
            },
            "training": {
                "model_path": str(representative_model_path),
                "representative_epoch": aggregate["representative_epoch"],
                "final_model_path": str(final_model_path),
                "best_val_model_path": str(paths["model"].resolve()),
                "resume_path": str(paths["resume"].resolve()),
                "history_path": str(paths["history"].resolve()),
                "checkpoint_index_path": str(paths["checkpoint_index"].resolve()),
                "epoch_checkpoint_count": epoch_checkpoint_count,
                "best_val_loss": metadata["best_val_loss"],
                "best_epoch": metadata["best_epoch"],
                "checkpoint_completed": metadata["completed"],
            },
            "validation": {
                "trajectory_path": str(paths["trajectory"].resolve()),
                "evaluated_epochs": len(trajectory_rows),
                "aggregate": aggregate,
            },
            "wall_time_this_invocation_sec": time.monotonic() - started,
        }
        atomic_write_json(paths["result"], result)
        return result
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_at_utc": utc_now(),
            "study_fingerprint": study_id,
            "order": trial["order"],
            "tid": tid,
            "is_baseline": trial["is_baseline"],
            "seed": SEED,
            "params": params,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_write_json(paths["result"], failure)
        raise


def run_search(
    layout: StudyLayout,
    settings: dict[str, Any],
) -> dict[str, Any]:
    output_root = layout.results_root
    tokenizer = Path(settings["tokenizer"]["path"])
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {tokenizer}")
    if not settings["smoke"] and settings["max_stocks"] != 0:
        raise RuntimeError("Formal Exp 04-B must use all stocks")

    plan = build_trial_plan(settings["n_trials"])
    manifest = prepare_manifest(output_root, settings, plan, layout)
    study_id = manifest["study_fingerprint"]
    manifest["status"] = "running"
    manifest.setdefault("started_at_utc", utc_now())
    manifest.pop("failed_at_utc", None)
    manifest.pop("error", None)
    atomic_write_json(output_root / "study_manifest.json", manifest)

    print("=" * 78)
    print("Exp 04-B | one-stage full-data HPO")
    print(f"Python: {sys.executable}")
    print(
        f"Trials: {len(plan)} (baseline first), seed={SEED}, "
        f"epochs={settings['epochs']}, max_stocks={settings['max_stocks']}"
    )
    print(
        f"Architecture: {settings['architecture']['config']} "
        f"({settings['architecture']['dim']}d/"
        f"{settings['architecture']['depth']}L); "
        f"accumulation={settings['accumulation_steps']} constant"
    )
    runtime = settings["runtime_estimate"]
    if not settings["smoke"]:
        print(
            f"Budget estimate: {runtime['estimated_total_minutes'] / 60:.2f}h "
            f"({runtime['total_minutes_per_trial']:.1f} min/trial + "
            f"{runtime['fixed_reserve_minutes']:.0f} min reserve)"
        )
    print(
        f"Validation offsets={settings['validation_offsets']}, "
        f"days/window={settings['eval_days']}"
    )
    print("=" * 78)

    try:
        for trial in plan:
            params_text = ", ".join(
                f"{key}={value}" for key, value in trial["params"].items()
            )
            print(
                f"\n[{trial['order']}/{len(plan)}] {trial['tid']} "
                f"({'baseline' if trial['is_baseline'] else 'preregistered'})\n"
                f"  {params_text}",
                flush=True,
            )
            run_trial(
                trial=trial,
                layout=layout,
                settings=settings,
                study_id=study_id,
                cache_root=layout.weights_root / "shared",
            )
            write_leaderboard(output_root)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output_root / "study_manifest.json", manifest)
        raise

    manifest["status"] = "search_completed"
    manifest["completed_at_utc"] = utc_now()
    manifest.pop("error", None)
    atomic_write_json(output_root / "study_manifest.json", manifest)
    leaderboard = write_leaderboard(output_root)
    write_download_manifest(layout)
    return leaderboard


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_daily_da_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    n_bootstrap: int = 10_000,
    block_length: int = 5,
) -> dict[str, Any]:
    candidate_daily = candidate["per_date"]
    baseline_daily = baseline["per_date"]
    dates = sorted(set(candidate_daily) & set(baseline_daily))
    if not dates:
        raise RuntimeError("No common dates for paired holdout comparison")
    differences = [
        float(candidate_daily[date]["da"]) - float(baseline_daily[date]["da"])
        for date in dates
    ]
    observed = sum(differences) / len(differences)

    rng = random.Random(SEED)
    bootstrap = []
    block_length = min(max(block_length, 1), len(differences))
    for _ in range(n_bootstrap):
        sample = []
        while len(sample) < len(differences):
            start = rng.randrange(len(differences))
            sample.extend(
                differences[(start + step) % len(differences)]
                for step in range(block_length)
            )
        bootstrap.append(sum(sample[:len(differences)]) / len(differences))
    bootstrap.sort()
    lower = percentile(bootstrap, 0.025)
    upper = percentile(bootstrap, 0.975)
    return {
        "n_common_dates": len(dates),
        "candidate_minus_baseline_da": observed,
        "moving_block_bootstrap_95_ci": [lower, upper],
        "bootstrap_samples": n_bootstrap,
        "block_length_dates": block_length,
        "candidate_daily_win_rate": (
            sum(delta > 0 for delta in differences) / len(differences)
        ),
        "candidate_daily_tie_rate": (
            sum(delta == 0 for delta in differences) / len(differences)
        ),
        "ci_excludes_zero_positive": lower > 0,
    }


def run_holdout(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "study_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("No completed study manifest; run the search first")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "search_completed":
        raise RuntimeError("Holdout remains sealed until every planned trial completes")

    settings = manifest["settings"]
    tokenizer = Path(settings["tokenizer"]["path"])
    leaderboard = write_leaderboard(output_root)
    rows = leaderboard["rows"]
    if len(rows) != settings["n_trials"]:
        raise RuntimeError(
            f"Only {len(rows)}/{settings['n_trials']} trials are complete; "
            "holdout remains sealed"
        )
    healthy = [row for row in rows if row["healthy"]]
    if not healthy:
        raise RuntimeError("No trial passed the health gate")
    winner = healthy[0]
    baseline = next((row for row in rows if row["is_baseline"]), None)
    if baseline is None:
        raise RuntimeError("Baseline result is missing")

    def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
        paths = trial_paths(output_root, row["tid"])
        output = (
            paths["directory"]
            / f"holdout_offset_{HOLDOUT_OFFSET:04d}_days_{HOLDOUT_DAYS}.json"
        )
        return evaluate_window(
            model=Path(row["model_path"]),
            tokenizer=tokenizer,
            output=output,
            override_path=paths["override"],
            n_stocks=0,
            n_days=HOLDOUT_DAYS,
            start_offset=HOLDOUT_OFFSET,
            batch_size=settings["eval_batch_size"],
            sample_strategy="random",
        )

    print(
        f"\nOpening final holdout once for winner={winner['tid']} "
        f"and baseline={baseline['tid']}..."
    )
    winner_metrics = evaluate_row(winner)
    baseline_metrics = (
        winner_metrics if winner["tid"] == baseline["tid"] else evaluate_row(baseline)
    )
    comparison = paired_daily_da_comparison(winner_metrics, baseline_metrics)
    result = {
        "experiment": "Exp 04-B final holdout",
        "evaluated_at_utc": utc_now(),
        "study_fingerprint": manifest["study_fingerprint"],
        "selection_used_only_validation_windows": True,
        "winner": winner,
        "baseline": baseline,
        "holdout": {
            "start_offset": HOLDOUT_OFFSET,
            "n_days_requested": HOLDOUT_DAYS,
            "winner_metrics": {
                key: value for key, value in winner_metrics.items()
                if key != "per_date"
            },
            "baseline_metrics": {
                key: value for key, value in baseline_metrics.items()
                if key != "per_date"
            },
            "paired_daily_da": comparison,
        },
    }
    atomic_write_json(output_root / "holdout_comparison.json", result)
    print(
        "Holdout result: "
        f"winner DA={winner_metrics['avg_da_per_date'] * 100:.2f}%, "
        f"baseline DA={baseline_metrics['avg_da_per_date'] * 100:.2f}%, "
        f"delta={comparison['candidate_minus_baseline_da'] * 100:+.2f}pp, "
        f"95% CI=[{comparison['moving_block_bootstrap_95_ci'][0] * 100:+.2f}, "
        f"{comparison['moving_block_bootstrap_95_ci'][1] * 100:+.2f}]pp"
    )
    return result


def run_smoke(
    args: argparse.Namespace,
    tokenizer: dict[str, Any],
    architecture: dict[str, Any],
    upstream_selections: dict[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(prefix="kronos_exp04b_smoke_") as temp:
        temporary_root = Path(temp)
        layout = StudyLayout.create(
            temporary_root / "weights",
            temporary_root / "results",
        )
        settings = study_settings(
            n_trials=1,
            epochs=1,
            tokenizer=tokenizer,
            architecture=architecture,
            upstream_selections=upstream_selections,
            validation_offsets=(0,),
            eval_days=2,
            batch_tokens=2048,
            accumulation_steps=args.accumulation_steps,
            eval_batch_size=args.eval_batch_size,
            max_collapse_rate=1.0,
            min_unique_tokens=1,
            time_budget_hours=None,
            trial_count_source="smoke",
            smoke=True,
        )
        leaderboard = run_search(layout, settings)
        rows = leaderboard.get("rows", [])
        if len(rows) != 1 or not rows[0]["healthy"]:
            raise RuntimeError("Smoke test did not produce one healthy completed trial")
        epoch_snapshots = list(layout.weights_root.rglob("*_ep*.pt"))
        checkpoint_indexes = list(
            layout.results_root.rglob("*_checkpoints.json")
        )
        if len(epoch_snapshots) != 1 or len(checkpoint_indexes) != 1:
            raise RuntimeError(
                "Smoke test did not retain exactly one epoch snapshot and index: "
                f"snapshots={epoch_snapshots}, indexes={checkpoint_indexes}"
            )
        index_payload = load_json(checkpoint_indexes[0])
        entries = index_payload.get("checkpoints", [])
        if (
            len(entries) != 1
            or entries[0].get("epoch") != 1
            or not Path(entries[0].get("path", "")).is_file()
            or not math.isfinite(float(entries[0].get("val_loss", math.nan)))
        ):
            raise RuntimeError(f"Invalid checkpoint index: {index_payload}")
        print(
            "\nSMOKE TEST PASSED: train -> checkpoints -> all-epoch shifted "
            "window eval -> mature aggregation -> leaderboard"
        )
        print(
            f"Temporary artifacts will now be removed: {temporary_root}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp 04-B one-stage full-data HPO (fixed seed=42)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Prefix length of the preregistered plan, including baseline; "
             "default derives from the time budget",
    )
    parser.add_argument(
        "--time_budget_hours",
        type=float,
        default=DEFAULT_TIME_BUDGET_HOURS,
        help="Planning budget used when --n_trials is omitted (default: 10)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--weights_root", type=Path, default=DEFAULT_WEIGHTS_ROOT
    )
    parser.add_argument(
        "--results_root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Default reads the Exp 02 seed-42 selection",
    )
    parser.add_argument(
        "--architecture_selection",
        type=Path,
        default=DEFAULT_ARCH_SELECTION,
        help="Exp 03 selection.json (required before formal Exp 04)",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help=(
            "Read the tokenizer, Exp 03 architecture, and Exp 04-A muon "
            "selection from the portable bundle at --portable_dir instead of "
            "server_runs/. Use this on servers that carry only the repo + "
            "dataset. Output still honours KRONOS_WEIGHTS_ROOT/"
            "KRONOS_RESULTS_ROOT."
        ),
    )
    parser.add_argument(
        "--portable_dir",
        type=Path,
        default=AB_COMMON.PORTABLE_DIR,
        help="Portable bundle directory (tokenizer.pt + upstream.json)",
    )
    parser.add_argument(
        "--validation_offsets",
        type=parse_offsets,
        default=DEFAULT_VALIDATION_OFFSETS,
        help="Comma-separated validation offsets (default: full-coverage tiling"
        " 0,20,...,380)",
    )
    parser.add_argument("--eval_days", type=int, default=EVAL_DAYS)
    parser.add_argument(
        "--batch_tokens", type=int, default=0,
        help="0 inherits the selected Exp 03 architecture setting (6144). "
        "Raising this improves single-microbatch GPU utilization on larger "
        "cards; pair with --accumulation_steps so that "
        "batch_tokens * accumulation_steps stays ~196608 (= 6144 * 32) to "
        "preserve the Exp 04-A effective batch and optimizer-step count.",
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=32,
        help=(
            "Optimizer steps accumulate this many microbatches per update. "
            "Default 32 matches Exp 03/04-A. When raising --batch_tokens on a "
            "large GPU, lower this proportionally (e.g. batch_tokens=24576 "
            "with accumulation_steps=8) to keep the effective batch token "
            "budget and step count identical to Exp 04-A."
        ),
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=0,
        help="0 inherits the selected Exp 03 architecture setting",
    )
    parser.add_argument("--max_collapse_rate", type=float, default=0.35)
    parser.add_argument("--min_unique_tokens", type=int, default=32)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny end-to-end trial in an auto-deleted temporary directory",
    )
    mode.add_argument(
        "--holdout",
        action="store_true",
        help="After search completion, evaluate the selected winner and baseline once",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.time_budget_hours <= 0:
        parser.error("--time_budget_hours must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_tokens < 0 or args.eval_batch_size < 0:
        parser.error("batch sizes must be non-negative")
    if args.eval_days <= 0:
        parser.error("--eval_days must be positive")
    if any(
        offset + args.eval_days > HOLDOUT_OFFSET
        for offset in args.validation_offsets
    ):
        parser.error("validation windows may not enter the sealed holdout")
    ensure_runtime()
    results_root = args.results_root.expanduser().resolve()
    if args.holdout:
        run_holdout(results_root)
        return

    if args.portable:
        portable = AB_COMMON.load_portable_assets(args.portable_dir)
        tokenizer = portable["tokenizer"]
        architecture = portable["architecture"]
        upstream_selections = portable["upstream_selections"]
        print(
            f"Portable mode: upstream assets loaded from "
            f"{args.portable_dir.resolve()}"
        )
    else:
        tokenizer = AB_COMMON.resolve_tokenizer(args.tokenizer)
        architecture = AB_COMMON.resolve_architecture(
            args.architecture_selection
        )
        requirements = {
            "exp04a_optimizer": (EXP04A_SELECTION, "muon"),
        }
        if args.smoke:
            existing = {
                name: requirement
                for name, requirement in requirements.items()
                if requirement[0].is_file()
            }
            upstream_selections = AB_COMMON.load_required_selections(existing)
            for name, (path, expected_arm) in requirements.items():
                if name not in upstream_selections:
                    upstream_selections[name] = {
                        "source": "smoke_preregistered_stub",
                        "expected_path": str(path.resolve()),
                        "expected_arm": expected_arm,
                    }
        else:
            upstream_selections = AB_COMMON.load_required_selections(requirements)
    if args.smoke:
        run_smoke(args, tokenizer, architecture, upstream_selections)
        return

    recommended = recommended_trial_count(args.time_budget_hours, args.epochs)
    n_trials = recommended if args.n_trials is None else args.n_trials
    trial_count_source = (
        "time_budget_estimate" if args.n_trials is None else "explicit"
    )
    settings = study_settings(
        n_trials=n_trials,
        epochs=args.epochs,
        tokenizer=tokenizer,
        architecture=architecture,
        upstream_selections=upstream_selections,
        validation_offsets=args.validation_offsets,
        eval_days=args.eval_days,
        batch_tokens=args.batch_tokens,
        accumulation_steps=args.accumulation_steps,
        eval_batch_size=args.eval_batch_size,
        max_collapse_rate=args.max_collapse_rate,
        min_unique_tokens=args.min_unique_tokens,
        time_budget_hours=args.time_budget_hours,
        trial_count_source=trial_count_source,
    )
    layout = StudyLayout.create(args.weights_root, args.results_root)
    free_gib = shutil.disk_usage(layout.weights_root).free / (1024**3)
    checkpoint_gib = (
        n_trials
        * args.epochs
        * int(architecture["parameter_count"])
        * 4
        / (1024**3)
    )
    required_gib = checkpoint_gib + 3.0
    if free_gib < required_gib:
        raise RuntimeError(
            f"Exp 04-B estimates at least {required_gib:.1f} GiB free is "
            f"needed for checkpoints/caches; only {free_gib:.1f} GiB remains"
        )
    run_search(layout, settings)


if __name__ == "__main__":
    main()
