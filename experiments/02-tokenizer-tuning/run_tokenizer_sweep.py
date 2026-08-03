"""Run the current tokenizer-architecture sweep with epoch-wise GPT evaluation.

This is the canonical Exp 02 entry point.  It consumes the bit-width decision
from the completed Exp 01 sweep, trains one tokenizer per encoder/decoder
capacity point, then trains and evaluates one 50-epoch CE+AdamW GPT for every
tokenizer.  Every GPT epoch is retained and evaluated on the same four
pre-holdout windows used by Exp 01.

What Exp 01 taught this stage
-----------------------------
Exp 01 varied the nominal codebook size, so raw daily Collapse and raw daily
Unique-token counts were not comparable across arms: a larger codebook slices
the data more finely, which mechanically lowers Collapse and raises Unique
without proving that the GPT can exploit the extra classes.  Measured against
the target distribution instead, support/effective-code/collapse alignment fell
as the codebook grew, and the predictive information the GPT extracted saturated
near ~1.3 bits per token from 2^12 to 2^18 joint codes.  Exp 02 therefore
carries the per-layer codebook diagnostics (coarse/fine utilization, effective
codes, collapse) into the combined summary so the architecture decision is made
on capacity-normalized evidence rather than on raw behaviour counts.

Checkpoints/caches and downloadable diagnostics use separate resumable roots.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parent
ROOT = EXPERIMENT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_io import (
    StudyLayout,
    default_study_roots,
    runtime_environment,
    write_download_manifest,
)

EXP01_DIR = ROOT / "experiments" / "01-bitsweep"
_, EXP01_ROOT = default_study_roots("01-bitsweep", seed=42)
EXP01_SELECTION = EXP01_ROOT / "selection.json"
EXP01_PIPELINE_PATH = EXP01_DIR / "run_bitsweep.py"
TOKENIZER_SCRIPT = ROOT / "train_tokenizer.py"
GPT_SCRIPT = ROOT / "train_base.py"
TOKENIZER_EVAL_SCRIPT = EXP01_DIR / "evaluate_tokenizer.py"
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "c-hpo" / "evaluate_epoch_trajectory.py"
)
ANALYSIS_SCRIPT = EXPERIMENT_DIR / "analyze_tokenizer_epochwise.py"
DEFAULT_WEIGHTS_ROOT, DEFAULT_RESULTS_ROOT = default_study_roots(
    "02-tokenizer-tuning", seed=42
)

SEED = 42
# (embedding_dim, hidden_dim).  The first six points are the original grid.
# The last three extend the encoder/decoder capacity axis because Exp 01 showed
# that the binding constraint at fixed bits is codebook utilization, not the
# nominal vocabulary: at embedding_dim=64/hidden_dim=192 the coarse layer of the
# selected bit split used only a fraction of its nominal codes, so the sweep has
# to reach far enough to tell "the encoder is too small" from "the codebook is
# intrinsically hard to fill".
DEFAULT_CONFIGS = (
    (48, 192),
    (48, 256),
    (64, 192),
    (64, 256),
    (96, 192),
    (96, 256),
    (96, 384),
    (128, 256),
    (128, 384),
)
HOLDOUT_OFFSET = 400
EVAL_DAYS = 1
# Full-coverage, single-day-resolution evaluation. The pre-holdout region
# [0, HOLDOUT_OFFSET) is tiled with contiguous EVAL_DAYS-day windows (offsets
# 0, 1, ..., 399), so every pre-holdout trading day is scored as its own window
# for the finest robustness granularity. Offset HOLDOUT_OFFSET and beyond remain
# the sealed holdout.
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, EVAL_DAYS))


def _load_exp01_pipeline():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp01_pipeline", EXP01_PIPELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXP01_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPE = _load_exp01_pipeline()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_key(embedding_dim: int, hidden_dim: int) -> str:
    return f"{embedding_dim}x{hidden_dim}"


def json_safe_float(value: Any) -> float | None:
    """Return a JSON-writable float; the study writers reject NaN/Inf."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def config_directory_name(embedding_dim: int, hidden_dim: int) -> str:
    return f"emb_{embedding_dim:03d}_hid_{hidden_dim:03d}"


def parse_configs(raw: str) -> list[tuple[int, int]]:
    if not raw.strip():
        return list(DEFAULT_CONFIGS)
    output: list[tuple[int, int]] = []
    for item in raw.split(","):
        embedding, hidden = item.strip().lower().split("x", maxsplit=1)
        value = (int(embedding), int(hidden))
        if value not in DEFAULT_CONFIGS:
            raise ValueError(f"Unsupported tokenizer config: {item}")
        if value not in output:
            output.append(value)
    return output


def parse_offsets(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("At least one validation offset is required")
    if any(value < 0 for value in values):
        raise ValueError("Validation offsets must be non-negative")
    return values


def resolve_bits(raw: str) -> tuple[int, int, dict[str, Any]]:
    if raw.strip():
        left, right = raw.strip().split("+", maxsplit=1)
        return int(left), int(right), {
            "source": "command_line",
            "path": None,
            "sha256": None,
        }
    if not EXP01_SELECTION.is_file():
        raise RuntimeError(
            f"Exp 01 selection is missing: {EXP01_SELECTION}. "
            "Record the reviewed bit split on the server with "
            "experiments/01-bitsweep/narrate_bitsweep.py "
            "--select_config L1+L2 --rationale '...', or pass --bits L1+L2."
        )
    payload = PIPE.load_json(EXP01_SELECTION)
    if not payload.get("upstream_eligible", False):
        raise RuntimeError(
            f"Exp 01 selection has not been recorded after review: "
            f"{EXP01_SELECTION}"
        )
    selected = payload.get("selected", payload)
    l1 = int(selected["bits_l1"])
    l2 = int(selected["bits_l2"])
    return l1, l2, {
        "source": "exp01_selection",
        "path": str(EXP01_SELECTION.resolve()),
        "sha256": PIPE.file_sha256(EXP01_SELECTION),
        "selection": payload,
    }


def config_paths(
    weights_root: Path,
    results_root: Path,
    embedding_dim: int,
    hidden_dim: int,
) -> dict[str, Path]:
    name = config_directory_name(embedding_dim, hidden_dim)
    weights = weights_root / "configs" / name
    results = results_root / "configs" / name
    return {
        "directory": results,
        "weights_directory": weights,
        "override": results / "override.json",
        "run": results / "run.json",
        "tokenizer": weights / "tokenizer.pt",
        "tokenizer_resume": weights / "tokenizer.pt.ckpt",
        "tokenizer_metrics": results / "tokenizer_metrics.json",
        "tokenizer_history": results / "tokenizer_training_history.json",
        "tokenizer_distributions": results / "tokenizer_distributions.npz",
        "model": weights / "model.pt",
        "model_resume": weights / "model.pt.ckpt",
        "checkpoint_index": results / "model_checkpoints.json",
        "history": (
            results
            / f"history_toksweep_{embedding_dim}_{hidden_dim}.json"
        ),
        "trajectory": results / "epoch_trajectory",
        "cache": weights / "cache",
        "logs": results / "logs",
        "tokenizer_log": results / "logs" / "tokenizer.log",
        "tokenizer_eval_log": results / "logs" / "tokenizer_eval.log",
        "gpt_log": results / "logs" / "gpt.log",
        "trajectory_log": results / "logs" / "trajectory.log",
    }


def build_settings(
    *,
    configs: list[tuple[int, int]],
    bits_l1: int,
    bits_l2: int,
    dependency: dict[str, Any],
    tok_epochs: int,
    gpt_epochs: int,
    batch_tokens: int,
    eval_offsets: tuple[int, ...],
    eval_days: int,
    eval_batch_size: int,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "seed": SEED,
        "exp01_dependency": dependency,
        "bits_l1": bits_l1,
        "bits_l2": bits_l2,
        "configs": [config_key(*value) for value in configs],
        "tokenizer": {
            "epochs": tok_epochs,
            "batch_size": 256 if smoke else 8192,
            "learning_rate": 1e-4,
            "scheduler": "warmup_5pct_cosine",
            "early_stop_patience": 0,
            "validation_every": 1 if smoke else 5,
            "selection": "lowest tokenizer validation loss",
        },
        "gpt": {
            "epochs": gpt_epochs,
            "architecture": "baseline_256x2_h4_kv1",
            "loss": "ce",
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "fine_weight": 0.3,
            "heteroscedastic": True,
            "het_weight": 0.1,
            "warmup_ratio": 0.05,
            "max_stocks": 24 if smoke else 0,
            "max_seq_len": 256 if smoke else 0,
            "batch_tokens": 2048 if smoke else batch_tokens,
            "batch_cap": 64,
            "accumulation_steps": 8 if smoke else 32,
            "controlled_loader_seed": SEED,
            "exact_accumulation_boundaries": True,
            "curriculum": False,
            "early_stop_patience": 0,
            "retain_every_epoch": True,
        },
        "evaluation": {
            "offsets": list(eval_offsets),
            "days_per_window": eval_days,
            "n_stocks": 6 if smoke else 0,
            "sample_strategy": "shortest" if smoke else "random",
            "batch_size": eval_batch_size,
            "health_gate": {
                "max_daily_collapse_rate": 1.0 if smoke else 0.35,
                "min_daily_unique_tokens": 1 if smoke else 32,
            },
            "holdout_offset": HOLDOUT_OFFSET,
            "holdout_used": False,
            "selection_policy": (
                "No weighted score. Read three separate groups: (1) "
                "codebook-size-invariant quality (DA, daily RankIC, MAPE) "
                "including the per-window floors min_window_da and "
                "min_window_daily_rank_ic; (2) capacity-normalized behaviour "
                "(collapse/unique/effective/distribution alignment against the "
                "same-day target distribution, and the codebook balance "
                "score) instead of raw Collapse and raw Unique counts; (3) "
                "tokenizer sufficiency (reconstruction MAE plus per-layer "
                "code utilization and effective code counts)."
            ),
            "exp01_measurement_lesson": (
                "Raw daily Collapse and raw daily Unique-token counts scale "
                "with the nominal codebook and are only comparable when the "
                "bit split is held fixed, as it is inside Exp 02. They still "
                "move with how much of the codebook each architecture "
                "actually fills, so alignment-against-target metrics and the "
                "per-layer utilization diagnostics remain the primary "
                "behaviour evidence here."
            ),
        },
        "smoke": smoke,
    }


def source_hashes() -> dict[str, str]:
    files = (
        "experiment_io.py",
        "config.py",
        "data_processor.py",
        "training_utils.py",
        "train_tokenizer.py",
        "train_base.py",
        "eval_helpers.py",
        "model/tokenizer.py",
        "model/kronos_preview.py",
        "model/layers.py",
        "experiments/01-bitsweep/evaluate_tokenizer.py",
        "experiments/01-bitsweep/run_bitsweep.py",
        "experiments/02-tokenizer-tuning/run_tokenizer_sweep.py",
        "experiments/02-tokenizer-tuning/analyze_tokenizer_epochwise.py",
        "experiments/04/c-hpo/evaluate_epoch_trajectory.py",
    )
    return {
        relative: PIPE.file_sha256(ROOT / relative)
        for relative in files
        if (ROOT / relative).is_file()
    }


def prepare_manifest(
    results_root: Path,
    settings: dict[str, Any],
    layout: StudyLayout,
) -> dict[str, Any]:
    implementation = source_hashes()
    dataset = PIPE.dataset_signature()
    fingerprint = PIPE.canonical_sha256(
        {
            "settings": settings,
            "implementation_sha256": implementation,
            "dataset": dataset,
        }
    )
    path = results_root / "study_manifest.json"
    if path.exists():
        payload = PIPE.load_json(path)
        if payload.get("study_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{path} has a different protocol/source fingerprint. "
                "Archive it instead of mixing results."
            )
        return payload
    payload = {
        "experiment": "Exp 02 Tokenizer architecture sweep",
        "design": "full-data tokenizer grid plus epoch-wise downstream GPT",
        "status": "planned",
        "created_at_utc": utc_now(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "study_fingerprint": fingerprint,
        "settings": settings,
        "implementation_sha256": implementation,
        "dataset": dataset,
        "artifact_layout": layout.metadata(),
        "runtime_environment": runtime_environment(ROOT),
        "clean_results_root": True,
    }
    PIPE.atomic_write_json(path, payload)
    return payload


def update_manifest(results_root: Path, **updates: Any) -> dict[str, Any]:
    path = results_root / "study_manifest.json"
    payload = PIPE.load_json(path)
    payload.update(updates)
    PIPE.atomic_write_json(path, payload)
    return payload


def update_config_run(
    paths: dict[str, Path],
    embedding_dim: int,
    hidden_dim: int,
    **updates: Any,
) -> dict[str, Any]:
    if paths["run"].exists():
        payload = PIPE.load_json(paths["run"])
    else:
        payload = {
            "config": config_key(embedding_dim, hidden_dim),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "status": "planned",
            "stages": {},
            "created_at_utc": utc_now(),
        }
    payload.update(updates)
    payload["updated_at_utc"] = utc_now()
    PIPE.atomic_write_json(paths["run"], payload)
    return payload


def write_override(
    path: Path,
    cache_directory: Path,
    embedding_dim: int,
    hidden_dim: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "DataConfig": {
            "random_seed": SEED,
            "max_stocks": settings["gpt"]["max_stocks"],
        },
        "TokenizerConfig": {
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "bits_l1": settings["bits_l1"],
            "bits_l2": settings["bits_l2"],
            "bits_per_quantizer": 0,
            "random_seed": SEED,
        },
        "TrainingConfig": {
            "random_seed": SEED,
            "warmup_ratio": settings["gpt"]["warmup_ratio"],
            "batch_size": 1,
            "accumulation_steps": settings["gpt"]["accumulation_steps"],
            "save_dir": str(cache_directory.resolve()),
        },
    }
    if path.exists() and PIPE.load_json(path) != payload:
        raise RuntimeError(f"Refusing to overwrite incompatible {path}")
    PIPE.atomic_write_json(path, payload)
    return payload


def run_one_config(
    layout: StudyLayout,
    feature_cache: Path,
    eval_cache: Path,
    settings: dict[str, Any],
    embedding_dim: int,
    hidden_dim: int,
) -> None:
    paths = config_paths(
        layout.weights_root,
        layout.results_root,
        embedding_dim,
        hidden_dim,
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["weights_directory"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    write_override(
        paths["override"],
        paths["cache"],
        embedding_dim,
        hidden_dim,
        settings,
    )
    run_state = update_config_run(
        paths,
        embedding_dim,
        hidden_dim,
        status="running",
        started_at_utc=utc_now(),
    )
    env = os.environ.copy()
    env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(paths["override"].resolve())

    tokenizer_complete = PIPE.tokenizer_is_complete(
        paths,
        l1=settings["bits_l1"],
        l2=settings["bits_l2"],
        epochs=settings["tokenizer"]["epochs"],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    if not tokenizer_complete:
        run_state["stages"]["tokenizer"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        PIPE.atomic_write_json(paths["run"], run_state)
        command = [
            sys.executable,
            str(TOKENIZER_SCRIPT),
            "--save_path",
            str(paths["tokenizer"]),
            "--metrics_path",
            str(paths["tokenizer_history"]),
            "--bits_l1",
            str(settings["bits_l1"]),
            "--bits_l2",
            str(settings["bits_l2"]),
            "--embedding_dim",
            str(embedding_dim),
            "--hidden_dim",
            str(hidden_dim),
            "--epochs",
            str(settings["tokenizer"]["epochs"]),
            "--batch_size",
            str(settings["tokenizer"]["batch_size"]),
            "--val_every",
            str(settings["tokenizer"]["validation_every"]),
            "--early_stop_patience",
            "0",
            "--scheduler",
            "--seed",
            str(SEED),
            "--feature_cache_dir",
            str(feature_cache),
        ]
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["tokenizer_log"],
            label=f"{config_key(embedding_dim, hidden_dim)} tokenizer",
        )
        if not PIPE.tokenizer_is_complete(
            paths,
            l1=settings["bits_l1"],
            l2=settings["bits_l2"],
            epochs=settings["tokenizer"]["epochs"],
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ):
            raise RuntimeError(
                "Tokenizer completion check failed for "
                f"{config_key(embedding_dim, hidden_dim)}"
            )
    run_state = update_config_run(paths, embedding_dim, hidden_dim)
    run_state["stages"]["tokenizer"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "checkpoint": str(paths["tokenizer"].resolve()),
        "sha256": PIPE.file_sha256(paths["tokenizer"]),
    }
    PIPE.atomic_write_json(paths["run"], run_state)

    validation_features = feature_cache / "tok_feat_val_2024-02-01.npz"
    if not PIPE.tokenizer_metrics_are_current(paths):
        PIPE.run_command(
            [
                sys.executable,
                str(TOKENIZER_EVAL_SCRIPT),
                "--tokenizer",
                str(paths["tokenizer"]),
                "--features",
                str(validation_features),
                "--output",
                str(paths["tokenizer_metrics"]),
                "--distribution_output",
                str(paths["tokenizer_distributions"]),
                "--seed",
                str(SEED),
                "--chunk_size",
                "8192" if settings["smoke"] else "65536",
            ],
            env=env,
            log_path=paths["tokenizer_eval_log"],
            label=f"{config_key(embedding_dim, hidden_dim)} tokenizer diagnostics",
        )
    run_state = update_config_run(paths, embedding_dim, hidden_dim)
    run_state["stages"]["tokenizer_diagnostics"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "metrics": str(paths["tokenizer_metrics"].resolve()),
    }
    PIPE.atomic_write_json(paths["run"], run_state)

    if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
        run_state = update_config_run(paths, embedding_dim, hidden_dim)
        run_state["stages"]["gpt"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        PIPE.atomic_write_json(paths["run"], run_state)
        tag = f"toksweep_{embedding_dim}_{hidden_dim}"
        command = [
            sys.executable,
            str(GPT_SCRIPT),
            "--save_path",
            str(paths["model"]),
            "--tokenizer_path",
            str(paths["tokenizer"]),
            "--epochs",
            str(settings["gpt"]["epochs"]),
            "--tag",
            tag,
            "--loss",
            "ce",
            "--gamma",
            "0",
            "--optimizer",
            "adamw",
            "--lr",
            str(settings["gpt"]["learning_rate"]),
            "--dropout",
            str(settings["gpt"]["dropout"]),
            "--weight_decay",
            str(settings["gpt"]["weight_decay"]),
            "--fine_weight",
            str(settings["gpt"]["fine_weight"]),
            "--heteroscedastic",
            "--het_weight",
            str(settings["gpt"]["het_weight"]),
            "--label_smoothing",
            "0",
            "--entropy_alpha",
            "0",
            "--max_stocks",
            str(settings["gpt"]["max_stocks"]),
            "--max_seq_len",
            str(settings["gpt"]["max_seq_len"]),
            "--batch_tokens",
            str(settings["gpt"]["batch_tokens"]),
            "--batch_cap",
            str(settings["gpt"]["batch_cap"]),
            "--controlled_loader_seed",
            str(settings["gpt"]["controlled_loader_seed"]),
            "--exact_accumulation_boundaries",
            "--early_stop_patience",
            "0",
            "--history_per_epoch",
            "--metrics_dir",
            str(paths["directory"]),
        ]
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["gpt_log"],
            label=f"{config_key(embedding_dim, hidden_dim)} GPT",
        )
        if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(
                f"GPT completion check failed for "
                f"{config_key(embedding_dim, hidden_dim)}"
            )
    run_state = update_config_run(paths, embedding_dim, hidden_dim)
    run_state["stages"]["gpt"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "checkpoint_index": str(paths["checkpoint_index"].resolve()),
    }
    PIPE.atomic_write_json(paths["run"], run_state)

    if not PIPE.trajectory_is_complete(paths, settings["gpt"]["epochs"]):
        run_state = update_config_run(paths, embedding_dim, hidden_dim)
        run_state["stages"]["epoch_evaluation"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        PIPE.atomic_write_json(paths["run"], run_state)
        evaluation = settings["evaluation"]
        gate = evaluation["health_gate"]
        command = [
            sys.executable,
            str(TRAJECTORY_SCRIPT),
            "--trial_dir",
            str(paths["directory"]),
            "--tokenizer",
            str(paths["tokenizer"]),
            "--output_dir",
            str(paths["trajectory"]),
            "--prepared_cache_dir",
            str(eval_cache),
            "--epochs",
            f"1-{settings['gpt']['epochs']}",
            "--offsets",
            ",".join(str(value) for value in evaluation["offsets"]),
            "--n_days",
            str(evaluation["days_per_window"]),
            "--n_stocks",
            str(evaluation["n_stocks"]),
            "--batch_size",
            str(evaluation["batch_size"]),
            "--seed",
            str(SEED),
            "--sample_strategy",
            evaluation["sample_strategy"],
            "--max_collapse_rate",
            str(gate["max_daily_collapse_rate"]),
            "--min_unique_tokens",
            str(gate["min_daily_unique_tokens"]),
            "--no_reference_check",
            "--experiment_label",
            f"Exp 02 Tokenizer {config_key(embedding_dim, hidden_dim)}",
        ]
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["trajectory_log"],
            label=f"{config_key(embedding_dim, hidden_dim)} epoch trajectory",
        )
        if not PIPE.trajectory_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(
                f"Trajectory completion check failed for "
                f"{config_key(embedding_dim, hidden_dim)}"
            )
    stages = update_config_run(
        paths, embedding_dim, hidden_dim
    ).get("stages", {})
    stages["epoch_evaluation"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "output": str(paths["trajectory"].resolve()),
    }
    update_config_run(
        paths,
        embedding_dim,
        hidden_dim,
        status="completed",
        completed_at_utc=utc_now(),
        stages=stages,
    )


def write_combined_summary(
    layout: StudyLayout,
    configs: list[tuple[int, int]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bits_l1 = int(settings["bits_l1"])
    bits_l2 = int(settings["bits_l2"])
    for embedding_dim, hidden_dim in configs:
        paths = config_paths(
            layout.weights_root,
            layout.results_root,
            embedding_dim,
            hidden_dim,
        )
        summary_path = paths["trajectory"] / "epoch_summary.json"
        if not summary_path.is_file() or not paths["tokenizer_metrics"].is_file():
            continue
        tokenizer = PIPE.load_json(paths["tokenizer_metrics"])
        # Exp 01 could not see that one bit split had a badly under-filled
        # coarse layer, because only the joint-code aggregate reached the
        # combined summary. Exp 02 tunes exactly the module that decides code
        # occupancy, so every level is exported per config-epoch row.
        codebook: dict[str, Any] = {}
        for level in ("coarse", "fine", "joint"):
            metrics = tokenizer[f"{level}_codes"]
            codebook.update(
                {
                    f"tokenizer_{level}_unique": metrics["n_unique"],
                    f"tokenizer_{level}_entropy_bits": metrics["entropy_bits"],
                    f"tokenizer_{level}_effective_codes": metrics[
                        "effective_codes"
                    ],
                    f"tokenizer_{level}_utilization": metrics["utilization"],
                    f"tokenizer_{level}_collapse": metrics["collapse_rate"],
                }
            )
        for item in PIPE.load_json(summary_path):
            row = dict(item)
            row.update(
                {
                    "config": config_key(embedding_dim, hidden_dim),
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "bits_l1": bits_l1,
                    "bits_l2": bits_l2,
                    "theoretical_bits": bits_l1 + bits_l2,
                    "coarse_vocab": 2 ** bits_l1,
                    "fine_vocab": 2 ** bits_l2,
                    "joint_vocab": 2 ** (bits_l1 + bits_l2),
                    "tokenizer_mae": tokenizer["mae"],
                    "tokenizer_rmse": tokenizer["rmse"],
                    "tokenizer_best_val_loss": json_safe_float(
                        tokenizer["checkpoint_best_val_loss"]
                    ),
                    "tokenizer_best_epoch": tokenizer[
                        "checkpoint_best_epoch"
                    ],
                    **codebook,
                }
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["embedding_dim"],
            row["hidden_dim"],
            row["epoch"],
        )
    )
    PIPE.atomic_write_json(
        layout.results_root / "combined_epoch_summary.json", rows
    )
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        temporary = layout.results_root / "combined_epoch_summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(
            temporary, layout.results_root / "combined_epoch_summary.csv"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current epoch-wise tokenizer architecture sweep"
    )
    parser.add_argument(
        "--weights_root", type=Path, default=DEFAULT_WEIGHTS_ROOT
    )
    parser.add_argument(
        "--results_root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--configs",
        default="",
        help=(
            "Comma-separated execution subset of the preregistered grid: "
            + ",".join(config_key(*value) for value in DEFAULT_CONFIGS)
        ),
    )
    parser.add_argument(
        "--bits",
        default="",
        help="L1+L2; default reads Exp 01 selection.json",
    )
    parser.add_argument("--tok_epochs", type=int, default=100)
    parser.add_argument("--gpt_epochs", type=int, default=50)
    parser.add_argument(
        "--batch_tokens",
        type=int,
        default=65536,
        help=(
            "Adaptive GPT microbatch token budget. The 65536 default targets "
            "24 GiB RTX 4090-class GPUs while preserving the fixed "
            "sequence-level accumulation schedule."
        ),
    )
    parser.add_argument(
        "--eval_offsets",
        default=",".join(str(value) for value in VALIDATION_OFFSETS),
    )
    parser.add_argument("--eval_days", type=int, default=EVAL_DAYS)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help=(
            "Length-bucketed inference batch size. Evaluation automatically "
            "halves this value and retries if CUDA memory is exhausted."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    PIPE.verify_runtime()
    configs = parse_configs(args.configs)
    eval_offsets = parse_offsets(args.eval_offsets)
    bits_l1, bits_l2, dependency = resolve_bits(args.bits)
    if args.smoke:
        configs = [configs[0]]
        args.tok_epochs = 1
        args.gpt_epochs = 1
        args.eval_days = 2
        eval_offsets = (3,)
    if args.tok_epochs <= 0 or args.gpt_epochs <= 0:
        raise ValueError("Training epochs must be positive")
    if any(offset + args.eval_days > HOLDOUT_OFFSET for offset in eval_offsets):
        raise ValueError("Evaluation may not enter the sealed holdout")

    temporary_root: Path | None = None
    if args.smoke:
        temporary_root = Path(
            tempfile.mkdtemp(prefix="kronos_exp02_smoke_")
        ).resolve()
        layout = StudyLayout.create(
            temporary_root / "weights", temporary_root / "results"
        )
    else:
        layout = StudyLayout.create(args.weights_root, args.results_root)
        free_gb = shutil.disk_usage(layout.weights_root).free / (1024**3)
        if free_gb < 10:
            raise RuntimeError(
                f"At least 10 GiB free is required; {free_gb:.1f} GiB remains"
            )
    settings = build_settings(
        configs=configs,
        bits_l1=bits_l1,
        bits_l2=bits_l2,
        dependency=dependency,
        tok_epochs=args.tok_epochs,
        gpt_epochs=args.gpt_epochs,
        batch_tokens=args.batch_tokens,
        eval_offsets=eval_offsets,
        eval_days=args.eval_days,
        eval_batch_size=args.eval_batch_size,
        smoke=args.smoke,
    )
    prepare_manifest(layout.results_root, settings, layout)
    update_manifest(
        layout.results_root,
        status="running",
        started_at_utc=utc_now(),
        completed_configs=[],
    )
    feature_cache = layout.weights_root / "shared" / "tokenizer_features"
    eval_cache = layout.weights_root / "shared" / "eval_cache"
    feature_cache.mkdir(parents=True, exist_ok=True)
    eval_cache.mkdir(parents=True, exist_ok=True)
    print(f"Weights/cache: {layout.weights_root}", flush=True)
    print(f"Downloadable results: {layout.results_root}", flush=True)
    print(
        f"Bits from Exp 01: {bits_l1}+{bits_l2} "
        f"(coarse={2 ** bits_l1}, fine={2 ** bits_l2}, "
        f"joint={2 ** (bits_l1 + bits_l2)}); source="
        f"{dependency['source']}",
        flush=True,
    )
    print(
        f"Tokenizer configs: {[config_key(*value) for value in configs]}",
        flush=True,
    )

    completed: list[str] = []
    succeeded = False
    try:
        for index, (embedding_dim, hidden_dim) in enumerate(configs, start=1):
            print(
                f"\n{'=' * 72}\n"
                f"[{index}/{len(configs)}] Tokenizer "
                f"{config_key(embedding_dim, hidden_dim)}\n"
                f"{'=' * 72}",
                flush=True,
            )
            run_one_config(
                layout,
                feature_cache,
                eval_cache,
                settings,
                embedding_dim,
                hidden_dim,
            )
            completed.append(config_key(embedding_dim, hidden_dim))
            rows = write_combined_summary(layout, configs, settings)
            update_manifest(
                layout.results_root,
                status="running",
                completed_configs=completed,
                combined_rows=len(rows),
                last_progress_at_utc=utc_now(),
            )
        if ANALYSIS_SCRIPT.is_file() and not args.smoke:
            analysis_env = os.environ.copy()
            analysis_env.pop("KRONOS_PREVIEW_OVERRIDE_JSON", None)
            PIPE.run_command(
                [
                    sys.executable,
                    str(ANALYSIS_SCRIPT),
                    "--root",
                    str(layout.results_root),
                ],
                env=analysis_env,
                log_path=layout.results_root / "analysis.log",
                label="Exp 02 analysis",
            )
        update_manifest(
            layout.results_root,
            status="completed",
            completed_at_utc=utc_now(),
            completed_configs=completed,
            combined_rows=len(
                PIPE.load_json(
                    layout.results_root / "combined_epoch_summary.json"
                )
            ),
        )
        write_download_manifest(layout)
        succeeded = True
        print(
            f"Completed Exp 02: {layout.results_root}", flush=True
        )
        return 0
    except BaseException as exc:
        update_manifest(
            layout.results_root,
            status="failed",
            failed_at_utc=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            completed_configs=completed,
        )
        raise
    finally:
        if temporary_root is not None:
            if succeeded:
                shutil.rmtree(temporary_root, ignore_errors=False)
            else:
                print(f"Smoke artifacts retained at {temporary_root}")


if __name__ == "__main__":
    raise SystemExit(main())
