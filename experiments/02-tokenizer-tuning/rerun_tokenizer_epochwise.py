"""Run the current tokenizer-architecture sweep with epoch-wise GPT evaluation.

This is the canonical Exp 02 entry point.  It consumes the bit-width decision
from the completed Exp 01 rerun, trains six tokenizer architectures, then
trains and evaluates one 50-epoch CE+AdamW GPT for every tokenizer.  Every GPT
epoch is retained and evaluated on the same four pre-holdout windows used by
Exp 01.

The historical ``sweep_tokenizer*.py`` outputs are never reused or overwritten.
All current outputs live under ``rerun_seed42`` and every stage is resumable.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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
EXP01_DIR = ROOT / "experiments" / "01-bitsweep"
EXP01_ROOT = EXP01_DIR / "rerun_seed42"
EXP01_SELECTION = EXP01_ROOT / "selection.json"
EXP01_PIPELINE_PATH = EXP01_DIR / "rerun_bits_epochwise.py"
TOKENIZER_SCRIPT = ROOT / "train_tokenizer.py"
GPT_SCRIPT = ROOT / "train_base.py"
TOKENIZER_EVAL_SCRIPT = EXP01_DIR / "evaluate_tokenizer.py"
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "c-hpo" / "evaluate_epoch_trajectory.py"
)
ANALYSIS_SCRIPT = EXPERIMENT_DIR / "analyze_tokenizer_epochwise.py"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "rerun_seed42"

SEED = 42
DEFAULT_CONFIGS = (
    (48, 192),
    (48, 256),
    (64, 192),
    (64, 256),
    (96, 192),
    (96, 256),
)
VALIDATION_OFFSETS = (0, 100, 200, 300)
HOLDOUT_OFFSET = 400


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
            "Complete and analyze Exp 01 first, or pass --bits L1+L2."
        )
    payload = PIPE.load_json(EXP01_SELECTION)
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
    output_root: Path, embedding_dim: int, hidden_dim: int
) -> dict[str, Path]:
    directory = (
        output_root
        / "configs"
        / config_directory_name(embedding_dim, hidden_dim)
    )
    return {
        "directory": directory,
        "override": directory / "override.json",
        "run": directory / "run.json",
        "tokenizer": directory / "tokenizer.pt",
        "tokenizer_resume": directory / "tokenizer.pt.ckpt",
        "tokenizer_metrics": directory / "tokenizer_metrics.json",
        "model": directory / "model.pt",
        "model_resume": directory / "model.pt.ckpt",
        "checkpoint_index": directory / "model_checkpoints.json",
        "trajectory": directory / "epoch_trajectory",
        "logs": directory / "logs",
        "tokenizer_log": directory / "logs" / "tokenizer.log",
        "tokenizer_eval_log": directory / "logs" / "tokenizer_eval.log",
        "gpt_log": directory / "logs" / "gpt.log",
        "trajectory_log": directory / "logs" / "trajectory.log",
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
                "No weighted score. Tokenizer reconstruction, prediction "
                "quality, and prediction behaviour remain separate."
            ),
        },
        "smoke": smoke,
    }


def source_hashes() -> dict[str, str]:
    files = (
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
        "experiments/01-bitsweep/rerun_bits_epochwise.py",
        "experiments/02-tokenizer-tuning/rerun_tokenizer_epochwise.py",
        "experiments/02-tokenizer-tuning/analyze_tokenizer_epochwise.py",
        "experiments/04/c-hpo/evaluate_epoch_trajectory.py",
    )
    return {
        relative: PIPE.file_sha256(ROOT / relative)
        for relative in files
        if (ROOT / relative).is_file()
    }


def prepare_manifest(
    output_root: Path, settings: dict[str, Any]
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
    path = output_root / "study_manifest.json"
    if path.exists():
        payload = PIPE.load_json(path)
        if payload.get("study_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{path} has a different protocol/source fingerprint. "
                "Archive it instead of mixing results."
            )
        return payload
    payload = {
        "experiment": "Exp 02 Tokenizer architecture rerun",
        "design": "full-data tokenizer grid plus epoch-wise downstream GPT",
        "status": "planned",
        "created_at_utc": utc_now(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "study_fingerprint": fingerprint,
        "settings": settings,
        "implementation_sha256": implementation,
        "dataset": dataset,
        "historical_outputs_overwritten": False,
    }
    PIPE.atomic_write_json(path, payload)
    return payload


def update_manifest(output_root: Path, **updates: Any) -> dict[str, Any]:
    path = output_root / "study_manifest.json"
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
    directory: Path,
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
            "accumulation_steps": 32,
            "save_dir": str((directory / "cache").resolve()),
        },
    }
    if path.exists() and PIPE.load_json(path) != payload:
        raise RuntimeError(f"Refusing to overwrite incompatible {path}")
    PIPE.atomic_write_json(path, payload)
    return payload


def run_one_config(
    output_root: Path,
    feature_cache: Path,
    settings: dict[str, Any],
    embedding_dim: int,
    hidden_dim: int,
) -> None:
    paths = config_paths(output_root, embedding_dim, hidden_dim)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    write_override(
        paths["override"],
        paths["directory"],
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
            "--early_stop_patience",
            "0",
            "--history_per_epoch",
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
    output_root: Path, configs: list[tuple[int, int]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for embedding_dim, hidden_dim in configs:
        paths = config_paths(output_root, embedding_dim, hidden_dim)
        summary_path = paths["trajectory"] / "epoch_summary.json"
        if not summary_path.is_file() or not paths["tokenizer_metrics"].is_file():
            continue
        tokenizer = PIPE.load_json(paths["tokenizer_metrics"])
        for item in PIPE.load_json(summary_path):
            row = dict(item)
            row.update(
                {
                    "config": config_key(embedding_dim, hidden_dim),
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "bits_l1": settings["bits_l1"],
                    "bits_l2": settings["bits_l2"],
                    "joint_vocab": 2
                    ** (settings["bits_l1"] + settings["bits_l2"]),
                    "tokenizer_mae": tokenizer["mae"],
                    "tokenizer_rmse": tokenizer["rmse"],
                    "tokenizer_joint_unique": tokenizer["joint_codes"][
                        "n_unique"
                    ],
                    "tokenizer_joint_entropy_bits": tokenizer["joint_codes"][
                        "entropy_bits"
                    ],
                    "tokenizer_joint_utilization": tokenizer["joint_codes"][
                        "utilization"
                    ],
                    "tokenizer_joint_collapse": tokenizer["joint_codes"][
                        "collapse_rate"
                    ],
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
    PIPE.atomic_write_json(output_root / "combined_epoch_summary.json", rows)
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        temporary = output_root / "combined_epoch_summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_root / "combined_epoch_summary.csv")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current epoch-wise tokenizer architecture sweep"
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--configs", default="")
    parser.add_argument(
        "--bits",
        default="",
        help="L1+L2; default reads Exp 01 rerun selection.json",
    )
    parser.add_argument("--tok_epochs", type=int, default=100)
    parser.add_argument("--gpt_epochs", type=int, default=50)
    parser.add_argument("--batch_tokens", type=int, default=12288)
    parser.add_argument(
        "--eval_offsets",
        default=",".join(str(value) for value in VALIDATION_OFFSETS),
    )
    parser.add_argument("--eval_days", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=4)
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
        output_root = temporary_root
    else:
        output_root = args.output_root.resolve()
        free_gb = shutil.disk_usage(output_root.parent).free / (1024**3)
        if free_gb < 10:
            raise RuntimeError(
                f"At least 10 GiB free is required; {free_gb:.1f} GiB remains"
            )
    output_root.mkdir(parents=True, exist_ok=True)

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
    prepare_manifest(output_root, settings)
    update_manifest(
        output_root,
        status="running",
        started_at_utc=utc_now(),
        completed_configs=[],
    )
    feature_cache = output_root / "shared" / "tokenizer_features"
    feature_cache.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Bits from Exp 01: {bits_l1}+{bits_l2}", flush=True)
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
                output_root,
                feature_cache,
                settings,
                embedding_dim,
                hidden_dim,
            )
            completed.append(config_key(embedding_dim, hidden_dim))
            rows = write_combined_summary(output_root, configs, settings)
            update_manifest(
                output_root,
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
                    str(output_root),
                ],
                env=analysis_env,
                log_path=output_root / "analysis.log",
                label="Exp 02 analysis",
            )
        update_manifest(
            output_root,
            status="completed",
            completed_at_utc=utc_now(),
            completed_configs=completed,
            combined_rows=len(
                PIPE.load_json(output_root / "combined_epoch_summary.json")
            ),
        )
        succeeded = True
        print(f"Completed Exp 02: {output_root}", flush=True)
        return 0
    except BaseException as exc:
        update_manifest(
            output_root,
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
