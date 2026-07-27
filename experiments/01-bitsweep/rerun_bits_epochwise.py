"""Exp 01 BitSweep rerun with checkpoint-level, multi-window evaluation.

The historical sweep is kept intact for provenance.  This orchestrator runs a
new isolated study using the current production tokenizer architecture and the
controlled CE+AdamW recipe selected by Exp 04-A/B.

Formal protocol:
  * seed=42 only
  * ten (L1, L2) configurations
  * tokenizer: embedding_dim=64, hidden_dim=192, 100 epochs
  * GPT: CE + AdamW, full data, 50 epochs
  * every GPT epoch is saved
  * every saved epoch is evaluated on offsets 0/100/200/300, 20 days each
  * offset 400+ is a sealed holdout and is never used here

All stages are resumable and all formal outputs live below ``rerun_seed42`` so
the 2026-07-11 historical report cannot be overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[2]
EXPECTED_PYTHON = Path(r"D:\conda_envs\llm-t\Scripts\python.exe")
TOKENIZER_SCRIPT = ROOT / "train_tokenizer.py"
GPT_SCRIPT = ROOT / "train_base.py"
TOKENIZER_EVAL_SCRIPT = EXPERIMENT_DIR / "evaluate_tokenizer.py"
ANALYSIS_SCRIPT = EXPERIMENT_DIR / "analyze_bitsweep_epochwise.py"
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "c-hpo" / "evaluate_epoch_trajectory.py"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "rerun_seed42"

SEED = 42
ALL_CONFIGS = (
    (6, 6),
    (7, 6),
    (7, 7),
    (8, 6),
    (8, 7),
    (8, 8),
    (9, 6),
    (9, 7),
    (9, 8),
    (9, 9),
)
VALIDATION_OFFSETS = (0, 100, 200, 300)
HOLDOUT_OFFSET = 400

IMPLEMENTATION_FILES = (
    "config.py",
    "data_processor.py",
    "training_utils.py",
    "train_tokenizer.py",
    "train_base.py",
    "eval_helpers.py",
    "model/__init__.py",
    "model/tokenizer.py",
    "model/kronos_preview.py",
    "model/layers.py",
    "experiments/01-bitsweep/evaluate_tokenizer.py",
    "experiments/01-bitsweep/analyze_bitsweep_epochwise.py",
    "experiments/01-bitsweep/rerun_bits_epochwise.py",
    "experiments/04/c-hpo/evaluate_epoch_trajectory.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_signature() -> dict[str, Any]:
    files = sorted((ROOT / "dataset").glob("*.csv"))
    rows = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "metadata_sha256": canonical_sha256(rows),
    }


def parse_configs(specification: str) -> list[tuple[int, int]]:
    if not specification.strip():
        return list(ALL_CONFIGS)
    configs: list[tuple[int, int]] = []
    for raw in specification.split(","):
        parts = raw.strip().split("+")
        if len(parts) != 2:
            raise ValueError(f"Invalid bit configuration: {raw!r}")
        config = (int(parts[0]), int(parts[1]))
        if config not in ALL_CONFIGS:
            raise ValueError(
                f"{config[0]}+{config[1]} is outside the preregistered sweep"
            )
        if config not in configs:
            configs.append(config)
    return configs


def parse_offsets(specification: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in specification.split(","))
    if not values or any(value < 0 for value in values):
        raise ValueError("Evaluation offsets must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError("Evaluation offsets must be unique")
    return values


def verify_runtime() -> None:
    actual = Path(sys.executable).resolve()
    if actual != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"This study must use {EXPECTED_PYTHON}; current Python is {actual}"
        )
    if sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError(
            f"Expected Python 3.12.10, got {sys.version.split()[0]}"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal BitSweep rerun")


def config_key(l1: int, l2: int) -> str:
    return f"{l1}+{l2}"


def config_directory_name(l1: int, l2: int) -> str:
    return f"bits_{l1:02d}_{l2:02d}"


def config_paths(output_root: Path, l1: int, l2: int) -> dict[str, Path]:
    directory = output_root / "configs" / config_directory_name(l1, l2)
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
    tok_epochs: int,
    gpt_epochs: int,
    embedding_dim: int,
    hidden_dim: int,
    batch_tokens: int,
    eval_offsets: tuple[int, ...],
    eval_days: int,
    eval_batch_size: int,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "seed": SEED,
        "configs": [config_key(*config) for config in configs],
        "tokenizer": {
            "epochs": tok_epochs,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "batch_size": 256 if smoke else 8192,
            "learning_rate": 1e-4,
            "scheduler": "warmup_5pct_cosine",
            "early_stop_patience": 0,
            "validation_every": 1 if smoke else 5,
            "selection": "lowest tokenizer validation loss",
        },
        "gpt": {
            "epochs": gpt_epochs,
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
            "dense_date_min_coverage": 0.80,
            "health_gate": {
                "max_daily_collapse_rate": 1.0 if smoke else 0.35,
                "min_daily_unique_tokens": 1 if smoke else 32,
            },
            "holdout_offset": HOLDOUT_OFFSET,
            "holdout_used": False,
            "selection_policy": (
                "No weighted score. Inspect Collapse/AmpRatio/Unique and "
                "DA/MAPE/daily-RankIC as separate metric groups."
            ),
        },
        "smoke": smoke,
    }


def source_hashes() -> dict[str, str]:
    return {
        relative: file_sha256(ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }


def prepare_manifest(
    output_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    implementation = source_hashes()
    data = dataset_signature()
    fingerprint = canonical_sha256(
        {
            "settings": settings,
            "implementation_sha256": implementation,
            "dataset": data,
        }
    )
    path = output_root / "study_manifest.json"
    if path.exists():
        manifest = load_json(path)
        if manifest.get("study_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{path} belongs to a different study definition or source "
                "state. Archive it instead of mixing results."
            )
        return manifest

    manifest = {
        "experiment": "Exp 01 BitSweep rerun",
        "design": "full-data epoch-wise bit-width sweep",
        "status": "planned",
        "created_at_utc": utc_now(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "study_fingerprint": fingerprint,
        "settings": settings,
        "implementation_sha256": implementation,
        "dataset": data,
        "historical_outputs_overwritten": False,
    }
    atomic_write_json(path, manifest)
    return manifest


def update_manifest(output_root: Path, **updates: Any) -> dict[str, Any]:
    path = output_root / "study_manifest.json"
    manifest = load_json(path)
    manifest.update(updates)
    atomic_write_json(path, manifest)
    return manifest


def update_config_run(
    paths: dict[str, Path],
    l1: int,
    l2: int,
    **updates: Any,
) -> dict[str, Any]:
    payload = (
        load_json(paths["run"])
        if paths["run"].exists()
        else {
            "config": config_key(l1, l2),
            "bits_l1": l1,
            "bits_l2": l2,
            "status": "planned",
            "stages": {},
            "created_at_utc": utc_now(),
        }
    )
    payload.update(updates)
    payload["updated_at_utc"] = utc_now()
    atomic_write_json(paths["run"], payload)
    return payload


def write_override(
    path: Path,
    directory: Path,
    l1: int,
    l2: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "DataConfig": {
            "random_seed": SEED,
            "max_stocks": settings["gpt"]["max_stocks"],
        },
        "TokenizerConfig": {
            "hidden_dim": settings["tokenizer"]["hidden_dim"],
            "embedding_dim": settings["tokenizer"]["embedding_dim"],
            "bits_l1": l1,
            "bits_l2": l2,
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
    if path.exists() and load_json(path) != payload:
        raise RuntimeError(f"Refusing to overwrite incompatible {path}")
    atomic_write_json(path, payload)
    return payload


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = subprocess.list2cmdline(command)
    print(f"\n[{label}] {rendered}", flush=True)
    child_env = env.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    creationflags = (
        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[{utc_now()}] {label}\n{rendered}\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}; see {log_path}"
        )


def tokenizer_is_complete(
    paths: dict[str, Path],
    *,
    l1: int,
    l2: int,
    epochs: int,
    embedding_dim: int,
    hidden_dim: int,
) -> bool:
    if not paths["tokenizer"].is_file() or not paths["tokenizer_resume"].is_file():
        return False
    try:
        import torch

        model = torch.load(
            paths["tokenizer"], map_location="cpu", weights_only=False
        )
        resume = torch.load(
            paths["tokenizer_resume"], map_location="cpu", weights_only=False
        )
        config = model.get("config", {})
        bits = config.get("bits_per_quantizer")
        return bool(
            model.get("completed", False)
            and int(model.get("total_epochs", -1)) == epochs
            and list(bits) == [l1, l2]
            and int(config.get("embedding_dim", -1)) == embedding_dim
            and int(config.get("hidden_dim", -1)) == hidden_dim
            and int(resume.get("epoch", -1)) + 1 >= epochs
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def tokenizer_metrics_are_current(paths: dict[str, Path]) -> bool:
    if not paths["tokenizer_metrics"].is_file():
        return False
    try:
        payload = load_json(paths["tokenizer_metrics"])
        return bool(
            payload.get("status") == "completed"
            and payload.get("tokenizer_sha256")
            == file_sha256(paths["tokenizer"])
            and payload.get("n_rows", 0) > 0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def gpt_is_complete(paths: dict[str, Path], epochs: int) -> bool:
    required = (
        paths["model"].is_file()
        and paths["model_resume"].is_file()
        and paths["checkpoint_index"].is_file()
    )
    if not required:
        return False
    try:
        import torch

        index = load_json(paths["checkpoint_index"])
        rows = {
            int(item["epoch"]): item
            for item in index.get("checkpoints", [])
        }
        if set(range(1, epochs + 1)) - set(rows):
            return False
        for epoch in range(1, epochs + 1):
            checkpoint = Path(rows[epoch]["path"])
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size
                != int(rows[epoch]["size_bytes"])
            ):
                return False
        resume = torch.load(
            paths["model_resume"], map_location="cpu", weights_only=False
        )
        return int(resume.get("epoch", -1)) + 1 >= epochs
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def trajectory_is_complete(paths: dict[str, Path], epochs: int) -> bool:
    manifest_path = paths["trajectory"] / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
        completed = {int(value) for value in manifest.get("epochs_completed", [])}
        return bool(
            manifest.get("status") == "completed"
            and completed == set(range(1, epochs + 1))
            and all(
                (paths["trajectory"] / f"epoch_{epoch:03d}.json").is_file()
                for epoch in range(1, epochs + 1)
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_one_config(
    output_root: Path,
    shared_feature_cache: Path,
    settings: dict[str, Any],
    l1: int,
    l2: int,
) -> None:
    paths = config_paths(output_root, l1, l2)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    write_override(paths["override"], paths["directory"], l1, l2, settings)
    run_state = update_config_run(
        paths, l1, l2, status="running", started_at_utc=utc_now()
    )
    env = os.environ.copy()
    env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(paths["override"].resolve())

    tokenizer_complete = tokenizer_is_complete(
        paths,
        l1=l1,
        l2=l2,
        epochs=settings["tokenizer"]["epochs"],
        embedding_dim=settings["tokenizer"]["embedding_dim"],
        hidden_dim=settings["tokenizer"]["hidden_dim"],
    )
    if not tokenizer_complete:
        run_state["stages"]["tokenizer"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        atomic_write_json(paths["run"], run_state)
        command = [
            sys.executable,
            str(TOKENIZER_SCRIPT),
            "--save_path",
            str(paths["tokenizer"]),
            "--bits_l1",
            str(l1),
            "--bits_l2",
            str(l2),
            "--embedding_dim",
            str(settings["tokenizer"]["embedding_dim"]),
            "--hidden_dim",
            str(settings["tokenizer"]["hidden_dim"]),
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
            str(shared_feature_cache),
        ]
        run_command(
            command,
            env=env,
            log_path=paths["tokenizer_log"],
            label=f"{config_key(l1, l2)} tokenizer",
        )
        if not tokenizer_is_complete(
            paths,
            l1=l1,
            l2=l2,
            epochs=settings["tokenizer"]["epochs"],
            embedding_dim=settings["tokenizer"]["embedding_dim"],
            hidden_dim=settings["tokenizer"]["hidden_dim"],
        ):
            raise RuntimeError(
                f"Tokenizer completion check failed for {config_key(l1, l2)}"
            )
    run_state = update_config_run(paths, l1, l2)
    run_state["stages"]["tokenizer"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "checkpoint": str(paths["tokenizer"].resolve()),
        "sha256": file_sha256(paths["tokenizer"]),
    }
    atomic_write_json(paths["run"], run_state)

    validation_features = (
        shared_feature_cache / "tok_feat_val_2024-02-01.npz"
    )
    if not tokenizer_metrics_are_current(paths):
        command = [
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
        ]
        run_command(
            command,
            env=env,
            log_path=paths["tokenizer_eval_log"],
            label=f"{config_key(l1, l2)} tokenizer diagnostics",
        )
    run_state = update_config_run(paths, l1, l2)
    run_state["stages"]["tokenizer_diagnostics"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "metrics": str(paths["tokenizer_metrics"].resolve()),
    }
    atomic_write_json(paths["run"], run_state)

    if not gpt_is_complete(paths, settings["gpt"]["epochs"]):
        run_state = update_config_run(paths, l1, l2)
        run_state["stages"]["gpt"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        atomic_write_json(paths["run"], run_state)
        tag = f"bitsweep_{l1}_{l2}"
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
        run_command(
            command,
            env=env,
            log_path=paths["gpt_log"],
            label=f"{config_key(l1, l2)} GPT",
        )
        if not gpt_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(
                f"GPT completion check failed for {config_key(l1, l2)}"
            )
    run_state = update_config_run(paths, l1, l2)
    run_state["stages"]["gpt"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "checkpoint_index": str(paths["checkpoint_index"].resolve()),
    }
    atomic_write_json(paths["run"], run_state)

    if not trajectory_is_complete(paths, settings["gpt"]["epochs"]):
        run_state = update_config_run(paths, l1, l2)
        run_state["stages"]["epoch_evaluation"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        atomic_write_json(paths["run"], run_state)
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
            f"Exp 01 BitSweep {config_key(l1, l2)}",
        ]
        run_command(
            command,
            env=env,
            log_path=paths["trajectory_log"],
            label=f"{config_key(l1, l2)} epoch trajectory",
        )
        if not trajectory_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(
                f"Trajectory completion check failed for {config_key(l1, l2)}"
            )
    update_config_run(
        paths,
        l1,
        l2,
        status="completed",
        completed_at_utc=utc_now(),
        stages={
            **update_config_run(paths, l1, l2).get("stages", {}),
            "epoch_evaluation": {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "epochs": settings["gpt"]["epochs"],
                "output": str(paths["trajectory"].resolve()),
            },
        },
    )


def write_combined_summary(
    output_root: Path,
    configs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for l1, l2 in configs:
        paths = config_paths(output_root, l1, l2)
        summary_path = paths["trajectory"] / "epoch_summary.json"
        if not summary_path.is_file() or not paths["tokenizer_metrics"].is_file():
            continue
        tokenizer = load_json(paths["tokenizer_metrics"])
        for item in load_json(summary_path):
            row = dict(item)
            row.update(
                {
                    "config": config_key(l1, l2),
                    "bits_l1": l1,
                    "bits_l2": l2,
                    "theoretical_bits": l1 + l2,
                    "coarse_vocab": 2 ** l1,
                    "fine_vocab": 2 ** l2,
                    "joint_vocab": 2 ** (l1 + l2),
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
                }
            )
            rows.append(row)
    rows.sort(key=lambda row: (row["bits_l1"], row["bits_l2"], row["epoch"]))
    atomic_write_json(output_root / "combined_epoch_summary.json", rows)
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        csv_path = output_root / "combined_epoch_summary.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current, epoch-wise BitSweep protocol"
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--configs", default="")
    parser.add_argument("--tok_epochs", type=int, default=100)
    parser.add_argument("--gpt_epochs", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--batch_tokens", type=int, default=12288)
    parser.add_argument(
        "--eval_offsets",
        default=",".join(str(value) for value in VALIDATION_OFFSETS),
    )
    parser.add_argument("--eval_days", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One-config, one-epoch end-to-end run in a temporary directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_runtime()
    configs = parse_configs(args.configs)
    eval_offsets = parse_offsets(args.eval_offsets)
    if args.smoke:
        configs = [configs[0]]
        args.tok_epochs = 1
        args.gpt_epochs = 1
        args.eval_days = 2
        eval_offsets = (3,)
    if args.tok_epochs <= 0 or args.gpt_epochs <= 0:
        raise ValueError("Training epochs must be positive")
    if args.eval_days <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Evaluation dimensions must be positive")
    if any(offset + args.eval_days > HOLDOUT_OFFSET for offset in eval_offsets):
        raise ValueError(
            f"Epoch-wise evaluation may not enter the sealed holdout at "
            f"offset {HOLDOUT_OFFSET}"
        )

    temporary_root: Path | None = None
    if args.smoke:
        temporary_root = Path(
            tempfile.mkdtemp(prefix="kronos_bitsweep_smoke_")
        ).resolve()
        output_root = temporary_root
    else:
        output_root = args.output_root.resolve()
        free_gb = shutil.disk_usage(output_root.parent).free / (1024 ** 3)
        if free_gb < 12:
            raise RuntimeError(
                f"At least 12 GiB free is required; only {free_gb:.1f} GiB remains"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    settings = build_settings(
        configs=configs,
        tok_epochs=args.tok_epochs,
        gpt_epochs=args.gpt_epochs,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
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
    shared_feature_cache = output_root / "shared" / "tokenizer_features"
    shared_feature_cache.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Configs: {[config_key(*value) for value in configs]}", flush=True)
    print(
        f"Protocol: tokenizer={args.tok_epochs}ep, GPT={args.gpt_epochs}ep, "
        f"windows={eval_offsets}x{args.eval_days}d, holdout_used=False",
        flush=True,
    )

    completed: list[str] = []
    succeeded = False
    try:
        for index, (l1, l2) in enumerate(configs, start=1):
            print(
                f"\n{'=' * 72}\n"
                f"[{index}/{len(configs)}] Bit configuration {l1}+{l2}\n"
                f"{'=' * 72}",
                flush=True,
            )
            run_one_config(
                output_root, shared_feature_cache, settings, l1, l2
            )
            completed.append(config_key(l1, l2))
            rows = write_combined_summary(output_root, configs)
            update_manifest(
                output_root,
                status="running",
                completed_configs=completed,
                combined_rows=len(rows),
                last_progress_at_utc=utc_now(),
            )
        if args.smoke:
            print(
                "\nRe-running the completed smoke config to verify all "
                "resume/cache guards...",
                flush=True,
            )
            run_one_config(
                output_root,
                shared_feature_cache,
                settings,
                configs[0][0],
                configs[0][1],
            )
        rows = write_combined_summary(output_root, configs)
        update_manifest(
            output_root,
            status="completed",
            completed_at_utc=utc_now(),
            completed_configs=completed,
            combined_rows=len(rows),
            holdout_used=False,
        )
        analysis_env = os.environ.copy()
        analysis_env.pop("KRONOS_PREVIEW_OVERRIDE_JSON", None)
        run_command(
            [
                sys.executable,
                str(ANALYSIS_SCRIPT),
                "--root",
                str(output_root),
            ],
            env=analysis_env,
            log_path=output_root / "analysis.log",
            label="BitSweep non-composite analysis",
        )
        update_manifest(
            output_root,
            analysis_status="completed",
            analysis_completed_at_utc=utc_now(),
        )
        print(
            f"\nBitSweep completed: {len(completed)} configs, "
            f"{len(rows)} config-epoch rows.",
            flush=True,
        )
        succeeded = True
    except BaseException as exc:
        update_manifest(
            output_root,
            status="failed",
            failed_at_utc=utc_now(),
            completed_configs=completed,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if temporary_root is not None and succeeded:
            # The path was created by tempfile.mkdtemp in this invocation.
            shutil.rmtree(temporary_root)
            print(f"Smoke artifacts removed: {temporary_root}", flush=True)
        elif temporary_root is not None:
            print(f"Failed smoke artifacts retained: {temporary_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
