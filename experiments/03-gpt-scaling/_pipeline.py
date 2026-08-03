"""Shared full-data, full-sequence GPT capacity experiment pipeline.

The tokenizer is inherited from the Exp 02 selection. Capacity configurations
share one seed, the CE+AdamW recipe, the full stock set, complete sequences,
and four pre-holdout evaluation windows. Every epoch checkpoint is retained
and evaluated. Checkpoints/caches and downloadable diagnostics use separate
roots.

Protocol guarantees:

- CE + AdamW, no early stopping, and epoch-wise multi-window evaluation.
- Large configurations train on full sequences via gradient checkpointing.
- All architectures share one token cache (identical tokenizer; the cache is
  validated by a hash of tokenizer weights in ``data_processor.pack_stocks_v2``),
  instead of building five identical per-config caches.
- Every JSON state write retries on Windows ``PermissionError`` file locks —
  the exact failure that corrupted the Exp 02 finalization metadata.

Shared helpers are imported from the Exp 01 pipeline module so completion
checks, subprocess handling, and fingerprints stay identical across the
experiment line.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
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
)

EXP01_PIPELINE_PATH = (
    ROOT / "experiments" / "01-bitsweep" / "run_bitsweep.py"
)
_, EXP02_ROOT = default_study_roots("02-tokenizer-tuning", seed=42)
EXP02_SELECTION = EXP02_ROOT / "selection.json"
GPT_SCRIPT = ROOT / "train_base.py"
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
)
SEED = 42
HOLDOUT_OFFSET = 400
EVAL_DAYS = 1
# Full-coverage, single-day-resolution evaluation: tile [0, HOLDOUT_OFFSET) with
# contiguous EVAL_DAYS-day windows (offsets 0, 1, ..., 399) so every pre-holdout
# trading day is scored as its own window. Offset HOLDOUT_OFFSET+ stays sealed.
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, EVAL_DAYS))


def _load_exp01_pipeline():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp01_pipeline_for_exp03", EXP01_PIPELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXP01_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPE = _load_exp01_pipeline()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write that survives transient Windows file locks.

    Antivirus/indexer handles intermittently hold target files open on
    Windows; a bare ``os.replace`` then raises ``PermissionError`` (this
    corrupted the Exp 02 finalization).  Retry briefly before giving up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def parse_offsets(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise ValueError("Validation offsets must be non-empty and non-negative")
    if len(set(values)) != len(values):
        raise ValueError("Validation offsets must be unique")
    return values


def tokenizer_metadata(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    bits = config.get("bits_per_quantizer")
    if not isinstance(bits, (list, tuple)) or len(bits) != 2:
        raise RuntimeError(f"Tokenizer bit metadata is missing in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": PIPE.file_sha256(path),
        "bits_l1": int(bits[0]),
        "bits_l2": int(bits[1]),
        "embedding_dim": int(config["embedding_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
    }


def resolve_tokenizer(raw: str) -> dict[str, Any]:
    if raw.strip():
        path = Path(raw).resolve()
        source = {"source": "command_line", "selection_path": None}
    else:
        if not EXP02_SELECTION.is_file():
            raise RuntimeError(
                f"Exp 02 selection is missing: {EXP02_SELECTION}. "
                "Complete and analyze Exp 02 first, or pass --tokenizer."
            )
        selection = PIPE.load_json(EXP02_SELECTION)
        if not selection.get("upstream_eligible", False):
            raise RuntimeError(
                f"Exp 02 selection has not been recorded after review: "
                f"{EXP02_SELECTION}"
            )
        selected = selection.get("selected", selection)
        path = Path(selected["tokenizer_path"]).resolve()
        source = {
            "source": "exp02_selection",
            "selection_path": str(EXP02_SELECTION.resolve()),
            "selection_sha256": PIPE.file_sha256(EXP02_SELECTION),
            "selection": selection,
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = tokenizer_metadata(path)
    metadata.update(source)
    return metadata


def config_paths(
    weights_root: Path, results_root: Path, name: str
) -> dict[str, Path]:
    weights = weights_root / "configs" / name
    results = results_root / "configs" / name
    return {
        "directory": results,
        "weights_directory": weights,
        "override": results / "override.json",
        "run": results / "run.json",
        "model": weights / "model.pt",
        "model_resume": weights / "model.pt.ckpt",
        "checkpoint_index": results / "model_checkpoints.json",
        "history": results / f"history_arch_{name}.json",
        "trajectory": results / "epoch_trajectory",
        "logs": results / "logs",
        "gpt_log": results / "logs" / "gpt.log",
        "trajectory_log": results / "logs" / "trajectory.log",
    }


def build_settings(
    *,
    configs: list[dict[str, Any]],
    tokenizer: dict[str, Any],
    gpt_epochs: int,
    eval_offsets: tuple[int, ...],
    eval_days: int,
    smoke: bool,
) -> dict[str, Any]:
    config_settings = []
    for value in configs:
        item = dict(value)
        if smoke:
            item["batch_tokens"] = 2048
            item["eval_batch_size"] = 1
        config_settings.append(item)
    return {
        "seed": SEED,
        "exp02_dependency": tokenizer,
        "configs": config_settings,
        "shared_token_cache": True,
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
            "batch_cap": 64,
            "curriculum": False,
            "early_stop_patience": 0,
            "retain_every_epoch": True,
            "full_sequences_for_all_architectures": not smoke,
        },
        "evaluation": {
            "offsets": list(eval_offsets),
            "days_per_window": eval_days,
            "n_stocks": 6 if smoke else 0,
            "sample_strategy": "shortest" if smoke else "random",
            "health_gate": {
                "max_daily_collapse_rate": 1.0 if smoke else 0.35,
                "min_daily_unique_tokens": 1 if smoke else 32,
            },
            "holdout_offset": HOLDOUT_OFFSET,
            "holdout_used": False,
            "selection_policy": (
                "No weighted score. Capacity, prediction quality, and "
                "prediction behaviour remain separate."
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
        "train_base.py",
        "eval_helpers.py",
        "model/kronos_preview.py",
        "model/layers.py",
        "experiments/01-bitsweep/run_bitsweep.py",
        "experiments/02-tokenizer-tuning/analyze_tokenizer_epochwise.py",
        "experiments/03-gpt-scaling/_pipeline.py",
        "experiments/03-gpt-scaling/analyze_gpt_capacity.py",
        "experiments/04/b-hpo/evaluate_epoch_trajectory.py",
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
                f"{path} has a different protocol/source fingerprint; "
                "refusing to mix runs. Move the old output root aside first."
            )
        return payload
    payload = {
        "experiment": "Exp 03 GPT architecture scaling",
        "design": "full-data full-sequence epoch-wise architecture sweep",
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
    }
    atomic_write_json(path, payload)
    return payload


def update_manifest(results_root: Path, **updates: Any) -> dict[str, Any]:
    path = results_root / "study_manifest.json"
    payload = PIPE.load_json(path)
    payload.update(updates)
    atomic_write_json(path, payload)
    return payload


def update_config_run(
    paths: dict[str, Path], config: dict[str, Any], **updates: Any
) -> dict[str, Any]:
    if paths["run"].exists():
        payload = PIPE.load_json(paths["run"])
    else:
        payload = {
            "config": config["name"],
            "architecture": config,
            "status": "planned",
            "stages": {},
            "created_at_utc": utc_now(),
        }
    payload.update(updates)
    payload["updated_at_utc"] = utc_now()
    atomic_write_json(paths["run"], payload)
    return payload


def write_override(
    path: Path,
    shared_cache_root: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = settings["exp02_dependency"]
    payload = {
        "DataConfig": {
            "random_seed": SEED,
            "max_stocks": settings["gpt"]["max_stocks"],
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
            "dim": config["dim"],
            "depth": config["depth"],
            "heads": config["heads"],
            "num_kv_heads": config["kv_heads"],
            "ffn_multiplier": config["ffn_multiplier"],
        },
        "TrainingConfig": {
            "random_seed": SEED,
            "warmup_ratio": settings["gpt"]["warmup_ratio"],
            "batch_size": 1,
            "accumulation_steps": settings["gpt"].get(
                "accumulation_steps", 32
            ),
            "use_gradient_checkpointing": config["gradient_checkpointing"],
            # One shared cache root for every architecture: the tokenizer is
            # identical, and pack_stocks_v2 validates entries by a hash of
            # tokenizer weights, so sharing is exact and saves 4x re-encoding.
            "save_dir": str(shared_cache_root.resolve()),
        },
    }
    if path.exists() and PIPE.load_json(path) != payload:
        raise RuntimeError(f"Refusing to overwrite incompatible {path}")
    atomic_write_json(path, payload)
    return payload


def run_one_config(
    layout: StudyLayout,
    shared_cache_root: Path,
    settings: dict[str, Any],
    config: dict[str, Any],
) -> None:
    name = str(config["name"])
    paths = config_paths(
        layout.weights_root, layout.results_root, name
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["weights_directory"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    write_override(paths["override"], shared_cache_root, config, settings)
    run_state = update_config_run(
        paths, config, status="running", started_at_utc=utc_now()
    )
    env = os.environ.copy()
    env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(paths["override"].resolve())

    if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
        run_state["stages"]["gpt"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        atomic_write_json(paths["run"], run_state)
        command = [
            sys.executable,
            str(GPT_SCRIPT),
            "--save_path",
            str(paths["model"]),
            "--tokenizer_path",
            str(settings["exp02_dependency"]["path"]),
            "--epochs",
            str(settings["gpt"]["epochs"]),
            "--tag",
            f"arch_{name}",
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
            str(config["batch_tokens"]),
            "--batch_cap",
            str(settings["gpt"]["batch_cap"]),
            "--early_stop_patience",
            "0",
            "--history_per_epoch",
            "--metrics_dir",
            str(paths["directory"]),
            "--dim",
            str(config["dim"]),
            "--depth",
            str(config["depth"]),
            "--heads",
            str(config["heads"]),
            "--num_kv_heads",
            str(config["kv_heads"]),
            "--ffn_multiplier",
            str(config["ffn_multiplier"]),
        ]
        if config["gradient_checkpointing"]:
            command.append("--gradient_checkpointing")
        controlled_loader_seed = settings["gpt"].get(
            "controlled_loader_seed"
        )
        if controlled_loader_seed is not None:
            command.extend(
                [
                    "--controlled_loader_seed",
                    str(controlled_loader_seed),
                ]
            )
        if settings["gpt"].get("exact_accumulation_boundaries", False):
            command.append("--exact_accumulation_boundaries")
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["gpt_log"],
            label=f"{name} GPT",
        )
        if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(f"GPT completion check failed for {name}")
    run_state = update_config_run(paths, config)
    run_state["stages"]["gpt"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "checkpoint_index": str(paths["checkpoint_index"].resolve()),
    }
    atomic_write_json(paths["run"], run_state)

    expected_epochs = settings["gpt"]["epochs"]

    def trajectory_is_complete() -> bool:
        return (
            PIPE.trajectory_is_complete(paths, expected_epochs)
            and len(
                list(
                    paths["trajectory"].glob(
                        "token_distributions_epoch_*.npz"
                    )
                )
            )
            == expected_epochs
            and len(
                list(
                    paths["trajectory"].glob(
                        "prediction_records_epoch_*.npz"
                    )
                )
            )
            == expected_epochs
        )

    if not trajectory_is_complete():
        run_state = update_config_run(paths, config)
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
            str(settings["exp02_dependency"]["path"]),
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
            str(config["eval_batch_size"]),
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
            f"Exp 03 GPT scaling {name}",
        ]
        prepared_cache_dir = evaluation.get("prepared_cache_dir")
        if prepared_cache_dir:
            command.extend(
                ["--prepared_cache_dir", str(prepared_cache_dir)]
            )
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["trajectory_log"],
            label=f"{name} epoch trajectory",
        )
        if not trajectory_is_complete():
            raise RuntimeError(
                f"Trajectory completion check failed for {name}"
            )
    stages = update_config_run(paths, config).get("stages", {})
    stages["epoch_evaluation"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "output": str(paths["trajectory"].resolve()),
    }
    update_config_run(
        paths,
        config,
        status="completed",
        completed_at_utc=utc_now(),
        stages=stages,
    )


def parameter_count(model_path: Path) -> int:
    import torch

    from eval_helpers import load_gpt

    model = load_gpt(model_path, torch.device("cpu"))
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return int(count)


def write_combined_summary(
    layout: StudyLayout,
    configs: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        paths = config_paths(
            layout.weights_root,
            layout.results_root,
            str(config["name"]),
        )
        summary_path = paths["trajectory"] / "epoch_summary.json"
        if not summary_path.is_file() or not paths["model"].is_file():
            continue
        count = parameter_count(paths["model"])
        for item in PIPE.load_json(summary_path):
            row = dict(item)
            row.update(
                {
                    "config": config["name"],
                    "dim": config["dim"],
                    "depth": config["depth"],
                    "heads": config["heads"],
                    "kv_heads": config["kv_heads"],
                    "ffn_multiplier": config["ffn_multiplier"],
                    "gradient_checkpointing": config["gradient_checkpointing"],
                    "batch_tokens": config["batch_tokens"],
                    "parameter_count": count,
                    "tokenizer_sha256": settings["exp02_dependency"]["sha256"],
                }
            )
            rows.append(row)
    order = {config["name"]: index for index, config in enumerate(configs)}
    rows.sort(key=lambda row: (order[row["config"]], row["epoch"]))
    atomic_write_json(
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
