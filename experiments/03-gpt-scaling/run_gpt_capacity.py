"""Exp 03: controlled GPT capacity interpolation around deep/xlarge.

The formal grid contains two clean one-factor lines:

* depth line: deep (256x4) -> depth6 -> depth8, with dim/heads/kv_heads fixed;
* width line: deep (256) -> width384 -> width512, with depth/kv_heads fixed.

Heads scale with width only to preserve head_dim=64.  Comparing width512_kv1
against the replayed xlarge (width512_kv2) then isolates the remaining KV-head
change.  A local loader seed plus exact accumulation blocks guarantee the same
data order and 4,160 optimizer steps for every architecture.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
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

PARENT_SCRIPT = (
    ROOT / "experiments" / "03-gpt-scaling" / "_pipeline.py"
)
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "c-hpo" / "evaluate_epoch_trajectory.py"
)
ANALYSIS_SCRIPT = EXPERIMENT_DIR / "analyze_gpt_capacity.py"
DEFAULT_WEIGHTS_ROOT, DEFAULT_RESULTS_ROOT = default_study_roots(
    "03-gpt-scaling", seed=42
)

SEED = 42
EPOCHS = 50
EVAL_DAYS = 1
HOLDOUT_OFFSET = 400
# Full-coverage, single-day-resolution evaluation. The pre-holdout region
# [0, HOLDOUT_OFFSET) is tiled with contiguous EVAL_DAYS-day windows (offsets
# 0, 1, ..., 399), so every pre-holdout trading day is scored as its own window.
# Offset HOLDOUT_OFFSET and beyond remain the sealed holdout.
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, EVAL_DAYS))

CAPACITY_CONFIGS = (
    {
        "name": "deep",
        "axis": "common_origin",
        "dim": 256,
        "depth": 4,
        "heads": 4,
        "kv_heads": 1,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "replayed common origin under exact controlled step/data schedule"
        ),
    },
    {
        "name": "depth6",
        "axis": "depth",
        "dim": 256,
        "depth": 6,
        "heads": 4,
        "kv_heads": 1,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "depth-only point: deep +2 blocks; dim/heads/kv_heads fixed"
        ),
    },
    {
        "name": "depth8",
        "axis": "depth",
        "dim": 256,
        "depth": 8,
        "heads": 4,
        "kv_heads": 1,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "depth-only point: deep +4 blocks; dim/heads/kv_heads fixed"
        ),
    },
    {
        "name": "width384_d4",
        "axis": "width",
        "dim": 384,
        "depth": 4,
        "heads": 6,
        "kv_heads": 1,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "width-only point: dim 384; depth/kv_heads fixed, head_dim=64"
        ),
    },
    {
        "name": "width512_d4_kv1",
        "axis": "width",
        "dim": 512,
        "depth": 4,
        "heads": 8,
        "kv_heads": 1,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "width-only endpoint: dim 512; depth/kv_heads fixed, head_dim=64"
        ),
    },
    {
        "name": "xlarge",
        "axis": "kv_heads",
        "dim": 512,
        "depth": 4,
        "heads": 8,
        "kv_heads": 2,
        "ffn_multiplier": 4,
        "gradient_checkpointing": True,
        "batch_tokens": 6144,
        "eval_batch_size": 1,
        "description": (
            "replayed upper endpoint; isolates kv_heads=2 from width512_d4_kv1"
        ),
    },
)
CONFIG_MAP = {item["name"]: item for item in CAPACITY_CONFIGS}


def _load_parent():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp03_parent_pipeline", PARENT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import parent pipeline: {PARENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARENT = _load_parent()
PIPE = PARENT.PIPE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    PARENT.atomic_write_json(path, payload)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_configs(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return [dict(item) for item in CAPACITY_CONFIGS]
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in CONFIG_MAP]
    if unknown:
        raise ValueError(f"Unknown Exp 03 configs: {unknown}")
    return [dict(CONFIG_MAP[name]) for name in names]


def build_settings(
    tokenizer: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    configs = [dict(item) for item in CAPACITY_CONFIGS]
    settings = PARENT.build_settings(
        configs=configs,
        tokenizer=tokenizer,
        gpt_epochs=1 if smoke else EPOCHS,
        eval_offsets=(3,) if smoke else VALIDATION_OFFSETS,
        eval_days=2 if smoke else EVAL_DAYS,
        smoke=smoke,
    )
    for config in settings["configs"]:
        config["axis"] = CONFIG_MAP[config["name"]]["axis"]
        config["batch_tokens"] = CONFIG_MAP[config["name"]]["batch_tokens"]
        config["eval_batch_size"] = 1
    settings.update(
        {
            "experiment": "Exp 03 controlled GPT capacity interpolation",
            "single_seed_stage": True,
            "axis_design": {
                "depth": ["deep", "depth6", "depth8"],
                "width": [
                    "deep",
                    "width384_d4",
                    "width512_d4_kv1",
                ],
                "kv_heads_check": ["width512_d4_kv1", "xlarge"],
                "width_definition": (
                    "heads scale with dim to hold head_dim=64; depth and "
                    "kv_heads remain fixed"
                ),
            },
            "protocol_invariants": {
                # Derived from the resolved Exp 02 dependency instead of a
                # literal, so a re-reviewed upstream selection can never leave
                # a stale tokenizer claim in the manifest.
                "tokenizer": (
                    f"Exp 02 {tokenizer['embedding_dim']}x"
                    f"{tokenizer['hidden_dim']} @ bits "
                    f"{tokenizer['bits_l1']}+{tokenizer['bits_l2']}"
                ),
                "seed": SEED,
                "epochs": EPOCHS,
                "optimizer_step_policy": (
                    "corrected exact sequence-count schedule: 32 sequences/update "
                    "through epoch 15, 64 from epoch 16; no adaptive microbatch "
                    "may cross an optimizer boundary"
                ),
                "data_order_policy": (
                    "architecture-independent local loader seed with the same "
                    "epoch-addressable sequence order for all six retrains"
                ),
                "full_sequences": True,
                "rope_angles": (
                    "fp32 with autocast disabled; cast to Q/K dtype only when "
                    "the rotary transform is applied"
                ),
                "evaluation_offsets": list(VALIDATION_OFFSETS),
                "days_per_window": EVAL_DAYS,
                "holdout_offset": HOLDOUT_OFFSET,
                "holdout_used": False,
            },
            "hierarchical_head_invariant": {
                "training": (
                    "fine target t is teacher-conditioned on current coarse "
                    "target t"
                ),
                "inference": (
                    "fine prediction t is conditioned on current predicted "
                    "coarse token t"
                ),
                "fine_ignore_index": -100,
                "valid_fine_code_zero_is_trained": True,
            },
        }
    )
    settings["evaluation"].update(
        {
            "selection_policy": (
                "Rank stable mature five-epoch windows by target-relative "
                "coarse-codebook balance and its components. DA, daily RankIC, "
                "MAPE, AmpRatio, and the now-aligned fine/joint distributions "
                "are mandatory secondary diagnostics."
            ),
            "recorded_code_levels": ["coarse", "fine", "joint"],
            "full_distribution_sidecars": True,
            "common_eval_batch_size": 1,
        }
    )
    settings["gpt"].update(
        {
            "controlled_loader_seed": SEED,
            "exact_accumulation_boundaries": True,
            "common_batch_tokens": 6144,
            "accumulation_steps": 8 if smoke else 32,
            "expected_optimizer_steps": 4160 if not smoke else 2,
            "expected_steps_per_epoch": (
                {"epochs_1_15": 128, "epochs_16_50": 64}
                if not smoke
                else {"epoch_1": 2}
            ),
            "same_data_order_across_architectures": True,
            "historical_endpoint_checkpoints_reused": False,
        }
    )
    return settings


def source_hashes() -> dict[str, str]:
    paths = (
        SCRIPT_PATH,
        ANALYSIS_SCRIPT,
        PARENT_SCRIPT,
        TRAJECTORY_SCRIPT,
        ROOT / "eval_helpers.py",
        ROOT / "experiment_io.py",
        ROOT / "train_base.py",
        ROOT / "data_processor.py",
        ROOT / "model" / "kronos_preview.py",
        ROOT / "model" / "layers.py",
        ROOT / "model" / "tokenizer.py",
    )
    return {
        str(path.relative_to(ROOT)): PIPE.file_sha256(path)
        for path in paths
        if path.is_file()
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
                "refusing to mix runs"
            )
        return payload
    payload = {
        "experiment": "Exp 03 controlled GPT capacity interpolation",
        "design": "single-seed controlled two-axis capacity grid",
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


def trajectory_ready(path: Path, epochs: int) -> bool:
    summary = path / "epoch_summary.json"
    if not summary.is_file():
        return False
    if len(PIPE.load_json(summary)) != epochs:
        return False
    return (
        len(list(path.glob("token_distributions_epoch_*.npz"))) == epochs
        and len(list(path.glob("prediction_records_epoch_*.npz")))
        == epochs
    )


def validate_optimizer_schedule(
    config_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    index_path = config_dir / "model_checkpoints.json"
    payload = PIPE.load_json(index_path)
    rows = sorted(
        payload.get("checkpoints", []),
        key=lambda row: int(row["epoch"]),
    )
    expected_epochs = 1 if smoke else EPOCHS
    if len(rows) != expected_epochs:
        raise RuntimeError(
            f"{config_dir.name}: incomplete checkpoint schedule"
        )
    cumulative = 0
    for row in rows:
        epoch = int(row["epoch"])
        expected = 2 if smoke else (128 if epoch <= 15 else 64)
        observed = int(row.get("optimizer_steps_this_epoch", -1))
        cumulative += expected
        if observed != expected or int(row.get("global_step", -1)) != cumulative:
            raise RuntimeError(
                f"{config_dir.name} epoch {epoch}: optimizer schedule "
                f"observed steps={observed}, global={row.get('global_step')}; "
                f"expected steps={expected}, global={cumulative}"
            )
    return {
        "epochs": expected_epochs,
        "final_global_step": cumulative,
        "exact": True,
    }


def add_architecture_fields(
    row: dict[str, Any],
    config: dict[str, Any],
    parameter_count: int,
    *,
    source: str,
) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "config": config["name"],
            "axis": config.get("axis", "controlled"),
            "source": source,
            "dim": config["dim"],
            "depth": config["depth"],
            "heads": config["heads"],
            "kv_heads": config["kv_heads"],
            "ffn_multiplier": config["ffn_multiplier"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "batch_tokens": config["batch_tokens"],
            "eval_batch_size": config["eval_batch_size"],
            "parameter_count": parameter_count,
        }
    )
    return result


def write_combined_summary(
    layout: StudyLayout,
    tokenizer: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in CAPACITY_CONFIGS:
        paths = PARENT.config_paths(
            layout.weights_root, layout.results_root, config["name"]
        )
        summary = paths["trajectory"] / "epoch_summary.json"
        if not summary.is_file() or not paths["model"].is_file():
            continue
        count = PARENT.parameter_count(paths["model"])
        for item in PIPE.load_json(summary):
            row = add_architecture_fields(
                item, dict(config), count, source="exp03_controlled_retrain"
            )
            row["tokenizer_sha256"] = tokenizer["sha256"]
            rows.append(row)

    order = {
        name: index
        for index, name in enumerate(
            (
                "deep",
                "depth6",
                "depth8",
                "width384_d4",
                "width512_d4_kv1",
                "xlarge",
            )
        )
    }
    rows.sort(key=lambda row: (order[row["config"]], int(row["epoch"])))
    atomic_write_json(
        layout.results_root / "combined_epoch_summary.json", rows
    )
    atomic_write_csv(
        layout.results_root / "combined_epoch_summary.csv", rows
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 03 controlled GPT capacity interpolation"
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
        help="Comma-separated execution subset: "
        + ",".join(item["name"] for item in CAPACITY_CONFIGS),
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Default reads the completed Exp 02 selection",
    )
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--no_analysis", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    PIPE.verify_runtime()
    requested_configs = parse_configs(args.configs)
    tokenizer = PARENT.resolve_tokenizer(args.tokenizer)
    settings = build_settings(tokenizer, smoke=args.smoke)

    temporary_root: Path | None = None
    if args.smoke:
        if args.analysis_only:
            raise ValueError("--analysis_only cannot be combined with --smoke")
        temporary_root = Path(
            tempfile.mkdtemp(prefix="kronos_exp03_smoke_")
        ).resolve()
        layout = StudyLayout.create(
            temporary_root / "weights", temporary_root / "results"
        )
    else:
        layout = StudyLayout.create(args.weights_root, args.results_root)
        free_gb = shutil.disk_usage(layout.weights_root).free / (1024**3)
        if not args.analysis_only and free_gb < 18:
            raise RuntimeError(
                f"At least 18 GiB free is required; {free_gb:.1f} GiB remains"
            )
    settings["evaluation"]["prepared_cache_dir"] = str(
        (layout.weights_root / "shared" / "eval_cache").resolve()
    )
    prepare_manifest(layout.results_root, settings, layout)

    succeeded = False
    try:
        if not args.analysis_only:
            update_manifest(
                layout.results_root,
                status="running",
                started_at_utc=utc_now(),
                requested_configs=[
                    config["name"] for config in requested_configs
                ],
            )
            shared_cache_root = layout.weights_root / "shared"
            schedule_validations: dict[str, Any] = {}
            for index, config in enumerate(requested_configs, start=1):
                print(
                    f"\n[{index}/{len(requested_configs)}] "
                    f"{config['name']}: {config['description']}",
                    flush=True,
                )
                PARENT.run_one_config(
                    layout, shared_cache_root, settings, config
                )
                schedule_validations[config["name"]] = (
                    validate_optimizer_schedule(
                        PARENT.config_paths(
                            layout.weights_root,
                            layout.results_root,
                            config["name"],
                        )["directory"],
                        smoke=args.smoke,
                    )
                )
            update_manifest(
                layout.results_root,
                optimizer_schedule_validation=schedule_validations,
            )

        if args.smoke:
            expected = 1
            for config in requested_configs:
                trajectory = PARENT.config_paths(
                    layout.weights_root,
                    layout.results_root,
                    config["name"],
                )["trajectory"]
                if not trajectory_ready(trajectory, expected):
                    raise RuntimeError(
                        f"Smoke trajectory incomplete: {config['name']}"
                    )
                validate_optimizer_schedule(
                    PARENT.config_paths(
                        layout.weights_root,
                        layout.results_root,
                        config["name"],
                    )["directory"],
                    smoke=True,
                )
            update_manifest(
                layout.results_root,
                status="smoke_completed",
                completed_at_utc=utc_now(),
            )
            succeeded = True
            print("Exp 03 smoke completed.", flush=True)
            return 0

        rows = write_combined_summary(layout, tokenizer)
        schedule_validations = {}
        completed_names = []
        for config in CAPACITY_CONFIGS:
            paths = PARENT.config_paths(
                layout.weights_root,
                layout.results_root,
                config["name"],
            )
            if not trajectory_ready(paths["trajectory"], EPOCHS):
                continue
            schedule_validations[config["name"]] = (
                validate_optimizer_schedule(paths["directory"], smoke=False)
            )
            completed_names.append(config["name"])
        complete = len(completed_names) == len(CAPACITY_CONFIGS)
        if complete and not args.no_analysis:
            env = os.environ.copy()
            env.pop("KRONOS_PREVIEW_OVERRIDE_JSON", None)
            PIPE.run_command(
                [
                    sys.executable,
                    str(ANALYSIS_SCRIPT),
                    "--root",
                    str(layout.results_root),
                ],
                env=env,
                log_path=layout.results_root / "analysis.log",
                label="Exp 03 mature target-relative codebook analysis",
            )
        update_manifest(
            layout.results_root,
            status="completed" if complete else "partial",
            completed_at_utc=utc_now() if complete else None,
            combined_rows=len(rows),
            completed_configs=completed_names,
            all_controlled_configs_completed=complete,
            optimizer_schedule_validation=schedule_validations,
            holdout_used=False,
        )
        write_download_manifest(layout)
        succeeded = True
        if complete:
            print(
                f"Completed Exp 03: {layout.results_root}", flush=True
            )
        else:
            print(
                "Partial Exp 03 run; selection was not finalized. "
                f"Output: {layout.results_root}",
                flush=True,
            )
        return 0
    except BaseException as exc:
        update_manifest(
            layout.results_root,
            status="failed",
            failed_at_utc=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
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
