"""Shared runner for the Exp 04-A controlled optimizer ablation.

The runner inherits the tokenizer selected by Exp 02 and the GPT architecture
selected by the controlled Exp 03 sweep. Every arm is trained on full data with
full sequences, retains every epoch checkpoint, and is evaluated over the same
pre-holdout windows. The module deliberately keeps quality and behaviour
metrics separate; it never constructs a weighted score.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import (
    StudyLayout,
    default_study_roots,
    runtime_environment,
    write_download_manifest,
)

_, EXP02_RESULTS_ROOT = default_study_roots(
    "02-tokenizer-tuning", seed=42
)
_, EXP03_RESULTS_ROOT = default_study_roots("03-gpt-scaling", seed=42)
EXP01_PIPELINE_PATH = (
    ROOT / "experiments" / "01-bitsweep" / "run_bitsweep.py"
)
EXP02_SELECTION = (
    EXP02_RESULTS_ROOT / "selection.json"
)
EXP03_SELECTION = (
    EXP03_RESULTS_ROOT / "selection.json"
)
TRAIN_SCRIPT = ROOT / "train_base.py"
TRAJECTORY_SCRIPT = (
    ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
)

# Portable upstream bundle: lets Exp 04 run on a server without copying
# server_runs/. The bundle (tokenizer.pt + upstream.json) is packaged once on
# a workstation that holds the reviewed Exp 02/03/04-A selections, then travels
# with the repo. Output still honours KRONOS_WEIGHTS_ROOT/KRONOS_RESULTS_ROOT,
# so results remain downloadable into the local server_runs/results tree.
PORTABLE_DIR = SCRIPT_PATH.parent / "portable_assets"
PORTABLE_TOKENIZER = PORTABLE_DIR / "tokenizer.pt"
PORTABLE_MANIFEST = PORTABLE_DIR / "upstream.json"

SEED = 42
HOLDOUT_OFFSET = 400
EVAL_DAYS = 1
# Full-coverage, single-day-resolution evaluation: tile [0, HOLDOUT_OFFSET) with
# contiguous EVAL_DAYS-day windows (offsets 0, 1, ..., 399) so every pre-holdout
# trading day is scored as its own window. Offset HOLDOUT_OFFSET+ stays sealed.
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, EVAL_DAYS))

# All-epoch summarization policy, mirroring Exp 03's eligible_windows: each arm
# is summarized over its full 1..N epoch trajectory as a single window. The
# maturity onset and loss basin are still recorded for audit only and no longer
# restrict which epochs enter the ranking.
LOSS_BASIN_RATIO = 1.01

# Downstream quality guardrails. These never participate in ranking; they only
# gate whether an arm may lead. Tolerance is max(practical floor, 2x reference
# arm within-window SD). ``envelope_field`` maps the canonical guardrail name to
# the arm_envelopes summary field holding its all-epoch window median, and
# ``std_field`` is the matching within-window standard deviation.
GUARDRAILS = {
    "avg_da_per_date": {
        "direction": "higher",
        "floor": 0.01,
        "label": "DA",
        "envelope_field": "late_median_da",
    },
    "avg_daily_rank_ic": {
        "direction": "higher",
        "floor": 0.01,
        "label": "daily RankIC",
        "envelope_field": "late_median_rankic",
    },
    "avg_mape": {
        "direction": "lower",
        "floor": 0.05,
        "label": "MAPE",
        "envelope_field": "late_median_mape",
    },
    "ampratio_log_error": {
        "direction": "lower",
        "floor": 0.05,
        "label": "|log AmpRatio|",
        "envelope_field": "late_median_ampratio_log_error",
    },
}


def _load_exp01_pipeline():
    spec = importlib.util.spec_from_file_location(
        "kronos_exp01_pipeline_for_exp04", EXP01_PIPELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXP01_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPE = _load_exp01_pipeline()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(temporary: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    _replace_with_retry(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    _replace_with_retry(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temporary, path)


def verify_python_runtime() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python >=3.10 is required, got {sys.version.split()[0]}"
        )


def parse_offsets(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation offsets must be comma-separated integers"
        ) from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(
            "evaluation offsets must be non-empty and non-negative"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("evaluation offsets must be unique")
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
        source: dict[str, Any] = {
            "source": "command_line",
            "selection_path": None,
        }
    else:
        if not EXP02_SELECTION.is_file():
            raise FileNotFoundError(
                f"Exp 02 selection is missing: {EXP02_SELECTION}"
            )
        selection = PIPE.load_json(EXP02_SELECTION)
        if not selection.get("upstream_eligible", False):
            raise RuntimeError(
                f"Exp 02 selection has not been recorded after review: "
                f"{EXP02_SELECTION}"
            )
        path = Path(selection["selected"]["tokenizer_path"]).resolve()
        source = {
            "source": "exp02_selection",
            "selection_path": str(EXP02_SELECTION.resolve()),
            "selection_sha256": PIPE.file_sha256(EXP02_SELECTION),
            "selection": selection,
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    result = tokenizer_metadata(path)
    result.update(source)
    return result


def resolve_architecture(selection_path: Path) -> dict[str, Any]:
    path = selection_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Controlled capacity selection is missing: {path}. "
            "Complete and analyze Exp 03 before running Exp 04."
        )
    selection = PIPE.load_json(path)
    if not selection.get("upstream_eligible", False):
        raise RuntimeError(
            f"Capacity selection has not been recorded after review: {path}"
        )
    selected = dict(selection.get("selected", {}))
    required = {
        "config",
        "dim",
        "depth",
        "heads",
        "kv_heads",
        "ffn_multiplier",
        "gradient_checkpointing",
        "batch_tokens",
        "eval_batch_size",
        "parameter_count",
    }
    missing = sorted(required - set(selected))
    if missing:
        raise RuntimeError(f"Capacity selection lacks fields: {missing}")
    return {
        **selected,
        "source": "capacity_selection",
        "selection_path": str(path),
        "selection_sha256": PIPE.file_sha256(path),
        "selection": selection,
    }


def load_required_selections(
    requirements: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, (path, expected_arm) in requirements.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Required upstream selection {name!r} is missing: {resolved}"
            )
        payload = PIPE.load_json(resolved)
        if not payload.get("upstream_eligible", False):
            raise RuntimeError(
                f"{name} selection is not eligible for downstream use: {resolved}. "
                "Run the complete formal arm set (not smoke/subset mode)."
            )
        actual = (
            payload.get("selected", {}).get("arm")
            or payload.get("selected", {}).get("config")
        )
        if actual != expected_arm:
            raise RuntimeError(
                f"{name} selected {actual!r}; this preregistration expects "
                f"{expected_arm!r}. Review the downstream protocol explicitly."
            )
        loaded[name] = {
            "path": str(resolved),
            "sha256": PIPE.file_sha256(resolved),
            "selection": payload,
        }
    return loaded


_PORTABLE_ARCH_FIELDS = (
    "config",
    "dim",
    "depth",
    "heads",
    "kv_heads",
    "ffn_multiplier",
    "gradient_checkpointing",
    "batch_tokens",
    "eval_batch_size",
    "parameter_count",
)


def write_portable_bundle(
    portable_dir: Path,
    *,
    tokenizer_path: Path,
    exp02_selection_path: Path,
    exp03_selection_path: Path,
    exp04a_selection_path: Path,
) -> Path:
    """Package the reviewed upstream artifacts so Exp 04 can run without server_runs.

    Copies the selected tokenizer and bundles the Exp 02/03/04-A selection JSONs
    into one ``upstream.json`` manifest under ``portable_dir``. Run this once on a
    workstation that has the formal ``server_runs/`` selections; the resulting
    directory then travels with the repo to the server.
    """
    portable_dir = portable_dir.resolve()
    portable_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_path.resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer is missing: {tokenizer_path}")
    tok_dest = portable_dir / "tokenizer.pt"
    if not tok_dest.exists() or (
        tok_dest.stat().st_size != tokenizer_path.stat().st_size
    ):
        shutil.copyfile(tokenizer_path, tok_dest)

    exp02_sel = PIPE.load_json(exp02_selection_path.resolve())
    exp03_sel = PIPE.load_json(exp03_selection_path.resolve())
    exp04a_sel = PIPE.load_json(exp04a_selection_path.resolve())
    for label, sel, path in (
        ("exp02", exp02_sel, exp02_selection_path),
        ("exp03", exp03_sel, exp03_selection_path),
        ("exp04a", exp04a_sel, exp04a_selection_path),
    ):
        if not sel.get("upstream_eligible", False):
            raise RuntimeError(
                f"{label} selection is not upstream-eligible: {path}"
            )

    tok_meta = tokenizer_metadata(tok_dest)
    selected = exp03_sel.get("selected", {})
    architecture = {
        field: selected[field]
        for field in _PORTABLE_ARCH_FIELDS
        if field in selected
    }
    missing = sorted(set(_PORTABLE_ARCH_FIELDS) - set(architecture))
    if missing:
        raise RuntimeError(
            f"Exp 03 selection lacks architecture fields: {missing}"
        )

    manifest = {
        "schema": 1,
        "created_at_utc": utc_now(),
        "description": (
            "Portable upstream bundle for Exp 04 server runs. Removes the "
            "input dependency on server_runs/ for the tokenizer, the Exp 03 "
            "architecture, and the Exp 04-A optimizer selection. Study output "
            "still honours KRONOS_WEIGHTS_ROOT/KRONOS_RESULTS_ROOT."
        ),
        "tokenizer": {
            "relative_path": "tokenizer.pt",
            "sha256": tok_meta["sha256"],
            "bits_l1": tok_meta["bits_l1"],
            "bits_l2": tok_meta["bits_l2"],
            "embedding_dim": tok_meta["embedding_dim"],
            "hidden_dim": tok_meta["hidden_dim"],
        },
        "architecture": architecture,
        "exp02_selection": exp02_sel,
        "exp03_selection": exp03_sel,
        "exp04a_selection": exp04a_sel,
    }
    atomic_write_json(PORTABLE_MANIFEST, manifest)
    return PORTABLE_MANIFEST


def load_portable_assets(
    portable_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve tokenizer/architecture/04-A selection from the portable bundle.

    Returns the same shapes the non-portable path produces
    (``resolve_tokenizer`` / ``resolve_architecture`` /
    ``load_required_selections``), so callers can branch without touching the
    rest of the pipeline.
    """
    base = (portable_dir or PORTABLE_DIR).resolve()
    manifest_path = base / "upstream.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Portable manifest is missing: {manifest_path}. "
            "Run write_portable_bundle on a workstation that has server_runs/, "
            "or pass --portable_dir pointing at an existing bundle."
        )
    manifest = PIPE.load_json(manifest_path)
    schema = manifest.get("schema", 1)
    if schema != 1:
        raise RuntimeError(
            f"Unsupported portable manifest schema {schema} at {manifest_path}"
        )

    tok_rel = manifest["tokenizer"]["relative_path"]
    tok_path = (base / tok_rel).resolve()
    tokenizer = tokenizer_metadata(tok_path)
    tokenizer.update(
        {
            "source": "portable_bundle",
            "portable_dir": str(base),
            "portable_manifest_path": str(manifest_path),
            "portable_manifest_sha256": PIPE.file_sha256(manifest_path),
            "bundled_selection": manifest.get("exp02_selection"),
        }
    )

    architecture = dict(manifest["architecture"])
    architecture.update(
        {
            "source": "portable_bundle",
            "portable_dir": str(base),
            "portable_manifest_path": str(manifest_path),
            "bundled_selection": manifest.get("exp03_selection"),
        }
    )

    upstream: dict[str, Any] = {}
    for name, expected in (("exp04a_optimizer", "muon"),):
        sel = manifest.get("exp04a_selection")
        if sel is None:
            raise RuntimeError(
                f"Portable manifest lacks the {name} selection"
            )
        if not sel.get("upstream_eligible", False):
            raise RuntimeError(
                f"Portable {name} selection is not upstream-eligible"
            )
        actual = (
            sel.get("selected", {}).get("arm")
            or sel.get("selected", {}).get("config")
        )
        if actual != expected:
            raise RuntimeError(
                f"Portable {name} selected {actual!r}; "
                f"this preregistration expects {expected!r}"
            )
        upstream[name] = {
            "path": str(manifest_path),
            "sha256": PIPE.file_sha256(manifest_path),
            "selection": sel,
            "source": "portable_bundle",
        }

    return {
        "tokenizer": tokenizer,
        "architecture": architecture,
        "upstream_selections": upstream,
    }


def arm_paths(
    results_root: Path,
    name: str,
    weights_root: Path | None = None,
) -> dict[str, Path]:
    if weights_root is None:
        manifest_path = results_root / "study_manifest.json"
        if manifest_path.is_file():
            manifest = PIPE.load_json(manifest_path)
            raw = manifest.get("artifact_layout", {}).get("weights_root")
            weights_root = Path(raw) if raw else results_root
        else:
            weights_root = results_root
    results = results_root / "arms" / name
    weights = weights_root / "arms" / name
    return {
        "directory": results,
        "weights_directory": weights,
        "override": results / "override.json",
        "run": results / "run.json",
        "model": weights / "model.pt",
        "model_resume": weights / "model.pt.ckpt",
        "checkpoint_index": results / "model_checkpoints.json",
        "trajectory": results / "epoch_trajectory",
        "logs": results / "logs",
        "gpt_log": results / "logs" / "gpt.log",
        "trajectory_log": results / "logs" / "trajectory.log",
    }


def merged_recipe(fixed_recipe: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(fixed_recipe)
    recipe.update(arm.get("train", {}))
    return recipe


def build_settings(
    *,
    experiment_key: str,
    experiment_label: str,
    arms: list[dict[str, Any]],
    protocol_arm_names: list[str],
    fixed_recipe: dict[str, Any],
    tokenizer: dict[str, Any],
    architecture: dict[str, Any],
    upstream: dict[str, Any],
    epochs: int,
    offsets: tuple[int, ...],
    eval_days: int,
    batch_tokens: int,
    eval_batch_size: int,
    smoke: bool,
) -> dict[str, Any]:
    effective_batch_tokens = (
        2048 if smoke else batch_tokens or int(architecture["batch_tokens"])
    )
    effective_eval_batch = (
        1 if smoke else eval_batch_size or int(architecture["eval_batch_size"])
    )
    return {
        "experiment_key": experiment_key,
        "experiment_label": experiment_label,
        "seed": SEED,
        "arms": [
            {
                "name": arm["name"],
                "description": arm["description"],
                "recipe": merged_recipe(fixed_recipe, arm),
            }
            for arm in arms
        ],
        "protocol_arm_names": protocol_arm_names,
        "subset_run": [arm["name"] for arm in arms] != protocol_arm_names,
        "tokenizer_dependency": tokenizer,
        "architecture_dependency": architecture,
        "upstream_selections": upstream,
        "gpt": {
            "epochs": epochs,
            "max_stocks": 24 if smoke else 0,
            "max_seq_len": 256 if smoke else 0,
            "batch_tokens": effective_batch_tokens,
            "batch_cap": 64,
            "accumulation_steps": 8 if smoke else 32,
            "constant_accumulation": True,
            "controlled_loader_seed": SEED,
            "exact_accumulation_boundaries": True,
            "early_stop_patience": 0,
            "retain_every_epoch": True,
        },
        "evaluation": {
            "offsets": list(offsets),
            "days_per_window": eval_days,
            "n_stocks": 6 if smoke else 0,
            "sample_strategy": "shortest" if smoke else "random",
            "batch_size": effective_eval_batch,
            "health_gate": {
                "max_daily_collapse_rate": 1.0 if smoke else 0.35,
                "min_daily_unique_tokens": 1 if smoke else 32,
            },
            "holdout_offset": HOLDOUT_OFFSET,
            "holdout_used": False,
            "selection_policy": (
                "No weighted score. Each arm is summarized over its full epoch "
                "trajectory as a single window. Primary ranking is by target-relative "
                "coarse codebook balance (median, p10), token support F1, token JSD, "
                "effective token alignment, collapse, and unique tokens. Downstream "
                "DA/RankIC/MAPE/AmpRatio serve only as guardrails (must not regress "
                "beyond max(practical floor, 2x baseline within-window SD) vs the "
                "reference arm). Paired moving-block bootstrap diagnostics are "
                "reported separately and do not affect ranking."
            ),
        },
        "smoke": smoke,
    }


def source_hashes(wrapper_path: Path) -> dict[str, str]:
    paths = (
        ROOT / "experiment_io.py",
        ROOT / "config.py",
        ROOT / "data_processor.py",
        ROOT / "training_utils.py",
        ROOT / "train_base.py",
        ROOT / "eval_helpers.py",
        ROOT / "model" / "kronos_preview.py",
        ROOT / "model" / "layers.py",
        EXP01_PIPELINE_PATH,
        TRAJECTORY_SCRIPT,
        SCRIPT_PATH,
        wrapper_path,
    )
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): PIPE.file_sha256(path)
        for path in paths
        if path.is_file()
    }


def prepare_manifest(
    results_root: Path,
    settings: dict[str, Any],
    wrapper_path: Path,
    layout: StudyLayout,
) -> dict[str, Any]:
    implementation = source_hashes(wrapper_path)
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
                "move the old output root aside before starting a new study."
            )
        return payload
    payload = {
        "experiment": settings["experiment_label"],
        "design": "full-data full-sequence epoch-wise controlled ablation",
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


def update_manifest(output_root: Path, **updates: Any) -> dict[str, Any]:
    path = output_root / "study_manifest.json"
    payload = PIPE.load_json(path)
    payload.update(updates)
    atomic_write_json(path, payload)
    return payload


def write_override(
    path: Path,
    shared_cache_root: Path,
    settings: dict[str, Any],
    recipe: dict[str, Any],
) -> None:
    tokenizer = settings["tokenizer_dependency"]
    architecture = settings["architecture_dependency"]
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
            "dim": architecture["dim"],
            "depth": architecture["depth"],
            "heads": architecture["heads"],
            "num_kv_heads": architecture["kv_heads"],
            "ffn_multiplier": architecture["ffn_multiplier"],
        },
        "TrainingConfig": {
            "random_seed": SEED,
            "warmup_ratio": recipe.get("warmup_ratio", 0.05),
            "batch_size": 1,
            "accumulation_steps": settings["gpt"]["accumulation_steps"],
            "use_gradient_checkpointing": architecture[
                "gradient_checkpointing"
            ],
            "save_dir": str(shared_cache_root.resolve()),
        },
    }
    if path.exists() and PIPE.load_json(path) != payload:
        raise RuntimeError(f"Refusing to overwrite incompatible {path}")
    atomic_write_json(path, payload)


def build_train_command(
    paths: dict[str, Path],
    settings: dict[str, Any],
    arm: dict[str, Any],
    recipe: dict[str, Any],
) -> list[str]:
    architecture = settings["architecture_dependency"]
    gpt = settings["gpt"]
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--save_path",
        str(paths["model"]),
        "--tokenizer_path",
        str(settings["tokenizer_dependency"]["path"]),
        "--epochs",
        str(gpt["epochs"]),
        "--tag",
        f"{settings['experiment_key']}_{arm['name']}",
        "--loss",
        str(recipe["loss"]),
        "--gamma",
        str(recipe.get("gamma", 0.0)),
        "--optimizer",
        str(recipe["optimizer"]),
        "--lr",
        str(recipe["lr"]),
        "--dropout",
        str(recipe["dropout"]),
        "--weight_decay",
        str(recipe["weight_decay"]),
        "--fine_weight",
        str(recipe["fine_weight"]),
        "--het_weight",
        str(recipe["het_weight"]),
        "--label_smoothing",
        str(recipe.get("label_smoothing", 0.0)),
        "--entropy_alpha",
        str(recipe.get("entropy_alpha", 0.0)),
        "--max_stocks",
        str(gpt["max_stocks"]),
        "--max_seq_len",
        str(gpt["max_seq_len"]),
        "--batch_tokens",
        str(gpt["batch_tokens"]),
        "--batch_cap",
        str(gpt["batch_cap"]),
        "--early_stop_patience",
        "0",
        "--history_per_epoch",
        "--metrics_dir",
        str(paths["directory"]),
        "--constant_accumulation",
        "--controlled_loader_seed",
        str(gpt["controlled_loader_seed"]),
        "--exact_accumulation_boundaries",
        "--dim",
        str(architecture["dim"]),
        "--depth",
        str(architecture["depth"]),
        "--heads",
        str(architecture["heads"]),
        "--num_kv_heads",
        str(architecture["kv_heads"]),
        "--ffn_multiplier",
        str(architecture["ffn_multiplier"]),
    ]
    if recipe.get("heteroscedastic", True):
        command.append("--heteroscedastic")
    else:
        command.append("--no-heteroscedastic")
    if architecture["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    if recipe["optimizer"] == "muon":
        command.extend(["--lr_muon", str(recipe["lr_muon"])])
    return command


def update_arm_run(
    paths: dict[str, Path], arm: dict[str, Any], **updates: Any
) -> dict[str, Any]:
    if paths["run"].exists():
        payload = PIPE.load_json(paths["run"])
    else:
        payload = {
            "arm": arm["name"],
            "description": arm["description"],
            "status": "planned",
            "stages": {},
            "created_at_utc": utc_now(),
        }
    payload.update(updates)
    payload["updated_at_utc"] = utc_now()
    atomic_write_json(paths["run"], payload)
    return payload


def run_one_arm(
    layout: StudyLayout,
    shared_cache_root: Path,
    settings: dict[str, Any],
    arm: dict[str, Any],
    fixed_recipe: dict[str, Any],
) -> None:
    paths = arm_paths(
        layout.results_root, arm["name"], layout.weights_root
    )
    paths["weights_directory"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    recipe = merged_recipe(fixed_recipe, arm)
    write_override(paths["override"], shared_cache_root, settings, recipe)
    run_state = update_arm_run(
        paths,
        arm,
        status="running",
        started_at_utc=utc_now(),
        recipe=recipe,
    )
    env = os.environ.copy()
    env["KRONOS_PREVIEW_OVERRIDE_JSON"] = str(paths["override"].resolve())

    if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
        run_state["stages"]["gpt"] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        atomic_write_json(paths["run"], run_state)
        PIPE.run_command(
            build_train_command(paths, settings, arm, recipe),
            env=env,
            log_path=paths["gpt_log"],
            label=f"{settings['experiment_key']} {arm['name']} GPT",
        )
        if not PIPE.gpt_is_complete(paths, settings["gpt"]["epochs"]):
            raise RuntimeError(f"GPT completion check failed for {arm['name']}")
    run_state = update_arm_run(paths, arm)
    run_state["stages"]["gpt"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "checkpoint_index": str(paths["checkpoint_index"].resolve()),
    }
    atomic_write_json(paths["run"], run_state)

    if not PIPE.trajectory_is_complete(paths, settings["gpt"]["epochs"]):
        run_state = update_arm_run(paths, arm)
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
            str(settings["tokenizer_dependency"]["path"]),
            "--output_dir",
            str(paths["trajectory"]),
            "--prepared_cache_dir",
            str((shared_cache_root / "prepared_eval").resolve()),
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
            f"{settings['experiment_label']} · {arm['name']}",
        ]
        PIPE.run_command(
            command,
            env=env,
            log_path=paths["trajectory_log"],
            label=f"{settings['experiment_key']} {arm['name']} trajectory",
        )
        if not PIPE.trajectory_is_complete(
            paths, settings["gpt"]["epochs"]
        ):
            raise RuntimeError(
                f"Trajectory completion check failed for {arm['name']}"
            )
    stages = update_arm_run(paths, arm).get("stages", {})
    stages["epoch_evaluation"] = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "epochs": settings["gpt"]["epochs"],
        "output": str(paths["trajectory"].resolve()),
    }
    update_arm_run(
        paths,
        arm,
        status="completed",
        completed_at_utc=utc_now(),
        stages=stages,
    )


def write_combined_summary(
    output_root: Path,
    arms: list[dict[str, Any]],
    fixed_recipe: dict[str, Any],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        paths = arm_paths(output_root, arm["name"])
        summary_path = paths["trajectory"] / "epoch_summary.json"
        if not summary_path.is_file():
            continue
        recipe = merged_recipe(fixed_recipe, arm)
        for item in PIPE.load_json(summary_path):
            row = dict(item)
            row.update(
                {
                    "arm": arm["name"],
                    "arm_description": arm["description"],
                    "recipe": json.dumps(recipe, sort_keys=True),
                    "parameter_count": settings["architecture_dependency"][
                        "parameter_count"
                    ],
                    "tokenizer_sha256": settings["tokenizer_dependency"][
                        "sha256"
                    ],
                }
            )
            rows.append(row)
    order = {arm["name"]: index for index, arm in enumerate(arms)}
    rows.sort(key=lambda row: (order[row["arm"]], int(row["epoch"])))
    atomic_write_json(output_root / "combined_epoch_summary.json", rows)
    write_csv(output_root / "combined_epoch_summary.csv", rows)
    return rows


def _arg_extreme(
    rows: list[dict[str, Any]], field: str, mode: str
) -> dict[str, Any]:
    key = (lambda row: float(row[field]))
    return (max if mode == "max" else min)(rows, key=key)


def _median_rows(rows: list[dict[str, Any]], field: str) -> float:
    values = sorted(float(row[field]) for row in rows)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _std_rows(rows: list[dict[str, Any]], field: str) -> float:
    """Sample standard deviation (ddof=1) of a field across the window.

    Used for guardrail tolerance: max(practical floor, 2x reference within-window
    SD). Returns 0.0 when fewer than two finite values are present.
    """
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    return math.sqrt(variance)


def select_mature_window(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize an arm over its full epoch trajectory as a single window.

    Mirrors Exp 03's all-epoch ``eligible_windows`` policy: every epoch 1..N
    enters the summary as one window. The maturity onset and 1%-of-best
    validation-loss basin are still recorded for audit, but no longer restrict
    which epochs enter the ranking. Token-balance metrics dominate downstream
    ordering; the previous lexicographic health/DA/RankIC/MAPE scan is removed.
    """
    if not rows:
        raise ValueError("No epoch rows")
    ordered = sorted(rows, key=lambda row: int(row["epoch"]))
    minimum_loss = min(float(row["val_loss"]) for row in ordered)
    loss_ceiling = minimum_loss * LOSS_BASIN_RATIO
    maturity_onset = min(
        int(row["epoch"])
        for row in ordered
        if float(row["val_loss"]) <= loss_ceiling
    )
    metadata = {
        "window_epochs": [
            int(ordered[0]["epoch"]), int(ordered[-1]["epoch"])
        ],
        "window_size": len(ordered),
        "maturity_onset_epoch": maturity_onset,
        "loss_basin_threshold": loss_ceiling,
        "minimum_val_loss": minimum_loss,
        "near_best_loss_ceiling": loss_ceiling,
        "used_near_best_loss_basin": False,
        "candidate_windows": 1,
        "window_policy": "all_epochs",
        "selection_order": [
            "higher median target-relative coarse codebook balance",
            "higher p10 target-relative coarse codebook balance",
            "higher median coarse token support F1",
            "lower median coarse token JSD",
            "higher median effective token alignment",
            "lower median daily collapse rate",
            "higher median daily unique tokens",
        ],
    }
    return ordered, metadata


def arm_envelopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    arm_order = list(dict.fromkeys(str(row["arm"]) for row in rows))
    for arm in arm_order:
        group = sorted(
            [row for row in rows if row["arm"] == arm],
            key=lambda row: int(row["epoch"]),
        )
        mature, mature_metadata = select_mature_window(group)
        best_da = _arg_extreme(group, "avg_da_per_date", "max")
        best_rank = _arg_extreme(group, "avg_daily_rank_ic", "max")
        best_mape = _arg_extreme(group, "avg_mape", "min")
        best_collapse = _arg_extreme(
            group, "p90_daily_collapse_rate", "min"
        )
        best_amp = _arg_extreme(group, "ampratio_log_error", "min")

        def median(field: str) -> float:
            return _median_rows(mature, field)

        def optional_median(field: str) -> float | None:
            values = [
                float(row[field])
                for row in mature
                if row.get(field) is not None
            ]
            if not values:
                return None
            values.sort()
            middle = len(values) // 2
            if len(values) % 2:
                return values[middle]
            return (values[middle - 1] + values[middle]) / 2.0

        envelope: dict[str, Any] = {
            "arm": arm,
            "description": group[0]["arm_description"],
            "recipe": group[0]["recipe"],
            "n_epochs": len(group),
            "healthy_epochs": sum(bool(row["healthy"]) for row in group),
            "late_healthy_epochs": sum(
                bool(row["healthy"]) for row in mature
            ),
            "mature_window_epochs": mature_metadata["window_epochs"],
            "window_policy": mature_metadata["window_policy"],
            "maturity_onset_epoch": mature_metadata[
                "maturity_onset_epoch"
            ],
            "loss_basin_threshold": mature_metadata[
                "loss_basin_threshold"
            ],
            "minimum_val_loss": mature_metadata["minimum_val_loss"],
            "near_best_loss_ceiling": mature_metadata[
                "near_best_loss_ceiling"
            ],
            "best_da": best_da["avg_da_per_date"],
            "best_da_epoch": best_da["epoch"],
            "best_rankic": best_rank["avg_daily_rank_ic"],
            "best_rankic_epoch": best_rank["epoch"],
            "best_mape": best_mape["avg_mape"],
            "best_mape_epoch": best_mape["epoch"],
            "best_p90_collapse": best_collapse[
                "p90_daily_collapse_rate"
            ],
            "best_p90_collapse_epoch": best_collapse["epoch"],
            "best_ampratio": best_amp["avg_ampratio"],
            "best_ampratio_epoch": best_amp["epoch"],
            "late_median_val_loss": median("val_loss"),
            "late_median_da": median("avg_da_per_date"),
            "late_median_rankic": median("avg_daily_rank_ic"),
            "late_median_mape": median("avg_mape"),
            "late_median_p90_collapse": median(
                "p90_daily_collapse_rate"
            ),
            "late_median_worst_collapse": median(
                "worst_daily_collapse_rate"
            ),
            "late_median_collapse_rate": median(
                "median_daily_collapse_rate"
            ),
            "late_median_unique": median("median_daily_unique_tokens"),
            "late_median_min_unique": median("min_daily_unique_tokens"),
            "late_median_ampratio": median("avg_ampratio"),
            "late_median_ampratio_log_error": median(
                "ampratio_log_error"
            ),
            # Coarse target-relative token layer (primary ranking).
            "mature_median_codebook_balance": optional_median(
                "median_daily_codebook_balance_score"
            ),
            "mature_p10_codebook_balance": optional_median(
                "p10_daily_codebook_balance_score"
            ),
            "mature_median_token_support_f1": optional_median(
                "median_daily_token_support_f1"
            ),
            "mature_median_token_jsd": optional_median(
                "median_daily_token_jsd"
            ),
            "mature_median_effective_token_alignment": optional_median(
                "median_daily_effective_token_alignment"
            ),
            "mature_median_collapse_alignment": optional_median(
                "median_daily_collapse_alignment"
            ),
            "mature_median_distribution_alignment": optional_median(
                "median_daily_distribution_alignment"
            ),
            "mature_median_target_support_recall": optional_median(
                "median_daily_target_support_recall"
            ),
            "mature_median_prediction_support_precision": optional_median(
                "median_daily_prediction_support_precision"
            ),
            "mature_median_coarse_token_accuracy": optional_median(
                "median_daily_coarse_token_accuracy"
            ),
            # Fine target-relative token layer (diagnostic).
            "mature_median_fine_codebook_balance": optional_median(
                "median_daily_fine_codebook_balance_score"
            ),
            "mature_median_fine_token_support_f1": optional_median(
                "median_daily_fine_token_support_f1"
            ),
            "mature_median_fine_token_jsd": optional_median(
                "median_daily_fine_token_jsd"
            ),
            "mature_median_fine_effective_token_alignment": optional_median(
                "median_daily_fine_effective_token_alignment"
            ),
            "mature_median_fine_collapse_alignment": optional_median(
                "median_daily_fine_collapse_alignment"
            ),
            "mature_median_fine_n_unique_tokens": optional_median(
                "median_daily_fine_n_unique_tokens"
            ),
            # Joint target-relative token layer (diagnostic).
            "mature_median_joint_codebook_balance": optional_median(
                "median_daily_joint_codebook_balance_score"
            ),
            "mature_median_joint_token_support_f1": optional_median(
                "median_daily_joint_token_support_f1"
            ),
            "mature_median_joint_token_jsd": optional_median(
                "median_daily_joint_token_jsd"
            ),
            "mature_median_joint_effective_token_alignment": optional_median(
                "median_daily_joint_effective_token_alignment"
            ),
            "mature_median_joint_collapse_alignment": optional_median(
                "median_daily_joint_collapse_alignment"
            ),
            "mature_median_joint_n_unique_tokens": optional_median(
                "median_daily_joint_n_unique_tokens"
            ),
        }
        # Within-window standard deviations for the downstream guardrails.
        # Tolerance = max(practical floor, 2x reference-arm SD).
        for field in GUARDRAILS:
            envelope[f"window_std_{field}"] = _std_rows(mature, field)
        output.append(envelope)
    return output


def _envelope_value(row: dict[str, Any], field: str, default: float) -> float:
    """Robust float accessor for primary_key over envelope dicts."""
    value = row.get(field)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def primary_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Token-balance-dominated ranking key, mirroring Exp 03.

    Downstream DA/RankIC/MAPE/AmpRatio are deliberately absent: they are
    guardrails, not ranking signals. Lower-is-better fields are negated so
    that ``max`` / ``reverse=True`` always means "better".
    """
    return (
        _envelope_value(row, "mature_median_codebook_balance", -math.inf),
        _envelope_value(row, "mature_p10_codebook_balance", -math.inf),
        _envelope_value(
            row, "mature_median_token_support_f1", -math.inf
        ),
        -_envelope_value(row, "mature_median_token_jsd", math.inf),
        _envelope_value(
            row, "mature_median_effective_token_alignment", -math.inf
        ),
        -_envelope_value(row, "late_median_collapse_rate", math.inf),
        _envelope_value(row, "late_median_unique", -math.inf),
    )


def guardrail_reference(
    baseline_envelope: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Reference values and tolerances drawn from the reference arm.

    The reference arm is the first arm (adamw in Exp 04-A), analogous to Exp
    03's ``deep`` config. Tolerance is max(practical floor, 2x the reference
    arm's within-window standard deviation).
    """
    reference = {
        field: _envelope_value(
            baseline_envelope, spec["envelope_field"], 0.0
        )
        for field, spec in GUARDRAILS.items()
    }
    tolerances = {
        field: max(
            float(spec["floor"]),
            2.0
            * _envelope_value(
                baseline_envelope, f"window_std_{field}", 0.0
            ),
        )
        for field, spec in GUARDRAILS.items()
    }
    return reference, tolerances


def apply_guardrails(
    envelope: dict[str, Any],
    reference: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Annotate an envelope with per-guardrail delta/pass/fail status.

    Guardrails never reorder arms; they only flag whether an arm is allowed to
    lead. Returns a copy of ``envelope`` with ``guardrail_delta_<field>``,
    ``guardrail_tolerance_<field>``, ``guardrail_pass``, and
    ``guardrail_failures`` added.
    """
    result = dict(envelope)
    failures: list[str] = []
    for field, spec in GUARDRAILS.items():
        value = _envelope_value(result, spec["envelope_field"], 0.0)
        baseline = reference[field]
        tolerance = tolerances[field]
        delta = value - baseline
        result[f"guardrail_delta_{field}"] = delta
        result[f"guardrail_tolerance_{field}"] = tolerance
        if spec["direction"] == "higher":
            failed = value < baseline - tolerance
        else:
            failed = value > baseline + tolerance
        if failed:
            failures.append(str(spec["label"]))
    result["guardrail_pass"] = not failures
    result["guardrail_failures"] = ",".join(failures)
    return result


DAILY_DIAGNOSTIC_FIELDS = {
    "daily_da": ("da", "higher"),
    "daily_rank_ic": ("rank_ic", "higher"),
    "daily_mape": ("mape", "lower"),
    "daily_collapse": ("collapse_rate", "lower"),
    "daily_unique": ("n_unique_tokens", "higher"),
    "daily_ampratio_log_error": ("ampratio", "lower"),
    "daily_codebook_balance": ("codebook_balance_score", "higher"),
    "daily_target_support_recall": ("target_support_recall", "higher"),
    "daily_effective_tokens": ("pred_effective_tokens", "higher"),
    "daily_token_jsd": ("token_jsd", "lower"),
}


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mature_daily_values(
    output_root: Path,
    arm: str,
) -> dict[str, dict[int, dict[str, float]]]:
    trajectory = arm_paths(output_root, arm)["trajectory"]
    epoch_paths: list[Path] = []
    for path in trajectory.glob("epoch_*.json"):
        try:
            int(path.stem.removeprefix("epoch_"))
        except ValueError:
            continue
        epoch_paths.append(path)
    payloads = [PIPE.load_json(path) for path in sorted(epoch_paths)]
    payloads.sort(key=lambda payload: int(payload["epoch"]))
    # All-epoch policy: the paired bootstrap now consumes per-date records from
    # every epoch (select_mature_window returns the full trajectory), not just
    # a narrow 5-epoch mature plateau.
    collected: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {} for name in DAILY_DIAGNOSTIC_FIELDS
    }
    for payload in payloads:
        for raw_offset, window in payload["windows"].items():
            offset = int(raw_offset)
            for date_key, daily in window["per_date"].items():
                for name, (source, _) in DAILY_DIAGNOSTIC_FIELDS.items():
                    value = daily.get(source)
                    if value is None:
                        continue
                    number = float(value)
                    if not math.isfinite(number):
                        continue
                    if name == "daily_ampratio_log_error":
                        if number <= 0:
                            continue
                        number = abs(math.log(number))
                    collected[name].setdefault(offset, {}).setdefault(
                        date_key, []
                    ).append(number)
    return {
        name: {
            offset: {
                date_key: sum(values) / len(values)
                for date_key, values in dates.items()
            }
            for offset, dates in windows.items()
        }
        for name, windows in collected.items()
    }


def paired_epoch_diagnostics(
    output_root: Path,
    arms: list[str],
    selected_arm: str,
    *,
    replicates: int = 10_000,
    block_length: int = 5,
) -> list[dict[str, Any]]:
    """All-epoch paired diagnostics; not an estimate of multi-seed uncertainty.

    Per-date means are pooled across the full epoch trajectory (one window per
    arm) and a circular moving-block bootstrap runs inside each validation
    window. Diagnostic only; it never affects ranking or selection.
    """
    if len(arms) < 2:
        return []
    by_arm = {
        arm: _mature_daily_values(output_root, arm)
        for arm in arms
    }
    output: list[dict[str, Any]] = []
    for comparator_index, comparator in enumerate(arms):
        if comparator == selected_arm:
            continue
        for metric_index, (metric, (_, favorable)) in enumerate(
            DAILY_DIAGNOSTIC_FIELDS.items()
        ):
            differences: dict[int, list[float]] = {}
            selected_windows = by_arm[selected_arm][metric]
            comparator_windows = by_arm[comparator][metric]
            for offset in sorted(set(selected_windows) & set(comparator_windows)):
                selected_dates = selected_windows[offset]
                comparator_dates = comparator_windows[offset]
                common_dates = sorted(set(selected_dates) & set(comparator_dates))
                if common_dates:
                    differences[offset] = [
                        selected_dates[date] - comparator_dates[date]
                        for date in common_dates
                    ]
            # Pool all per-date differences into a single time series and
            # resample with circular moving blocks. Resampling within each
            # offset separately degenerates when an offset has only one date
            # (rng.randrange(1) is always 0), producing zero-variance CIs.
            flat = [value for values in differences.values() for value in values]
            if not flat:
                continue
            n_flat = len(flat)
            point = sum(flat) / n_flat
            rng = random.Random(
                SEED * 10_000 + comparator_index * 100 + metric_index
            )
            bootstrap: list[float] = []
            for _ in range(replicates):
                sampled: list[float] = []
                while len(sampled) < n_flat:
                    start = rng.randrange(n_flat)
                    sampled.extend(
                        flat[(start + step) % n_flat]
                        for step in range(block_length)
                    )
                bootstrap.append(sum(sampled[:n_flat]) / n_flat)
            output.append(
                {
                    "selected_arm": selected_arm,
                    "comparator_arm": comparator,
                    "metric": metric,
                    "favorable_when": favorable,
                    "selected_minus_comparator": point,
                    "moving_block_bootstrap_95_ci": [
                        _quantile(bootstrap, 0.025),
                        _quantile(bootstrap, 0.975),
                    ],
                    "n_dates": len(flat),
                    "n_windows": len(differences),
                    "block_length_days": block_length,
                    "replicates": replicates,
                }
            )
    return output


def make_comparison_plots(
    output_root: Path, rows: list[dict[str, Any]], experiment_label: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(str(row["arm"]) for row in rows))
    definitions = {
        "quality_trajectories.png": (
            ("avg_da_per_date", "Mean daily DA (%)", 100.0),
            ("avg_daily_rank_ic", "Mean daily RankIC", 1.0),
            ("avg_mape", "MAPE (%)", 1.0),
        ),
        "behaviour_trajectories.png": (
            ("p90_daily_collapse_rate", "P90 daily Collapse (%)", 100.0),
            ("median_daily_unique_tokens", "Median daily Unique", 1.0),
            ("avg_ampratio", "AmpRatio", 1.0),
        ),
        "token_quality_trajectories.png": (
            (
                "median_daily_codebook_balance_score",
                "Median daily codebook balance",
                1.0,
            ),
            (
                "median_daily_token_support_f1",
                "Median daily token support F1",
                1.0,
            ),
            ("median_daily_token_jsd", "Median daily token JSD", 1.0),
            (
                "median_daily_effective_token_alignment",
                "Median daily effective token alignment",
                1.0,
            ),
        ),
    }
    for filename, metrics in definitions.items():
        n_panels = len(metrics)
        figure, axes = plt.subplots(
            n_panels,
            1,
            figsize=(11, 4 * n_panels),
            sharex=True,
        )
        if n_panels == 1:
            axes = (axes,)
        for arm in arms:
            group = sorted(
                [row for row in rows if row["arm"] == arm],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in group]
            for axis, (field, label, scale) in zip(axes, metrics):
                axis.plot(
                    epochs,
                    [float(row[field]) * scale for row in group],
                    label=arm,
                    linewidth=1.8,
                )
                axis.set_ylabel(label)
                axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Epoch")
        axes[0].legend()
        figure.suptitle(experiment_label)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)


def analyze(
    output_root: Path,
    settings: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    selected_arm: str,
    selection_rationale: str,
    record_selection: bool,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No completed epoch rows to analyze")
    summary = arm_envelopes(rows)
    # Downstream guardrails: reference arm = first arm (adamw in Exp 04-A),
    # analogous to Exp 03's ``deep`` config. Guardrails gate leadership but
    # never reorder arms.
    baseline_envelope = summary[0]
    reference, tolerances = guardrail_reference(baseline_envelope)
    summary = [
        apply_guardrails(envelope, reference, tolerances)
        for envelope in summary
    ]
    # Primary ranking is purely target-relative token balance.
    ranked = sorted(summary, key=primary_key, reverse=True)
    for rank, envelope in enumerate(ranked, start=1):
        envelope["primary_rank"] = rank
    by_arm = {row["arm"]: row for row in summary}
    if selected_arm not in by_arm:
        raise ValueError(
            f"Selected arm {selected_arm!r} is absent; available={sorted(by_arm)}"
        )
    paired = paired_epoch_diagnostics(
        output_root,
        [row["arm"] for row in summary],
        selected_arm,
    )
    analysis = {
        "experiment": settings["experiment_label"],
        "status": "completed",
        "n_arms": len(summary),
        "n_checkpoints": len(rows),
        "n_healthy": sum(bool(row["healthy"]) for row in rows),
        "arm_envelopes": summary,
        "primary_rank_order": [row["arm"] for row in ranked],
        "guardrail_reference": {
            "reference_arm": baseline_envelope["arm"],
            "values": reference,
            "tolerances": tolerances,
            "rule": (
                "adverse delta larger than max(practical floor, 2x "
                "reference-arm within-window epoch SD)"
            ),
        },
        "paired_epoch_diagnostics": paired,
        "paired_diagnostic_scope": (
            "All-epoch per-date means pooled across each arm's full "
            "trajectory, with a 5-day circular moving-block bootstrap "
            "inside each validation window. Diagnostic only; it does not "
            "capture training-seed uncertainty and does not affect ranking."
        ),
        "selection_policy": settings["evaluation"]["selection_policy"],
    }
    selection = {
        "experiment": settings["experiment_label"],
        "created_at_utc": utc_now(),
        "selected": {
            "arm": selected_arm,
            **by_arm[selected_arm],
        },
        "selection_rule": settings["evaluation"]["selection_policy"],
        "rationale": selection_rationale,
        "paired_epoch_diagnostics": paired,
        "upstream_eligible": (
            record_selection
            and not settings.get("smoke", False)
            and not settings.get("subset_run", False)
        ),
        "human_review_recorded": record_selection,
        "holdout_used": False,
        "study_manifest": str((output_root / "study_manifest.json").resolve()),
    }
    selection_filename = (
        "selection.json" if record_selection else "selection_proposal.json"
    )
    existing_selection_path = output_root / selection_filename
    if existing_selection_path.is_file():
        existing_selection = PIPE.load_json(existing_selection_path)
        if (
            existing_selection.get("selected", {}).get("arm") == selected_arm
            and existing_selection.get("study_manifest")
            == selection["study_manifest"]
        ):
            selection["created_at_utc"] = existing_selection.get(
                "created_at_utc", selection["created_at_utc"]
            )
    atomic_write_json(output_root / "analysis.json", analysis)
    atomic_write_json(output_root / selection_filename, selection)
    write_csv(output_root / "arm_envelopes.csv", summary)
    make_comparison_plots(output_root, rows, settings["experiment_label"])

    def fmt_opt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    lines = [
        f"# {settings['experiment_label']}",
        "",
        "- Status: **completed**",
        f"- Architecture: **{settings['architecture_dependency']['config']}** "
        f"({settings['architecture_dependency']['dim']}d/"
        f"{settings['architecture_dependency']['depth']}L)",
        f"- Tokenizer: **{settings['tokenizer_dependency']['embedding_dim']}x"
        f"{settings['tokenizer_dependency']['hidden_dim']} / "
        f"{settings['tokenizer_dependency']['bits_l1']}+"
        f"{settings['tokenizer_dependency']['bits_l2']} bits**",
        f"- Evaluated checkpoints: **{len(rows)}**",
        f"- Health-passing checkpoints: **{analysis['n_healthy']}**",
        f"- Reference (guardrail) arm: **{baseline_envelope['arm']}**",
        f"- Primary rank order: **{', '.join(analysis['primary_rank_order'])}**",
        f"- {'Reviewed' if record_selection else 'Provisional'} selection: "
        f"**{selected_arm}**",
        "",
        "## Full-trajectory envelopes",
        "",
        "Each arm is summarized over its full epoch trajectory as a single "
        "window (policy `all_epochs`). Primary ranking is target-relative "
        "token balance; downstream DA/RankIC/MAPE/AmpRatio are guardrails only.",
        "",
        "| Arm | Window | Best DA | Mature loss | Mature DA | Mature RankIC | "
        "Mature MAPE | P90 collapse | CB balance | Support F1 | Token JSD | "
        "Eff. align | Collapse | Unique | Amp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {row['mature_window_epochs'][0]}–"
            f"{row['mature_window_epochs'][1]} | "
            f"{row['best_da'] * 100:.2f}%@"
            f"{row['best_da_epoch']} | {row['late_median_val_loss']:.4f} | "
            f"{row['late_median_da'] * 100:.2f}% | "
            f"{row['late_median_rankic']:.4f} | "
            f"{row['late_median_mape']:.3f} | "
            f"{row['late_median_p90_collapse'] * 100:.1f}% | "
            f"{fmt_opt(row['mature_median_codebook_balance'], '.3f')} | "
            f"{fmt_opt(row['mature_median_token_support_f1'], '.3f')} | "
            f"{fmt_opt(row['mature_median_token_jsd'], '.4f')} | "
            f"{fmt_opt(row['mature_median_effective_token_alignment'], '.3f')} | "
            f"{row['late_median_collapse_rate'] * 100:.1f}% | "
            f"{row['late_median_unique']:.1f} | "
            f"{row['late_median_ampratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Selection order: coarse codebook balance (median, p10) → token "
            "support F1 → token JSD (lower) → effective token alignment → "
            "collapse rate (lower) → unique tokens. DA/RankIC/MAPE/AmpRatio "
            "only gate leadership via max(practical floor, 2x reference-arm "
            "within-window SD). A proposal is not eligible downstream until "
            "explicitly recorded after review.",
            "",
        ]
    )
    if paired:
        lines.extend(
            [
                "## Paired all-epoch diagnostics",
                "",
                f"Differences are {selected_arm} minus comparator. Per-date "
                "means are pooled across each arm's full epoch trajectory, with "
                "a 5-day circular moving-block bootstrap inside each validation "
                "window; they do not measure training-seed uncertainty.",
                "",
                "| Comparator | Metric | Difference | 95% interval | Better when |",
                "|---|---|---:|---:|---|",
            ]
        )
        for item in paired:
            low, high = item["moving_block_bootstrap_95_ci"]
            lines.append(
                f"| {item['comparator_arm']} | {item['metric']} | "
                f"{item['selected_minus_comparator']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | {item['favorable_when']} |"
            )
        lines.append("")
    atomic_write_text(output_root / "ANALYSIS.md", "\n".join(lines))
    return analysis


def build_parser(
    *,
    default_weights_root: Path,
    default_results_root: Path,
    arm_names: list[str],
    default_epochs: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_root", type=Path, default=default_weights_root
    )
    parser.add_argument(
        "--results_root", type=Path, default=default_results_root
    )
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated subset of: " + ",".join(arm_names),
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Default reads the reviewed Exp 02 results selection.json",
    )
    parser.add_argument(
        "--architecture_selection",
        type=Path,
        default=EXP03_SELECTION,
        help="Exp 03 selection.json (required before formal Exp 04)",
    )
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument(
        "--eval_offsets",
        type=parse_offsets,
        default=VALIDATION_OFFSETS,
    )
    parser.add_argument("--eval_days", type=int, default=EVAL_DAYS)
    parser.add_argument(
        "--batch_tokens",
        type=int,
        default=0,
        help="0 inherits the Exp 03 architecture setting",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="0 inherits the Exp 03 architecture setting",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument(
        "--record_selection",
        action="store_true",
        help=(
            "Write downstream-eligible selection.json after human review; "
            "requires --select_arm and --selection_rationale"
        ),
    )
    parser.add_argument("--select_arm", default="")
    parser.add_argument("--selection_rationale", default="")
    return parser


def select_arms(raw: str, definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw.strip():
        return [dict(arm) for arm in definitions]
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise ValueError("--arms did not contain an arm name")
    by_name = {arm["name"]: arm for arm in definitions}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}")
    return [dict(by_name[name]) for name in requested]


def run_ablation(
    *,
    wrapper_path: Path,
    experiment_key: str,
    experiment_label: str,
    arm_definitions: list[dict[str, Any]],
    fixed_recipe: dict[str, Any],
    default_weights_root: Path,
    default_results_root: Path,
    selected_arm: str,
    selection_rationale: str,
    required_selections: dict[str, tuple[Path, str]] | None = None,
    default_epochs: int = 30,
) -> int:
    parser = build_parser(
        default_weights_root=default_weights_root,
        default_results_root=default_results_root,
        arm_names=[arm["name"] for arm in arm_definitions],
        default_epochs=default_epochs,
    )
    args = parser.parse_args()
    verify_python_runtime()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.eval_days <= 0:
        parser.error("--eval_days must be positive")
    if args.batch_tokens < 0 or args.eval_batch_size < 0:
        parser.error("batch sizes must be non-negative")
    if any(offset + args.eval_days > HOLDOUT_OFFSET for offset in args.eval_offsets):
        parser.error("evaluation windows may not enter the sealed holdout")
    if args.record_selection and (
        not args.select_arm.strip() or not args.selection_rationale.strip()
    ):
        parser.error(
            "--record_selection requires --select_arm and "
            "--selection_rationale"
        )

    arms = select_arms(args.arms, arm_definitions)
    protocol_arm_names = [arm["name"] for arm in arm_definitions]
    if (
        len(arms) != len(arm_definitions)
        and not args.smoke
        and args.results_root.resolve() == default_results_root.resolve()
    ):
        parser.error(
            "A formal --arms subset must use separate --weights_root and "
            "--results_root paths; the defaults are reserved for the complete "
            "arm set"
        )
    requested_selected_arm = args.select_arm.strip() or selected_arm
    if requested_selected_arm not in {arm["name"] for arm in arms}:
        parser.error(
            f"--select_arm {requested_selected_arm!r} is not in the active arms"
        )
    effective_selected_arm = (
        requested_selected_arm
        if requested_selected_arm in {arm["name"] for arm in arms}
        else arms[0]["name"]
    )
    effective_rationale = (
        args.selection_rationale.strip()
        if args.record_selection
        else selection_rationale
    )
    if effective_selected_arm != selected_arm:
        effective_rationale = (
            f"Subset run excludes preregistered arm {selected_arm!r}; "
            f"{effective_selected_arm!r} is recorded only as this subset's "
            "working output. "
            + selection_rationale
        )
    tokenizer = resolve_tokenizer(args.tokenizer)
    architecture = resolve_architecture(args.architecture_selection)
    smoke = bool(args.smoke)
    requirements = required_selections or {}
    if smoke:
        existing_requirements = {
            name: requirement
            for name, requirement in requirements.items()
            if requirement[0].is_file()
        }
        upstream = load_required_selections(existing_requirements)
        for name, (path, expected_arm) in requirements.items():
            if name not in upstream:
                upstream[name] = {
                    "source": "smoke_preregistered_stub",
                    "expected_path": str(path.resolve()),
                    "expected_arm": expected_arm,
                }
    else:
        upstream = load_required_selections(requirements)
    epochs = 1 if smoke else args.epochs
    offsets = (3,) if smoke else tuple(args.eval_offsets)
    eval_days = 2 if smoke else args.eval_days
    settings = build_settings(
        experiment_key=experiment_key,
        experiment_label=experiment_label,
        arms=arms,
        protocol_arm_names=protocol_arm_names,
        fixed_recipe=fixed_recipe,
        tokenizer=tokenizer,
        architecture=architecture,
        upstream=upstream,
        epochs=epochs,
        offsets=offsets,
        eval_days=eval_days,
        batch_tokens=args.batch_tokens,
        eval_batch_size=args.eval_batch_size,
        smoke=smoke,
    )

    temporary_root: Path | None = None
    if smoke:
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f"kronos_{experiment_key}_smoke_")
        ).resolve()
        layout = StudyLayout.create(
            temporary_root / "weights", temporary_root / "results"
        )
    else:
        layout = (
            StudyLayout(
                args.weights_root.expanduser().resolve(),
                args.results_root.expanduser().resolve(),
            )
            if args.analyze_only
            else StudyLayout.create(args.weights_root, args.results_root)
        )
    output_root = layout.results_root
    shared_cache_root = layout.weights_root / "shared"

    if args.analyze_only:
        manifest_path = output_root / "study_manifest.json"
        if manifest_path.is_file():
            settings = PIPE.load_json(manifest_path)["settings"]
        rows = PIPE.load_json(output_root / "combined_epoch_summary.json")
        analyze(
            output_root,
            settings,
            rows,
            selected_arm=effective_selected_arm,
            selection_rationale=effective_rationale,
            record_selection=args.record_selection,
        )
        return 0

    PIPE.verify_runtime()
    if not smoke:
        free_gib = shutil.disk_usage(layout.weights_root).free / (1024**3)
        if free_gib < 5:
            raise RuntimeError(
                f"At least 5 GiB free is required; {free_gib:.1f} GiB remains"
            )
    prepare_manifest(output_root, settings, wrapper_path, layout)
    update_manifest(
        output_root,
        status="running",
        started_at_utc=utc_now(),
        completed_arms=[],
    )
    completed: list[str] = []
    succeeded = False
    try:
        for index, arm in enumerate(arms, start=1):
            print(
                f"\n{'=' * 72}\n[{index}/{len(arms)}] {arm['name']}: "
                f"{arm['description']}\n{'=' * 72}",
                flush=True,
            )
            run_one_arm(
                layout,
                shared_cache_root,
                settings,
                arm,
                fixed_recipe,
            )
            completed.append(arm["name"])
            rows = write_combined_summary(
                output_root, arms, fixed_recipe, settings
            )
            update_manifest(
                output_root,
                status="running",
                completed_arms=completed,
                combined_rows=len(rows),
                last_progress_at_utc=utc_now(),
            )
        update_manifest(
            output_root,
            status="completed",
            completed_at_utc=utc_now(),
            completed_arms=completed,
            combined_rows=len(rows),
        )
        analysis = analyze(
            output_root,
            settings,
            rows,
            selected_arm=effective_selected_arm,
            selection_rationale=effective_rationale,
            record_selection=args.record_selection,
        )
        update_manifest(
            output_root,
            status="completed",
            analysis_completed_at_utc=utc_now(),
            health_passing_checkpoints=analysis["n_healthy"],
        )
        write_download_manifest(layout)
        succeeded = True
        print(f"Completed {experiment_label}: {output_root}")
        return 0
    except BaseException as exc:
        update_manifest(
            output_root,
            status="failed",
            failed_at_utc=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            completed_arms=completed,
        )
        raise
    finally:
        if temporary_root is not None:
            if succeeded:
                shutil.rmtree(temporary_root, ignore_errors=False)
            else:
                print(f"Smoke artifacts retained at {temporary_root}")
