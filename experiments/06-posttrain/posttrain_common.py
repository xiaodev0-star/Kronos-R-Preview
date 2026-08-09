"""PT-00 shared PostTrain infrastructure.

Responsibilities:
  - load + validate the reviewed CPT selection (upstream_eligible, holdout seals)
  - resolve dual roots (weights / results) via experiment_io.StudyLayout
  - file/dict hashing for cache keys and manifests
  - hard offset<400 guard for ordinary runners
  - trial ledger for every attempt (including failures/OOM/aborts)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_io import StudyLayout, default_study_roots, file_sha256

ROOT = Path(__file__).resolve().parents[2]
POSTTRAIN_DIR = Path(__file__).resolve().parent

HOLDOUT_OFFSET = 400
HOLDOUT_DAYS = 80
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, 1))

CPT_SELECTION_PATH = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "selection.json"
POSTTRAIN_SELECTION_PATH = ROOT / "server_runs" / "results" / "06-posttrain" / "seed42" / "selection.json"

EXP_KEY = "06-posttrain"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_roots(seed: int = 42) -> StudyLayout:
    """Resolve and create the PostTrain dual roots for one seed."""
    weights, results = default_study_roots(EXP_KEY, seed=seed)
    return StudyLayout.create(weights, results)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def dict_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


# ============================================================================
# Reviewed selection
# ============================================================================

class SelectionError(RuntimeError):
    pass


def _first_true(*candidates) -> bool:
    return any(bool(c) for c in candidates if c)


def load_reviewed_selection(path: Path = CPT_SELECTION_PATH) -> dict:
    """Load and validate a reviewed selection for PostTrain consumption.

    Accepts both the revised CPT selection (``upstream.upstream_eligible``,
    per-branch ``human_review_recorded``) and the PostTrain selection
    (top-level ``upstream_eligible`` / ``human_review_recorded`` / ``protocol``).

    Raises SelectionError unless: human_review_recorded, upstream_eligible,
    holdout_used false, and the upstream checkpoint exists with the recorded
    SHA256.  ``formal`` runner must not offer a checkpoint override.
    """
    sel = load_json(path)
    protocol = sel.get("protocol", {})
    reviewed = _first_true(
        sel.get("human_review_recorded"),
        sel.get("upstream", {}).get("human_review_recorded"),
        any(br.get("human_review_recorded") for br in sel.get("branches", {}).values()),
    )
    if not reviewed:
        raise SelectionError(f"selection {path} is not human-reviewed")
    eligible = _first_true(
        sel.get("upstream_eligible"),
        sel.get("upstream", {}).get("upstream_eligible"),
    )
    if not eligible:
        raise SelectionError(f"selection {path} is not upstream-eligible")
    holdout_used = _first_true(sel.get("holdout_used"),
                               protocol.get("holdout_used"))
    if holdout_used:
        raise SelectionError(f"selection {path} claims holdout already used")

    # upstream pointer can be at ``upstream`` (CPT) or ``parent_selection`` (PostTrain)
    up = sel.get("upstream") or sel.get("parent_selection", {})
    ckpt_rel = up.get("checkpoint") or up.get("checkpoint_id")
    if not ckpt_rel:
        raise SelectionError(f"selection {path} has no upstream checkpoint pointer")
    ckpt = ROOT / ckpt_rel
    if not ckpt.exists():
        raise SelectionError(f"upstream checkpoint missing: {ckpt}")
    recorded_sha = up.get("checkpoint_sha256")
    if recorded_sha:
        actual = file_sha256(ckpt)
        if actual != recorded_sha:
            raise SelectionError(
                f"upstream SHA256 mismatch: recorded {recorded_sha} != actual {actual}")
    return sel


def upstream_checkpoint_path(sel: dict) -> Path:
    up = sel.get("upstream") or sel.get("parent_selection", {})
    return ROOT / Path(up["checkpoint"])


# ============================================================================
# Offset guard
# ============================================================================

def require_offsets(offsets):
    """Ordinary runners must not interpret any offset >= 400 as authorized."""
    offsets = list(offsets)
    if any(o >= HOLDOUT_OFFSET for o in offsets):
        raise SelectionError(
            f"refusing offsets >= {HOLDOUT_OFFSET}: {[o for o in offsets if o >= HOLDOUT_OFFSET]} "
            "is the sealed final holdout window; only an explicit final command may open it")
    if any(o < 0 for o in offsets):
        raise ValueError(f"negative offset: {offsets}")
    return tuple(offsets)


# ============================================================================
# Trial ledger
# ============================================================================

LEDGER_PATH = POSTTRAIN_DIR / "trial_ledger.jsonl"


def append_trial(entry: dict) -> None:
    """Append a trial record (including failures/OOM/aborts)."""
    row = {"ts": utc_now(), **entry}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_trial_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    rows = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows
