"""07-bert-critic shared infrastructure (inherits 06 posttrain_common).

Responsibilities:
  - resolve the 07 dual roots (weights / results)
  - reuse the reviewed CPT selection (upstream_eligible, holdout seals)
  - reuse the offset<400 guard, trial ledger, hashing
  - hard leakage constants (cutoff date, calibration slice)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIX_DIR = ROOT / "experiments" / "06-posttrain"
SEVEN_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SIX_DIR, SEVEN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import StudyLayout, default_study_roots  # noqa: E402
from posttrain_common import (  # noqa: E402
    load_reviewed_selection,
    upstream_checkpoint_path,
    require_offsets,
    dict_sha256,
    file_fingerprint,
    append_trial,
    load_trial_ledger,
    load_json,
    write_json,
    utc_now,
    HOLDOUT_OFFSET,
    HOLDOUT_DAYS,
    VALIDATION_OFFSETS,
    CPT_SELECTION_PATH,
)

EXP_KEY = "07-bert-critic"
CUTOFF_DATE = "2024-02-01"
CALIB_START, CALIB_STOP = "2023-02-01", "2024-02-01"


def resolve_roots(seed: int = 42) -> StudyLayout:
    """Resolve and create the 07 dual roots for one seed."""
    weights, results = default_study_roots(EXP_KEY, seed=seed)
    return StudyLayout.create(weights, results)


def upstream_paths(selection_path=CPT_SELECTION_PATH):
    """Return (checkpoint_path, tokenizer_path) from the reviewed selection."""
    sel = load_reviewed_selection(selection_path)
    up = sel.get("upstream") or sel.get("parent_selection", {})
    ckpt = ROOT / Path(up["checkpoint"])
    tok = ROOT / Path(up["tokenizer"])
    return ckpt, tok
