"""Explicit bridge to the canonical Exp 07 helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_path = ROOT / "experiments" / "07-bert-critic" / "common.py"
_spec = importlib.util.spec_from_file_location("kronos_exp07_common_09", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load Exp 07 helpers from {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

MlpRankHead = _module.MlpRankHead
upstream_paths = _module.upstream_paths

