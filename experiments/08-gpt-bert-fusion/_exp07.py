"""Explicit import bridge to the consolidated Exp 07 helpers.

Exp 08 has its own ``common.py``-like modules and therefore cannot rely on
Python's short module name ``common``.  This bridge keeps the dependency
explicit while Exp 07 remains a three-entry-point experiment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "07-bert-critic" / "common.py"
_spec = importlib.util.spec_from_file_location("kronos_exp07_common", SOURCE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load Exp 07 helpers from {SOURCE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = getattr(_module, _name)
