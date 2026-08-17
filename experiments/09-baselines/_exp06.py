"""Explicit bridge to the canonical Exp 06 helpers.

The 06 experiment is intentionally a directory of stage entry points rather
than a collection of importable legacy modules.  Baseline code may still use
the shared data helpers through this small, explicit bridge.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_path = ROOT / "experiments" / "06-posttrain" / "common.py"
_spec = importlib.util.spec_from_file_location("kronos_exp06_common_09", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load Exp 06 helpers from {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

load_stocks_uid = _module.load_stocks_uid
attach_close_prices_uid = _module.attach_close_prices_uid
prepare_stocks_uid = _module.prepare_stocks_uid

