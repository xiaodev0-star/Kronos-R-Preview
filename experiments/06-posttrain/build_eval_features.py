"""Build point-in-time C1 features for the eval-region rows (offsets 0..399).

The training cache already carries c1_feats; the eval hidden cache does not.
This script computes the same point-in-time features (last return, 5/20-day
momentum, 20-day realized vol, 20-day log-volume) for every eval row from the
stock's raw features, so the C1 feature-only control can be evaluated on the
400-day window.  Only rows visible strictly before position p are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig  # noqa: E402
from posttrain_data import load_stocks_uid, build_stock_arrays_uid  # noqa: E402
from build_training_cache import c1_features, MIN_POS  # noqa: E402
from posttrain_common import resolve_roots  # noqa: E402


def main():
    eval_hidden = np.load(
        "server_runs/weights/06-posttrain/seed42/hidden_cache.npz", allow_pickle=True)
    uids = eval_hidden["stock_uid"]
    positions = eval_hidden["position"]
    n = len(uids)

    stocks = load_stocks_uid(DataConfig.data_dir)
    arrays = {}
    for s in stocks:
        a = build_stock_arrays_uid(s)
        if a is not None:
            arrays[s["stock_uid"]] = a
    print(f"[eval-features] built {len(arrays)} stock arrays")

    feats = np.zeros((n, 5), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    uid_arr = np.asarray(uids)
    for i in range(n):
        a = arrays.get(str(uid_arr[i]))
        if a is None:
            continue
        cf = c1_features(a["feat"], int(positions[i]))
        if cf is None:
            continue
        feats[i] = cf
        ok[i] = True
    print(f"[eval-features] {int(ok.sum())}/{n} rows have c1 features")

    roots = resolve_roots()
    out = roots.weights_root / "eval_c1_feats.npz"
    np.savez(out, c1_feats=feats, valid=ok)
    print(f"[eval-features] wrote {out}")


if __name__ == "__main__":
    main()
