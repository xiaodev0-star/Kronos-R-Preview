"""data.py — shared data layer for the 09-baselines comparison.

Two dataset families, both point-in-time (no future leakage):

  A) Feature rows for XGBoost / MLP
       fit : c1_feats [905072, 5] + true_logret (06 training_cache)
       eval: c1_feats [1798899, 5]              (06 eval_c1_feats, eval-row order)
     c1_feats = [last-return, 5d momentum, 20d momentum, 20d vol, 20d volume].

  B) Return-window sequences for Transformer / TimeFM-style
       X[i, t] = raw log-return at (stock_i, position_i - W + t),
       for t in [0, W)  →  all BEFORE the target row p_i (p_i predicts feat[p_i]).
       fit : [905072, W], eval: [1798899, W]; targets = true_logret.
     Built once from the per-stock feature arrays and CACHED to
     ``checkpoints/seq_<region>_w<W>.npz``.

Every array is aligned to the shared eval-row contract in common.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from common import ROOT, W06, W07, CHECKPOINTS, OUTPUTS

# make project modules importable
for _p in (ROOT / "experiments" / "06-posttrain",
           ROOT / "experiments" / "07-bert-critic", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# A) Feature rows (XGBoost / MLP)
# ---------------------------------------------------------------------------
def load_fit_features():
    """c1_feats + true_logret + date_key + stock_uid for the fit region."""
    tc = np.load(W06 / "training_cache.npz", allow_pickle=True)
    return {"X": tc["c1_feats"], "y": tc["true_logret"].astype(np.float64),
            "date_key": tc["date_key"], "stock_uid": tc["stock_uid"]}


def load_eval_features():
    """c1_feats for the eval rows (aligned to candidates_eval order)."""
    ec = np.load(W06 / "eval_c1_feats.npz", allow_pickle=True)
    return {"X": ec["c1_feats"], "valid": ec["valid"].astype(bool)}


# ---------------------------------------------------------------------------
# B) Return-window sequences (Transformer / TimeFM-style)
# ---------------------------------------------------------------------------
def _load_stock_series():
    """per-stock {dates_int, rets} from the prepared feature arrays."""
    from posttrain_data import load_stocks_uid, attach_close_prices_uid, prepare_stocks_uid
    from model import load_tokenizer
    from critic_common import upstream_paths
    from config import DataConfig
    import torch
    _, tok_path = upstream_paths()
    tok = load_tokenizer(str(tok_path), torch.device("cpu"))
    # data_dir in config is relative; resolve against the repo root so this
    # works regardless of the process CWD.
    data_dir = Path(DataConfig.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    stocks = load_stocks_uid(str(data_dir))
    attach_close_prices_uid(stocks)
    prepped = prepare_stocks_uid(stocks, tok, torch.device("cpu"))
    out = {}
    for p in prepped:
        dr = np.asarray(p.get("dates_raw") if p.get("dates_raw") is not None else p.get("dates"))
        if dr is None or np.ndim(dr) == 0 or len(dr) == 0:
            continue
        dates_int = np.asarray(
            [int(str(d)[:10].replace("-", "")) for d in dr], dtype=np.int32)
        out[p["stock_uid"]] = {"dates_int": dates_int, "rets": p["feat"][:, 0]}
    return out


def build_sequences(region, window, max_rows=0, cache=True):
    """Build [N, window] return-window inputs for fit or eval.

    fit : from training_cache rows (positions recomputed via stock date lookup)
    eval: from candidates_eval rows (positions stored in candidates)
    Cached to checkpoints/seq_<region>_w<window>.npz for fast re-runs.
    """
    cache_path = CHECKPOINTS / f"seq_{region}_w{window}.npz"
    if cache and cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print(f"[data] building {region} sequences (w={window})...", flush=True)
    stocks = _load_stock_series()
    if region == "fit":
        tc = np.load(W06 / "training_cache.npz", allow_pickle=True)
        uids = np.asarray([str(u) for u in tc["stock_uid"]])
        dates = np.asarray([str(d)[:10] for d in tc["date_key"]])
        y = tc["true_logret"].astype(np.float64)
    elif region == "calib":
        c = np.load(W07 / "candidates_calib_K8.npz", allow_pickle=True)
        uids = np.asarray([str(u) for u in c["stock_uid"]])
        dates = np.asarray([str(d)[:10] for d in c["date_key"]])
        y = c["true_logret"].astype(np.float64)
    else:
        c = np.load(W07 / "candidates_eval_K8.npz", allow_pickle=True)
        uids = np.asarray([str(u) for u in c["stock_uid"]])
        dates = np.asarray([str(d)[:10] for d in c["date_key"]])
        y = c["true_logret"].astype(np.float64)
    if max_rows:
        uids, dates, y = uids[:max_rows], dates[:max_rows], y[:max_rows]

    dint = np.asarray([int(d.replace("-", "")) for d in dates])
    uniq, inv = np.unique(uids, return_inverse=True)
    N = len(uids)
    X = np.full((N, window), np.nan, dtype=np.float32)
    for ui, uid in enumerate(uniq):
        st = stocks.get(str(uid))
        if st is None:
            continue
        ref = st["dates_int"]
        rets = st["rets"]
        here = np.where(inv == ui)[0]
        didx = np.searchsorted(ref, dint[here], side="left")
        ok = (didx < len(ref)) & (ref[didx] == dint[here])
        pos = np.where(ok, didx, -1)
        valid = ok & (pos >= window)
        pv = pos[valid]
        idx_rows = np.arange(window)[None, :].repeat(len(pv), 0)
        col = pv[:, None] - window + idx_rows
        X[here[valid]] = rets[col]
        if ui % 500 == 0:
            print(f"[data] {region} stock {ui}/{len(uniq)} valid={int(valid.sum())}", flush=True)

    out = {"X": X, "y": y, "date_key": dates, "stock_uid": uids}
    if cache:
        np.savez(cache_path, **{k: np.asarray(v) for k, v in out.items()})
        print(f"[data] cached -> {cache_path}")
    return out


def load_fit_sequences(window=32, max_rows=0):
    return build_sequences("fit", window, max_rows=max_rows)


def load_eval_sequences(window=32, max_rows=0):
    return build_sequences("eval", window, max_rows=max_rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--region", choices=["fit", "eval", "calib"], default="fit")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()
    d = build_sequences(args.region, args.window, max_rows=args.max_rows)
    print(f"[data] {args.region}: X{d['X'].shape} y{d['y'].shape} "
          f"finite_frac={np.isfinite(d['X']).mean():.3f}")
