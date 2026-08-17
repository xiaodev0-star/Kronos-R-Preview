"""Exp 06 PostTrain shared infrastructure.

The experiment entry points are deliberately small, while this module owns
the numerical/data helpers and the canonical artifact names. Keeping the
paths here prevents downstream experiments from having to know the old
``P1/P6/S3`` trial labels.
"""
from __future__ import annotations

import sys
from pathlib import Path
from glob import glob
from scipy.stats import spearmanr
from torch.utils.data import IterableDataset
from typing import Any
import argparse
import hashlib
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from config import DataConfig, NormConfig
from data_processor import document_normalize, _stock_cutoff_idx
from data_processor import split_stocks
from eval_helpers import load_gpt
from experiment_io import StudyLayout, default_study_roots, file_sha256
from model import load_tokenizer

# ===== posttrain_common =====
HOLDOUT_OFFSET = 400
HOLDOUT_DAYS = 80
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, 1))

CPT_SELECTION_PATH = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "selection.json"
POSTTRAIN_SELECTION_PATH = ROOT / "server_runs" / "results" / "06-posttrain" / "seed42" / "selection.json"

EXP_KEY = "06-posttrain"


def resolve_roots(seed: int = 42) -> StudyLayout:
    """Resolve and create the PostTrain dual roots for one seed."""
    weights, results = default_study_roots(EXP_KEY, seed=seed)
    return StudyLayout.create(weights, results)


def artifact_paths(seed: int = 42, roots: StudyLayout | None = None) -> dict[str, Path]:
    """Return canonical Exp 06 artifact paths.

    ``shared`` contains reusable caches, while ``A/B/C`` mirrors the three
    experiment stages. Human-readable names are intentional: the old PT
    trial labels were implementation IDs, not useful artifact names.
    """
    layout = roots or resolve_roots(seed)
    shared = layout.weights_root / "shared"
    stage_a = layout.weights_root / "A-decode"
    stage_b = layout.weights_root / "B-heads"
    stage_c = layout.weights_root / "C-cross-sectional"
    return {
        "prepared": shared / "prepared-inputs.pt",
        "hidden": shared / "hidden-cache.npz",
        "training": shared / "training-cache.npz",
        "calibration": shared / "calibration-cache.npz",
        "records": stage_a / "posterior-records.npz",
        "eval_features": stage_b / "eval-features.npz",
        "head_dir": stage_b,
        "cross_dir": stage_c,
        "head_return_linear": stage_b / "return-linear.pt",
        "head_return_mlp": stage_b / "return-mlp.pt",
        "head_direction_linear": stage_b / "direction-linear.pt",
        "head_direction_mlp": stage_b / "direction-mlp.pt",
        "head_rank_linear_pairwise": stage_b / "rank-linear-pairwise.pt",
        "head_rank_mlp_spearman": stage_b / "rank-mlp-spearman.pt",
        "head_rank_mlp_pairwise": stage_b / "rank-mlp-pairwise.pt",
        "head_feature_only": stage_b / "feature-only.pt",
        "head_independent_mlp": stage_c / "independent-mlp.pt",
        "head_deepsets": stage_c / "deepsets.pt",
        "head_isab": stage_c / "isab.pt",
    }


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


# ===== posttrain_data =====
CUTOFF_DATE = DataConfig.cutoff_date          # 2024-02-01
SPLIT_VERSION = "posttrain_split_v1"


# ============================================================================
# stock_uid
# ============================================================================

def source_relpath_for(data_dir, fpath):
    """Normalized source relative path (posix) under ``data_dir``."""
    full = Path(fpath).resolve()
    base = Path(data_dir).resolve()
    rel = full.relative_to(base)
    return rel.as_posix()


def stock_uid_from_relpath(relpath):
    """Stable UID from a normalized source relpath."""
    rel = relpath.replace("\\", "/")
    # Keep it readable and stable; no hashing needed since relpaths are stable.
    return rel


# ============================================================================
# UID-keyed CSV loading
# ============================================================================

def _uid_cache_fingerprint(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    relpaths = sorted(source_relpath_for(data_dir, f) for f in files)
    total = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    h = hashlib.sha256()
    for r in relpaths:
        h.update(r.encode("utf-8"))
        h.update(b"\x00")
    h.update(str(total).encode("utf-8"))
    return {
        "schema": 1,
        "file_count": len(files),
        "total_bytes": total,
        "relpath_sha256": h.hexdigest(),
        "max_stocks": int(DataConfig.max_stocks),
    }


def _uid_cache_path(data_dir):
    fp = _uid_cache_fingerprint(data_dir)
    digest = hashlib.sha256(
        json.dumps(fp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(data_dir, ".cache", f"posttrain_stocks_{digest}.pkl"), fp


def load_stocks_uid(data_dir=DataConfig.data_dir, use_cache=True):
    """Load all stocks with ``stock_uid``/``source_relpath`` attached.

    Returns list of dicts with keys: stock_uid, source_relpath, symbol,
    features_raw [T,6], dates_dt [T], day/month/year [T].  No normalization
    stats here (they are per-cutoff, computed by build_stock_arrays_uid).
    """
    if use_cache:
        cache_path, fp = _uid_cache_path(data_dir)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fp:
                    print(f"  [uid-cache] Loaded {len(cached['stocks'])} stocks")
                    return cached["stocks"]
            except Exception:
                pass

    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    if DataConfig.max_stocks and DataConfig.max_stocks < len(files):
        rng = np.random.RandomState(DataConfig.random_seed)
        idx = rng.choice(len(files), DataConfig.max_stocks, replace=False)
        files = [files[i] for i in sorted(idx)]

    feature_cols = DataConfig.feature_cols
    stocks = []
    for fpath in files:
        try:
            rel = source_relpath_for(data_dir, fpath)
            df = pd.read_csv(fpath)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close", "volume"])
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) < NormConfig.min_lookback + 10:
                continue
            symbol = (
                str(df["symbol"].iloc[0])
                if "symbol" in df.columns
                else os.path.basename(fpath).split(".")[0]
            )
            if not all(c in df.columns for c in
                       ["date", "close", "high", "low", "open", "volume"]):
                continue
            prev_close = df["close"].shift(1)
            df["log_ret"] = np.log(df["close"] / prev_close).replace(
                [np.inf, -np.inf], np.nan)
            df["log_high"] = np.log1p(df["high"] / df["close"] - 1)
            df["log_low"] = np.log1p(df["low"] / df["close"] - 1)
            df["log_open"] = np.log1p(df["open"] / df["close"] - 1)
            df["log_vol"] = np.log1p(df["volume"])
            df["log_amt"] = (
                np.log1p(df["amount"])
                if "amount" in df.columns
                else np.log1p(df["volume"] * df["close"])
            )
            df = df.dropna().reset_index(drop=True)
            features = df[feature_cols].values.astype(np.float32)
            dates_dt = df["date"].values
            stocks.append({
                "stock_uid": stock_uid_from_relpath(rel),
                "source_relpath": rel,
                "symbol": symbol,
                "features_raw": features,
                "dates_dt": dates_dt,
                "day": df["date"].dt.day.values.astype(np.int64),
                "month": df["date"].dt.month.values.astype(np.int64),
                "year": (df["date"].dt.year - 2010).clip(0, 99).values.astype(np.int64),
            })
        except Exception:
            continue

    if use_cache and stocks:
        cache_path, fp = _uid_cache_path(data_dir)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"fingerprint": fp, "stocks": stocks}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    print(f"  [uid] Loaded {len(stocks)} stocks (uid-keyed)")
    return stocks


# ============================================================================
# posttrain_split_v1
# ============================================================================

def posttrain_split_v1(stocks, split_ratio=0.875, seed=42):
    """Assign each stock_uid a role on the UID axis (fit / audit).

    Sorts by stock_uid, permutes once with RandomState(seed), takes the first
    ``floor(0.875*N)`` as fit_uids and the rest as audit_uids.  Missing labels in
    some date interval only mark absence; they never move a UID between groups.
    """
    uids = sorted({s["stock_uid"] for s in stocks})
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uids))
    ordered = [uids[i] for i in perm]
    n_fit = max(1, int(np.floor(split_ratio * len(ordered))))
    fit_uids = set(ordered[:n_fit])
    audit_uids = set(ordered[n_fit:])
    return fit_uids, audit_uids


# time boundaries (target-date, left-closed right-open)
SPLIT_FOLDS = {
    # fold: (fit_dates_start, fit_dates_stop, val_start, val_stop)
    "R0": (None, "2020-02-01", "2020-02-01", "2021-02-01"),
    "R1": (None, "2021-02-01", "2021-02-01", "2022-02-01"),
    "R2": (None, "2022-02-01", "2022-02-01", "2023-02-01"),
    "final_head_fit": (None, "2023-02-01", None, None),
}
CALIBRATION = ("2023-02-01", "2024-02-01")


def in_date_interval(d64, start, stop):
    """True if datetime64 ``d64`` in [start, stop) (both string or None)."""
    if start is not None and np.datetime64(start) > d64:
        return False
    if stop is not None and d64 >= np.datetime64(stop):
        return False
    return True


def _fold_label_for_target_date(d64, uids, fit_uids, audit_uids):
    """Return the fold role(s) a (target_date, uid) belongs to, or None."""
    return None  # resolved per fold table below


def write_split_definition(stocks, fit_uids, audit_uids, output_path):
    """Write ``split_definition.json`` with every UID role and sample counts."""
    rows = []
    for s in sorted(stocks, key=lambda x: x["stock_uid"]):
        role = "fit" if s["stock_uid"] in fit_uids else "audit"
        rows.append({"stock_uid": s["stock_uid"], "role": role})
    n_fit = len(fit_uids)
    n_audit = len(audit_uids)
    payload = {
        "schema_version": SPLIT_VERSION,
        "created_at_utc": "2026-08-05",
        "uid_axis": {
            "method": f"RandomState({42}) permutation of sorted uid; first floor(0.875*N)",
            "fit_count": n_fit,
            "audit_count": n_audit,
            "total": n_fit + n_audit,
        },
        "time_axis": {
            "cutoff_date": CUTOFF_DATE,
            "folds": {k: {"fit_dates": [str(v[0]), str(v[1])],
                          "validation_dates": [str(v[2]), str(v[3])]}
                      for k, v in SPLIT_FOLDS.items()},
            "calibration": {"uids": "audit", "dates": list(CALIBRATION)},
        },
        "rows": rows,
    }
    import hashlib
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload["sha256"]


# ============================================================================
# Per-stock GPT inputs (mirrors eval_helpers._prepare_stocks_batch, + uid)
# ============================================================================

def attach_close_prices_uid(stocks):
    """Attach date-aligned ``close_prices`` [T] float64 per stock (by UID)."""
    csv_map = {}
    for s in stocks:
        csv_map[s["source_relpath"]] = s["source_relpath"]
    for s in stocks:
        fpath = os.path.join(DataConfig.data_dir, s["source_relpath"])
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (df.dropna(subset=["date", "close"])
                    .sort_values("date").drop_duplicates("date", keep="last"))
            close_by_date = df.set_index("date")["close"]
            target_dates = pd.to_datetime(s["dates_dt"], errors="coerce")
            aligned = close_by_date.reindex(target_dates).to_numpy(dtype=np.float64)
            if len(aligned) == len(s["features_raw"]) and np.isfinite(aligned).all():
                s["close_prices"] = aligned
                continue
        lr = s["features_raw"][:, 0]
        s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)


def build_stock_arrays_uid(stock):
    """Per-stock arrays mirroring ``eval_helpers.build_stock_arrays`` + uid."""
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    close = stock.get("close_prices")
    if close is None:
        close = np.exp(np.cumsum(feat[:, 0])).astype(np.float64)
    ci = _stock_cutoff_idx(stock, CUTOFF_DATE)
    T_total = len(feat)
    m = NormConfig.min_lookback
    if T_total < m + 10 or ci < m:
        return None
    price_normed, va_normed = document_normalize(feat, cutoff_idx=ci)
    price_feat = feat[:, :4]
    p_mean = price_feat[:ci].mean(axis=0)
    p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)
    return {
        "stock_uid": stock["stock_uid"],
        "source_relpath": stock["source_relpath"],
        "symbol": str(stock.get("symbol", "unknown")),
        "feat": feat, "close": close, "ci": ci, "T_total": T_total,
        "p_mean": p_mean, "p_std": p_std,
        "price_normed": price_normed, "va_normed": va_normed,
        "day": day, "month": month, "year": year,
        "dates_dt": stock.get("dates_dt", None),
    }


def prepare_stocks_uid(stocks, tokenizer, device, max_T_chunk=64):
    """Batch-tokenize and return per-stock prepared dicts (mirrors
    ``eval_helpers._prepare_stocks_batch``) with stock_uid attached."""
    arrays_list = []
    for s in stocks:
        a = build_stock_arrays_uid(s)
        if a is not None:
            arrays_list.append(a)
    if not arrays_list:
        return []

    cpu = torch.device("cpu")
    max_T = max(a["T_total"] for a in arrays_list)
    N = len(arrays_list)
    all_idx_np = np.zeros((N, max_T), dtype=np.int32)
    all_fine_idx_np = np.zeros((N, max_T), dtype=np.int16)

    for i in range(0, N, max_T_chunk):
        end = min(i + max_T_chunk, N)
        chunk_len = end - i
        chunk_np = np.zeros((chunk_len, max_T, 4), dtype=np.float32)
        for k, a in enumerate(arrays_list[i:end]):
            T = a["T_total"]
            chunk_np[k, :T] = a["price_normed"][:T]
        with torch.no_grad():
            all_idx = tokenizer.encode_all(torch.from_numpy(chunk_np).to(device))
        host = all_idx.cpu().numpy()
        all_idx_np[i:end, :] = host[..., 0].astype(np.int32)
        all_fine_idx_np[i:end, :] = host[..., 1].astype(np.int16)
        del chunk_np, all_idx, host

    vocab = tokenizer.vocab_coarse
    bos_id = vocab
    results = []
    for j, a in enumerate(arrays_list):
        T = a["T_total"]
        token_ids = all_idx_np[j, :T]
        fine_ids = all_fine_idx_np[j, :T]
        day, month, year = a["day"], a["month"], a["year"]
        inp_ids = [bos_id] + token_ids.tolist()
        seq_len = len(inp_ids) - 1
        va_seq = np.concatenate([np.zeros((1, 2), dtype=np.float32),
                                 a["va_normed"][:T - 1]], axis=0)
        dates_dt = a.get("dates_dt")
        if dates_dt is not None and len(dates_dt) >= T:
            dates_full = [str(d)[:10] for d in dates_dt]
        else:
            dates_full = ["unknown"] * T
        dates_aligned = [dates_full[0]] + dates_full[:T - 1]
        results.append({
            "stock_uid": a["stock_uid"],
            "source_relpath": a["source_relpath"],
            "symbol": a["symbol"],
            "inp_ids": inp_ids[:-1],
            "coarse_token_ids": token_ids,
            "fine_token_ids": fine_ids,
            "day": [int(day[0])] + day[:T - 1].tolist(),
            "month": [int(month[0])] + month[:T - 1].tolist(),
            "year": [int(year[0])] + year[:T - 1].tolist(),
            "dates": dates_aligned,
            "dates_raw": dates_full,
            "va": va_seq,
            "seq_len": seq_len,
            "test_pos": a["ci"],
            "p_mean": a["p_mean"], "p_std": a["p_std"],
            "feat": a["feat"], "close": a["close"],
        })
    return results


def build_prepared_batches(prepped, offsets, n_days, batch_size):
    """Pack per-stock prepped dicts into batches (mirrors
    ``evaluate_epoch_trajectory.prepare_stocks`` packing, + stock_uid)."""
    from eval_helpers import _bucket_by_length

    min_required = min(n_days, 10)
    valid = []
    for stock in prepped:
        selections = []
        for window_start in offsets:
            first_position = stock["test_pos"] + window_start
            if first_position + min_required > stock["seq_len"]:
                continue
            for within_window in range(n_days):
                position = first_position + within_window
                if position >= stock["seq_len"]:
                    break
                date_key = (stock["dates_raw"][position]
                            if position < len(stock["dates_raw"]) else "unknown")
                selections.append((window_start, position, date_key))
        if not selections:
            continue
        required_length = stock["seq_len"]
        positions = np.asarray([item[1] for item in selections], dtype=np.int64)
        features = stock["feat"]
        coarse_ids = stock["coarse_token_ids"]
        fine_ids = stock["fine_token_ids"]
        closes = stock["close"]
        time_ids = torch.empty(required_length, 3, dtype=torch.long)
        time_ids[:, 0] = torch.as_tensor(stock["day"][:required_length], dtype=torch.long)
        time_ids[:, 1] = torch.as_tensor(stock["month"][:required_length], dtype=torch.long)
        time_ids[:, 2] = torch.as_tensor(stock["year"][:required_length], dtype=torch.long)
        valid.append({
            "stock_uid": stock["stock_uid"],
            "symbol": stock["symbol"],
            "seq_len": required_length,
            "input_ids": torch.as_tensor(stock["inp_ids"][:required_length], dtype=torch.long),
            "time_ids": time_ids,
            "va_values": torch.as_tensor(stock["va"][:required_length], dtype=torch.float32),
            "selection_positions": torch.from_numpy(positions),
            "window_starts": torch.as_tensor([item[0] for item in selections], dtype=torch.int32),
            "date_keys": [item[2] for item in selections],
            "p_mean_0": float(stock["p_mean"][0]),
            "p_std_0": float(stock["p_std"][0]),
            "true_logret": torch.as_tensor([float(features[p, 0]) for p in positions],
                                           dtype=torch.float64),
            "true_coarse_ids": torch.as_tensor([int(coarse_ids[p]) for p in positions],
                                               dtype=torch.int16),
            "true_fine_ids": torch.as_tensor([int(fine_ids[p]) for p in positions],
                                             dtype=torch.int16),
            "base_close": torch.as_tensor([float(closes[p - 1]) for p in positions],
                                          dtype=torch.float64),
            "true_close": torch.as_tensor([float(closes[p]) for p in positions],
                                          dtype=torch.float64),
            "regime_ids": None,
        })

    if not valid:
        raise RuntimeError("No stocks have usable observations for requested windows")
    buckets = _bucket_by_length(valid, tolerance=100)
    packed = []
    for bucket in buckets:
        for start in range(0, len(bucket), batch_size):
            stocks_batch = bucket[start:start + batch_size]
            batch_count = len(stocks_batch)
            max_len = max(s["seq_len"] for s in stocks_batch)
            input_ids = torch.zeros(batch_count, max_len, dtype=torch.long)
            time_ids = torch.zeros(batch_count, max_len, 3, dtype=torch.long)
            va_values = torch.zeros(batch_count, max_len, 2, dtype=torch.float32)
            lengths = torch.empty(batch_count, dtype=torch.long)
            selection_ptr = [0]
            selection_rows, selection_positions, window_starts = [], [], []
            date_keys, stock_uids, symbols = [], [], []
            p_means, p_stds, true_logrets = [], [], []
            true_coarse_ids, true_fine_ids = [], []
            base_closes, true_closes = [], []
            for row, stock in enumerate(stocks_batch):
                length = stock["seq_len"]
                lengths[row] = length
                input_ids[row, :length] = stock["input_ids"]
                time_ids[row, :length] = stock["time_ids"]
                va_values[row, :length] = stock["va_values"]
                count = len(stock["selection_positions"])
                selection_rows.append(torch.full((count,), row, dtype=torch.long))
                selection_positions.append(stock["selection_positions"])
                window_starts.append(stock["window_starts"])
                date_keys.extend(stock["date_keys"])
                stock_uids.extend([stock["stock_uid"]] * count)
                symbols.extend([stock["symbol"]] * count)
                p_means.append(torch.full((count,), stock["p_mean_0"], dtype=torch.float64))
                p_stds.append(torch.full((count,), stock["p_std_0"], dtype=torch.float64))
                true_logrets.append(stock["true_logret"])
                true_coarse_ids.append(stock["true_coarse_ids"])
                true_fine_ids.append(stock["true_fine_ids"])
                base_closes.append(stock["base_close"])
                true_closes.append(stock["true_close"])
                selection_ptr.append(selection_ptr[-1] + count)
            packed.append({
                "input_ids": input_ids,
                "time_ids": time_ids,
                "va_values": va_values,
                "lengths": lengths,
                "selection_ptr": torch.as_tensor(selection_ptr, dtype=torch.long),
                "selection_rows": torch.cat(selection_rows),
                "selection_positions": torch.cat(selection_positions),
                "window_starts": torch.cat(window_starts),
                "date_keys": date_keys,
                "stock_uids": stock_uids,
                "symbols": symbols,
                "p_means": torch.cat(p_means),
                "p_stds": torch.cat(p_stds),
                "true_logrets": torch.cat(true_logrets),
                "true_coarse_ids": torch.cat(true_coarse_ids),
                "true_fine_ids": torch.cat(true_fine_ids),
                "base_closes": torch.cat(base_closes),
                "true_closes": torch.cat(true_closes),
                "regime_ids": None,
            })
    return packed


# ============================================================================
# DailyCrossSectionLoader (PT-00D / PT-04)
# ============================================================================

class DailyCrossSectionLoader(IterableDataset):
    """Yield one microbatch per target date with the full legal cross-section.

    Every microbatch is a single date with no duplicate ``stock_uid``.  Used by
    cross-sectional rank/set/decision loss and context loaders (PT-04/05/08).
    It consumes per-row hidden-cache records.

    Input ``records`` is a list of dicts, each with at least: ``date_key``,
    ``stock_uid``, ``offset``, and any extra feature/target fields (e.g.
    ``hidden``, ``raw_logret``, ``direction``, ``rank``).
    """

    def __init__(self, records, min_stocks=30, seed=42, shuffle_dates=True,
                 group_by="date_key"):
        super().__init__()
        self.min_stocks = int(min_stocks)
        self.seed = int(seed)
        self.shuffle_dates = shuffle_dates
        self.group_by = group_by
        self._index = {}
        for rec in records:
            key = rec[group_by]
            self._index.setdefault(key, []).append(rec)
        self._date_list = sorted(self._index.keys())
        self._date_list = [d for d in self._date_list
                           if len(self._index[d]) >= self.min_stocks]

    def __len__(self):
        return len(self._date_list)

    def _iter_dates(self):
        rng = np.random.RandomState(self.seed)
        dates = list(self._date_list)
        if self.shuffle_dates:
            rng.shuffle(dates)
        return dates

    def __iter__(self):
        for date in self._iter_dates():
            rows = self._index[date]
            # By construction the hidden cache holds one row per stock_uid per
            # date; assert uniqueness to catch any cache build defect.
            uids = [r["stock_uid"] for r in rows]
            if len(set(uids)) != len(uids):
                raise RuntimeError(
                    f"DailyCrossSectionLoader found duplicate stock_uid on {date}")
            yield date, rows

# ===== posttrain_heads =====
# ============================================================================
# Losses
# ============================================================================

def soft_rank(scores, tau=1.0):
    """Differentiable soft rank (sigmoid trick): r_i = sum_j sigmoid((s_i - s_j)/tau).

    Monotone in ``scores``, matches order exactly as tau -> 0.  O(n^2).
    """
    d = scores.unsqueeze(-1) - scores.unsqueeze(-2)   # [B, B]
    return torch.sigmoid(d / tau).sum(-1)


def soft_spearman_loss(scores, true_ranks, tau=1.0):
    """1 - Pearson(sigmoid soft-rank, true rank percentile), per date batch."""
    r = soft_rank(scores, tau).float()
    t = true_ranks.float()
    r = r - r.mean()
    t = t - t.mean()
    denom = (r * r).sum().clamp_min(1e-9) * (t * t).sum().clamp_min(1e-9)
    corr = (r * t).sum() / denom.sqrt()
    return (1.0 - corr).clamp_min(0.0)


def pairwise_logistic_loss(scores, true_logret, tau=1.0, dead_zone=None,
                           max_pairs=100_000, seed=42):
    """Pairwise logistic rank loss within one date batch, with near-tie dead zone.

    Pairs are sampled within the batch (one date).  ``dead_zone`` (absolute raw
    log-return gap) pre-registered from the pre-cutoff train quantile; pairs
    with |y_i - y_j| < dead_zone are skipped.
    """
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    if dead_zone is None:
        dead_zone = 0.0
    rng = torch.Generator(device=scores.device).manual_seed(seed)
    idx_i = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    idx_j = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    # exclude self-pairs and near-ties
    keep = (idx_i != idx_j) & ((true_logret[idx_i] - true_logret[idx_j]).abs() >= dead_zone)
    if keep.sum() < 1:
        return torch.tensor(0.0, device=scores.device)
    i, j = idx_i[keep], idx_j[keep]
    target = (true_logret[i] > true_logret[j]).float()
    logit = (scores[i] - scores[j]) / tau
    loss = F.binary_cross_entropy_with_logits(logit, target)
    return loss


def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta)


# ============================================================================
# Return / direction heads (PT-03)
# ============================================================================

class LinearReturnHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = "huber"

    def forward(self, h):
        return self.fc(h).squeeze(-1)          # [B]


class MlpReturnHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.loss = "huber"

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LinearDirectionHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h).squeeze(-1))   # direction_prob in [0,1]


class MlpDirectionHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        return torch.sigmoid(self.net(h).squeeze(-1))


# ============================================================================
# Rank heads (PT-03 / PT-04)
# ============================================================================

class LinearRankHead(nn.Module):
    """Rank score head; loss is pairwise logistic (fallback) or soft-Spearman."""
    def __init__(self, dim=256, loss="soft_spearman"):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = loss

    def forward(self, h):
        return self.fc(h).squeeze(-1)          # rank_score (any monotone scale)


class MlpRankHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0, loss="soft_spearman"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.loss = loss

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ============================================================================
# Cross-sectional heads (PT-04)
# ============================================================================

class IndependentMLP(nn.Module):
    """score_i = g(h_i): per-stock capacity control (no cross-stock context)."""
    def __init__(self, dim=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class DeepSetsHead(nn.Module):
    """score_i = psi([h_i, mean_pool_phi(h)]): permutation-equivariant context.

    ``phi`` pools the same-date hidden states; ``psi`` conditions each stock on
    the pooled context.  Invariant to input permutation (see test_10).
    """
    def __init__(self, dim=256, latent=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, latent), nn.SiLU())
        self.psi = nn.Sequential(
            nn.Linear(dim + latent, latent), nn.SiLU(), nn.Linear(latent, 1))

    def forward(self, h):
        pooled = self.phi(h).mean(dim=0, keepdim=True)   # [1, latent]
        ctx = pooled.expand(h.shape[0], -1)
        return self.psi(torch.cat([h, ctx], dim=-1)).squeeze(-1)


class ISABSetTransformer(nn.Module):
    """Inducing-point Set Transformer (Lee et al., ICML 2019), pilot config.

    One ISAB block: multihead attention from stocks to inducing points and back,
    then a per-stock score head.  Uses genuine (inducing-point) cross-stock
    attention, NOT independent chunks.  permutation equivariant.
    """
    def __init__(self, dim=256, latent=64, n_inducing=32, heads=4):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, latent) * 0.02)
        self.to_latent = nn.Linear(dim, latent)
        self.attn = nn.MultiheadAttention(latent, heads, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, 1))

    def forward(self, h):
        # The batch IS one date's set: B stocks each contribute one [latent] token.
        x = self.to_latent(h)                      # [B, latent]
        x = x.unsqueeze(0)                         # [1, B, latent] (one set)
        # stocks attend to inducing points (pooling), then stocks attend back
        pooled, _ = self.attn(self.inducing, x, x)  # [1, n_inducing, latent]
        back, _ = self.attn(x, pooled, pooled)      # [1, B, latent]
        return self.score(back.squeeze(0)).squeeze(-1)


# ============================================================================
# Controls (PT-03 C0/C1/C2)
# ============================================================================

class C1FeatureHead(nn.Module):
    """Feature-only control head: last-return, 5/20-day momentum, 20-day vol/vol.

    Input features are point-in-time (visible at position p-1).  Same capacity
    as the hidden MLP probe (dim -> 64 -> 1).
    """
    def __init__(self, n_features=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class C2PosteriorHead(nn.Module):
    """Posterior-statistics control: mean/std/entropy/P(up) same-capacity head."""
    def __init__(self, n_stats=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, stats):
        return self.net(stats).squeeze(-1)

# ===== joint_decoder =====
EPS = 1e-12


class DecodeTable:
    """Precomputed ``[V_c, V_f, feature_dim]`` tokenizer decode table.

    ``decode_all`` is a batched decoder, not a lookup table, so we enumerate the
    joint support once at construction.
    """

    def __init__(self, tokenizer, device="cpu"):
        self.v_c = int(tokenizer.vocab_coarse)
        self.v_f = int(tokenizer.bsq_fine.vocab_size)
        if self.v_c * self.v_f > 2_000_000:
            raise ValueError(f"joint support too large: {self.v_c}x{self.v_f}")
        self.device = torch.device(device)

        cc = torch.arange(self.v_c, dtype=torch.long).repeat_interleave(self.v_f)
        ff = torch.arange(self.v_f, dtype=torch.long).repeat(self.v_c)
        idx = torch.stack([cc, ff], dim=-1).unsqueeze(0)  # [1, V_c*V_f, 2]
        # Build on the tokenizer's own device, then move the fixed lookup.
        tok_device = next(tokenizer.parameters()).device
        with torch.no_grad():
            decoded = tokenizer.decode_all(idx.to(tok_device)).squeeze(0)
        decoded = decoded.to(self.device)
        # The decode table is a fixed lookup; never let the tokenizer's weights
        # drag it into autograd graphs (breaks .numpy() in downstream code).
        self.table = decoded.detach().view(self.v_c, self.v_f, decoded.shape[-1])
        # Feature 0 is normalized log_ret.
        self.norm_logret = self.table[..., 0].detach()  # [V_c, V_f]

        # Shared sort permutation over flattened [V_c*V_f] normalized log-returns.
        self.flat_norm = self.norm_logret.reshape(-1)
        self.order = torch.argsort(self.flat_norm, stable=True)

    def raw_logret(self, p_mean0, p_std0):
        """Raw-space return table ``[..., V_c, V_f]`` for per-sample scalars.

        ``p_mean0`` / ``p_std0`` broadcast from the right-most dims.
        """
        return self.norm_logret * p_std0 + p_mean0


class JointStats:
    """Mutable container for per-row joint posterior statistics (tensors [K, ...])."""

    def __init__(self, k, device):
        self.device = torch.device(device)
        self.k = k
        # MAP (J1)
        self.map_c = None
        self.map_f = None
        self.map_return = None
        self.map_joint_prob = None
        # Posterior moments
        self.mean = None
        self.median = None
        self.std = None
        self.q10 = None
        self.q90 = None
        self.p_up = None
        # Entropies
        self.coarse_entropy = None
        self.cond_fine_entropy = None
        self.joint_entropy = None
        # Greedy baseline (J0)
        self.greedy_c = None
        self.greedy_f = None
        self.greedy_return = None
        # NLLs (need true coarse/fine ids)
        self.full_joint_nll = None
        self.ordinary_joint_nll = None
        # Marginal diagnostic
        self.special_mass = None

    def as_dict(self):
        out = {}
        for k, v in vars(self).items():
            if k in ("device", "k"):
                continue
            if v is not None:
                out[k] = v
        return out


@torch.no_grad()
def decode_joint(
    model,
    decode_table,
    hidden,
    p_mean0,
    p_std0,
    true_coarse_ids=None,
    true_fine_ids=None,
    true_logret=None,
    t_c=1.0,
    t_f=1.0,
    chunk=512,
):
    """Compute exact joint posterior statistics for ``hidden [K, dim]``.

    Args:
        model: KronosPreview exposing ``coarse_logits_from_hidden`` and
            ``fine_logits_for_coarse``.
        decode_table: precomputed DecodeTable.
        hidden: [K, dim] final-norm hidden states.
        p_mean0 / p_std0: [K] per-sample raw-space normalization scalars for
            feature 0 (log_ret).  ``p_std0`` must be finite and > 0.
        true_coarse_ids / true_fine_ids: [K] optional true joint token for NLLs.
        t_c / t_f: coarse / conditional-fine softmax temperatures.
        chunk: number of rows to process per fine-logits expansion pass.

    Returns:
        (JointStats, quality_flags).  ``quality_flags`` is [K] bool: True where
        p_std0 is finite and positive (valid for raw-space recovery).
    """
    device = hidden.device
    k = hidden.shape[0]
    stats = JointStats(k, device)

    p_mean0 = torch.as_tensor(p_mean0, dtype=torch.float32, device=device).reshape(-1)
    p_std0 = torch.as_tensor(p_std0, dtype=torch.float32, device=device).reshape(-1)
    if p_mean0.numel() == 1:
        p_mean0 = p_mean0.expand(k)
    if p_std0.numel() == 1:
        p_std0 = p_std0.expand(k)
    quality = torch.isfinite(p_mean0) & torch.isfinite(p_std0) & (p_std0 > 0)

    v_c = decode_table.v_c
    v_f = decode_table.v_f
    dim = hidden.shape[-1]

    # ---- coarse posterior -------------------------------------------------
    coarse_logits = model.coarse_logits_from_hidden(hidden)  # [K, V_c+2]
    p_full = F.softmax(coarse_logits / t_c, dim=-1)          # [K, V_c+2]
    special_mass = p_full[:, v_c:].sum(dim=-1)               # [K]
    # q(c) = P(c | ordinary) = p_full(c) / (1 - special_mass).  When the model
    # is (near-)certain about BOS/EOS (special_mass ~ 1), the float division
    # underflows, so q is renormalized to a proper distribution over ordinary
    # codes.  full_joint_nll still uses p_full (spec §4.1); ordinary_nll uses q.
    q = p_full[:, :v_c] / (1.0 - special_mass).clamp_min(EPS).unsqueeze(-1)  # [K, V_c]
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)

    # ---- fine posterior for every candidate coarse ------------------------
    p_f_given_c = torch.empty(k, v_c, v_f, dtype=hidden.dtype, device=device)
    for start in range(0, k, chunk):
        stop = min(start + chunk, k)
        hid_exp = hidden[start:stop].unsqueeze(1).expand(stop - start, v_c, dim)
        hid_exp = hid_exp.reshape(-1, dim)
        coarse_ids = torch.arange(v_c, device=device).repeat(stop - start)
        fl = model.fine_logits_for_coarse(hid_exp, coarse_ids)
        p_f_given_c[start:stop] = F.softmax(
            fl / t_f, dim=-1).view(stop - start, v_c, v_f)

    joint = q.unsqueeze(-1) * p_f_given_c                    # [K, V_c, V_f]

    # ---- raw-space returns ------------------------------------------------
    flat_norm = decode_table.flat_norm.to(device)            # [M]
    order = decode_table.order.to(device)                    # [M]
    M = v_c * v_f
    raw_flat = flat_norm * p_std0.unsqueeze(-1) + p_mean0.unsqueeze(-1)  # [K, M]

    # ---- statistics that don't need the flattened sort ---------------------
    stats.special_mass = special_mass
    stats.coarse_entropy = -(q * (q + EPS).log()).sum(-1)
    cond_fine = -(joint * (p_f_given_c + EPS).log()).sum(dim=(-1, -2))
    stats.cond_fine_entropy = cond_fine
    stats.joint_entropy = -(joint * (joint + EPS).log()).sum(dim=(-1, -2))

    # joint MAP (J1): argmax over full (c, f) grid of q(c)*p(f|c)
    joint_flat = joint.reshape(k, M)
    map_idx = joint_flat.argmax(-1)
    stats.map_c = map_idx // v_f
    stats.map_f = map_idx % v_f
    stats.map_return = raw_flat[torch.arange(k, device=device), map_idx]
    stats.map_joint_prob = joint_flat[torch.arange(k, device=device), map_idx]

    # posterior mean / variance (J2)
    stats.mean = (joint_flat * raw_flat).sum(-1)
    mean_sq = (joint_flat * raw_flat.square()).sum(-1)
    var = (mean_sq - stats.mean.square()).clamp_min(0.0)
    stats.std = var.sqrt()

    # P(raw_logret > 0) in RAW space (J4).  With p_std>0 the threshold on the
    # normalized table is norm > -p_mean/p_std.
    with torch.no_grad():
        thr = (-p_mean0 / p_std0.clamp_min(EPS))               # [K]
        norm_sorted = flat_norm[order]                          # [M] shared
        probs_sorted = joint_flat[:, order]                     # [K, M]
        idx_up = torch.searchsorted(
            norm_sorted.unsqueeze(0).expand(k, M).contiguous(),
            thr.unsqueeze(-1), right=True,
        )[:, 0]                                                # [K]
        up_mask = torch.arange(M, device=device).unsqueeze(0) >= idx_up.unsqueeze(-1)
        stats.p_up = (probs_sorted * up_mask).sum(-1)

        # median / quantiles: left quantile where CDF first reaches q.
        # torch.searchsorted needs values shaped [B, ...] when boundaries are 2D.
        cdf = torch.cumsum(probs_sorted, dim=-1)               # [K, M]
        rows_idx = torch.arange(k, device=device)
        for qval, attr in ((0.10, "q10"), (0.50, "median"), (0.90, "q90")):
            qi = torch.searchsorted(
                cdf, torch.full((k, 1), qval, device=device), right=False
            )[:, 0].clamp_max(M - 1)
            stats.__dict__[attr] = raw_flat[rows_idx, order[qi]]

        # CRPS of the discrete posterior vs the true raw return (if provided).
        # CRPS = E|X-y| - 0.5 E|X-X'|; the E|X-X'| term uses the O(M) sorted form
        # term_i = x_i*(2*F_i - P_total) - (2*PX_i - PX_total).
        if true_logret is not None:
            y = torch.as_tensor(true_logret, dtype=torch.float32,
                                device=device).reshape(-1)
            raw_sorted = raw_flat[rows_idx[:, None], order[None, :]]  # [K, M] ascending
            P = torch.cumsum(probs_sorted, dim=-1)
            PX = torch.cumsum(probs_sorted * raw_sorted, dim=-1)
            total_px = PX[:, -1:]
            term = raw_sorted * (2.0 * P - 1.0) - (2.0 * PX - total_px)
            e_abs = (probs_sorted * (raw_sorted - y.unsqueeze(-1)).abs()).sum(-1)
            e_pair = (probs_sorted * term).sum(-1)
            stats.crps = e_abs - 0.5 * e_pair

    # greedy baseline (J0): argmax coarse over ordinary codes, then conditional
    # fine argmax, matching the legacy forward_selected decode path.
    stats.greedy_c = p_full[:, :v_c].argmax(-1)
    g_idx = torch.arange(k, device=device), stats.greedy_c
    stats.greedy_f = p_f_given_c[g_idx].argmax(-1)
    stats.greedy_return = raw_flat[
        torch.arange(k, device=device), stats.greedy_c * v_f + stats.greedy_f]

    # ---- NLLs (require true ids) -------------------------------------------
    if true_coarse_ids is not None and true_fine_ids is not None:
        tc = torch.as_tensor(true_coarse_ids, dtype=torch.long, device=device).reshape(-1)
        tf = torch.as_tensor(true_fine_ids, dtype=torch.long, device=device).reshape(-1)
        valid = (tc >= 0) & (tc < v_c) & (tf >= 0) & (tf < v_f)
        full_c_prob = p_full[torch.arange(k, device=device), tc]
        fine_prob = p_f_given_c[torch.arange(k, device=device), tc, tf]
        q_c_prob = q[torch.arange(k, device=device), tc]
        full_joint_nll = torch.full((k,), float("nan"), device=device)
        ordinary_nll = torch.full((k,), float("nan"), device=device)
        full_joint_nll[valid] = -torch.log((full_c_prob * fine_prob).clamp_min(EPS))[valid]
        ordinary_nll[valid] = -torch.log((q_c_prob * fine_prob).clamp_min(EPS))[valid]
        stats.full_joint_nll = full_joint_nll
        stats.ordinary_joint_nll = ordinary_nll

    return stats, quality


def greedy_logits_from_hidden(model, hidden):
    """Legacy greedy decode logits for parity checks (returns coarse, fine).

    Mirrors ``forward_selected``: fine conditioned on argmax coarse over ordinary
    codes.  Only for the J0 compatibility baseline / guardrail checks.
    """
    coarse_logits = model.coarse_logits_from_hidden(hidden)
    coarse_pred = coarse_logits[:, : model._vocab_l1].argmax(dim=-1)
    fine_logits = model.fine_logits_for_coarse(hidden, coarse_pred)
    return coarse_logits, fine_logits

# ===== compare_posttrain =====

from bootstrap_utils import (  # noqa: E402
    circular_moving_block_bootstrap as _root_circular_moving_block_bootstrap,
)


def daily_metric_from_rows(rows, metric, dense_min=None):
    """Compute a per-date metric over per-row dicts.

    ``rows``: list of dicts with ``date_key`` and, depending on metric:
        - ``rank_ic``: needs ``rank_score`` and ``true_logret``
        - ``da``: needs ``direction_prob`` and ``true_logret``
        - ``mape``: needs ``pred_logret`` and ``true_logret``
        - ``rank_ic_mae``: needs ``pred_logret`` and ``true_logret`` (MAE on logret)
    Returns {date_key: value} restricted to dates with >= dense_min finite rows.
    """
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date_key"], []).append(r)
    out = {}
    for d, rs in by_date.items():
        rs = [r for r in rs if _finite(r)]
        if dense_min is not None and len(rs) < dense_min:
            continue
        if len(rs) < 2:
            continue
        if metric == "rank_ic":
            score = np.asarray([r["rank_score"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            if np.all(score == score[0]):
                out[d] = 0.0
            else:
                out[d] = spearmanr(score, tru)[0]
                if not np.isfinite(out[d]):
                    out[d] = 0.0
        elif metric == "da":
            prob = np.asarray([r["direction_prob"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean((prob > 0.5) == (tru > 0)))
        elif metric == "mape":
            pred = np.asarray([r["pred_logret"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean(np.abs(pred - tru)))
        elif metric == "mae":
            pred = np.asarray([r["pred_logret"] for r in rs], dtype=float)
            tru = np.asarray([r["true_logret"] for r in rs], dtype=float)
            out[d] = float(np.mean(np.abs(pred - tru)))
        else:
            raise ValueError(f"unknown metric {metric}")
    return out


def _finite(r):
    if "true_logret" in r and not np.isfinite(r.get("true_logret")):
        return False
    for key in ("rank_score", "direction_prob", "pred_logret"):
        if key in r and r.get(key) is not None and not np.isfinite(r[key]):
            return False
    return True


def shared_universe(candidate_rows, reference_rows):
    """Intersect candidate/reference to common date x stock_uid set."""
    c = {(r["date_key"], r["stock_uid"]) for r in candidate_rows}
    r_ = {(r["date_key"], r["stock_uid"]) for r in reference_rows}
    common = c & r_
    c_by_key = {(r["date_key"], r["stock_uid"]): r for r in candidate_rows}
    r_by_key = {(r["date_key"], r["stock_uid"]): r for r in reference_rows}
    return [c_by_key[k] for k in common], [r_by_key[k] for k in common]


def circular_moving_block_bootstrap(
    deltas,
    block_length=5,
    n_replicates=10_000,
    seed=42,
):
    """Circular moving-block bootstrap over a contiguous date delta series.

    ``deltas`` must be ordered by date.  Samples contiguous circular blocks
    (start uniformly random, advance with wraparound).  Returns the replicate
    mean distribution.
    """
    # Canonical implementation lives in the root-level bootstrap_utils module
    # so every experiment package shares the exact same sampler.
    return _root_circular_moving_block_bootstrap(
        deltas,
        block_length=block_length,
        n_replicates=n_replicates,
        seed=seed,
    )


def paired_bootstrap_ci(
    candidate_rows,
    reference_rows,
    metric,
    dense_min=None,
    block_lengths=(5, 10, 20),
    n_replicates=10_000,
    seed=42,
    order_by_date=True,
):
    """Full paired comparison with moving-block CIs at several block lengths.

    Returns dict with point estimate (mean daily delta), per-block-length CI,
    directional significance, and block robustness.
    """
    cand, ref = shared_universe(candidate_rows, reference_rows)
    c_daily = daily_metric_from_rows(cand, metric, dense_min=dense_min)
    r_daily = daily_metric_from_rows(ref, metric, dense_min=dense_min)
    common_dates = sorted(set(c_daily) & set(r_daily))
    if len(common_dates) < 2:
        return {
            "metric": metric, "n_dates": len(common_dates),
            "point": None, "candidate_ci": None, "reference_ci": None,
            "error": "fewer than 2 common dense dates",
        }
    c_series = np.asarray([c_daily[d] for d in common_dates], dtype=float)
    r_series = np.asarray([r_daily[d] for d in common_dates], dtype=float)
    deltas = c_series - r_series
    point = float(deltas.mean())
    minimize = metric in ("mape", "mae")
    cis = {}
    for L in block_lengths:
        means = circular_moving_block_bootstrap(
            deltas, block_length=L, n_replicates=n_replicates, seed=seed)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        if minimize:
            signif = hi < 0.0
        else:
            signif = lo > 0.0
        cis[str(L)] = {
            "block_length": int(L), "ci_lower": lo, "ci_upper": hi,
            "significant_directional": bool(signif),
            "ci_excludes_zero": bool(lo <= 0.0 <= hi) is False,
        }
    block_robust = all(v["significant_directional"] for v in cis.values())
    return {
        "metric": metric,
        "n_dates": len(common_dates),
        "point": point,
        "minimize": minimize,
        "candidate_mean": float(c_series.mean()),
        "reference_mean": float(r_series.mean()),
        "block_cis": cis,
        "block_robust": block_robust,
        "offset_scope": "development_0_299_or_300_399_as_declared",
    }


def bootstrap_for_series(deltas, block_lengths=(5, 10, 20),
                         n_replicates=10_000, seed=42):
    """Convenience: CI for a raw per-date delta series (already computed)."""
    out = {}
    for L in block_lengths:
        means = circular_moving_block_bootstrap(
            deltas, block_length=L, n_replicates=n_replicates, seed=seed)
        out[str(L)] = {
            "block_length": int(L),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
        }
    return out

# ===== metrics =====
def daily_rank_ic(score, true, dates, dense_min):
    """Per-date RankIC / DA / MAE with a cross-section density threshold."""
    uniq, inv = np.unique(dates, return_inverse=True)
    n = len(uniq)
    ic = np.full(n, np.nan)
    da = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = inv == i
        c = int(m.sum())
        cnt[i] = c
        if c < dense_min:
            continue
        s = score[m]
        t = true[m]
        if c >= 2 and not np.all(s == s[0]):
            ic[i] = spearmanr(s, t)[0]
        else:
            ic[i] = 0.0
        da[i] = np.mean((np.sign(s) > 0) == (t > 0))
        mae[i] = np.mean(np.abs(s - t))
    dense = cnt >= dense_min
    return ic, da, mae, cnt, dense


def _per_date(rec, dense_threshold):
    """Group row indices by date; returns (unique_dates, inv)."""
    uniq, inv = np.unique(rec["date_key"], return_inverse=True)
    return uniq, inv


def arm_metrics(rec, score_field, dense_threshold, proper=None):
    """Aggregate + per-date metrics for a continuous-score arm.

    ``rec``: dict of arrays. ``score_field``: e.g. greedy_return / post_median.
    Returns dict with avg_daily_rank_ic, avg_da_per_date, avg_mape, avg_mae,
    n_dense_dates, per_date series.
    """
    score = rec[score_field]
    true = rec["true_logret"]
    quality = rec.get("quality", np.ones(len(score), dtype=bool))
    valid = np.isfinite(score) & np.isfinite(true) & quality
    uniq, inv = _per_date(rec, dense_threshold)
    n_dates = len(uniq)
    da = np.full(n_dates, np.nan)
    ic = np.full(n_dates, np.nan)
    mae_arr = np.full(n_dates, np.nan)
    mape = np.full(n_dates, np.nan)
    cnt = np.zeros(n_dates, dtype=np.int64)
    for i in range(n_dates):
        m = (inv == i) & valid
        c = int(m.sum())
        cnt[i] = c
        if c < dense_threshold:
            continue
        s = score[m]
        t = true[m]
        da[i] = float(np.mean((np.sign(s) > 0) == (t > 0)))
        if c >= 2 and not np.all(s == s[0]):
            ic[i] = float(spearmanr(s, t)[0])
        else:
            ic[i] = 0.0
        mae_arr[i] = float(np.mean(np.abs(s - t)))
        mape[i] = float(np.mean(np.abs(np.exp(s - t) - 1.0)))
    dense = cnt >= dense_threshold
    out = {
        "score_field": score_field,
        "n_dense_dates": int(dense.sum()),
        "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
        "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        "avg_mape": float(np.nanmean(mape[dense])) if dense.any() else None,
        "avg_mae": float(np.nanmean(mae_arr[dense])) if dense.any() else None,
        "per_date": {
            str(uniq[i])[:10]: {"da": da[i], "rank_ic": ic[i], "mae": mae_arr[i], "n": int(cnt[i])}
            for i in range(n_dates)
        },
    }
    if proper is not None:
        out["proper"] = proper
    return out


def _contrast(base_ic, arm_ic, base_mae, arm_mae, n_replicates=10000):
    """Paired circular moving-block bootstrap contrast (L=5/10/20) vs a base arm."""
    common = np.isfinite(base_ic) & np.isfinite(arm_ic)
    d_ic = arm_ic[common] - base_ic[common]
    d_mae = arm_mae[common] - base_mae[common]
    cis = {}
    for L in (5, 10, 20):
        b = circular_moving_block_bootstrap(d_ic, block_length=L,
                                            n_replicates=n_replicates, seed=42)
        lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        cis[str(L)] = {"ci_lower": lo, "ci_upper": hi, "significant": lo > 0.0}
    return {
        "rank_ic_delta_vs_J0": float(np.nanmean(d_ic)),
        "mae_delta_vs_J0": float(np.nanmean(d_mae)),
        "block_cis": cis,
        "block_robust": all(v["significant"] for v in cis.values()),
        "n_dates": int(common.sum()),
    }

# ===== cache_hidden =====
def prepared_cache_key(sel, tokenizer_path, split_sha):
    """Hash key for the prepared-input cache (uid + upstream + data + split)."""
    ckpt = upstream_checkpoint_path(sel)
    payload = {
        "schema": 2,
        "upstream_sha256": file_sha256(ckpt),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "data_uid_fingerprint": None,  # set by caller
        "split_sha256": split_sha,
        "offsets": list(VALIDATION_OFFSETS),
        "n_days": 1,
    }
    return payload


def build_and_extract_hidden(
    *,
    device,
    batch_size=4,
    seed=42,
    roots=None,
):
    """Build prepared inputs and extract hidden states; returns cache paths.

    Returns (prepared_path, hidden_path, n_rows, dense_threshold).
    """
    import numpy as np
    from eval_helpers import _bucket_by_length  # noqa: F401

    sel = load_reviewed_selection()
    ckpt_path = upstream_checkpoint_path(sel)
    tok_path = Path(sel["upstream"]["tokenizer"])
    tokenizer = load_tokenizer(str(tok_path), device)

    # ---- stocks + split ----
    stocks = load_stocks_uid(DataConfig.data_dir)
    _, _, test_stocks = split_stocks(stocks)  # train/val/test by cutoff date
    attach_close_prices_uid(test_stocks)
    prepped = prepare_stocks_uid(test_stocks, tokenizer, device)
    batches = build_prepared_batches(
        prepped, list(VALIDATION_OFFSETS), n_days=1, batch_size=batch_size)
    n_rows = sum(len(b["stock_uids"]) for b in batches)
    print(f"[hidden] prepared {len(batches)} batches, {n_rows} rows")

    # dense threshold from the reference universe (one date's max cross-section)
    by_date = {}
    for b in batches:
        for d, u in zip(b["date_keys"], b["stock_uids"]):
            by_date.setdefault(d, set()).add(u)
    max_cs = max(len(v) for v in by_date.values())
    dense_threshold = max(5, int(np.ceil(0.8 * max_cs)))
    print(f"[hidden] max cross-section {max_cs}, dense_threshold {dense_threshold}")

    # ---- model ----
    model = load_gpt(str(ckpt_path), device, tokenizer=tokenizer)
    model.eval()

    # ---- extract hidden ----
    dim = model.head_coarse.in_features
    hidden_list = []
    meta = {k: [] for k in ("stock_uid", "date_key", "symbol", "offset", "position",
                            "p_mean0", "p_std0", "true_logret", "true_coarse_id",
                            "true_fine_id", "base_close", "true_close")}
    total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for bi, batch in enumerate(batches):
            inp = batch["input_ids"].to(device)
            tids = batch["time_ids"].to(device)
            va = batch["va_values"].to(device)
            pos_ids = torch.arange(inp.shape[1], device=device).unsqueeze(0).expand(inp.shape[0], -1)
            rows = batch["selection_rows"].to(device)
            poss = batch["selection_positions"].to(device)
            h = model.encode_selected(inp, tids, pos_ids, rows, poss, va_values=va)
            hidden_list.append(h.float().cpu())
            cnt = h.shape[0]
            total += cnt
            meta["stock_uid"].extend(batch["stock_uids"])
            meta["date_key"].extend(batch["date_keys"])
            meta["symbol"].extend(batch["symbols"])
            meta["offset"].extend(batch["window_starts"].tolist())
            meta["position"].extend(batch["selection_positions"].tolist())
            meta["p_mean0"].extend(batch["p_means"].tolist())
            meta["p_std0"].extend(batch["p_stds"].tolist())
            meta["true_logret"].extend(batch["true_logrets"].tolist())
            meta["true_coarse_id"].extend(batch["true_coarse_ids"].tolist())
            meta["true_fine_id"].extend(batch["true_fine_ids"].tolist())
            meta["base_close"].extend(batch["base_closes"].tolist())
            meta["true_close"].extend(batch["true_closes"].tolist())
            if (bi + 1) % 200 == 0:
                print(f"[hidden] batch {bi+1}/{len(batches)} rows {total}")

    hidden = torch.cat(hidden_list, dim=0)  # [N, dim]
    print(f"[hidden] total hidden rows {hidden.shape}")

    # ---- write caches ----
    paths = artifact_paths(roots=roots)
    paths["prepared"].parent.mkdir(parents=True, exist_ok=True)
    prep_path = paths["prepared"]
    torch.save({"batches": batches, "dense_threshold": int(dense_threshold),
                "n_rows": int(n_rows)}, prep_path)

    hidden_path = paths["hidden"]
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        hidden_path,
        hidden=hidden.numpy(),
        stock_uid=np.array(meta["stock_uid"]),
        date_key=np.array(meta["date_key"]),
        symbol=np.array(meta["symbol"]),
        offset=np.array(meta["offset"], dtype=np.int16),
        position=np.array(meta["position"], dtype=np.int32),
        p_mean0=np.array(meta["p_mean0"], dtype=np.float64),
        p_std0=np.array(meta["p_std0"], dtype=np.float64),
        true_logret=np.array(meta["true_logret"], dtype=np.float64),
        true_coarse_id=np.array(meta["true_coarse_id"], dtype=np.int16),
        true_fine_id=np.array(meta["true_fine_id"], dtype=np.int16),
        base_close=np.array(meta["base_close"], dtype=np.float64),
        true_close=np.array(meta["true_close"], dtype=np.float64),
        dense_threshold=np.array([dense_threshold]),
    )
    print(f"[hidden] wrote {hidden_path} ({hidden_path.stat().st_size/1e9:.2f} GB)")
    return prep_path, hidden_path, int(n_rows), int(dense_threshold)


# ===== training utilities (shared by b-frozen-heads + c-cross-sectional) =====
FOLDS = {
    "R0": (None, "2020-02-01", "2020-02-01", "2021-02-01"),
    "R1": (None, "2021-02-01", "2021-02-01", "2022-02-01"),
    "R2": (None, "2022-02-01", "2022-02-01", "2023-02-01"),
}


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def in_interval(d, start, stop):
    if start is not None and d < start:
        return False
    if stop is not None and d >= stop:
        return False
    return True


def fold_split(rows, fold):
    start, fstop, vstart, vstop = FOLDS[fold]
    fit_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], start, fstop)]
    val_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], vstart, vstop)]
    return np.asarray(fit_idx), np.asarray(val_idx)


def final_fit_split(rows):
    idx = [i for i, d in enumerate(rows["date_key"])
           if str(d)[:10] < FINAL_FIT_STOP]
    return np.asarray(idx)


def bce_direction(head_out, y):
    return nn.functional.binary_cross_entropy(head_out, y, reduction="mean")


def compute_head_loss(head, h, feats, y, loss_kind, tau=1.0, dead_zone=None):
    if isinstance(head, (C1FeatureHead,)):
        out = head(feats)
    elif isinstance(head, (C2PosteriorHead,)):
        out = head(h)
    else:
        out = head(h)
    if loss_kind in ("huber", "mse", "reg"):
        return huber_loss(out, y, delta=1.0)
    if loss_kind in ("bce", "direction"):
        return bce_direction(out, y)
    if loss_kind in ("soft_spearman",):
        ranks = np.argsort(np.argsort(y.detach().cpu().numpy())).astype(np.float32)
        t = torch.from_numpy(ranks).to(y.device)
        return soft_spearman_loss(out, t, tau=tau)
    if loss_kind in ("pairwise",):
        return pairwise_logistic_loss(out, y, tau=tau, dead_zone=dead_zone)
    raise ValueError(loss_kind)


def train_one(head, rows, fit_idx, val_idx, loss_kind, *, lr, epochs, batch_size,
              seed=42, tau=1.0, dead_zone=None, val_every=1, max_grad_norm=1.0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    device = next(head.parameters()).device
    H = torch.from_numpy(rows["hidden"]).to(device)
    Y = torch.from_numpy(rows["true_logret"].astype(np.float32)).to(device)
    if loss_kind in ("bce", "direction"):
        Y = (Y > 0.0).float()
    feats = torch.from_numpy(rows["c1_feats"].astype(np.float32)).to(device) \
        if "c1_feats" in rows else None
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.RandomState(seed)
    n_fit = len(fit_idx)
    best_val = None
    for ep in range(epochs):
        perm = rng.permutation(n_fit)
        head.train()
        ep_loss = 0.0
        steps = 0
        for s in range(0, n_fit, batch_size):
            ids = fit_idx[perm[s:s + batch_size]]
            if len(ids) < 2:
                continue
            hb = H[ids]
            yb = Y[ids]
            fb = feats[ids] if feats is not None else None
            opt.zero_grad()
            loss = compute_head_loss(head, hb, fb, yb, loss_kind, tau=tau,
                                     dead_zone=dead_zone)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            opt.step()
            ep_loss += loss.item()
            steps += 1
        ep_loss /= max(1, steps)
        history["train_loss"].append(ep_loss)
        if (ep + 1) % val_every == 0:
            head.eval()
            with torch.no_grad():
                vloss = 0.0
                vsteps = 0
                for s in range(0, len(val_idx), batch_size):
                    ids = val_idx[s:s + batch_size]
                    if len(ids) < 2:
                        continue
                    vloss += compute_head_loss(
                        head, H[ids], feats[ids] if feats is not None else None,
                        Y[ids], loss_kind, tau=tau, dead_zone=dead_zone).item()
                    vsteps += 1
                vloss /= max(1, vsteps)
            history["val_loss"].append(vloss)
            if best_val is None or vloss < best_val:
                best_val = vloss
    return head, history


def _build_cross_section_recs(rows, idx):
    recs = []
    for i in idx:
        recs.append({"date_key": str(rows["date_key"][i])[:10],
                     "stock_uid": str(rows["stock_uid"][i]),
                     "hidden": rows["hidden"][i],
                     "true_logret": float(rows["true_logret"][i]),
                     "raw_logret": float(rows["true_logret"][i])})
    return recs


def _eval_rank_loss(head, rows, val_idx, loss_kind, tau=1.0, dead_zone=None):
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, val_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=0, shuffle_dates=False)
    head.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            total += compute_head_loss(head, hb, None, yb, loss_kind, tau=tau,
                                       dead_zone=dead_zone).item()
            steps += 1
    return total / max(1, steps)


def train_rank_per_date(head, rows, fit_idx, val_idx, loss_kind, *, lr, epochs,
                        seed=42, tau=1.0, dead_zone=None):
    """Rank heads train one date cross-section per step (soft-Spearman/pairwise)."""
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, fit_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=seed, shuffle_dates=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        head.train()
        total = 0.0
        steps = 0
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            opt.zero_grad()
            loss = compute_head_loss(head, hb, None, yb, loss_kind, tau=tau,
                                     dead_zone=dead_zone)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        history["train_loss"].append(total / max(1, steps))
        history["val_loss"].append(_eval_rank_loss(
            head, rows, val_idx, loss_kind, tau=tau, dead_zone=dead_zone))
    return head, history


def select_recipe(head_factory, rows, loss_kind, hyper_grid, seed=42, **loss_kw):
    """R0-R2 rolling selection of a fixed-step recipe (lr/dropout/epochs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for hp in hyper_grid:
        val_scores = []
        for fold in ("R0", "R1", "R2"):
            fit_idx, val_idx = fold_split(rows, fold)
            if len(fit_idx) < 50 or len(val_idx) < 20:
                val_scores.append(None)
                continue
            head = head_factory(**hp["head"]).to(device)
            if loss_kind in ("soft_spearman", "pairwise"):
                h, hist = train_rank_per_date(
                    head, rows, fit_idx, val_idx, loss_kind, lr=hp["lr"],
                    epochs=hp["epochs"], seed=seed, **loss_kw)
            else:
                h, hist = train_one(
                    head, rows, fit_idx, val_idx, loss_kind, lr=hp["lr"],
                    epochs=hp["epochs"], batch_size=hp["batch_size"], seed=seed,
                    **loss_kw)
            val_scores.append(hist["val_loss"][-1] if hist.get("val_loss") else None)
        valid = [v for v in val_scores if v is not None]
        results.append({"hp": hp, "fold_val_loss": dict(zip(("R0", "R1", "R2"),
                                                            val_scores)),
                        "mean_val_loss": float(np.mean(valid)) if valid else None})
    best = min([r for r in results if r["mean_val_loss"] is not None],
               key=lambda r: r["mean_val_loss"])
    return best


def final_fit(head_factory, rows, recipe, loss_kind, seed=42, **loss_kw):
    """Final head fit on all <2023-02-01 with the locked recipe (no early stop)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_idx = final_fit_split(rows)
    hp = recipe["hp"]
    head = head_factory(**hp["head"]).to(device)
    if loss_kind in ("soft_spearman", "pairwise"):
        h, hist = train_rank_per_date(
            head, rows, fit_idx, fit_idx, loss_kind, lr=hp["lr"],
            epochs=hp["epochs"], seed=seed, **loss_kw)
    else:
        h, hist = train_one(
            head, rows, fit_idx, fit_idx, loss_kind, lr=hp["lr"],
            epochs=hp["epochs"], batch_size=hp["batch_size"], seed=seed,
            **loss_kw)
    return head


# backward alias
recipe_select = select_recipe


def build_hidden(argv=None):
    ap = argparse.ArgumentParser(description="Build the PostTrain frozen hidden cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_cuda", action="store_true")
    args = ap.parse_args(argv)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal hidden cache")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    roots = resolve_roots(seed=args.seed)
    build_and_extract_hidden(device=device, batch_size=args.batch_size,
                             seed=args.seed, roots=roots)

# ===== build_training_cache =====
FINAL_FIT_STOP = "2023-02-01"
CALIB_START, CALIB_STOP = "2023-02-01", "2024-02-01"
MIN_POS = NormConfig.min_lookback + 1  # need history before p


def c1_features(feat, p, window=20):
    """Point-in-time C1 features at position p (uses rows < p only)."""
    if p < MIN_POS:
        return None
    lr = feat[:, 0]
    last = lr[p - 1]
    mom5 = float(np.sum(lr[max(0, p - 5):p]))
    mom20 = float(np.sum(lr[max(0, p - 20):p]))
    vol20 = float(np.std(lr[max(0, p - 20):p]))
    vol_amt = float(np.std(feat[max(0, p - 20):p, 4]))  # log_vol column
    return np.array([last, mom5, mom20, vol20, vol_amt], dtype=np.float32)


def select_positions(arrays, stop_date, stride):
    """Positions p with ci <= p (post-cutoff? no) and date(p) < stop_date.

    We select positions in the TRAINING period: p in [MIN_POS, ci) with
    date(p) < stop_date.  Stride keeps the cache at budget while covering the
    full training span.
    """
    dates = arrays["dates_dt"]
    ci = arrays["ci"]
    stop64 = np.datetime64(stop_date)
    cand = []
    for p in range(MIN_POS, ci):
        if dates[p] >= stop64:
            break
        cand.append(p)
    if not cand:
        return []
    return cand[::stride]


def extract_caches(*, device, batch_size=4, target_rows=1_200_000, roots=None):
    sel = load_reviewed_selection()
    ckpt_path = upstream_checkpoint_path(sel)
    tok_path = Path(sel["upstream"]["tokenizer"])
    tokenizer = load_tokenizer(str(tok_path), device)
    stocks = load_stocks_uid(DataConfig.data_dir)
    fit_uids, audit_uids = posttrain_split_v1(stocks)
    print(f"[train-cache] fit_uids={len(fit_uids)} audit_uids={len(audit_uids)}")

    model = load_gpt(str(ckpt_path), device, tokenizer=tokenizer)
    model.eval()
    dim = model.head_coarse.in_features

    # ---- build per-stock arrays for fit and audit ----
    attach_close_prices_uid(stocks)
    fit_arrays = [a for s in stocks if (a := build_stock_arrays_uid(s)) is not None
                  and s["stock_uid"] in fit_uids]
    audit_arrays = [a for s in stocks if (a := build_stock_arrays_uid(s)) is not None
                    and s["stock_uid"] in audit_uids]
    print(f"[train-cache] fit arrays={len(fit_arrays)} audit arrays={len(audit_arrays)}")

    # per-stock stride to target approx target_rows across fit stocks
    n_fit = len(fit_arrays)
    total_train_days = sum(len(a["dates_dt"]) for a in fit_arrays)
    stride = max(1, int(total_train_days / max(1, target_rows)))
    print(f"[train-cache] fit stride={stride}")

    # ---- extract fit positions (hidden + targets + c1) ----
    def pack_batches(arr_list, stop_date, stride_val):
        prepped = []
        for a in arr_list:
            poss = select_positions(a, stop_date, stride_val)
            if not poss:
                continue
            prepped.append((a, poss))
        batches = []
        # group by seq_len buckets (reuse prepared-input shape building)
        by_len = {}
        for a, poss in prepped:
            by_len.setdefault(a["T_total"], []).append((a, poss))
        for length, group in sorted(by_len.items()):
            for start in range(0, len(group), batch_size):
                batches.append(group[start:start + batch_size])
        return batches

    def run_batches(arr_groups, stop_date, stride_val, out_prefix):
        """arr_groups: list of (arrays, positions) tuples -> returns rows dict."""
        rows = {k: [] for k in ("stock_uid", "date_key", "hidden", "true_logret",
                                "true_coarse_id", "true_fine_id", "c1_feats",
                                "p_mean0", "p_std0", "offset_note")}
        n_rows = 0
        by_len = {}
        for a, poss in arr_groups:
            by_len.setdefault(a["T_total"], []).append((a, poss))
        import itertools
        # process by length bucket
        for length, group in sorted(by_len.items()):
            for gstart in range(0, len(group), batch_size):
                chunk = group[gstart:gstart + batch_size]
                max_len = length + 1  # BOS + tokens
                B = len(chunk)
                input_ids = torch.zeros(B, max_len, dtype=torch.long)
                time_ids = torch.zeros(B, max_len, 3, dtype=torch.long)
                va_values = torch.zeros(B, max_len, 2, dtype=torch.float32)
                sel_rows, sel_pos = [], []
                all_meta = []
                for bi, (a, poss) in enumerate(chunk):
                    T = a["T_total"]
                    idx_c = a["_coarse"]
                    idx_f = a["_fine"]
                    day = np.asarray(a["day"]); month = np.asarray(a["month"])
                    year = np.asarray(a["year"])
                    # BOS-aligned time arrays: position p uses day of feature row
                    # p-1, matching the evaluator (input position p predicts day p).
                    day_a = np.concatenate([[day[0]], day[:T - 1]])
                    month_a = np.concatenate([[month[0]], month[:T - 1]])
                    year_a = np.concatenate([[year[0]], year[:T - 1]])
                    inp_ids = [int(tokenizer.vocab_coarse)] + idx_c[:T].tolist()
                    seq_len = len(inp_ids) - 1
                    va_seq = np.concatenate([np.zeros((1, 2), dtype=np.float32),
                                             a["va_normed"][:T - 1]], axis=0)
                    input_ids[bi, :seq_len] = torch.as_tensor(inp_ids[:-1])
                    time_ids[bi, :seq_len, 0] = torch.as_tensor(day_a[:seq_len], dtype=torch.long)
                    time_ids[bi, :seq_len, 1] = torch.as_tensor(month_a[:seq_len], dtype=torch.long)
                    time_ids[bi, :seq_len, 2] = torch.as_tensor(year_a[:seq_len], dtype=torch.long)
                    va_values[bi, :seq_len] = torch.as_tensor(va_seq[:seq_len])
                    for p in poss:
                        if p >= seq_len:
                            continue
                        feat = a["feat"]
                        cf = c1_features(feat, p)
                        if cf is None:
                            continue
                        dkey = str(a["dates_dt"][p])[:10]
                        sel_rows.append(bi)
                        sel_pos.append(p)
                        all_meta.append((a["stock_uid"], dkey, float(feat[p, 0]),
                                         int(a["_coarse"][p]), int(a["_fine"][p]),
                                         cf, float(a["p_mean"][0]), float(a["p_std"][0])))
                if not all_meta:
                    continue
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
                    h = model.encode_selected(
                        input_ids.to(device), time_ids.to(device), pos_ids,
                        torch.tensor(sel_rows, device=device),
                        torch.tensor(sel_pos, device=device),
                        va_values=va_values.to(device))
                    h = h.float().cpu().numpy()
                for r, (uid, dkey, tlr, tco, tfi, cf, pm, ps) in enumerate(all_meta):
                    rows["stock_uid"].append(uid)
                    rows["date_key"].append(dkey)
                    rows["hidden"].append(h[r])
                    rows["true_logret"].append(tlr)
                    rows["true_coarse_id"].append(tco)
                    rows["true_fine_id"].append(tfi)
                    rows["c1_feats"].append(cf)
                    rows["p_mean0"].append(pm)
                    rows["p_std0"].append(ps)
                n_rows += len(all_meta)
                if n_rows % 200_000 == 0:
                    print(f"[{out_prefix}] rows {n_rows}")
        # tokenize once per stock up front
        return rows, n_rows

    # tokenize fit + audit stocks
    from model.tokenizer import HierarchicalQuantizer
    for a in fit_arrays + audit_arrays:
        a["_coarse"], a["_fine"] = None, None
    # batch tokenize
    def tokenize_all(arr_list):
        with torch.no_grad():
            for a in arr_list:
                T = a["T_total"]
                chunk = torch.from_numpy(a["price_normed"][:T]).float().unsqueeze(0).to(device)
                idx = tokenizer.encode_all(chunk).squeeze(0).cpu().numpy()
                a["_coarse"] = idx[:, 0].astype(np.int32)
                a["_fine"] = idx[:, 1].astype(np.int16)
    print("[train-cache] tokenizing...")
    tokenize_all(fit_arrays + audit_arrays)

    # fit groups
    fit_groups = []
    for a in fit_arrays:
        poss = select_positions(a, FINAL_FIT_STOP, stride)
        if poss:
            fit_groups.append((a, poss))
    rows_fit, n_fit_rows = run_batches(fit_groups, FINAL_FIT_STOP, stride, "fit")

    # calibration groups (audit)
    cal_groups = []
    for a in audit_arrays:
        # calibration positions: date in [CALIB_START, CALIB_STOP)
        dates = a["dates_dt"]
        ci = a["ci"]
        poss = [p for p in range(MIN_POS, ci)
                if CALIB_START <= str(dates[p])[:10] < CALIB_STOP]
        if poss:
            cal_groups.append((a, poss))
    rows_cal, n_cal_rows = run_batches(cal_groups, CALIB_STOP, 1, "cal")

    paths = artifact_paths(roots=roots)
    paths["training"].parent.mkdir(parents=True, exist_ok=True)

    def save(rows, n_rows, path):
        arr = {k: (np.stack(v) if k == "hidden" or k == "c1_feats"
                   else np.asarray(v)) for k, v in rows.items()}
        np.savez(path, hidden=arr["hidden"],
                 c1_feats=arr["c1_feats"],
                 stock_uid=arr["stock_uid"], date_key=arr["date_key"],
                 true_logret=arr["true_logret"].astype(np.float64),
                 true_coarse_id=arr["true_coarse_id"].astype(np.int16),
                 true_fine_id=arr["true_fine_id"].astype(np.int16),
                 p_mean0=arr["p_mean0"].astype(np.float64),
                 p_std0=arr["p_std0"].astype(np.float64))
        print(f"[train-cache] wrote {path} ({n_rows} rows, {path.stat().st_size/1e6:.0f} MB)")

    save(rows_fit, n_fit_rows, paths["training"])
    save(rows_cal, n_cal_rows, paths["calibration"])
    return paths["training"], paths["calibration"]


def build_training(argv=None):
    ap = argparse.ArgumentParser(description="Build PT-03 training/calibration caches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target_rows", type=int, default=1_200_000)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extract_caches(device=device, batch_size=args.batch_size,
                   target_rows=args.target_rows, roots=resolve_roots())

# ===== build_eval_features =====
def build_eval_features(argv=None):
    paths = artifact_paths()
    eval_hidden = np.load(paths["hidden"], allow_pickle=True)
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

    out = paths["eval_features"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, c1_feats=feats, valid=ok)
    print(f"[eval-features] wrote {out}")
def main():
    import sys as _s
    cmd = _s.argv[1] if len(_s.argv) > 1 else "hidden"
    rest = _s.argv[2:]
    if cmd == "hidden":
        build_hidden(rest)
    elif cmd == "training":
        build_training(rest)
    elif cmd == "c1":
        build_eval_features(rest)
    else:
        print("usage: python common.py [hidden|training|c1]")

if __name__ == "__main__":
    main()
