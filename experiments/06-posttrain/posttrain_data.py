"""PT-00C/PT-00D: PostTrain data contract, splits, and cross-section loader.

Stable identity is ``stock_uid`` (generated from the normalized source relative
path at CSV-load time); ``symbol`` is display-only and must never be used as a
join/group key (4 index/stock symbol collisions are a known limitation).

Row contract (per basic sample):

    stock_uid / source_relpath / symbol / date / offset
    selection_position == target_feature_index == p   (input position p predicts
        source feature/token row p; NO extra ``+1``)
    raw_logret_1d   = features_raw[p, 0]
    target_coarse_id / target_fine_id   (real joint token at p)
    p_mean[0] / p_std[0]   (train-period normalization of feature 0)

Old ``stocks_*.pkl`` symbol-keyed caches are NOT consumed here; a new
UID-keyed cache under ``dataset/.cache/posttrain_stocks_*.pkl`` is used, and its
fingerprint includes the sorted CSV relpath list so renames invalidate it.

Splits are fixed as ``posttrain_split_v1`` (see write_split_definition).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from config import DataConfig, NormConfig
from data_processor import document_normalize, _stock_cutoff_idx

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
