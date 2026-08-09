"""数据处理：CSV 加载、归一化、股票打包。

单股票独立文档 (4D OHLC token + 2D VA continuous)。
归一化：historical Z-Score（统计量仅来自 train 数据）。
"""
from glob import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig, NormConfig
import regime

# --- globals ---
_price_cols = NormConfig.price_features   # ["log_ret", "log_high", "log_low", "log_open"]
_va_cols = NormConfig.va_features         # ["log_vol", "log_amt"]
_n_price = len(_price_cols)               # 4
_n_va = len(_va_cols)                     # 2


# ============================================================================
# Per-stock document-level normalization
# ============================================================================

def document_normalize(features_raw, cutoff_idx=None):
    """Per-stock historical Z-Score normalization.

    Statistics (mean/std) are computed from the stock's train-period data only
    (features_raw[:cutoff_idx]), then applied to the full sequence.  Each position
    sees the same normalization — no rolling window, no future data.

    Price features (OHLC): Z-Score using train-period stats.
    Volume/Amount: first-day baseline then Z-Score.

    Args:
        features_raw: [T, 6] raw features (log_ret, log_high, log_low, log_open, log_vol, log_amt)
        cutoff_idx: if provided, stats are computed ONLY from features_raw[:cutoff_idx]
                    (train data).  If None, stats come from the whole array (tokenizer training).
    Returns:
        price_normed: [T, 4] Z-Score normalized OHLC features
        va_normed:    [T, 2] Z-Score normalized Volume/Amount (first-day baseline)
    """
    # Split into price (OHLC) and VA
    price = features_raw[:, :_n_price]   # [T, 4]
    va = features_raw[:, _n_price:]      # [T, 2] (log_vol, log_amt)

    # --- Price: historical Z-Score ---
    stats_slice = price[:cutoff_idx] if cutoff_idx is not None else price
    p_mean = stats_slice.mean(axis=0)
    p_std = stats_slice.std(axis=0)
    p_std = np.maximum(p_std, 1e-8)
    price_normed = (price - p_mean) / p_std

    # --- VA: first-day baseline + Z-Score ---
    va_base = va[0:1, :]               # [1, 2] first day's (log_vol, log_amt)
    va_rel = va - va_base               # log-ratio relative to day 0

    stats_slice_va = va_rel[:cutoff_idx] if cutoff_idx is not None else va_rel
    va_mean = stats_slice_va.mean(axis=0)
    va_std = stats_slice_va.std(axis=0)
    va_std = np.maximum(va_std, 1e-8)
    va_normed = (va_rel - va_mean) / va_std

    return price_normed.astype(np.float32), va_normed.astype(np.float32)


def _stock_cutoff_idx(stock, cutoff_date):
    cutoff = np.datetime64(pd.Timestamp(cutoff_date))
    return int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))


def _stock_cache_path(data_dir, max_stocks):
    """Path to binary stock cache file."""
    os.makedirs(os.path.join(data_dir, ".cache"), exist_ok=True)
    tag = f"all_{max_stocks}" if max_stocks else "all"
    return os.path.join(data_dir, ".cache", f"stocks_{tag}.pkl")


def _csv_fingerprint(data_dir, max_stocks):
    """Fast fingerprint of CSV directory: (file_count, total_size_bytes)."""
    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    if max_stocks and max_stocks < len(files):
        files = files[:max_stocks]  # approximate: first N after sort
    total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    return len(files), total_size


def load_stocks(data_dir=DataConfig.data_dir, max_stocks=DataConfig.max_stocks):
    """加载 CSV，返回 list[dict]。每个 dict: symbol, features_raw, dates_dt, day, month, year

    首次加载解析CSV并缓存为pickle，后续直接加载缓存（<1s）。
    """
    import pickle
    cache_path = _stock_cache_path(data_dir, max_stocks)
    fp = _csv_fingerprint(data_dir, max_stocks)

    # Try loading from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fp:
                stocks = cached["stocks"]
                print(f"  [cache] Loaded {len(stocks)} stocks from {os.path.basename(cache_path)}")
                return stocks
        except Exception:
            pass  # cache invalid, fall through

    # CSV loading path
    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    if max_stocks and max_stocks < len(files):
        rng = np.random.RandomState(DataConfig.random_seed)
        idx = rng.choice(len(files), max_stocks, replace=False)
        files = [files[i] for i in sorted(idx)]

    feature_cols = DataConfig.feature_cols
    stocks = []
    for fpath in tqdm(files, desc="Loading CSV"):
        try:
            df = pd.read_csv(fpath)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close", "volume"])
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) < NormConfig.min_lookback + 10:
                continue

            symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns else os.path.basename(fpath).split(".")[0]
            ohlcv_cols = ["date", "close", "high", "low", "open", "volume"]
            if not all(c in df.columns for c in ohlcv_cols):
                continue  # skip stocks with missing OHLCV columns
            prev_close = df["close"].shift(1)
            df["log_ret"] = np.log(df["close"] / prev_close).replace([np.inf, -np.inf], np.nan)
            df["log_high"] = np.log1p(df["high"] / df["close"] - 1)
            df["log_low"] = np.log1p(df["low"] / df["close"] - 1)
            df["log_open"] = np.log1p(df["open"] / df["close"] - 1)
            df["log_vol"] = np.log1p(df["volume"])
            df["log_amt"] = np.log1p(df["amount"]) if "amount" in df.columns else np.log1p(df["volume"] * df["close"])
            df = df.dropna().reset_index(drop=True)

            features = df[feature_cols].values.astype(np.float32)
            dates_dt = df["date"].values
            day = df["date"].dt.day.values.astype(np.int64)
            month = df["date"].dt.month.values.astype(np.int64)
            year = (df["date"].dt.year - 2010).clip(0, 99).values.astype(np.int64)

            stocks.append({
                "symbol": symbol,
                "features_raw": features,
                "dates_dt": dates_dt,
                "day": day, "month": month, "year": year,
            })
        except Exception:
            continue

    # Save to binary cache for fast subsequent loads
    if stocks:
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump({"fingerprint": fp, "stocks": stocks}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [cache] Saved {len(stocks)} stocks to {os.path.basename(cache_path)}")
        except Exception:
            pass  # non-fatal

    return stocks


def split_stocks(stocks, cutoff_date=DataConfig.cutoff_date, train_ratio=DataConfig.train_ratio):
    """按 cutoff_date 切分。返回 (train, val, test)。"""
    cutoff = pd.Timestamp(cutoff_date)
    tv, test = [], []
    for s in stocks:
        if s["dates_dt"][-1] < np.datetime64(cutoff):
            tv.append(s)
        else:
            ci = _stock_cutoff_idx(s, cutoff_date)
            if ci > 0:
                tv.append(s)
            test.append(s)

    n_train = max(1, int(len(tv) * train_ratio))
    rng = np.random.RandomState(DataConfig.random_seed)
    perm = rng.permutation(len(tv))
    train_idx = set(perm[:n_train])
    train, val = [], []
    for i in range(len(tv)):
        if i in train_idx:
            train.append(tv[i])
        else:
            val.append(tv[i])
    return train, val, test


def get_tokenizer_features_v2(stocks, cutoff_date=None):
    """v2: Per-stock historical normalize (4D OHLC only). Returns [N_total, 4]."""
    parts = []
    for s in tqdm(stocks, desc="Document normalize (4D)"):
        feat = s["features_raw"]
        if cutoff_date is not None:
            ci = _stock_cutoff_idx(s, cutoff_date)
            feat = feat[:ci]
        if len(feat) < NormConfig.min_doc_length:
            continue
        price_normed, _ = document_normalize(feat)
        parts.append(price_normed)
    if not parts:
        return np.zeros((0, _n_price), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _token_cache_path(symbol, cache_dir):
    """Path for cached tokenized sequence of a single stock."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{symbol}.npz")


def _tokenizer_hash(tokenizer):
    """Compute a short hash of tokenizer weights for cache validation."""
    import hashlib
    buf = b""
    for k, v in sorted(tokenizer.state_dict().items()):
        buf += k.encode() + v.cpu().numpy().tobytes()
    return hashlib.md5(buf).hexdigest()[:16]


# ============================================================================
# v2: Single-stock document packing (OHLC token + VA continuous)
# ============================================================================

def _build_causal_mask(S, device="cpu"):
    """Causal mask for a single-stock sequence.

    Returns [S, S] bool: True = attend, False = block.
    Lower-triangular: position i attends to positions [0..i].
    """
    return torch.ones(S, S, dtype=torch.bool, device=device).tril()


def _trailing_vol20(logret: np.ndarray, window: int = 20) -> np.ndarray:
    """Trailing ``window``-day realized volatility, strict point-in-time.

    ``out[i]`` = std of raw log_ret over ``[i-window, i)`` (days before i only,
    day i itself excluded); rows ``i < window`` lack a full window and are NaN.
    Returns [T] float32.
    """
    logret = np.asarray(logret, dtype=np.float64)
    T = logret.shape[0]
    out = np.full(T, np.nan, dtype=np.float32)
    if window <= 0:
        return out
    for i in range(window, T):
        out[i] = np.std(logret[i - window:i])
    return out


def pack_stocks_v2(stocks, tokenizer, mode="train", cutoff_date=DataConfig.cutoff_date,
                   context_len=DataConfig.context_len, cache_dir=None, max_seq_len=0,
                   regime_thresholds=None, regime_window=20):
    """v2: One stock per sequence.  Returns list[dict] with va_values.

    Each dict:
      input_ids:   [S] long   (BOS + tokens + EOS)
      targets:     [S-1] long (shifted input_ids)
      time_ids:    [S, 3] long (day, month, year)
      position_ids:[S] long
      va_values:   [S, 2] float32 (vol_normed, amt_normed; BOS/EOS = 0)
      regime_ids:  [S] long   (Branch F: per-input-position regime, aligned to
                               input_ids; -1 = BOS/EOS/insufficient-window/None)

    Branch F (LoRA-per-regime): when ``regime_thresholds`` is not None, per-day
    regime labels are computed from the stock's raw ``log_ret`` (features_raw
    col 0) using ``regime.trailing_realized_vol`` (inclusive trailing window,
    strict point-in-time — the ONLY regime source for Branch F; never
    ``_trailing_vol20``) and ``regime.label_regime``.  Input position p holds
    the token for day p-1, so ``regime_ids[p] = label(rv[p-1])`` for p in
    1..S-2; BOS (0) and EOS (S-1) are -1.  When ``regime_thresholds`` is None
    every regime id is -1 (dense/Branch C behaviour unchanged).
    """
    vocab = tokenizer.vocab_coarse  # coarse-only for BOS/EOS IDs
    bos_id, eos_id = vocab, vocab + 1
    device = next(tokenizer.parameters()).device
    tok_hash = _tokenizer_hash(tokenizer) if cache_dir else None

    # Phase 1: tokenize each stock with historical_normalize
    encoded = []
    for s in tqdm(stocks, desc="Encoding v2 (" + mode + ")"):
        feat = s["features_raw"]
        day, month, year = s["day"], s["month"], s["year"]
        ci = _stock_cutoff_idx(s, cutoff_date) if mode == "train" else len(feat)
        if ci < NormConfig.min_doc_length:
            continue

        price_normed, va_normed = document_normalize(feat[:ci], cutoff_idx=ci if mode == "train" else None)
        reg_target = np.abs(price_normed[:, 0]).copy()  # [T] volatility: |normalized log_ret|
        # Branch B: signed normalized log_ret (price feature 0) — the realized
        # value for cross-sectional ListNet ranking.  Same units as the coarse
        # codebook's expected-log_ret score.
        reg_signed = price_normed[:, 0].copy()  # [T] signed normalized log_ret

        # Branch F (LoRA-per-regime): per-day regime labels over the SAME rows
        # that get tokenized (feat[:ci]).  Uses regime.py exclusively (inclusive
        # trailing window, strict point-in-time); never data_processor's
        # _trailing_vol20.  None when the caller did not pass thresholds.
        regime_labels = None
        if regime_thresholds is not None:
            rvol = regime.trailing_realized_vol(feat[:ci, 0], regime_window)
            regime_labels = regime.label_regime(rvol, regime_thresholds)  # [ci] int64

        # Token cache
        if cache_dir:
            cache_path = _token_cache_path(s["symbol"] + "_v2", cache_dir)
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                cached_hash = str(data["_tok_hash"]) if "_tok_hash" in data else None
                has_required = "reg_signed" in data and "reg_target" in data and "va_values" in data
                if tok_hash and cached_hash != tok_hash:
                    data.close()
                    os.remove(cache_path)
                elif not has_required:
                    data.close()
                    os.remove(cache_path)
                else:
                    enc = {
                        "token_ids": data["token_ids"],
                        "fine_ids": data["fine_ids"] if "fine_ids" in data else data["token_ids"],
                        "day": data["day"], "month": data["month"], "year": data["year"],
                        "va_values": data["va_values"],
                        "reg_target": data["reg_target"],
                        "reg_signed": data["reg_signed"],
                        "symbol": s["symbol"],
                        "regime_labels": regime_labels,
                    }
                    data.close()
                    if len(enc["token_ids"]) >= NormConfig.min_doc_length:
                        encoded.append(enc)
                    continue

        with torch.no_grad():
            all_idx = tokenizer.encode_all(
                torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
            token_ids = all_idx[0, :, 0].cpu().numpy()    # coarse IDs
            fine_ids = all_idx[0, :, 1].cpu().numpy()     # fine IDs

        enc = {
            "token_ids": token_ids,
            "fine_ids": fine_ids,
            "day": day[:ci], "month": month[:ci], "year": year[:ci],
            "va_values": va_normed,
            "reg_target": reg_target,
            "reg_signed": reg_signed,
            "symbol": s["symbol"],
            "regime_labels": regime_labels,
        }

        if cache_dir:
            np.savez_compressed(
                _token_cache_path(s["symbol"] + "_v2", cache_dir),
                token_ids=token_ids, fine_ids=fine_ids,
                day=day[:ci], month=month[:ci], year=year[:ci],
                va_values=va_normed, reg_target=reg_target, reg_signed=reg_signed,
                _tok_hash=tok_hash or "",
            )

        if len(token_ids) < NormConfig.min_doc_length:
            continue
        encoded.append(enc)

    # Phase 2: pack each stock as one sequence
    zero_va_list = np.zeros((_n_va,), dtype=np.float32).tolist()
    sequences = []
    for enc in encoded:
        ids_list = enc["token_ids"].tolist()
        fine_list = enc["fine_ids"].tolist()
        va_list = enc["va_values"].tolist()
        rt_list = enc["reg_target"].tolist()
        srt_list = enc["reg_signed"].tolist()
        d_list, m_list, y_list = enc["day"].tolist(), enc["month"].tolist(), enc["year"].tolist()

        # BOS has no fine target. EOS uses the same ignore sentinel as padded
        # fine targets so valid fine code 0 remains trainable.
        ids = torch.tensor([bos_id] + ids_list + [eos_id], dtype=torch.long)
        fine = torch.tensor([0] + fine_list + [-100], dtype=torch.long)
        d = torch.tensor([d_list[0]] + d_list + [d_list[-1]], dtype=torch.long)
        m = torch.tensor([m_list[0]] + m_list + [m_list[-1]], dtype=torch.long)
        y = torch.tensor([y_list[0]] + y_list + [y_list[-1]], dtype=torch.long)
        va = torch.tensor(
            [zero_va_list] + va_list + [zero_va_list], dtype=torch.float32)
        reg_targets = torch.tensor([-999.0] + rt_list + [-999.0], dtype=torch.float32)
        reg_signed_t = torch.tensor([-999.0] + srt_list + [-999.0], dtype=torch.float32)

        # Branch F: per-position regime ids aligned to input_ids.  Position p
        # predicts day p (input_ids[p+1]); the regime of that prediction is
        # label(rv[p-1]) (strict point-in-time), so regime_ids[1:S-1] =
        # labels[0:S-2].  BOS (0), EOS (S-1) and any token with an insufficient
        # trailing window (label -1) carry -1 and fire no LoRA adapter.
        regime_labels = enc.get("regime_labels")
        if regime_labels is not None and len(regime_labels) >= len(ids_list):
            regime_ids_np = np.full(len(ids), -1, dtype=np.int64)
            n_tok = len(ids_list)
            regime_ids_np[1:1 + n_tok] = regime_labels[:n_tok]
            regime_ids = torch.from_numpy(regime_ids_np)
        else:
            # Symbol collision (index/ETF CSV shares a stock symbol) or a stale
            # cache: regime_labels length may not match the token count.  Fall
            # back to all -1 (no regime adapter fires for this stock).
            regime_ids = torch.full((len(ids),), -1, dtype=torch.long)

        # Truncate long sequences from the beginning (keep most recent data)
        source_offset = 0
        if max_seq_len > 0 and len(ids) > max_seq_len:
            source_offset = len(ids_list) - (max_seq_len - 2)
            ids = torch.cat([ids[:1], ids[-(max_seq_len-1):]])
            fine = torch.cat([fine[:1], fine[-(max_seq_len-1):]])
            d = torch.cat([d[:1], d[-(max_seq_len-1):]])
            m = torch.cat([m[:1], m[-(max_seq_len-1):]])
            y = torch.cat([y[:1], y[-(max_seq_len-1):]])
            va = torch.cat([va[:1], va[-(max_seq_len-1):]], dim=0)
            reg_targets = torch.cat([reg_targets[:1], reg_targets[-(max_seq_len-1):]])
            reg_signed_t = torch.cat([reg_signed_t[:1], reg_signed_t[-(max_seq_len-1):]])
            regime_ids = torch.cat([regime_ids[:1], regime_ids[-(max_seq_len-1):]])

        sequences.append({
            "input_ids": ids,
            "regime_ids": regime_ids,
            "targets": ids[1:].clone(),
            "fine_targets": fine[1:].clone(),  # shifted fine IDs for dual-head
            "time_ids": torch.stack([d, m, y], dim=-1),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
            "va_values": va,
            "reg_targets": reg_targets,
            "reg_signed": reg_signed_t,
            # Branch C: map seq position p back to this stock's features_raw row
            # (source_offset = head rows dropped by truncation, 0 when untruncated).
            "symbol": enc["symbol"],
            "source_offset": source_offset,
        })

    return sequences


def attach_sample_weights(train_seqs, stocks_by_symbol, *, mode="combined",
                          recency_tau_days=504, weight_clip=(0.05, 20.0),
                          regime_percentiles=(1/3, 2/3)):
    """Branch C: per-position loss re-weights (recency / regime balance).

    Attaches to each ``train_seqs`` dict a ``sample_weights`` tensor [S-1]
    aligned to the ``targets`` grid (last/EOS position = 0) and a ``regime_ids``
    tensor [S-1] (0=low-vol, 2=mid-vol, 3=high-vol, 1=insufficient history/NaN),
    then returns ``{"weight_config": ..., "regime_stats": ...}``.

    recency:  w_rec = exp(-d / tau) with d = (n_days_kept - 1) - p the trading-day
              distance from position p's day to the cutoff.  All stocks share the
              same cutoff (DataConfig.cutoff_date) and use their own trading-day
              index, so d naturally spans holidays.
    regime:   trailing 20-day realized volatility (strict point-in-time, days
              [p-20, p) only) tercile thresholds are global over the TRAIN split
              only.  w_reg = n_total / (3 * n_reg) makes the three tercile
              classes equally-weighted (importance-weighting equivalent to
              stratified resampling); the NaN regime keeps its own
              inverse-proportional weight.
    combined: w = clip(w_rec * w_reg, [lo, hi]); EOS target position = 0.

    ``stocks_by_symbol`` maps symbol -> raw stock dict (features_raw) for the
    same train split; the caller builds it from ``train_s``.
    """
    lo, hi = float(weight_clip[0]), float(weight_clip[1])
    use_recency = mode in ("recency", "combined")
    use_regime = mode in ("regime", "combined")

    # ---- regime: per-symbol trailing vol20 + train-split global terciles ----
    stock_rvol = {}
    thresholds = (0.0, 0.0)
    if use_regime:
        all_vals = []
        for seq in train_seqs:
            sym = seq.get("symbol")
            if sym is None or sym in stock_rvol:
                continue
            s = stocks_by_symbol.get(sym)
            if s is None or "features_raw" not in s:
                continue
            rvol = _trailing_vol20(s["features_raw"][:, 0])
            stock_rvol[sym] = rvol
            ci = _stock_cutoff_idx(s, DataConfig.cutoff_date)
            vals = rvol[:ci]
            all_vals.append(vals[np.isfinite(vals)])
        if all_vals:
            v = np.concatenate(all_vals)
            p33, p67 = (float(np.quantile(v, q)) for q in regime_percentiles)
            thresholds = (p33, p67)

    # ---- pass 1: regime id + class counts over effective positions ----
    # reg: 0 = low, 2 = mid, 3 = high, 1 = NaN (insufficient trailing window).
    regime_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_effective = 0
    for seq in train_seqs:
        n = int(seq["targets"].shape[0])
        off = int(seq.get("source_offset", 0))
        rvol = stock_rvol.get(seq.get("symbol")) if use_regime else None
        # Symbol collision guard: index/ETF CSVs share symbol names with stocks
        # (e.g. 科创50 == "688"), so a seq's rvol may come from a *different*
        # ticker whose length does not match. Fall back to the nan-regime group
        # (reg=1) instead of indexing out of bounds.
        if use_regime and rvol is not None and off + (n - 2) >= len(rvol):
            rvol = None
        reg_ids = np.full(n, -1, dtype=np.int64)
        for p in range(n - 1):           # last position = EOS target -> weight 0
            total_effective += 1
            if use_regime and rvol is not None:
                rv = rvol[off + p]
                if np.isnan(rv):
                    reg = 1
                elif rv <= thresholds[0]:
                    reg = 0
                elif rv <= thresholds[1]:
                    reg = 2
                else:
                    reg = 3
                reg_ids[p] = reg
                regime_counts[reg] += 1
            elif use_regime:
                reg_ids[p] = 1
                regime_counts[1] += 1
        seq["regime_ids"] = torch.from_numpy(reg_ids)

    # ---- pass 2: per-position weight ----
    regime_weights = {}
    if use_regime:
        for reg, cnt in regime_counts.items():
            regime_weights[reg] = total_effective / (3.0 * max(cnt, 1))
    for seq in train_seqs:
        n = int(seq["targets"].shape[0])
        w = np.ones(n, dtype=np.float32)
        if use_recency:
            tau = max(int(recency_tau_days), 1)
        for p in range(n - 1):
            if use_recency:
                d = float((n - 2) - p)   # trading days from day p to cutoff
                w[p] = float(np.exp(-d / tau))
            if use_regime:
                w[p] = w[p] * regime_weights[int(seq["regime_ids"][p])]
            w[p] = float(np.clip(w[p], lo, hi))
        w[n - 1] = 0.0                   # EOS target position is meaningless
        seq["sample_weights"] = torch.from_numpy(w)

    # ---- summary ----
    regime_labels = {0: "low_vol", 1: "nan_insufficient", 2: "mid_vol", 3: "high_vol"}
    regime_stats = {
        regime_labels[reg]: {
            "regime_id": reg,
            "n_positions": regime_counts[reg],
            "effective_uniform_weight": (
                regime_weights.get(reg, 1.0) if use_regime else 1.0
            ),
        }
        for reg in sorted(regime_counts)
    }
    weight_config = {
        "mode": mode,
        "recency_tau_days": recency_tau_days,
        "weight_clip": [lo, hi],
        "regime_percentiles": list(regime_percentiles),
        "regime_thresholds": list(thresholds) if use_regime else None,
        "regime_window_days": 20,
        "n_effective_positions": total_effective,
        "regime_counts": dict(regime_counts),
    }
    return {"weight_config": weight_config, "regime_stats": regime_stats}


class PackedDatasetV2(Dataset):
    """Dataset: returns sequence dict (mask built in collate)."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_dataloader_v2(sequences, batch_size=1, shuffle=True, num_workers=0):
    """DataLoader for single-sequence batches. Returns 10-tuple matching _pad_batch format.

    When batch_size=1, no explicit mask is needed — SDPA uses is_causal=True instead.
    """
    def collate(batch):
        s = batch[0]
        # Return None for mask — SDPA will use is_causal=True for causal attention
        sample_weights = s.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones_like(s["targets"], dtype=torch.float32)
        # Branch F: only surface input_ids-aligned regime_ids ([S]) into the
        # batch.  Branch C's targets-aligned regime_ids ([S-1]) is a different
        # grid and is left as None (it never routes the model).
        regime_ids = s.get("regime_ids")
        if regime_ids is not None and regime_ids.shape[0] != s["input_ids"].shape[0]:
            regime_ids = None
        return (
            s["input_ids"],
            s["targets"],
            s.get(
                "fine_targets",
                torch.full_like(s["targets"], -100),
            ),
            s["time_ids"],
            s["position_ids"],
            None,  # mask: let SDPA handle causal attention via is_causal=True
            s["va_values"],
            s["reg_targets"],
            sample_weights,  # Branch C: [S-1] per-position loss weights (all-1 default)
            regime_ids,      # Branch F: [S] per-input-position regime ids (None when absent)
        )
    return DataLoader(
        PackedDatasetV2(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True,
        num_workers=num_workers,
    )
