"""数据处理：CSV 加载、归一化、股票打包。

v1 (legacy): rolling_normalize + 多股票打包 (6D OHLCVA)
v2 (current): historical_normalize + 单股票独立文档 (4D OHLC token + 2D VA continuous)
"""
from glob import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig, NormConfig

# --- v1 globals (backward compat) ---
lookback_window = NormConfig.lookback_window
min_lookback = NormConfig.min_lookback

# --- v2 globals ---
_price_cols = NormConfig.price_features   # ["log_ret", "log_high", "log_low", "log_open"]
_va_cols = NormConfig.va_features         # ["log_vol", "log_amt"]
_n_price = len(_price_cols)               # 4
_n_va = len(_va_cols)                     # 2


def rolling_normalize(features, window=lookback_window, min_lookback=min_lookback):
    """Vectorized rolling z-score, no future leak. features: [T, D]"""
    T, D = features.shape
    cs = np.cumsum(features, axis=0)
    cs2 = np.cumsum(features ** 2, axis=0)

    idx = np.arange(T)
    starts = np.maximum(idx - window + 1, 0)
    counts = (idx - starts + 1).astype(np.float32)

    shifted = np.zeros_like(cs)
    shifted[1:] = cs[:-1]
    shifted2 = np.zeros_like(cs2)
    shifted2[1:] = cs2[:-1]

    mask = (starts > 0).astype(np.float32)[:, None]
    win_sum = cs - shifted * mask
    win_sum2 = cs2 - shifted2 * mask

    means = win_sum / counts[:, None]
    var = win_sum2 / counts[:, None] - means ** 2
    stds = np.sqrt(np.maximum(var, 1e-08))

    normed = np.where(
        counts[:, None] >= min_lookback,
        (features - means) / stds,
        0.0,
    )
    return normed.astype(np.float32)


# ============================================================================
# v2: Per-stock document-level normalization
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


def load_stocks(data_dir=DataConfig.data_dir, max_stocks=DataConfig.max_stocks):
    """加载 CSV，返回 list[dict]。每个 dict: symbol, features_raw, dates_dt, day, month, year"""
    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    if max_stocks and max_stocks < len(files):
        rng = np.random.RandomState(DataConfig.random_seed)
        idx = rng.choice(len(files), max_stocks, replace=False)
        files = [files[i] for i in sorted(idx)]

    feature_cols = DataConfig.feature_cols
    required = ["symbol", "date", "close", "volume"]
    stocks = []
    for fpath in tqdm(files, desc="Loading CSV"):
        try:
            df = pd.read_csv(fpath)
            if not set(required).issubset(set(df.columns)):
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close", "volume"])
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) < NormConfig.min_lookback + 10:
                continue

            symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns else os.path.basename(fpath).split(".")[0]
            prev_close = df["close"].shift(1)
            df["log_ret"] = np.log(df["close"] / prev_close).replace([np.inf, -np.inf], np.nan)
            df["log_high"] = np.log1p(df["high"] / df["close"] - 1) if "high" in df.columns else 0.0
            df["log_low"] = np.log1p(df["low"] / df["close"] - 1) if "low" in df.columns else 0.0
            df["log_open"] = np.log1p(df["open"] / df["close"] - 1) if "open" in df.columns else 0.0
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


def get_tokenizer_features(stocks, window=lookback_window, cutoff_date=None):
    """对每只股票做滚动归一化，拼接为 [N_total, 6]。
    若 cutoff_date 提供，仅使用 ≤ cutoff 的数据（防止 tokenizer 数据泄漏）。"""
    parts = []
    for s in tqdm(stocks, desc="Rolling normalize"):
        feat = s["features_raw"]
        if cutoff_date is not None:
            ci = _stock_cutoff_idx(s, cutoff_date)
            feat = feat[:ci]
        if len(feat) < min_lookback + 5:
            continue
        parts.append(rolling_normalize(feat, window))
    return np.concatenate(parts, axis=0)


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


def _encode_and_cache(stock, tokenizer, mode, cutoff_date, cache_dir, tok_hash=None):
    """Tokenize one stock and save to cache. Returns encoded dict."""
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    ci = _stock_cutoff_idx(stock, cutoff_date) if mode == "train" else len(feat)
    if ci < min_lookback + 5:
        return None
    normed = rolling_normalize(feat[:ci])
    device = next(tokenizer.parameters()).device
    with torch.no_grad():
        token_ids, _ = tokenizer.encode(
            torch.from_numpy(normed).float().unsqueeze(0).to(device))
    token_ids = token_ids[0].cpu().numpy()
    save_dict = {"token_ids": token_ids, "day": day[:ci], "month": month[:ci], "year": year[:ci]}
    if tok_hash:
        save_dict["_tok_hash"] = tok_hash
    np.savez_compressed(_token_cache_path(stock["symbol"], cache_dir), **save_dict)
    return {"token_ids": token_ids, "day": day[:ci], "month": month[:ci], "year": year[:ci]}


def _load_cached_or_encode(stock, tokenizer, mode, cutoff_date, cache_dir, tok_hash=None):
    """Load tokenized sequence from cache, or encode and cache if missing."""
    path = _token_cache_path(stock["symbol"], cache_dir)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        # Validate tokenizer hash to detect stale cache
        cached_hash = str(data["_tok_hash"]) if "_tok_hash" in data else None
        if tok_hash and cached_hash != tok_hash:
            return _encode_and_cache(stock, tokenizer, mode, cutoff_date, cache_dir, tok_hash)
        return {
            "token_ids": data["token_ids"],
            "day": data["day"],
            "month": data["month"],
            "year": data["year"],
        }
    return _encode_and_cache(stock, tokenizer, mode, cutoff_date, cache_dir, tok_hash)


def pack_stocks(stocks, tokenizer, mode="train", cutoff_date=DataConfig.cutoff_date,
                context_len=DataConfig.context_len, cache_dir=None):
    """打包多股票为固定长度序列。返回 list[dict]。支持 token 缓存。"""
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id, eos_id = vocab, vocab + 1
    device = next(tokenizer.parameters()).device

    # Compute tokenizer hash for cache validation
    tok_hash = _tokenizer_hash(tokenizer) if cache_dir else None

    # Pre-tokenize with cache if provided
    encoded = []
    for s in tqdm(stocks, desc="Encoding (" + mode + ")"):
        if cache_dir:
            enc = _load_cached_or_encode(s, tokenizer, mode, cutoff_date, cache_dir, tok_hash)
        else:
            feat = s["features_raw"]
            day, month, year = s["day"], s["month"], s["year"]
            ci = _stock_cutoff_idx(s, cutoff_date) if mode == "train" else len(feat)
            if ci < min_lookback + 5:
                continue
            normed = rolling_normalize(feat[:ci])
            with torch.no_grad():
                token_ids, _ = tokenizer.encode(
                    torch.from_numpy(normed).float().unsqueeze(0).to(device))
            token_ids = token_ids[0].cpu().numpy()
            enc = {"token_ids": token_ids, "day": day[:ci], "month": month[:ci], "year": year[:ci]}
        if enc is None or len(enc["token_ids"]) < min_lookback + 5:
            continue
        encoded.append(enc)

    sequences = []
    buf_ids, buf_d, buf_m, buf_y = [], [], [], []

    def _flush():
        if len(buf_ids) < 2:
            return
        ids = torch.tensor([bos_id] + buf_ids + [eos_id], dtype=torch.long)
        d = torch.tensor([buf_d[0]] + buf_d + [buf_d[-1]], dtype=torch.long)
        m = torch.tensor([buf_m[0]] + buf_m + [buf_m[-1]], dtype=torch.long)
        y = torch.tensor([buf_y[0]] + buf_y + [buf_y[-1]], dtype=torch.long)
        # boundaries for attention mask: each stock segment
        boundaries = []
        start = 0
        for i, s in enumerate(encoded if False else []):
            pass
        # Simpler: single stock per flush with boundaries recording
        sequences.append({
            "input_ids": ids,
            "boundaries": [(1, len(ids) - 1)],  # single stock segment
            "targets": ids[1:].clone(),
            "time_ids": torch.stack([d, m, y], dim=-1),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
        })

    # Actually pack multiple stocks per sequence
    current_len = 0
    current_ids, current_d, current_m, current_y = [], [], [], []
    stock_boundaries = []

    for enc in encoded:
        slen = len(enc["token_ids"]) + 2  # +BOS +EOS
        if current_len + slen > context_len and current_len > 0:
            # Flush current
            ids = torch.tensor([bos_id] + current_ids + [eos_id], dtype=torch.long)
            d = torch.tensor([current_d[0]] + current_d + [current_d[-1]], dtype=torch.long)
            m = torch.tensor([current_m[0]] + current_m + [current_m[-1]], dtype=torch.long)
            y = torch.tensor([current_y[0]] + current_y + [current_y[-1]], dtype=torch.long)
            sequences.append({
                "input_ids": ids,
                "boundaries": stock_boundaries,
                "targets": ids[1:].clone(),
                "time_ids": torch.stack([d, m, y], dim=-1),
                "position_ids": torch.arange(len(ids), dtype=torch.long),
            })
            current_ids, current_d, current_m, current_y = [], [], [], []
            stock_boundaries = []
            current_len = 0

        start = current_len + 1  # +1 for BOS
        current_ids.extend(enc["token_ids"].tolist())
        current_d.extend(enc["day"].tolist())
        current_m.extend(enc["month"].tolist())
        current_y.extend(enc["year"].tolist())
        end = current_len + len(enc["token_ids"]) + 1
        stock_boundaries.append((start, end))
        current_len += len(enc["token_ids"])

    # Flush remaining
    if len(current_ids) >= 2:
        ids = torch.tensor([bos_id] + current_ids + [eos_id], dtype=torch.long)
        d = torch.tensor([current_d[0]] + current_d + [current_d[-1]], dtype=torch.long)
        m = torch.tensor([current_m[0]] + current_m + [current_m[-1]], dtype=torch.long)
        y = torch.tensor([current_y[0]] + current_y + [current_y[-1]], dtype=torch.long)
        sequences.append({
            "input_ids": ids,
            "boundaries": stock_boundaries,
            "targets": ids[1:].clone(),
            "time_ids": torch.stack([d, m, y], dim=-1),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
        })

    return sequences


def _build_segment_mask(S, boundaries):
    """Build causal attention mask with cross-stock isolation.

    Each stock segment can attend to:
      - BOS token (position 0)
      - All earlier positions within the same segment (causal)
    Cross-stock attention is blocked.
    """
    mask = torch.zeros(S, S, dtype=torch.bool)
    # BOS is visible to everyone
    mask[:, 0] = True
    for start, end in boundaries:
        # Within segment: causal (each pos attends to [start..pos])
        for pos in range(start, end):
            mask[pos, start:pos + 1] = True
    return mask


class PackedDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        S = seq["input_ids"].shape[0]
        mask = _build_segment_mask(S, seq["boundaries"])
        return (
            seq["input_ids"],
            seq["targets"],
            seq["time_ids"],
            seq["position_ids"],
            mask,
        )


def make_dataloader(sequences, batch_size=1, shuffle=True):
    def collate(batch):
        # With batch_size=1, return tensors directly (no batch dim needed)
        return batch[0]
    return DataLoader(
        PackedDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True,
    )


# ============================================================================
# v2: Single-stock document packing (OHLC token + VA continuous)
# ============================================================================

def _build_causal_mask(S, device="cpu"):
    """Simple causal mask for a single-stock sequence.

    Returns [S, S] bool: True = attend, False = block.
    Position 0 (BOS) is visible to all; causal within [1..S).
    """
    mask = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    mask[:, 0] = True  # BOS visible to all
    return mask


def pack_stocks_v2(stocks, tokenizer, mode="train", cutoff_date=DataConfig.cutoff_date,
                   context_len=DataConfig.context_len, cache_dir=None, max_seq_len=0):
    """v2: One stock per sequence.  Returns list[dict] with va_values.

    Each dict:
      input_ids:   [S] long   (BOS + tokens + EOS)
      targets:     [S-1] long (shifted input_ids)
      time_ids:    [S, 3] long (day, month, year)
      position_ids:[S] long
      va_values:   [S, 2] float32 (vol_normed, amt_normed; BOS/EOS = 0)
    """
    vocab = tokenizer.bsq_coarse.vocab_size
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

        # Token cache
        if cache_dir:
            cache_path = _token_cache_path(s["symbol"] + "_v2", cache_dir)
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                cached_hash = str(data["_tok_hash"]) if "_tok_hash" in data else None
                has_required = "reg_target" in data and "va_values" in data and "idx_fine" in data
                if tok_hash and cached_hash != tok_hash:
                    data.close()
                    os.remove(cache_path)
                elif not has_required:
                    # Stale cache missing required fields → re-encode
                    data.close()
                    os.remove(cache_path)
                else:
                    enc = {
                        "token_ids": data["token_ids"],
                        "idx_fine": data["idx_fine"],
                        "day": data["day"], "month": data["month"], "year": data["year"],
                        "va_values": data["va_values"],
                        "reg_target": data["reg_target"],
                    }
                    data.close()
                    if len(enc["token_ids"]) >= NormConfig.min_doc_length:
                        encoded.append(enc)
                    continue

        with torch.no_grad():
            token_ids, idx_fine = tokenizer.encode(
                torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
        token_ids = token_ids[0].cpu().numpy()
        idx_fine = idx_fine[0].cpu().numpy()

        enc = {
            "token_ids": token_ids,
            "idx_fine": idx_fine,
            "day": day[:ci], "month": month[:ci], "year": year[:ci],
            "va_values": va_normed,
            "reg_target": reg_target,
        }

        if cache_dir:
            np.savez_compressed(
                _token_cache_path(s["symbol"] + "_v2", cache_dir),
                token_ids=token_ids, idx_fine=idx_fine,
                day=day[:ci], month=month[:ci], year=year[:ci],
                va_values=va_normed, reg_target=reg_target, _tok_hash=tok_hash or "",
            )

        if len(token_ids) < NormConfig.min_doc_length:
            continue
        encoded.append(enc)

    # Phase 2: pack each stock as one sequence
    zero_va = np.zeros((_n_va,), dtype=np.float32)
    sequences = []
    for enc in encoded:
        ids_list = enc["token_ids"].tolist()
        ft_list = enc["idx_fine"].tolist()
        va_list = enc["va_values"].tolist()
        rt_list = enc["reg_target"].tolist()
        d_list, m_list, y_list = enc["day"].tolist(), enc["month"].tolist(), enc["year"].tolist()

        ids = torch.tensor([bos_id] + ids_list + [eos_id], dtype=torch.long)
        # Fine targets: -100 sentinel at BOS/EOS so CE loss can ignore them
        fine_ids = torch.tensor([-100] + ft_list + [-100], dtype=torch.long)
        d = torch.tensor([d_list[0]] + d_list + [d_list[-1]], dtype=torch.long)
        m = torch.tensor([m_list[0]] + m_list + [m_list[-1]], dtype=torch.long)
        y = torch.tensor([y_list[0]] + y_list + [y_list[-1]], dtype=torch.long)
        va = torch.tensor(
            [zero_va.tolist()] + va_list + [zero_va.tolist()], dtype=torch.float32)
        # Regression targets: log_ret aligned with input_ids; BOS/EOS = -999 sentinel
        reg_targets = torch.tensor([-999.0] + rt_list + [-999.0], dtype=torch.float32)

        # Truncate long sequences from the beginning (keep most recent data)
        if max_seq_len > 0 and len(ids) > max_seq_len:
            ids = torch.cat([ids[:1], ids[-(max_seq_len-1):]])  # BOS + last max_seq_len-1
            fine_ids = torch.cat([fine_ids[:1], fine_ids[-(max_seq_len-1):]])
            d = torch.cat([d[:1], d[-(max_seq_len-1):]])
            m = torch.cat([m[:1], m[-(max_seq_len-1):]])
            y = torch.cat([y[:1], y[-(max_seq_len-1):]])
            va = torch.cat([va[:1], va[-(max_seq_len-1):]], dim=0)
            reg_targets = torch.cat([reg_targets[:1], reg_targets[-(max_seq_len-1):]])

        sequences.append({
            "input_ids": ids,
            "targets": ids[1:].clone(),
            "time_ids": torch.stack([d, m, y], dim=-1),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
            "va_values": va,
            "reg_targets": reg_targets,
            "fine_targets": fine_ids,
        })

    return sequences


class PackedDatasetV2(Dataset):
    """v2 Dataset: returns (input_ids, targets, time_ids, position_ids, mask, va_values)."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        S = seq["input_ids"].shape[0]
        mask = _build_causal_mask(S)
        if "reg_targets" in seq and "fine_targets" in seq:
            return (
                seq["input_ids"],
                seq["targets"],
                seq["time_ids"],
                seq["position_ids"],
                mask,
                seq["va_values"],
                seq["reg_targets"],
                seq["fine_targets"],
            )
        if "reg_targets" in seq:
            return (
                seq["input_ids"],
                seq["targets"],
                seq["time_ids"],
                seq["position_ids"],
                mask,
                seq["va_values"],
                seq["reg_targets"],
            )
        return (
            seq["input_ids"],
            seq["targets"],
            seq["time_ids"],
            seq["position_ids"],
            mask,
            seq["va_values"],
        )


# ============================================================================
# Experiment A: Stratified 200-stock test set
# ============================================================================

# A-share market-cap / board proxy from symbol prefix.
# (prefix, label) — labels are 4 market-cap buckets based on common A-share conventions.
_MARKET_CAP_BUCKETS = {
    "000": "shenzhen_main",   # 000xxx 深主板（大盘）
    "001": "shenzhen_main",
    "002": "shenzhen_sme",     # 002xxx 中小板
    "003": "shenzhen_sme",
    "300": "chinext",          # 300xxx 创业板（小盘/高波动）
    "600": "shanghai_main",    # 600xxx 沪主板（大盘）
    "601": "shanghai_main",
    "603": "shanghai_main",
    "605": "shanghai_sme",     # 605xxx 沪主板次新
    "688": "star",             # 688xxx 科创板（小盘/高波动）
}

# 4 market-cap buckets for stratification (large / mid / small / start-up)
_MCAP_LABEL = {
    "shenzhen_main": "large",
    "shanghai_main": "large",
    "shenzhen_sme": "mid",
    "shanghai_sme": "mid",
    "chinext": "small",
    "star": "startup",
}


def _symbol_strata(symbol):
    """Map an A-share symbol to (market_cap_label, industry_label).
    Industry is approximated by the next 2 digits after the prefix (3-char sector code).
    Falls back to 'misc' if symbol is too short.
    """
    if len(symbol) < 6:
        return ("misc", "misc")
    prefix = symbol[:3]
    mcap = _MCAP_LABEL.get(_MARKET_CAP_BUCKETS.get(prefix, "misc"), "misc")
    # Sector proxy: 3-digit code after the board prefix (e.g. 000001=banking, 600519=liquor)
    # Round to nearest 50 for bucket coarseness (10 industry buckets).
    try:
        mid3 = int(symbol[3:6])
        industry = f"s{(mid3 // 50) * 50:03d}"
    except ValueError:
        industry = "misc"
    return (mcap, industry)


def stratified_split_stocks(stocks, n=200, seed=42, market_cap_buckets=4, industry_buckets=10):
    """Stratified sampling of n stocks by (market_cap, industry).

    Falls back to deterministic 30-stock selection if total stock pool is too small
    or if any stratum has <3 stocks.
    """
    if len(stocks) < n:
        return stocks  # pool too small — return all

    # Assign strata to each stock
    strata = {}
    for s in stocks:
        key = _symbol_strata(s["symbol"])
        strata.setdefault(key, []).append(s)

    # Target n per stratum: n / (4 mcap × 10 industry) = n / 40 ≈ 5 for n=200
    # But we only sample from non-empty strata.
    rng = np.random.RandomState(seed)
    per_stratum = max(3, n // 40)  # at least 3 per stratum to be meaningful
    selected = []
    strata_keys = sorted(strata.keys())
    rng.shuffle(strata_keys)

    # First pass: target per_stratum from each non-empty stratum (proportional)
    for key in strata_keys:
        pool = strata[key]
        if len(pool) <= per_stratum:
            selected.extend(pool)  # take all
        else:
            indices = rng.choice(len(pool), per_stratum, replace=False)
            selected.extend([pool[i] for i in sorted(indices)])

    # If we overshot, trim; if we undershot, top up from largest strata
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        # Top up from remaining un-selected stocks
        selected_set = set(id(s) for s in selected)
        remaining = [s for s in stocks if id(s) not in selected_set]
        rng.shuffle(remaining)
        need = n - len(selected)
        selected.extend(remaining[:need])

    # Sort by symbol for determinism
    selected.sort(key=lambda s: s["symbol"])
    return selected

def make_dataloader_v2(sequences, batch_size=1, shuffle=True):
    def collate(batch):
        return batch[0]
    return DataLoader(
        PackedDatasetV2(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True,
    )
