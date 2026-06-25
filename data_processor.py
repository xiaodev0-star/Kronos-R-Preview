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

        # Token cache
        if cache_dir:
            cache_path = _token_cache_path(s["symbol"] + "_v2", cache_dir)
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                cached_hash = str(data["_tok_hash"]) if "_tok_hash" in data else None
                has_required = "reg_target" in data and "va_values" in data
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
        }

        if cache_dir:
            np.savez_compressed(
                _token_cache_path(s["symbol"] + "_v2", cache_dir),
                token_ids=token_ids, fine_ids=fine_ids,
                day=day[:ci], month=month[:ci], year=year[:ci],
                va_values=va_normed, reg_target=reg_target, _tok_hash=tok_hash or "",
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
        d_list, m_list, y_list = enc["day"].tolist(), enc["month"].tolist(), enc["year"].tolist()

        # BOS/EOS use 0 as placeholder for fine (never used in loss)
        ids = torch.tensor([bos_id] + ids_list + [eos_id], dtype=torch.long)
        fine = torch.tensor([0] + fine_list + [0], dtype=torch.long)
        d = torch.tensor([d_list[0]] + d_list + [d_list[-1]], dtype=torch.long)
        m = torch.tensor([m_list[0]] + m_list + [m_list[-1]], dtype=torch.long)
        y = torch.tensor([y_list[0]] + y_list + [y_list[-1]], dtype=torch.long)
        va = torch.tensor(
            [zero_va_list] + va_list + [zero_va_list], dtype=torch.float32)
        reg_targets = torch.tensor([-999.0] + rt_list + [-999.0], dtype=torch.float32)

        # Truncate long sequences from the beginning (keep most recent data)
        if max_seq_len > 0 and len(ids) > max_seq_len:
            ids = torch.cat([ids[:1], ids[-(max_seq_len-1):]])
            fine = torch.cat([fine[:1], fine[-(max_seq_len-1):]])
            d = torch.cat([d[:1], d[-(max_seq_len-1):]])
            m = torch.cat([m[:1], m[-(max_seq_len-1):]])
            y = torch.cat([y[:1], y[-(max_seq_len-1):]])
            va = torch.cat([va[:1], va[-(max_seq_len-1):]], dim=0)
            reg_targets = torch.cat([reg_targets[:1], reg_targets[-(max_seq_len-1):]])

        sequences.append({
            "input_ids": ids,
            "targets": ids[1:].clone(),
            "fine_targets": fine[1:].clone(),  # shifted fine IDs for dual-head
            "time_ids": torch.stack([d, m, y], dim=-1),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
            "va_values": va,
            "reg_targets": reg_targets,
        })

    return sequences


class PackedDatasetV2(Dataset):
    """Dataset: returns sequence dict (mask built in collate)."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_dataloader_v2(sequences, batch_size=1, shuffle=True):
    """DataLoader for single-sequence batches. Returns 8-tuple matching _pad_batch format."""
    def collate(batch):
        s = batch[0]
        S = s["input_ids"].shape[0]
        mask = _build_causal_mask(S)
        return (
            s["input_ids"],
            s["targets"],
            s.get("fine_targets", torch.zeros_like(s["targets"])),
            s["time_ids"],
            s["position_ids"],
            mask,
            s["va_values"],
            s["reg_targets"],
        )
    return DataLoader(
        PackedDatasetV2(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True,
    )
