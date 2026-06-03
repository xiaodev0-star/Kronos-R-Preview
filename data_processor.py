"""数据处理：CSV 加载、滚动归一化、多股票打包。"""
from glob import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig, NormConfig

lookback_window = NormConfig.lookback_window
min_lookback = NormConfig.min_lookback


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


def get_tokenizer_features(stocks, window=lookback_window):
    """对每只股票做滚动归一化，拼接为 [N_total, 6]。"""
    parts = []
    for s in tqdm(stocks, desc="Rolling normalize"):
        parts.append(rolling_normalize(s["features_raw"], window))
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
