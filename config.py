"""Kronos-R-Preview 全局配置。"""
import os
import json
import random

import numpy as np
import torch


# ============================================================================
# Config classes
# ============================================================================

class NormConfig:
    # --- per-stock historical normalize ---
    # Price features (OHLC): historical Z-Score (stats from full train history)
    # Volume/Amount: log1p → first-day baseline → Z-Score
    price_features: list = None   # set below after class
    va_features: list = None      # set below after class
    # Minimum days of history required for a stock to be included
    min_lookback: int = 20
    min_doc_length: int = 30


NormConfig.price_features = ["log_ret", "log_high", "log_low", "log_open"]
NormConfig.va_features = ["log_vol", "log_amt"]


class DataConfig:
    data_dir: str = "dataset/"
    cutoff_date: str = "2024-02-01"
    context_len: int = 8192
    train_ratio: float = 0.875
    max_stocks: int = 0          # 0 = all
    # RAW feature layout (6 columns) loaded by `load_stocks`.
    # The pipeline tokenizes only the first 4 columns (OHLC) — see
    # TokenizerConfig.input_dim = 4 below — and feeds the last 2 columns (log_vol,
    # log_amt) to the model as continuous VA embeddings (no quantization).
    feature_cols: list = None    # set below after class
    random_seed: int = 42


DataConfig.feature_cols = [
    "log_ret", "log_high", "log_low", "log_open", "log_vol", "log_amt",
]


class TokenizerConfig:
    input_dim: int = 4           # v2: OHLC only (was 6 with VA)
    hidden_dim: int = 192
    embedding_dim: int = 48
    num_quantizers: int = 2
    bits_per_quantizer: int = 10  # single int → all layers same; list → per-layer
    bits_l1: int = 0              # >0 overrides coarse layer bits
    bits_l2: int = 0              # >0 overrides fine layer bits
    bsq_commitment_cost: float = 0.194
    bsq_entropy_weight: float = 0.01
    epochs: int = 100
    random_seed: int = 42
    learning_rate: float = 1e-4
    batch_size: int = 8192       # v2: was 512, now 8x larger for GPU saturation
    num_workers: int = 2
    grad_clip: float = 1.0
    save_path: str = "checkpoints/tokenizer_v2_ohlc.pt"


class ModelConfig:
    dim: int = 256
    depth: int = 2
    heads: int = 4
    num_kv_heads: int = 1
    position_encoding: str = "rope"
    rope_base: float = 10000.0
    dropout: float = 0.1
    vocab_size: int = 1024       # coarse vocab (GPT prediction target)
    vocab_fine: int = 256        # fine vocab (dual-head auxiliary)
    ffn_multiplier: int = 4
    va_hidden_dim: int = 64     # v2: Volume/Amount MLP hidden dim


class TrainingConfig:
    epochs: int = 10
    batch_size: int = 4             # batch_size=1 (stable; batching needs more VRAM)
    accumulation_steps: int = 8     # was 16 — halved for faster updates
    num_workers: int = 2            # was 0 — parallel data loading
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    use_gradient_checkpointing: bool = False  # was True — minimal benefit for 2 layers
    random_seed: int = 42
    max_train_updates: int = 0
    save_dir: str = "checkpoints"
    tokenizer_path: str = "checkpoints/tokenizer_v2_ohlc.pt"
    # Production GPT checkpoint. HPO 2026-06-18 best (phase3_t000, DA 48.12% with V2
    # calibration) trains with focal γ=4 + heteroscedastic=ON, 10 epochs. The old
    # expA_v2.pt (γ=6, ls=0.05) is kept for backward compatibility but is no longer
    # the recommended baseline.
    base_model_path: str = "checkpoints/expA_v2_hpo.pt"
    token_cache_dir: str = "checkpoints/token_cache"  # NEW: pre-tokenize cache


# ============================================================================
# Reproducibility
# ============================================================================

def set_global_seed(seed, deterministic=True):
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """Worker init function for DataLoader (ensures reproducibility)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Runtime overrides
# ============================================================================

def _apply_runtime_overrides():
    path = os.environ.get("KRONOS_PREVIEW_OVERRIDE_JSON", "").strip()
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    config_map = {
        "NormConfig": NormConfig,
        "DataConfig": DataConfig,
        "TokenizerConfig": TokenizerConfig,
        "ModelConfig": ModelConfig,
        "TrainingConfig": TrainingConfig,
    }
    for name, overrides in payload.items():
        target = config_map.get(name)
        if target is None:
            raise KeyError(f"Unknown config: {name}")
        for key, value in overrides.items():
            setattr(target, key, value)


_apply_runtime_overrides()
