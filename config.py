"""Kronos-R-Preview 全局配置。"""
import os
import json


class NormConfig:
    lookback_window: int = 252
    min_lookback: int = 20


class DataConfig:
    data_dir: str = "dataset/"
    cutoff_date: str = "2024-02-01"
    context_len: int = 8192
    train_ratio: float = 0.875
    max_stocks: int = 0          # 0 = all
    feature_cols: list = None    # set below after class
    random_seed: int = 42


DataConfig.feature_cols = [
    "log_ret", "log_high", "log_low", "log_open", "log_vol", "log_amt",
]


class TokenizerConfig:
    input_dim: int = 6
    hidden_dim: int = 192
    embedding_dim: int = 48
    num_quantizers: int = 2
    bits_per_quantizer: int = 10
    bsq_commitment_cost: float = 0.194
    bsq_entropy_weight: float = 0.01
    epochs: int = 100
    random_seed: int = 42
    learning_rate: float = 1e-4
    batch_size: int = 512
    grad_clip: float = 1.0
    save_path: str = "checkpoints/tokenizer.pt"


class ModelConfig:
    dim: int = 256
    depth: int = 2
    heads: int = 4
    num_kv_heads: int = 1
    position_encoding: str = "rope"
    rope_base: float = 10000.0
    dropout: float = 0.1
    vocab_size: int = 1024
    ffn_multiplier: int = 4


class TrainingConfig:
    epochs: int = 10
    batch_size: int = 1             # batch_size=1 (stable; batching needs more VRAM)
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
    tokenizer_path: str = "checkpoints/tokenizer.pt"
    base_model_path: str = "checkpoints/base_model.pt"
    token_cache_dir: str = "checkpoints/token_cache"  # NEW: pre-tokenize cache


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
