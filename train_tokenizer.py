"""Stage A: 训练 BSQ Tokenizer on train+val data (no test data)."""
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import DataConfig, TokenizerConfig, NormConfig
from data_processor import load_stocks, split_stocks, get_tokenizer_features
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs, export_tokenizer_config
from reproducibility import set_global_seed


def main():
    set_global_seed(TokenizerConfig.random_seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all stocks, split to get train+val (exclude test)
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s
    print(f"Train: {len(train_s)}, Val: {len(val_s)}, Total train+val: {len(tv_stocks)}")

    # Get features from train+val stocks
    features = get_tokenizer_features(tv_stocks)
    print(f"Total feature vectors: {features.shape}")

    # Hold out 5% for monitoring
    n = len(features)
    n_val = max(1, int(n * 0.05))
    indices = np.random.permutation(n)
    train_feat = features[indices[n_val:]]
    val_feat = features[indices[:n_val]]
    print(f"Tokenizer train: {len(train_feat)}, val: {len(val_feat)}")

    train_ds = TensorDataset(torch.from_numpy(train_feat))
    val_ds = TensorDataset(torch.from_numpy(val_feat))
    train_loader = DataLoader(train_ds, batch_size=TokenizerConfig.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TokenizerConfig.batch_size,
                            shuffle=False, pin_memory=True)

    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)
    print(f"Tokenizer params: {sum(p.numel() for p in tok.parameters()):,}")

    optimizer = torch.optim.Adam(tok.parameters(), lr=TokenizerConfig.learning_rate)

    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(TokenizerConfig.epochs):
        tok.train()
        train_losses = []
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TokenizerConfig.epochs}", leave=False):
            batch = batch.to(device)
            loss = tok(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), TokenizerConfig.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        tok.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                loss = tok(batch)
                val_losses.append(loss.item())

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        avg_val = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - t0

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in tok.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} {elapsed:.0f}s")

    # Save best
    if best_state is not None:
        tok.load_state_dict(best_state)

    os.makedirs(os.path.dirname(TokenizerConfig.save_path), exist_ok=True)
    torch.save({
        "model_state_dict": tok.state_dict(),
        "config": export_tokenizer_config(),
        "best_val_loss": best_val,
    }, TokenizerConfig.save_path)
    print(f"\nSaved tokenizer to {TokenizerConfig.save_path} (best val={best_val:.4f})")


if __name__ == "__main__":
    main()
