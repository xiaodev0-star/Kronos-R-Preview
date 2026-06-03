"""Stage A: Train BSQ Tokenizer on train+val data.
Supports epoch-level checkpoint/resume."""
import argparse
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

from config import DataConfig, TokenizerConfig
from data_processor import load_stocks, split_stocks, get_tokenizer_features
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs, export_tokenizer_config
from reproducibility import set_global_seed


def main(args=None):
    set_global_seed(TokenizerConfig.random_seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = args.batch_size if args else TokenizerConfig.batch_size
    epochs = args.epochs if args else TokenizerConfig.epochs
    save_path = args.save_path if args else TokenizerConfig.save_path
    ckpt_path = save_path + ".ckpt"

    print(f"Device: {device}")
    print(f"  batch_size={bs}, epochs={epochs}, save={save_path}")

    # Data
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s
    print(f"Train: {len(train_s)}, Val: {len(val_s)}, TV: {len(tv_stocks)}")

    # Split by stock ID (not random feature vectors) to prevent temporal leakage
    n_tv = len(tv_stocks)
    n_val_stocks = max(1, int(n_tv * 0.05))
    rng = np.random.RandomState(DataConfig.random_seed)
    perm = rng.permutation(n_tv)
    tok_train_stocks = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
    tok_val_stocks = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]

    train_feat = get_tokenizer_features(tok_train_stocks, cutoff_date=DataConfig.cutoff_date)
    val_feat = get_tokenizer_features(tok_val_stocks, cutoff_date=DataConfig.cutoff_date)
    print(f"Feature vectors: train={train_feat.shape}, val={val_feat.shape}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_feat)),
                              batch_size=bs, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_feat)),
                            batch_size=bs, shuffle=False, pin_memory=True)

    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)
    optimizer = torch.optim.Adam(tok.parameters(), lr=TokenizerConfig.learning_rate)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    use_amp = amp_dtype != torch.float32
    if use_amp:
        print(f"  AMP: {amp_dtype}")

    # Resume
    start_epoch, best_val, best_state = 0, float("inf"), None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tok.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    t0 = time.time()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(start_epoch, epochs):
        tok.train()
        train_losses = []
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = tok(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), TokenizerConfig.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        tok.eval()
        val_losses = []
        with torch.inference_mode():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = tok(batch)
                val_losses.append(loss.item())

        avg_tr = sum(train_losses) / max(len(train_losses), 1)
        avg_va = sum(val_losses) / max(len(val_losses), 1)

        improved = avg_va < best_val
        if improved:
            best_val = avg_va

        # Save resume checkpoint
        torch.save({"model_state_dict": tok.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "best_val": best_val}, ckpt_path)

        # Save best model for downstream
        if improved:
            torch.save({"model_state_dict": tok.state_dict(),
                        "config": export_tokenizer_config(),
                        "best_val_loss": best_val,
                        "best_epoch": epoch,
                        "total_epochs": epochs,
                        "completed": epoch == epochs - 1}, save_path)

        log_iv = max(1, epochs // 6)
        if (epoch + 1) % log_iv == 0 or epoch == start_epoch or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1}: train={avg_tr:.4f} val={avg_va:.4f} best={best_val:.4f} "
                  f"{time.time()-t0:.0f}s")

    # Mark completed if not already
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if not ckpt.get("completed", False):
            ckpt["completed"] = True
            ckpt["total_epochs"] = epochs
            torch.save(ckpt, save_path)

    print(f"\nSaved tokenizer to {save_path} (best val={best_val:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=TokenizerConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TokenizerConfig.epochs)
    p.add_argument("--save_path", type=str, default=TokenizerConfig.save_path)
    main(p.parse_args())
