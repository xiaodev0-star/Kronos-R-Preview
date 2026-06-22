"""Stage A: Train BSQ Tokenizer — CUDA Graphs optimized.

Core optimizations:
  - CUDA Graphs: capture forward+backward+step as single replayed graph
  - GPU-resident data: entire training set on GPU, zero CPU-GPU transfer
  - In-place forward: add_() for graph-compatible accumulation
  - Feature caching: skip repeat normalization on re-runs

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --bits_l1 7 --bits_l2 6
    python train_tokenizer.py --val_every 10 --epochs 100
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import DataConfig, TokenizerConfig, set_global_seed
from data_processor import load_stocks, split_stocks, get_tokenizer_features_v2
from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs, export_tokenizer_config


# ============================================================================
# Feature cache
# ============================================================================

_FEATURE_CACHE_DIR = "checkpoints/feature_cache"


def _feature_cache_path(tag, cutoff_date):
    os.makedirs(_FEATURE_CACHE_DIR, exist_ok=True)
    return os.path.join(_FEATURE_CACHE_DIR, f"tok_feat_{tag}_{cutoff_date}.npz")


def _load_or_compute_features(stocks, tag, cutoff_date):
    cache_path = _feature_cache_path(tag, cutoff_date)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        feat = data["features"]
        print(f"  [cache] Loaded {tag} features: {feat.shape}")
        return feat
    feat = get_tokenizer_features_v2(stocks, cutoff_date=cutoff_date)
    np.savez(cache_path, features=feat)
    print(f"  [cache] Saved {tag} features: {feat.shape}")
    return feat


# ============================================================================
# Validation (no CUDA graph, runs outside graph context)
# ============================================================================

def _validate(tok, val_feat_gpu, bs, device):
    tok.eval()
    N = len(val_feat_gpu)
    n_batches = (N + bs - 1) // bs
    val_loss_sum = 0.0
    with torch.inference_mode():
        for i in range(n_batches):
            batch = val_feat_gpu[i * bs : (i + 1) * bs]
            if batch.shape[0] != bs:
                # Pad last batch to fixed size for consistency
                pad = torch.zeros(bs - batch.shape[0], *batch.shape[1:], device=device, dtype=batch.dtype)
                batch = torch.cat([batch, pad], dim=0)
            loss = tok(batch)
            val_loss_sum += loss.item()
    tok.train()
    return val_loss_sum / n_batches


# ============================================================================
# CUDA Graph training
# ============================================================================

def _train_cuda_graph(tok, train_feat_gpu, val_feat_gpu, bs, epochs, val_every,
                      save_path, ckpt_path, lr, grad_clip, device):
    """torch.compile(reduce-overhead) training — automatic CUDA graphs."""
    N = len(train_feat_gpu)
    n_steps = N // bs
    print(f"  Compiled mode: {n_steps} steps/epoch, bs={bs}")

    optimizer = torch.optim.Adam(tok.parameters(), lr=lr)

    # Pre-allocate static buffer
    static_input = torch.empty(bs, 4, device=device, dtype=torch.float32)

    # torch.compile with cudagraphs backend — uses CUDA graphs internally
    if hasattr(torch, "compile"):
        tok = torch.compile(tok, backend="cudagraphs")
        print("  torch.compile(backend='cudagraphs') enabled")

    # Warmup
    print("  Warming up (5 steps)...")
    tok.train()
    for _ in range(5):
        idx = torch.randint(0, N, (bs,), device=device)
        static_input.copy_(train_feat_gpu[idx])
        loss = tok(static_input)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("  Warmup done. Training...")

    # Resume
    start_epoch, best_val = 0, float("inf")
    raw_tok = tok._orig_mod if hasattr(tok, "_orig_mod") else tok
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_tok.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        train_loss_acc = 0.0

        for step in range(n_steps):
            idx = torch.randint(0, N, (bs,), device=device)  # O(1) vs randperm O(N)
            static_input.copy_(train_feat_gpu[idx])

            optimizer.zero_grad(set_to_none=True)
            loss = tok(static_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), grad_clip)
            optimizer.step()
            train_loss_acc += loss.detach()

        train_loss = (train_loss_acc / n_steps).item()

        do_val = (epoch + 1) % val_every == 0 or (epoch + 1) == epochs
        if do_val:
            val_loss = _validate(raw_tok, val_feat_gpu, bs, device)
        else:
            val_loss = best_val

        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        # Compute state_dict once per checkpoint save
        sd = raw_tok.state_dict()
        torch.save({"model_state_dict": sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "best_val": best_val}, ckpt_path)
        if improved:
            torch.save({"model_state_dict": sd,
                        "config": export_tokenizer_config(),
                        "best_val_loss": best_val,
                        "best_epoch": epoch,
                        "total_epochs": epochs,
                        "completed": epoch == epochs - 1}, save_path)

        log_interval = max(1, epochs // 6)
        if do_val and ((epoch + 1) % log_interval == 0 or epoch == start_epoch or (epoch + 1) == epochs):
            print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} "
                  f"{time.time()-t0:.0f}s")

    return best_val


# ============================================================================
# Fallback: standard training (no CUDA graph)
# ============================================================================

def _train_standard(tok, train_loader, val_loader, epochs, val_every,
                    save_path, ckpt_path, lr, grad_clip, device):
    optimizer = torch.optim.Adam(tok.parameters(), lr=lr)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    use_amp = amp_dtype != torch.float32

    t0 = time.time()
    best_val = float("inf")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tok.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    raw_tok = tok._orig_mod if hasattr(tok, "_orig_mod") else tok

    for epoch in range(start_epoch, epochs):
        tok.train()
        loss_acc = 0.0
        count = 0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = tok(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), grad_clip)
            optimizer.step()
            loss_acc += loss.detach()
            count += 1

        train_loss = (loss_acc / max(count, 1)).item()
        do_val = (epoch + 1) % val_every == 0 or (epoch + 1) == epochs
        val_loss = _validate(raw_tok, next(iter(val_loader))[0].to(device),
                             train_loader.batch_size, device) if do_val else best_val

        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        torch.save({"model_state_dict": raw_tok.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "best_val": best_val}, ckpt_path)
        if improved:
            torch.save({"model_state_dict": raw_tok.state_dict(),
                        "config": export_tokenizer_config(),
                        "best_val_loss": best_val,
                        "best_epoch": epoch,
                        "total_epochs": epochs,
                        "completed": epoch == epochs - 1}, save_path)

        log_interval = max(1, epochs // 6)
        if do_val and ((epoch + 1) % log_interval == 0 or epoch == start_epoch or (epoch + 1) == epochs):
            print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} "
                  f"{time.time()-t0:.0f}s")

    return best_val


# ============================================================================
# Main
# ============================================================================

def main(args=None):
    set_global_seed(TokenizerConfig.random_seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = args.batch_size if args else TokenizerConfig.batch_size
    epochs = args.epochs if args else TokenizerConfig.epochs
    save_path = args.save_path if args else TokenizerConfig.save_path
    val_every = args.val_every if args else 5
    use_cuda_graph = args.cuda_graph if args else True
    ckpt_path = save_path + ".ckpt"

    print(f"Device: {device}")
    print(f"  batch_size={bs}, epochs={epochs}, val_every={val_every}, cuda_graph={use_cuda_graph}")
    b1 = getattr(TokenizerConfig, "bits_l1", 0)
    b2 = getattr(TokenizerConfig, "bits_l2", 0)
    if b1 > 0 and b2 > 0:
        print(f"  bits: L1={b1}, L2={b2}, joint_vocab={2**b1 * 2**b2}")
    else:
        bpq = getattr(TokenizerConfig, "bits_per_quantizer", 10)
        print(f"  bits_per_quantizer={bpq}, vocab_per_layer={2**bpq}")

    # Data + feature cache
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s
    print(f"Train: {len(train_s)}, Val: {len(val_s)}, TV: {len(tv_stocks)}")

    n_tv = len(tv_stocks)
    n_val_stocks = max(1, int(n_tv * 0.05))
    rng = np.random.RandomState(DataConfig.random_seed)
    perm = rng.permutation(n_tv)
    tok_train_stocks = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
    tok_val_stocks = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]

    train_feat = _load_or_compute_features(tok_train_stocks, "train", DataConfig.cutoff_date)
    val_feat = _load_or_compute_features(tok_val_stocks, "val", DataConfig.cutoff_date)
    print(f"Feature vectors: train={train_feat.shape}, val={val_feat.shape}")

    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)

    # Check if CUDA graph is feasible
    can_cuda_graph = (
        use_cuda_graph
        and device.type == "cuda"
        and hasattr(torch.cuda, "CUDAGraph")
        and len(train_feat) >= bs
    )

    if can_cuda_graph:
        # Load entire dataset onto GPU (only ~40MB for 10M × 4 × float32)
        print("  Loading training data to GPU...")
        train_feat_gpu = torch.from_numpy(train_feat).to(device, non_blocking=True)
        val_feat_gpu = torch.from_numpy(val_feat).to(device, non_blocking=True)
        print(f"  GPU memory for data: {train_feat_gpu.nelement() * 4 / 1e6:.1f}MB + {val_feat_gpu.nelement() * 4 / 1e6:.1f}MB")

        best_val = _train_cuda_graph(
            tok, train_feat_gpu, val_feat_gpu,
            bs=bs, epochs=epochs, val_every=val_every,
            save_path=save_path, ckpt_path=ckpt_path,
            lr=TokenizerConfig.learning_rate,
            grad_clip=TokenizerConfig.grad_clip,
            device=device,
        )
    else:
        print("  Falling back to standard training loop")
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_feat)),
                                  batch_size=bs, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=TokenizerConfig.num_workers)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(val_feat)),
                                batch_size=bs, shuffle=False, pin_memory=True,
                                num_workers=TokenizerConfig.num_workers)
        best_val = _train_standard(tok, train_loader, val_loader,
                                   epochs=epochs, val_every=val_every,
                                   save_path=save_path, ckpt_path=ckpt_path,
                                   lr=TokenizerConfig.learning_rate,
                                   grad_clip=TokenizerConfig.grad_clip,
                                   device=device)

    # Mark completed
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
    p.add_argument("--save_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    p.add_argument("--bits_l1", type=int, default=0, help="Coarse layer bits (0=use config)")
    p.add_argument("--bits_l2", type=int, default=0, help="Fine layer bits (0=use config)")
    p.add_argument("--bits_per_quantizer", type=int, default=0, help="Override all layers (0=use config)")
    p.add_argument("--val_every", type=int, default=5, help="Validate every N epochs")
    p.add_argument("--cuda_graph", action="store_true", default=True, help="Use CUDA Graphs (default: on)")
    p.add_argument("--no_cuda_graph", dest="cuda_graph", action="store_false", help="Disable CUDA Graphs")
    parsed = p.parse_args()
    if parsed.bits_l1 > 0:
        TokenizerConfig.bits_l1 = parsed.bits_l1
    if parsed.bits_l2 > 0:
        TokenizerConfig.bits_l2 = parsed.bits_l2
    if parsed.bits_per_quantizer > 0:
        TokenizerConfig.bits_per_quantizer = parsed.bits_per_quantizer
    main(parsed)
