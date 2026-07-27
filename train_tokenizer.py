"""Stage A: Train BSQ Tokenizer — CUDA Graphs optimized.

Core optimizations:
  - CUDA Graphs: capture forward+backward+gradient clipping; keep Adam eager
    so the update trajectory remains bit-identical to the historical trainer
  - GPU-resident data: entire training set on GPU, zero CPU-GPU transfer
  - In-place forward: add_() for graph-compatible accumulation
  - Feature caching: skip repeat normalization on re-runs
  - Early stopping: patience-based to avoid over-training
  - LR scheduler: warmup + cosine decay for better convergence

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --bits_l1 7 --bits_l2 6
    python train_tokenizer.py --val_every 10 --epochs 100
    python train_tokenizer.py --early_stop_patience 15
    python train_tokenizer.py --scheduler
"""
import argparse
import os
import random
import sys
import time

if os.name != "nt":
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
from training_utils import (
    FixedBatchCudaGraphStep,
    clip_grad_norm_,
    optimizer_lr,
)


class EarlyStopping:
    """Early stopping with patience."""
    def __init__(self, patience=15, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, metric, epoch=0):
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_tokenizer_scheduler(optimizer, total_steps, warmup_frac=0.05,
                               min_lr_ratio=0.1):
    """Warmup + cosine LR scheduler."""
    import math
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Feature cache
# ============================================================================

_FEATURE_CACHE_DIR = os.environ.get(
    "KRONOS_TOKENIZER_FEATURE_CACHE_DIR", "checkpoints/feature_cache"
)


def _feature_cache_path(tag, cutoff_date):
    os.makedirs(_FEATURE_CACHE_DIR, exist_ok=True)
    return os.path.join(_FEATURE_CACHE_DIR, f"tok_feat_{tag}_{cutoff_date}.npz")


def _atomic_torch_save(payload, path):
    """Write a checkpoint without exposing a partially written target."""
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(checkpoint):
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )


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
    # Include the final short batch directly; no padding is introduced.
    losses = []
    counts = []
    with torch.inference_mode():
        for start in range(0, N, bs):
            batch = val_feat_gpu[start : start + bs]
            if len(batch) == 0:
                continue
            loss = tok(batch)
            losses.append(loss.detach())
            counts.append(len(batch))
    tok.train()
    values = torch.stack(losses).cpu().tolist() if losses else []
    return sum(
        value * count for value, count in zip(values, counts)
    ) / max(sum(counts), 1)


def _validate_loader(tok, val_loader, device, use_amp=False,
                     amp_dtype=torch.float32):
    tok.eval()
    losses = []
    counts = []
    with torch.inference_mode():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=use_amp):
                loss = tok(batch)
            losses.append(loss.detach())
            counts.append(len(batch))
    tok.train()
    values = torch.stack(losses).cpu().tolist() if losses else []
    return sum(
        value * count for value, count in zip(values, counts)
    ) / max(sum(counts), 1)


# ============================================================================
# CUDA Graph training
# ============================================================================

def _train_cuda_graph(tok, train_feat_gpu, val_feat_gpu, bs, epochs, val_every,
                      save_path, ckpt_path, lr, grad_clip, device,
                      early_stop=None, use_scheduler=False):
    """Native fixed-shape CUDA graph training (including Windows)."""
    N = len(train_feat_gpu)
    n_steps = N // bs
    print(f"  Native CUDA Graph: {n_steps} steps/epoch, bs={bs}")

    params = list(tok.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Optional LR scheduler
    scheduler = None
    if use_scheduler:
        total_steps = n_steps * epochs
        scheduler = build_tokenizer_scheduler(optimizer, total_steps, warmup_frac=0.05)
        print(f"  LR scheduler: warmup+cosine ({total_steps} total steps)")

    # Resume a coherent model/optimizer/scheduler snapshot before compilation.
    # RNG is restored after the dry warmup so continuation stays reproducible.
    start_epoch, best_val = 0, float("inf")
    resume_checkpoint = None
    if os.path.exists(ckpt_path):
        resume_checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        tok.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(
                resume_checkpoint["scheduler_state_dict"]
            )
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val = resume_checkpoint.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    # Pre-allocate static buffer
    static_input = torch.empty(bs, 4, device=device, dtype=torch.float32)

    if resume_checkpoint is not None:
        _restore_rng_state(resume_checkpoint)

    raw_tok = tok
    tok.train()
    static_input.copy_(train_feat_gpu[:bs])
    train_loss_acc = torch.zeros((), device=device)
    print("  Capturing forward + backward + gradient clipping...")
    graph_step = FixedBatchCudaGraphStep.capture(
        tok,
        optimizer,
        params,
        loss_closure=lambda: tok(static_input),
        grad_clip=grad_clip,
        loss_accumulator=train_loss_acc,
    )
    print("  CUDA graph captured; Adam remains on the exact eager path.")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        train_loss_acc.zero_()

        for step in range(n_steps):
            idx = torch.randint(0, N, (bs,), device=device)  # O(1) vs randperm O(N)
            static_input.copy_(train_feat_gpu[idx])

            graph_step.replay()
            if scheduler:
                scheduler.step()

        train_loss = (train_loss_acc / n_steps).item()
        cur_lr = optimizer_lr(optimizer)

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
        resume_payload = {
            "model_state_dict": sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            **_rng_state(),
        }
        if scheduler:
            resume_payload["scheduler_state_dict"] = scheduler.state_dict()
        _atomic_torch_save(resume_payload, ckpt_path)
        if improved:
            _atomic_torch_save(
                {
                    "model_state_dict": sd,
                    "config": export_tokenizer_config(),
                    "best_val_loss": best_val,
                    "best_epoch": epoch,
                    "total_epochs": epochs,
                    "completed": epoch == epochs - 1,
                },
                save_path,
            )

        log_interval = max(1, epochs // 6)
        if do_val and ((epoch + 1) % log_interval == 0 or epoch == start_epoch or (epoch + 1) == epochs):
            print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} "
                  f"lr={cur_lr:.2e} {time.time()-t0:.0f}s")

        # Early stopping
        if early_stop and do_val and early_stop(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch+1} (best={early_stop.best:.4f} "
                  f"at epoch {early_stop.best_epoch+1}, patience={early_stop.patience})")
            break

    return best_val


# ============================================================================
# Fallback: standard training (no CUDA graph)
# ============================================================================

def _train_standard(tok, train_loader, val_loader, epochs, val_every,
                    save_path, ckpt_path, lr, grad_clip, device,
                    early_stop=None, use_scheduler=False):
    optimizer = torch.optim.Adam(tok.parameters(), lr=lr)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    use_amp = amp_dtype != torch.float32
    scheduler = None
    if use_scheduler:
        scheduler = build_tokenizer_scheduler(
            optimizer, max(1, len(train_loader) * epochs), warmup_frac=0.05
        )

    t0 = time.time()
    best_val = float("inf")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        tok.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", float("inf"))
        _restore_rng_state(checkpoint)

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
            clip_grad_norm_(tok.parameters(), grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            loss_acc += loss.detach()
            count += 1

        train_loss = (loss_acc / max(count, 1)).item()
        do_val = (epoch + 1) % val_every == 0 or (epoch + 1) == epochs
        val_loss = (
            _validate_loader(
                raw_tok, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            if do_val
            else best_val
        )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        resume_payload = {
            "model_state_dict": raw_tok.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            **_rng_state(),
        }
        if scheduler:
            resume_payload["scheduler_state_dict"] = scheduler.state_dict()
        _atomic_torch_save(resume_payload, ckpt_path)
        if improved:
            _atomic_torch_save(
                {
                    "model_state_dict": raw_tok.state_dict(),
                    "config": export_tokenizer_config(),
                    "best_val_loss": best_val,
                    "best_epoch": epoch,
                    "total_epochs": epochs,
                    "completed": epoch == epochs - 1,
                },
                save_path,
            )

        log_interval = max(1, epochs // 6)
        if do_val and ((epoch + 1) % log_interval == 0 or epoch == start_epoch or (epoch + 1) == epochs):
            print(f"  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} "
                  f"{time.time()-t0:.0f}s")

        if early_stop and do_val and early_stop(val_loss, epoch):
            print(
                f"  Early stopping at epoch {epoch+1} "
                f"(best={early_stop.best:.4f} at epoch "
                f"{early_stop.best_epoch+1}, patience={early_stop.patience})"
            )
            break

    return best_val


# ============================================================================
# Main
# ============================================================================

def main(args=None):
    global _FEATURE_CACHE_DIR
    seed = (
        getattr(args, "seed", TokenizerConfig.random_seed)
        if args else TokenizerConfig.random_seed
    )
    TokenizerConfig.random_seed = int(seed)
    DataConfig.random_seed = int(seed)
    if args and getattr(args, "feature_cache_dir", ""):
        _FEATURE_CACHE_DIR = os.path.abspath(args.feature_cache_dir)
    set_global_seed(seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs = args.batch_size if args else TokenizerConfig.batch_size
    epochs = args.epochs if args else TokenizerConfig.epochs
    save_path = args.save_path if args else TokenizerConfig.save_path
    val_every = args.val_every if args else 5
    use_cuda_graph = args.cuda_graph if args else True
    early_stop_patience = getattr(args, "early_stop_patience", 0)
    use_scheduler = getattr(args, "scheduler", False)
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

    # Early stopping
    early_stop = None
    if early_stop_patience > 0:
        early_stop = EarlyStopping(patience=early_stop_patience, min_delta=1e-5)

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
    if len(train_feat) == 0 or len(val_feat) == 0:
        raise RuntimeError("Tokenizer train/validation feature split is empty")
    if bs > len(train_feat):
        bs = len(train_feat)
        print(f"  Adjusted batch_size to {bs} for available training features")

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
            early_stop=early_stop,
            use_scheduler=use_scheduler,
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
                                   device=device,
                                   early_stop=early_stop,
                                   use_scheduler=use_scheduler)

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if not ckpt.get("completed", False):
            ckpt["completed"] = True
            ckpt["total_epochs"] = epochs
            _atomic_torch_save(ckpt, save_path)

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
    # ── Training improvement args ──
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Early stopping patience (0=disabled, recommended 15-20)")
    p.add_argument("--scheduler", action="store_true", default=False,
                   help="Enable warmup+cosine LR scheduler")
    p.add_argument("--embedding_dim", type=int, default=0,
                   help="Override TokenizerConfig.embedding_dim (0=use config default)")
    p.add_argument("--hidden_dim", type=int, default=0,
                   help="Override TokenizerConfig.hidden_dim (0=use config default)")
    p.add_argument("--tag", type=str, default="",
                   help="Tag appended to save_path for distinguishing runs")
    p.add_argument("--seed", type=int, default=TokenizerConfig.random_seed)
    p.add_argument(
        "--feature_cache_dir",
        type=str,
        default="",
        help="Isolated tokenizer feature-cache directory",
    )
    parsed = p.parse_args()
    if parsed.bits_l1 > 0:
        TokenizerConfig.bits_l1 = parsed.bits_l1
    if parsed.bits_l2 > 0:
        TokenizerConfig.bits_l2 = parsed.bits_l2
    if parsed.bits_per_quantizer > 0:
        TokenizerConfig.bits_per_quantizer = parsed.bits_per_quantizer
    if parsed.embedding_dim > 0:
        TokenizerConfig.embedding_dim = parsed.embedding_dim
    if parsed.hidden_dim > 0:
        TokenizerConfig.hidden_dim = parsed.hidden_dim
    if parsed.tag:
        base, ext = os.path.splitext(parsed.save_path)
        parsed.save_path = f"{base}_{parsed.tag}{ext}"
    main(parsed)
