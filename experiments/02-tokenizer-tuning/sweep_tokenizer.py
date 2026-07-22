"""Tokenizer 超参扫描：embedding_dim × hidden_dim，只看重建损失。

扫描 6 个组合（embedding_dim ∈ {48,64,96} × hidden_dim ∈ {192,256}），
每个只训 tokenizer（~5-10 分钟），评估 val MAE/MSE + token 分布。

Usage:
    python sweep_tokenizer.py                    # 全部 6 个组合
    python sweep_tokenizer.py --configs "48x192,64x256"  # 指定子集
    python sweep_tokenizer.py --epochs 50        # 自定义 epoch 数

Output:
    checkpoints/tok_sweep_results.json
    checkpoints/tok_sweep_emb{X}_hid{Y}.pt
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from config import DataConfig, TokenizerConfig, set_global_seed

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════

BITS_L1, BITS_L2 = 8, 6
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "tok_sweep_results.json")

ALL_CONFIGS = [
    (48, 192), (48, 256),
    (64, 192), (64, 256),
    (96, 192), (96, 256),
]


def tok_save_path(emb, hid):
    return os.path.join(CHECKPOINT_DIR, f"tok_sweep_emb{emb}_hid{hid}.pt")


# ═══════════════════════════════════════════════════════════════════
#  Tokenizer 训练（复用 train_tokenizer.py 的逻辑）
# ═══════════════════════════════════════════════════════════════════

def train_one(emb, hid, train_feat, val_feat, epochs, device):
    """训练一个 tokenizer 配置，返回 save_path。"""
    from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs, export_tokenizer_config

    save_path = tok_save_path(emb, hid)
    ckpt_path = save_path + ".ckpt"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(save_path):
        print(f"  [skip] Already exists: {save_path}")
        return save_path

    # 设置 TokenizerConfig（每次重置，防止交叉污染）
    TokenizerConfig.embedding_dim = emb
    TokenizerConfig.hidden_dim = hid
    TokenizerConfig.bits_l1 = BITS_L1
    TokenizerConfig.bits_l2 = BITS_L2

    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)
    bs = TokenizerConfig.batch_size
    lr = TokenizerConfig.learning_rate
    grad_clip = TokenizerConfig.grad_clip

    train_gpu = torch.from_numpy(train_feat).to(device)
    val_gpu = torch.from_numpy(val_feat).to(device)
    N = len(train_gpu)
    n_steps = N // bs

    opt = torch.optim.Adam(tok.parameters(), lr=lr)

    # Scheduler
    from train_tokenizer import build_tokenizer_scheduler
    total_steps = n_steps * epochs
    scheduler = build_tokenizer_scheduler(opt, total_steps, warmup_frac=0.05)

    # Early stopping
    from train_tokenizer import EarlyStopping
    early_stop = EarlyStopping(patience=15, min_delta=1e-5)

    static_input = torch.empty(bs, 4, device=device, dtype=torch.float32)

    # Warmup
    tok.train()
    for _ in range(5):
        idx = torch.randint(0, N, (bs,), device=device)
        static_input.copy_(train_gpu[idx])
        loss = tok(static_input)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    best_val = float("inf")
    t0 = time.time()
    val_every = 10

    for epoch in range(epochs):
        tok.train()
        train_loss_acc = 0.0
        for step in range(n_steps):
            idx = torch.randint(0, N, (bs,), device=device)
            static_input.copy_(train_gpu[idx])
            opt.zero_grad(set_to_none=True)
            loss = tok(static_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            train_loss_acc += loss.detach()

        train_loss = (train_loss_acc / n_steps).item()

        # Validate
        do_val = (epoch + 1) % val_every == 0 or epoch == epochs - 1
        if do_val:
            tok.eval()
            val_loss_sum, val_count = 0.0, 0
            n_full = len(val_gpu) // bs
            with torch.inference_mode():
                for i in range(n_full):
                    batch = val_gpu[i * bs : (i + 1) * bs]
                    val_loss_sum += tok(batch).item()
                    val_count += 1
            val_loss = val_loss_sum / max(val_count, 1)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": tok.state_dict(),
                    "config": export_tokenizer_config(),
                    "best_val_loss": best_val,
                    "bits_l1": BITS_L1, "bits_l2": BITS_L2,
                    "embedding_dim": emb, "hidden_dim": hid,
                }, save_path)

            print(f"    Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"best={best_val:.4f} {time.time()-t0:.0f}s")

            if early_stop(val_loss, epoch):
                print(f"    Early stop at epoch {epoch+1} (best={early_stop.best:.4f})")
                break

    # Save resume checkpoint
    torch.save({"model_state_dict": tok.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch, "best_val": best_val}, ckpt_path)

    print(f"  Saved: {save_path} (best_val={best_val:.4f})")
    return save_path


# ═══════════════════════════════════════════════════════════════════
#  评估：重建 MAE/MSE + tokenizer 级 Unique/Collapse
# ═══════════════════════════════════════════════════════════════════

def evaluate_tokenizer(save_path, val_feat, device):
    """评估一个 tokenizer 的重建质量和 token 分布。"""
    from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs

    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(cfg)).to(device)
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.eval()

    val_gpu = torch.from_numpy(val_feat).float().to(device)
    bs = 8192
    N = len(val_gpu)

    all_mae, all_mse = [], []
    all_tokens = []

    with torch.inference_mode():
        for i in range(0, N, bs):
            batch = val_gpu[i:i+bs].unsqueeze(0)  # [1, bs, 4]
            all_idx = tok.encode_all(batch)          # [1, bs, 2]
            recon = tok.decode_all(all_idx)           # [1, bs, 4]
            diff = recon - batch
            all_mae.append(diff.abs().mean().item())
            all_mse.append(diff.pow(2).mean().item())
            # Coarse tokens for distribution analysis
            coarse_ids = all_idx[0, :, 0].cpu().numpy()
            all_tokens.append(coarse_ids)

    tokens = np.concatenate(all_tokens)
    unique, counts = np.unique(tokens, return_counts=True)
    collapse_rate = float(counts.max() / len(tokens))
    n_unique = int(len(unique))

    return {
        "recon_mae": float(np.mean(all_mae)),
        "recon_mse": float(np.mean(all_mse)),
        "tokenizer_unique": n_unique,
        "tokenizer_collapse": collapse_rate,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Tokenizer hyperparameter sweep")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated configs (e.g. '48x192,64x256'). Empty = all.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.configs:
        configs = []
        for c in args.configs.split(","):
            e, h = c.strip().split("x")
            configs.append((int(e), int(h)))
    else:
        configs = ALL_CONFIGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=False)
    print(f"Device: {device}, Seed: {args.seed}")
    print(f"Configs: {len(configs)}, epochs={args.epochs}")
    print(f"Bits: L1={BITS_L1}, L2={BITS_L2}")

    # ── 加载数据（共享） ──
    from data_processor import load_stocks, split_stocks, get_tokenizer_features_v2
    from train_tokenizer import _load_or_compute_features

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s

    n_tv = len(tv_stocks)
    n_val_stocks = max(1, int(n_tv * 0.05))
    rng = np.random.RandomState(DataConfig.random_seed)
    perm = rng.permutation(n_tv)
    tok_train = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
    tok_val = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]

    train_feat = _load_or_compute_features(tok_train, "train", DataConfig.cutoff_date)
    val_feat = _load_or_compute_features(tok_val, "val", DataConfig.cutoff_date)
    print(f"Features: train={train_feat.shape}, val={val_feat.shape}")

    results = load_results()
    t_total = time.time()

    for i, (emb, hid) in enumerate(configs):
        key = f"{emb}x{hid}"
        if key in results:
            print(f"\n[{i+1}/{len(configs)}] {key}: already done — "
                  f"MAE={results[key]['recon_mae']:.4f}")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] embedding_dim={emb}, hidden_dim={hid}")
        print(f"{'='*60}")

        t0 = time.time()
        save_path = train_one(emb, hid, train_feat, val_feat, args.epochs, device)
        metrics = evaluate_tokenizer(save_path, val_feat, device)
        metrics["time_s"] = time.time() - t0
        metrics["embedding_dim"] = emb
        metrics["hidden_dim"] = hid

        results[key] = metrics
        save_results(results)

        print(f"  Result: MAE={metrics['recon_mae']:.4f} MSE={metrics['recon_mse']:.4f} "
              f"Uniq={metrics['tokenizer_unique']} Coll={metrics['tokenizer_collapse']*100:.1f}% "
              f"({metrics['time_s']:.0f}s)")

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print("TOKENIZER SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<12} {'Emb':>4} {'Hid':>4} {'MAE':>8} {'MSE':>8} "
          f"{'Uniq':>5} {'Coll%':>6} {'Time':>6}")
    print("-" * 70)

    sorted_keys = sorted(results.keys(),
                         key=lambda k: results[k]["recon_mae"])
    for key in sorted_keys:
        r = results[key]
        print(f"{key:<12} {r['embedding_dim']:>4} {r['hidden_dim']:>4} "
              f"{r['recon_mae']:>8.4f} {r['recon_mse']:>8.4f} "
              f"{r['tokenizer_unique']:>5} {r['tokenizer_collapse']*100:>5.1f}% "
              f"{r['time_s']:>5.0f}s")

    # Best
    best_key = sorted_keys[0]
    best = results[best_key]
    print(f"\nBest: {best_key} (MAE={best['recon_mae']:.4f})")
    print(f"Baseline (48x192): MAE={results.get('48x192', {}).get('recon_mae', 'N/A')}")
    print(f"Total time: {time.time()-t_total:.0f}s")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
