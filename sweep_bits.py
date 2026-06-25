"""Bit-width Sweep: systematically scan quantizer bit configurations.

Trains 10 tokenizers (L1≥L2, bits 6-9) + corresponding GPT models,
then evaluates each with GPT-only inference. Results saved to JSON.

Supports resume: skips steps where checkpoints already exist.

Usage:
    python sweep_bits.py                               # all 10 configs
    python sweep_bits.py --configs "7+6,8+7"            # subset
    python sweep_bits.py --tok_epochs 50 --gpt_epochs 5 # fast preview
    python sweep_bits.py --skip_eval                     # train only
    python sweep_bits.py --early_stop_patience 15        # tokenizer early stopping
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch

from config import DataConfig, ModelConfig, TokenizerConfig, set_global_seed


# ═══════════════════════════════════════════════════════════════════
#  Inline training utilities
# ═══════════════════════════════════════════════════════════════════

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

# 10 configurations: L1 >= L2, bits 6-9
ALL_CONFIGS = [
    (6, 6), (7, 6), (7, 7), (8, 6), (8, 7), (8, 8),
    (9, 6), (9, 7), (9, 8), (9, 9),
]

SWEEP_DIR = "checkpoints/sweep"


def tok_path(l1, l2, seed=42):
    return os.path.join(SWEEP_DIR, f"seed{seed}", f"bits_{l1}_{l2}_tok.pt")


def gpt_path(l1, l2, seed=42):
    return os.path.join(SWEEP_DIR, f"seed{seed}", f"bits_{l1}_{l2}_gpt.pt")


def results_path(seed=42):
    return os.path.join(SWEEP_DIR, f"results_seed{seed}.json")


def load_results(seed=42):
    p = results_path(seed)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_results(results, seed=42):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(results_path(seed), "w") as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Step 1: Train tokenizer
# ============================================================================

def train_tokenizer(l1, l2, tok_epochs, device, seed=42,
                    early_stop_patience=0, use_scheduler=False):
    """Train tokenizer for given bit config. Returns path to checkpoint."""
    path = tok_path(l1, l2, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  [skip] Tokenizer exists: {path}")
        return path

    print(f"  Training tokenizer: L1={l1}, L2={l2}, epochs={tok_epochs}")

    # Import and configure
    from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs, export_tokenizer_config
    from data_processor import load_stocks, split_stocks, get_tokenizer_features_v2

    # Save and restore config to prevent cross-config pollution
    saved_bits_l1 = getattr(TokenizerConfig, "bits_l1", 0)
    saved_bits_l2 = getattr(TokenizerConfig, "bits_l2", 0)
    saved_bpq = TokenizerConfig.bits_per_quantizer
    TokenizerConfig.bits_l1 = l1
    TokenizerConfig.bits_l2 = l2
    TokenizerConfig.bits_per_quantizer = 0  # force use of l1/l2

    # Early stopping
    early_stop = None
    if early_stop_patience > 0:
        early_stop = EarlyStopping(patience=early_stop_patience, min_delta=1e-5)

    # Load data
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s

    n_tv = len(tv_stocks)
    n_val_stocks = max(1, int(n_tv * 0.05))
    rng = __import__("numpy").random.RandomState(seed)
    perm = rng.permutation(n_tv)
    tok_train = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
    tok_val = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]

    train_feat = get_tokenizer_features_v2(tok_train, cutoff_date=DataConfig.cutoff_date)
    val_feat = get_tokenizer_features_v2(tok_val, cutoff_date=DataConfig.cutoff_date)

    # Load to GPU
    train_gpu = torch.from_numpy(train_feat).to(device)
    val_gpu = torch.from_numpy(val_feat).to(device)

    # Build tokenizer
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)
    opt = torch.optim.Adam(tok.parameters(), lr=TokenizerConfig.learning_rate)

    # Optional scheduler
    scheduler = None
    if use_scheduler:
        bs = TokenizerConfig.batch_size
        n_steps = len(train_gpu) // bs
        total_steps = n_steps * tok_epochs
        scheduler = build_tokenizer_scheduler(opt, total_steps, warmup_frac=0.05)

    bs = TokenizerConfig.batch_size
    N = len(train_gpu)
    n_steps = N // bs
    static_input = torch.empty(bs, 4, device=device, dtype=torch.float32)

    # Compile with cudagraphs
    if hasattr(torch, "compile"):
        try:
            tok_compiled = torch.compile(tok, backend="cudagraphs")
        except Exception:
            tok_compiled = tok
    else:
        tok_compiled = tok

    # Warmup
    tok_compiled.train()
    for _ in range(5):
        idx = torch.randint(0, N, (bs,), device=device)
        static_input.copy_(train_gpu[idx])
        loss = tok_compiled(static_input)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    raw_tok = tok._orig_mod if hasattr(tok, "_orig_mod") else tok
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(tok_epochs):
        tok_compiled.train()
        for step in range(n_steps):
            idx = torch.randint(0, N, (bs,), device=device)
            static_input.copy_(train_gpu[idx])
            opt.zero_grad(set_to_none=True)
            loss = tok_compiled(static_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()

        cur_lr = opt.param_groups[0]["lr"]

        # Validate every 10 epochs or last
        if (epoch + 1) % 10 == 0 or epoch == tok_epochs - 1:
            tok.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.inference_mode():
                for i in range(0, len(val_gpu), bs):
                    batch = val_gpu[i:i+bs]
                    if batch.shape[0] != bs:
                        pad = torch.zeros(bs - batch.shape[0], 4, device=device)
                        batch = torch.cat([batch, pad], dim=0)
                    val_loss_sum += tok(batch).item()
                    val_count += 1
            val_loss = val_loss_sum / val_count

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": raw_tok.state_dict(),
                    "config": export_tokenizer_config(),
                    "best_val_loss": best_val,
                    "bits_l1": l1, "bits_l2": l2,
                }, path)
            print(f"    Epoch {epoch+1}: val={val_loss:.4f} best={best_val:.4f} "
                  f"lr={cur_lr:.2e} {time.time()-t0:.0f}s")

            # Early stopping check
            if early_stop and early_stop(val_loss, epoch):
                print(f"    Early stopping at epoch {epoch+1} (best={early_stop.best:.4f} "
                      f"at epoch {early_stop.best_epoch+1}, patience={early_stop.patience})")
                break

    print(f"  Tokenizer saved: {path} (best_val={best_val:.4f})")

    # Restore config
    TokenizerConfig.bits_l1 = saved_bits_l1
    TokenizerConfig.bits_l2 = saved_bits_l2
    TokenizerConfig.bits_per_quantizer = saved_bpq
    return path


# ============================================================================
# Step 2: Train GPT
# ============================================================================

def train_gpt(l1, l2, tokenizer_path, gpt_epochs, device, seed=42):
    """Train GPT model with Muon+AdamW. Returns path to checkpoint."""
    path = gpt_path(l1, l2, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  [skip] GPT exists: {path}")
        return path

    print(f"  Training GPT: epochs={gpt_epochs}")

    from train_base import main as train_main
    from config import TrainingConfig, ModelConfig

    TrainingConfig.batch_size = 1
    TrainingConfig.accumulation_steps = 32
    TrainingConfig.random_seed = seed  # propagate seed to GPT training

    # Set vocab sizes from tokenizer for dual-head architecture
    from model import load_tokenizer as _load_tok
    _tok = _load_tok(tokenizer_path, device)
    ModelConfig.vocab_size = _tok.vocab_coarse
    ModelConfig.vocab_fine = _tok.bsq_fine.vocab_size
    del _tok

    args = type("Args", (), {
        "save_path": path,
        "tokenizer_path": tokenizer_path,
        "epochs": gpt_epochs,
        "tag": f"sweep_{l1}_{l2}",
        "loss": "focal",
        "weight_decay": 0.01,
        "lr": 3e-4,
        "optimizer": "muon",
        "lr_muon": 0.02,
        "light_eval": True,
        "gamma": 4.0,
        "label_smoothing": 0.0,
        "entropy_alpha": 0.0,
        "heteroscedastic": True,
        "het_weight": 0.1,
        "fine_weight": 0.3,
        "reasoning": False,
        "reasoning_frozen": False,
        "base_checkpoint": None,
        "dropout": 0.1,
        "max_stocks": 0,
        "max_seq_len": 0,   # no limit — long context is core design
        "force_repack": False,
        "history_per_epoch": False,
        "early_stop_patience": 0,
    })()

    train_main(args)
    return path


# ============================================================================
# Step 3: Evaluate
# ============================================================================

def evaluate_config(l1, l2, gpt_ckpt, tokenizer_ckpt, device, n_stocks=2400):
    """Run GPT evaluation (multi-day windowed). Returns metrics dict."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from eval_windowed import evaluate_windowed as _win_eval
    print(f"  Evaluating ({n_stocks} stocks, window=20 days)...")
    metrics = _win_eval(gpt_ckpt, tokenizer_ckpt, device,
                        n_stocks=n_stocks, n_days=20, batch_size=4, silent=False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ============================================================================
# Main orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bit-width Sweep")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated configs (e.g. '7+6,8+7'). Empty = all.")
    parser.add_argument("--tok_epochs", type=int, default=100)
    parser.add_argument("--gpt_epochs", type=int, default=10)
    parser.add_argument("--n_stocks", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_eval", action="store_true")
    # ── Training improvement args ──
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience for tokenizer training (0=disabled, recommended 15-20)")
    parser.add_argument("--scheduler", action="store_true", default=False,
                        help="Enable warmup+cosine LR scheduler for tokenizer training")
    args = parser.parse_args()

    seed = args.seed
    os.makedirs(SWEEP_DIR, exist_ok=True)
    set_global_seed(seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse configs
    if args.configs:
        configs = []
        for c in args.configs.split(","):
            l1, l2 = c.strip().split("+")
            configs.append((int(l1), int(l2)))
    else:
        configs = ALL_CONFIGS

    results = load_results(seed)
    print(f"Device: {device}, Seed: {seed}")
    print(f"Configs: {len(configs)}, tok_epochs={args.tok_epochs}, gpt_epochs={args.gpt_epochs}")
    if args.early_stop_patience > 0 or args.scheduler:
        print(f"Improvements: early_stop_patience={args.early_stop_patience}, "
              f"scheduler={args.scheduler}")
    print(f"Results so far: {len(results)} configs completed")
    print()

    t_total = time.time()

    for i, (l1, l2) in enumerate(configs):
        key = f"{l1}+{l2}"
        if key in results:
            print(f"[{i+1}/{len(configs)}] {key}: already done -- DA={results[key]['da']*100:.2f}%")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] Config: L1={l1}, L2={l2}  "
              f"(joint_vocab={2**l1 * 2**l2})")
        print(f"{'='*60}")

        t_config = time.time()

        # Step 1: Tokenizer
        tok_ckpt = train_tokenizer(
            l1, l2, args.tok_epochs, device, seed=seed,
            early_stop_patience=args.early_stop_patience,
            use_scheduler=args.scheduler,
        )

        # Step 2: GPT
        gpt_ckpt = train_gpt(l1, l2, tok_ckpt, args.gpt_epochs, device, seed=seed)

        # Step 3: Evaluate
        if args.skip_eval:
            metrics = {"da": 0, "mape": 999, "ampratio": 0,
                       "collapse_rate": 1.0, "n_unique_tokens": 0,
                       "rank_ic": 0, "n_predictions": 0,
                       "skipped": True}
        else:
            metrics = evaluate_config(l1, l2, gpt_ckpt, tok_ckpt, device,
                                      n_stocks=args.n_stocks)

        metrics["bits_l1"] = l1
        metrics["bits_l2"] = l2
        metrics["joint_vocab"] = 2**l1 * 2**l2
        metrics["time_s"] = time.time() - t_config
        results[key] = metrics
        save_results(results, seed)

        print(f"  Config {key} done in {metrics['time_s']:.0f}s: "
              f"avgDA={metrics['avg_da_per_date']*100:.2f}% "
              f"xBaseline={metrics['avg_da_above_baseline']*100:+.2f}% "
              f"Coll={metrics['collapse_rate']*100:.1f}% Uniq={metrics['n_unique_tokens']}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS (multi-day windowed, avg_da_per_date)")
    print(f"{'='*70}")
    print(f"{'Config':<10} {'Vocab':>8} {'avgDA%':>8} {'xBase%':>8} {'AmpR':>8} "
          f"{'Coll%':>8} {'Uniq':>6} {'RankIC':>8} {'Time':>6}")
    print("-" * 70)

    for key in sorted(results.keys(), key=lambda k: results[k].get("avg_da_per_date", 0), reverse=True):
        r = results[key]
        print(f"{key:<10} {r.get('joint_vocab', 0):>8} "
              f"{r['avg_da_per_date']*100:>7.2f}% {r['avg_da_above_baseline']*100:>+7.2f}% "
              f"{r['ampratio']:>7.3f}x "
              f"{r['collapse_rate']*100:>7.1f}% {r['n_unique_tokens']:>6} "
              f"{r.get('rank_ic', 0):>8.4f} {r.get('time_s', 0):>5.0f}s")

    # Best config
    best = max(results.keys(), key=lambda k: results[k].get("avg_da_per_date", 0))
    print(f"\nBest: {best} (avgDA={results[best]['avg_da_per_date']*100:.2f}%)")
    print(f"Total time: {time.time()-t_total:.0f}s")
    print(f"Results: {results_path()}")


if __name__ == "__main__":
    main()
