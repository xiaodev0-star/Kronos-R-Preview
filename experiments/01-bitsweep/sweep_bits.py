"""Bit-width Sweep: systematically scan quantizer bit configurations.

Trains 10 tokenizers (L1≥L2, bits 6-9) + corresponding GPT models,
then evaluates each with GPT-only inference. Results saved to JSON.

Supports resume: skips steps where checkpoints already exist.

Usage:
    python sweep_bits.py                               # all 10 configs
    python sweep_bits.py --configs "7+6,8+7"            # subset
    python sweep_bits.py --tok_epochs 50 --gpt_epochs 30 # custom epochs
    python sweep_bits.py --skip_eval                     # train only
    python sweep_bits.py --ablation "7+7"                 # 1-epoch ablation study
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

from config import DataConfig, ModelConfig, TokenizerConfig, set_global_seed


# ═══════════════════════════════════════════════════════════════════
#  Inline training utilities
# ═══════════════════════════════════════════════════════════════════

class TrainingProfiler:
    """Lightweight profiler: manual timing + CUDA memory tracking.

    Usage:
        prof = TrainingProfiler()
        prof.start("data_load")
        ... do work ...
        prof.end("data_load")
        prof.start("gpt_train")
        prof.record("gpt_forward", 0.05)  # accumulate sub-timings
        prof.end("gpt_train")
        prof.report()  # prints + saves JSON
    """
    def __init__(self):
        self._timings = {}
        self._starts = {}
        self._gpu_mem = {}

    def start(self, phase):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[phase] = time.perf_counter()

    def end(self, phase):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._starts.pop(phase, time.perf_counter())
        self._timings[phase] = self._timings.get(phase, 0.0) + elapsed
        if torch.cuda.is_available():
            self._gpu_mem[phase] = torch.cuda.max_memory_allocated() / 1e6

    def record(self, phase, seconds):
        """Accumulate a sub-timing (e.g. forward pass sampled every 20 steps)."""
        self._timings[phase] = self._timings.get(phase, 0.0) + seconds

    def gpu_snapshot(self, phase):
        """Capture current GPU memory stats for a phase."""
        if torch.cuda.is_available():
            self._gpu_mem[phase] = torch.cuda.max_memory_allocated() / 1e6

    def report(self, save_path=None):
        """Print summary and optionally save JSON."""
        total = sum(self._timings.values())
        print(f"\n{'='*60}")
        print("PROFILER REPORT")
        print(f"{'='*60}")
        sorted_phases = sorted(self._timings.items(), key=lambda x: -x[1])
        for phase, secs in sorted_phases:
            pct = secs / total * 100 if total > 0 else 0
            mem = self._gpu_mem.get(phase, 0)
            mem_str = f"  GPU_peak={mem:.0f}MB" if mem > 0 else ""
            print(f"  {phase:<25} {secs:>8.2f}s  ({pct:>5.1f}%){mem_str}")
        print(f"  {'TOTAL':<25} {total:>8.2f}s")
        print(f"{'='*60}")

        # Top 5 bottlenecks
        print("\nTop 5 bottlenecks:")
        for i, (phase, secs) in enumerate(sorted_phases[:5]):
            pct = secs / total * 100 if total > 0 else 0
            print(f"  {i+1}. {phase}: {secs:.2f}s ({pct:.1f}%)")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            data = {"total_s": total, "phases": {}}
            for phase, secs in self._timings.items():
                entry = {"seconds": round(secs, 3), "pct": round(secs / total * 100, 1) if total > 0 else 0}
                if phase in self._gpu_mem:
                    entry["gpu_peak_mb"] = round(self._gpu_mem[phase], 1)
                data["phases"][phase] = entry
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Profiler report saved: {save_path}")
        return self._timings

    def reset(self):
        self._timings.clear()
        self._starts.clear()
        self._gpu_mem.clear()


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

# Feature cache directory (shared across all configs since cutoff_date is the same)
_FEATURE_CACHE_DIR = os.path.join(SWEEP_DIR, "feature_cache")


def _load_or_compute_tok_features(stocks_train, stocks_val, cutoff_date, cache_dir=None):
    """Compute or load cached tokenizer features (shared across bit configs)."""
    from data_processor import get_tokenizer_features_v2
    cache_dir = cache_dir or _FEATURE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, f"tok_train_{cutoff_date}.npz")
    val_cache = os.path.join(cache_dir, f"tok_val_{cutoff_date}.npz")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        train_feat = np.load(train_cache)["features"]
        val_feat = np.load(val_cache)["features"]
        print(f"  [cache] Loaded tok features: train={train_feat.shape}, val={val_feat.shape}")
        return train_feat, val_feat

    train_feat = get_tokenizer_features_v2(stocks_train, cutoff_date=cutoff_date)
    val_feat = get_tokenizer_features_v2(stocks_val, cutoff_date=cutoff_date)
    np.savez(train_cache, features=train_feat)
    np.savez(val_cache, features=val_feat)
    print(f"  [cache] Saved tok features: train={train_feat.shape}, val={val_feat.shape}")
    return train_feat, val_feat


def train_tokenizer(l1, l2, tok_epochs, device, seed=42,
                    early_stop_patience=15, use_scheduler=True,
                    precomputed_features=None, profiler=None):
    """Train tokenizer for given bit config. Returns path to checkpoint.

    Args:
        precomputed_features: (train_feat, val_feat) numpy arrays, or None to compute fresh.
        profiler: optional TrainingProfiler instance.
    """
    path = tok_path(l1, l2, seed)
    ckpt_path = path + ".ckpt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  [skip] Tokenizer exists: {path}")
        return path

    print(f"  Training tokenizer: L1={l1}, L2={l2}, epochs={tok_epochs}, "
          f"scheduler={use_scheduler}, early_stop={early_stop_patience}")

    from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs, export_tokenizer_config

    # Save and restore config to prevent cross-config pollution
    saved_bits_l1 = getattr(TokenizerConfig, "bits_l1", 0)
    saved_bits_l2 = getattr(TokenizerConfig, "bits_l2", 0)
    saved_bpq = TokenizerConfig.bits_per_quantizer
    TokenizerConfig.bits_l1 = l1
    TokenizerConfig.bits_l2 = l2
    TokenizerConfig.bits_per_quantizer = 0

    # Early stopping
    early_stop = None
    if early_stop_patience > 0:
        early_stop = EarlyStopping(patience=early_stop_patience, min_delta=1e-5)

    # Use precomputed features or compute fresh
    if precomputed_features is not None:
        train_feat, val_feat = precomputed_features
    else:
        from data_processor import load_stocks, split_stocks
        stocks = load_stocks(max_stocks=DataConfig.max_stocks)
        train_s, val_s, _ = split_stocks(stocks)
        tv_stocks = train_s + val_s
        n_tv = len(tv_stocks)
        n_val_stocks = max(1, int(n_tv * 0.05))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_tv)
        tok_train = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
        tok_val = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]
        train_feat, val_feat = _load_or_compute_tok_features(
            tok_train, tok_val, DataConfig.cutoff_date)

    # Load to GPU
    train_gpu = torch.from_numpy(train_feat).to(device)
    val_gpu = torch.from_numpy(val_feat).to(device)

    # Reset seed before model init + training — ensures each config gets identical
    # RNG state (weight init, batch sampling), making sweep results comparable.
    set_global_seed(seed, deterministic=False)

    # Build tokenizer
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs()).to(device)
    opt = torch.optim.Adam(tok.parameters(), lr=TokenizerConfig.learning_rate)

    # Scheduler (default enabled for 100-epoch training)
    scheduler = None
    bs = TokenizerConfig.batch_size
    N = len(train_gpu)
    n_steps = N // bs
    if use_scheduler:
        total_steps = n_steps * tok_epochs
        scheduler = build_tokenizer_scheduler(opt, total_steps, warmup_frac=0.05)

    static_input = torch.empty(bs, 4, device=device, dtype=torch.float32)

    # torch.compile(backend="cudagraphs") — disabled on Windows (Jinja2/triton errors)
    tok_compiled = tok
    if sys.platform != "win32" and hasattr(torch, "compile"):
        try:
            tok_compiled = torch.compile(tok, backend="cudagraphs")
        except Exception:
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
    if profiler:
        profiler.start("tok_train")

    for epoch in range(tok_epochs):
        tok_compiled.train()
        train_loss_acc = 0.0
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
            train_loss_acc += loss.detach()

        cur_lr = opt.param_groups[0]["lr"]
        train_loss = (train_loss_acc / n_steps).item()

        # Validate every 10 epochs or last
        val_every = 10
        if (epoch + 1) % val_every == 0 or epoch == tok_epochs - 1:
            tok.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.inference_mode():
                # Iterate only full batches — skip last incomplete batch to avoid
                # zero-padding rows diluting the reconstruction loss.
                n_full_batches = len(val_gpu) // bs
                for i in range(n_full_batches):
                    batch = val_gpu[i * bs : (i + 1) * bs]
                    val_loss_sum += tok(batch).item()
                    val_count += 1
            val_loss = val_loss_sum / max(val_count, 1)

            # Save resume checkpoint every validation epoch
            sd = raw_tok.state_dict()
            torch.save({"model_state_dict": sd,
                        "optimizer_state_dict": opt.state_dict(),
                        "epoch": epoch, "best_val": best_val}, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": sd,
                    "config": export_tokenizer_config(),
                    "best_val_loss": best_val,
                    "bits_l1": l1, "bits_l2": l2,
                }, path)

            print(f"    Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"best={best_val:.4f} lr={cur_lr:.2e} {time.time()-t0:.0f}s")

            # Early stopping check
            if early_stop and early_stop(val_loss, epoch):
                print(f"    Early stopping at epoch {epoch+1} (best={early_stop.best:.4f} "
                      f"at epoch {early_stop.best_epoch+1}, patience={early_stop.patience})")
                break

    if profiler:
        profiler.end("tok_train")

    print(f"  Tokenizer saved: {path} (best_val={best_val:.4f})")

    # Restore config
    TokenizerConfig.bits_l1 = saved_bits_l1
    TokenizerConfig.bits_l2 = saved_bits_l2
    TokenizerConfig.bits_per_quantizer = saved_bpq
    return path


# ============================================================================
# Step 2: Train GPT
# ============================================================================

def train_gpt(l1, l2, tokenizer_path, gpt_epochs, device, seed=42, profiler=None):
    """Train GPT model with Muon+AdamW. Returns path to checkpoint."""
    path = gpt_path(l1, l2, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  [skip] GPT exists: {path}")
        return path

    print(f"  Training GPT: epochs={gpt_epochs}, batch_size=1, accum=32→64 (ep15+), no early_stop")

    from train_base import main as train_main
    from config import TrainingConfig, ModelConfig

    TrainingConfig.batch_size = 1       # single-seq fastest for variable-length stocks
    TrainingConfig.accumulation_steps = 32  # effective batch = 32
    TrainingConfig.warmup_ratio = 0.02  # shorter warmup (~1 epoch, was 0.05)
    TrainingConfig.random_seed = seed

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
        "max_seq_len": 0,
        "force_repack": False,
        "history_per_epoch": False,
        "early_stop_patience": 0,  # no early stopping — let full 40 epochs complete
        "profiler": profiler,
        "curriculum": True,  # enable curriculum learning
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
    print(f"  Evaluating ({n_stocks} stocks, window=20 days, batch_size=2)...")
    metrics = _win_eval(gpt_ckpt, tokenizer_ckpt, device,
                        n_stocks=n_stocks, n_days=20, batch_size=2, silent=False)
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
    parser.add_argument("--gpt_epochs", type=int, default=40,
                        help="GPT training epochs (default 40, no early stop)")
    parser.add_argument("--n_stocks", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_eval", action="store_true")
    # ── Training improvement args ──
    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="Early stopping patience for tokenizer (default 15)")
    parser.add_argument("--no_scheduler", action="store_true", default=False,
                        help="Disable warmup+cosine LR scheduler for tokenizer")
    # ── Ablation mode ──
    parser.add_argument("--ablation", type=str, default="",
                        help="Run 1-epoch ablation study on specified config (e.g. '7+7')")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable performance profiler")
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

    # Ablation mode: run 1-epoch experiments with individual optimizations
    if args.ablation:
        l1, l2 = [int(x) for x in args.ablation.strip().split("+")]
        _run_ablation(l1, l2, args, device, seed)
        return

    results = load_results(seed)
    use_scheduler = not args.no_scheduler
    print(f"Device: {device}, Seed: {seed}")
    print(f"Configs: {len(configs)}, tok_epochs={args.tok_epochs}, gpt_epochs={args.gpt_epochs}")
    print(f"Tokenizer: scheduler={use_scheduler}, early_stop={args.early_stop_patience}")
    print(f"GPT: batch_size=1, accum=32→64(ep15+), no early_stop, curriculum=True")
    print(f"Results so far: {len(results)} configs completed")
    print()

    # Profiler
    profiler = TrainingProfiler() if args.profile else None

    # ── Shared data loading (once for all configs) ──
    profiler_data = profiler
    if profiler_data:
        profiler_data.start("data_load")

    from data_processor import load_stocks, split_stocks
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    tv_stocks = train_s + val_s

    if profiler_data:
        profiler_data.end("data_load")
        profiler_data.start("feature_extract")

    # Compute tokenizer features once (shared across all bit configs)
    n_tv = len(tv_stocks)
    n_val_stocks = max(1, int(n_tv * 0.05))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_tv)
    tok_train = [tv_stocks[i] for i in sorted(perm[n_val_stocks:])]
    tok_val = [tv_stocks[i] for i in sorted(perm[:n_val_stocks])]
    precomputed_features = _load_or_compute_tok_features(
        tok_train, tok_val, DataConfig.cutoff_date)

    if profiler_data:
        profiler_data.end("feature_extract")

    t_total = time.time()

    for i, (l1, l2) in enumerate(configs):
        key = f"{l1}+{l2}"
        if key in results:
            print(f"[{i+1}/{len(configs)}] {key}: already done -- DA={results[key].get('avg_da_per_date', 0)*100:.2f}%")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] Config: L1={l1}, L2={l2}  "
              f"(joint_vocab={2**l1 * 2**l2})")
        print(f"{'='*60}")

        t_config = time.time()

        # Step 1: Tokenizer (with shared features + profiler)
        tok_ckpt = train_tokenizer(
            l1, l2, args.tok_epochs, device, seed=seed,
            early_stop_patience=args.early_stop_patience,
            use_scheduler=use_scheduler,
            precomputed_features=precomputed_features,
            profiler=profiler,
        )

        # Step 2: GPT (batch_size=4, 30 epochs, early stop, curriculum)
        gpt_ckpt = train_gpt(l1, l2, tok_ckpt, args.gpt_epochs, device, seed=seed,
                             profiler=profiler)

        # Step 3: Evaluate
        if profiler:
            profiler.start("eval")

        if args.skip_eval:
            metrics = {"avg_da_per_date": 0, "avg_da_above_baseline": 0,
                       "ampratio": 0, "collapse_rate": 1.0, "n_unique_tokens": 0,
                       "rank_ic": 0, "n_predictions": 0, "skipped": True}
        else:
            metrics = evaluate_config(l1, l2, gpt_ckpt, tok_ckpt, device,
                                      n_stocks=args.n_stocks)

        if profiler:
            profiler.end("eval")

        metrics["bits_l1"] = l1
        metrics["bits_l2"] = l2
        metrics["joint_vocab"] = 2**l1 * 2**l2
        metrics["time_s"] = time.time() - t_config
        results[key] = metrics
        save_results(results, seed)

        print(f"  Config {key} done in {metrics['time_s']:.0f}s: "
              f"avgDA={metrics.get('avg_da_per_date', 0)*100:.2f}% "
              f"xBaseline={metrics.get('avg_da_above_baseline', 0)*100:+.2f}% "
              f"Coll={metrics.get('collapse_rate', 0)*100:.1f}% Uniq={metrics.get('n_unique_tokens', 0)}")

    # Profiler final report
    if profiler:
        profiler.report(save_path=os.path.join(SWEEP_DIR, "profiler_report.json"))

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
              f"{r.get('avg_da_per_date', 0)*100:>7.2f}% {r.get('avg_da_above_baseline', 0)*100:>+7.2f}% "
              f"{r.get('ampratio', 0):>7.3f}x "
              f"{r.get('collapse_rate', 0)*100:>7.1f}% {r.get('n_unique_tokens', 0):>6} "
              f"{r.get('rank_ic', 0):>8.4f} {r.get('time_s', 0):>5.0f}s")

    # Best config
    if results:
        best = max(results.keys(), key=lambda k: results[k].get("avg_da_per_date", 0))
        print(f"\nBest: {best} (avgDA={results[best].get('avg_da_per_date', 0)*100:.2f}%)")
    print(f"Total time: {time.time()-t_total:.0f}s")
    print(f"Results: {results_path()}")


# ============================================================================
# Ablation study: 1-epoch experiments to measure per-optimization impact
# ============================================================================

def _run_ablation(l1, l2, args, device, seed):
    """Run 1-epoch ablation: baseline vs each optimization, on the same config."""
    from data_processor import load_stocks, split_stocks
    from train_base import main as train_main
    from config import TrainingConfig, ModelConfig
    from model import load_tokenizer as _load_tok

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: Config L1={l1}, L2={l2}")
    print(f"{'='*60}")

    # Ensure tokenizer exists
    tok_ckpt = tok_path(l1, l2, seed)
    if not os.path.exists(tok_ckpt):
        print("Training tokenizer first...")
        train_tokenizer(l1, l2, args.tok_epochs, device, seed=seed,
                        early_stop_patience=args.early_stop_patience,
                        use_scheduler=not args.no_scheduler)

    # Set vocab sizes
    _tok = _load_tok(tok_ckpt, device)
    ModelConfig.vocab_size = _tok.vocab_coarse
    ModelConfig.vocab_fine = _tok.bsq_fine.vocab_size
    del _tok

    ablation_dir = os.path.join(SWEEP_DIR, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    results = {}

    ablation_configs = [
        ("baseline", {"batch_size": 1, "accumulation_steps": 32, "curriculum": False}),
        ("batch4", {"batch_size": 4, "accumulation_steps": 8, "curriculum": False}),
        ("curriculum", {"batch_size": 1, "accumulation_steps": 32, "curriculum": True}),
    ]

    for name, opts in ablation_configs:
        print(f"\n--- Ablation: {name} ---")
        save_p = os.path.join(ablation_dir, f"abl_{l1}_{l2}_{name}.pt")

        TrainingConfig.batch_size = opts["batch_size"]
        TrainingConfig.accumulation_steps = opts["accumulation_steps"]
        TrainingConfig.random_seed = seed

        ab_args = type("Args", (), {
            "save_path": save_p,
            "tokenizer_path": tok_ckpt,
            "epochs": 1,
            "tag": f"abl_{l1}_{l2}_{name}",
            "loss": "focal", "weight_decay": 0.01, "lr": 3e-4,
            "optimizer": "muon", "lr_muon": 0.02,
            "light_eval": True, "gamma": 4.0,
            "label_smoothing": 0.0, "entropy_alpha": 0.0,
            "heteroscedastic": True, "het_weight": 0.1, "fine_weight": 0.3,
            "reasoning": False, "reasoning_frozen": False, "base_checkpoint": None,
            "dropout": 0.1, "max_stocks": 0, "max_seq_len": 0,
            "force_repack": False, "history_per_epoch": False,
            "early_stop_patience": 0, "profiler": None,
            "curriculum": opts["curriculum"],
        })()

        t0 = time.time()
        # Remove old checkpoint if exists for clean run
        for p in [save_p, save_p + ".ckpt"]:
            if os.path.exists(p):
                os.remove(p)
        train_main(ab_args)
        elapsed = time.time() - t0

        # Load history to get train/val loss
        hist_path = os.path.join(os.path.dirname(save_p) or ".",
                                 f"history_abl_{l1}_{l2}_{name}.json")
        hist = {}
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)

        results[name] = {
            "opts": opts,
            "elapsed_s": round(elapsed, 2),
            "train_loss": hist.get("train_loss", [None])[-1],
            "val_loss": hist.get("val_loss", [None])[-1],
            "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0,
        }
        print(f"  {name}: elapsed={elapsed:.1f}s, "
              f"train_loss={results[name]['train_loss']}, "
              f"val_loss={results[name]['val_loss']}, "
              f"gpu={results[name]['gpu_peak_mb']}MB")

    # Save ablation report
    report_path = os.path.join(ablation_dir, "ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION REPORT")
    print(f"{'='*60}")
    print(f"{'Name':<15} {'Elapsed':>8} {'TrainLoss':>10} {'ValLoss':>10} {'GPU_MB':>8}")
    print("-" * 55)
    for name, r in results.items():
        tl = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "N/A"
        vl = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "N/A"
        print(f"{name:<15} {r['elapsed_s']:>7.1f}s {tl:>10} {vl:>10} {r['gpu_peak_mb']:>7.0f}")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
