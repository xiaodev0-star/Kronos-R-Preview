"""阶段二：轻量 GPT 验证 — 对比 embedding_dim=48 vs 64 的下游效果。

对两个最优 tokenizer（48x192 baseline 和 64x192 candidate）各训一个 GPT，
然后用 windowed eval 对比 DA / Collapse / Unique。

每个评估跑在独立子进程（内存完全隔离，避免 GPT 训练后内存碎片化导致 OOM）。

Usage:
    python sweep_tokenizer_validate.py

Output:
    checkpoints/tok_sweep_gpt_validation.json
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "tok_sweep_gpt_validation.json")

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════

VALIDATION_CONFIGS = [
    {
        "name": "baseline_48x192",
        "tok_path": os.path.join(CHECKPOINT_DIR, "tok_sweep_emb48_hid192.pt"),
        "embedding_dim": 48,
        "hidden_dim": 192,
        "description": "Baseline: embedding_dim=48 (BitSweep default)",
    },
    {
        "name": "candidate_64x192",
        "tok_path": os.path.join(CHECKPOINT_DIR, "tok_sweep_emb64_hid192.pt"),
        "embedding_dim": 64,
        "hidden_dim": 192,
        "description": "Best from sweep: embedding_dim=64 (MAE -14.5%)",
    },
]

GPT_MAX_STOCKS = 0
GPT_EPOCHS = 100
GPT_EARLY_STOP = 5
EVAL_STOCKS = 0
EVAL_DAYS = 999

PYTHON = sys.executable  # same Python interpreter


# ═══════════════════════════════════════════════════════════════════
#  GPT 训练（主进程，因为需要检查 checkpoint 是否存在）
# ═══════════════════════════════════════════════════════════════════

def train_gpt_if_needed(cfg):
    """如果 GPT checkpoint 不存在，训练一个。"""
    gpt_path = os.path.join(CHECKPOINT_DIR, f"tok_val_gpt_{cfg['name']}.pt")
    if os.path.exists(gpt_path):
        print(f"  [skip] GPT exists: {gpt_path}")
        return gpt_path

    import torch
    from config import DataConfig, ModelConfig, set_global_seed
    from model import load_tokenizer

    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = load_tokenizer(cfg["tok_path"], device)
    ModelConfig.vocab_size = tok.vocab_coarse
    ModelConfig.vocab_fine = tok.bsq_fine.vocab_size
    del tok

    saved_max_stocks = DataConfig.max_stocks
    DataConfig.max_stocks = GPT_MAX_STOCKS

    from train_base import main as train_main

    args = type("Args", (), {
        "save_path": gpt_path,
        "tokenizer_path": cfg["tok_path"],
        "epochs": GPT_EPOCHS,
        "tag": f"tok_val_{cfg['name']}",
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
        "max_stocks": GPT_MAX_STOCKS,
        "max_seq_len": 0,
        "force_repack": False,
        "history_per_epoch": False,
        "early_stop_patience": GPT_EARLY_STOP,
        "profiler": None,
        "curriculum": True,
    })()

    t0 = time.time()
    train_main(args)
    print(f"  GPT trained in {time.time()-t0:.0f}s")

    DataConfig.max_stocks = saved_max_stocks
    return gpt_path


# ═══════════════════════════════════════════════════════════════════
#  评估（子进程，内存完全隔离）
# ═══════════════════════════════════════════════════════════════════

def evaluate_in_subprocess(cfg, gpt_path):
    """在独立子进程中运行 windowed eval，返回 metrics dict。"""
    result_path = os.path.join(CHECKPOINT_DIR, f"_eval_{cfg['name']}.json")

    # 子进程脚本：加载模型、评估、写结果到 JSON
    script = f"""
import json, os, sys, torch
os.chdir({_PROJECT_ROOT!r})
sys.path.insert(0, {_PROJECT_ROOT!r})
from config import ModelConfig, set_global_seed
from model import load_tokenizer
from eval_helpers import load_gpt, attach_close_prices, batched_gpt_eval_windowed, compute_windowed_metrics
from data_processor import load_stocks, split_stocks

set_global_seed(42, deterministic=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tok = load_tokenizer({cfg['tok_path']!r}, device)
ModelConfig.vocab_size = tok.vocab_coarse
ModelConfig.vocab_fine = tok.bsq_fine.vocab_size
gpt = load_gpt({gpt_path!r}, device, tokenizer=tok)

stocks = load_stocks(max_stocks=0)
_, _, test_stocks = split_stocks(stocks)
attach_close_prices(test_stocks)

preds = batched_gpt_eval_windowed(
    gpt, tok, test_stocks, device,
    batch_size=2, n_days={EVAL_DAYS}, silent=False)
metrics = compute_windowed_metrics(preds)

# Save only the keys we need
result = {{
    'da': metrics.get('avg_da_per_date', 0),
    'da_std': metrics.get('da_std', 0),
    'da_above_baseline': metrics.get('avg_da_above_baseline', 0),
    'collapse_rate': metrics.get('collapse_rate', 0),
    'n_unique_tokens': metrics.get('n_unique_tokens', 0),
    'rank_ic': metrics.get('rank_ic', 0),
    'ampratio': metrics.get('ampratio', 0),
    'n_predictions': metrics.get('n_predictions', 0),
}}
with open({result_path!r}, 'w') as f:
    json.dump(result, f)
print(f"Saved: {result_path}")
"""

    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=False,  # show output in real-time
        text=True,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [ERROR] Evaluation subprocess failed (exit code {proc.returncode})")
        return None

    if not os.path.exists(result_path):
        print(f"  [ERROR] Result file not found: {result_path}")
        return None

    with open(result_path) as f:
        metrics = json.load(f)
    os.remove(result_path)  # cleanup temp file
    metrics["total_time_s"] = elapsed
    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"GPT: all stocks, epochs cap={GPT_EPOCHS}, early_stop={GPT_EARLY_STOP}")
    print(f"Eval: all test stocks, all available days (subprocess isolation)")
    print(f"Configs: {[c['name'] for c in VALIDATION_CONFIGS]}")
    print()

    results = {}
    t_total = time.time()

    for cfg in VALIDATION_CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"[{name}] {cfg['description']}")
        print(f"  tokenizer: {cfg['tok_path']}")
        print(f"{'='*60}")

        if not os.path.exists(cfg["tok_path"]):
            print(f"  [ERROR] Tokenizer not found: {cfg['tok_path']}")
            continue

        # Step 1: Train GPT (or skip if exists)
        print(f"\n  Training GPT...")
        gpt_path = train_gpt_if_needed(cfg)

        # Step 2: Evaluate in subprocess (memory isolated)
        print(f"\n  Evaluating in subprocess...")
        metrics = evaluate_in_subprocess(cfg, gpt_path)

        if metrics is None:
            print(f"  [SKIP] Evaluation failed for {name}")
            continue

        result = {
            "name": name,
            "embedding_dim": cfg["embedding_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "tok_path": cfg["tok_path"],
            "gpt_path": gpt_path,
            **metrics,
        }
        results[name] = result

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n  [{name}] Results:")
        print(f"    DA:            {result['da']*100:.2f}% (std={result['da_std']*100:.1f}%)")
        print(f"    DA above base: {result['da_above_baseline']*100:+.2f}%")
        print(f"    Collapse:      {result['collapse_rate']*100:.1f}%")
        print(f"    Unique:        {result['n_unique_tokens']}")
        print(f"    RankIC:        {result['rank_ic']:.4f}")
        print(f"    AmpRatio:      {result['ampratio']:.3f}")
        print(f"    Time:          {result['total_time_s']:.0f}s")

    # ── 对比 ──
    if len(results) == 2:
        names = list(results.keys())
        r0, r1 = results[names[0]], results[names[1]]

        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {names[0]:>16} {names[1]:>16} {'Delta':>10}")
        print("-" * 70)

        comparisons = [
            ("DA%", r0["da"]*100, r1["da"]*100, "%"),
            ("DA above base%", r0["da_above_baseline"]*100, r1["da_above_baseline"]*100, "%"),
            ("Collapse%", r0["collapse_rate"]*100, r1["collapse_rate"]*100, "pp"),
            ("Unique", r0["n_unique_tokens"], r1["n_unique_tokens"], ""),
            ("RankIC", r0["rank_ic"], r1["rank_ic"], ""),
            ("AmpRatio", r0["ampratio"], r1["ampratio"], ""),
        ]

        for label, v0, v1, unit in comparisons:
            delta = v1 - v0
            print(f"{label:<20} {v0:>15.4f} {v1:>15.4f} {delta:>+9.4f}")

        da_delta = r1["da"] - r0["da"]
        coll_delta = r1["collapse_rate"] - r0["collapse_rate"]

        print(f"\nDecision criteria (from plan-tokenizer.md):")
        print(f"  DA:     {r1['da']*100:.2f}% vs {r0['da']*100:.2f}% (delta={da_delta*100:+.2f}pp, threshold: ±1pp)")
        print(f"  Collapse: {r1['collapse_rate']*100:.1f}% vs {r0['collapse_rate']*100:.1f}% (delta={coll_delta*100:+.1f}pp, threshold: ±10pp)")

        if da_delta >= -0.01 and coll_delta <= 0.10:
            print(f"\n✅ PASS: candidate_64x192 adopted (DA within -1pp, Collapse within +10pp)")
        elif da_delta < -0.01:
            print(f"\n❌ FAIL: DA dropped >1pp — reject candidate_64x192, keep baseline_48x192")
        else:
            print(f"\n⚠️  BORDERLINE: needs full GPT training to confirm")

    print(f"\nTotal time: {time.time()-t_total:.0f}s")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
