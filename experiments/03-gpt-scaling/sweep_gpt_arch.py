"""GPT 架构扫描：5 个架构全量训练 + 全量评估，支持断点续算。

断点逻辑：
  - 训练：train_base.py 自带 .ckpt resume，中断后重跑自动从上次 epoch 续算
  - 评估：结果写入 JSON，已有结果的 config 自动跳过
  - 状态：checkpoint.json 记录每个 config 的训练/评估完成状态

Usage:
    python sweep_gpt_arch.py                        # 全部 5 个架构
    python sweep_gpt_arch.py --configs "baseline,wide"  # 指定子集
    python sweep_gpt_arch.py --skip_eval            # 只训练不评估
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

import torch
from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed

PYTHON = sys.executable
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "gpt_arch_sweep_results.json")
STATE_PATH = os.path.join(CHECKPOINT_DIR, "gpt_arch_sweep_state.json")

# ═══════════════════════════════════════════════════════════════════
#  架构配置
# ═══════════════════════════════════════════════════════════════════

ARCH_CONFIGS = [
    {
        "name": "baseline",
        "dim": 256, "depth": 2, "heads": 4, "kv_heads": 1,
        "gradient_checkpointing": False, "max_seq_len": 0,
        "description": "2.7M baseline (current best)",
    },
    {
        "name": "wide",
        "dim": 384, "depth": 2, "heads": 6, "kv_heads": 1,
        "gradient_checkpointing": False, "max_seq_len": 0,
        "description": "6M pure width scaling",
    },
    {
        "name": "deep",
        "dim": 256, "depth": 4, "heads": 4, "kv_heads": 1,
        "gradient_checkpointing": True, "max_seq_len": 0,
        "description": "5M pure depth scaling",
    },
    {
        "name": "large",
        "dim": 384, "depth": 3, "heads": 6, "kv_heads": 1,
        "gradient_checkpointing": True, "max_seq_len": 0,
        "description": "10M width + depth",
    },
    {
        "name": "xlarge",
        "dim": 512, "depth": 4, "heads": 8, "kv_heads": 2,
        "gradient_checkpointing": True, "max_seq_len": 4096,
        "description": "25M extreme (may OOM)",
    },
]

CONFIG_MAP = {c["name"]: c for c in ARCH_CONFIGS}

# 训练参数
GPT_EPOCHS = 100
GPT_EARLY_STOP = 5


# ═══════════════════════════════════════════════════════════════════
#  状态管理（断点续算核心）
# ═══════════════════════════════════════════════════════════════════

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def gpt_save_path(name):
    return os.path.join(CHECKPOINT_DIR, f"gpt_arch_{name}.pt")


def gpt_ckpt_path(name):
    return gpt_save_path(name) + ".ckpt"


def eval_result_path(name):
    return os.path.join(CHECKPOINT_DIR, f"_eval_arch_{name}.json")


def is_trained(name, state):
    """检查训练是否完成：state 标记完成 或 best checkpoint 存在。"""
    if state.get(name, {}).get("train_status") == "completed":
        return True
    return os.path.exists(gpt_save_path(name))


def is_evaluated(name, results):
    """检查评估是否完成：results JSON 里有该 config 的完整数据。"""
    return name in results and "da" in results[name]


# ═══════════════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════════════

def train_arch(cfg, state):
    """训练一个架构配置，支持 resume。返回 save_path。"""
    name = cfg["name"]
    save_path = gpt_save_path(name)
    ckpt_path = gpt_ckpt_path(name)

    if is_trained(name, state):
        # 检查是否真的训练完了（有 best checkpoint）
        if os.path.exists(save_path):
            print(f"  [skip] Training completed: {save_path}")
            return save_path
        # state 说完成但文件不在，重训
        print(f"  [warn] State says completed but checkpoint missing, retraining")

    print(f"  Training: dim={cfg['dim']}, depth={cfg['depth']}, "
          f"heads={cfg['heads']}, kv={cfg['kv_heads']}, "
          f"gc={'ON' if cfg['gradient_checkpointing'] else 'OFF'}, "
          f"max_seq={cfg['max_seq_len'] or 'unlimited'}")

    # 检查是否可以 resume
    if os.path.exists(ckpt_path):
        print(f"  [resume] Found .ckpt: {ckpt_path}")
    else:
        print(f"  [fresh] No .ckpt found, starting from scratch")

    from train_base import main as train_main

    args = type("Args", (), {
        "save_path": save_path,
        "tokenizer_path": os.path.join(CHECKPOINT_DIR, "tok_sweep_emb64_hid192.pt"),
        "epochs": GPT_EPOCHS,
        "tag": f"arch_{name}",
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
        "max_seq_len": cfg["max_seq_len"],
        "force_repack": False,
        "history_per_epoch": False,
        "early_stop_patience": GPT_EARLY_STOP,
        "profiler": None,
        "curriculum": True,
        # Architecture overrides
        "dim": cfg["dim"],
        "depth": cfg["depth"],
        "heads": cfg["heads"],
        "num_kv_heads": cfg["kv_heads"],
        "ffn_multiplier": 0,
        "gradient_checkpointing": cfg["gradient_checkpointing"],
    })()

    t0 = time.time()
    train_main(args)
    elapsed = time.time() - t0

    # 更新状态
    state.setdefault(name, {})["train_status"] = "completed"
    state[name]["train_time_s"] = elapsed
    save_state(state)

    print(f"  Trained in {elapsed:.0f}s → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════
#  评估（子进程隔离）
# ═══════════════════════════════════════════════════════════════════

def evaluate_arch(cfg, gpt_path):
    """在子进程中评估，返回 metrics dict。"""
    name = cfg["name"]
    result_path = eval_result_path(name)
    tok_path = os.path.join(CHECKPOINT_DIR, "tok_sweep_emb64_hid192.pt")

    script = f"""
import json, os, sys, torch
os.chdir({os.getcwd()!r})
sys.path.insert(0, {os.getcwd()!r})
from config import ModelConfig, set_global_seed
from model import load_tokenizer
from eval_helpers import load_gpt, attach_close_prices, batched_gpt_eval_windowed, compute_windowed_metrics
from data_processor import load_stocks, split_stocks

set_global_seed(42, deterministic=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tok = load_tokenizer({tok_path!r}, device)
ModelConfig.vocab_size = tok.vocab_coarse
ModelConfig.vocab_fine = tok.bsq_fine.vocab_size
gpt = load_gpt({gpt_path!r}, device, tokenizer=tok)

stocks = load_stocks(max_stocks=0)
_, _, test_stocks = split_stocks(stocks)
attach_close_prices(test_stocks)

preds = batched_gpt_eval_windowed(
    gpt, tok, test_stocks, device,
    batch_size=2, n_days=999, silent=False)
metrics = compute_windowed_metrics(preds)

result = {{
    'da': metrics.get('avg_da_per_date', 0),
    'da_std': metrics.get('da_std', 0),
    'da_above_baseline': metrics.get('avg_da_above_baseline', 0),
    'collapse_rate': metrics.get('collapse_rate', 0),
    'n_unique_tokens': metrics.get('n_unique_tokens', 0),
    'rank_ic': metrics.get('rank_ic', 0),
    'ampratio': metrics.get('ampratio', 0),
    'mape': metrics.get('mape', 0),
    'baseline_mape': metrics.get('baseline_mape', 0),
    'n_predictions': metrics.get('n_predictions', 0),
}}
with open({result_path!r}, 'w') as f:
    json.dump(result, f)
"""

    t0 = time.time()
    proc = subprocess.run([PYTHON, "-c", script], text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [ERROR] Eval subprocess failed (exit {proc.returncode})")
        return None

    if not os.path.exists(result_path):
        print(f"  [ERROR] Result file missing: {result_path}")
        return None

    with open(result_path) as f:
        metrics = json.load(f)
    os.remove(result_path)
    metrics["eval_time_s"] = elapsed
    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPT architecture sweep")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated config names (empty=all)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only train, skip evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate already-trained models")
    args = parser.parse_args()

    if args.configs:
        names = [c.strip() for c in args.configs.split(",")]
        configs = [CONFIG_MAP[n] for n in names if n in CONFIG_MAP]
    else:
        configs = ARCH_CONFIGS

    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_state()
    results = load_results()

    print(f"Device: {device}")
    print(f"Configs: {[c['name'] for c in configs]}")
    print(f"Training: all 4695 stocks, epochs cap={GPT_EPOCHS}, early_stop={GPT_EARLY_STOP}")
    print(f"Eval: {'skip' if args.skip_eval else 'all test stocks, all days (subprocess)'}")
    print(f"Resume state: {len(state)} entries")
    print(f"Existing results: {list(results.keys())}")
    print()

    # 检查 tokenizer
    tok_path = os.path.join(CHECKPOINT_DIR, "tok_sweep_emb64_hid192.pt")
    if not os.path.exists(tok_path):
        print(f"[ERROR] Tokenizer not found: {tok_path}")
        print("Run experiments/02-tokenizer-tuning/ first!")
        return

    t_total = time.time()

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"[{name}] {cfg['description']}")
        print(f"  dim={cfg['dim']}, depth={cfg['depth']}, heads={cfg['heads']}, "
              f"kv={cfg['kv_heads']}, gc={'ON' if cfg['gradient_checkpointing'] else 'OFF'}")
        print(f"{'='*60}")

        # ── 训练 ──
        if not args.eval_only:
            try:
                gpt_path = train_arch(cfg, state)
            except Exception as e:
                print(f"  [ERROR] Training failed: {e}")
                state.setdefault(name, {})["train_status"] = "failed"
                state[name]["error"] = str(e)
                save_state(state)
                continue
        else:
            gpt_path = gpt_save_path(name)
            if not os.path.exists(gpt_path):
                print(f"  [skip] No checkpoint: {gpt_path}")
                continue

        # ── 评估 ──
        if args.skip_eval:
            print("  [skip] Evaluation skipped")
            continue

        if is_evaluated(name, results):
            print(f"  [skip] Already evaluated: DA={results[name]['da']*100:.2f}%")
            continue

        print(f"\n  Evaluating in subprocess...")
        metrics = evaluate_arch(cfg, gpt_path)

        if metrics is None:
            print(f"  [SKIP] Evaluation failed for {name}")
            continue

        result = {
            "name": name,
            "dim": cfg["dim"],
            "depth": cfg["depth"],
            "heads": cfg["heads"],
            "kv_heads": cfg["kv_heads"],
            "gradient_checkpointing": cfg["gradient_checkpointing"],
            "max_seq_len": cfg["max_seq_len"],
            "gpt_path": gpt_path,
            **metrics,
        }
        results[name] = result
        save_results(results)

        # 更新状态
        state.setdefault(name, {})["eval_status"] = "completed"
        save_state(state)

        print(f"\n  [{name}] Results:")
        print(f"    DA:            {result['da']*100:.2f}% (std={result['da_std']*100:.1f}%)")
        print(f"    DA above base: {result['da_above_baseline']*100:+.2f}%")
        print(f"    MAPE:          {result.get('mape', 0):.2f}% (BL: {result.get('baseline_mape', 0):.2f}%)")
        print(f"    Collapse:      {result['collapse_rate']*100:.1f}%")
        print(f"    Unique:        {result['n_unique_tokens']}")
        print(f"    RankIC:        {result['rank_ic']:.4f}")
        print(f"    AmpRatio:      {result['ampratio']:.3f}")
        print(f"    Eval time:     {result['eval_time_s']:.0f}s")

    # ── 汇总 ──
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Config':<10} {'Dim':>4} {'Dep':>3} {'H':>2} {'DA%':>7} {'Coll%':>7} "
              f"{'Uniq':>5} {'RankIC':>7} {'AmpR':>6} {'MAPE%':>7}")
        print("-" * 80)
        for name in sorted(results.keys(),
                           key=lambda n: results[n].get("da", 0), reverse=True):
            r = results[name]
            print(f"{name:<10} {r.get('dim',0):>4} {r.get('depth',0):>3} {r.get('heads',0):>2} "
                  f"{r['da']*100:>6.2f}% {r['collapse_rate']*100:>6.1f}% "
                  f"{r['n_unique_tokens']:>5} {r['rank_ic']:>7.4f} {r['ampratio']:>5.3f} "
                  f"{r.get('mape',0):>6.2f}%")

        best = max(results.keys(), key=lambda n: results[n].get("da", 0))
        print(f"\nBest: {best} (DA={results[best]['da']*100:.2f}%)")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total/3600:.1f}h")
    print(f"Results: {RESULTS_PATH}")
    print(f"State: {STATE_PATH}")


if __name__ == "__main__":
    main()
