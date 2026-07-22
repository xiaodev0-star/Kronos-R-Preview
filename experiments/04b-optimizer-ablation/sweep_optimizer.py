"""Exp 04-B: Optimizer Ablation — AdamW vs Muon 收敛对比实验

两个 arm 以相同超参（仅 optimizer 不同）训练至自然收敛（early_stop patience=5），
对比训练动态、收敛速度、最终 val_loss 和下游 eval 指标。

前置：Exp 04-A (Loss Ablation) 确定最优 loss 后，本实验使用该 loss 配置。

用法:
    python experiments/04b-optimizer-ablation/sweep_optimizer.py
    python experiments/04b-optimizer-ablation/sweep_optimizer.py --arm adamw
    python experiments/04b-optimizer-ablation/sweep_optimizer.py --resume
    python experiments/04b-optimizer-ablation/sweep_optimizer.py --eval_only
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

SCRIPT = os.path.join(ROOT, "train_base.py")
EVAL = os.path.join(ROOT, "eval.py")
TOK = os.path.join(ROOT, "experiments", "02-tokenizer-tuning", "tok_sweep_emb64_hid192.pt")
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Arm 定义：仅 optimizer 不同，其余超参完全一致 ──
ARMS = {
    "adamw": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "tag": "exp04a_adamw",
    },
    "muon": {
        "optimizer": "muon",
        "lr": 3e-4,
        "lr_muon": 0.02,
        "tag": "exp04a_muon",
    },
}

# 共享超参（除 optimizer 外全部相同）
SHARED = dict(
    epochs=50,              # 足够长，让 early stopping 决定何时停
    loss="focal",
    gamma=4.0,
    dropout=0.1,
    weight_decay=0.01,
    heteroscedastic=True,
    het_weight=0.1,
    fine_weight=0.3,
    early_stop_patience=5,
    batch_tokens=12288,
    history_per_epoch=True,
    max_stocks=0,           # 全量 4695 只
    max_seq_len=0,          # 不限 seq len，保留全部历史上下文
)


def build_cmd(arm_name: str, resume: bool = False) -> list[str]:
    arm = ARMS[arm_name]
    tag = arm["tag"]
    save = os.path.join(OUT, f"gpt_{arm_name}.pt")
    cmd = [
        sys.executable, SCRIPT,
        "--save_path", save,
        "--tokenizer_path", TOK,
        "--tag", tag,
        "--loss", SHARED["loss"],
        "--gamma", str(SHARED["gamma"]),
        "--dropout", str(SHARED["dropout"]),
        "--weight_decay", str(SHARED["weight_decay"]),
        "--het_weight", str(SHARED["het_weight"]),
        "--fine_weight", str(SHARED["fine_weight"]),
        "--epochs", str(SHARED["epochs"]),
        "--early_stop_patience", str(SHARED["early_stop_patience"]),
        "--batch_tokens", str(SHARED["batch_tokens"]),
        "--max_stocks", str(SHARED["max_stocks"]),
        "--max_seq_len", str(SHARED["max_seq_len"]),
        "--optimizer", arm["optimizer"],
        "--lr", str(arm["lr"]),
    ]
    if SHARED["heteroscedastic"]:
        cmd.append("--heteroscedastic")
    if SHARED["history_per_epoch"]:
        cmd.append("--history_per_epoch")
    if "lr_muon" in arm:
        cmd += ["--lr_muon", str(arm["lr_muon"])]
    return cmd


def run_arm(arm_name: str, resume: bool = False):
    """训练单个 arm（阻塞式，输出实时打印）"""
    print(f"\n{'='*60}")
    print(f"  ARM: {arm_name.upper()}")
    print(f"{'='*60}")
    cmd = build_cmd(arm_name, resume)
    print(f"  CMD: {' '.join(cmd[-10:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"\n  [{arm_name}] {'OK' if proc.returncode == 0 else f'FAILED (rc={proc.returncode})'}"
          f"  elapsed={elapsed/60:.1f}min")
    return proc.returncode


def run_eval(arm_name: str, n_stocks: int = 0) -> dict:
    """对训练好的 checkpoint 做 windowed eval"""
    ckpt = os.path.join(OUT, f"gpt_{arm_name}.pt")
    if not os.path.exists(ckpt):
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return {}
    out_json = os.path.join(OUT, f"eval_{arm_name}.json")
    cmd = [
        sys.executable, EVAL, "windowed",
        "--gpt_ckpt", ckpt,
        "--tokenizer", TOK,
        "--n_days", "20",
        "--n_stocks", str(n_stocks),
        "--output", out_json,
    ]
    print(f"\n  [eval:{arm_name}] Running windowed eval (n_stocks={n_stocks})...")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode == 0 and os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    return {}


def load_history(arm_name: str) -> dict:
    """加载训练 history JSON"""
    path = os.path.join(OUT, f"history_exp04a_{arm_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def print_comparison(results: dict):
    """打印对比表"""
    print(f"\n{'='*70}")
    print(f"  Exp 04-A: Optimizer Ablation — Results")
    print(f"{'='*70}")

    # 训练指标
    print(f"\n  {'Metric':<28} {'AdamW':>12} {'Muon':>12} {'Delta':>12}")
    print(f"  {'-'*64}")

    metrics = [
        ("Final train_loss", "final_train_loss", "min"),
        ("Final val_loss", "final_val_loss", "min"),
        ("Best val_loss", "best_val_loss", "best"),
        ("Converged epoch", "converged_epoch", "N/A"),
        ("Total epochs", "total_epochs", "N/A"),
    ]

    for label, key, direction in metrics:
        v_adamw = results.get("adamw", {}).get("training", {}).get(key)
        v_muon = results.get("muon", {}).get("training", {}).get(key)
        if isinstance(v_adamw, (int, float)) and isinstance(v_muon, (int, float)):
            delta = v_muon - v_adamw
            marker = " <<" if (direction in ("min", "best") and delta < 0) else ""
            print(f"  {label:<28} {v_adamw:>12.4f} {v_muon:>12.4f} {delta:>+12.4f} {marker}")
        else:
            print(f"  {label:<28} {str(v_adamw or '—'):>12} {str(v_muon or '—'):>12}")

    # 评估指标
    has_eval = "adamw_eval" in results and "muon_eval" in results
    if has_eval:
        print(f"\n  {'Metric':<28} {'AdamW':>12} {'Muon':>12} {'Delta':>12}")
        print(f"  {'-'*64}")
        eval_metrics = [
            ("DA (%)", "avg_da_per_date", "+pp"),
            ("DA std", "da_std", "N/A"),
            ("RankIC", "rank_ic", "+"),
            ("Collapse (%)", "collapse_rate", "−"),
            ("Unique tokens", "unique_tokens", "+"),
            ("AmpRatio", "amp_ratio", "N/A"),
            ("MAPE (%)", "mape", "−"),
        ]
        for label, key, direction in eval_metrics:
            va = results.get("adamw_eval", {}).get(key, None)
            vm = results.get("muon_eval", {}).get(key, None)
            if va is not None and vm is not None:
                delta = vm - va
                if "pp" in direction:
                    print(f"  {label:<28} {va:>12.2f} {vm:>12.2f} {delta:>+12.2f}")
                else:
                    print(f"  {label:<28} {va:>12.4f} {vm:>12.4f} {delta:>+12.4f}")

    print()


def main():
    global SHARED
    parser = argparse.ArgumentParser(description="Exp 04-B: Optimizer Ablation")
    parser.add_argument("--arm", choices=["adamw", "muon", "both"], default="both",
                        help="Which arm to run (default: both)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only eval")
    parser.add_argument("--no_eval", action="store_true", help="Skip eval, only train")
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick smoke test: 50 stocks, 2 epochs, no eval")
    args = parser.parse_args()

    if args.dry_run:
        SHARED = {**SHARED, "max_stocks": 50, "epochs": 2,
                  "early_stop_patience": 5, "batch_tokens": 4096}
        print("  [DRY RUN] max_stocks=50, epochs=2")

    results = {}
    arms = ["adamw", "muon"] if args.arm == "both" else [args.arm]

    # ── 训练 ──
    if not args.eval_only:
        for arm in arms:
            rc = run_arm(arm, resume=args.resume)
            if rc != 0:
                print(f"  [WARN] {arm} training failed, skipping eval")

    # ── 评估 ──
    eval_n_stocks = 50 if args.dry_run else 0
    if not args.no_eval:
        for arm in arms:
            ev = run_eval(arm, n_stocks=eval_n_stocks)
            if ev:
                results[f"{arm}_eval"] = ev

    # ── 收集训练指标 ──
    for arm in arms:
        hist = load_history(arm)
        if hist:
            vl = hist.get("val_loss", [])
            tl = hist.get("train_loss", [])
            converged_epoch = len(vl)
            best_val = min(vl) if vl else float("inf")
            final_train = tl[-1] if tl else float("inf")
            final_val = vl[-1] if vl else float("inf")
            results[arm] = {
                "training": {
                    "final_train_loss": final_train,
                    "final_val_loss": final_val,
                    "best_val_loss": best_val,
                    "converged_epoch": converged_epoch,
                    "total_epochs": len(vl),
                }
            }

    # ── 保存 ──
    summary_path = os.path.join(OUT, "comparison.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    # ── 打印对比 ──
    print_comparison(results)


if __name__ == "__main__":
    main()
