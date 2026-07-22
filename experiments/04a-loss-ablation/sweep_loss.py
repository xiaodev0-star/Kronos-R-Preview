"""Exp 04-A: Loss Ablation — Focal (γ=4) vs Cross-Entropy

对比 focal loss 和标准 cross-entropy 在完全相同训练条件下的收敛行为和预测质量。

动机：focal loss 源自目标检测（类别不平衡场景），用于金融时序预测缺乏理论先验。
HPO 只在 focal 内部扫了 γ，从未与标准 CE 做过公平对比。本实验验证这一根本性选择。

用法:
    python experiments/04a-loss-ablation/sweep_loss.py
    python experiments/04a-loss-ablation/sweep_loss.py --arm focal   # 只跑一个
    python experiments/04a-loss-ablation/sweep_loss.py --resume      # 断点续算
    python experiments/04a-loss-ablation/sweep_loss.py --eval_only   # 只做评估
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

# ── Arm 定义：仅 loss 不同，其余超参完全一致 ──
ARMS = {
    "focal": {
        "loss": "focal",
        "gamma": 4.0,
        "tag": "exp04a_focal",
    },
    "ce": {
        "loss": "ce",
        "gamma": 0.0,       # CE 不使用 gamma
        "tag": "exp04a_ce",
    },
}

# 共享超参（除 loss 外全部相同 —— 均为 HPO 最优默认值）
SHARED = dict(
    epochs=50,
    dropout=0.1,
    weight_decay=0.01,
    heteroscedastic=True,
    het_weight=0.1,
    fine_weight=0.3,
    early_stop_patience=5,
    batch_tokens=12288,
    max_stocks=0,           # 全量 4695 只
    max_seq_len=0,          # 不限 seq len
    optimizer="adamw",
    lr=3e-4,
    label_smoothing=0.0,    # 不加 label smoothing，隔离 loss 变量
    entropy_alpha=0.0,      # 不加 entropy 正则
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
        "--loss", arm["loss"],
        "--gamma", str(arm["gamma"]),
        "--dropout", str(SHARED["dropout"]),
        "--weight_decay", str(SHARED["weight_decay"]),
        "--het_weight", str(SHARED["het_weight"]),
        "--fine_weight", str(SHARED["fine_weight"]),
        "--epochs", str(SHARED["epochs"]),
        "--early_stop_patience", str(SHARED["early_stop_patience"]),
        "--batch_tokens", str(SHARED["batch_tokens"]),
        "--max_stocks", str(SHARED["max_stocks"]),
        "--max_seq_len", str(SHARED["max_seq_len"]),
        "--optimizer", SHARED["optimizer"],
        "--lr", str(SHARED["lr"]),
        "--label_smoothing", str(SHARED["label_smoothing"]),
        "--entropy_alpha", str(SHARED["entropy_alpha"]),
    ]
    if SHARED["heteroscedastic"]:
        cmd.append("--heteroscedastic")
    cmd.append("--history_per_epoch")
    return cmd


def run_arm(arm_name: str, resume: bool = False):
    print(f"\n{'='*60}")
    print(f"  ARM: {arm_name.upper()}")
    print(f"{'='*60}")
    cmd = build_cmd(arm_name, resume)
    # 打印关键差异
    arm = ARMS[arm_name]
    print(f"  loss={arm['loss']}, gamma={arm['gamma']}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"\n  [{arm_name}] {status}  elapsed={elapsed/60:.1f}min")
    return proc.returncode


def run_eval(arm_name: str, n_stocks: int = 0) -> dict:
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
    path = os.path.join(OUT, f"history_exp04a_{arm_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def print_comparison(results: dict):
    print(f"\n{'='*70}")
    print(f"  Exp 04-A: Loss Ablation — Focal (γ=4) vs CE")
    print(f"{'='*70}")

    # 训练指标
    print(f"\n  {'Metric':<28} {'Focal γ=4':>12} {'CE':>12} {'Delta':>12}")
    print(f"  {'-'*64}")

    for key, label, direction in [
        ("best_val_loss", "Best val_loss", "min"),
        ("final_val_loss", "Final val_loss", "min"),
        ("final_train_loss", "Final train_loss", "min"),
        ("converged_epoch", "Converged epoch", "fewer"),
        ("total_epochs", "Total epochs", "fewer"),
    ]:
        vf = results.get("focal", {}).get("training", {}).get(key)
        vc = results.get("ce", {}).get("training", {}).get(key)
        if vf is not None and vc is not None:
            if isinstance(vf, float):
                delta = vc - vf
                marker = " <<" if (direction == "min" and delta < 0) else ""
                print(f"  {label:<28} {vf:>12.4f} {vc:>12.4f} {delta:>+12.4f} {marker}")
            else:
                print(f"  {label:<28} {str(vf):>12} {str(vc):>12}")

    # 评估指标
    has_eval = "focal_eval" in results and "ce_eval" in results
    if has_eval:
        print(f"\n  {'Metric':<28} {'Focal γ=4':>12} {'CE':>12} {'Delta':>12}")
        print(f"  {'-'*64}")
        for label, key, fmt in [
            ("DA (%)", "avg_da_per_date", ".2f"),
            ("DA std", "da_std", ".2f"),
            ("RankIC", "rank_ic", ".4f"),
            ("Collapse (%)", "collapse_rate", ".2f"),
            ("Unique tokens", "unique_tokens", "d"),
            ("AmpRatio", "amp_ratio", ".4f"),
            ("MAPE (%)", "mape", ".2f"),
        ]:
            vf = results.get("focal_eval", {}).get(key)
            vc = results.get("ce_eval", {}).get(key)
            if vf is not None and vc is not None:
                delta = vc - vf
                print(f"  {label:<28} {vf:>12{fmt}} {vc:>12{fmt}} {delta:>+12{fmt}}")

    print(f"\n  结论: ", end="")
    vf = results.get("focal", {}).get("training", {}).get("best_val_loss")
    vc = results.get("ce", {}).get("training", {}).get("best_val_loss")
    vd = results.get("focal_eval", {}).get("avg_da_per_date", 0) or 0
    ve = results.get("ce_eval", {}).get("avg_da_per_date", 0) or 0
    if vf is None or vc is None:
        print("训练数据不完整，无法判断。")
    elif vf < vc and vd > ve:
        print("Focal loss 在 val_loss 和 DA 上均优于 CE，γ=4 的 hard-example 聚焦有效。")
    elif vf > vc and vd < ve:
        print("CE 优于 Focal —— hard-example 聚焦对金融时序无正向作用，建议切换为 CE。")
    elif vf < vc:
        print("Focal val_loss 更低但 DA 无显著差异 —— loss 下降不等于预测提升。")
    else:
        print("两者接近，需结合 DA/RankIC 综合判断。")
    print()


def main():
    global SHARED
    parser = argparse.ArgumentParser(description="Exp 04-A: Loss Ablation")
    parser.add_argument("--arm", choices=["focal", "ce", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick smoke test: 50 stocks, 2 epochs, no eval")
    args = parser.parse_args()

    if args.dry_run:
        SHARED = {**SHARED, "max_stocks": 50, "epochs": 2,
                  "early_stop_patience": 5, "batch_tokens": 4096}
        print("  [DRY RUN] max_stocks=50, epochs=2")

    results = {}
    arms = ["focal", "ce"] if args.arm == "both" else [args.arm]

    # ── 训练 ──
    if not args.eval_only:
        for arm in arms:
            rc = run_arm(arm, resume=args.resume)
            if rc != 0:
                print(f"  [WARN] {arm} training failed")

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
            results[arm] = {
                "training": {
                    "final_train_loss": tl[-1] if tl else None,
                    "final_val_loss": vl[-1] if vl else None,
                    "best_val_loss": min(vl) if vl else None,
                    "converged_epoch": len(vl),
                    "total_epochs": len(vl),
                }
            }

    # ── 保存 ──
    summary_path = os.path.join(OUT, "comparison.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    print_comparison(results)


if __name__ == "__main__":
    main()
