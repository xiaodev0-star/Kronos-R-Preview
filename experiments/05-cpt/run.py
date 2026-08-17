"""Exp 05 (CPT)：单脚本全流程。

读取 Exp 04-B 的两个最优配置权重（top-2）的 ep100 终态，跑 400 窗推理，记录各项
数据，并写成 Parquet：

  cpt_predictions.parquet  每 (weight, epoch, date, symbol) 的预测/真实 token + logret/close
  cpt_daily.parquet        每 (weight, epoch, date) 的逐日指标
  cpt_epochs.parquet       每 (weight, epoch) 的汇总

推理复用 ``experiments/04/b-hpo/evaluate_epoch_trajectory.py``（输入准备/缓存共享，
不触碰 holdout）。写 Parquet 需要 ``pyarrow``。逐 epoch 快照已精简删除，仅保留 ep100。

用法：
    python experiments/05-cpt/run.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
TOKENIZER = ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt"
RESULTS = ROOT / "server_runs" / "results" / "05-cpt" / "seed42"
CACHE = RESULTS / "cache"

# Eval 期模型/tokenizer 配置 == config.py 默认（no-op override，仅供 eval 读取）。
OVERRIDE = {
    "DataConfig": {"max_stocks": 0, "random_seed": 42},
    "ModelConfig": {"dim": 256, "depth": 6, "heads": 4,
                    "num_kv_heads": 1, "ffn_multiplier": 4},
    "TokenizerConfig": {"bits_l1": 7, "bits_l2": 7, "embedding_dim": 64,
                        "hidden_dim": 192, "random_seed": 42},
}

# Exp 04-B HPO top-2 -> CPT ep100 终态 checkpoint。
WEIGHTS = {
    "4c72": ROOT / "checkpoints" / "exp04b_best_ep100.pt",   # lr_muon=0.005
    "8ceb": ROOT / "checkpoints" / "exp04b_8ceb_ep100.pt",   # lr_muon=0.01
}

# 逐股票×逐日的长表（joint token = coarse*128 + fine，不落盘）。
PRED_SCHEMA = pa.schema([
    ("weight", pa.string()),
    ("epoch", pa.int8()),
    ("date", pa.string()),
    ("symbol", pa.string()),
    ("pred_coarse", pa.uint8()),
    ("pred_fine", pa.uint8()),
    ("true_coarse", pa.uint8()),
    ("true_fine", pa.uint8()),
    ("pred_logret", pa.float32()),
    ("true_logret", pa.float32()),
    ("base_close", pa.float32()),
    ("true_close", pa.float32()),
])


def run_eval(code: str, ckpt: Path) -> Path:
    """Build a single-ep100 trial dir and run the 400-window evaluation."""
    trial = RESULTS / "trials" / code
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "override.json").write_text(
        json.dumps(OVERRIDE, indent=2), encoding="utf-8")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    val_loss = payload.get("val_loss")
    val_loss = float(val_loss) if isinstance(val_loss, (int, float)) else 3.6
    index = {
        "tag": code,
        "save_path": str(ckpt.resolve()),
        "updated_epoch": 100,
        "checkpoints": [{
            "epoch": 100,
            "path": str(ckpt.resolve()),
            "size_bytes": ckpt.stat().st_size,
            "train_loss": val_loss,
            "val_loss": val_loss,
            "learning_rate": 0.0,
            "learning_rate_adam": 0.0,
            "optimizer_steps_this_epoch": 0,
            "global_step": 0,
            "best_so_far": False,
        }],
    }
    (trial / "model_checkpoints.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--trial_dir", str(trial),
        "--tokenizer", str(TOKENIZER),
        "--output_dir", str(trial / "epoch_trajectory"),
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(CACHE),
        "--no_reference_check",
    ]
    print(f"[{code}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return trial


def epoch_files(trial: Path) -> list[Path]:
    """Completed epoch_*.json payloads, sorted by epoch."""
    pattern = re.compile(r"^epoch_(\d{3})\.json$")
    matches = []
    for path in (trial / "epoch_trajectory").iterdir():
        m = pattern.match(path.name)
        if m:
            matches.append((int(m.group(1)), path))
    return [p for _, p in sorted(matches)]


def epoch_row(weight: str, payload: dict) -> dict:
    agg, tr = payload["aggregate"], payload["training"]
    return {
        "weight": weight, "epoch": payload["epoch"],
        "val_loss": tr.get("val_loss"), "val_coarse_loss": tr.get("val_coarse_loss"),
        "val_fine_loss": tr.get("val_fine_loss"),
        "avg_da": agg.get("avg_da_per_date"), "avg_mape": agg.get("avg_mape"),
        "baseline_mape": agg.get("avg_baseline_mape"), "avg_ampratio": agg.get("avg_ampratio"),
        "avg_rank_ic": agg.get("avg_daily_rank_ic"),
        "median_collapse": agg.get("median_daily_collapse_rate"),
        "p90_collapse": agg.get("p90_daily_collapse_rate"),
        "worst_collapse": agg.get("worst_daily_collapse_rate"),
        "median_unique": agg.get("median_daily_unique_tokens"),
        "min_unique": agg.get("min_daily_unique_tokens"),
        "coarse_balance": agg.get("median_daily_codebook_balance_score"),
        "p10_coarse_balance": agg.get("p10_daily_codebook_balance_score"),
        "joint_balance": agg.get("median_daily_joint_codebook_balance_score"),
        "token_jsd": agg.get("median_daily_token_jsd"),
        "support_f1": agg.get("median_daily_token_support_f1"),
        "healthy": agg.get("healthy"),
    }


def daily_rows(weight: str, payload: dict):
    for window in payload["windows"].values():
        for date_key, m in window.get("per_date", {}).items():
            yield {
                "weight": weight, "epoch": payload["epoch"], "date": date_key,
                "mape": m.get("mape"), "baseline_mape": m.get("baseline_mape"),
                "da": m.get("da"), "da_above_baseline": m.get("da_above_baseline"),
                "ampratio": m.get("ampratio"), "rank_ic": m.get("rank_ic"),
                "collapse_rate": m.get("collapse_rate"),
                "n_unique_tokens": m.get("n_unique_tokens"),
                "coarse_acc": m.get("coarse_token_accuracy"),
                "fine_acc": m.get("fine_token_accuracy"),
                "joint_acc": m.get("joint_token_accuracy"),
                "coarse_balance": m.get("codebook_balance_score"),
                "fine_balance": m.get("fine_codebook_balance_score"),
                "joint_balance": m.get("joint_codebook_balance_score"),
                "n_predictions": m.get("n"),
            }


def write_metrics(trials: dict[str, Path]) -> None:
    """Read per-epoch JSONs and write daily/epoch metrics to Parquet."""
    epoch_rows, daily = [], []
    for weight, trial in trials.items():
        for path in epoch_files(trial):
            payload = json.loads(path.read_text(encoding="utf-8"))
            epoch_rows.append(epoch_row(weight, payload))
            daily.extend(daily_rows(weight, payload))
    pd.DataFrame(epoch_rows).to_parquet(RESULTS / "cpt_epochs.parquet", index=False)
    pd.DataFrame(daily).to_parquet(RESULTS / "cpt_daily.parquet", index=False)
    print(f"wrote cpt_epochs.parquet ({len(epoch_rows)} rows), "
          f"cpt_daily.parquet ({len(daily)} rows)", flush=True)


def write_predictions(trials: dict[str, Path]) -> None:
    """Stream the per-epoch prediction_records NPZ into one long Parquet table."""
    out = RESULTS / "cpt_predictions.parquet"
    writer = None
    total = 0
    try:
        for weight, trial in trials.items():
            tgt = np.load(trial / "epoch_trajectory" / "evaluation_targets.npz",
                          allow_pickle=True)
            symbol = tgt["symbols"][tgt["symbol_index"]]
            date = tgt["dates"][tgt["date_index"]]
            for epoch in (100,):
                preds = np.load(
                    trial / "epoch_trajectory" / f"prediction_records_epoch_{epoch:03d}.npz",
                    allow_pickle=True,
                )
                n = preds["pred_coarse_id"].shape[0]
                table = pa.Table.from_arrays(
                    [
                        pa.array(np.full(n, weight, dtype=object), type=pa.string()),
                        pa.array(np.full(n, epoch, dtype=np.int8), type=pa.int8()),
                        pa.array(date, type=pa.string()),
                        pa.array(symbol, type=pa.string()),
                        pa.array(preds["pred_coarse_id"].astype(np.uint8), type=pa.uint8()),
                        pa.array(preds["pred_fine_id"].astype(np.uint8), type=pa.uint8()),
                        pa.array(tgt["true_coarse_id"].astype(np.uint8), type=pa.uint8()),
                        pa.array(tgt["true_fine_id"].astype(np.uint8), type=pa.uint8()),
                        pa.array(preds["pred_logret"].astype(np.float32), type=pa.float32()),
                        pa.array(tgt["true_logret"].astype(np.float32), type=pa.float32()),
                        pa.array(tgt["base_close"].astype(np.float32), type=pa.float32()),
                        pa.array(tgt["true_close"].astype(np.float32), type=pa.float32()),
                    ],
                    schema=PRED_SCHEMA,
                )
                if writer is None:
                    writer = pq.ParquetWriter(out, PRED_SCHEMA, compression="zstd")
                writer.write_table(table)
                total += n
                del preds, table
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote cpt_predictions.parquet ({total/1e6:.1f}M rows)", flush=True)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    trials = {code: run_eval(code, ckpt) for code, ckpt in WEIGHTS.items()}
    write_metrics(trials)
    write_predictions(trials)
    for trial in trials.values():
        shutil.rmtree(trial, ignore_errors=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
