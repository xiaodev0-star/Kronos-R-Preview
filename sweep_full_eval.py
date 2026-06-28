"""大规模全量推理：复用 BitSweep 10 组已训练权重，预测 cutoff 后所有股票所有日期。

与 sweep_bits.py / eval_windowed.py 的区别：
  - 不限 n_stocks=2400，使用全部 test_stocks
  - 不限 n_days=20，预测每只股票 cutoff 后的所有可测日期
  - 记录每个预测点的原始值：symbol / date / pred_logret / true_logret / coarse_id 等
  - 每个 config 一个最终 npz，按 bucket 保存 partial 文件，支持断点续算

Usage:
    python sweep_full_eval.py                              # 全部 10 个 config
    python sweep_full_eval.py --configs "8+6,9+6"          # 指定 config
    python sweep_full_eval.py --batch_size 2               # 减小 batch 防 OOM
    python sweep_full_eval.py --resume                     # 断点续算
"""
import argparse
import gc
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from scipy.stats import spearmanr

from config import DataConfig, ModelConfig, set_global_seed
from data_processor import load_stocks, split_stocks
from eval_helpers import (
    AMP_DTYPE,
    _bucket_by_length,
    decode_coarse_batch,
    load_gpt,
    load_tokenizer,
)
from sweep_bits import ALL_CONFIGS, tok_path, gpt_path
from tqdm import tqdm

SEED = 42


def config_key(l1, l2):
    return f"{l1}+{l2}"


def full_eval_dir(seed=SEED):
    return os.path.join("checkpoints", "sweep", f"full_eval_seed{seed}")


def checkpoint_path(seed=SEED):
    return os.path.join(full_eval_dir(seed), "checkpoint.json")


def partial_dir(config, seed=SEED):
    return os.path.join(full_eval_dir(seed), "partial", config)


def final_npz_path(config, seed=SEED):
    return os.path.join(full_eval_dir(seed), f"{config}_predictions.npz")


def metrics_path(seed=SEED):
    return os.path.join(full_eval_dir(seed), "metrics.json")


def load_checkpoint(seed=SEED):
    p = checkpoint_path(seed)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"configs": {}}


def save_checkpoint(state, seed=SEED):
    p = checkpoint_path(seed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def save_partial_npz(path, records):
    """将一批预测记录保存为 npz。"""
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # dtype=object prevents numpy from inferring a fixed-width string dtype
    # that would silently truncate long symbols (e.g. "000001" → "00").
    np.savez_compressed(
        path,
        symbol=np.array([r["symbol"] for r in records], dtype=object),
        date=np.array([r["date"] for r in records], dtype=object),
        pred_logret=np.array([r["pred_logret"] for r in records], dtype=np.float32),
        true_logret=np.array([r["true_logret"] for r in records], dtype=np.float32),
        coarse_id=np.array([r["coarse_id"] for r in records], dtype=np.int64),
        decoded_feat=np.array([r["decoded_feat"] for r in records], dtype=np.float32),
        base_close=np.array([r["base_close"] for r in records], dtype=np.float64),
        true_close=np.array([r["true_close"] for r in records], dtype=np.float64),
        config=np.array([r["config"] for r in records], dtype=object),
    )


def merge_partial_npz(config, seed=SEED, remove_partial=True):
    """合并一个 config 的所有 partial npz 为最终文件。"""
    pdir = partial_dir(config, seed)
    out_path = final_npz_path(config, seed)

    if not os.path.exists(pdir):
        return False

    files = sorted([f for f in os.listdir(pdir) if f.endswith(".npz")])
    if not files:
        return False

    arrays = {}
    keys = [
        "symbol", "date", "pred_logret", "true_logret", "coarse_id",
        "decoded_feat", "base_close", "true_close", "config",
    ]
    for k in keys:
        arrays[k] = []

    for fname in files:
        data = np.load(os.path.join(pdir, fname), allow_pickle=True)
        for k in keys:
            arrays[k].append(data[k])
        data.close()

    merged = {k: np.concatenate(arrays[k], axis=0) for k in keys}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **merged)

    if remove_partial:
        for fname in files:
            os.remove(os.path.join(pdir, fname))
        os.rmdir(pdir)

    return True


@torch.no_grad()
def infer_bucket(gpt, tokenizer, bucket, device, symbol_map, key):
    """对一个 bucket 的股票做前向，并返回所有 cutoff 后日期的预测记录。"""
    B = len(bucket)
    max_len = max(s["seq_len"] for s in bucket)
    vocab = tokenizer.vocab_coarse

    inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
    tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
    mask = torch.zeros(B, max_len, max_len, dtype=torch.bool, device=device)
    va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

    for j, s in enumerate(bucket):
        L = s["seq_len"]
        inp[j, :L] = torch.tensor(s["inp_ids"], dtype=torch.long)
        tids[j, :L, 0] = torch.tensor(s["day"], dtype=torch.long)
        tids[j, :L, 1] = torch.tensor(s["month"], dtype=torch.long)
        tids[j, :L, 2] = torch.tensor(s["year"], dtype=torch.long)
        va[j, :L] = torch.tensor(s["va"], dtype=torch.float32)
        mask[j, :L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))
        mask[j, L:, 0] = True

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(autocast_device, dtype=AMP_DTYPE):
        coarse_logits, fine_logits = gpt(inp, tids, pos, mask, va_values=va)

    coarse_cpu = coarse_logits.float().cpu()
    fine_cpu = fine_logits.float().cpu()

    # 收集所有 cutoff 后日期的预测
    raw_preds = []
    for j, s in enumerate(bucket):
        symbol = symbol_map[id(s)]
        p_start = s["test_pos"]
        seq_len_real = min(coarse_cpu.shape[1], fine_cpu.shape[1], s["seq_len"])
        for offset in range(seq_len_real - p_start):
            p = p_start + offset
            if p >= seq_len_real:
                break
            cid = int(coarse_cpu[j, p, :vocab].argmax().item())
            fl = fine_cpu[j, p]
            date_key = s["dates_raw"][p] if p < len(s["dates_raw"]) else "unknown"
            raw_preds.append({
                "coarse_id": cid,
                "fine_logits": fl,
                "test_pos": p,
                "date_key": date_key,
                "symbol": symbol,
                "p_mean": s["p_mean"],
                "p_std": s["p_std"],
                "feat": s["feat"],
                "close": s["close"],
                "config": key,
            })

    if not raw_preds:
        return []

    # Batch decode
    fine_tensor = torch.stack([p["fine_logits"] for p in raw_preds])
    coarse_ids = [p["coarse_id"] for p in raw_preds]
    decoded = decode_coarse_batch(coarse_ids, fine_tensor, tokenizer, device)

    records = []
    for i, pred in enumerate(raw_preds):
        dec = decoded[i]
        tp = pred["test_pos"]
        feat = pred["feat"]
        close = pred["close"]
        records.append({
            "symbol": pred["symbol"],
            "date": str(pred["date_key"]),
            "pred_logret": float(dec[0]) * pred["p_std"][0] + pred["p_mean"][0],
            "true_logret": float(feat[tp, 0]) if tp < len(feat) else 0.0,
            "coarse_id": pred["coarse_id"],
            "decoded_feat": dec.astype(np.float32),
            "base_close": float(close[tp - 1]) if tp - 1 >= 0 else float(close[0]),
            "true_close": float(close[tp]) if tp < len(close) else 0.0,
            "config": pred["config"],
        })

    return records


def prepare_stocks_batch_with_symbol(stocks, tokenizer, device):
    """Batch-tokenize all stocks and keep symbol mapping.

    Same logic as eval_helpers._prepare_stocks_batch, but returns
    (prepped_list, symbol_list).
    """
    from eval_helpers import build_stock_arrays
    from tqdm import tqdm

    all_arrays = []
    symbols = []
    for stock in tqdm(stocks, desc="  build arrays", leave=False):
        arrays = build_stock_arrays(stock)
        if arrays is not None:
            all_arrays.append(arrays)
            symbols.append(stock["symbol"])

    if not all_arrays:
        return [], []

    cpu_device = torch.device("cpu")
    max_T = max(a["T_total"] for a in all_arrays)
    batch_np = np.zeros((len(all_arrays), max_T, 4), dtype=np.float32)
    lengths = []
    for j, a in enumerate(all_arrays):
        T = a["T_total"]
        batch_np[j, :T] = a["price_normed"][:T]
        lengths.append(T)

    chunk_size = 256
    all_idx_chunks = []
    for i in range(0, len(batch_np), chunk_size):
        chunk = torch.from_numpy(batch_np[i:i+chunk_size]).float()
        with torch.no_grad():
            idx, _ = tokenizer.encode(chunk.to(device))
        all_idx_chunks.append(idx.cpu().numpy())
    all_idx_np = np.concatenate(all_idx_chunks, axis=0)

    vocab = tokenizer.vocab_coarse
    bos_id = vocab
    results = []
    result_symbols = []
    for j, arrays in enumerate(all_arrays):
        T = lengths[j]
        token_ids = all_idx_np[j, :T]
        day, month, year = arrays["day"], arrays["month"], arrays["year"]

        inp_ids = [bos_id] + token_ids.tolist()
        seq_len = len(inp_ids) - 1

        va_seq = np.concatenate([
            np.zeros((1, 2), dtype=np.float32),
            arrays["va_normed"][:T - 1],
        ], axis=0)

        dates_np = arrays.get("dates_dt", None)
        if dates_np is not None and len(dates_np) >= T:
            dates_full = [str(d)[:10] if "T" in str(d) else str(d) for d in dates_np]
        else:
            dates_full = ["unknown"] * T
        dates_aligned = [dates_full[0]] + dates_full[:T-1]

        results.append({
            "inp_ids": inp_ids[:-1],
            "day": [int(day[0])] + day[:T-1].tolist(),
            "month": [int(month[0])] + month[:T-1].tolist(),
            "year": [int(year[0])] + year[:T-1].tolist(),
            "dates": dates_aligned,
            "dates_raw": dates_full,
            "va": va_seq,
            "seq_len": seq_len,
            "test_pos": arrays["ci"],
            "p_mean": arrays["p_mean"], "p_std": arrays["p_std"],
            "feat": arrays["feat"], "close": arrays["close"],
        })
        result_symbols.append(symbols[j])

    return results, result_symbols


def infer_config(l1, l2, test_stocks, device, batch_size=4, resume=True, seed=SEED):
    """对一个 config 做全量推理，支持断点续算。"""
    key = config_key(l1, l2)
    ckpt = load_checkpoint(seed)
    cfg_state = ckpt.setdefault("configs", {}).setdefault(key, {})

    if cfg_state.get("status") == "completed":
        print(f"[{key}] already completed -- skipping")
        return True

    if not resume and cfg_state.get("status") != "completed":
        # 不续算时重置该 config 的 partial 文件
        pdir = partial_dir(key, seed)
        if os.path.exists(pdir):
            for f in os.listdir(pdir):
                os.remove(os.path.join(pdir, f))

    cfg_state["status"] = "running"
    save_checkpoint(ckpt, seed)

    tok_ckpt = tok_path(l1, l2, seed)
    gpt_ckpt = gpt_path(l1, l2, seed)
    if not os.path.exists(tok_ckpt) or not os.path.exists(gpt_ckpt):
        print(f"[{key}] missing checkpoint: tok={tok_ckpt}, gpt={gpt_ckpt}")
        cfg_state["status"] = "failed"
        cfg_state["error"] = "missing checkpoint"
        save_checkpoint(ckpt, seed)
        return False

    print(f"\n{'='*60}")
    print(f"[{key}] Loading tokenizer + GPT ...")
    tokenizer = load_tokenizer(tok_ckpt, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    gpt = load_gpt(gpt_ckpt, device, tokenizer=tokenizer)

    print(f"[{key}] Preparing {len(test_stocks)} stocks ...")
    from eval_helpers import attach_close_prices
    attach_close_prices(test_stocks)
    prepped, symbols = prepare_stocks_batch_with_symbol(test_stocks, tokenizer, device)
    print(f"[{key}] {len(prepped)} valid stocks prepared")
    # GPU memory status for capacity planning
    if device.type == 'cuda':
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{key}] GPU memory: {alloc:.2f}G allocated, {reserved:.2f}G reserved")

    # 建立 prepped item -> symbol 的映射
    symbol_map = {id(p): sym for p, sym in zip(prepped, symbols)}

    # 只保留 test_pos < seq_len 的股票
    valid = [(p, symbol_map[id(p)]) for p in prepped if p["test_pos"] < p["seq_len"]]
    if not valid:
        print(f"[{key}] no valid stocks with test dates")
        cfg_state["status"] = "completed"
        cfg_state["n_predictions"] = 0
        save_checkpoint(ckpt, seed)
        return True

    # 仅按 prepped 长度分 bucket，symbol 跟随
    prepped_valid = [v[0] for v in valid]
    symbol_map_valid = {id(v[0]): v[1] for v in valid}
    buckets = _bucket_by_length(prepped_valid, tolerance=100)
    buckets = [[(p, symbol_map_valid[id(p)]) for p in b] for b in buckets]
    print(f"[{key}] {len(buckets)} length buckets, batch_size={batch_size}")

    pdir = partial_dir(key, seed)
    os.makedirs(pdir, exist_ok=True)

    total_preds = 0
    t0 = time.time()

    for bidx, bucket in enumerate(buckets):
        # Reset effective batch_size per bucket — OOM in a long-sequence bucket
        # shouldn't permanently penalize shorter later buckets.
        effective_bs = batch_size
        pbar = tqdm(total=len(bucket), desc=f"  bucket {bidx+1}/{len(buckets)}", leave=False, unit="stock")
        bucket_preds = 0
        start = 0
        while start < len(bucket):
            # Scan for any existing partial file covering offset `start` (any actual batch size).
            # Filename format: bucket_{bidx:05d}_start_{offset:05d}_n{actual_n}.npz
            existing_prefix = os.path.join(pdir, f"bucket_{bidx:05d}_start_{start:05d}_n")
            existing_match = None
            for f in os.listdir(pdir):
                if f.startswith(os.path.basename(existing_prefix)) and f.endswith(".npz"):
                    existing_match = f
                    break
            if existing_match:
                # Extract actual_n from filename to advance start correctly
                try:
                    actual_n = int(existing_match.replace(".npz", "").rsplit("_n", 1)[-1])
                except ValueError:
                    actual_n = effective_bs
                pbar.update(actual_n)
                start += actual_n
                continue

            sub = bucket[start:start+effective_bs]
            try:
                sub_prepped = [p for p, _ in sub]
                sub_symbols = {id(p): sym for p, sym in sub}
                records = infer_bucket(gpt, tokenizer, sub_prepped, device, sub_symbols, key)
                # Encode actual stock count in filename so resume never skips.
                partial_path = os.path.join(
                    pdir, f"bucket_{bidx:05d}_start_{start:05d}_n{len(sub)}.npz")
                save_partial_npz(partial_path, records)
                bucket_preds += len(records)
                pbar.update(len(sub))
                start += len(sub)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective_bs > 1:
                    effective_bs = max(1, effective_bs // 2)
                    pbar.set_postfix_str(f"OOM→bs={effective_bs}")
                    continue
                else:
                    pbar.set_postfix_str("OOM skip")
                    start += 1
                    pbar.update(1)

        pbar.close()
        total_preds += bucket_preds
        print(f"[{key}] bucket {bidx+1} saved {bucket_preds} predictions")

    # 合并 partial
    print(f"[{key}] merging partial results ...")
    merged = merge_partial_npz(key, seed, remove_partial=True)
    elapsed = time.time() - t0

    cfg_state["status"] = "completed"
    cfg_state["n_predictions"] = total_preds
    cfg_state["elapsed_s"] = elapsed
    cfg_state["n_buckets"] = len(buckets)
    cfg_state["batch_size_final"] = effective_bs
    save_checkpoint(ckpt, seed)

    del gpt, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{key}] done in {elapsed:.1f}s, {total_preds} predictions")
    return True


def compute_metrics_from_npz(config, seed=SEED):
    """从最终 npz 计算汇总指标。"""
    from collections import defaultdict

    path = final_npz_path(config, seed)
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    pred = data["pred_logret"]
    true = data["true_logret"]
    dates = data["date"]
    coarse = data["coarse_id"]

    by_date = defaultdict(list)
    for i in range(len(dates)):
        by_date[dates[i]].append((pred[i], true[i]))

    per_date = {}
    for d, vals in sorted(by_date.items()):
        if len(vals) < 5:
            continue
        ps = np.array([v[0] for v in vals])
        ts = np.array([v[1] for v in vals])
        da = float((np.sign(ps) == np.sign(ts)).mean())
        up_frac = float((ts > 0).mean())
        baseline_da = max(up_frac, 1 - up_frac)
        per_date[d] = {
            "da": da,
            "n": len(vals),
            "up_frac": up_frac,
            "baseline_da": baseline_da,
            "da_above_baseline": da - baseline_da,
        }

    das = np.array([v["da"] for v in per_date.values()])
    valid = ~np.isnan(pred) & ~np.isnan(true)
    rank_ic = float(spearmanr(pred[valid], true[valid])[0]) if valid.sum() > 2 else 0.0
    rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic

    unique, counts = np.unique(coarse, return_counts=True)
    collapse_rate = float(counts.max() / max(len(coarse), 1))

    eps = 1e-8
    amp_ratio = float(np.mean(np.abs(pred)) / max(np.mean(np.abs(true)), eps))

    return {
        "avg_da_per_date": float(das.mean()) if len(das) else 0.0,
        "da_std": float(das.std()) if len(das) > 1 else 0.0,
        "n_dates": len(per_date),
        "n_predictions": len(pred),
        "avg_da_above_baseline": float(np.mean([v["da_above_baseline"] for v in per_date.values()])) if per_date else 0.0,
        "collapse_rate": collapse_rate,
        "n_unique_tokens": int(len(unique)),
        "rank_ic": rank_ic,
        "ampratio": amp_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Full-scale sweep evaluation")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated configs, e.g. '8+6,9+6'. Empty = all.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint / partial files")
    parser.add_argument("--n_stocks", type=int, default=0,
                        help="Limit number of test stocks for dry-run (0 = all)")
    args = parser.parse_args()

    seed = args.seed
    os.makedirs(full_eval_dir(seed), exist_ok=True)
    set_global_seed(seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {seed}")

    if args.configs:
        configs = []
        for c in args.configs.split(","):
            l1, l2 = c.strip().split("+")
            configs.append((int(l1), int(l2)))
    else:
        configs = ALL_CONFIGS

    # 加载 test_stocks（cutoff 后的所有股票，或仅用于 dry-run 的子集）
    print("Loading all stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks = split_stocks(stocks)
    if args.n_stocks > 0 and args.n_stocks < len(test_stocks):
        test_stocks = test_stocks[:args.n_stocks]
    print(f"Total test stocks: {len(test_stocks)}")

    # 运行每个 config
    for l1, l2 in configs:
        infer_config(l1, l2, test_stocks, device,
                     batch_size=args.batch_size, resume=args.resume, seed=seed)

    # 汇总 metrics
    print("\n" + "="*60)
    print("Computing summary metrics ...")
    metrics = {}
    for l1, l2 in configs:
        key = config_key(l1, l2)
        m = compute_metrics_from_npz(key, seed)
        if m:
            metrics[key] = m
            print(f"  {key}: n_pred={m['n_predictions']}, "
                  f"avgDA={m['avg_da_per_date']*100:.2f}%, "
                  f"Coll={m['collapse_rate']*100:.1f}%, "
                  f"Uniq={m['n_unique_tokens']}, RankIC={m['rank_ic']:.4f}")

    with open(metrics_path(seed), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path(seed)}")
    print(f"Predictions dir: {full_eval_dir(seed)}")


if __name__ == "__main__":
    main()
