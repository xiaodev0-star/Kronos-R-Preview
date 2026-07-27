"""统一评估入口：GPT / GPT+BERT 评估，通过 --mode 切换。

Modes:
  windowed  多日滑动窗口评估（默认，主要评估工具）
  grpo      全量单点评估 + per-stock RankIC + MAPE
  bert      GPT 提案 top-K + BERT V2 一致性校准

Usage:
    # 多日窗口评估（推荐）
    python eval.py windowed --gpt_ckpt ckpt.pt --tokenizer tok.pt

    # GRPO 全量评估
    python eval.py grpo --gpt_ckpt ckpt.pt --tokenizer tok.pt

    # GPT + BERT 双模型评估
    python eval.py bert --gpt_ckpt gpt.pt --bert_ckpt bert.pt --tokenizer tok.pt
"""
import argparse
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())


# ═══════════════════════════════════════════════════════════════════
#  Mode: windowed — 多日滑动窗口评估
# ═══════════════════════════════════════════════════════════════════

def run_windowed(args):
    import json
    from config import set_global_seed
    import torch

    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from eval_helpers import evaluate_windowed
    metrics = evaluate_windowed(
        args.gpt_ckpt, args.tokenizer, device,
        n_stocks=args.n_stocks, n_days=args.n_days,
        batch_size=args.batch_size, start_offset=args.start_offset,
        silent=False, seed=args.seed, sample_strategy=args.sample_strategy)

    if args.output:
        output = (dict(metrics) if args.include_per_date
                  else {k: v for k, v in metrics.items() if k != "per_date"})
        output["mode"] = "windowed"
        output["gpt_ckpt"] = args.gpt_ckpt
        output["tokenizer"] = args.tokenizer
        output["seed"] = args.seed
        output["n_stocks_requested"] = args.n_stocks
        output["n_days_requested"] = args.n_days
        output["start_offset"] = args.start_offset
        output["sample_strategy"] = args.sample_strategy
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {args.output}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Mode: grpo — 全量单点评估
# ═══════════════════════════════════════════════════════════════════

def run_grpo(args):
    import json
    import time
    import numpy as np
    import torch
    from collections import defaultdict
    from scipy.stats import spearmanr
    from tqdm import tqdm

    from config import DataConfig, ModelConfig, set_global_seed
    from data_processor import load_stocks, split_stocks
    from eval_helpers import (
        AMP_DTYPE, _bucket_by_length, decode_code_ids_tensor,
        attach_close_prices, predict_selected_ids, load_gpt, load_tokenizer,
        _prepare_stocks_batch,
    )

    SEED = args.seed
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPT: {args.gpt_ckpt}")
    print(f"Tokenizer: {args.tokenizer}")

    # Tokenizer
    tok = load_tokenizer(args.tokenizer, device)
    vocab = tok.vocab_coarse
    print(f"  Vocab: L1={tok.bits_l1}b (vocab={vocab}), "
          f"L2={tok.bits_l2}b (vocab={tok.bsq_fine.vocab_size})")

    # Stocks
    DataConfig.max_stocks = args.max_stocks
    stocks = load_stocks(max_stocks=args.max_stocks)
    _, _, test_stocks = split_stocks(stocks)
    if args.max_stocks > 0:
        import random
        random.seed(SEED)
        test_stocks = random.sample(test_stocks, min(len(test_stocks), args.max_stocks))
    attach_close_prices(test_stocks)
    print(f"  Test stocks: {len(test_stocks)}")

    # Model
    ModelConfig.vocab_size = vocab
    gpt = load_gpt(args.gpt_ckpt, device, tokenizer=tok)
    gpt.eval()
    n_params = sum(p.numel() for p in gpt.parameters())
    print(f"  Model params: {n_params:,}")

    # Prepare
    prepped = _prepare_stocks_batch(test_stocks, tok, device)
    valid = [p for p in prepped if p is not None]
    buckets = _bucket_by_length(valid, tolerance=100)
    print(f"  Valid: {len(valid)}, Buckets: {len(buckets)}, bs={args.batch_size}")

    # Forward
    records = []
    n_forward = 0
    t0 = time.time()

    for bucket in tqdm(buckets, desc="Buckets", ncols=70):
        for batch_start in range(0, len(bucket), args.batch_size):
            batch = bucket[batch_start:batch_start + args.batch_size]
            B = len(batch)
            max_len = max(s["seq_len"] for s in batch)

            try:
                inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
                tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
                pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
                va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)

                for j, s in enumerate(batch):
                    L = s["seq_len"]
                    inp[j, :L] = torch.tensor(s["inp_ids"], dtype=torch.long)
                    tids[j, :L, 0] = torch.tensor(s["day"], dtype=torch.long)
                    tids[j, :L, 1] = torch.tensor(s["month"], dtype=torch.long)
                    tids[j, :L, 2] = torch.tensor(s["year"], dtype=torch.long)
                    va[j, :L] = torch.tensor(s["va"], dtype=torch.float32)

                selected_rows = []
                selected_positions = []
                selected_meta = []
                for j, s in enumerate(batch):
                    p_start = s["test_pos"]
                    # The full forward's fine head spans max_len - 1 positions;
                    # that bound is known from the input without running it.
                    seq_len_real = min(inp.shape[1] - 1, s["seq_len"])
                    for offset in range(seq_len_real - p_start):
                        p = p_start + offset
                        selected_rows.append(j)
                        selected_positions.append(p)
                        selected_meta.append((s, p))

                coarse_ids, fine_ids = predict_selected_ids(
                    gpt, inp, tids, pos, va,
                    selected_rows, selected_positions, tok, device,
                )
                n_forward += 1
                id_pairs = torch.stack(
                    (coarse_ids, fine_ids), dim=1
                ).cpu().tolist()
                for ids, (stock, position) in zip(
                    id_pairs, selected_meta
                ):
                    date_key = (
                        stock["dates_raw"][position]
                        if position < len(stock["dates_raw"])
                        else "unknown"
                    )
                    records.append({
                        "coarse_id": int(ids[0]),
                        "fine_id": int(ids[1]),
                        "test_pos": position,
                        "date_key": date_key,
                        "p_mean": stock["p_mean"],
                        "p_std": stock["p_std"],
                        "feat": stock["feat"],
                        "close": stock["close"],
                    })

                del inp, tids, pos, va
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [OOM] Skipping batch ({max_len} tokens, bs={B})")
                continue

    elapsed = time.time() - t0
    print(f"\nForward: {n_forward}, Records: {len(records)}, Time: {elapsed:.0f}s")

    if not records:
        print("No predictions!")
        return None

    # Batch decode
    print("Batch decoding...")
    decode_bs = 10000
    for start in range(0, len(records), decode_bs):
        end = min(start + decode_bs, len(records))
        chunk = records[start:end]
        decoded = decode_code_ids_tensor(
            [r["coarse_id"] for r in chunk],
            [r["fine_id"] for r in chunk],
            tok,
            device,
        ).cpu().numpy()
        for i, r in enumerate(chunk):
            dec = decoded[i]
            tp = r["test_pos"]
            feat, close = r["feat"], r["close"]
            n_close = len(close)
            r["pred_logret"] = float(dec[0]) * r["p_std"][0] + r["p_mean"][0]
            r["true_logret"] = float(feat[tp, 0]) if tp < len(feat) else 0.0
            r["base_close"] = float(close[tp - 1]) if 0 <= tp - 1 < n_close else float(close[-1])
            r["true_close"] = float(close[tp]) if 0 <= tp < n_close else float(close[-1])
            del r["fine_id"], r["feat"], r["close"]

    # Metrics
    pred_lrs = np.array([r["pred_logret"] for r in records])
    true_lrs = np.array([r["true_logret"] for r in records])
    token_ids = np.array([r["coarse_id"] for r in records], dtype=np.int64)
    eps = 1e-8

    # Per-date DA
    by_date = defaultdict(list)
    for r in records:
        by_date[r["date_key"]].append((r["pred_logret"], r["true_logret"]))

    per_date_da = {}
    for dk, group in sorted(by_date.items()):
        pl = np.array([g[0] for g in group])
        tl = np.array([g[1] for g in group])
        if len(pl) >= 5:
            per_date_da[dk] = float((np.sign(pl) == np.sign(tl)).mean())

    avg_da = float(np.mean(list(per_date_da.values()))) if per_date_da else 0.0
    da_std = float(np.std(list(per_date_da.values()))) if len(per_date_da) > 1 else 0.0

    # Collapse & Unique
    unique, counts = np.unique(token_ids, return_counts=True)
    collapse_rate = float(counts.max() / max(len(token_ids), 1))
    n_unique = int(len(unique))

    # RankIC
    valid_idx = ~np.isnan(pred_lrs) & ~np.isnan(true_lrs)
    rank_ic = float(spearmanr(pred_lrs[valid_idx], true_lrs[valid_idx])[0]) if valid_idx.sum() > 2 else 0.0
    rank_ic = 0.0 if np.isnan(rank_ic) else rank_ic

    # AmpRatio & MAPE
    amp_ratio = float(np.mean(np.abs(pred_lrs)) / max(np.mean(np.abs(true_lrs)), eps))
    base_closes = np.array([r["base_close"] for r in records])
    true_closes = np.array([r["true_close"] for r in records])
    pred_prices = base_closes * np.exp(pred_lrs.astype(np.float64))
    mape = float(np.mean(np.abs(pred_prices - true_closes) / np.maximum(np.abs(true_closes), eps))) * 100

    results = {
        "mode": "grpo",
        "checkpoint": os.path.basename(args.gpt_ckpt),
        "n_predictions": len(records),
        "n_stocks": len(valid),
        "n_dates": len(per_date_da),
        "DA": round(avg_da, 4),
        "DA_std": round(da_std, 4),
        "MAPE": round(mape, 4),
        "Collapse%": round(collapse_rate * 100, 2),
        "UniqueTokens": n_unique,
        "AmpRatio": round(amp_ratio, 4),
        "RankIC": round(rank_ic, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"\n{'='*60}")
    print(f"  GRPO Evaluation: {os.path.basename(args.gpt_ckpt)}")
    print(f"{'='*60}")
    print(f"  Predictions:   {results['n_predictions']:,}")
    print(f"  Test stocks:   {results['n_stocks']}")
    print(f"  Trading dates: {results['n_dates']}")
    print(f"  DA (per-date): {results['DA']*100:.2f}% ± {results['DA_std']*100:.2f}%")
    print(f"  MAPE:          {results['MAPE']:.2f}%")
    print(f"  Collapse%:     {results['Collapse%']:.2f}%")
    print(f"  Unique Tokens: {results['UniqueTokens']} / {vocab}")
    print(f"  AmpRatio:      {results['AmpRatio']:.3f}x")
    print(f"  RankIC:        {results['RankIC']:.4f}")
    print(f"  Elapsed:       {results['elapsed_sec']:.0f}s")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {args.output}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Mode: bert — GPT 提案 + BERT V2 一致性校准
# ═══════════════════════════════════════════════════════════════════

def run_bert(args):
    import json
    import time
    import types
    import warnings
    import numpy as np
    import torch
    import torch.nn.functional as F
    import pandas as pd
    from glob import glob

    warnings.filterwarnings("ignore")

    from config import DataConfig, ModelConfig, NormConfig, set_global_seed
    from data_processor import load_stocks, split_stocks
    from model import load_tokenizer as _load_tok
    from model.kronos_preview import KronosPreview
    from model.kronos_bert import KronosBert
    from data_processor import document_normalize

    SEED = args.seed
    AMP_DTYPE = torch.bfloat16

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"  K={args.K}, mask_strategy={args.mask_strategy}, combine={args.combine}")

    # Load tokenizer
    tokenizer = _load_tok(args.tokenizer, device)
    vocab_base = tokenizer.vocab_size
    bos_id = vocab_base
    mask_id = vocab_base + 2
    print(f"  vocab_size={vocab_base}, BOS={bos_id}, MASK={mask_id}")
    ModelConfig.vocab_size = vocab_base

    # Load GPT
    gpt_ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    gpt = KronosPreview().to(device)
    gpt.load_state_dict(gpt_ckpt["model_state_dict"], strict=False)
    gpt.eval()
    print(f"  GPT params: {sum(p.numel() for p in gpt.parameters()):,}")

    # Load BERT
    bert_ckpt = torch.load(args.bert_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = bert_ckpt.get("config", {})
    cfg = types.SimpleNamespace(**{k: v for k, v in vars(ModelConfig).items()
                                   if not k.startswith("_")})
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    bert = KronosBert(cfg=cfg).to(device)
    bert.load_state_dict(bert_ckpt["model_state_dict"], strict=False)
    bert.eval()
    print(f"  BERT params: {sum(p.numel() for p in bert.parameters()):,}")

    # Helper functions (inline from eval_bert_calibration_v2.py)
    def _cutoff_idx(stock):
        return int(np.searchsorted(stock["dates_dt"],
                                   np.datetime64(pd.Timestamp(DataConfig.cutoff_date)),
                                   side="left"))

    def _build_stock_arrays(stock):
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        close = stock["close_prices"]
        ci = _cutoff_idx(stock)
        T_total = len(feat)
        m = NormConfig.min_lookback
        if T_total < m + 10 or ci < m:
            return None
        price_feat = feat[:, :4]
        price_normed, va_normed = document_normalize(feat, cutoff_idx=ci)
        p_mean = price_feat[:ci].mean(axis=0)
        p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)
        return {"feat": feat, "close": close, "ci": ci, "T_total": T_total,
                "p_mean": p_mean, "p_std": p_std,
                "price_normed": price_normed, "va_normed": va_normed,
                "day": day, "month": month, "year": year}

    def _build_gpt_inputs(arrays, tok, dev):
        price_normed = arrays["price_normed"]
        day, month, year = arrays["day"], arrays["month"], arrays["year"]
        T_total = arrays["T_total"]
        ci = arrays["ci"]
        idx_c, _ = tok.encode(torch.from_numpy(price_normed).float().unsqueeze(0).to(dev))
        token_ids = idx_c[0].cpu().numpy()
        N = T_total
        ids = [vocab_base] + token_ids[:N].tolist()
        day_list = [day[0]] + day[:N].tolist()
        month_list = [month[0]] + month[:N].tolist()
        year_list = [year[0]] + year[:N].tolist()
        S = len(ids)
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
        tids = torch.stack([
            torch.tensor([day_list[:-1]], dtype=torch.long),
            torch.tensor([month_list[:-1]], dtype=torch.long),
            torch.tensor([year_list[:-1]], dtype=torch.long),
        ], dim=-1).to(dev)
        pos = torch.arange(S - 1, device=dev).unsqueeze(0)
        va_seq = np.concatenate([np.zeros((1, 2), dtype=np.float32),
                                 arrays["va_normed"][:N - 1]], axis=0)
        va = torch.tensor(va_seq, dtype=torch.float32, device=dev).unsqueeze(0)
        return {"inp": inp, "tids": tids, "pos": pos, "mask": None,
                "va_values": va, "S": S, "token_ids": token_ids,
                "test_start": ci, "test_end": T_total - 2}

    def _build_bert_inputs(arrays, prefix_len, candidate_token_id,
                           mask_positions, token_ids, bos_id, mask_id):
        day, month, year = arrays["day"], arrays["month"], arrays["year"]
        prefix = [bos_id] + token_ids[:prefix_len].tolist()
        inp_list = list(prefix)
        for mp in mask_positions:
            inp_list[mp] = mask_id
        inp_list.append(int(candidate_token_id))
        S = len(inp_list)
        inp = torch.tensor([inp_list], dtype=torch.long)
        day_list = [int(day[0])] + [int(day[i]) for i in range(prefix_len)] + [int(day[prefix_len])]
        month_list = [int(month[0])] + [int(month[i]) for i in range(prefix_len)] + [int(month[prefix_len])]
        year_list = [int(year[0])] + [int(year[i]) for i in range(prefix_len)] + [int(year[prefix_len])]
        tids = torch.tensor([list(zip(day_list, month_list, year_list))], dtype=torch.long)
        pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
        va_data = arrays["va_normed"]
        va_seq = np.concatenate([np.zeros((1, 2), dtype=np.float32),
                                 va_data[:prefix_len],
                                 va_data[prefix_len:prefix_len + 1]], axis=0)
        va = torch.tensor(va_seq, dtype=torch.float32).unsqueeze(0)
        return {"inp": inp, "tids": tids, "pos": pos, "va_values": va, "S": S}

    def _bert_validate(bert, token_ids, candidates, mask_positions, p_value,
                       device, arrays, bos_id, mask_id):
        K = len(candidates)
        inputs = []
        for cand in candidates:
            ba = _build_bert_inputs(arrays, p_value, cand, mask_positions,
                                    token_ids, bos_id, mask_id)
            inputs.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in ba.items()})
        max_len = max(it["S"] for it in inputs)
        inp_t = torch.zeros(K, max_len, dtype=torch.long, device=device)
        tids = torch.zeros(K, max_len, 3, dtype=torch.long, device=device)
        pos = torch.zeros(K, max_len, dtype=torch.long, device=device)
        va = torch.zeros(K, max_len, 2, device=device)
        for bi, it in enumerate(inputs):
            L = it["S"]
            inp_t[bi, :L] = it["inp"][0]
            tids[bi, :L] = it["tids"][0]
            pos[bi, :L] = it["pos"][0]
            va[bi, :L] = it["va_values"][0]
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            logits = bert(inp_t, tids, pos, va_values=va)
        scores = np.zeros(K, dtype=np.float32)
        n_masks = len(mask_positions)
        for bi in range(K):
            total = 0.0
            for mp in mask_positions:
                probs = F.softmax(logits[bi, mp].float(), dim=-1).cpu().numpy()
                original_token = int(token_ids[mp - 1])
                total += probs[original_token]
            scores[bi] = total / n_masks
        return scores

    def _decode_token(token_id, tok, dev):
        pred_indices = (torch.tensor([token_id], dtype=torch.long, device=dev)
                        .unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous())
        with torch.no_grad():
            pred_feat = tok.decode_all(pred_indices)[0].cpu().numpy()
        return pred_feat[0]

    # Load stocks
    print("Loading test stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)

    # Attach close prices
    csv_map = {os.path.basename(f).split(".")[0]: f for f in sorted(glob("dataset/*.csv"))}
    for s in test_stocks_all:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            s["close_prices"] = df["close"].values.astype(np.float64)
        else:
            lr = s["features_raw"][:, 0]
            s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks_all),
                         min(args.n_stocks, len(test_stocks_all)), replace=False)
    test_stocks = [test_stocks_all[i] for i in sorted(indices)]
    print(f"  Test stocks: {len(test_stocks)}")

    # Evaluate
    pred_toks, pred_lrs, true_lrs = [], [], []
    base_closes, true_closes, was_resampled = [], [], []
    n_done = 0
    t0 = time.time()

    for si, stock in enumerate(test_stocks):
        arrays = _build_stock_arrays(stock)
        if arrays is None:
            continue

        inputs = _build_gpt_inputs(arrays, tokenizer, device)
        test_start = inputs["test_start"]
        test_end = inputs["test_end"]
        if test_end <= test_start:
            continue

        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc = gpt(inputs["inp"], inputs["tids"], inputs["pos"],
                     inputs["mask"], va_values=inputs["va_values"])
        gpt_logits = lc.float().cpu()
        token_ids = inputs["token_ids"]
        p_mean, p_std = arrays["p_mean"], arrays["p_std"]
        feat, close = arrays["feat"], arrays["close"]

        if args.max_test_pos > 0 and (test_end - test_start + 1) > args.max_test_pos:
            test_end = test_start + args.max_test_pos - 1

        for p in range(test_start, test_end + 1):
            gpt_lp = gpt_logits[0, p, :vocab_base].float()
            gpt_probs = F.softmax(gpt_lp, dim=-1).numpy()
            top_k_indices = np.argpartition(-gpt_probs, args.K)[:args.K]
            top_k_indices = top_k_indices[np.argsort(-gpt_probs[top_k_indices])]
            candidates = top_k_indices.tolist()
            candidate_probs = gpt_probs[candidates]

            if args.mask_strategy == "boundary":
                mask_positions = [p]
            elif args.mask_strategy == "middle":
                mask_positions = [max(1, p // 2)]
            else:
                rng_local = np.random.RandomState(p)
                mask_pool = list(range(1, p + 1))
                if len(mask_pool) > args.n_masks:
                    mask_positions = sorted(
                        rng_local.choice(mask_pool, args.n_masks, replace=False).tolist())
                else:
                    mask_positions = mask_pool

            bert_scores = _bert_validate(bert, token_ids, candidates, mask_positions,
                                         p, device, arrays=arrays, bos_id=bos_id, mask_id=mask_id)

            if args.combine == "product":
                final_scores = candidate_probs * bert_scores
            elif args.combine == "bert_only":
                final_scores = bert_scores
            else:
                final_scores = candidate_probs

            best_idx = int(np.argmax(final_scores))
            chosen_token = candidates[best_idx]
            gpt_argmax = int(np.argmax(gpt_probs))
            was_resampled.append(1 if chosen_token != gpt_argmax else 0)

            pred_feat = _decode_token(chosen_token, tokenizer, device)
            pred_lr = pred_feat[0] * p_std[0] + p_mean[0]

            pred_toks.append(chosen_token)
            pred_lrs.append(pred_lr)
            true_lrs.append(feat[p + 1, 0])
            base_closes.append(close[p])
            true_closes.append(close[p + 1])

        n_done += 1
        if (si + 1) % 5 == 0 or si == len(test_stocks) - 1:
            elapsed = time.time() - t0
            n_total = len(pred_lrs)
            if n_total > 0:
                running_da = (np.sign(np.array(pred_lrs)) == np.sign(np.array(true_lrs))).mean() * 100
            print(f"  [{si+1}/{len(test_stocks)}] elapsed={elapsed:.0f}s n_pred={n_total} "
                  f"DA={running_da:.2f}% resampled={np.mean(was_resampled)*100:.1f}%", flush=True)

    # Aggregate
    pred_lr_arr = np.array(pred_lrs)
    true_lr_arr = np.array(true_lrs)
    pred_toks_arr = np.array(pred_toks, dtype=np.int64)
    base_close_arr = np.array(base_closes)
    true_close_arr = np.array(true_closes)
    eps = 1e-8

    if len(pred_lr_arr) == 0:
        print("No predictions!")
        return None

    avg_da = (np.sign(pred_lr_arr) == np.sign(true_lr_arr)).mean()
    pred_close_arr = base_close_arr * np.exp(pred_lr_arr.astype(np.float64))
    mape_pt = np.abs(pred_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps) * 100
    avg_mape = mape_pt.mean()
    avg_ampratio = np.mean(np.abs(pred_lr_arr)) / max(np.mean(np.abs(true_lr_arr)), eps)
    unique, counts = np.unique(pred_toks_arr, return_counts=True)
    collapse = counts.max() / len(pred_toks_arr)
    n_unique = len(unique)
    top_tok = int(unique[np.argmax(counts)])

    results = {
        "mode": "bert",
        "gpt_ckpt": args.gpt_ckpt,
        "bert_ckpt": args.bert_ckpt,
        "K": args.K,
        "mask_strategy": args.mask_strategy,
        "combine": args.combine,
        "n_stocks": n_done,
        "n_predictions": len(pred_lr_arr),
        "da": float(avg_da),
        "mape": float(avg_mape),
        "ampratio": float(avg_ampratio),
        "collapse_rate": float(collapse),
        "n_unique_tokens": int(n_unique),
        "top_token": int(top_tok),
        "resample_rate": float(np.mean(was_resampled)),
    }

    print(f"\n{'='*60}")
    print(f"  BERT-V2 Calibration (n_stocks={n_done}, n_pos={len(pred_lr_arr)})")
    print(f"  K={args.K}, mask={args.mask_strategy}, combine={args.combine}")
    print(f"{'='*60}")
    print(f"  DA:           {avg_da*100:.2f}%")
    print(f"  MAPE:         {avg_mape:.2f}%")
    print(f"  AmpRatio:     {avg_ampratio:.3f}x")
    print(f"  Collapse:     {collapse*100:.1f}%")
    print(f"  Unique:       {n_unique}")
    print(f"  Resampled:    {np.mean(was_resampled)*100:.1f}%")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {args.output}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation: GPT / GPT+BERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("mode", choices=["windowed", "grpo", "bert"],
                        help="Evaluation mode")

    # Shared args
    parser.add_argument("--gpt_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    # windowed args
    parser.add_argument("--n_stocks", type=int, default=0,
                        help="[windowed/bert] 0=all test stocks")
    parser.add_argument("--n_days", type=int, default=20,
                        help="[windowed] days per stock")
    parser.add_argument("--start_offset", type=int, default=0,
                        help="[windowed] start this many test observations after cutoff")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--include_per_date", action="store_true",
                        help="[windowed] retain per-date metrics in the output JSON")
    parser.add_argument("--sample_strategy", choices=["random", "shortest"],
                        default="random",
                        help="[windowed] stock sampling; shortest is for smoke tests")

    # grpo args
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="[grpo] 0=all test stocks")

    # bert args
    parser.add_argument("--bert_ckpt", type=str, default="",
                        help="[bert] BERT checkpoint path")
    parser.add_argument("--K", type=int, default=20,
                        help="[bert] GPT top-K candidates")
    parser.add_argument("--mask_strategy", type=str, default="boundary",
                        choices=["boundary", "middle", "random_k", "all_history"])
    parser.add_argument("--n_masks", type=int, default=4,
                        help="[bert] masks for random_k / all_history")
    parser.add_argument("--combine", type=str, default="product",
                        choices=["product", "bert_only", "gpt_only"])
    parser.add_argument("--max_test_pos", type=int, default=0,
                        help="[bert] limit test positions per stock")

    args = parser.parse_args()

    if args.mode == "bert" and not args.bert_ckpt:
        parser.error("--bert_ckpt is required for bert mode")

    runners = {
        "windowed": run_windowed,
        "grpo": run_grpo,
        "bert": run_bert,
    }
    runners[args.mode](args)


if __name__ == "__main__":
    main()
