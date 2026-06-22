"""BERT-consistency v2: GPT proposes candidates, BERT validates by checking middle-token interpretability.

Algorithm (per test position p, where we want to predict tok_p):

1. GPT top-K: get top-K candidates {y_1, ..., y_K} and their probabilities.
2. For each candidate y_k, construct BERT input:
       [BOS, tok_0, tok_1, ..., tok_{p-2}, MASK, y_k]
   where:
     - length = p + 2 (BOS + p history tokens + MASK + y_k)
     - MASK at position p replaces tok_{p-1} (the last history token)
     - y_k is appended as "future" information
3. BERT predicts at MASK position:
       score_k = P_BERT(tok_{p-1} | [BOS, tok_0, ..., MASK, y_k])
   This is "how well can BERT reconstruct the original history token,
   given y_k as the future?"
4. Combine scores:
       final_score_k = P_GPT(y_k) * score_k
   Pick y* = argmax_k final_score_k.

Interpretation: BERT validates that the proposed continuation y_k is consistent
with the history. If y_k "fits naturally", the middle token (tok_{p-1}) remains
predictable. If y_k disrupts the sequence, BERT's confidence in tok_{p-1} drops.

This is V2 of the BERT calibration; V1 used joint scoring P_GPT^α * P_BERT^(1-α)
at the prediction position. V2 uses BERT purely as a consistency checker for
GPT's proposed candidate tokens, with y_k playing the role of "future" info.

Usage:
    python eval_bert_calibration_v2.py
    python eval_bert_calibration_v2.py --K 10
    python eval_bert_calibration_v2.py --K 20 --mask_pos boundary
"""
import argparse
import os
import sys
import json
import time
import types
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from glob import glob

from config import DataConfig, ModelConfig, NormConfig, set_global_seed
from data_processor import load_stocks, split_stocks
from model import load_tokenizer
from model.kronos_preview import KronosPreview
from model.kronos_bert import KronosBert
from eval_helpers import (
    load_gpt, attach_close_prices, build_stock_arrays,
    build_gpt_eval_inputs, decode_predicted_token, get_gpt_full_seq_logits,
)

N_TEST_STOCKS = 30
SEED = 42
AMP_DTYPE = torch.bfloat16
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Special token IDs — derived from tokenizer at runtime (not hardcoded)
# BOS_ID = tokenizer.vocab_size
# EOS_ID = tokenizer.vocab_size + 1
# MASK_ID = tokenizer.vocab_size + 2
# VOCAB_BASE = tokenizer.vocab_size


# ============================================================================
# Model loading
# ============================================================================

def load_gpt(path, device):
    """Load a KronosPreview GPT model for evaluation."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_bert(path, device):
    """Load a KronosBert calibrator for evaluation (reads size config from ckpt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = types.SimpleNamespace(**{k: v for k, v in vars(ModelConfig).items()
                                   if not k.startswith("_")})
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    model = KronosBert(cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


# ============================================================================
# Evaluation data preparation
# ============================================================================

def attach_close_prices(test_stocks):
    """Attach `close_prices` [T] float64 to each stock by re-reading its CSV."""
    csv_map = {os.path.basename(f).split(".")[0]: f
               for f in sorted(glob("dataset/*.csv"))}
    for s in test_stocks:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            s["close_prices"] = df["close"].values.astype(np.float64)
        else:
            lr = s["features_raw"][:, 0]
            s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)


def _cutoff_idx(stock):
    return int(np.searchsorted(stock["dates_dt"],
                               np.datetime64(pd.Timestamp(DataConfig.cutoff_date)),
                               side="left"))


def build_stock_arrays(stock):
    """Build per-stock arrays matching the v2 training pipeline.

    Returns dict with token_ids, day/month/year, va_values, normalization stats,
    test_start/test_end, and raw features. None if stock is too short.
    """
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    close = stock["close_prices"]
    ci = _cutoff_idx(stock)
    T_total = len(feat)
    m = NormConfig.min_lookback
    if T_total < m + 10 or ci < m:
        return None

    price_feat = feat[:, :4]  # [T, 4] OHLC
    price_normed, va_normed = document_normalize(feat, cutoff_idx=ci)
    p_mean = price_feat[:ci].mean(axis=0)
    p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)

    return {
        "feat": feat,
        "close": close,
        "ci": ci,
        "T_total": T_total,
        "p_mean": p_mean,
        "p_std": p_std,
        "price_normed": price_normed,
        "va_normed": va_normed,
        "day": day,
        "month": month,
        "year": year,
    }


def build_gpt_eval_inputs(arrays, tokenizer, device):
    """Build the GPT forward inputs matching v2 training exactly.

    Returns dict with inp, tids, pos, mask, va_values, S, token_ids,
    test_start, test_end.
    """
    price_normed = arrays["price_normed"]
    day, month, year = arrays["day"], arrays["month"], arrays["year"]
    T_total = arrays["T_total"]
    ci = arrays["ci"]

    idx_c, _ = tokenizer.encode(
        torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
    token_ids = idx_c[0].cpu().numpy()  # [T_total]
    vocab = tokenizer.vocab_size
    bos_id = vocab

    N = T_total
    ids = [bos_id] + token_ids[:N].tolist()
    day_list = [day[0]] + day[:N].tolist()
    month_list = [month[0]] + month[:N].tolist()
    year_list = [year[0]] + year[:N].tolist()
    S = len(ids)

    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tids = torch.stack([
        torch.tensor([day_list[:-1]], dtype=torch.long),
        torch.tensor([month_list[:-1]], dtype=torch.long),
        torch.tensor([year_list[:-1]], dtype=torch.long),
    ], dim=-1).to(device)
    pos = torch.arange(S - 1, device=device).unsqueeze(0)
    mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))

    # VA: [S-1, 2] matching v2 packing: BOS=zero, then va_normed[0..T-2]
    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        arrays["va_normed"][:N - 1],
    ], axis=0)
    va = torch.tensor(va_seq, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "inp": inp,
        "tids": tids,
        "pos": pos,
        "mask": mask,
        "va_values": va,
        "S": S,
        "token_ids": token_ids,
        "test_start": ci,
        "test_end": T_total - 2,
    }


def build_bert_position_arrays(arrays, prefix_len, candidate_token_id,
                                mask_positions, token_ids=None,
                                bos_id=0, mask_id=0):
    """Build a [1, S] BERT forward input for one (prefix_len, candidate, mask) triple.

    Mask positions are replaced with mask_id. Per-position day/month/year from
    the original stock are used (NOT the predicted-position's date copied to all).
    """
    day, month, year = arrays["day"], arrays["month"], arrays["year"]
    if token_ids is None:
        token_ids = np.full(prefix_len, bos_id, dtype=np.int64)

    # prefix: [BOS, tok_0, ..., tok_{prefix_len-1}]
    prefix = [bos_id] + token_ids[:prefix_len].tolist()
    inp_list = list(prefix)
    for mp in mask_positions:
        if mp < 0 or mp >= len(inp_list):
            raise ValueError(f"mask_pos {mp} out of range [0, {len(inp_list)})")
        inp_list[mp] = mask_id
    inp_list.append(int(candidate_token_id))
    S = len(inp_list)
    inp = torch.tensor([inp_list], dtype=torch.long)

    # Per-position dates from original stock sequence
    day_list = [int(day[0])] + [int(day[i]) for i in range(prefix_len)] + [int(day[prefix_len])]
    month_list = [int(month[0])] + [int(month[i]) for i in range(prefix_len)] + [int(month[prefix_len])]
    year_list = [int(year[0])] + [int(year[i]) for i in range(prefix_len)] + [int(year[prefix_len])]
    tids = torch.tensor([list(zip(day_list, month_list, year_list))], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0)

    # VA: BOS=zero, history from stock, candidate from predicted day
    va_data = arrays["va_normed"]
    va_prefix = va_data[:prefix_len]
    va_candidate = va_data[prefix_len:prefix_len + 1]
    va_seq = np.concatenate([
        np.zeros((1, 2), dtype=np.float32),
        va_prefix,
        va_candidate,
    ], axis=0)
    va = torch.tensor(va_seq, dtype=torch.float32).unsqueeze(0)
    return {
        "inp": inp,
        "tids": tids,
        "pos": pos,
        "va_values": va,
        "S": S,
    }


# ============================================================================
# Core evaluation logic
# ============================================================================

@torch.no_grad()
def get_gpt_full_seq_logits(gpt, tokenizer, stock, device):
    """Run GPT once on the full sequence. Returns logits at every position."""
    arrays = build_stock_arrays(stock)
    if arrays is None:
        return None
    inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        lc = gpt(inputs["inp"], inputs["tids"], inputs["pos"],
                 inputs["mask"], va_values=inputs["va_values"])
    return {
        "logits": lc.float().cpu(),
        "token_ids": inputs["token_ids"],
        "test_start": inputs["test_start"],
        "test_end": inputs["test_end"],
        "p_mean": arrays["p_mean"],
        "p_std": arrays["p_std"],
        "day": arrays["day"], "month": arrays["month"], "year": arrays["year"],
        "feat": arrays["feat"],
        "close": arrays["close"],
        "T_total": arrays["T_total"],
        "_arrays": arrays,
    }


@torch.no_grad()
def bert_validate_candidates(bert, token_ids, candidates, mask_positions,
                             p_value, device, arrays, bos_id=0, mask_id=0):
    """Run BERT validation for one test position against K candidates.

    Returns scores [K] = mean probability of original token at each masked position.
    """
    K = len(candidates)
    inputs = []
    for cand in candidates:
        ba = build_bert_position_arrays(arrays, p_value, cand, mask_positions,
                                        token_ids=token_ids, bos_id=bos_id, mask_id=mask_id)
        inputs.append({
            "inp": ba["inp"].to(device),
            "tids": ba["tids"].to(device),
            "pos": ba["pos"].to(device),
            "va_values": ba["va_values"].to(device),
            "S": ba["S"],
        })
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


def decode_predicted_token(token_id, tokenizer, device):
    """Decode a single predicted token to a feature vector."""
    pred_indices = (torch.tensor([token_id], dtype=torch.long, device=device)
                    .unsqueeze(0).unsqueeze(-1)
                    .expand(-1, -1, 2).contiguous())
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]


# ============================================================================
# Main evaluation loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_ckpt", type=str, default="checkpoints/best_gpt_regime.pt")
    parser.add_argument("--bert_ckpt", type=str, default="checkpoints/kronos_bert_big_v1.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--K", type=int, default=20,
                        help="Number of GPT candidates to validate per test position")
    parser.add_argument("--n_stocks", type=int, default=N_TEST_STOCKS)
    parser.add_argument("--max_test_pos", type=int, default=0,
                        help="If >0, only evaluate on first N test positions per stock")
    parser.add_argument("--mask_strategy", type=str, default="boundary",
                        choices=["boundary", "middle", "random_k", "all_history"])
    parser.add_argument("--n_masks", type=int, default=4,
                        help="For random_k and all_history, how many positions to mask")
    parser.add_argument("--combine", type=str, default="product",
                        choices=["product", "bert_only", "gpt_only"])
    parser.add_argument("--output", type=str, default="eval_bert_v2_results.json")
    args = parser.parse_args()

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"  K={args.K}, mask_strategy={args.mask_strategy}, combine={args.combine}")

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(args.tokenizer, device)
    # Derive special token IDs from tokenizer (not hardcoded)
    vocab_base = tokenizer.vocab_size
    bos_id = vocab_base
    mask_id = vocab_base + 2
    print(f"  vocab_size={vocab_base}, BOS={bos_id}, MASK={mask_id}")
    ModelConfig.vocab_size = vocab_base
    print("Loading GPT ...")
    gpt = load_gpt(args.gpt_ckpt, device)
    print("Loading BERT ...")
    bert = load_bert(args.bert_ckpt, device)
    print(f"GPT params: {sum(p.numel() for p in gpt.parameters()):,}")
    print(f"BERT params: {sum(p.numel() for p in bert.parameters()):,}")

    print("Loading test stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)
    attach_close_prices(test_stocks_all)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks_all),
                         min(args.n_stocks, len(test_stocks_all)), replace=False)
    test_stocks = [test_stocks_all[i] for i in sorted(indices)]
    print(f"Test stocks: {len(test_stocks)}")

    # Accumulators
    pred_toks = []
    pred_lrs = []
    true_lrs = []
    base_closes = []
    true_closes = []
    was_resampled = []
    n_done = 0
    t0 = time.time()

    for si, stock in enumerate(test_stocks):
        info = get_gpt_full_seq_logits(gpt, tokenizer, stock, device)
        if info is None:
            continue

        test_start = info["test_start"]
        test_end = info["test_end"]
        if test_end <= test_start:
            continue

        if args.max_test_pos > 0 and (test_end - test_start + 1) > args.max_test_pos:
            test_end = test_start + args.max_test_pos - 1

        token_ids = info["token_ids"]
        p_mean = info["p_mean"]
        p_std = info["p_std"]
        feat = info["feat"]
        close = info["close"]
        gpt_logits = info["logits"]  # [1, T_total, vocab_full]
        arrays = info["_arrays"]

        for p in range(test_start, test_end + 1):
            # GPT distribution at this position
            gpt_lp = gpt_logits[0, p, :vocab_base].float()
            gpt_probs = F.softmax(gpt_lp, dim=-1).numpy()
            # Top-K candidates
            top_k_indices = np.argpartition(-gpt_probs, args.K)[:args.K]
            top_k_indices = top_k_indices[np.argsort(-gpt_probs[top_k_indices])]
            candidates = top_k_indices.tolist()
            candidate_probs = gpt_probs[candidates]

            # Decide mask positions
            if args.mask_strategy == "boundary":
                mask_positions = [p]
            elif args.mask_strategy == "middle":
                mid = max(1, p // 2)
                mask_positions = [mid]
            else:
                rng_local = np.random.RandomState(p)
                mask_pool = list(range(1, p + 1))
                if len(mask_pool) > args.n_masks:
                    mask_positions = sorted(
                        rng_local.choice(mask_pool, args.n_masks, replace=False).tolist())
                else:
                    mask_positions = mask_pool

            bert_scores = bert_validate_candidates(
                bert, token_ids, candidates, mask_positions, p, device,
                arrays=arrays, bos_id=bos_id, mask_id=mask_id)

            # Combine scores
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

            # Decode predicted token → log return
            pred_feat = decode_predicted_token(chosen_token, tokenizer, device)
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
                pred_arr = np.array(pred_lrs)
                true_arr = np.array(true_lrs)
                running_da = (np.sign(pred_arr) == np.sign(true_arr)).mean() * 100
            print(f"  [{si+1}/{len(test_stocks)}] elapsed={elapsed:.0f}s n_pred={n_total} "
                  f"running DA={running_da:.2f}% "
                  f"frac_resampled={np.mean(was_resampled)*100:.1f}%", flush=True)

    # Aggregate
    pred_lr_arr = np.array(pred_lrs)
    true_lr_arr = np.array(true_lrs)
    pred_toks_arr = np.array(pred_toks, dtype=np.int64)
    base_close_arr = np.array(base_closes)
    true_close_arr = np.array(true_closes)

    if len(pred_lr_arr) == 0:
        print("No predictions!")
        return

    # DA
    avg_da = (np.sign(pred_lr_arr) == np.sign(true_lr_arr)).mean()

    # MAPE (price space)
    pred_close_arr = base_close_arr * np.exp(pred_lr_arr.astype(np.float64))
    eps = 1e-8
    mape_pt = np.abs(pred_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps) * 100
    avg_mape = mape_pt.mean()

    # AmpRatio
    pred_amp = np.mean(np.abs(pred_lr_arr))
    true_amp = np.mean(np.abs(true_lr_arr))
    avg_ampratio = pred_amp / max(true_amp, 1e-8)

    # Collapse
    unique, counts = np.unique(pred_toks_arr, return_counts=True)
    collapse = counts.max() / len(pred_toks_arr)
    n_unique = len(unique)
    top_tok = int(unique[np.argmax(counts)])

    # Baseline MAPE
    bl_mape = np.mean(np.abs(base_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps)) * 100

    print("\n" + "=" * 100)
    print(f"  BERT-V2 CONSISTENCY CALIBRATION (n_stocks={n_done}, n_pos={len(pred_lr_arr)})")
    print(f"  K={args.K}, mask_strategy={args.mask_strategy}, combine={args.combine}")
    print("=" * 100)
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'DA':<20} {avg_da*100:>9.2f}%")
    print(f"  {'MAPE':<20} {avg_mape:>9.2f}%")
    print(f"  {'Baseline MAPE':<20} {bl_mape:>9.2f}%")
    print(f"  {'AmpRatio':<20} {avg_ampratio:>9.3f}x")
    print(f"  {'Collapse':<20} {collapse*100:>9.1f}%")
    print(f"  {'Unique tokens':<20} {n_unique:>9}")
    print(f"  {'Top token':<20} {top_tok:>9}")
    print(f"  {'Resampled':<20} {np.mean(was_resampled)*100:>9.1f}%")
    print("=" * 100)

    out = {
        "gpt_ckpt": args.gpt_ckpt,
        "bert_ckpt": args.bert_ckpt,
        "K": args.K,
        "mask_strategy": args.mask_strategy,
        "n_masks": args.n_masks,
        "combine": args.combine,
        "n_stocks": n_done,
        "n_predictions": len(pred_lr_arr),
        "metrics": {
            "da": float(avg_da),
            "mape": float(avg_mape),
            "ampratio": float(avg_ampratio),
            "baseline_mape": float(bl_mape),
            "collapse_rate": float(collapse),
            "n_unique_tokens": int(n_unique),
            "top_token": int(top_tok),
            "resample_rate": float(np.mean(was_resampled)),
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
