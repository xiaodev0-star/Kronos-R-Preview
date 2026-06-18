"""BERT-calibrated 1-step evaluation.

Compares:
  - GPT-only baseline (KronosPreview argmax)
  - GPT+BERT: P_combined = P_GPT^alpha * P_BERT^(1-alpha), argmax

For each test position t (in 1-step AR):
  GPT input:  [BOS, tok_0, ..., tok_{t-1}]              (length t, predicts tok_t)
  BERT input: [BOS, tok_0, ..., tok_{t-1}, MASK]        (length t+1, predicts at MASK)

No future context is used (true 1-step AR scenario). The bidirectional nature of BERT
still helps because it can attend bidirectionally over history (not just causally).

Usage:
    python eval_bert_calibration.py
    python eval_bert_calibration.py --alpha 0.5
    python eval_bert_calibration.py --alphas 0 0.3 0.5 0.7 1.0
"""
import argparse
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np

from data_processor import load_stocks, split_stocks
from reproducibility import set_global_seed
from eval_helpers import (
    build_stock_arrays, build_gpt_eval_inputs, build_bert_position_arrays, MASK_ID,
    load_tokenizer, load_gpt, load_bert, attach_close_prices, _cutoff_idx,
)

N_TEST_STOCKS = 30
SEED = 42
AMP_DTYPE = torch.bfloat16
BERT_BATCH_SIZE = 4  # reduced from 32 to fit big BERT (16M params) on long sequences
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@torch.no_grad()
def get_gpt_predictions_full_seq(gpt, tokenizer, stock, device):
    """Run GPT once on the full sequence (BOS + all tokens). Returns:
      - P_GPT at all positions (logits at position t predict token t+1)
      - token_ids [T_total] (no BOS)
      - test_start, test_end (in 1-step AR space: positions where t+1 <= T_total - 1)
      - p_mean, p_std (for denorm)
    """
    arrays = build_stock_arrays(stock)
    if arrays is None:
        return None
    inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        lc, _ = gpt(inputs["inp"], inputs["tids"], inputs["pos"],
                    inputs["mask"], va_values=inputs["va_values"])
    return {
        "logits": lc.float().cpu(),     # [1, S-1, vocab_full]
        "token_ids": inputs["token_ids"],
        "test_start": inputs["test_start"],
        "test_end": inputs["test_end"],
        "p_mean": arrays["p_mean"],
        "p_std": arrays["p_std"],
        "day": arrays["day"],
        "month": arrays["month"],
        "year": arrays["year"],
        "feat": arrays["feat"],
        "close": arrays["close"],
        "T_total": arrays["T_total"],
        "_arrays": arrays,  # for BERT position-arrays (per-position time/va)
    }


@torch.no_grad()
def get_bert_scores_for_positions(bert, stock_info, positions, tokenizer, device, mask_id, batch_size=32):
    """For each position t in `positions`, compute BERT([BOS, tok_0, ..., tok_{t-1}, MASK]).
    Returns tensor [len(positions), vocab_base] of P_BERT(.|history) at MASK.

    FIXED (was buggy): per-position day/month/year used; padding mask passed to BERT
    so that padded positions cannot attend each other (and the MASK prediction is
    not contaminated by padding tokens).
    """
    T_total = stock_info["T_total"]
    token_ids = stock_info["token_ids"]  # [T_total] - no BOS
    arrays = stock_info["_arrays"]

    # Build per-position sequences, then batch by length
    per_pos = []
    for t in positions:
        # The MASK is the last position in the input, length t+1
        hist = token_ids[:t].tolist()
        # prefix length (history tokens) = t  →  BERT input length = 1 (BOS) + t + 1 (MASK)
        per_pos.append((t, hist))

    # Sort by length for efficient batching
    per_pos.sort(key=lambda x: len(x[1]))

    results = np.zeros((len(positions), 1024), dtype=np.float32)  # P_BERT

    for batch_start in range(0, len(per_pos), batch_size):
        batch = per_pos[batch_start:batch_start + batch_size]
        max_len = max(len(x[1]) for x in batch) + 2  # +1 BOS +1 MASK
        B = len(batch)
        # Build padded batch — pad with 0 (a non-special, harmless token), not BOS
        inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
        tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
        pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
        va = torch.zeros(B, max_len, 2, device=device)
        real_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        mask_positions = []
        for bi, (t_orig, hist) in enumerate(batch):
            L = len(hist) + 2  # +BOS +MASK
            # BOS at position 0
            inp[bi, 0] = 1024
            tids[bi, 0, 0] = int(arrays["day"][0])
            tids[bi, 0, 1] = int(arrays["month"][0])
            tids[bi, 0, 2] = int(arrays["year"][0])
            # History
            for j, tok in enumerate(hist):
                inp[bi, 1 + j] = tok
                tids[bi, 1 + j, 0] = int(arrays["day"][j])
                tids[bi, 1 + j, 1] = int(arrays["month"][j])
                tids[bi, 1 + j, 2] = int(arrays["year"][j])
            # MASK at the last real position
            inp[bi, L - 1] = mask_id
            tids[bi, L - 1, 0] = int(arrays["day"][t_orig])  # date of predicted position
            tids[bi, L - 1, 1] = int(arrays["month"][t_orig])
            tids[bi, L - 1, 2] = int(arrays["year"][t_orig])
            pos[bi, :L] = torch.arange(L, device=device)
            real_mask[bi, :L] = True
            # VA: zero at BOS, real at history positions, real at MASK from predicted day
            va[bi, 0] = 0
            if L - 1 > 0:
                va[bi, 1:L - 1] = torch.tensor(arrays["va_normed"][:L - 2], device=device)
            # MASK at position L-1: va_normed at the predicted day
            if t_orig < len(arrays["va_normed"]):
                va[bi, L - 1] = torch.tensor(arrays["va_normed"][t_orig], device=device)
            mask_positions.append(L - 1)

        # Build attention mask so padded positions cannot attend or be attended.
        # real_mask[b, i] is True for real positions. We want attention to be
        # allowed only between real positions. Build [B, L, L] True=attend mask.
        L = max_len
        attn = real_mask.unsqueeze(1) & real_mask.unsqueeze(2)  # [B, L, L]
        # Convert to boolean; SDPA expects True=attend
        attn = attn.bool()

        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            logits = bert(inp, tids, pos, attn_mask=attn, va_values=va)  # [B, L, 1024]

        for bi, mp in enumerate(mask_positions):
            results[batch_start + bi] = F.softmax(logits[bi, mp].float(), dim=-1).cpu().numpy()

    return results, [t for t, _ in per_pos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_ckpt", type=str, default="checkpoints/expA_v2_hpo.pt")
    parser.add_argument("--bert_ckpt", type=str, default="checkpoints/kronos_bert_big_v1.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0],
                        help="alpha values: P_GPT^(1-alpha) * P_BERT^alpha (alpha=0 → GPT only, 1 → BERT only)")
    parser.add_argument("--n_stocks", type=int, default=N_TEST_STOCKS)
    parser.add_argument("--max_test_pos", type=int, default=0,
                        help="If >0, only evaluate on first N test positions per stock (for speed)")
    parser.add_argument("--output", type=str, default="eval_bert_results.json")
    args = parser.parse_args()

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(args.tokenizer, device)
    print("Loading GPT ...")
    gpt = load_gpt(args.gpt_ckpt, device)
    print("Loading BERT ...")
    bert = load_bert(args.bert_ckpt, device)
    mask_id = 1026  # KronosBert.mask_id
    vocab_base = 1024
    print(f"GPT params: {sum(p.numel() for p in gpt.parameters()):,}")
    print(f"BERT params: {sum(p.numel() for p in bert.parameters()):,}")

    print("Loading test stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)

    # Attach close_prices
    attach_close_prices(test_stocks_all)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks_all), min(args.n_stocks, len(test_stocks_all)), replace=False)
    test_stocks = [test_stocks_all[i] for i in sorted(indices)]
    print(f"Test stocks: {len(test_stocks)}")

    # For each alpha, accumulate metrics
    alpha_results = {a: {"mape": [], "da": [], "ampratio": [], "preds": [], "true_lr": [],
                          "pred_lr": [], "all_toks": []} for a in args.alphas}

    n_done = 0
    t0 = time.time()
    for si, stock in enumerate(test_stocks):
        info = get_gpt_predictions_full_seq(gpt, tokenizer, stock, device)
        if info is None:
            continue

        test_start = info["test_start"]
        test_end = info["test_end"]
        if test_end <= test_start:
            continue

        # positions in logits space: [test_start, test_end] (predicts token_ids[test_start..test_end])
        # Limit if requested
        if args.max_test_pos > 0 and (test_end - test_start + 1) > args.max_test_pos:
            test_end = test_start + args.max_test_pos - 1

        positions = list(range(test_start, test_end + 1))
        n_pos = len(positions)
        if n_pos < 5:
            continue

        # Get GPT log-probs at these positions (use log-space for numerical stability)
        gpt_logits = info["logits"][0, test_start:test_end + 1, :vocab_base]  # [n_pos, 1024]
        gpt_log_probs = F.log_softmax(gpt_logits.float(), dim=-1).numpy()  # [n_pos, 1024]

        # Get BERT probs at these positions: BERT([BOS, history, MASK]) at MASK
        bert_probs, ordered_positions = get_bert_scores_for_positions(
            bert, info, positions, tokenizer, device, mask_id, batch_size=BERT_BATCH_SIZE)
        # Reorder to match positions
        bert_probs_dict = {p: bert_probs[i] for i, p in enumerate(ordered_positions)}
        bert_probs_aligned = np.stack([bert_probs_dict[p] for p in positions], axis=0)
        bert_log_probs = np.log(np.maximum(bert_probs_aligned, 1e-12))

        # True tokens
        true_toks = info["token_ids"][test_start:test_end + 1]

        # Denormalize target log return
        feat = info["feat"]
        p_mean = info["p_mean"]
        p_std = info["p_std"]
        true_lr = feat[test_start + 1:test_end + 2, 0]

        for alpha in args.alphas:
            if alpha == 0.0:
                # GPT only
                combined_log_probs = gpt_log_probs
            elif alpha == 1.0:
                # BERT only
                combined_log_probs = bert_log_probs
            else:
                # Normalize each to roughly equal scale before combining
                # (log probabilities have different baseline magnitudes)
                gpt_lp = gpt_log_probs - gpt_log_probs.max(axis=-1, keepdims=True)
                bert_lp = bert_log_probs - bert_log_probs.max(axis=-1, keepdims=True)
                combined_log_probs = (1 - alpha) * gpt_lp + alpha * bert_lp

            pred_toks = combined_log_probs.argmax(axis=-1)  # [n_pos]
            # Decode: replicate coarse to both levels
            pred_indices = torch.from_numpy(pred_toks).to(device).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous()
            with torch.no_grad():
                pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
            pred_lr = pred_feat[:, 0] * p_std[0] + p_mean[0]

            base_close = info["close"][test_start:test_end + 1]
            pred_close = base_close * np.exp(pred_lr.astype(np.float64))
            true_close = info["close"][test_start + 1:test_end + 2]

            eps = 1e-8
            mape_pt = np.abs(pred_close - true_close) / (np.abs(true_close) + eps) * 100
            da_pt = (np.sign(pred_lr) == np.sign(true_lr)).astype(float)
            pred_amp = float(np.mean(np.abs(pred_lr)))
            true_amp = float(np.mean(np.abs(true_lr)))
            ampratio = pred_amp / max(true_amp, 1e-8)

            alpha_results[alpha]["mape"].append(float(np.mean(mape_pt)))
            alpha_results[alpha]["da"].append(float(np.mean(da_pt)))
            alpha_results[alpha]["ampratio"].append(ampratio)
            alpha_results[alpha]["preds"].extend(pred_toks.tolist())
            alpha_results[alpha]["true_lr"].extend(true_lr.tolist())
            alpha_results[alpha]["pred_lr"].extend(pred_lr.tolist())
            alpha_results[alpha]["all_toks"].extend(pred_toks.tolist())

        n_done += 1
        if (si + 1) % 5 == 0 or si == len(test_stocks) - 1:
            elapsed = time.time() - t0
            if 0.0 in alpha_results:
                avg_mape_gpt = float(np.mean(alpha_results[0.0]["mape"])) if alpha_results[0.0]["mape"] else 0
                avg_da_gpt = float(np.mean(alpha_results[0.0]["da"])) if alpha_results[0.0]["da"] else 0
                print(f"  [{si+1}/{len(test_stocks)}] elapsed={elapsed:.0f}s "
                      f"GPT-only running DA={avg_da_gpt*100:.2f}% MAPE={avg_mape_gpt:.2f}%", flush=True)
            else:
                print(f"  [{si+1}/{len(test_stocks)}] elapsed={elapsed:.0f}s", flush=True)

    # Aggregate
    print("\n" + "=" * 100)
    print(f"  BERT-CALIBRATION 1-STEP EVALUATION (n_stocks={n_done}, n_pos per stock≈{n_pos})")
    print("=" * 100)
    print(f"  {'alpha':>6} {'label':<14} {'DA':>8} {'MAPE':>8} {'AmpRatio':>9} {'Collapse':>9} {'Unique':>7} {'TopTok':>7}")
    print("  " + "-" * 90)
    final = {}
    for alpha in args.alphas:
        d = alpha_results[alpha]
        if not d["mape"]:
            continue
        all_toks = np.array(d["all_toks"])
        if len(all_toks):
            unique, counts = np.unique(all_toks, return_counts=True)
            collapse = counts.max() / len(all_toks)
            n_unique = len(unique)
            top_tok = int(unique[np.argmax(counts)])
        else:
            collapse, n_unique, top_tok = 0, 0, -1
        avg_da = float(np.mean(d["da"]))
        avg_mape = float(np.mean(d["mape"]))
        avg_ampratio = float(np.mean(d["ampratio"]))
        if alpha == 0.0:
            label = "GPT-only"
        elif alpha == 1.0:
            label = "BERT-only"
        else:
            label = f"GPT+BERT"
        print(f"  {alpha:>6.2f} {label:<14} {avg_da*100:>7.2f}% {avg_mape:>7.2f}% "
              f"{avg_ampratio:>8.3f}x {collapse*100:>8.1f}% {n_unique:>6} {top_tok:>6}")
        final[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "label": label,
            "da": avg_da,
            "mape": avg_mape,
            "ampratio": avg_ampratio,
            "collapse_rate": float(collapse),
            "n_unique_tokens": int(n_unique),
            "top_token": int(top_tok),
            "n_stocks": n_done,
        }
    # Best by DA
    best = max(final.values(), key=lambda x: x["da"])
    print(f"\n  >>> BEST BY DA: alpha={best['alpha']} [{best['label']}] "
          f"DA={best['da']*100:.2f}% MAPE={best['mape']:.2f}%")

    out = {
        "gpt_ckpt": args.gpt_ckpt,
        "bert_ckpt": args.bert_ckpt,
        "n_stocks": n_done,
        "alphas": args.alphas,
        "results": final,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
