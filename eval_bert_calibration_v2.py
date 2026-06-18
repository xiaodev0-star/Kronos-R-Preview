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
import warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from glob import glob

from config import DataConfig, NormConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, document_normalize, _stock_cutoff_idx
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from model.kronos_bert import KronosBert
from reproducibility import set_global_seed
from eval_helpers import build_stock_arrays, build_gpt_eval_inputs, build_bert_position_arrays, BOS_ID, MASK_ID, VOCAB_BASE

N_TEST_STOCKS = 30
SEED = 42
AMP_DTYPE = torch.bfloat16
BOS_ID = 1024
EOS_ID = 1025
MASK_ID = 1026
VOCAB_BASE = 1024
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def load_gpt(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_bert(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Read config from checkpoint to support different model sizes
    cfg_dict = ckpt.get("config", {})
    from config import ModelConfig as _MC
    class _Cfg:
        pass
    cfg = _Cfg()
    for attr in dir(_MC):
        if attr.startswith("_"):
            continue
        setattr(cfg, attr, getattr(_MC, attr))
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    model = KronosBert(cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _cutoff_idx(stock):
    return int(np.searchsorted(stock["dates_dt"],
                               np.datetime64(pd.Timestamp(DataConfig.cutoff_date)), side="left"))


@torch.no_grad()
def get_gpt_full_seq_logits(gpt, tokenizer, stock, device):
    """Run GPT once on the full sequence. Returns logits at every position.
    Logit at position i predicts ids[i+1] = tok_i (where ids = [BOS, tok_0, ..., tok_{T-1}]).
    """
    arrays = build_stock_arrays(stock)
    if arrays is None:
        return None
    inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        lc, _, _ = gpt(inputs["inp"], inputs["tids"], inputs["pos"],
                       inputs["mask"], va_values=inputs["va_values"])
    return {
        "logits": lc.float().cpu(),  # [1, T, vocab_full]
        "token_ids": inputs["token_ids"],
        "test_start": inputs["test_start"],
        "test_end": inputs["test_end"],
        "p_mean": arrays["p_mean"],
        "p_std": arrays["p_std"],
        "day": arrays["day"], "month": arrays["month"], "year": arrays["year"],
        "feat": arrays["feat"],
        "close": arrays["close"],
        "T_total": arrays["T_total"],
        "_arrays": arrays,  # for per-position BERT input building
    }


@torch.no_grad()
def bert_validate_candidates(bert, token_ids, candidates, mask_pos, day, month, year,
                              p_value, device, batch_size=64, arrays=None):
    """Run BERT validation for one test position p against K candidates.

    FIXED (was buggy): per-position day/month/year used (each input position gets
    its own date, not the predicted-position's date copied to all).
    `arrays` is the dict returned by build_stock_arrays and carries the per-position
    VA values; pass it through to the BERT forward.
    """
    K = len(candidates)
    # Build the K candidate-augmented inputs using the shared helper.
    inputs = []
    for cand in candidates:
        ba = build_bert_position_arrays(arrays, p_value, cand, [mask_pos],
                                         token_ids=token_ids)
        inputs.append({
            "inp": ba["inp"].to(device),
            "tids": ba["tids"].to(device),
            "pos": ba["pos"].to(device),
            "va_values": ba["va_values"].to(device),
            "S": ba["S"],
        })
    max_len = max(it["S"] for it in inputs)
    B = K
    inp_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
    tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
    pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
    va = torch.zeros(B, max_len, 2, device=device)
    for bi, it in enumerate(inputs):
        L = it["S"]
        inp_t[bi, :L] = it["inp"][0]
        tids[bi, :L] = it["tids"][0]
        pos[bi, :L] = it["pos"][0]
        va[bi, :L] = it["va_values"][0]

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        logits = bert(inp_t, tids, pos, va_values=va)  # [B, max_len, vocab_base]

    scores = np.zeros(K, dtype=np.float32)
    for bi in range(B):
        probs = F.softmax(logits[bi, mask_pos].float(), dim=-1).cpu().numpy()  # [vocab_base]
        original_token = int(token_ids[mask_pos - 1])
        scores[bi] = probs[original_token]

    return scores


@torch.no_grad()
def bert_validate_candidates_multi_mask(bert, token_ids, candidates, mask_positions,
                                          day, month, year, p_value, device, arrays=None):
    """Variant: mask MULTIPLE positions, return mean score across masked positions.
    FIXED (was buggy): per-position dates used via build_bert_position_arrays.
    """
    K = len(candidates)
    inputs = []
    for cand in candidates:
        ba = build_bert_position_arrays(arrays, p_value, cand, mask_positions,
                                         token_ids=token_ids)
        inputs.append({
            "inp": ba["inp"].to(device),
            "tids": ba["tids"].to(device),
            "pos": ba["pos"].to(device),
            "va_values": ba["va_values"].to(device),
            "S": ba["S"],
        })
    max_len = max(it["S"] for it in inputs)
    B = K
    inp_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
    tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
    pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
    va = torch.zeros(B, max_len, 2, device=device)
    for bi, it in enumerate(inputs):
        L = it["S"]
        inp_t[bi, :L] = it["inp"][0]
        tids[bi, :L] = it["tids"][0]
        pos[bi, :L] = it["pos"][0]
        va[bi, :L] = it["va_values"][0]

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        logits = bert(inp_t, tids, pos, va_values=va)

    scores = np.zeros(K, dtype=np.float32)
    for bi in range(B):
        total = 0.0
        for mp in mask_positions:
            probs = F.softmax(logits[bi, mp].float(), dim=-1).cpu().numpy()
            original_token = int(token_ids[mp - 1])
            total += probs[original_token]
        scores[bi] = total / len(mask_positions)

    return scores


def decode_predicted_token(token_id, tokenizer, device):
    """Decode a single predicted token to a feature vector."""
    pred_indices = torch.tensor([token_id], dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous()
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_ckpt", type=str, default="checkpoints/expA_v2_hpo.pt")
    parser.add_argument("--bert_ckpt", type=str, default="checkpoints/kronos_bert_big_v1.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--K", type=int, default=20,
                        help="Number of GPT candidates to validate per test position")
    parser.add_argument("--n_stocks", type=int, default=N_TEST_STOCKS)
    parser.add_argument("--max_test_pos", type=int, default=0,
                        help="If >0, only evaluate on first N test positions per stock (for speed)")
    parser.add_argument("--mask_strategy", type=str, default="boundary",
                        choices=["boundary", "middle", "random_k", "all_history"],
                        help="Where to mask the BERT input")
    parser.add_argument("--n_masks", type=int, default=4,
                        help="For random_k and all_history, how many positions to mask")
    parser.add_argument("--combine", type=str, default="product",
                        choices=["product", "bert_only", "gpt_only"],
                        help="How to combine GPT prob and BERT score")
    parser.add_argument("--output", type=str, default="eval_bert_v2_results.json")
    args = parser.parse_args()

    set_global_seed(SEED, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"  K={args.K}, mask_strategy={args.mask_strategy}, combine={args.combine}")

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(args.tokenizer, device)
    print("Loading GPT ...")
    gpt = load_gpt(args.gpt_ckpt, device)
    print("Loading BERT ...")
    bert = load_bert(args.bert_ckpt, device)
    print(f"GPT params: {sum(p.numel() for p in gpt.parameters()):,}")
    print(f"BERT params: {sum(p.numel() for p in bert.parameters()):,}")

    print("Loading test stocks ...")
    stocks = load_stocks(max_stocks=0)
    _, _, test_stocks_all = split_stocks(stocks)

    csv_map = {os.path.basename(f).split(".")[0]: f for f in sorted(glob("dataset/*.csv"))}
    for s in test_stocks_all:
        fpath = csv_map.get(s["symbol"])
        if fpath:
            df = pd.read_csv(fpath, usecols=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            prev = df["close"].shift(1)
            df["log_ret"] = np.log(df["close"] / prev).replace([np.inf, -np.inf], np.nan)
            df = df.dropna().reset_index(drop=True)
            s["close_prices"] = df["close"].values.astype(np.float64)
        else:
            lr = s["features_raw"][:, 0]
            s["close_prices"] = np.exp(np.cumsum(lr)).astype(np.float64)

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(test_stocks_all), min(args.n_stocks, len(test_stocks_all)), replace=False)
    test_stocks = [test_stocks_all[i] for i in sorted(indices)]
    print(f"Test stocks: {len(test_stocks)}")

    # Accumulators
    all_pred_toks = []
    all_pred_lr = []
    all_true_lr = []
    all_base_close = []
    all_true_close = []
    all_pred_chosen = []  # whether the chosen token was NOT GPT's argmax
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
        day = info["day"]
        month = info["month"]
        year = info["year"]
        T_total = info["T_total"]
        p_mean = info["p_mean"]
        p_std = info["p_std"]
        feat = info["feat"]
        close = info["close"]
        gpt_logits = info["logits"]  # [1, T_total, vocab_full]
        arrays = info["_arrays"]  # for BERT per-position time/VA

        # For each test position
        for p in range(test_start, test_end + 1):
            # GPT distribution at this position (predicting tok_p)
            gpt_lp = gpt_logits[0, p, :VOCAB_BASE].float()  # [vocab_base]
            gpt_probs = F.softmax(gpt_lp, dim=-1).numpy()  # [vocab_base]
            # Top-K candidates
            top_k_indices = np.argpartition(-gpt_probs, args.K)[:args.K]
            # Sort by descending probability
            top_k_indices = top_k_indices[np.argsort(-gpt_probs[top_k_indices])]
            candidates = top_k_indices.tolist()
            candidate_probs = gpt_probs[candidates]

            # Decide mask positions
            if args.mask_strategy == "boundary":
                mask_positions = [p]  # last history position
            elif args.mask_strategy == "middle":
                # Middle of history (not including BOS)
                mid = max(1, p // 2)
                mask_positions = [mid]
            elif args.mask_strategy == "random_k":
                rng_local = np.random.RandomState(p)  # deterministic per position
                candidates_for_mask = list(range(1, p + 1))
                if len(candidates_for_mask) > args.n_masks:
                    mask_positions = sorted(rng_local.choice(candidates_for_mask, args.n_masks, replace=False).tolist())
                else:
                    mask_positions = candidates_for_mask
            elif args.mask_strategy == "all_history":
                # Sample n_masks random positions in history (excluding BOS)
                rng_local = np.random.RandomState(p)
                candidates_for_mask = list(range(1, p + 1))
                if len(candidates_for_mask) > args.n_masks:
                    mask_positions = sorted(rng_local.choice(candidates_for_mask, args.n_masks, replace=False).tolist())
                else:
                    mask_positions = candidates_for_mask

            # Get BERT validation scores for each candidate
            if len(mask_positions) == 1:
                bert_scores = bert_validate_candidates(
                    bert, token_ids, candidates, mask_positions[0],
                    day, month, year, p, device, arrays=arrays)
            else:
                bert_scores = bert_validate_candidates_multi_mask(
                    bert, token_ids, candidates, mask_positions,
                    day, month, year, p, device, arrays=arrays)

            # Combine scores
            if args.combine == "product":
                final_scores = candidate_probs * bert_scores
            elif args.combine == "bert_only":
                final_scores = bert_scores
            elif args.combine == "gpt_only":
                final_scores = candidate_probs

            best_idx = int(np.argmax(final_scores))
            chosen_token = candidates[best_idx]
            # Also track whether GPT's argmax matches our choice (for diagnostics)
            gpt_argmax = int(np.argmax(gpt_probs))
            all_pred_chosen.append(1 if chosen_token != gpt_argmax else 0)

            # Decode predicted token → log return
            pred_feat = decode_predicted_token(chosen_token, tokenizer, device)
            pred_lr = pred_feat[0] * p_std[0] + p_mean[0]

            all_pred_toks.append(chosen_token)
            all_pred_lr.append(pred_lr)
            all_true_lr.append(feat[p + 1, 0])  # the actual log return at position p+1
            all_base_close.append(close[p])
            all_true_close.append(close[p + 1])

        n_done += 1
        if (si + 1) % 5 == 0 or si == len(test_stocks) - 1:
            elapsed = time.time() - t0
            n_total = len(all_pred_lr)
            if n_total > 0:
                pred_lr_arr = np.array(all_pred_lr)
                true_lr_arr = np.array(all_true_lr)
                running_da = (np.sign(pred_lr_arr) == np.sign(true_lr_arr)).mean() * 100
            print(f"  [{si+1}/{len(test_stocks)}] elapsed={elapsed:.0f}s n_pred={n_total} "
                  f"running DA={running_da:.2f}% "
                  f"frac_resampled={np.mean(all_pred_chosen)*100:.1f}%", flush=True)

    # Aggregate
    pred_lr = np.array(all_pred_lr)
    true_lr = np.array(all_true_lr)
    pred_toks = np.array(all_pred_toks, dtype=np.int64)
    base_close_arr = np.array(all_base_close)
    true_close_arr = np.array(all_true_close)

    if len(pred_lr) == 0:
        print("No predictions!")
        return

    # DA
    avg_da = (np.sign(pred_lr) == np.sign(true_lr)).mean()

    # MAPE (price space)
    pred_close_arr = base_close_arr * np.exp(pred_lr.astype(np.float64))
    eps = 1e-8
    mape_pt = np.abs(pred_close_arr - true_close_arr) / (np.abs(true_close_arr) + eps) * 100
    avg_mape = mape_pt.mean()

    # AmpRatio
    pred_amp = np.mean(np.abs(pred_lr))
    true_amp = np.mean(np.abs(true_lr))
    avg_ampratio = pred_amp / max(true_amp, 1e-8)

    # Collapse
    unique, counts = np.unique(pred_toks, return_counts=True)
    collapse = counts.max() / len(pred_toks)
    n_unique = len(unique)
    top_tok = int(unique[np.argmax(counts)])

    # Baseline MAPE (zero prediction)
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
    print(f"  {'Resampled':<20} {np.mean(all_pred_chosen)*100:>9.1f}%")
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
            "resample_rate": float(np.mean(all_pred_chosen)),
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
