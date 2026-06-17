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
    feat = stock["features_raw"]
    day, month, year = stock["day"], stock["month"], stock["year"]
    close = stock["close_prices"]
    ci = _cutoff_idx(stock)
    T_total = len(feat)
    m = NormConfig.min_lookback
    if T_total < m + 10 or ci < m:
        return None

    price_feat = feat[:, :4]
    price_normed, _ = document_normalize(feat, cutoff_idx=ci)
    idx_c, _ = tokenizer.encode(torch.from_numpy(price_normed).float().unsqueeze(0).to(device))
    token_ids = idx_c[0].cpu().numpy()

    p_mean = price_feat[:ci].mean(axis=0)
    p_std = np.maximum(price_feat[:ci].std(axis=0), 1e-8)

    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab
    N = T_total
    ids = [bos_id] + token_ids[:N].tolist()
    d_l = [day[0]] + day[:N].tolist()
    m_l = [month[0]] + month[:N].tolist()
    y_l = [year[0]] + year[:N].tolist()
    S = len(ids)
    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tids = torch.stack([
        torch.tensor([d_l[:-1]], dtype=torch.long),
        torch.tensor([m_l[:-1]], dtype=torch.long),
        torch.tensor([y_l[:-1]], dtype=torch.long),
    ], dim=-1).to(device)
    pos = torch.arange(S - 1, device=device).unsqueeze(0)
    mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))
    va = torch.zeros(1, S - 1, 2, device=device)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        lc, lf, _ = gpt(inp, tids, pos, mask, va_values=va)
    return {
        "logits": lc.float().cpu(),  # [1, T, vocab_full]
        "token_ids": token_ids,      # [T_total] - no BOS
        "test_start": ci,
        "test_end": T_total - 2,
        "p_mean": p_mean,
        "p_std": p_std,
        "day": day, "month": month, "year": year,
        "feat": feat,
        "close": close,
        "T_total": T_total,
    }


@torch.no_grad()
def bert_validate_candidates(bert, token_ids, candidates, mask_pos, day, month, year,
                              p_value, device, batch_size=64):
    """Run BERT validation for one test position p against K candidates.

    Args:
        bert: BERT model
        token_ids: [T] the full sequence (no BOS), only used for the prefix
        candidates: list of K candidate token ids (y's to be appended as future)
        mask_pos: int, the position in [1, p] to mask (1-indexed in the BERT input,
                  where position 0 is BOS and position p+1 is the candidate)
        day/month/year: arrays for time embedding at the predicted position
        p_value: int, the test position p (predicting tok_p). The BERT input will be:
                 [BOS, tok_0, ..., MASK_at_mask_pos, ..., tok_{p-1}, candidate]
                 where MASK is at position mask_pos in [0, p+1]
        device: torch device
        batch_size: BERT batch size

    Returns:
        scores: [K] array of P_BERT(tok_at_mask_pos | context_with_candidate)
    """
    K = len(candidates)
    # Build prefix: [BOS, tok_0, ..., tok_{p-1}] (length p+1)
    prefix = [BOS_ID] + token_ids[:p_value].tolist()  # length p+1
    # prefix has positions 0..p, with prefix[0]=BOS, prefix[i]=tok_{i-1} for i>=1

    # Build K inputs: each is prefix with mask_pos replaced by MASK and candidate appended
    inputs = []
    mask_positions = []
    for cand in candidates:
        inp = list(prefix)  # copy
        inp[mask_pos] = MASK_ID  # mask the boundary (or chosen) history position
        inp.append(int(cand))   # append candidate as future
        inputs.append(inp)
        mask_positions.append(mask_pos)

    # We can batch all K inputs together if they all have the same length (which they do).
    max_len = len(inputs[0])
    B = len(inputs)
    inp_t = torch.tensor(inputs, dtype=torch.long, device=device)
    tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
    pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
    va = torch.zeros(B, max_len, 2, device=device)

    # Time embeddings: use the time at the predicted position (test position p)
    tids[:, :, 0] = int(day[p_value])
    tids[:, :, 1] = int(month[p_value])
    tids[:, :, 2] = int(year[p_value])
    pos[:] = torch.arange(max_len, device=device).unsqueeze(0)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        logits = bert(inp_t, tids, pos, va_values=va)  # [B, max_len, vocab_base]

    # Extract probability at the original (masked) token position for each candidate
    scores = np.zeros(K, dtype=np.float32)
    for bi, mp in enumerate(mask_positions):
        probs = F.softmax(logits[bi, mp].float(), dim=-1).cpu().numpy()  # [vocab_base]
        original_token = token_ids[p_value - (p_value + 1 - mask_pos)]  # = token_ids[mask_pos - 1]
        # Wait let me recompute: prefix[mask_pos] = tok_{mask_pos - 1}. So original token is token_ids[mask_pos - 1].
        original_token = int(token_ids[mask_pos - 1])
        scores[bi] = probs[original_token]

    return scores


@torch.no_grad()
def bert_validate_candidates_multi_mask(bert, token_ids, candidates, mask_positions,
                                          day, month, year, p_value, device):
    """Variant: mask MULTIPLE positions, return mean score across masked positions.

    Args:
        mask_positions: list of positions (in [1, p]) to mask
    """
    K = len(candidates)
    prefix = [BOS_ID] + token_ids[:p_value].tolist()
    M = len(mask_positions)

    inputs = []
    for cand in candidates:
        inp = list(prefix)
        for mp in mask_positions:
            inp[mp] = MASK_ID
        inp.append(int(cand))
        inputs.append(inp)

    max_len = len(inputs[0])
    B = len(inputs)
    inp_t = torch.tensor(inputs, dtype=torch.long, device=device)
    tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
    pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
    va = torch.zeros(B, max_len, 2, device=device)
    tids[:, :, 0] = int(day[p_value])
    tids[:, :, 1] = int(month[p_value])
    tids[:, :, 2] = int(year[p_value])
    pos[:] = torch.arange(max_len, device=device).unsqueeze(0)

    with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
        logits = bert(inp_t, tids, pos, va_values=va)

    # For each candidate, average P_BERT at each masked position
    scores = np.zeros(K, dtype=np.float32)
    for bi in range(B):
        total = 0.0
        for mp in mask_positions:
            probs = F.softmax(logits[bi, mp].float(), dim=-1).cpu().numpy()
            original_token = int(token_ids[mp - 1])
            total += probs[original_token]
        scores[bi] = total / M

    return scores


def decode_predicted_token(token_id, tokenizer, device):
    """Decode a single predicted token to a feature vector."""
    pred_indices = torch.tensor([token_id], dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).contiguous()
    with torch.no_grad():
        pred_feat = tokenizer.decode_all(pred_indices)[0].cpu().numpy()
    return pred_feat[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_ckpt", type=str, default="checkpoints/expA_v2.pt")
    parser.add_argument("--bert_ckpt", type=str, default="checkpoints/kronos_bert_v1.pt")
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
                    day, month, year, p, device)
            else:
                bert_scores = bert_validate_candidates_multi_mask(
                    bert, token_ids, candidates, mask_positions,
                    day, month, year, p, device)

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
