"""Branch E (DPO PostTrain): offline preference-pair construction.

Uses the FROZEN reference model ``pi_ref`` (Branch A output, e.g.
``checkpoints/branchA_dm030_8ceb_ep5.pt``) to sample K=8 coarse next-day tokens
at every train-region position ``p`` strictly before the cutoff
(``p in [min_lookback, min(ci, T_total-1))``, ``ci = cutoff index``), decodes
each to a price-space log-return, and scores each candidate by

    reward = alpha * 1[sign(pred_lr) == sign(true_lr)]
           + (1 - alpha) * (1 - |pct_pred - pct_true|)

where ``pct_true`` is the position's cross-sectional realized-return percentile
within its trading date and ``pct_pred`` is the sampled log-return's percentile
within the date's pooled candidate pool (all stocks x K).  The argmax/argmin
candidates become chosen/rejected, kept only when the reward gap clears
``min_reward_gap`` and the two sampled tokens differ.

Crucially, ``pi_ref``'s log-probability (``log_softmax(coarse_logits)[token]``)
of BOTH candidates is precomputed here and stored in the pairs file, so DPO
training needs only ONE forward pass (the policy's) per optimizer step.

Leakage guard (ToDo §9): preference labels come from ``features_raw[p, 0]``
(the raw next-day log-return, matching the 400-window protocol's
``true_logret`` convention) and all pair positions are strictly pre-cutoff.
``reg_targets`` in data_processor.py is |normalized log_ret| (volatility) and
must NEVER be used as the preference label.

Usage:
    python experiments/05-cpt/e_build_pairs.py \\
        --ref-ckpt checkpoints/branchA_dm030_8ceb_ep5.pt \\
        --tokenizer checkpoints/tokenizer_v2_ohlc.pt \\
        --k 8 --temperature 1.0 --alpha 0.5 --min_reward_gap 0.15 \\
        --seed 43 --out server_runs/results/04b-cpt/seed42/trials/branchE_b0.1/pairs_s43.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "experiments" / "04" / "b-hpo"))

from config import DataConfig, ModelConfig, NormConfig, set_global_seed  # noqa: E402
from data_processor import load_stocks, split_stocks  # noqa: E402
from eval_helpers import (  # noqa: E402
    AMP_DTYPE,
    attach_close_prices,
    build_gpt_eval_inputs,
    build_stock_arrays,
    decode_code_ids_tensor,
    load_gpt,
    load_tokenizer,
)
from evaluate_epoch_trajectory import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-ckpt", type=Path, required=True,
                        help="Frozen pi_ref (Branch A output).")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--k", type=int, default=8,
                        help="Coarse candidates sampled per position.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature used for sampling AND for the "
                             "precomputed pi_ref log-probs (must match training).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on the sign-hit reward term; the percentile "
                             "term gets (1 - alpha).")
    parser.add_argument("--min-reward-gap", dest="min_reward_gap", type=float,
                        default=0.15,
                        help="Minimum chosen-minus-rejected reward gap to keep a pair.")
    parser.add_argument("--min-lookback", type=int, default=20,
                        help="First pair position (>= min_lookback days of context).")
    parser.add_argument("--entropy-threshold", type=float, default=0.9,
                        help="Skip positions whose top-1 sampled probability "
                             "exceeds this (too low-entropy to yield useful pairs).")
    parser.add_argument("--min-date-candidates", type=int, default=5,
                        help="Drop dates with fewer than this many candidate "
                             "positions (cross-sectional percentile unstable).")
    parser.add_argument("--seed", type=int, default=43,
                        help="Sampling seed for K draws (pair_seed in {43,44,45}).")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Debug: limit the number of train stocks processed (0 = all).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output pairs npz path.")
    return parser.parse_args()


def _date_key(dates_dt, p: int) -> str:
    d = dates_dt[p] if p < len(dates_dt) else None
    if d is None:
        return "unknown"
    s = str(d)
    return s[:10]


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed, deterministic=False)
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = load_tokenizer(str(args.tokenizer), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    vocab = int(tokenizer.vocab_coarse)

    ref = load_gpt(str(args.ref_ckpt), device, tokenizer=tokenizer)
    ref.eval()
    print(f"pi_ref: {args.ref_ckpt} (coarse vocab={vocab})", flush=True)

    stocks = load_stocks(max_stocks=args.max_stocks)
    train_s, _val_s, _test_s = split_stocks(stocks)
    if args.max_stocks > 0:
        train_s = train_s[: args.max_stocks]
    attach_close_prices(train_s)
    print(f"Train stocks: {len(train_s)}", flush=True)

    # ---------------- Pass 1: sample K coarse candidates per position ----------
    # date -> list of candidate records.  Each record holds the K sampled coarse
    # ids, their decoded log-returns, the pi_ref log-probs, and the realized
    # next-day log-return at position p.
    true_lrs_by_date: dict[str, list[float]] = {}
    pred_lrs_by_date: dict[str, list[np.ndarray]] = {}
    meta_by_date: dict[str, list[tuple]] = {}
    n_positions = 0
    n_skipped_entropy = 0
    n_skipped_support = 0
    n_stocks_used = 0

    for si, stock in enumerate(train_s):
        arrays = build_stock_arrays(stock)
        if arrays is None:
            continue
        ci = int(arrays["ci"])
        t_total = int(arrays["T_total"])
        # pack_stocks_v2 (train mode) skips stocks with ci < min_doc_length;
        # drop them here too so every pair maps to a real training sequence.
        if ci < NormConfig.min_doc_length:
            continue
        inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
        positions = list(range(args.min_lookback, min(ci, t_total - 1)))
        if not positions:
            continue

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE,
                                    enabled=device.type == "cuda"):
                coarse, fine = ref.forward_selected(
                    inputs["inp"], inputs["tids"], inputs["pos"],
                    [0] * len(positions), positions,
                    va_values=inputs["va_values"],
                )
        lps = torch.log_softmax(
            coarse[:, :vocab].float() / args.temperature, dim=-1
        ).cpu()
        probs = lps.exp()
        top1 = probs.max(dim=-1).values

        feat = arrays["feat"]
        p_mean0 = float(arrays["p_mean"][0])
        p_std0 = float(arrays["p_std"][0])
        token_ids = inputs["token_ids"]
        dates_dt = arrays.get("dates_dt", None)

        for i, p in enumerate(positions):
            n_positions += 1
            if top1[i].item() > args.entropy_threshold:
                n_skipped_entropy += 1
                continue
            pi = probs[i]
            n_nonzero = int((pi > 0).sum())
            if n_nonzero < 2:
                n_skipped_support += 1
                continue
            k_eff = min(args.k, n_nonzero)
            sampled = torch.multinomial(
                pi, k_eff, replacement=(n_nonzero < args.k)
            ).tolist()
            fine_id = int(fine[i].argmax().item())
            coarse_t = torch.as_tensor(sampled, dtype=torch.long, device=device)
            fine_t = torch.full_like(coarse_t, fine_id)
            decoded = decode_code_ids_tensor(
                coarse_t, fine_t, tokenizer, device
            ).cpu().numpy()[:, 0]
            pred_lrs = decoded * p_std0 + p_mean0
            true_lr = float(feat[p, 0])
            true_id = int(token_ids[p]) if p < len(token_ids) else -1
            sampled_lps = lps[i].numpy()[sampled].astype(np.float32)
            dkey = _date_key(dates_dt, p) if dates_dt is not None else "unknown"

            if dkey not in true_lrs_by_date:
                true_lrs_by_date[dkey] = []
                pred_lrs_by_date[dkey] = []
                meta_by_date[dkey] = []
            true_lrs_by_date[dkey].append(true_lr)
            pred_lrs_by_date[dkey].append(np.asarray(pred_lrs, dtype=np.float32))
            meta_by_date[dkey].append(
                (si, str(stock["symbol"]), int(p), dkey,
                 np.asarray(sampled, dtype=np.int32),
                 sampled_lps, int(true_id), float(true_lr))
            )
        n_stocks_used += 1
        if (si + 1) % 200 == 0:
            print(f"  pass1 stocks {si+1}/{len(train_s)}, positions={n_positions}, "
                  f"entropy-skip={n_skipped_entropy}", flush=True)

    print(f"Pass1 done: {n_stocks_used} stocks, {n_positions} positions, "
          f"{len(true_lrs_by_date)} dates", flush=True)
    if not true_lrs_by_date:
        raise RuntimeError("No candidate positions survived; check min_lookback/"
                           "entropy_threshold or the reference checkpoint.")

    # ---------------- Pass 2/3: cross-sectional percentiles + pair selection ---
    from scipy.stats import rankdata

    pair_symbols: list[str] = []
    pair_dates: list[str] = []
    pair_positions: list[int] = []
    pair_chosen: list[int] = []
    pair_rejected: list[int] = []
    pair_refw: list[float] = []
    pair_refl: list[float] = []
    pair_true: list[int] = []
    pair_rew: list[float] = []
    pair_rel: list[float] = []
    pair_gap: list[float] = []

    dates = sorted(true_lrs_by_date)
    for dkey in dates:
        true_arr = np.asarray(true_lrs_by_date[dkey], dtype=np.float64)
        n = true_arr.shape[0]
        if n < args.min_date_candidates:
            continue
        pooled = np.concatenate(pred_lrs_by_date[dkey]).astype(np.float64)
        n_pool = pooled.shape[0]
        pct_true_all = rankdata(true_arr, method="average") / n
        pct_pred_all = rankdata(pooled, method="average") / n_pool

        cursor = 0
        true_cursor = 0
        for record in meta_by_date[dkey]:
            (si, symbol, p, dkey2, sampled_ids, sampled_lps,
             true_id, true_lr) = record
            k_eff = sampled_ids.shape[0]
            pred_lrs = pooled[cursor:cursor + k_eff]
            pct_pred = pct_pred_all[cursor:cursor + k_eff]
            cursor += k_eff
            pct_true = float(pct_true_all[true_cursor])
            true_cursor += 1

            sign_hit = np.sign(pred_lrs) == np.sign(true_lr)
            pct_term = 1.0 - np.abs(pct_pred - pct_true)
            rewards = args.alpha * sign_hit.astype(np.float64) + \
                (1.0 - args.alpha) * pct_term
            w = int(np.argmax(rewards))
            l = int(np.argmin(rewards))
            gap = float(rewards[w] - rewards[l])
            if gap < args.min_reward_gap:
                continue
            if sampled_ids[w] == sampled_ids[l]:
                continue

            pair_symbols.append(symbol)
            pair_dates.append(dkey2)
            pair_positions.append(p)
            pair_chosen.append(int(sampled_ids[w]))
            pair_rejected.append(int(sampled_ids[l]))
            pair_refw.append(float(sampled_lps[w]))
            pair_refl.append(float(sampled_lps[l]))
            pair_true.append(true_id)
            pair_rew.append(float(rewards[w]))
            pair_rel.append(float(rewards[l]))
            pair_gap.append(gap)

    n_pairs = len(pair_chosen)
    print(f"Pass2/3 done: {n_pairs} pairs across {len(pair_dates)} dates", flush=True)
    if n_pairs == 0:
        raise RuntimeError("No pairs passed min_reward_gap; relax the threshold.")

    # ---------------- Persist pairs npz + summary.json --------------------------
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_out = {
        "schema": np.asarray([1], dtype=np.int16),
        "temperature": np.asarray([args.temperature], dtype=np.float32),
        "symbols": np.asarray(pair_symbols, dtype=object),
        "dates": np.asarray(pair_dates, dtype=object),
        "position": np.asarray(pair_positions, dtype=np.int32),
        "chosen_id": np.asarray(pair_chosen, dtype=np.int32),
        "rejected_id": np.asarray(pair_rejected, dtype=np.int32),
        "ref_logp_chosen": np.asarray(pair_refw, dtype=np.float32),
        "ref_logp_rejected": np.asarray(pair_refl, dtype=np.float32),
        "true_id": np.asarray(pair_true, dtype=np.int32),
        "reward_chosen": np.asarray(pair_rew, dtype=np.float32),
        "reward_rejected": np.asarray(pair_rel, dtype=np.float32),
        "reward_gap": np.asarray(pair_gap, dtype=np.float32),
    }
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays_out)
    os.replace(temporary, out_path)
    pairs_sha256 = file_sha256(out_path)

    gaps = np.asarray(pair_gap, dtype=np.float64)
    rew_w = np.asarray(pair_rew, dtype=np.float64)
    n_dates_with_pairs = len(set(pair_dates))
    summary = {
        "schema": 1,
        "ref_ckpt": {
            "path": str(args.ref_ckpt.resolve()),
            "sha256": file_sha256(args.ref_ckpt),
        },
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "sampling": {
            "k": args.k,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "min_reward_gap": args.min_reward_gap,
            "min_lookback": args.min_lookback,
            "entropy_threshold": args.entropy_threshold,
            "min_date_candidates": args.min_date_candidates,
            "seed": args.seed,
        },
        "data": {
            "n_train_stocks": n_stocks_used,
            "n_positions_scanned": n_positions,
            "n_skipped_entropy": n_skipped_entropy,
            "n_skipped_support": n_skipped_support,
            "cutoff_date": DataConfig.cutoff_date,
        },
        "pairs": {
            "n_pairs": n_pairs,
            "n_dates": n_dates_with_pairs,
            "reward_gap": {
                "mean": float(gaps.mean()),
                "median": float(np.median(gaps)),
                "p10": float(np.percentile(gaps, 10)),
                "p90": float(np.percentile(gaps, 90)),
                "min": float(gaps.min()),
                "max": float(gaps.max()),
            },
            "reward_chosen_mean": float(rew_w.mean()),
            "token_match_rate": float(np.mean(np.asarray(pair_chosen) == np.asarray(pair_true))),
        },
        "pairs_path": str(out_path.resolve()),
        "pairs_sha256": pairs_sha256,
    }
    summary_path = out_path.with_suffix(".summary.json")
    temporary_summary = summary_path.with_suffix(".tmp")
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_summary, summary_path)

    print("=== e_build_pairs summary ===")
    print(f"  pairs={n_pairs} dates={n_dates_with_pairs} "
          f"scanned_positions={n_positions}")
    print(f"  reward_gap: mean={gaps.mean():.3f} median={np.median(gaps):.3f}")
    print(f"  saved to {out_path}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
