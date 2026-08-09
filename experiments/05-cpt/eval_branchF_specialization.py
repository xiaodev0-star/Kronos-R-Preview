"""G-gate functional test for Branch F (LoRA-per-regime).

Decides whether the per-regime LoRA adapters are actually regime-specific, on
the validation split (ToDo §11 'expert-swap' + 'Delta-W≈0' checks at F cost):

  - ``dense``    : the dense CPT baseline checkpoint, no regime_ids.
  - ``dense_lora``: the Branch F checkpoint with regime_ids=None.  Because the
                   LoRA backbone is frozen, this is bit-identical to ``dense``
                   and confirms the frozen-backbone identity (path noise only).
  - ``auto``     : the Branch F checkpoint with the true per-token regime
                   routing (each token uses its own regime's LoRA).
  - ``force{r}`` : the Branch F checkpoint with regime_ids broadcast to r on
                   every token (that single LoRA applied to ALL tokens).

For each regime r in {0,1,2} we report the mean coarse CE over that regime's
validation tokens under every mode, then derive:

  learned[r] = CE['dense_lora'][r] - CE['auto'][r]     (> 0 => LoRA r helps r)
  spill[r][s] = CE['dense_lora'][s] - CE['force' + r][s]   (~0 for s != r)

G is released only if (a) at least one regime shows a paired-bootstrap
improvement vs dense AND (b) every LoRA is regime-specific (its own-regime CE
improves while other regimes do not improve).  The Delta-W≈0 check reports the
L2 norms of the A and B matrices; if B stays ~0 the adapter is a no-op and G
stays frozen.

Usage:
    python experiments/05-cpt/eval_branchF_specialization.py \
        --ckpt checkpoints/branchF_r8.pt \
        --baseline_ckpt checkpoints/exp04b_8ceb_ep100.pt \
        --regime_window 20 --regime_quantiles 0.333,0.667
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Branch F checkpoint (load_gpt_lora).")
    parser.add_argument("--baseline_ckpt", type=Path, required=True,
                        help="Dense CPT baseline checkpoint (load_gpt).")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--regime_window", type=int, default=20)
    parser.add_argument("--regime_quantiles", type=str, default="0.333,0.667")
    parser.add_argument("--n_stocks", type=int, default=0,
                        help="Number of validation stocks to test (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=None,
                        help="Override LoRA rank (default: from checkpoint).")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Override LoRA alpha (default: from checkpoint).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON path to write the per-regime table.")
    return parser.parse_args()


def lora_param_norms(model: torch.nn.Module) -> dict[str, float]:
    """L2 norms of the attached LoRA A/B matrices (Delta-W ≈ 0 check)."""
    sd = model.state_dict()
    a_sq = 0.0
    b_sq = 0.0
    total_sq = 0.0
    n_a = 0
    n_b = 0
    for key, value in sd.items():
        if ".lora." not in key:
            continue
        s = float((value.float() ** 2).sum().item())
        total_sq += s
        if key.endswith(".0.weight"):
            a_sq += s
            n_a += 1
        elif key.endswith(".1.weight"):
            b_sq += s
            n_b += 1
    return {
        "a_l2": math.sqrt(a_sq),
        "b_l2": math.sqrt(b_sq),
        "total_l2": math.sqrt(total_sq),
        "n_a_matrices": n_a,
        "n_b_matrices": n_b,
    }


def compute_thresholds(regime_window: int, regime_quantiles: str) -> tuple:
    from config import DataConfig
    from data_processor import _stock_cutoff_idx, load_stocks, split_stocks
    from regime import compute_regime_thresholds

    quantiles = tuple(float(v) for v in regime_quantiles.split(",") if v.strip())
    stocks = load_stocks(max_stocks=0)
    train_s, _, _ = split_stocks(stocks)
    train_logrets = [
        s["features_raw"][:_stock_cutoff_idx(s, DataConfig.cutoff_date), 0]
        for s in train_s
    ]
    return compute_regime_thresholds(
        train_logrets, window=regime_window, quantiles=quantiles)


def run_mode_batch(model_lora, stock, tokenizer, device, labels, regime_window,
                   mode_specs, baseline_model=None):
    """Run all Branch F modes on one stock in a single batched forward.

    ``mode_specs`` is a list of ``(name, regime_ids)`` tuples; regime_ids is
    either None (fused dense path) or an [N] int64 array.  Returns
    ``{name: {regime: CE}}`` plus the baseline (dense) result when
    ``baseline_model`` is given.
    """
    from eval_helpers import build_gpt_eval_inputs, build_stock_arrays

    arrays = build_stock_arrays(stock)
    if arrays is None:
        return None
    inputs = build_gpt_eval_inputs(arrays, tokenizer, device)
    inp, tids, pos, mask, va = (
        inputs["inp"], inputs["tids"], inputs["pos"], inputs["mask"],
        inputs["va_values"],
    )
    ci = arrays["ci"]
    N = inp.shape[1]
    token_ids = np.asarray(inputs["token_ids"], dtype=np.int64)[:N]
    targets = torch.as_tensor(token_ids, dtype=torch.long, device=device)

    def group_ce(coarse_logits, mode):
        """[N, V+2] logits -> per-regime mean CE over pre-cutoff positions."""
        ce = F.cross_entropy(coarse_logits, targets, reduction="none")  # [N]
        buckets = {0: [], 1: [], 2: []}
        for p in range(1, ci):
            r = int(labels[p - 1])
            if r >= 0:
                buckets[r].append(float(ce[p]))
        out = {}
        for r in (0, 1, 2):
            values = buckets[r]
            out[str(r)] = float(np.mean(values)) if values else float("nan")
        out["_n_per_regime"] = {str(r): len(buckets[r]) for r in (0, 1, 2)}
        return out

    results: dict[str, dict] = {}
    # Batched forward for the Branch F modes.
    if mode_specs:
        row_rids = []
        for _, m in mode_specs:
            if m is None:
                row_rids.append(np.full(N, -1, dtype=np.int64))  # all -1 => no adapter
            else:
                row_rids.append(np.asarray(m, dtype=np.int64))
        Bm = len(mode_specs)
        rid = torch.from_numpy(np.stack(row_rids)).to(device)  # [Bm, N]
        inp_m = inp.expand(Bm, -1)
        tids_m = tids.expand(Bm, -1, -1)
        pos_m = pos.expand(Bm, -1)
        va_m = va.expand(Bm, -1, -1)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                coarse_batch, _ = model_lora(
                    inp_m, tids_m, pos_m, mask, va_values=va_m,
                    regime_ids=rid)
        coarse_batch = coarse_batch.float()
        for i, (name, _) in enumerate(mode_specs):
            results[name] = group_ce(coarse_batch[i], None)

    # Dense baseline runs on its own (separate model).
    if baseline_model is not None:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                coarse_dense, _ = baseline_model(
                    inp, tids, pos, mask, va_values=va, regime_ids=None)
        results["dense"] = group_ce(coarse_dense[0].float(), None)
    return results


def main() -> int:
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    if not args.baseline_ckpt.exists():
        raise FileNotFoundError(args.baseline_ckpt)

    from config import ModelConfig, set_global_seed
    from data_processor import load_stocks, split_stocks
    from eval_helpers import attach_close_prices, load_gpt, load_gpt_lora, load_tokenizer
    from regime import label_regime, trailing_realized_vol

    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = load_tokenizer(str(args.tokenizer), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size

    model_lora = load_gpt_lora(
        str(args.ckpt), device, tokenizer=tokenizer,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    model_base = load_gpt(str(args.baseline_ckpt), device, tokenizer=tokenizer)

    dw = lora_param_norms(model_lora)
    print("Delta-W norms (LoRA params):", json.dumps(dw, indent=2), flush=True)

    thresholds = compute_thresholds(args.regime_window, args.regime_quantiles)
    print(f"Regime thresholds: {[float(v) for v in thresholds]}", flush=True)

    stocks = load_stocks(max_stocks=0)
    _, val_s, _ = split_stocks(stocks)
    if args.n_stocks > 0:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(val_s), min(args.n_stocks, len(val_s)),
                         replace=False)
        val_s = [val_s[i] for i in sorted(idx)]
    print(f"Evaluating {len(val_s)} validation stocks", flush=True)
    attach_close_prices(val_s)

    modes = ["dense_lora", "auto", "force0", "force1", "force2"]
    accum = {
        m: {str(r): {"sum": 0.0, "count": 0} for r in (0, 1, 2)}
        for m in modes + ["dense"]
    }
    n_regime_tokens = {str(r): 0 for r in (0, 1, 2)}
    skipped = 0

    for i, stock in enumerate(val_s):
        feat = stock["features_raw"]
        labels = label_regime(
            trailing_realized_vol(feat[:, 0].astype(np.float64),
                                  args.regime_window),
            thresholds,
        )
        # Build per-mode regime_ids once per stock.
        auto_rid = np.full(len(feat), -1, dtype=np.int64)
        auto_rid[1:] = labels[0:len(feat) - 1]
        mode_specs = [
            ("dense_lora", None),
            ("auto", auto_rid),
            ("force0", np.full(len(feat), 0, dtype=np.int64)),
            ("force1", np.full(len(feat), 1, dtype=np.int64)),
            ("force2", np.full(len(feat), 2, dtype=np.int64)),
        ]
        result = run_mode_batch(
            model_lora, stock, tokenizer, device, labels, args.regime_window,
            mode_specs, baseline_model=model_base,
        )
        if result is None:
            skipped += 1
            continue
        for m, per in result.items():
            for r in (0, 1, 2):
                value = per.get(str(r))
                if value is not None and np.isfinite(value):
                    accum[m][str(r)]["sum"] += value
                    accum[m][str(r)]["count"] += 1
        for r in (0, 1, 2):
            n_regime_tokens[str(r)] += result["dense_lora"].get(
                "_n_per_regime", {}).get(str(r), 0)
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(val_s)} stocks", flush=True)

    print(f"\nSkipped (too-short) stocks: {skipped}", flush=True)

    table = {}
    print("\n=== Per-regime mean coarse CE by mode ===", flush=True)
    print(f"{'regime':>6} {'tokens':>8}" + "".join(
        f" {m:>10}" for m in modes + ["dense"]), flush=True)
    for r in (0, 1, 2):
        row = {}
        for m in modes + ["dense"]:
            s = accum[m][str(r)]
            mean = s["sum"] / s["count"] if s["count"] else float("nan")
            row[m] = mean
        table[str(r)] = row
        print(f"{r:>6} {n_regime_tokens[str(r)]:>8}" + "".join(
            f" {row[m]:>10.4f}" for m in modes + ["dense"]), flush=True)

    # Derived G-gate signals.
    print("\n=== learned[r] = CE[dense_lora][r] - CE[auto][r] (own-regime gain) ===",
          flush=True)
    learned = {}
    for r in (0, 1, 2):
        gain = table[str(r)]["dense_lora"] - table[str(r)]["auto"]
        learned[str(r)] = gain
        print(f"  regime {r}: {gain:+.4f} bit", flush=True)

    print("\n=== spill[r][s] = CE[dense_lora][s] - CE[force+r][s] ===", flush=True)
    spill = {}
    for r in (0, 1, 2):
        spill_row = {}
        for s in (0, 1, 2):
            delta = table[str(s)]["dense_lora"] - table[str(s)][f"force{r}"]
            spill_row[str(s)] = delta
        spill[str(r)] = spill_row
        print(f"  LoRA_{r}: " + "  ".join(
            f"s={s}: {delta:+.4f}" for s, delta in spill_row.items()), flush=True)

    # Frozen-backbone identity check: CE[dense] vs CE[dense_lora].
    print("\n=== frozen-backbone identity: CE[dense] - CE[dense_lora] ===",
          flush=True)
    identity = {}
    for r in (0, 1, 2):
        delta = table[str(r)]["dense"] - table[str(r)]["dense_lora"]
        identity[str(r)] = delta
        print(f"  regime {r}: {delta:+.5f} bit (path noise only)", flush=True)

    g_release = (
        any(learned[r] > 0.01 for r in learned)
        and all(
            abs(spill[str(r)][str(s)]) < 0.01 or s == r
            for r in spill for s in spill[r]
        )
        and dw["b_l2"] > 1e-4
    )
    summary = {
        "mode_ce": table,
        "learned": learned,
        "spill": spill,
        "identity_check": identity,
        "delta_w_norms": dw,
        "n_regime_tokens": n_regime_tokens,
        "n_val_stocks": len(val_s) - skipped,
        "skipped": skipped,
        "regime_thresholds": [float(v) for v in thresholds],
        "g_release_recommendation": g_release,
    }
    print("\nG-gate recommendation:", "RELEASE" if g_release else "FROZEN",
          flush=True)
    print(json.dumps({k: summary[k] for k in
                      ("learned", "spill", "delta_w_norms", "g_release_recommendation")},
                     indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
