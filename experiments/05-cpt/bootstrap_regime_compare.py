"""Per-regime paired bootstrap between Branch F and the dense baseline.

Branch F (LoRA-per-regime) acceptance requires, per market regime, a paired
bootstrap of the candidate vs the dense baseline's SAME-regime slice over
overlapping dates (ToDo §10 + §11).  Both epoch JSONs carry a ``per_regime``
block produced by ``evaluate_epoch_trajectory.py --regime_window ...``; each
slice's ``per_date`` holds the per-date metric values needed here.

For each regime r in {0,1,2} (or a single --regime), this script
  - extracts the per-date chosen-metric value from candidate and reference,
  - keeps only dates with >= --min_cross_section stocks in BOTH,
  - block-bootstraps the per-date delta over dates (2000 resamples, seed 42),
  - prints mean delta, 95% CI, significance, and the sample size.

Usage:
    python experiments/05-cpt/bootstrap_regime_compare.py \
        --candidate <branchF_epN_XXX.json> \
        --reference <dense_regime_XXX.json> \
        --key median_daily_codebook_balance_score \
        --regime 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Aggregate metric key -> per-date field inside per_regime[r]['per_date'].
PER_DATE_KEY = {
    "median_daily_codebook_balance_score": "codebook_balance_score",
    "avg_da_per_date": "da",
    "avg_daily_rank_ic": "rank_ic",
    "mape": "mape",
    "ampratio": "ampratio",
}


def load_per_regime_per_date(path: Path, regime: str, key: str,
                             min_cross_section: int) -> dict[str, float]:
    """Extract per-date values of ``key`` from one regime slice.

    Returns {date: metric} restricted to dates whose regime cross-section
    ``per_date[date]['n'] >= min_cross_section``.
    """
    payload = json.load(open(path))
    slice_ = payload.get("per_regime", {}).get(regime)
    if slice_ is None:
        raise KeyError(f"{path.name} has no per_regime[{regime}] (was it run "
                       f"with --regime_window?)")
    per_date_key = PER_DATE_KEY.get(key)
    if per_date_key is None:
        raise ValueError(
            f"unsupported key {key!r}; choose from {sorted(PER_DATE_KEY)}")
    out: dict[str, float] = {}
    for date, metrics in slice_.get("per_date", {}).items():
        if int(metrics.get("n", 0)) < min_cross_section:
            continue
        value = metrics.get(per_date_key)
        if value is not None and np.isfinite(value):
            out[date] = float(value)
    return out


def report_regime(cand: dict[str, float], ref: dict[str, float], key: str,
                  regime: str, bootstrap: int, seed: int) -> None:
    """Paired bootstrap over dates present in both candidate and reference."""
    common = sorted(set(cand) & set(ref))
    if not common:
        raise RuntimeError(
            f"regime {regime}: no overlapping dates between candidate and "
            f"reference (try lowering --min_cross_section)")
    c = np.asarray([cand[d] for d in common])
    r = np.asarray([ref[d] for d in common])
    delta = c - r
    mean_delta = float(delta.mean())

    rng = np.random.RandomState(seed)
    n = len(common)
    means = np.empty(bootstrap, dtype=np.float64)
    for i in range(bootstrap):
        idx = rng.randint(0, n, n)
        means[i] = delta[idx].mean()
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    significant = not (lo <= 0 <= hi)

    print(f"Regime {regime} | key={key}")
    print(f"  Overlapping dates: {n}")
    print(f"  Candidate mean: {c.mean():+.4f}, Reference mean: {r.mean():+.4f}")
    print(f"  Mean delta: {mean_delta:+.4f}")
    print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Significant (CI excludes 0): {significant}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--key", type=str,
                        default="median_daily_codebook_balance_score",
                        choices=sorted(PER_DATE_KEY))
    parser.add_argument("--regime", type=str, default="all",
                        help="0, 1, 2, or 'all' (default).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_cross_section", type=int, default=20,
                        help="Require >= this many stocks on a date in the "
                             "regime slice (in BOTH candidate and reference).")
    args = parser.parse_args()

    regimes = ["0", "1", "2"] if args.regime == "all" else [args.regime]
    if args.regime not in ("all", "0", "1", "2"):
        raise ValueError("--regime must be 0, 1, 2, or 'all'")

    # Report the per-regime aggregate (headline numbers) from both files too.
    cand_agg = json.load(open(args.candidate)).get("per_regime", {})
    ref_agg = json.load(open(args.reference)).get("per_regime", {})

    for regime in regimes:
        print(f"\n=== Regime {regime} ===", flush=True)
        c = load_per_regime_per_date(args.candidate, regime, args.key,
                                     args.min_cross_section)
        r = load_per_regime_per_date(args.reference, regime, args.key,
                                     args.min_cross_section)
        report_regime(c, r, args.key, regime, args.bootstrap, args.seed)
        # Headline aggregate context (ampratio log error for the high-vol regime).
        c_slice = cand_agg.get(regime, {})
        r_slice = ref_agg.get(regime, {})
        if args.key == "ampratio":
            for label, slice_ in (("cand", c_slice), ("ref", r_slice)):
                amp = slice_.get("ampratio")
                if isinstance(amp, (int, float)) and amp > 0:
                    print(f"  {label} ampratio={amp:.3f} "
                          f"log_error={abs(float(np.log(amp))):.3f}")
        print(f"  cand n_dates={c_slice.get('n_dates', 0)} "
              f"n_pred={c_slice.get('n_predictions', 0)} | "
              f"ref n_dates={r_slice.get('n_dates', 0)} "
              f"n_pred={r_slice.get('n_predictions', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
