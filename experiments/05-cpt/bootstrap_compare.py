"""Paired bootstrap comparison between two 400-window evaluations.

Branch A acceptance (ToDo §5): balance improves >= +0.02 over the CPT baseline
with a CI that excludes zero, AND H-CE drops <= 0.05 bit. JSD must not breach
the red line. This script computes per-date deltas between a candidate
evaluation and the reference baseline, then block-bootstraps over dates.

Usage:
    python experiments/05-cpt/bootstrap_compare.py \
        --candidate <cand_epoch_XXX.json> \
        --reference <ref_epoch_100.json> \
        --key median_daily_codebook_balance_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_per_date(path: Path, key: str) -> dict[str, float]:
    """Extract per-date values for a balance/distribution key."""
    payload = json.load(open(path))
    out = {}
    for offset, window in payload.get("windows", {}).items():
        for date, metrics in window.get("per_date", {}).items():
            if key in metrics and metrics[key] is not None:
                out[date] = float(metrics[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--key", type=str,
                        default="codebook_balance_score")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cand = load_per_date(args.candidate, args.key)
    ref = load_per_date(args.reference, args.key)
    common = sorted(set(cand) & set(ref))
    if not common:
        raise RuntimeError("No overlapping dates between candidate and reference")

    c = np.asarray([cand[d] for d in common])
    r = np.asarray([ref[d] for d in common])
    delta = c - r
    mean_delta = float(delta.mean())

    rng = np.random.RandomState(args.seed)
    n = len(common)
    means = []
    for _ in range(args.bootstrap):
        idx = rng.randint(0, n, n)
        means.append(delta[idx].mean())
    means = np.asarray(means)
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    significant = not (lo <= 0 <= hi)

    print(f"Key: {args.key}")
    print(f"Overlapping dates: {n}")
    print(f"Candidate mean: {c.mean():.4f}, Reference mean: {r.mean():.4f}")
    print(f"Delta: {mean_delta:+.4f}")
    print(f"95% CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"Significant (CI excludes 0): {significant}")
    print(f"Accept (delta >= +0.02): {mean_delta >= 0.02}")

    # Also report per-window aggregate from the JSON for the headline numbers
    def headline(path: Path):
        p = json.load(open(path))
        return p.get("aggregate", {}).get(args.key)
    print()
    print(f"Candidate aggregate {args.key}: {headline(args.candidate)}")
    print(f"Reference aggregate {args.key}: {headline(args.reference)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
