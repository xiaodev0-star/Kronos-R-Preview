"""Canonical paired circular moving-block bootstrap helpers.

All experiment scripts should import bootstrap logic from this root module.
Scripts under ``experiments/`` must not duplicate their own bootstrap code and
must not import bootstrap code from another experiment package.

The reference implementation follows Politis & Romano: block starts are drawn
iid over ``0..n-1`` with wraparound, blocks are concatenated, and the tail is
truncated to ``n`` observations.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def circular_moving_block_bootstrap(
    deltas: Iterable[float],
    *,
    block_length: int = 5,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Return replicate means of a daily delta series under circular MBB.

    ``deltas`` must already be ordered by trading date.  The pairing unit is the
    date: callers must first compute per-date candidate-minus-reference deltas on
    the common dense-date set, then pass that one-dimensional series here.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1:
        raise ValueError(f"deltas must be 1-D, got shape {deltas.shape}")
    n = int(deltas.shape[0])
    if n == 0:
        raise ValueError("empty delta series")
    n_replicates = max(1, int(n_replicates))
    if n == 1:
        return np.full(n_replicates, float(deltas[0]), dtype=float)

    block = max(1, int(block_length))
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.randint(0, n, size=(n_replicates, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    flat_indices = indices.reshape(n_replicates, -1)[:, :n]
    return deltas[flat_indices].mean(axis=1)


def paired_moving_block_bootstrap(
    candidate: Iterable[float],
    reference: Iterable[float],
    *,
    direction: str = "higher",
    block_lengths: tuple[int, ...] = (5, 10, 20),
    n_replicates: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired circular moving-block bootstrap for two aligned daily series.

    ``direction`` is ``"higher"`` when larger candidate values are better
    (RankIC, DA) and ``"lower"`` when smaller values are better (MAPE, MAE).

    Returns:
        point_estimate: mean of per-date candidate-minus-reference deltas.
        block_cis: per block length, 95% percentile CI and directional verdicts.
        favorable_fraction: fraction of bootstrap replicate means that favor the
            candidate (not a p-value; report it as a probability, not ``p``).
        block_robust: True when every requested block length gives a directional
            CI that excludes zero.
    """
    if direction not in ("higher", "lower"):
        raise ValueError(f"direction must be 'higher' or 'lower', got {direction!r}")
    if not block_lengths:
        raise ValueError("block_lengths must not be empty")

    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.ndim != 1 or ref.ndim != 1:
        raise ValueError("candidate and reference must be 1-D daily series")
    if cand.shape != ref.shape:
        raise ValueError(
            f"candidate and reference shapes differ: {cand.shape} vs {ref.shape}"
        )

    deltas = cand - ref
    point = float(deltas.mean())
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    block_cis: dict[str, dict] = {}
    for block_length in block_lengths:
        means = circular_moving_block_bootstrap(
            deltas,
            block_length=block_length,
            n_replicates=n_replicates,
            seed=seed,
        )
        lo = float(np.percentile(means, lower_pct))
        hi = float(np.percentile(means, upper_pct))
        if direction == "higher":
            favorable_fraction = float(np.mean(means > 0.0))
            directional_significant = bool(lo > 0.0)
        else:
            favorable_fraction = float(np.mean(means < 0.0))
            directional_significant = bool(hi < 0.0)
        block_cis[str(int(block_length))] = {
            "block_length": int(block_length),
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_excludes_zero": not (lo <= 0.0 <= hi),
            "significant_directional": directional_significant,
            "favorable_fraction": favorable_fraction,
        }

    favorable_fractions = [
        float(item["favorable_fraction"]) for item in block_cis.values()
    ]
    return {
        "direction": direction,
        "n_observations": int(deltas.shape[0]),
        "point_estimate": point,
        "candidate_mean": float(cand.mean()),
        "reference_mean": float(ref.mean()),
        "block_cis": block_cis,
        "block_robust": all(
            bool(item["significant_directional"]) for item in block_cis.values()
        ),
        "favorable_fraction": (
            float(np.mean(favorable_fractions)) if favorable_fractions else None
        ),
    }
