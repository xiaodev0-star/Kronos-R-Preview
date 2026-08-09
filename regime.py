"""Regime definition for Branch F (LoRA-per-regime) — single source of truth.

Training and the 400-window evaluation MUST compute identical per-day regime
labels, so all regime logic lives here (pure numpy, no model / data_processor
dependencies).

Regime metric
-------------
Realized volatility = std of raw daily log_ret over a ``window``-day trailing
window ending at day t (INCLUSIVE of t)::

    rv[t] = std(log_ret[t-window+1 : t+1])        # population std (ddof=0)

Strict point-in-time: a prediction of day ``p`` may only use information through
day ``p-1``, so the caller aligns ``regime_ids[p] = label(rv[p-1])``.  That gives
the design's ``[p-20, p-1]`` window — the 20 days ending at day p-1.

    NOTE: this differs from ``data_processor._trailing_vol20`` (used by Branch C
    weights), which uses the half-open window ``[i-window, i)`` and EXCLUDES day
    i.  Branch F must use the functions in THIS file only and never mix in
    ``_trailing_vol20``; the two branches intentionally keep separate windows.

Labels
------
Cross-sectional terciles over the TRAIN split only (computed once, cached).
  0 = low vol, 1 = med vol, 2 = high vol, -1 = insufficient window (NaN).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trailing_realized_vol(logret, window=20):
    """Realized volatility of raw daily log-returns over a trailing window.

    ``out[t] = std(logret[max(0, t-window+1) : t+1])`` — the window INCLUDES day
    ``t`` (population std, ddof=0, matching ``np.std``'s default).  Rows with
    fewer than ``window`` days of history (``t < window-1``) are NaN, which maps
    to regime id ``-1`` downstream.

    Returns a ``[T] float32`` array.
    """
    logret = np.asarray(logret, dtype=np.float64)
    T = logret.shape[0]
    out = np.full(T, np.nan, dtype=np.float32)
    if T == 0 or window <= 0:
        return out
    if T >= window:
        win = sliding_window_view(logret, window)     # [T-window+1, window]
        out[window - 1:] = win.std(axis=1).astype(np.float32)
    return out


def compute_regime_thresholds(train_logrets, window=20, quantiles=(1 / 3, 2 / 3)):
    """Cross-sectional realized-vol tercile cutoffs over the TRAIN split.

    Args:
        train_logrets: iterable of per-stock raw daily log-ret arrays (float),
            each already trimmed to TRAIN (pre-cutoff) rows only.
        window: trailing window length passed to ``trailing_realized_vol``.
        quantiles: the two cutoffs (default 1/3, 2/3 terciles).

    Returns:
        (q_lo, q_hi) float cutoffs: ~1/3 of train token-days land in each of
        the low / med / high regime buckets.
    """
    vals = []
    for lr in train_logrets:
        rv = trailing_realized_vol(lr, window)
        vals.append(rv[np.isfinite(rv)])
    if not vals:
        raise ValueError("no finite trailing-vol values in the train split")
    v = np.concatenate(vals)
    return tuple(float(np.quantile(v, q)) for q in quantiles)


def label_regime(rvol, thresholds):
    """Bucket per-day realized-vol values into regime ids.

    Args:
        rvol: array of realized-vol values (e.g. output of
            ``trailing_realized_vol``).
        thresholds: (lo, hi) tercile cutoffs.

    Returns:
        int64 array matching ``rvol``: 0 = low, 1 = med, 2 = high; any
        non-finite value (insufficient trailing window, NaN/Inf) -> -1.
    """
    rvol = np.asarray(rvol, dtype=np.float32)
    out = np.full(rvol.shape, -1, dtype=np.int64)
    lo, hi = thresholds
    finite = np.isfinite(rvol)
    out[finite & (rvol < lo)] = 0
    out[finite & (rvol >= lo) & (rvol < hi)] = 1
    out[finite & (rvol >= hi)] = 2
    return out


if __name__ == "__main__":
    # Synthetic sanity check for the point-in-time / off-by-one contract that is
    # the top correctness risk in the design (risk note: regime_ids[0]==-1,
    # regime_ids[1:20]==-1, regime_ids[20]==label(rv[19])).
    rng = np.random.default_rng(0)
    lr = rng.normal(0.01, 0.03, size=60)
    rv = trailing_realized_vol(lr, window=20)
    assert rv.shape == (60,)
    assert np.isnan(rv[:19]).all(), "first 19 days must be NaN (insufficient window)"
    assert np.isfinite(rv[19:]).all()
    # rv[19] = std(lr[0:20]) — inclusive window [0, 19]
    assert abs(float(rv[19]) - float(np.std(lr[0:20]))) < 1e-6
    # rv[40] = std(lr[21:41]) — window [21, 40]
    assert abs(float(rv[40]) - float(np.std(lr[21:41]))) < 1e-6
    # label boundaries
    th = (float(np.quantile(rv[19:], 1 / 3)), float(np.quantile(rv[19:], 2 / 3)))
    lab = label_regime(rv, th)
    assert (lab[:19] == -1).all()
    assert set(lab[19:].tolist()) <= {0, 1, 2}
    assert ((rv[19:] < th[0]) == (lab[19:] == 0)).all()
    assert ((rv[19:] >= th[1]) == (lab[19:] == 2)).all()
    assert (((rv[19:] >= th[0]) & (rv[19:] < th[1])) == (lab[19:] == 1)).all()
    # NaN/Inf -> -1
    lab2 = label_regime(np.array([np.nan, np.inf, -np.inf, 0.0]), th)
    assert (lab2[:3] == -1).all() and lab2[3] == 0
    # compute_regime_thresholds agrees with direct quantiles on the finite rows
    th2 = compute_regime_thresholds([lr], window=20)
    assert abs(th2[0] - float(np.quantile(rv[19:], 1 / 3))) < 1e-6
    assert abs(th2[1] - float(np.quantile(rv[19:], 2 / 3))) < 1e-6
    print("regime.py self-check OK")
