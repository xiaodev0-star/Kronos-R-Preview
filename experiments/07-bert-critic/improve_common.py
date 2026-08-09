"""improve_common.py — shared helpers for the BERT-Critic Improvement Plan (R/T series).

Implements the plan §2 E-1 fix: row-level rank scores MUST be decoded-return
quantities (E[r] / median / P(up)) in raw space, never raw token likelihoods.
All R-series arms decode a coarse posterior over the full 128-code vocabulary
with ``checkpoints/coarse_logret_centers.npy`` (normalized-space per-token
log_ret) and restore raw space with per-row ``p_mean0`` / ``p_std0``
(``raw = norm * p_std0 + p_mean0``, joint_decoder.py:69).

Statistics inherit the 06 protocol: daily RankIC / DA / MAE, paired circular
moving-block bootstrap (L=5/10/20), dev 0..299 / confirm 300..399 splits.
0..399 is pure inference; every fusion/calibration parameter is fit on the
calibration slice (audit_uids x [2023-02-01, 2024-02-01)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scipy.stats import spearmanr  # noqa: E402

from critic_common import resolve_roots, write_json, append_trial  # noqa: E402
from compare_posttrain import circular_moving_block_bootstrap  # noqa: E402
from evaluate_posttrain import arm_metrics  # noqa: E402

EPS = 1e-12

# ---------------------------------------------------------------------------
# Paths (07 seed42 dual roots + inherited 06 assets)
# ---------------------------------------------------------------------------

CENTERS_PATH = ROOT / "checkpoints" / "coarse_logret_centers.npy"
PT01_REC = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "pt01_records.npz"
EVAL_HIDDEN = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "hidden_cache.npz"
P6_HEAD = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "head_P6_mlp_rank_spearman.pt"


def roots(seed=42):
    return resolve_roots(seed=seed)


def weights_root(seed=42):
    return roots(seed).weights_root


def results_root(seed=42):
    return roots(seed).results_root


def cand_path(region, seed=42):
    return weights_root(seed) / f"candidates_{region}_K8.npz"


def scores_path(region, seed=42, suffix=""):
    sfx = f"_{suffix}" if suffix else ""
    return weights_root(seed) / f"scores_{region}_K8_w512_stride1{sfx}.npz"


def gpt_q_path(region, seed=42):
    """Full-128 GPT q for a region (calib cached in candidates; eval recomputed)."""
    return weights_root(seed) / f"gpt_q_{region}_full128.npz"


# ---------------------------------------------------------------------------
# Coarse posterior -> raw-space return decoding  (plan §2 E-1 fix, §3 R1/R2/R3)
# ---------------------------------------------------------------------------

def load_centers():
    """[128] float32 normalized-space log_ret per coarse token (fine=0 decode)."""
    return np.load(CENTERS_PATH)


def softmax_rows(logp):
    """[N,128] log-space (log_softmax output) -> probability (float32).

    NaN-safe: rows with non-finite logp stay NaN.  float32 keeps the [N,128]
    posterior half the size of float64 (a 1.8M-row posterior is 0.86 GiB vs
    1.72 GiB) — ranking-scale precision is unaffected.
    """
    p = np.exp(np.asarray(logp, dtype=np.float32))
    fin = np.isfinite(p).all(axis=1)
    pn = np.where(fin[:, None], p, np.float32(0.0))
    pn = pn / np.maximum(pn.sum(axis=1, keepdims=True), np.float32(EPS))
    return np.where(fin[:, None], pn, np.nan)


def decode_coarse(p_or_logp, centers, p_mean0, p_std0, log_space=False, chunk=300_000):
    """Decode a [N,128] coarse posterior to raw-space return statistics.

    ``p_or_logp`` is either a per-row PROBABILITY posterior (``log_space=False``,
    rows sum to 1) or the log-space vector (``log_space=True``, exp within each
    chunk).  ``centers`` are normalized-space per-token log_ret; raw space is
    restored per row with ``raw = norm * p_std0 + p_mean0``.

    NaN-safe and chunked (bounded peak memory on the 1.8M-row eval region).

    Returns dict (all [N] float64):
      e_mean / e_norm    E[r] and its normalized version (J2's BERT version)
      e_median / med_norm  weighted median r (J3's BERT version)
      p_up_raw  P(raw r > 0) = mass over tokens with center > -p_mean0/p_std0
      p_up_naive  mass over center > 0 (plan §3 R1 literal "正 center")
    """
    centers = np.asarray(centers, dtype=np.float64)
    pstd = np.maximum(np.asarray(p_std0, dtype=np.float64), EPS)
    pm = np.asarray(p_mean0, dtype=np.float64)
    n = int(np.asarray(p_or_logp).shape[0])
    res = {k: np.full(n, np.nan, dtype=np.float64)
           for k in ("e_mean", "e_median", "p_up_raw", "p_up_naive",
                     "e_norm", "med_norm")}
    order = np.argsort(centers)
    cs = centers[order]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = np.asarray(p_or_logp[start:stop], dtype=np.float32)
        if log_space:
            x = np.exp(x)
        fin = np.isfinite(x).all(axis=1)
        pn = np.where(fin[:, None], x, np.float32(0.0))
        pn = pn / np.maximum(pn.sum(axis=1, keepdims=True), np.float32(EPS))
        m = stop - start
        pstd_c = pstd[start:stop]
        pm_c = pm[start:stop]

        e_norm = (pn @ centers).astype(np.float64)             # [m]
        ps = pn[:, order]
        cdf = np.cumsum(ps, axis=1)
        idx = np.sum(cdf < np.float32(0.5), axis=1).clip(max=len(centers) - 1)
        med_norm = cs[idx]
        thr = -pm_c / pstd_c
        pup_raw = np.sum(pn * (centers[None, :] > thr[:, None]), axis=1).astype(np.float64)
        pup_naive = np.sum(pn * (centers[None, :] > 0.0), axis=1).astype(np.float64)
        mask = np.where(fin, 1.0, np.nan)
        res["e_norm"][start:stop] = e_norm * mask
        res["e_mean"][start:stop] = (e_norm * pstd_c + pm_c) * mask
        res["med_norm"][start:stop] = med_norm * mask
        res["e_median"][start:stop] = (med_norm * pstd_c + pm_c) * mask
        res["p_up_raw"][start:stop] = pup_raw * mask
        res["p_up_naive"][start:stop] = pup_naive * mask
    return res


def poe_fused(logp_bert, logq_gpt, lam, chunk=300_000):
    """Full-128 log-linear pooling (product of experts), plan §3 R2 / §2 E-5.

    ``p_fused(c) ∝ q_GPT(c)^(1-lam) * p_BERT(c)^lam`` over the FULL 128-code
    support (not just GPT's top-8), then renormalized.  ``logp_bert`` is a
    [N,128] log-space vector; ``logq_gpt`` a [N,128] probability.

    Chunked (bounded peak memory on 1.8M rows).  Contract (plan §6 new test 2):
    every finite returned row sums to 1 within 1e-4.
    """
    n = int(np.asarray(logp_bert).shape[0])
    v = int(np.asarray(logp_bert).shape[1])
    pf_out = np.full((n, v), np.nan, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p_bert = softmax_rows(logp_bert[start:stop])            # float32
        fin = np.isfinite(p_bert).all(axis=1)
        p_bert = np.where(fin[:, None], p_bert, np.float32(0.0))
        q = np.asarray(logq_gpt[start:stop], dtype=np.float32)
        q = np.where(fin[:, None], q, np.float32(1.0))
        q = q / q.sum(axis=1, keepdims=True)
        q = np.maximum(q, np.float32(EPS))
        lf = ((1.0 - lam) * np.log(q)
              + lam * np.log(np.maximum(p_bert, np.float32(EPS))))
        lf -= lf.max(axis=1, keepdims=True)
        pf = np.exp(lf)
        pf = pf / pf.sum(axis=1, keepdims=True)
        pf_out[start:stop] = np.where(fin[:, None], pf, np.nan)
    good = np.isfinite(pf_out).all(axis=1)
    if good.any():
        assert np.allclose(pf_out[good].sum(axis=1), 1.0, atol=1e-4), "PoE row sums != 1"
    return pf_out


# ---------------------------------------------------------------------------
# Record construction & region slices
# ---------------------------------------------------------------------------

def build_rec(cand, pt01=None, extra=None):
    """Base per-row record from a candidates cache (+ optional pt01 for J2/J0).

    Fields shared by eval & calib: date_key, stock_uid, offset (eval), quality,
    p_mean0, p_std0, post_median (J3), p_up (J4), post_std, true_logret.
    ``pt01`` (eval only) adds post_mean (J2) and greedy_return (J0).
    """
    n = len(cand["stock_uid"])
    rec = {
        "date_key": cand["date_key"],
        "stock_uid": cand["stock_uid"],
        "true_logret": cand["true_logret"].astype(np.float64),
        "quality": cand["quality"].astype(bool),
        "p_mean0": cand["p_mean0"].astype(np.float64),
        "p_std0": cand["p_std0"].astype(np.float64),
        "post_median": cand["post_median"].astype(np.float64),
        "p_up": cand["p_up"].astype(np.float64),
        "post_std": cand["post_std"].astype(np.float64),
    }
    if "offset" in cand.files:
        rec["offset"] = cand["offset"].astype(np.int64)
    if pt01 is not None:
        rec["post_mean"] = pt01["post_mean"].astype(np.float64)
        rec["greedy_return"] = pt01["greedy_return"].astype(np.float64)
    if extra:
        rec.update(extra)
    return rec


def slice_rec(rec, keep_mask):
    """Return a new rec dict restricted to ``keep_mask`` (bool [N])."""
    out = {}
    for k, v in rec.items():
        a = np.asarray(v)
        out[k] = a[keep_mask]
    return out


def eval_dev_confirm(rec):
    """Split an eval rec into (dev, confirm) recs by offset (0..299 / 300..399)."""
    off = np.asarray(rec["offset"], dtype=np.int64)
    dev = slice_rec(rec, (off >= 0) & (off <= 299))
    conf = slice_rec(rec, (off >= 300) & (off < 400))
    return dev, conf


# ---------------------------------------------------------------------------
# Metrics & paired moving-block bootstrap (array-based; no per-row dicts)
# ---------------------------------------------------------------------------

def daily_rank_ic_series(rec, field, dense_min):
    """Per-date daily RankIC over dense dates (dict date -> ic).

    O(n_dates) string sorts avoided: dates are grouped once via argsort, then
    each field slices into the same pre-grouped windows.
    """
    score = np.asarray(rec[field], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(np.asarray(rec["true_logret"], dtype=np.float64)) \
        & np.asarray(rec["quality"]).astype(bool)
    order, bounds, uniq = _split_points(rec["date_key"])
    s = score[order]
    v = valid[order]
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        m = v[lo:hi]
        c = int(m.sum())
        if c < dense_min:
            continue
        ss = s[lo:hi][m]
        tt = np.asarray(rec["true_logret"], dtype=np.float64)[order[lo:hi]][m]
        if c >= 2 and not np.all(ss == ss[0]):
            ic = float(spearmanr(ss, tt)[0])
        else:
            ic = 0.0
        out[uniq[i]] = ic
    return out


def _split_points(dates):
    """Stable date grouping -> (order, bounds, uniq)."""
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(dates)]]).astype(np.int64)
    uniq = [str(sd[int(bounds[i])]) for i in range(len(bounds) - 1)]
    return order, bounds, uniq


class DailyIcCache:
    """Pre-group a rec's dates once; compute per-field daily IC series cheaply."""

    def __init__(self, rec, dense_min):
        self.dense_min = dense_min
        self.order, self.bounds, self.uniq = _split_points(rec["date_key"])
        self.true = np.asarray(rec["true_logret"], dtype=np.float64)[self.order]
        self.qual = np.asarray(rec["quality"]).astype(bool)[self.order]

    def series(self, score):
        s = np.asarray(score, dtype=np.float64)[self.order]
        out = {}
        for i in range(len(self.bounds) - 1):
            lo, hi = int(self.bounds[i]), int(self.bounds[i + 1])
            ss, tt, qq = s[lo:hi], self.true[lo:hi], self.qual[lo:hi]
            m = np.isfinite(ss) & np.isfinite(tt) & qq
            c = int(m.sum())
            if c < self.dense_min:
                continue
            sv, tv = ss[m], tt[m]
            if c >= 2 and not np.all(sv == sv[0]):
                ic = float(spearmanr(sv, tv)[0])
            else:
                ic = 0.0
            out[self.uniq[i]] = ic
        return out


def circular_block_means(deltas, block_length, n_replicates=10_000, seed=42):
    """Vectorized circular moving-block bootstrap (Politis & Romano).

    Statistically identical to ``compare_posttrain.circular_moving_block_bootstrap``
    (iid block starts over 0..n-1 with wraparound, tail truncated to n) but fully
    vectorized — ~100x faster for the many arm-vs-reference comparisons the
    R series needs.
    """
    n = len(deltas)
    rng = np.random.RandomState(seed)
    block = max(int(block_length), 1)
    n_starts = int(np.ceil(n / block))
    starts = rng.randint(0, n, size=(n_replicates, n_starts))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]) % n     # [R, n_starts, block]
    flat = idx.reshape(n_replicates, -1)[:, :n]
    return deltas[flat].mean(axis=1)


def bootstrap_vs(rec, field, ref_rec, ref_field, dense_min, block_lengths=(5, 10, 20),
                 n_replicates=10_000, seed=42, _cand_cache=None, _ref_cache=None):
    """Paired circular moving-block bootstrap of daily RankIC deltas.

    Per-date candidate IC minus per-date reference IC over the common dense-date
    set.  ``_cand_cache``/``_ref_cache`` (DailyIcCache) pre-group dates so a
    multi-arm summary pays the grouping cost once per rec.
    """
    cc = _cand_cache if _cand_cache is not None else DailyIcCache(rec, dense_min)
    rc = _ref_cache if _ref_cache is not None else DailyIcCache(ref_rec, dense_min)
    cs = cc.series(rec[field])
    rs = rc.series(ref_rec[ref_field])
    common = sorted(set(cs) & set(rs))
    if len(common) < 2:
        return {"n_dates": len(common), "point": None, "block_cis": None,
                "error": "fewer than 2 common dense dates"}
    c_series = np.asarray([cs[d] for d in common], dtype=float)
    r_series = np.asarray([rs[d] for d in common], dtype=float)
    deltas = c_series - r_series
    point = float(deltas.mean())
    cis = {}
    for L in block_lengths:
        means = circular_block_means(deltas, L, n_replicates, seed)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        cis[str(L)] = {"block_length": int(L), "ci_lower": lo, "ci_upper": hi,
                       "ci_excludes_zero": bool(lo <= 0.0 <= hi) is False,
                       "significant_directional": bool(lo > 0.0)}
    return {"metric": "rank_ic", "n_dates": len(common), "point": point,
            "candidate_mean": float(c_series.mean()),
            "reference_mean": float(r_series.mean()),
            "block_cis": cis,
            "block_robust": all(v["significant_directional"] for v in cis.values())}


def metrics_table(rec, fields, dense_threshold):
    """arm_metrics for each named field -> {name: metrics} (full 400 window)."""
    out = {}
    for name, field in fields.items():
        if field not in rec or rec[field] is None:
            continue
        try:
            m = arm_metrics(rec, field, dense_threshold)
        except Exception as e:  # noqa: BLE001
            m = {"error": str(e)}
        m.pop("per_date", None)
        out[name] = m
    return out


def region_dense_threshold(cand, frac=0.8, floor=5):
    """Per-region dense threshold = frac * max cross-section (plan uses 0.8)."""
    dates = cand["date_key"]
    _, counts = np.unique(dates, return_counts=True)
    return max(floor, int(np.ceil(frac * int(counts.max()))))


def calib_dense_threshold(seed=42):
    c = np.load(cand_path("calib", seed), allow_pickle=True)
    return region_dense_threshold(c)


# ---------------------------------------------------------------------------
# Evaluation summary helper
# ---------------------------------------------------------------------------

def summarize(rec, fields, dense_threshold, refs=None, label=""):
    """Full-window metrics + dev/confirm + optional paired bootstrap vs refs.

    ``refs``: dict ref_name -> (ref_rec, ref_field).  Returns a JSON-able dict.
    """
    out = {"label": label, "dense_threshold": dense_threshold,
           "full": metrics_table(rec, fields, dense_threshold)}
    dev, conf = eval_dev_confirm(rec)
    if dev and len(dev["stock_uid"]):
        out["dev_0_299"] = metrics_table(dev, fields, dense_threshold)
        out["confirm_300_399"] = metrics_table(conf, fields, dense_threshold)
    if refs:
        cc = DailyIcCache(rec, dense_threshold)
        out["bootstrap_vs"] = {}
        for ref_name, (ref_rec, ref_field) in refs.items():
            rc = DailyIcCache(ref_rec, dense_threshold)
            out["bootstrap_vs"][ref_name] = {}
            for name, field in fields.items():
                if field not in rec or rec[field] is None:
                    continue
                try:
                    out["bootstrap_vs"][ref_name][name] = bootstrap_vs(
                        rec, field, ref_rec, ref_field, dense_threshold,
                        _cand_cache=cc, _ref_cache=rc)
                except Exception as e:  # noqa: BLE001
                    out["bootstrap_vs"][ref_name][name] = {"error": str(e)}
    return out


def write_json_ledger(path, payload, event, **trial):
    """Write JSON + append a trial-ledger entry."""
    write_json(path, payload)
    append_trial({"event": event, "status": "ok", **trial})
    return path


# ---------------------------------------------------------------------------
# Improvement-plan protocol guards (plan §6 new contracts)
# ---------------------------------------------------------------------------

# Fields that are FORBIDDEN as rank scores (E-1): token likelihoods, entropy
# and acceptance aggregations.  Only decoded-return / return-semantic-head
# fields may enter arm_metrics as a row score (R4 abstention scores excepted —
# they never enter the RankIC ranking).
FORBIDDEN_RANK_FIELDS = ("logp", "margin", "entropy", "electra_acceptance")


def assert_score_semantics(field):
    """Contract 1 (plan §6): a rank-score field must not be a raw likelihood /
    entropy / acceptance aggregation.  Raises ValueError otherwise."""
    name = field.lower()
    if any(f in name for f in FORBIDDEN_RANK_FIELDS):
        raise ValueError(
            f"rank score '{field}' violates row-score semantics (E-1): "
            "token-likelihood / entropy / acceptance quantities are forbidden "
            "as row-level ranking scores; decode the posterior to a return")


def require_ca_not_regressed(ca_path, baseline_avg_rank=3.23, max_regression=0.3):
    """Contract 4 (plan §6): a fine-tuned BERT may proceed to R/T continuation
    only if C-a did not regress.  Raises RuntimeError otherwise."""
    from critic_common import load_json
    if not Path(ca_path).exists():
        raise RuntimeError(f"C-a artifact missing (must rerun after fine-tune): {ca_path}")
    ca = load_json(ca_path)
    avg_rank = float(ca["c_a"]["avg_true_rank"])
    if avg_rank > baseline_avg_rank + max_regression:
        raise RuntimeError(
            f"C-a regressed: avg_true_rank {avg_rank:.2f} > baseline "
            f"{baseline_avg_rank} + {max_regression} (max allowed regression)")
    return ca


def row_set_fingerprint(stock_uid, date_key, ckpt_path, out_json=None):
    """Contract 5 (plan §6): deterministic fingerprint of a BERT-hidden cache —
    the (uid, date) row set + the BERT checkpoint hash that produced it."""
    import hashlib
    from experiment_io import file_sha256
    u = np.asarray(stock_uid); d = np.asarray(date_key)
    row_hash = hashlib.sha256()
    row_hash.update(str(len(u)).encode())
    for uu, dd in zip(sorted(set(map(str, u))), sorted(set(map(str, d)))):
        row_hash.update(uu.encode()); row_hash.update(b"|"); row_hash.update(dd.encode())
    fp = {
        "schema": "bert-hidden-cache-fingerprint-v1",
        "n_rows": int(len(u)),
        "n_unique_uids": int(len(set(map(str, u)))),
        "n_unique_dates": int(len(set(map(str, d)))),
        "row_set_sha256": row_hash.hexdigest(),
        "bert_ckpt_sha256": file_sha256(Path(ckpt_path)),
        "ckpt_path": str(ckpt_path),
    }
    if out_json is not None:
        write_json(Path(out_json), fp)
    return fp
