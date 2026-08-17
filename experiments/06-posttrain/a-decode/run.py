"""PT-01: exact joint posterior decode over the formal 400-day window.

Consumes the frozen hidden cache (``cache_hidden.py``), runs the exact joint
decoder (``joint_decoder.decode_joint``), writes per-row prediction records, and
computes the J0-J5 arm metrics plus proper scores.

Prediction record fields (per date x stock_uid; PostTrain-ToDo §19):
    stock_uid, date_key, symbol, offset, position
    true_logret, true_coarse_id, true_fine_id
    greedy_c, greedy_f, greedy_return                      (J0)
    map_c, map_f, map_return                               (J1)
    post_mean, post_median, post_std, q10, q90             (J2/J3)
    p_up                                                   (J4)
    coarse_entropy, cond_fine_entropy, joint_entropy       (J5)
    full_joint_nll, ordinary_joint_nll                     (J5)
    p_mean0, p_std0, quality
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_io import file_sha256  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402

from common import (  # noqa: E402
    load_reviewed_selection, upstream_checkpoint_path, resolve_roots,
    artifact_paths, write_json,
)
from common import DecodeTable, decode_joint  # noqa: E402


# ============================================================================
# CRPS for a discrete posterior (O(M) per row using the sorted support)
# ============================================================================

def crps_discrete(sorted_returns, probs, y):
    """CRPS of a discrete distribution vs scalar y.

    ``sorted_returns``: [.., M] ascending raw returns (shared order)
    ``probs``:          [.., M] posterior probabilities (same order)
    ``y``:              [..] true raw return
    Returns [..] CRPS.  CRPS = E|X-y| - 0.5 E|X-X'|.
    """
    # cumulative sums over sorted support
    P = np.cumsum(probs, axis=-1)             # [.., M]
    PX = np.cumsum(probs * sorted_returns, axis=-1)  # [.., M]
    # term_i = E|X - x_i| = x_i*(2*F_i - P_total) - (2*PX_i - PX_total)
    # For sorted x_i: E|X - x_i| = x_i*(F_{<=i} - F_{>i}) - (PX_{<=i} - PX_{>i})
    #   = x_i*(2*P_i - 1) - (2*PX_i - PX_total)
    total_prob = P[..., -1]
    total_px = PX[..., -1]
    term = sorted_returns * (2.0 * P - total_prob) - (2.0 * PX - total_px)
    e_abs = np.sum(probs * np.abs(sorted_returns - y[..., None]), axis=-1)
    e_pair = np.sum(probs * term, axis=-1)
    return e_abs - 0.5 * e_pair


# ============================================================================
# Metrics
# ============================================================================

def _per_date(rec, dense_threshold):
    """Group row indices by date; returns (unique_dates, inv)."""
    uniq, inv = np.unique(rec["date_key"], return_inverse=True)
    return uniq, inv


def arm_metrics(rec, score_field, dense_threshold, proper=None):
    """Aggregate + per-date metrics for a continuous-score arm.

    ``rec``: dict of arrays. ``score_field``: one of greedy_return / map_return /
    post_mean / post_median.  Returns dict with avg_daily_rank_ic, avg_da_per_date,
    avg_mape, avg_mae, n_dense_dates, per_date series.
    """
    score = rec[score_field]
    true = rec["true_logret"]
    valid = np.isfinite(score) & np.isfinite(true) & rec["quality"]
    uniq, inv = _per_date(rec, dense_threshold)
    n_dates = len(uniq)
    da = np.full(n_dates, np.nan)
    ic = np.full(n_dates, np.nan)
    mae = np.full(n_dates, np.nan)
    mape = np.full(n_dates, np.nan)
    cnt = np.zeros(n_dates, dtype=np.int64)
    for i in range(n_dates):
        m = (inv == i) & valid
        c = int(m.sum())
        cnt[i] = c
        if c < dense_threshold:
            continue
        s = score[m]
        t = true[m]
        da[i] = float(np.mean((np.sign(s) > 0) == (t > 0)))
        if c >= 2 and not np.all(s == s[0]):
            from scipy.stats import spearmanr
            ic[i] = float(spearmanr(s, t)[0])
        else:
            ic[i] = 0.0
        mae[i] = float(np.mean(np.abs(s - t)))
        # MAPE in PRICE space (matches the project evaluator): pred_close/true_close
        # ratio cancels base_close -> |exp(pred - true) - 1|.
        mape[i] = float(np.mean(np.abs(np.exp(s - t) - 1.0)))
    dense = cnt >= dense_threshold
    out = {
        "score_field": score_field,
        "n_dense_dates": int(dense.sum()),
        "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
        "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        "avg_mape": float(np.nanmean(mape[dense])) if dense.any() else None,
        "avg_mae": float(np.nanmean(mae[dense])) if dense.any() else None,
        "per_date": {
            uniq[i]: {"da": da[i], "rank_ic": ic[i], "mae": mae[i], "n": int(cnt[i])}
            for i in range(n_dates)
        },
    }
    if proper is not None:
        out["proper"] = proper
    return out


def direction_arm_metrics(rec, dense_threshold):
    """Metrics for the J4 P(up) arm (direction probability)."""
    prob = rec["p_up"]
    true = rec["true_logret"]
    valid = np.isfinite(prob) & np.isfinite(true) & rec["quality"]
    y = (true > 0).astype(np.float64)
    brier = float(np.mean((prob[valid] - y[valid]) ** 2)) if valid.any() else None
    eps = 1e-12
    ll = -np.mean(y[valid] * np.log(prob[valid] + eps)
                  + (1 - y[valid]) * np.log(1 - prob[valid] + eps)) if valid.any() else None
    # DA + RankIC of the implied direction score
    rec_dir = {k: v for k, v in rec.items()}
    rec_dir["direction_prob"] = prob
    # DA per date
    uniq, inv = _per_date(rec, dense_threshold)
    n_dates = len(uniq)
    da = np.full(n_dates, np.nan)
    ic = np.full(n_dates, np.nan)
    cnt = np.zeros(n_dates, dtype=np.int64)
    for i in range(n_dates):
        m = (inv == i) & valid
        c = int(m.sum())
        cnt[i] = c
        if c < dense_threshold:
            continue
        da[i] = float(np.mean((prob[m] > 0.5) == (true[m] > 0)))
        if c >= 2 and not np.all(prob[m] == prob[m][0]):
            from scipy.stats import spearmanr
            ic[i] = float(spearmanr(prob[m], true[m])[0])
        else:
            ic[i] = 0.0
    dense = cnt >= dense_threshold
    ece = _ece(prob[valid], y[valid])
    return {
        "score_field": "p_up",
        "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
        "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        "brier": brier,
        "log_loss": ll,
        "ece": ece,
        "n_valid": int(valid.sum()),
        "per_date": {uniq[i]: {"da": da[i], "rank_ic": ic[i], "n": int(cnt[i])}
                     for i in range(n_dates)},
    }


def _ece(prob, y, n_bins=10):
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(prob[m].mean() - y[m].mean())
    return float(ece)


def distribution_metrics(rec):
    """J5 proper scores from the exact posterior NLLs + CRPS."""
    valid = rec["quality"] & (rec["true_coarse_id"] >= 0) & (rec["true_fine_id"] >= 0)
    n = int(valid.sum())
    out = {
        "n_valid": n,
        "full_joint_nll_mean": float(np.nanmean(rec["full_joint_nll"][valid])) if n else None,
        "ordinary_joint_nll_mean": float(np.nanmean(rec["ordinary_joint_nll"][valid])) if n else None,
        "coarse_entropy_mean": float(np.nanmean(rec["coarse_entropy"])) if len(rec["coarse_entropy"]) else None,
        "cond_fine_entropy_mean": float(np.nanmean(rec["cond_fine_entropy"])) if len(rec["cond_fine_entropy"]) else None,
        "joint_entropy_mean": float(np.nanmean(rec["joint_entropy"])) if len(rec["joint_entropy"]) else None,
        "special_mass_mean": float(np.nanmean(rec["special_mass"])) if len(rec["special_mass"]) else None,
        "posterior_std_mean": float(np.nanmean(rec["post_std"])) if len(rec["post_std"]) else None,
        "crps_mean": float(np.nanmean(rec["crps"])) if "crps" in rec and len(rec["crps"]) else None,
    }
    return out


# ============================================================================
# Main runner
# ============================================================================

def run_pt01(hidden_path, model_path, tokenizer_path, device, chunk=512,
             out_json=None):
    tok = load_tokenizer(str(tokenizer_path), device)
    model = load_gpt(str(model_path), device, tokenizer=tok)
    model.eval()
    dt = DecodeTable(tok, device="cpu")

    data = np.load(hidden_path, allow_pickle=True)
    hidden = torch.from_numpy(data["hidden"]).to(device)
    n = hidden.shape[0]

    rec = {k: [] for k in ("stock_uid", "date_key", "symbol", "offset", "position",
                           "true_logret", "true_coarse_id", "true_fine_id",
                           "greedy_c", "greedy_f", "greedy_return",
                           "map_c", "map_f", "map_return",
                           "post_mean", "post_median", "post_std", "q10", "q90",
                           "p_up", "coarse_entropy", "cond_fine_entropy",
                           "joint_entropy", "full_joint_nll", "ordinary_joint_nll",
                           "special_mass", "p_mean0", "p_std0", "quality", "crps")}
    meta_fields = {k: data[k] for k in ("stock_uid", "date_key", "symbol", "offset",
                                        "position", "p_mean0", "p_std0",
                                        "true_logret", "true_coarse_id", "true_fine_id")}
    dense_threshold = int(data["dense_threshold"][0])

    p_mean0 = torch.from_numpy(np.asarray(data["p_mean0"], dtype=np.float32)).to(device)
    p_std0 = torch.from_numpy(np.asarray(data["p_std0"], dtype=np.float32)).to(device)
    tc = torch.from_numpy(np.asarray(data["true_coarse_id"], dtype=np.int64)).to(device)
    tf = torch.from_numpy(np.asarray(data["true_fine_id"], dtype=np.int64)).to(device)

    tr = torch.from_numpy(np.asarray(data["true_logret"], dtype=np.float32)).to(device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        h = hidden[start:stop]
        stats, quality = decode_joint(
            model, dt, h, p_mean0[start:stop], p_std0[start:stop],
            true_coarse_ids=tc[start:stop], true_fine_ids=tf[start:stop],
            true_logret=tr[start:stop], chunk=256)
        crps = stats.crps
        s = stats
        rec["stock_uid"].extend(meta_fields["stock_uid"][start:stop].tolist())
        rec["date_key"].extend(meta_fields["date_key"][start:stop].tolist())
        rec["symbol"].extend(meta_fields["symbol"][start:stop].tolist())
        rec["offset"].extend(meta_fields["offset"][start:stop].tolist())
        rec["position"].extend(meta_fields["position"][start:stop].tolist())
        rec["true_logret"].extend(meta_fields["true_logret"][start:stop].tolist())
        rec["true_coarse_id"].extend(meta_fields["true_coarse_id"][start:stop].tolist())
        rec["true_fine_id"].extend(meta_fields["true_fine_id"][start:stop].tolist())
        for key, arr in (("greedy_c", s.greedy_c), ("greedy_f", s.greedy_f),
                         ("greedy_return", s.greedy_return), ("map_c", s.map_c),
                         ("map_f", s.map_f), ("map_return", s.map_return),
                         ("post_mean", s.mean), ("post_median", s.median),
                         ("post_std", s.std), ("q10", s.q10), ("q90", s.q90),
                         ("p_up", s.p_up), ("coarse_entropy", s.coarse_entropy),
                         ("cond_fine_entropy", s.cond_fine_entropy),
                         ("joint_entropy", s.joint_entropy),
                         ("full_joint_nll", s.full_joint_nll),
                         ("ordinary_joint_nll", s.ordinary_joint_nll),
                         ("special_mass", s.special_mass)):
            rec[key].extend(arr.detach().cpu().numpy().tolist())
        rec["p_mean0"].extend(meta_fields["p_mean0"][start:stop].tolist())
        rec["p_std0"].extend(meta_fields["p_std0"][start:stop].tolist())
        rec["quality"].extend(quality.cpu().numpy().tolist())
        rec["crps"].extend(crps.detach().cpu().numpy().tolist())
        if (start // chunk) % 50 == 0:
            print(f"[pt01] rows {start}/{n}")

    # pack to arrays
    rec_arr = {k: np.asarray(v) for k, v in rec.items()}
    # ensure float fields are float64
    for k in ("greedy_return", "map_return", "post_mean", "post_median", "post_std",
              "q10", "q90", "p_up", "coarse_entropy", "cond_fine_entropy",
              "joint_entropy", "full_joint_nll", "ordinary_joint_nll",
              "special_mass", "crps"):
        rec_arr[k] = rec_arr[k].astype(np.float64)
    for k in ("greedy_c", "greedy_f", "map_c", "map_f", "true_coarse_id",
              "true_fine_id", "offset", "position"):
        rec_arr[k] = rec_arr[k].astype(np.int64)

    # metrics per arm
    arms = {}
    for field in ("greedy_return", "map_return", "post_mean", "post_median"):
        arms[f"J0_greedy" if field == "greedy_return" else
             ("J1_joint_map" if field == "map_return" else
              ("J2_posterior_mean" if field == "post_mean" else "J3_posterior_median"))] = \
            arm_metrics(rec_arr, field, dense_threshold)
    arms["J4_p_up"] = direction_arm_metrics(rec_arr, dense_threshold)
    arms["J5_distribution"] = distribution_metrics(rec_arr)

    result = {
        "schema_version": "pt01-v1",
        "n_rows": int(n),
        "dense_threshold": dense_threshold,
        "arms": arms,
        "offsets_scope": "0_399",
        "holdout_used": False,
    }
    if out_json:
        write_json(out_json, result)
    return result, rec_arr


def main():
    ap = argparse.ArgumentParser(description="PT-01 exact joint decode baseline")
    ap.add_argument("--hidden", required=True, help="hidden cache NPZ path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sel = load_reviewed_selection()
    ckpt = upstream_checkpoint_path(sel)
    tok = Path(sel["upstream"]["tokenizer"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    roots = resolve_roots()
    paths = artifact_paths(roots=roots)
    out = args.out or (roots.results_root / "A-decode" / "decode.json")
    result, rec = run_pt01(args.hidden, ckpt, tok, device, chunk=args.chunk,
                           out_json=Path(out))
    # also write the per-row records to weights root (large artifact)
    npz_path = paths["records"]
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **{k: np.asarray(v) for k, v in rec.items()})
    print(f"[pt01] records -> {npz_path}")
    print(json.dumps(result["arms"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
