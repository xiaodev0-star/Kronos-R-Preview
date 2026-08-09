"""PT-03 training/calibration hidden caches over pre-cutoff positions.

The eval-region cache (cache_hidden.py) covers offsets 0..399 (post-cutoff) and
is used for PT-01 evaluation.  Probe/set heads must be TRAINED on pre-cutoff
data only, so this script builds:

  training cache   fit_uids   x target dates < 2023-02-01   (subsampled to budget)
  calibration cache audit_uids x [2023-02-01, 2024-02-01)   (for PT-02 / thresholds)

Each row stores hidden + raw targets + point-in-time C1 features (last return,
5/20-day momentum, 20-day realized vol and volume), all computed from data
visible strictly before position p.  No post-cutoff label is consumed.

Sampling uses a per-stock stride so every training year is covered; the exact
budget is a legitimate choice for tiny frozen probes (protocol §13.5 limits
frozen heads to 20 equivalent data passes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig, NormConfig  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402

from posttrain_common import (  # noqa: E402
    load_reviewed_selection, upstream_checkpoint_path, resolve_roots,
)
from posttrain_data import (  # noqa: E402
    load_stocks_uid, attach_close_prices_uid, posttrain_split_v1,
    build_stock_arrays_uid,
)

FINAL_FIT_STOP = "2023-02-01"
CALIB_START, CALIB_STOP = "2023-02-01", "2024-02-01"
MIN_POS = NormConfig.min_lookback + 1  # need history before p


def c1_features(feat, p, window=20):
    """Point-in-time C1 features at position p (uses rows < p only)."""
    if p < MIN_POS:
        return None
    lr = feat[:, 0]
    last = lr[p - 1]
    mom5 = float(np.sum(lr[max(0, p - 5):p]))
    mom20 = float(np.sum(lr[max(0, p - 20):p]))
    vol20 = float(np.std(lr[max(0, p - 20):p]))
    vol_amt = float(np.std(feat[max(0, p - 20):p, 4]))  # log_vol column
    return np.array([last, mom5, mom20, vol20, vol_amt], dtype=np.float32)


def select_positions(arrays, stop_date, stride):
    """Positions p with ci <= p (post-cutoff? no) and date(p) < stop_date.

    We select positions in the TRAINING period: p in [MIN_POS, ci) with
    date(p) < stop_date.  Stride keeps the cache at budget while covering the
    full training span.
    """
    dates = arrays["dates_dt"]
    ci = arrays["ci"]
    stop64 = np.datetime64(stop_date)
    cand = []
    for p in range(MIN_POS, ci):
        if dates[p] >= stop64:
            break
        cand.append(p)
    if not cand:
        return []
    return cand[::stride]


def extract_caches(*, device, batch_size=4, target_rows=1_200_000, roots=None):
    sel = load_reviewed_selection()
    ckpt_path = upstream_checkpoint_path(sel)
    tok_path = Path(sel["upstream"]["tokenizer"])
    tokenizer = load_tokenizer(str(tok_path), device)
    stocks = load_stocks_uid(DataConfig.data_dir)
    fit_uids, audit_uids = posttrain_split_v1(stocks)
    print(f"[train-cache] fit_uids={len(fit_uids)} audit_uids={len(audit_uids)}")

    model = load_gpt(str(ckpt_path), device, tokenizer=tokenizer)
    model.eval()
    dim = model.head_coarse.in_features

    # ---- build per-stock arrays for fit and audit ----
    attach_close_prices_uid(stocks)
    fit_arrays = [a for s in stocks if (a := build_stock_arrays_uid(s)) is not None
                  and s["stock_uid"] in fit_uids]
    audit_arrays = [a for s in stocks if (a := build_stock_arrays_uid(s)) is not None
                    and s["stock_uid"] in audit_uids]
    print(f"[train-cache] fit arrays={len(fit_arrays)} audit arrays={len(audit_arrays)}")

    # per-stock stride to target approx target_rows across fit stocks
    n_fit = len(fit_arrays)
    total_train_days = sum(len(a["dates_dt"]) for a in fit_arrays)
    stride = max(1, int(total_train_days / max(1, target_rows)))
    print(f"[train-cache] fit stride={stride}")

    # ---- extract fit positions (hidden + targets + c1) ----
    def pack_batches(arr_list, stop_date, stride_val):
        prepped = []
        for a in arr_list:
            poss = select_positions(a, stop_date, stride_val)
            if not poss:
                continue
            prepped.append((a, poss))
        batches = []
        # group by seq_len buckets (reuse prepared-input shape building)
        by_len = {}
        for a, poss in prepped:
            by_len.setdefault(a["T_total"], []).append((a, poss))
        for length, group in sorted(by_len.items()):
            for start in range(0, len(group), batch_size):
                batches.append(group[start:start + batch_size])
        return batches

    def run_batches(arr_groups, stop_date, stride_val, out_prefix):
        """arr_groups: list of (arrays, positions) tuples -> returns rows dict."""
        rows = {k: [] for k in ("stock_uid", "date_key", "hidden", "true_logret",
                                "true_coarse_id", "true_fine_id", "c1_feats",
                                "p_mean0", "p_std0", "offset_note")}
        n_rows = 0
        by_len = {}
        for a, poss in arr_groups:
            by_len.setdefault(a["T_total"], []).append((a, poss))
        import itertools
        # process by length bucket
        for length, group in sorted(by_len.items()):
            for gstart in range(0, len(group), batch_size):
                chunk = group[gstart:gstart + batch_size]
                max_len = length + 1  # BOS + tokens
                B = len(chunk)
                input_ids = torch.zeros(B, max_len, dtype=torch.long)
                time_ids = torch.zeros(B, max_len, 3, dtype=torch.long)
                va_values = torch.zeros(B, max_len, 2, dtype=torch.float32)
                sel_rows, sel_pos = [], []
                all_meta = []
                for bi, (a, poss) in enumerate(chunk):
                    T = a["T_total"]
                    idx_c = a["_coarse"]
                    idx_f = a["_fine"]
                    day = np.asarray(a["day"]); month = np.asarray(a["month"])
                    year = np.asarray(a["year"])
                    # BOS-aligned time arrays: position p uses day of feature row
                    # p-1, matching the evaluator (input position p predicts day p).
                    day_a = np.concatenate([[day[0]], day[:T - 1]])
                    month_a = np.concatenate([[month[0]], month[:T - 1]])
                    year_a = np.concatenate([[year[0]], year[:T - 1]])
                    inp_ids = [int(tokenizer.vocab_coarse)] + idx_c[:T].tolist()
                    seq_len = len(inp_ids) - 1
                    va_seq = np.concatenate([np.zeros((1, 2), dtype=np.float32),
                                             a["va_normed"][:T - 1]], axis=0)
                    input_ids[bi, :seq_len] = torch.as_tensor(inp_ids[:-1])
                    time_ids[bi, :seq_len, 0] = torch.as_tensor(day_a[:seq_len], dtype=torch.long)
                    time_ids[bi, :seq_len, 1] = torch.as_tensor(month_a[:seq_len], dtype=torch.long)
                    time_ids[bi, :seq_len, 2] = torch.as_tensor(year_a[:seq_len], dtype=torch.long)
                    va_values[bi, :seq_len] = torch.as_tensor(va_seq[:seq_len])
                    for p in poss:
                        if p >= seq_len:
                            continue
                        feat = a["feat"]
                        cf = c1_features(feat, p)
                        if cf is None:
                            continue
                        dkey = str(a["dates_dt"][p])[:10]
                        sel_rows.append(bi)
                        sel_pos.append(p)
                        all_meta.append((a["stock_uid"], dkey, float(feat[p, 0]),
                                         int(a["_coarse"][p]), int(a["_fine"][p]),
                                         cf, float(a["p_mean"][0]), float(a["p_std"][0])))
                if not all_meta:
                    continue
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
                    h = model.encode_selected(
                        input_ids.to(device), time_ids.to(device), pos_ids,
                        torch.tensor(sel_rows, device=device),
                        torch.tensor(sel_pos, device=device),
                        va_values=va_values.to(device))
                    h = h.float().cpu().numpy()
                for r, (uid, dkey, tlr, tco, tfi, cf, pm, ps) in enumerate(all_meta):
                    rows["stock_uid"].append(uid)
                    rows["date_key"].append(dkey)
                    rows["hidden"].append(h[r])
                    rows["true_logret"].append(tlr)
                    rows["true_coarse_id"].append(tco)
                    rows["true_fine_id"].append(tfi)
                    rows["c1_feats"].append(cf)
                    rows["p_mean0"].append(pm)
                    rows["p_std0"].append(ps)
                n_rows += len(all_meta)
                if n_rows % 200_000 == 0:
                    print(f"[{out_prefix}] rows {n_rows}")
        # tokenize once per stock up front
        return rows, n_rows

    # tokenize fit + audit stocks
    from model.tokenizer import HierarchicalQuantizer
    for a in fit_arrays + audit_arrays:
        a["_coarse"], a["_fine"] = None, None
    # batch tokenize
    def tokenize_all(arr_list):
        with torch.no_grad():
            for a in arr_list:
                T = a["T_total"]
                chunk = torch.from_numpy(a["price_normed"][:T]).float().unsqueeze(0).to(device)
                idx = tokenizer.encode_all(chunk).squeeze(0).cpu().numpy()
                a["_coarse"] = idx[:, 0].astype(np.int32)
                a["_fine"] = idx[:, 1].astype(np.int16)
    print("[train-cache] tokenizing...")
    tokenize_all(fit_arrays + audit_arrays)

    # fit groups
    fit_groups = []
    for a in fit_arrays:
        poss = select_positions(a, FINAL_FIT_STOP, stride)
        if poss:
            fit_groups.append((a, poss))
    rows_fit, n_fit_rows = run_batches(fit_groups, FINAL_FIT_STOP, stride, "fit")

    # calibration groups (audit)
    cal_groups = []
    for a in audit_arrays:
        # calibration positions: date in [CALIB_START, CALIB_STOP)
        dates = a["dates_dt"]
        ci = a["ci"]
        poss = [p for p in range(MIN_POS, ci)
                if CALIB_START <= str(dates[p])[:10] < CALIB_STOP]
        if poss:
            cal_groups.append((a, poss))
    rows_cal, n_cal_rows = run_batches(cal_groups, CALIB_STOP, 1, "cal")

    weights_root = (roots.weights_root if roots else resolve_roots().weights_root)
    weights_root.mkdir(parents=True, exist_ok=True)

    def save(rows, n_rows, path):
        arr = {k: (np.stack(v) if k == "hidden" or k == "c1_feats"
                   else np.asarray(v)) for k, v in rows.items()}
        np.savez(path, hidden=arr["hidden"],
                 c1_feats=arr["c1_feats"],
                 stock_uid=arr["stock_uid"], date_key=arr["date_key"],
                 true_logret=arr["true_logret"].astype(np.float64),
                 true_coarse_id=arr["true_coarse_id"].astype(np.int16),
                 true_fine_id=arr["true_fine_id"].astype(np.int16),
                 p_mean0=arr["p_mean0"].astype(np.float64),
                 p_std0=arr["p_std0"].astype(np.float64))
        print(f"[train-cache] wrote {path} ({n_rows} rows, {path.stat().st_size/1e6:.0f} MB)")

    save(rows_fit, n_fit_rows, weights_root / "training_cache.npz")
    save(rows_cal, n_cal_rows, weights_root / "calibration_cache.npz")
    return weights_root / "training_cache.npz", weights_root / "calibration_cache.npz"


def main():
    ap = argparse.ArgumentParser(description="Build PT-03 training/calibration caches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target_rows", type=int, default=1_200_000)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extract_caches(device=device, batch_size=args.batch_size,
                   target_rows=args.target_rows, roots=resolve_roots())


if __name__ == "__main__":
    main()
