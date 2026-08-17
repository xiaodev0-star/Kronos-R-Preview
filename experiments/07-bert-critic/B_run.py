"""07-B: BERT scoring + evaluation pipeline.

Usage is intentionally documented in README.md; this file is the B-stage
entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================================
# Path bootstrap
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ============================================================================
# Repo-module imports
# ============================================================================
from experiment_io import (  # noqa: E402
    StudyLayout, default_study_roots, file_sha256, assert_results_boundary,
)
from config import DataConfig  # noqa: E402
from model import load_tokenizer  # noqa: E402
from model.kronos_bert import KronosBert  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402

# ============================================================================
# common.py imports
# ============================================================================
from common import (  # noqa: E402
    VOCAB_BASE, MASK_ID, EPS,
    resolve_roots, upstream_paths, write_json, append_trial, dict_sha256,
    posttrain_artifacts, posttrain_common, weights_artifact, result_artifact, stage_weights,
    stage_results,
    MODEL_VARIANTS, model_checkpoint, model_variant,
    load_json, bert_time_for_target_date, selection_history_window,
    build_bert_input,
    MlpRankHead, fold_split, final_fit_split, train_rank_per_date,
    arm_metrics, paired_bootstrap_ci, circular_moving_block_bootstrap,
    load_model_cpu, coarse_q_for_hidden, gpt_q_path,
    write_prediction_parquet, PredictionParquetWriter,
    require_full_validation_coverage,
)

# ============================================================================
# common.py imports (merged improve_common)
# ============================================================================
from common import (  # noqa: E402
    weights_root, results_root, cand_path, scores_path,
    metrics_table, bootstrap_vs, build_rec,
    row_set_fingerprint, write_json_ledger,
)


# ############################################################################
#  B-1: Build GPT candidates (eval region)
# ############################################################################

TC = 1.4          # locked calibrated coarse temperature (plan s7.6)
TF = 1.1
K_DEFAULT = 8


def _load_model_cpu(ckpt, tok_path):
    """Load GPT model + tokenizer on CPU (from build_gpt_candidates)."""
    tok = load_tokenizer(str(tok_path), torch.device("cpu"))
    model = load_gpt(str(ckpt), torch.device("cpu"), tokenizer=tok)
    model.eval()
    return model, tok


@torch.no_grad()
def coarse_q_for_hidden(model, hidden, t_c=TC, vocab_base=128):
    """Coarse marginal q(c) over ordinary codes at temperature ``t_c``.

    hidden [B, dim] -> q [B, vocab_base] (Bayes-conditioned on ordinary codes,
    special mass excluded and renormalized -- same convention as joint_decoder).
    """
    logits = model.coarse_logits_from_hidden(hidden)          # [B, V+2]
    p_full = torch.softmax(logits / t_c, dim=-1)
    special_mass = p_full[:, vocab_base:].sum(dim=-1)          # [B]
    q = p_full[:, :vocab_base] / (1.0 - special_mass).clamp_min(EPS).unsqueeze(-1)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return q, special_mass


def _build_candidates(*, hidden_path, records_path, ckpt, tok_path,
                      out_npz, out_json, k=K_DEFAULT, chunk=100_000, device="cpu"):
    """Write the eval-region candidate cache."""
    model, tok = _load_model_cpu(ckpt, tok_path)

    hdata = np.load(hidden_path, allow_pickle=True)
    hidden = hdata["hidden"]                                  # [N, 256]
    n = hidden.shape[0]
    print(f"[cand] hidden rows {n}")

    rec = np.load(records_path, allow_pickle=True)
    if (len(hdata["stock_uid"]) != len(rec["stock_uid"])
            or not np.array_equal(hdata["stock_uid"], rec["stock_uid"])
            or not np.array_equal(hdata["date_key"], rec["date_key"])):
        raise RuntimeError(
            "BERT candidate source rows are not exactly aligned with the "
            "06 validation records; refusing to build a partial/misaligned cache"
        )
    # Posterior stats consumed as candidate-cache fields (PT-01, T=1.0 exact).
    stats = {
        "stock_uid": rec["stock_uid"], "date_key": rec["date_key"],
        "offset": rec["offset"].astype(np.int32),
        "position": rec["position"].astype(np.int32),
        "p_mean0": rec["p_mean0"].astype(np.float64),
        "p_std0": rec["p_std0"].astype(np.float64),
        "true_coarse_id": rec["true_coarse_id"].astype(np.int16),
        "true_logret": rec["true_logret"].astype(np.float64),
        "quality": rec["quality"].astype(bool),
        "post_median": rec["post_median"].astype(np.float64),
        "p_up": rec["p_up"].astype(np.float64),
        "post_std": rec["post_std"].astype(np.float64),
        "coarse_entropy": rec["coarse_entropy"].astype(np.float64),
        "joint_entropy": rec["joint_entropy"].astype(np.float64),
        "special_mass": rec["special_mass"].astype(np.float64),
        "greedy_return": rec["greedy_return"].astype(np.float64),
    }
    validation_coverage = require_full_validation_coverage(
        stats["offset"], stats["date_key"], label="GPT candidate source"
    )
    print(
        "[cand] row alignment OK (all rows match 06 records); "
        f"offsets={validation_coverage['offset_min']}.."
        f"{validation_coverage['offset_max']} rows={validation_coverage['n_rows']}"
    )

    topk_ids = np.zeros((n, k), dtype=np.int16)
    topk_logq = np.zeros((n, k), dtype=np.float32)
    special_mass = np.zeros(n, dtype=np.float64)
    dev = torch.device(device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        h = torch.from_numpy(hidden[start:stop]).to(dev)
        q, sm = coarse_q_for_hidden(model, h, t_c=TC, vocab_base=model._vocab_l1)
        logq = torch.log(q.clamp_min(EPS))
        ids = q.topk(k, dim=-1).indices.to(torch.int16)
        vals = logq.gather(-1, ids.to(torch.int64)).to(torch.float32)
        topk_ids[start:stop] = ids.cpu().numpy()
        topk_logq[start:stop] = vals.cpu().numpy()
        special_mass[start:stop] = sm.cpu().numpy()
        del h, q, logq, ids, vals
        if (start // chunk) % 5 == 0:
            print(f"[cand] rows {stop}/{n}")

    dense_threshold = int(hdata["dense_threshold"][0])
    out = {**stats,
           "topk_ids": topk_ids, "topk_logq": topk_logq,
           "special_mass": special_mass,
           "dense_threshold": np.array([dense_threshold]),
           "tc": np.array([TC]), "tf": np.array([TF]), "k": np.array([k])}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[cand] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")

    meta = {
        "schema": "candidates-v1",
        "n_rows": int(n),
        "k": k, "tc": TC, "tf": TF,
        "source": {
            "hidden_cache": str(hidden_path),
            "hidden_sha256": file_sha256(Path(hidden_path)),
            "posterior_records": str(records_path),
        },
        "upstream": {"checkpoint": str(ckpt), "sha256": file_sha256(ckpt),
                     "tokenizer": str(tok_path), "sha256": file_sha256(tok_path)},
        "dense_threshold": dense_threshold,
        "offsets_scope": "0_399",
        "validation_coverage": validation_coverage,
        "holdout_used": False,
        "candidate_sha256": dict_sha256({"topk_ids_shape": list(topk_ids.shape),
                                         "n": n, "k": k, "tc": TC}),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[cand] sidecar -> {out_json}")
    return out_npz, meta


def _stage_b1():
    """B-1: Build GPT candidates (eval region)."""
    ap = argparse.ArgumentParser(description="B-1: Build GPT candidates (eval)")
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    ckpt, tok_path = upstream_paths()
    try:
        pt06 = posttrain_artifacts(seed=42)
        hidden_path = Path(args.hidden) if args.hidden else pt06["hidden"]
        records_path = Path(args.records) if args.records else pt06["records"]
        out_npz = stage_weights("B") / f"candidates-eval-k{args.k}.npz"
        out_json = stage_results("B") / f"candidates-eval-k{args.k}.json"
        _build_candidates(hidden_path=hidden_path, records_path=records_path,
                          ckpt=ckpt, tok_path=tok_path, out_npz=out_npz,
                          out_json=out_json, k=args.k, device=args.device)
        append_trial({"event": "build_candidates", "region": "eval",
                      "k": args.k, "status": "ok"})
    except Exception as e:
        append_trial({"event": "build_candidates", "region": "eval",
                      "k": args.k, "status": "failed", "error": str(e)})
        raise


# ############################################################################
#  B-2: Build GPT candidates (calib region)
# ############################################################################

def _build_calib_candidates(*, calib_cache, ckpt, tok_path, out_npz, out_json,
                            k=K_DEFAULT, chunk=4096, device="cpu", light=False):
    """Calibration-region candidate cache (audit_uids x [2023-02-01, 2024-02-01)).

    Unlike the eval region (which reuses PT-01 posterior stats), the calibration
    slice has no existing posterior cache, so we run the exact joint decoder
    (decode_joint) over the calibration hidden to obtain f1-f3 features -- unless
    ``light=True`` (F-INT only needs the top-K candidates + targets, so the
    expensive fine expansion is skipped).
    """
    data = np.load(calib_cache, allow_pickle=True)
    n = len(data["stock_uid"])
    print(f"[calib-cand] rows {n} light={light}")
    tok = load_tokenizer(str(tok_path), torch.device(device))
    model = load_gpt(str(ckpt), torch.device(device), tokenizer=tok)
    model.eval()
    dev = torch.device(device)

    hidden = torch.from_numpy(data["hidden"]).to(dev)
    post = {kk: np.full(n, np.nan) for kk in
            ("post_median", "p_up", "post_std", "coarse_entropy",
             "joint_entropy", "special_mass")}
    topk_ids = np.zeros((n, k), dtype=np.int16)
    topk_logq = np.zeros((n, k), dtype=np.float32)
    q_accum = np.zeros((n, 128), dtype=np.float32)
    if not light:
        exp06 = posttrain_common()
        DecodeTable, decode_joint = exp06.DecodeTable, exp06.decode_joint
        dt = DecodeTable(tok, device="cpu")
        p_mean0 = torch.from_numpy(data["p_mean0"].astype(np.float32)).to(dev)
        p_std0 = torch.from_numpy(data["p_std0"].astype(np.float32)).to(dev)
        tc = torch.from_numpy(data["true_coarse_id"].astype(np.int64)).to(dev)
        tf = torch.from_numpy(data["true_fine_id"].astype(np.int64)).to(dev)
        tr = torch.from_numpy(data["true_logret"].astype(np.float32)).to(dev)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        if not light:
            stats, quality = decode_joint(
                model, dt, hidden[start:stop], p_mean0[start:stop], p_std0[start:stop],
                true_coarse_ids=tc[start:stop], true_fine_ids=tf[start:stop],
                true_logret=tr[start:stop], t_c=TC, t_f=TF, chunk=256)
            post["post_median"][start:stop] = stats.median.cpu().numpy()
            post["p_up"][start:stop] = stats.p_up.cpu().numpy()
            post["post_std"][start:stop] = stats.std.cpu().numpy()
            post["coarse_entropy"][start:stop] = stats.coarse_entropy.cpu().numpy()
            post["joint_entropy"][start:stop] = stats.joint_entropy.cpu().numpy()
            post["special_mass"][start:stop] = stats.special_mass.cpu().numpy()
        # top-K candidates from the calibrated q(c)
        q, sm = coarse_q_for_hidden(model, hidden[start:stop], t_c=TC,
                                    vocab_base=model._vocab_l1)
        logq = torch.log(q.clamp_min(EPS))
        ids = q.topk(k, dim=-1).indices.to(torch.int16)
        vals = logq.gather(-1, ids.to(torch.int64)).to(torch.float32)
        topk_ids[start:stop] = ids.cpu().numpy()
        topk_logq[start:stop] = vals.cpu().numpy()
        q_accum[start:stop] = q.cpu().numpy()
        if start % (chunk * 10) == 0:
            print(f"[calib-cand] rows {stop}/{n}")
    quality = (np.isfinite(data["p_mean0"]) & np.isfinite(data["p_std0"])
               & (np.asarray(data["p_std0"]) > 0))
    out = {
        "stock_uid": data["stock_uid"], "date_key": data["date_key"],
        "true_coarse_id": data["true_coarse_id"].astype(np.int16),
        "true_fine_id": data["true_fine_id"].astype(np.int16),
        "true_logret": data["true_logret"].astype(np.float64),
        "p_mean0": data["p_mean0"].astype(np.float64),
        "p_std0": data["p_std0"].astype(np.float64),
        "quality": quality.astype(bool),
        **post, "topk_ids": topk_ids, "topk_logq": topk_logq,
        "gpt_q": q_accum,
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[calib-cand] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    meta = {"schema": "candidates-calib-v1", "n_rows": n, "k": k, "tc": TC,
            "source": str(calib_cache), "upstream_sha256": file_sha256(ckpt)}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_npz


def _stage_b2():
    """B-2: Build GPT candidates (calib region)."""
    ap = argparse.ArgumentParser(description="B-2: Build GPT candidates (calib)")
    ap.add_argument("--calib_cache", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--light", action="store_true",
                    help="calib: skip decode_joint posterior (F-INT only needs top-K)")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    ckpt, tok_path = upstream_paths()
    try:
        pt06 = posttrain_artifacts(seed=42)
        calib_cache = Path(args.calib_cache) if args.calib_cache else pt06["calibration"]
        out_npz = stage_weights("B") / f"candidates-calib-k{args.k}.npz"
        out_json = stage_results("B") / f"candidates-calib-k{args.k}.json"
        _build_calib_candidates(calib_cache=calib_cache, ckpt=ckpt,
                                tok_path=tok_path, out_npz=out_npz,
                                out_json=out_json, k=args.k,
                                device=args.device, light=args.light)
        append_trial({"event": "build_candidates", "region": "calib",
                      "k": args.k, "status": "ok"})
    except Exception as e:
        append_trial({"event": "build_candidates", "region": "calib",
                      "k": args.k, "status": "failed", "error": str(e)})
        raise


# ############################################################################
#  B-3 / B-4: BERT scoring (from score_bert.py)
# ############################################################################

# ============================================================================
# BERT checkpoint loading
# ============================================================================

def load_bert(ckpt_path, device):
    """Load a KronosBert from a checkpoint with its recorded config."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = types.SimpleNamespace(
        vocab_size=int(cfg["vocab_size"]), vocab_fine=int(cfg.get("vocab_fine", 128)),
        dim=int(cfg["dim"]), depth=int(cfg["depth"]), heads=int(cfg["heads"]),
        num_kv_heads=int(cfg.get("num_kv_heads", 1)),
        ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        va_hidden_dim=int(cfg.get("va_hidden_dim", 64)),
        rope_base=float(cfg.get("rope_base", 10000.0)),
    )
    model = KronosBert(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg, ckpt


# ============================================================================
# Per-stock prepared-input index (shared across fit/calib/eval regions)
# ============================================================================

def build_index(tokenizer_path, device="cpu", max_stocks=0, cache_path=None):
    """Build (and optionally cache) the BertInputIndex over all stocks.

    Tokenizing all 4,695 stocks takes a few minutes; the index is reused by
    scoring and the C-a/C-b controls, so it is cached to ``cache_path``
    (a pickle).  ``max_stocks`` is for development smoke runs only.
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            idx = pickle.load(f)
        print(f"[index] loaded cached index {len(idx.by_uid)} stocks")
        return idx
    tokenizer = load_tokenizer(str(tokenizer_path), torch.device(device))
    exp06 = posttrain_common()
    load_stocks_uid = exp06.load_stocks_uid
    attach_close_prices_uid = exp06.attach_close_prices_uid
    prepare_stocks_uid = exp06.prepare_stocks_uid
    stocks = load_stocks_uid(DataConfig.data_dir)
    if max_stocks:
        stocks = stocks[:max_stocks]
    attach_close_prices_uid(stocks)
    prepped = prepare_stocks_uid(stocks, tokenizer, torch.device(device))
    idx = BertInputIndex(prepped)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[index] cached {len(prepped)} stocks -> {cache_path}")
    return idx


class BertInputIndex:
    """Compact per-stock store: lean tensors + int dates (fast pickle, no strings).

    ``by_uid`` holds, per stock, the pre-converted ``inp_ids`` / ``time_ids`` /
    ``va`` tensors plus an int32 ``dates_int`` array (YYYYMMDD) in chronological
    order.  ``position_for`` uses searchsorted on the sorted dates -- no
    string-keyed dict, so the cache pickle stays small and loads in seconds.
    """

    def __init__(self, prepped):
        self.by_uid = {}
        for p in prepped:
            uid = p["stock_uid"]
            day = np.asarray(p["day"], dtype=np.int64)
            month = np.asarray(p["month"], dtype=np.int64)
            year = np.asarray(p["year"], dtype=np.int64)
            time_ids = torch.from_numpy(
                np.stack([day, month, year], axis=-1)).to(torch.long)
            inp_ids = torch.as_tensor(p["inp_ids"], dtype=torch.long)
            va = torch.as_tensor(p["va"], dtype=torch.float32)
            dates_int = np.asarray(
                [int(str(d)[:10].replace("-", "")) for d in p["dates_raw"]],
                dtype=np.int32)
            self.by_uid[uid] = {"inp_ids": inp_ids, "time_ids": time_ids,
                                "va": va, "dates_int": dates_int}

    def position_for(self, uid, date_key):
        p = self.by_uid.get(uid)
        if p is None:
            return None
        d_int = int(str(date_key)[:10].replace("-", ""))
        i = int(np.searchsorted(p["dates_int"], d_int, side="left"))
        if i >= len(p["dates_int"]) or int(p["dates_int"][i]) != d_int:
            return None
        return i

    def bert_input(self, uid, date_key, window, vocab_base=VOCAB_BASE,
                   mask_id=VOCAB_BASE + 2, shuffle_seed=None, position=None):
        p = self.by_uid[uid]
        if position is not None:
            pos = int(position)
        else:
            pos = self.position_for(uid, date_key)
        if pos is None or pos < 0 or pos >= p["inp_ids"].shape[0]:
            return None
        # GPT input: inp_ids[0]=BOS, inp_ids[j]=token of row j-1; position pos
        # predicts feature row pos (date_key).  History window = positions
        # [pos-W+1, pos] of the SAME input arrays (all pre-converted tensors).
        start = max(0, pos + 1 - window)
        hist_ids = p["inp_ids"][start:pos + 1]
        hist_time = p["time_ids"][start:pos + 1]
        hist_va = p["va"][start:pos + 1]
        if shuffle_seed is not None:
            # C-b control: destroy the real ordering of the history window
            # (tokens AND their calendar/va together).  Discrimination that
            # survives must come from a side channel, not real context.
            rng = np.random.RandomState(int(shuffle_seed))
            perm = rng.permutation(hist_ids.shape[0])
            hist_ids = hist_ids[perm]
            hist_time = hist_time[perm]
            hist_va = hist_va[perm]
        target_time = bert_time_for_target_date(date_key)
        return build_bert_input(hist_ids, hist_time, hist_va, target_time,
                                vocab_base=vocab_base, mask_id=mask_id)

    def __contains__(self, uid):
        return uid in self.by_uid


# ============================================================================
# Scoring
# ============================================================================

@torch.no_grad()
def score_rows(index, rows, model, *, window=512, batch_size=32, device="cuda",
               shuffle_history=False):
    """Score rows against the BERT critic.

    ``rows`` is a lightweight row table: an object supporting ``__len__`` and
    per-index ``stock_uid(i)`` / ``date_key(i)`` / ``position(i)`` /
    ``topk_ids(i)`` accessors -- see ``NumpyRowTable`` below.  Passing numpy
    arrays directly (instead of 1.8M Python dicts) is what makes the full
    1.8M-row eval scoring tractable.

    ``shuffle_history`` (C-b control): randomly permute each row's history
    window before scoring, destroying real context ordering.

    Returns dict of arrays aligned to ``rows``:
      logp_bert_topk [N, K]  log p_BERT(c) on each candidate id
      bert_top1_id   [N]     argmax_c log p_BERT over all 128 codes
      bert_margin    [N]     top1 - top2 log p_BERT
      logp_bert_full [N, 128] the full log p_BERT vector (for f5/f6 + C-a)
    """
    n = len(rows)
    K = rows.topk_width() if hasattr(rows, "topk_width") else 0
    logp_full = np.full((n, VOCAB_BASE), np.nan, dtype=np.float32)
    logp_topk = np.full((n, max(K, 1)), np.nan, dtype=np.float32)
    top1_id = np.full(n, -1, dtype=np.int32)
    margin = np.full(n, np.nan, dtype=np.float32)

    dev = torch.device(device)
    idx = 0
    while idx < n:
        stop = min(idx + batch_size, n)
        built = []
        for j in range(idx, stop):
            shuf_seed = j if shuffle_history else None
            b = index.bert_input(rows.stock_uid(j), rows.date_key(j), window,
                                 shuffle_seed=shuf_seed,
                                 position=rows.position(j))
            built.append(b)
        valid = [b is not None for b in built]
        if not any(valid):
            idx = stop
            continue
        valid_sel = [b for b, ok in zip(built, valid) if ok]
        max_len = max(b[0].shape[0] for b in valid_sel)
        B = len(valid_sel)
        inp = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=dev)
        va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=dev)
        maskpos = torch.empty(B, dtype=torch.long, device=dev)
        for i, (ids, ti, v, _) in enumerate(valid_sel):
            L = ids.shape[0]
            inp[i, :L] = ids
            tids[i, :L] = ti
            va[i, :L] = v
            maskpos[i] = L - 1
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda"),
                                dtype=torch.bfloat16):
            logits = model(inp, tids, torch.arange(max_len, device=dev)
                           .unsqueeze(0).expand(B, -1), va_values=va)
        # logits: [B, max_len, 128]; read MASK positions
        mp = maskpos.view(B, 1, 1).expand(B, 1, logits.shape[-1])
        ml = torch.gather(logits, 1, mp).squeeze(1).float()     # [B, 128]
        lp = torch.log_softmax(ml, dim=-1).cpu().numpy()        # [B, 128]
        g = 0
        for j in range(idx, stop):
            if not valid[j - idx]:
                continue
            logp_full[j] = lp[g]
            top1_id[j] = int(np.argmax(lp[g]))
            slp = -np.sort(-lp[g])
            margin[j] = float(slp[0] - slp[1])
            tk = rows.topk_ids(j)
            if tk is not None and len(tk) > 0:
                logp_topk[j, :len(tk)] = lp[g][tk]
            g += 1
        idx = stop
    return {"logp_bert_topk": logp_topk, "bert_top1_id": top1_id,
            "bert_margin": margin, "logp_bert_full": logp_full}


class NumpyRowTable:
    """Lightweight row table over numpy arrays (no per-row Python objects)."""

    def __init__(self, stock_uid, date_key, position=None, topk_ids=None):
        self._u = stock_uid
        self._d = date_key
        self._p = position
        self._t = topk_ids
        self._n = len(stock_uid)

    def __len__(self):
        return self._n

    def stock_uid(self, i):
        return str(self._u[i])

    def date_key(self, i):
        return str(self._d[i])

    def position(self, i):
        return int(self._p[i]) if self._p is not None else None

    def topk_ids(self, i):
        if self._t is None:
            return None
        return self._t[i]

    def topk_width(self):
        return int(self._t.shape[1]) if self._t is not None else 0

    def select(self, keep_mask):
        """Return a filtered NumpyRowTable (boolean mask over rows)."""
        return NumpyRowTable(self._u[keep_mask], self._d[keep_mask],
                             position=self._p[keep_mask] if self._p is not None else None,
                             topk_ids=self._t[keep_mask] if self._t is not None else None)


# ============================================================================
# Region row tables
# ============================================================================

def _region_rows(region, candidates, calib_cache, training_cache, stride):
    """Build a NumpyRowTable for a region (subsampled by ``stride``)."""
    if region == "eval":
        c = np.load(candidates, allow_pickle=True)
        sl = slice(0, len(c["stock_uid"]), max(1, stride))
        return NumpyRowTable(c["stock_uid"][sl], c["date_key"][sl],
                             position=c["position"][sl],
                             topk_ids=c["topk_ids"][sl].astype(np.int64))
    elif region == "calib":
        c = np.load(calib_cache, allow_pickle=True)
        sl = slice(0, len(c["stock_uid"]), max(1, stride))
        # topk_ids come from the calib candidate cache (built by
        # build_gpt_candidates --region calib) so logp_bert_topk is defined.
        cand_calib = Path(str(candidates)).with_name(
            str(Path(candidates).name).replace("eval", "calib"))
        topk = None
        if cand_calib.exists():
            cc = np.load(cand_calib, allow_pickle=True)
            if len(cc["stock_uid"]) == len(c["stock_uid"]):
                topk = cc["topk_ids"][sl].astype(np.int64)
        return NumpyRowTable(c["stock_uid"][sl], c["date_key"][sl],
                             topk_ids=topk)
    elif region == "fit":
        c = np.load(training_cache, allow_pickle=True)
        sl = slice(0, len(c["stock_uid"]), max(1, stride))
        return NumpyRowTable(c["stock_uid"][sl], c["date_key"][sl])
    else:
        raise ValueError(f"unknown region {region}")


def _score_bert_main(region, stride=1, bert_path=None, window=512,
                     batch_size=32, device="cuda", max_stocks=0, suffix=""):
    """Shared BERT scoring logic for B-3 (eval) and B-4 (calib)."""
    roots = resolve_roots(seed=42)
    if bert_path is None:
        ckpt_path = model_checkpoint("BERT")
    else:
        ckpt_path = Path(bert_path)
    if not ckpt_path.exists():
        raise RuntimeError(f"BERT checkpoint missing: {ckpt_path}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    device = device if torch.cuda.is_available() else "cpu"

    # ---- load BERT ----
    model, cfg, ckpt_meta = load_bert(ckpt_path, torch.device(device))
    print(f"[score] BERT loaded: vocab={cfg.vocab_size} dim={cfg.dim} "
          f"depth={cfg.depth} heads={cfg.heads} mask_id={model.mask_id}")

    # ---- input index (cached across scripts) ----
    _, tok_path = upstream_paths()
    index_cache = weights_artifact("bert-index")
    index = build_index(tok_path, device="cpu", max_stocks=max_stocks,
                        cache_path=index_cache)
    print(f"[score] input index: {len(index.by_uid)} stocks")

    # ---- region rows ----
    candidates = weights_artifact("candidates-eval")
    validation_coverage = None
    if region == "eval" and stride == 1 and max_stocks == 0:
        eval_cache = np.load(candidates, allow_pickle=True)
        validation_coverage = require_full_validation_coverage(
            eval_cache["offset"], eval_cache["date_key"], label="BERT scoring"
        )
    pt06 = posttrain_artifacts(seed=42)
    calib_cache = pt06["calibration"]
    training_cache = pt06["training"]
    rows = _region_rows(region, candidates, calib_cache, training_cache, stride)
    keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
    rows = rows.select(keep)
    # sanity: for eval, the index-derived position must equal the candidate's
    if region == "eval" and rows:
        rng = np.random.RandomState(42)
        samp = rng.choice(len(rows), min(300, len(rows)), replace=False)
        mism = sum(1 for i in samp
                   if index.position_for(rows.stock_uid(int(i)), rows.date_key(int(i)))
                   != rows.position(int(i)))
        if mism:
            raise RuntimeError(
                f"eval position mismatch on {mism}/300 sampled rows; "
                "index and candidate cache disagree")
        print("[score] eval position alignment OK (300-row sample)")
    print(f"[score] region={region} rows to score: {len(rows)}")

    # ---- score ----
    out = score_rows(index, rows, model, window=window,
                     batch_size=batch_size, device=device)
    arr = {k: v for k, v in out.items()}
    arr["stock_uid"] = np.asarray(rows._u)
    arr["date_key"] = np.asarray(rows._d)

    sfx = f"_{suffix}" if suffix else ""
    out_npz = stage_weights("B") / f"scores-{region}-k8-w{window}-stride{stride}{sfx}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arr)
    print(f"[score] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    meta = {
        "schema": "scores-v1", "region": region, "stride": stride,
        "window": window, "n_rows": len(rows),
        "model": (model_variant(suffix or "BERT")["name"]
                  if (suffix or "BERT").upper() in
                  {v["name"].upper() for v in MODEL_VARIANTS}
                  else (suffix or "BERT")),
        "bert": {"checkpoint": str(ckpt_path), "sha256": file_sha256(ckpt_path),
                 "vocab_size": cfg.vocab_size, "mask_id": model.mask_id,
                 "val_loss": ckpt_meta.get("val_loss"),
                 "mlm_acc": ckpt_meta.get("mlm_acc"),
                 "epoch": ckpt_meta.get("epoch")},
        "validation_coverage": validation_coverage,
        "leakage": {"mask_va_zero": True, "mask_truncates_sequence": True},
    }
    out_json = stage_results("B") / f"scores-{region}-k8-w{window}-stride{stride}{sfx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    append_trial({"event": "score_bert", "region": region, "stride": stride,
                  "n_rows": len(rows), "status": "ok"})
    print(f"[score] sidecar -> {out_json}")


def _stage_b3():
    """B-3: Score BERT (eval region)."""
    ap = argparse.ArgumentParser(description="B-3: Score BERT (eval)")
    ap.add_argument("--bert", default=None,
                    help="BERT checkpoint (default A-train/BERT.pt)")
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth row (pilot)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_stocks", type=int, default=0,
                    help="limit the input index to the first N stocks (dev only)")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (for example scoring-aligned) so a selected "
                         "fine-tuned score cache has an explicit model name")
    args = ap.parse_args()
    _score_bert_main("eval", stride=args.stride, bert_path=args.bert,
                     window=args.window, batch_size=args.batch_size,
                     device=args.device, max_stocks=args.max_stocks,
                     suffix=args.suffix)


def _stage_b4():
    """B-4: Score BERT (calib region)."""
    ap = argparse.ArgumentParser(description="B-4: Score BERT (calib)")
    ap.add_argument("--bert", default=None,
                    help="BERT checkpoint (default A-train/BERT.pt)")
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth row (pilot)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_stocks", type=int, default=0,
                    help="limit the input index to the first N stocks (dev only)")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (for example scoring-aligned) so a selected "
                         "fine-tuned score cache has an explicit model name")
    args = ap.parse_args()
    _score_bert_main("calib", stride=args.stride, bert_path=args.bert,
                     window=args.window, batch_size=args.batch_size,
                     device=args.device, max_stocks=args.max_stocks,
                     suffix=args.suffix)


# ############################################################################
#  B-5 / B-6: Cache BERT hidden (from cache_bert_hidden.py)
# ############################################################################

@torch.no_grad()
def bert_hidden_rows(index, rows, model, *, dim, window=512, batch_size=32, device="cuda"):
    """MASK-position hidden [N, dim] for each row (same input as score_rows)."""
    model_cfg_dim = dim
    n = len(rows)
    hidden_out = np.full((n, model_cfg_dim), np.nan, dtype=np.float32)
    dev = torch.device(device)
    idx = 0
    while idx < n:
        stop = min(idx + batch_size, n)
        built = []
        for j in range(idx, stop):
            b = index.bert_input(rows.stock_uid(j), rows.date_key(j), window,
                                 position=rows.position(j))
            built.append(b)
        valid = [b is not None for b in built]
        if not any(valid):
            idx = stop
            continue
        valid_sel = [b for b, ok in zip(built, valid) if ok]
        max_len = max(b[0].shape[0] for b in valid_sel)
        B = len(valid_sel)
        inp = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=dev)
        va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=dev)
        maskpos = torch.empty(B, dtype=torch.long, device=dev)
        for i, (ids, ti, v, _) in enumerate(valid_sel):
            L = ids.shape[0]
            inp[i, :L] = ids
            tids[i, :L] = ti
            va[i, :L] = v
            maskpos[i] = L - 1
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda"),
                                dtype=torch.bfloat16):
            x = model._embed(inp, tids, va)
            sin, cos = model.rotary(torch.arange(max_len, device=dev)
                                    .unsqueeze(0).expand(B, -1))
            x = model._run_blocks(x, sin, cos, attn_mask=None)
            x = model.norm(x)                                     # [B, L, dim]
        mp = maskpos.view(B, 1, 1).expand(B, 1, x.shape[-1])
        h = torch.gather(x, 1, mp).squeeze(1).float().cpu().numpy()  # [B, dim]
        g = 0
        for j in range(idx, stop):
            if not valid[j - idx]:
                continue
            hidden_out[j] = h[g]
            g += 1
        idx = stop
        if stop % (batch_size * 5000) == 0 or stop == n:
            print(f"[hidden] rows {stop}/{n}", flush=True)
    return hidden_out


def _vectorized_dates_int(date_key):
    """'YYYY-MM-DD' (<U10) -> int32 YYYYMMDD, vectorized (no per-row Python)."""
    d = np.asarray(date_key).astype("<U10")
    digits = np.char.replace(d, "-", "")
    return digits.astype(np.int32)


def _precompute_positions(index, stock_uid, date_key):
    """Bulk per-row positions for a region (vectorized; avoids per-row searchsorted).

    Returns int64 array of positions, -1 where the (uid, date) is absent from
    the index.  O(#uids) searchsorted calls instead of #rows.
    """
    uids = np.asarray(stock_uid)
    dint = _vectorized_dates_int(date_key)
    n = len(uids)
    positions = np.full(n, -1, dtype=np.int64)
    uniq_uids, inv = np.unique(uids, return_inverse=True)
    for ui, uid in enumerate(uniq_uids):
        p = index.by_uid.get(str(uid))
        if p is None:
            continue
        ref = np.asarray(p["dates_int"], dtype=np.int32)
        rows_here = np.where(inv == ui)[0]
        didx = np.searchsorted(ref, dint[rows_here], side="left")
        ok = (didx < len(ref)) & (ref[didx] == dint[rows_here])
        positions[rows_here] = np.where(ok, didx, -1)
    return positions


def _fit_rows(stride=1, index=None):
    tc = np.load(posttrain_artifacts(seed=42)["training"], allow_pickle=True)
    sl = slice(0, len(tc["stock_uid"]), max(1, stride))
    u, d = tc["stock_uid"][sl], tc["date_key"][sl]
    pos = _precompute_positions(index, u, d) if index is not None else None
    return NumpyRowTable(u, d, position=pos)


def _eval_rows(stride=1):
    c = np.load(weights_artifact("candidates-eval"), allow_pickle=True)
    sl = slice(0, len(c["stock_uid"]), max(1, stride))
    return NumpyRowTable(c["stock_uid"][sl], c["date_key"][sl],
                         position=c["position"][sl])


def _cache_bert_hidden_main(region, bert_path=None, window=512, batch_size=32,
                            stride=1, max_rows=0, suffix=""):
    """Shared BERT hidden caching logic for B-5 (fit) and B-6 (eval)."""
    roots = resolve_roots(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(bert_path) if bert_path else model_checkpoint("BERT")
    model, cfg, ckpt = load_bert(ckpt_path, torch.device(device))
    print(f"[hidden] BERT dim={cfg.dim} depth={cfg.depth} on {device}")

    _, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=weights_artifact("bert-index"))

    rows = (_fit_rows(stride, index=index) if region == "fit"
            else _eval_rows(stride))
    if region == "eval" and stride == 1 and max_rows == 0:
        eval_cache = np.load(weights_artifact("candidates-eval"), allow_pickle=True)
        require_full_validation_coverage(
            eval_cache["offset"], eval_cache["date_key"], label="BERT hidden eval"
        )
    keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
    rows = rows.select(keep)
    if max_rows > 0:
        rows = rows.select(np.arange(min(max_rows, len(rows))))
    print(f"[hidden] region={region} rows={len(rows)}")

    h = bert_hidden_rows(index, rows, model, dim=cfg.dim, window=window,
                         batch_size=batch_size, device=device)

    sfx = f"_{suffix}" if suffix else ""
    out_npz = stage_weights("B") / f"hidden-{region}-w{window}{sfx}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, stock_uid=np.asarray(rows._u), date_key=np.asarray(rows._d),
             hidden=h, window=np.array([window]),
             bert_ckpt=str(ckpt_path))
    fp = row_set_fingerprint(np.asarray(rows._u), np.asarray(rows._d), str(ckpt_path),
                             out_json=stage_results("B") /
                             f"hidden-{region}-w{window}{sfx}.fingerprint.json")
    print(f"[hidden] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    print(f"[hidden] fingerprint: rows={fp['n_rows']} uids={fp['n_unique_uids']} "
          f"dates={fp['n_unique_dates']} ckpt_sha={fp['bert_ckpt_sha256'][:10]}")
    append_trial({"event": "cache_bert_hidden", "region": region,
                  "n_rows": len(rows), "status": "ok"})


def _stage_b5():
    """B-5: Cache BERT hidden (fit region)."""
    ap = argparse.ArgumentParser(description="B-5: Cache BERT hidden (fit)")
    ap.add_argument("--bert", default=str(model_checkpoint("BERT")))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="subsample (dev smoke)")
    ap.add_argument("--max_rows", type=int, default=0, help="dev smoke limit")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (for example scoring-aligned) so a selected "
                         "fine-tuned hidden cache has an explicit model name")
    args = ap.parse_args()
    _cache_bert_hidden_main("fit", bert_path=args.bert, window=args.window,
                            batch_size=args.batch_size, stride=args.stride,
                            max_rows=args.max_rows, suffix=args.suffix)


def _stage_b6():
    """B-6: Cache BERT hidden (eval region)."""
    ap = argparse.ArgumentParser(description="B-6: Cache BERT hidden (eval)")
    ap.add_argument("--bert", default=str(model_checkpoint("BERT")))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="subsample (dev smoke)")
    ap.add_argument("--max_rows", type=int, default=0, help="dev smoke limit")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (for example scoring-aligned) so a selected "
                         "fine-tuned hidden cache has an explicit model name")
    args = ap.parse_args()
    _cache_bert_hidden_main("eval", bert_path=args.bert, window=args.window,
                            batch_size=args.batch_size, stride=args.stride,
                            max_rows=args.max_rows, suffix=args.suffix)


# ############################################################################
#  B-7: Train MLP rank head (from train_bert_head.py)
# ############################################################################

# fixed recipe candidate grid (06 P6-style; R0-R2 roll selects one)
HP_GRID = [
    {"lr": 1e-3, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 8, "dropout": 0.0},
    {"lr": 3e-4, "epochs": 16, "dropout": 0.1},
    {"lr": 1e-3, "epochs": 12, "dropout": 0.1},
]


def _rows_from_hidden(hidden_path):
    d = np.load(hidden_path, allow_pickle=True)
    rows = {k: d[k] for k in d.files}
    # the fit hidden cache is built from the 06 training_cache rows in order;
    # join the target / normalization fields the head trainer needs.
    tc = np.load(posttrain_artifacts(seed=42)["training"], allow_pickle=True)
    if len(tc["stock_uid"]) == len(rows["stock_uid"]):
        for k in ("true_logret", "p_mean0", "p_std0", "quality"):
            if k in tc.files:
                rows[k] = tc[k]
    else:
        raise RuntimeError(
            f"fit hidden cache rows ({len(rows['stock_uid'])}) != training_cache "
            f"({len(tc['stock_uid'])}) -- cannot join true_logret")
    return rows


def _select_recipe(rows, loss_kind="soft_spearman", seed=42):
    """R0-R2 rolling: pick the hp minimizing mean fold val rank loss."""
    results = []
    for hp in HP_GRID:
        fold_losses = []
        for fold in ("R0", "R1", "R2"):
            fit_idx, val_idx = fold_split(rows, fold)
            head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                               dropout=hp["dropout"], loss=loss_kind)
            _, hist = train_rank_per_date(head, rows, fit_idx, val_idx,
                                          loss_kind, lr=hp["lr"],
                                          epochs=hp["epochs"], seed=seed)
            fold_losses.append(hist["val_loss"][-1])
        results.append({"hp": hp, "mean_fold_val_loss": float(np.mean(fold_losses)),
                        "fold_val_loss": dict(zip(("R0", "R1", "R2"), fold_losses))})
        print(f"[rank-head] hp={hp} mean_fold_val_loss={np.mean(fold_losses):.4f}", flush=True)
    results.sort(key=lambda r: r["mean_fold_val_loss"])
    return results[0]


def _stage_b7():
    """B-7: Train MLP rank head."""
    ap = argparse.ArgumentParser(description="B-7: Train MLP rank head")
    ap.add_argument("--hidden_fit", default=None)
    ap.add_argument("--suffix", type=str, default="",
                    help="hidden-cache suffix (for example scoring-aligned) matching "
                         "the selected BERT model")
    ap.add_argument("--model", type=str, default="BERT",
                    help="formal model name: BERT, BERT-FT, or BERT-PPS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply_eval", action="store_true",
                    help="apply to eval hidden cache if present")
    args = ap.parse_args()

    wr = stage_weights("B", seed=args.seed)
    if args.hidden_fit:
        hidden_fit = Path(args.hidden_fit)
    else:
        sfx = f"_{args.suffix}" if args.suffix else ""
        hidden_fit = wr / f"hidden-fit-w512{sfx}.npz"
    if not hidden_fit.exists():
        raise RuntimeError(f"fit hidden cache missing: {hidden_fit} "
                           "(run cache_bert_hidden.py --region fit first)")
    rows = _rows_from_hidden(hidden_fit)
    print(f"[rank-head] fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")

    best = _select_recipe(rows, seed=args.seed)
    print(f"[rank-head] selected recipe: {best['hp']}")

    # final fit on all pre-cutoff fit rows (no early stop)
    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                       dropout=best["hp"]["dropout"], loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=best["hp"]["lr"], epochs=best["hp"]["epochs"],
                                  seed=args.seed)
    model_name = model_variant(args.model)["name"]
    out_head = weights_artifact("bert-head", seed=args.seed, model=model_name)
    torch.save({"head_state": head.state_dict(),
                "recipe": best["hp"], "seed": args.seed,
                "fold_selection": best,
                "train_history": hist,
                "hidden_cache": str(hidden_fit),
                "schema": "rank-head-v1"}, out_head)
    print(f"[rank-head] saved {out_head}")

    result = {"schema": "rank-head-v1", "seed": args.seed,
              "recipe": best["hp"], "fold_selection": best,
              "final_train_loss": hist["train_loss"][-1],
              "n_fit_rows": int(len(fit_idx))}
    # ---- eval apply (offsets 0..399), if eval hidden cache present ----
    eval_hidden = wr / f"hidden-eval-w512{('_' + args.suffix) if args.suffix else ''}.npz"
    if args.apply_eval and eval_hidden.exists():
        cand = np.load(cand_path("eval"), allow_pickle=True)
        de = np.load(eval_hidden, allow_pickle=True)
        n = len(cand["stock_uid"])
        if (len(de["stock_uid"]) != n
                or not np.array_equal(de["stock_uid"], cand["stock_uid"])
                or not np.array_equal(de["date_key"], cand["date_key"])):
            raise RuntimeError("eval hidden cache stock/date rows are not aligned with candidates")
        H = torch.from_numpy(de["hidden"].astype(np.float32))
        head.eval()
        with torch.no_grad():
            score = head(H).numpy()
        rec = build_rec(cand)
        rec["bert_head"] = score.astype(np.float64)
        p6 = np.load(weights_artifact("p6-scores", seed=args.seed))
        if len(p6) != n:
            raise RuntimeError(f"P6 len {len(p6)} != candidates {n}")
        rec["p6_score"] = p6.astype(np.float64)
        dense = int(cand["dense_threshold"][0])
        result["eval_metrics"] = metrics_table(rec, {
            "BERT_head": "bert_head", "J3_median": "post_median", "P6": "p6_score"},
            dense)
        result["bootstrap_vs_J3"] = bootstrap_vs(rec, "bert_head", rec, "post_median", dense)
        result["bootstrap_vs_P6"] = bootstrap_vs(rec, "bert_head", rec, "p6_score", dense)
        print(json.dumps({"eval_metrics": result["eval_metrics"],
                          "vs_J3": result["bootstrap_vs_J3"],
                          "vs_P6": result["bootstrap_vs_P6"]}, indent=1, default=str))
    json_path = (stage_results("B", seed=args.seed) /
                 f"rank-head-training-{model_name}-seed{args.seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    append_trial({"event": "rank_head_train", "seed": args.seed, "status": "ok"})
    print(f"[rank-head] result -> {json_path}")


# ############################################################################
#  B-8: Apply head to eval (from apply_bert_head.py)
# ############################################################################

def _stage_b8():
    """Apply the trained BERT rank head to the eval hidden cache."""
    ap = argparse.ArgumentParser(description="Apply the trained BERT rank head to eval")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--model", type=str, default="BERT",
                    help="formal model name: BERT, BERT-FT, or BERT-PPS")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wr = stage_weights("B", seed=args.seed)

    model_name = model_variant(args.model)["name"]
    head_path = weights_artifact("bert-head", seed=args.seed, model=model_name)
    if not head_path.exists():
        raise RuntimeError(f"head missing: {head_path}")
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    print(f"[apply] head <- {head_path}")

    sfx = f"_{args.suffix}" if args.suffix else ""
    eval_hidden = wr / f"hidden-eval-w512{sfx}.npz"
    if not eval_hidden.exists():
        raise RuntimeError(f"eval hidden cache missing: {eval_hidden}")
    de = np.load(eval_hidden, allow_pickle=True)
    cand = np.load(weights_artifact("candidates-eval", seed=args.seed), allow_pickle=True)
    n = len(cand["stock_uid"])
    if (len(de["stock_uid"]) != n
            or not np.array_equal(de["stock_uid"], cand["stock_uid"])
            or not np.array_equal(de["date_key"], cand["date_key"])):
        raise RuntimeError("eval hidden cache stock/date rows are not aligned with candidates")
    print(f"[apply] eval hidden rows {n}")

    H = torch.from_numpy(de["hidden"].astype(np.float32))
    with torch.no_grad():
        score = head(H).numpy()
    rank_score_path = weights_artifact(
        "rank-head-scores", seed=args.seed, model=model_name
    )
    np.savez(rank_score_path, stock_uid=cand["stock_uid"],
             date_key=cand["date_key"], rank_head_score=score.astype(np.float32),
             model=np.array([model_name]))
    rec = build_rec(cand)
    rec["bert_head"] = score.astype(np.float64)
    p6 = np.load(weights_artifact("p6-scores", seed=args.seed))
    if len(p6) != n:
        raise RuntimeError(f"P6 len {len(p6)} != candidates {n}")
    rec["p6_score"] = p6.astype(np.float64)
    dense = int(cand["dense_threshold"][0])

    res = {
        "schema": "rank-head-eval-v1", "seed": args.seed,
        "model": model_name, "suffix": args.suffix,
        "head": str(head_path), "recipe": ck.get("recipe"),
        "eval_metrics": metrics_table(rec, {
            "BERT_head": "bert_head", "J3_median": "post_median",
            "J2_mean": "post_mean", "J4_pup": "p_up", "P6": "p6_score"},
            dense),
        "bootstrap_vs_J3": bootstrap_vs(rec, "bert_head", rec, "post_median", dense),
        "bootstrap_vs_P6": bootstrap_vs(rec, "bert_head", rec, "p6_score", dense),
    }
    out = (stage_results("B", seed=args.seed) /
           f"rank-head-evaluation-{model_name}-seed{args.seed}{sfx}.json")
    write_json_ledger(out, res, "rank_head_apply", seed=args.seed)
    print(json.dumps({
        "BERT_head": res["eval_metrics"]["BERT_head"],
        "vs_J3": res["bootstrap_vs_J3"],
        "vs_P6": res["bootstrap_vs_P6"],
    }, indent=1, default=str))
    print(f"[apply] -> {out}")


# ############################################################################
#  F-INT helpers (inlined from fuse_scores.py for B-9)
# ############################################################################

LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _fint_scores(cand, sc, lam):
    logq = cand["topk_logq"]
    lp = sc["logp_bert_topk"]
    if lp.shape[1] < logq.shape[1]:
        logq = logq[:, :lp.shape[1]]
    s = (1.0 - lam) * logq + lam * lp
    return np.nanmax(s, axis=1)


def _select_lambda_calibration(calib_cand, calib_sc, dense_threshold=None):
    if dense_threshold is None:
        by_date = {}
        for d in calib_cand["date_key"]:
            by_date[str(d)] = by_date.get(str(d), 0) + 1
        dense_threshold = max(5, int(np.ceil(0.8 * max(by_date.values()))))
    rec = {"date_key": calib_cand["date_key"],
           "true_logret": calib_cand["true_logret"].astype(np.float64),
           "quality": calib_cand["quality"].astype(bool)}
    best, best_ic = None, -1e9
    scores = {}
    for lam in LAMBDA_GRID:
        rec["fint"] = _fint_scores(calib_cand, calib_sc, lam)
        m = arm_metrics(rec, "fint", dense_threshold)
        ic = m["avg_daily_rank_ic"]
        scores[str(lam)] = {"lambda": lam, "calib_avg_daily_rank_ic": ic}
        if ic is not None and ic > best_ic:
            best_ic, best = ic, lam
    return best, best_ic, scores


def _run_fint(*, candidates, scores_eval, out_json=None,
              out_scores_npz=None, calib_candidates=None, calib_scores=None):
    cand = np.load(candidates, allow_pickle=True)
    sc = np.load(scores_eval, allow_pickle=True)
    if len(sc["stock_uid"]) != len(cand["stock_uid"]):
        raise RuntimeError("F-INT: scores_eval misaligned with candidates")
    dense_threshold = int(cand["dense_threshold"][0])
    if calib_candidates is not None and calib_scores is not None:
        cc = np.load(calib_candidates, allow_pickle=True)
        cs = np.load(calib_scores, allow_pickle=True)
        best, best_ic, lam_scores = _select_lambda_calibration(cc, cs, dense_threshold=None)
        fitted_on = "calibration_slice"
    else:
        raise RuntimeError("F-INT requires calibration-slice lambda selection")
    rec = {"date_key": cand["date_key"], "stock_uid": cand["stock_uid"],
           "true_logret": cand["true_logret"].astype(np.float64),
           "quality": cand["quality"].astype(bool),
           "offset": cand["offset"].astype(np.int64)}
    rec["fint"] = _fint_scores(cand, sc, best)
    full = arm_metrics(rec, "fint", dense_threshold)
    result = {"schema": "fint-v1", "chosen_lambda": best,
              "chosen_lambda_calib_rank_ic": best_ic, "lambda_scores": lam_scores,
              "chosen_lambda_full_400": {"avg_daily_rank_ic": full["avg_daily_rank_ic"],
                                         "avg_da_per_date": full["avg_da_per_date"]},
              "n_rows": len(cand["stock_uid"]), "fitted_on": fitted_on}
    if out_json:
        write_json(out_json, result)
    if out_scores_npz is not None:
        out_scores_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_scores_npz, stock_uid=cand["stock_uid"],
                 date_key=cand["date_key"], fint_score=rec["fint"],
                 chosen_lambda=np.array([best]))
    return result, rec, best


# ############################################################################
#  B-9: Evaluate controls + arms (from eval_critic.py)
# ############################################################################

def require_controls(controls_path) -> None:
    """Refuse to produce B1-B5 summaries without the C-a/C-b artifact."""
    if not Path(controls_path).exists():
        raise RuntimeError(
            "C-a/C-b controls must run and write their artifact before B1-B5 "
            f"(missing {controls_path})")


# ============================================================================
# C-a / C-b controls
# ============================================================================

def _sample_fit_rows(training_cache, n_target=50_000):
    """Sample n_target (uid, date) rows from the fit-region cache by stride.

    Returns (NumpyRowTable, cache_npz, idx, true_coarse_ids).
    """
    c = np.load(training_cache, allow_pickle=True)
    n = len(c["stock_uid"])
    stride = max(1, n // n_target)
    idx = np.arange(0, n, stride)
    table = NumpyRowTable(c["stock_uid"][idx], c["date_key"][idx])
    true_coarse = c["true_coarse_id"][idx].astype(np.int64)
    return table, c, idx, true_coarse


@torch.no_grad()
def _gpt_q_for_rows(model, hidden_cache, idx):
    """q(c) at T_c=1.4 for the sampled training-cache rows."""
    h = torch.from_numpy(hidden_cache["hidden"][idx]).float()
    h = h.to(next(model.parameters()).device)
    q, sm = coarse_q_for_hidden(model, h, t_c=1.4,
                                vocab_base=model._vocab_l1)
    return q.cpu().numpy(), sm.cpu().numpy()


def _sample_distractors(q_row, true_id, n_dist=7, seed=0):
    """Sample n_dist distractors from q (excluding the true token)."""
    rng = np.random.RandomState(seed)
    probs = q_row.copy()
    probs[true_id] = 0.0
    probs /= probs.sum()
    candidates = np.arange(VOCAB_BASE, dtype=np.int64)
    d = rng.choice(candidates, size=n_dist, p=probs, replace=False)
    return d


def _true_rank(logp_row, true_id, distractor_ids):
    """Rank of the true token among {true} union distractors (1 = best)."""
    scores = np.concatenate([[logp_row[true_id]], logp_row[distractor_ids]])
    # lower rank value = higher score (rank 1 = highest score)
    order = np.argsort(-scores)
    return int(np.where(order == 0)[0][0]) + 1


def run_controls(*, bert_path, index, rows, gpt_model, hidden_cache,
                 true_coarse, cache_idx, n_shuffle=10_000, window=512,
                 batch_size=32, device="cuda"):
    """Run C-a (true-token sorting) and C-b (shuffled-history) controls.

    ``rows`` is a NumpyRowTable sampled from the fit region; ``true_coarse`` and
    ``cache_idx`` give the aligned true coarse id and the training-cache row
    index per sampled row (used to pull hidden states for GPT q(c)).  Returns
    the controls dict.
    """
    model, cfg, _ = load_bert(bert_path, torch.device(device))
    n = len(rows)
    # ---- C-a: score the true next-day slot ----
    scores_a = score_rows(index, rows, model, window=window,
                          batch_size=batch_size, device=device)
    logp = scores_a["logp_bert_full"]                       # [N, 128]
    q, _ = _gpt_q_for_rows(gpt_model, hidden_cache, cache_idx)

    ranks = np.empty(n, dtype=np.float64)
    hits = np.empty(n, dtype=bool)
    auc_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = _sample_distractors(q[i], int(true_coarse[i]), n_dist=7, seed=i)
        rank = _true_rank(logp[i], int(true_coarse[i]), d)
        ranks[i] = rank
        hits[i] = (rank == 1)
        auc_vals[i] = np.mean(logp[i, int(true_coarse[i])] > logp[i, d])
    c_a = {
        "n_rows": n,
        "avg_true_rank": float(ranks.mean()),
        "uniform_rank_expectation": 4.5,           # (8+1)/2
        "top1_hit_rate": float(hits.mean()),
        "uniform_top1_expectation": 1.0 / 8.0,
        "auc_true_vs_distractors": float(auc_vals.mean()),
        "pass": bool(ranks.mean() < 4.5 - 0.5) and bool(hits.mean() > 1.0 / 8.0 + 0.02),
    }

    # ---- C-b: shuffle history order -> discrimination should collapse ----
    m = min(n_shuffle, n)
    shuf_rows = NumpyRowTable(rows._u[:m], rows._d[:m])
    shuf_scores = score_rows(index, shuf_rows, model, window=window,
                             batch_size=batch_size, device=device,
                             shuffle_history=True)
    logp_sh = shuf_scores["logp_bert_full"]
    q_sh, _ = _gpt_q_for_rows(gpt_model, hidden_cache, cache_idx[:m])
    ranks_sh = np.empty(m, dtype=np.float64)
    hits_sh = np.empty(m, dtype=bool)
    for i in range(m):
        d = _sample_distractors(q_sh[i], int(true_coarse[i]), n_dist=7, seed=i)
        ranks_sh[i] = _true_rank(logp_sh[i], int(true_coarse[i]), d)
        hits_sh[i] = (ranks_sh[i] == 1)
    c_b = {
        "n_rows": m,
        "shuffled_avg_true_rank": float(ranks_sh.mean()),
        "shuffled_top1_hit_rate": float(hits_sh.mean()),
        "collapsed_toward_uniform": bool(ranks_sh.mean() >= 4.5 - 0.3),
    }
    return {"c_a": c_a, "c_b": c_b, "config": {"window": window,
                                               "tc_candidates": 1.4,
                                               "n_distractors": 7}}


# ============================================================================
# B0-B5 arm evaluation
# ============================================================================

def load_p6_scores(head_path, hidden):
    """Evaluate the 06 P6 MLP rank head on hidden [N, dim] -> [N] scores."""
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    sd = ck["head_state"]
    # MlpRankHead(dim=256, hidden=64, dropout=0.0) -> Linear(0)/SiLU(1)/Identity(2)/Linear(3)
    net = torch.nn.Sequential(
        torch.nn.Linear(256, 64), torch.nn.SiLU(), torch.nn.Identity(),
        torch.nn.Linear(64, 1))
    net.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
    net.eval()
    with torch.no_grad():
        s = net(torch.from_numpy(hidden).float()).squeeze(-1).numpy()
    return s


def _dev_confirm(offsets):
    """Split offsets into dev (0..299) and confirm (300..399)."""
    dev = [o for o in offsets if o <= 299]
    conf = [o for o in offsets if 300 <= o < 400]
    return dev, conf


def bootstrap_vs_reference(cand_rec, ref_rec, score_field, dense_min=3634,
                           block_lengths=(5, 10, 20), n_replicates=10_000):
    """Paired moving-block bootstrap of candidate vs reference daily RankIC.

    Both recs must share the same (date, stock_uid) universe and finite mask.
    ``cand_rec[score_field]`` is the candidate's per-row rank score;
    ``ref_rec`` uses its own ``p6_score`` as the reference.  Returns the
    compare_posttrain result dict (point + L=5/10/20 CIs).
    """

    def to_rows(rec, field):
        rows = []
        valid = np.isfinite(rec[field]) & np.isfinite(rec["true_logret"]) & rec["quality"]
        for i in np.where(valid)[0]:
            rows.append({"date_key": str(rec["date_key"][i]),
                         "stock_uid": str(rec["stock_uid"][i]),
                         "rank_score": float(rec[field][i]),
                         "true_logret": float(rec["true_logret"][i])})
        return rows

    cand_rows = to_rows(cand_rec, score_field)
    ref_rows = to_rows(ref_rec, "p6_score")
    return paired_bootstrap_ci(cand_rows, ref_rows, "rank_ic",
                               dense_min=dense_min,
                               block_lengths=block_lengths,
                               n_replicates=n_replicates)


def evaluate_arms(*, candidates, scores_eval, p6_path, dense_threshold,
                  out_json=None, score_fields=None, fused_scores_npz=None):
    """Build B0/B1/B2 arm records and compute metrics + paired bootstrap.

    ``score_fields``: dict arm_name -> dict(score_field, source) where source
    is one of ``p6``, ``post_median``, ``bert_max``, ``fint`` (needs fused_scores).
    """
    cand = np.load(candidates, allow_pickle=True)
    sc = np.load(scores_eval, allow_pickle=True) if scores_eval else None
    n = len(cand["stock_uid"])

    rec = {
        "date_key": cand["date_key"], "stock_uid": cand["stock_uid"],
        "true_logret": cand["true_logret"].astype(np.float64),
        "quality": cand["quality"].astype(bool),
        "offset": cand["offset"].astype(np.int64) if "offset" in cand.files else None,
        "true_coarse_id": cand["true_coarse_id"].astype(np.int16),
        "gpt_top1_id": cand["topk_ids"][:, 0].astype(np.int16),
        "post_median": cand["post_median"].astype(np.float64),
        "p_up": cand["p_up"].astype(np.float64),
        "post_std": cand["post_std"].astype(np.float64),
        "topk_logq": cand["topk_logq"], "topk_ids": cand["topk_ids"],
    }
    if sc is not None:
        # row alignment of scores_eval vs candidates (same region/order)
        if len(sc["stock_uid"]) != n:
            raise RuntimeError(
                f"scores_eval rows {len(sc['stock_uid'])} != candidates rows {n}")
        if (not np.array_equal(sc["stock_uid"], cand["stock_uid"]) or
                not np.array_equal(sc["date_key"], cand["date_key"])):
            raise RuntimeError("scores_eval stock/date rows are not aligned with candidates")
        rec["logp_bert_topk"] = sc["logp_bert_topk"]
        rec["logp_bert_full"] = sc["logp_bert_full"]
        rec["bert_top1_id"] = sc["bert_top1_id"].astype(np.int16)
        rec["bert_margin"] = sc["bert_margin"]

    # P6 (B0 primary)
    p6 = load_p6_scores(
        p6_path,
        np.load(posttrain_artifacts(seed=42)["hidden"], allow_pickle=True)["hidden"])
    if len(p6) != n:
        raise RuntimeError(f"P6 scores {len(p6)} != candidates rows {n}")
    rec["p6_score"] = p6

    # B1: max over candidates of log p_BERT
    rec["bert_score"] = np.nanmax(rec["logp_bert_topk"], axis=1) if sc is not None else None
    # B2 (F-INT) fused scores injected via score_fields from fuse_scores.py
    if fused_scores_npz is not None and Path(fused_scores_npz).exists():
        fs = np.load(fused_scores_npz, allow_pickle=True)
        if len(fs["stock_uid"]) != n:
            raise RuntimeError("fused F-INT scores misaligned with candidates")
        rec["fint_score"] = fs["fint_score"].astype(np.float64)
    arms = {}
    for name, fld in (score_fields or {}).items():
        if fld in rec:
            arms[name] = arm_metrics(rec, fld, dense_threshold)

    result = {"schema": "critic-eval-v1", "n_rows": n,
              "dense_threshold": dense_threshold, "arms": arms,
              "offsets_scope": "0_399", "holdout_used": False}
    if out_json:
        write_json(out_json, result)
    return result, rec


def _stage_b9():
    """B-9: Evaluate controls + arms."""
    ap = argparse.ArgumentParser(description="B-9: BERT critic controls + arm evaluation")
    ap.add_argument("--mode", choices=["controls", "arms", "all"], default="all")
    ap.add_argument("--n_controls", type=int, default=50_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", type=str, default="BERT",
                    help="formal model name: BERT, BERT-FT, or BERT-PPS")
    ap.add_argument("--suffix", type=str, default="",
                    help="score-cache suffix; defaults to the formal model name")
    ap.add_argument("--bert", default=None,
                    help="BERT checkpoint for controls (default base MLM; use the "
                         "same selected model variant as the score cache)")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    model_name = model_variant(args.model)["name"]
    score_suffix = args.suffix or model_name
    controls_path = stage_results("B") / f"controls-{model_name}.json"
    pt06 = posttrain_artifacts(seed=42)
    training_cache = pt06["training"]
    candidates = weights_artifact("candidates-eval")
    scores_eval = scores_path("eval", suffix=score_suffix)
    p6_path = pt06["head_rank_mlp_spearman"]
    bert_path = Path(args.bert) if args.bert else model_checkpoint(model_name)

    if args.mode in ("controls", "all"):
        if not bert_path.exists():
            raise RuntimeError(f"Base BERT checkpoint missing: {bert_path}")
        ckpt, tok_path = upstream_paths()
        index_cache = weights_artifact("bert-index")
        index = build_index(tok_path, device="cpu", cache_path=index_cache)
        gpt_model, _ = _load_model_cpu(ckpt, tok_path)
        rows, cache, cache_idx, true_coarse = _sample_fit_rows(
            training_cache, args.n_controls)
        # filter to stocks present in the index (uid axis)
        keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
        rows = NumpyRowTable(rows._u[keep], rows._d[keep])
        cache_idx = cache_idx[keep]
        true_coarse = true_coarse[keep]
        device = args.device if torch.cuda.is_available() else "cpu"
        gpt_model = gpt_model.to(device)
        controls = run_controls(bert_path=bert_path, index=index, rows=rows,
                                gpt_model=gpt_model, hidden_cache=cache,
                                true_coarse=true_coarse, cache_idx=cache_idx,
                                device=device)
        write_json(controls_path, controls)
        append_trial({"event": "controls", "n": len(rows), "status": "ok",
                      "c_a_pass": controls["c_a"]["pass"]})
        print(json.dumps(controls, indent=2, ensure_ascii=False))

    if args.mode in ("arms", "all"):
        require_controls(controls_path)
        if not scores_eval.exists():
            raise RuntimeError(f"eval BERT scores missing: {scores_eval}")
        cand = np.load(candidates, allow_pickle=True)
        # F-INT fused scores -- lambda fit on this model's calibration scores.
        fused_npz = weights_artifact("fint-scores", model=model_name)
        calib_cand = weights_artifact("candidates-calib")
        calib_sc = scores_path("calib", suffix=score_suffix)
        _run_fint(candidates=candidates, scores_eval=scores_eval,
                  out_json=stage_results("B") / f"fint-{model_name}.json",
                  out_scores_npz=fused_npz,
                  calib_candidates=calib_cand if calib_cand.exists() else None,
                  calib_scores=calib_sc if calib_sc.exists() else None)
        result, rec = evaluate_arms(
            candidates=candidates, scores_eval=scores_eval, p6_path=p6_path,
            dense_threshold=int(cand["dense_threshold"][0]),
            out_json=stage_results("B") / f"arms-{model_name}.json",
            score_fields={"exp06_rank_head": "p6_score",
                          "gpt_posterior_median": "post_median",
                          "bert_critic": "bert_score",
                          "score_interpolation": "fint_score"},
            fused_scores_npz=fused_npz)
        rank_score_path = weights_artifact(
            "rank-head-scores", seed=42, model=model_name
        )
        if rank_score_path.exists():
            rank_scores = np.load(rank_score_path, allow_pickle=True)
            if (len(rank_scores["rank_head_score"]) != len(rec["stock_uid"])
                    or not np.array_equal(rank_scores["stock_uid"], rec["stock_uid"])
                    or not np.array_equal(rank_scores["date_key"], rec["date_key"])):
                raise RuntimeError(
                    f"rank-head score rows are not aligned with {model_name} evaluation rows"
                )
            rec["rank_head_score"] = rank_scores["rank_head_score"].astype(np.float64)
            result["arms"]["bert_rank_head"] = arm_metrics(
                rec, "rank_head_score", int(cand["dense_threshold"][0])
            )
        # paired moving-block bootstrap of each candidate arm vs B0 (P6)
        comparisons = {}
        for name, field in (("score_interpolation", "fint_score"),
                            ("bert_critic", "bert_score"),
                            ("bert_rank_head", "rank_head_score")):
            if field in rec:
                comparisons[name] = bootstrap_vs_reference(
                    rec, rec, field, dense_min=int(cand["dense_threshold"][0]))
        result["bootstrap_vs_exp06_rank_head"] = comparisons
        result["model"] = model_name
        result["score_suffix"] = score_suffix
        result["validation_coverage"] = require_full_validation_coverage(
            cand["offset"], cand["date_key"], label=f"{model_name} evaluation"
        )
        write_json(stage_results("B") / f"arms-{model_name}.json", result)
        print(json.dumps({"arms": {k: v for k, v in result["arms"].items()},
                          "bootstrap_vs_exp06_rank_head": comparisons},
                         indent=2, ensure_ascii=False))
        return result, rec
    return None, None


def _stage_b10():
    """B-10: Recompute full 128-code GPT q for the optional R2 arm."""
    ap = argparse.ArgumentParser(description="B-10: Recompute full GPT q")
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hidden_path = Path(args.hidden) if args.hidden else posttrain_artifacts()["hidden"]
    out_path = gpt_q_path("eval")
    if out_path.exists():
        print(f"[gpt-q] exists, skipping: {out_path}")
        return
    ckpt, tok_path = upstream_paths()
    model, _ = load_model_cpu(ckpt, tok_path)
    model = model.to(dev).eval()
    hdata = np.load(hidden_path, allow_pickle=True)
    hidden = hdata["hidden"]
    q = np.zeros((len(hidden), int(model._vocab_l1)), dtype=np.float32)
    for start in range(0, len(hidden), args.chunk):
        stop = min(start + args.chunk, len(hidden))
        qc, _ = coarse_q_for_hidden(
            model, torch.from_numpy(hidden[start:stop]).to(dev),
            t_c=TC, vocab_base=int(model._vocab_l1))
        q[start:stop] = qc.cpu().numpy()
        if stop == len(hidden) or stop % (args.chunk * 20) == 0:
            print(f"[gpt-q] rows {stop}/{len(hidden)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, stock_uid=hdata["stock_uid"], date_key=hdata["date_key"],
             q=q, tc=np.array([TC]))
    print(f"[gpt-q] wrote {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")


# ############################################################################
#  Complete B pipeline
# ############################################################################

FULL_PREDICTION_FIELDS = (
    "offset", "date_key", "stock_uid", "true_logret", "quality",
    "true_coarse_id", "gpt_top1_id", "bert_top1_id",
    "post_median", "p_up", "post_std", "p6_score", "bert_score",
    "bert_margin", "rank_head_score", "fint_score",
)


def _stage_ball():
    """Run every formal BERT variant through the complete B pipeline."""
    shared_steps = [
        ("GPT candidates — eval region", _stage_b1, ()),
        ("GPT candidates — calibration region", _stage_b2, ()),
    ]
    for label, fn, args in shared_steps:
        print("=" * 70)
        print(f"[B] {label}")
        print("=" * 70)
        sys.argv = [sys.argv[0], *args]
        fn()

    prediction_path = results_root(seed=42) / "bert_critic_predictions.parquet"
    summary = {
        "schema": "bert-critic-model-evaluation-v1",
        "models": [],
        "prediction_records": str(prediction_path),
        "prediction_fields": ["model", *FULL_PREDICTION_FIELDS],
        "holdout_used": False,
    }
    with PredictionParquetWriter(
        prediction_path, FULL_PREDICTION_FIELDS, include_model=True
    ) as prediction_writer:
        for variant in MODEL_VARIANTS:
            model_name = variant["name"]
            suffix = variant["suffix"]
            ckpt = model_checkpoint(model_name)
            model_steps = [
                (f"{model_name} — BERT scoring — eval", _stage_b3,
                 ("--bert", str(ckpt), "--suffix", suffix)),
                (f"{model_name} — BERT scoring — calibration", _stage_b4,
                 ("--bert", str(ckpt), "--suffix", suffix)),
                (f"{model_name} — hidden cache — fit", _stage_b5,
                 ("--bert", str(ckpt), "--suffix", suffix)),
                (f"{model_name} — hidden cache — eval", _stage_b6,
                 ("--bert", str(ckpt), "--suffix", suffix)),
                (f"{model_name} — rank-head training", _stage_b7,
                 ("--model", model_name, "--suffix", suffix, "--apply_eval")),
                (f"{model_name} — rank-head application", _stage_b8,
                 ("--model", model_name, "--suffix", suffix)),
            ]
            for label, fn, args in model_steps:
                print("=" * 70)
                print(f"[B] {label}")
                print("=" * 70)
                sys.argv = [sys.argv[0], *args]
                fn()

            label = f"{model_name} — controls and arm evaluation"
            print("=" * 70)
            print(f"[B] {label}")
            print("=" * 70)
            sys.argv = [sys.argv[0], "--mode", "all", "--model", model_name,
                        "--suffix", suffix, "--bert", str(ckpt)]
            model_result, model_records = _stage_b9()
            if model_result is None or model_records is None:
                raise RuntimeError(f"{model_name}: B evaluation returned no records")
            prediction_writer.write(model_records, model=model_name)
            summary["models"].append({
                "name": model_name,
                "checkpoint": str(ckpt),
                "prediction_rows": int(len(model_records["stock_uid"])),
                "validation_coverage": model_result["validation_coverage"],
                "arms": model_result["arms"],
            })
            del model_records, model_result

    summary["prediction_rows"] = int(prediction_writer.rows)
    write_json(results_root(seed=42) / "model-evaluation.json", summary)
    assert_results_boundary(resolve_roots(seed=42))
    print("=" * 70)
    print(f"[B] Complete model pipeline finished: {prediction_path}")
    print(f"[B] Combined prediction rows: {prediction_writer.rows}")
    print("=" * 70)


# ############################################################################
#  CLI dispatch
# ############################################################################

DEBUG_STAGES = {
    "candidates-eval": _stage_b1,
    "candidates-calibration": _stage_b2,
    "scores-eval": _stage_b3,
    "scores-calibration": _stage_b4,
    "hidden-fit": _stage_b5,
    "hidden-eval": _stage_b6,
    "rank-head": _stage_b7,
    "apply-rank-head": _stage_b8,
    "evaluate": _stage_b9,
    "full": _stage_ball,
}


def main():
    if len(sys.argv) == 1:
        _stage_ball()
        return

    requested = sys.argv[1]
    if requested in {"-h", "--help"}:
        print("B_run.py 无参数时会完成候选生成、BERT 评分、hidden 缓存、")
        print("排序头训练、应用和 controls/arms 评估，并自动包含校准区。")
        print("仅调试单个环节时可传入语义名称，完整流程名称为 full。")
        return
    if requested in {"all", "B-all"}:
        _stage_ball()
        return
    if requested in {"--stage", "-s"}:
        if len(sys.argv) < 3:
            print("缺少调试阶段名称；B_run.py 默认运行完整 B 流程。")
            sys.exit(2)
        requested = sys.argv[2]
        stage_args = sys.argv[3:]
    else:
        stage_args = sys.argv[2:]

    if requested not in DEBUG_STAGES:
        print("B_run.py 默认运行完整 B 流程；未知的调试阶段：", requested)
        print("可用名称：candidates-eval、candidates-calibration、scores-eval、")
        print("scores-calibration、hidden-fit、hidden-eval、rank-head、")
        print("apply-rank-head、evaluate、full")
        sys.exit(2)

    sys.argv = [sys.argv[0], *stage_args]
    DEBUG_STAGES[requested]()


if __name__ == "__main__":
    main()
