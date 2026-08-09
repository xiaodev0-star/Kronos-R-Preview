"""score_bert.py — plan §8: BERT critic scoring over a row region.

Main score (plan §4.2): one forward with the t+1 slot set to [MASK]; read the
MASK-position coarse logits, log_softmax -> log p_BERT(c) for all 128 candidates.
One forward yields every candidate's score (no per-candidate insertion).

Leakage red line (hard): the MASK row's va_values must be exactly 0; the
sequence is truncated at the MASK slot; every MASK row's time_ids is the target
day's real calendar (deterministic, known).  These are pinned by contract tests
T1/T2.

Regions (all share ``BertInputIndex``, built once from prepare_stocks_uid):
  - fit   : training_cache.npz rows  (fit_uids x pre-2023-02-01, for C-a/C-b)
  - calib : calibration_cache.npz    (audit_uids x [2023-02-01, 2024-02-01), for fusion)
  - eval  : candidate cache rows     (offsets 0..399, 1,798,899 rows, for B0-B5)

Usage:
    python experiments/07-bert-critic/score_bert.py --region eval --stride 10   # pilot
    python experiments/07-bert-critic/score_bert.py --region eval               # full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import DataConfig  # noqa: E402
from experiment_io import file_sha256  # noqa: E402
from model import load_tokenizer  # noqa: E402
from model.kronos_bert import KronosBert  # noqa: E402

from bert_data import (  # noqa: E402
    VOCAB_BASE, bert_time_for_target_date, selection_history_window,
    build_bert_input,
)
from critic_common import resolve_roots, upstream_paths, append_trial  # noqa: E402


# ============================================================================
# BERT checkpoint loading
# ============================================================================

def load_bert(ckpt_path, device):
    """Load a KronosBert from a checkpoint with its recorded config."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    import types
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
    import pickle
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            idx = pickle.load(f)
        print(f"[index] loaded cached index {len(idx.by_uid)} stocks")
        return idx
    tokenizer = load_tokenizer(str(tokenizer_path), torch.device(device))
    from posttrain_data import load_stocks_uid, attach_close_prices_uid, prepare_stocks_uid
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
    order.  ``position_for`` uses searchsorted on the sorted dates — no
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
    ``topk_ids(i)`` accessors — see ``NumpyRowTable`` below.  Passing numpy
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


def main():
    ap = argparse.ArgumentParser(description="BERT critic scoring")
    ap.add_argument("--bert", default=None,
                    help="BERT checkpoint (default checkpoints/bert_critic_mlm_v1.pt)")
    ap.add_argument("--region", choices=["eval", "calib", "fit"], default="eval")
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth row (pilot)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_stocks", type=int, default=0,
                    help="limit the input index to the first N stocks (dev only)")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (e.g. t1_w512) so fine-tuned scores do not "
                         "overwrite the mlm_v1 scores cache")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    ckpt_path = Path(args.bert) if args.bert else ROOT / "checkpoints" / "bert_critic_mlm_v1.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"BERT checkpoint missing: {ckpt_path}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- load BERT ----
    model, cfg, ckpt_meta = load_bert(ckpt_path, torch.device(device))
    print(f"[score] BERT loaded: vocab={cfg.vocab_size} dim={cfg.dim} "
          f"depth={cfg.depth} heads={cfg.heads} mask_id={model.mask_id}")

    # ---- input index (cached across scripts) ----
    _, tok_path = upstream_paths()
    index_cache = roots.weights_root / "bert_input_index.pkl"
    index = build_index(tok_path, device="cpu", max_stocks=args.max_stocks,
                        cache_path=index_cache)
    print(f"[score] input index: {len(index.by_uid)} stocks")

    # ---- region rows ----
    candidates = roots.weights_root / "candidates_eval_K8.npz"
    calib_cache = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "calibration_cache.npz"
    training_cache = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "training_cache.npz"
    rows = _region_rows(args.region, candidates, calib_cache, training_cache, args.stride)
    keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
    rows = rows.select(keep)
    # sanity: for eval, the index-derived position must equal the candidate's
    if args.region == "eval" and rows:
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
    print(f"[score] region={args.region} rows to score: {len(rows)}")

    # ---- score ----
    out = score_rows(index, rows, model, window=args.window,
                     batch_size=args.batch_size, device=device)
    arr = {k: v for k, v in out.items()}
    arr["stock_uid"] = np.asarray(rows._u)
    arr["date_key"] = np.asarray(rows._d)

    sfx = f"_{args.suffix}" if args.suffix else ""
    out_npz = roots.weights_root / f"scores_{args.region}_K8_w{args.window}_stride{args.stride}{sfx}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arr)
    print(f"[score] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    meta = {
        "schema": "scores-v1", "region": args.region, "stride": args.stride,
        "window": args.window, "n_rows": len(rows),
        "bert": {"checkpoint": str(ckpt_path), "sha256": file_sha256(ckpt_path),
                 "vocab_size": cfg.vocab_size, "mask_id": model.mask_id,
                 "val_loss": ckpt_meta.get("val_loss"),
                 "mlm_acc": ckpt_meta.get("mlm_acc"),
                 "epoch": ckpt_meta.get("epoch")},
        "leakage": {"mask_va_zero": True, "mask_truncates_sequence": True},
    }
    out_json = roots.results_root / f"scores_{args.region}_K8_w{args.window}_stride{args.stride}{sfx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    append_trial({"event": "score_bert", "region": args.region, "stride": args.stride,
                  "n_rows": len(rows), "status": "ok"})
    print(f"[score] sidecar -> {out_json}")


if __name__ == "__main__":
    main()
