"""cache_bert_hidden.py — plan §4 T5: cache the MASK-position BERT hidden states.

T5 trains a P6-style rank head on the frozen BERT's t+1-slot hidden vector
(the same scoring input as score_bert: truncated window + [MASK] at the last
position, MASK row va=0).  This script runs the BERT forward over a region and
saves the MASK-position hidden [N, dim] (float32) plus the §6 contract-5
fingerprint (BERT ckpt hash + row-set hash).

Regions:
  fit   training_cache.npz rows (fit_uids x pre-2023-02)  — head training
  eval  candidate cache rows (offsets 0..399)              — head application

Usage:
    python experiments/07-bert-critic/cache_bert_hidden.py --region fit
    python experiments/07-bert-critic/cache_bert_hidden.py --region eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from critic_common import resolve_roots, upstream_paths, append_trial  # noqa: E402
from score_bert import load_bert, build_index, NumpyRowTable  # noqa: E402
from improve_common import row_set_fingerprint, weights_root, results_root  # noqa: E402


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
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    sl = slice(0, len(tc["stock_uid"]), max(1, stride))
    u, d = tc["stock_uid"][sl], tc["date_key"][sl]
    pos = _precompute_positions(index, u, d) if index is not None else None
    return NumpyRowTable(u, d, position=pos)


def _eval_rows(stride=1):
    roots = resolve_roots(seed=42)
    c = np.load(roots.weights_root / "candidates_eval_K8.npz", allow_pickle=True)
    sl = slice(0, len(c["stock_uid"]), max(1, stride))
    return NumpyRowTable(c["stock_uid"][sl], c["date_key"][sl],
                         position=c["position"][sl])


def main():
    ap = argparse.ArgumentParser(description="Cache MASK-position BERT hidden")
    ap.add_argument("--region", choices=["fit", "eval"], default="fit")
    ap.add_argument("--bert", default=str(ROOT / "checkpoints" / "bert_critic_mlm_v1.pt"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="subsample (dev smoke)")
    ap.add_argument("--max_rows", type=int, default=0, help="dev smoke limit")
    ap.add_argument("--suffix", type=str, default="",
                    help="output suffix (e.g. t1_w512) so a fine-tuned BERT's hidden "
                         "cache does not collide with mlm_v1's")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_bert(Path(args.bert), torch.device(device))
    print(f"[hidden] BERT dim={cfg.dim} depth={cfg.depth} on {device}")

    _, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=roots.weights_root / "bert_input_index.pkl")

    rows = (_fit_rows(args.stride, index=index) if args.region == "fit"
            else _eval_rows(args.stride))
    keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
    rows = rows.select(keep)
    if args.max_rows > 0:
        rows = rows.select(np.arange(min(args.max_rows, len(rows))))
    print(f"[hidden] region={args.region} rows={len(rows)}")

    h = bert_hidden_rows(index, rows, model, dim=cfg.dim, window=args.window,
                         batch_size=args.batch_size, device=device)

    sfx = f"_{args.suffix}" if args.suffix else ""
    out_npz = roots.weights_root / f"bert_hidden_{args.region}_w{args.window}{sfx}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, stock_uid=np.asarray(rows._u), date_key=np.asarray(rows._d),
             hidden=h, window=np.array([args.window]),
             bert_ckpt=args.bert)
    fp = row_set_fingerprint(np.asarray(rows._u), np.asarray(rows._d), args.bert,
                             out_json=roots.results_root /
                             f"bert_hidden_{args.region}_w{args.window}{sfx}.fingerprint.json")
    print(f"[hidden] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    print(f"[hidden] fingerprint: rows={fp['n_rows']} uids={fp['n_unique_uids']} "
          f"dates={fp['n_unique_dates']} ckpt_sha={fp['bert_ckpt_sha256'][:10]}")
    append_trial({"event": "cache_bert_hidden", "region": args.region,
                  "n_rows": len(rows), "status": "ok"})


if __name__ == "__main__":
    main()
