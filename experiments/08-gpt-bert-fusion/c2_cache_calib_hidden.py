"""c2_cache_calib_hidden.py — cache MASK-position BERT hidden for the calib slice.

The 07 cache_bert_hidden.py only supports fit/eval regions; Round 2 (blend-weight
fitting + abstention thresholds) needs the calibration slice
(audit_uids x [2023-02-01, 2024-02-01) = candidates_calib_K8 rows).  This reuses
the exact 07 input construction (score_bert build_index + bert_hidden_rows) with
positions computed from the index (calib candidates lack a 'position' field).

Output: 07 weights_root/bert_hidden_calib_w512_t2.npz (+ fingerprint json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from critic_common import resolve_roots, upstream_paths  # noqa: E402
from score_bert import load_bert, build_index  # noqa: E402
from cache_bert_hidden import bert_hidden_rows, _vectorized_dates_int  # noqa: E402
from improve_common import row_set_fingerprint, append_trial  # noqa: E402


def _precompute_positions(index, stock_uid, date_key):
    """Same as cache_bert_hidden._precompute_positions (copy to avoid private import)."""
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


class SimpleRows:
    def __init__(self, u, d, position):
        self._u = np.asarray(u)
        self._d = np.asarray(d)
        self._position = np.asarray(position, dtype=np.int64)

    def __len__(self):
        return len(self._u)

    def stock_uid(self, i):
        return str(self._u[i])

    def date_key(self, i):
        return str(self._d[i])

    def position(self, i):
        return int(self._position[i])

    def select(self, mask):
        self._u = self._u[mask]
        self._d = self._d[mask]
        self._position = self._position[mask]
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default=str(ROOT / "checkpoints" / "bert_critic_mlm_t2.pt"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--suffix", type=str, default="t2")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_bert(Path(args.bert), torch.device(device))
    print(f"[c2] BERT dim={cfg.dim} depth={cfg.depth} on {device}")

    _, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=roots.weights_root / "bert_input_index.pkl")

    c = np.load(roots.weights_root / "candidates_calib_K8.npz", allow_pickle=True)
    u, d = c["stock_uid"], c["date_key"]
    pos = _precompute_positions(index, u, d)
    rows = SimpleRows(u, d, pos)
    keep = np.array([str(rows.stock_uid(i)) in index for i in range(len(rows))])
    rows.select(keep)
    print(f"[c2] calib rows after index filter: {len(rows)}")

    h = bert_hidden_rows(index, rows, model, dim=cfg.dim, window=args.window,
                         batch_size=args.batch_size, device=device)

    sfx = f"_{args.suffix}" if args.suffix else ""
    out_npz = roots.weights_root / f"bert_hidden_calib_w{args.window}{sfx}.npz"
    np.savez(out_npz, stock_uid=np.asarray(rows._u), date_key=np.asarray(rows._d),
             hidden=h, window=np.array([args.window]), bert_ckpt=args.bert)
    fp = row_set_fingerprint(np.asarray(rows._u), np.asarray(rows._d), args.bert,
                             out_json=roots.results_root /
                             f"bert_hidden_calib_w{args.window}{sfx}.fingerprint.json")
    print(f"[c2] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    append_trial({"event": "cache_bert_hidden", "region": "calib",
                  "n_rows": len(rows), "status": "ok"})
    print(f"[c2] fingerprint rows={fp['n_rows']}")


if __name__ == "__main__":
    main()
