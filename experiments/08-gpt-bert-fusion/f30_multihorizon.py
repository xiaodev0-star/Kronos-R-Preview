"""f30_multihorizon.py — multi-horizon consistency filter (07 plan N7, untested).

Trains a y5-direction head on BERT hidden (5-day cumulative log-return sign),
then tests a HORIZON-CONSISTENCY filter: keep a row only if the y1 rank head
direction and the y5 head direction AGREE.  This is a genuinely-new confidence
signal orthogonal to |F| (time-consistency rather than cross-sectional).

Training labels y5(p) = sign(sum_{k=0}^{4} raw_logret[p+k]) from per-stock
feature arrays (point-in-time, no leakage: only fit rows < 2023-02-01 trained).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
SIX = ROOT / "experiments" / "06-posttrain"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, SIX, EIGHT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


class DirHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


def _load_stock_feats():
    from posttrain_data import (load_stocks_uid, attach_close_prices_uid,
                                prepare_stocks_uid)
    from model import load_tokenizer
    from critic_common import upstream_paths
    from config import DataConfig
    _, tok_path = upstream_paths()
    tokenizer = load_tokenizer(str(tok_path), torch.device("cpu"))
    stocks = load_stocks_uid(DataConfig.data_dir)
    attach_close_prices_uid(stocks)
    prepped = prepare_stocks_uid(stocks, tokenizer, torch.device("cpu"))
    # precompute dates_int + y5 cumulative-sum array per stock (vectorized)
    out = {}
    for p in prepped:
        dr = np.asarray(p.get("dates_raw") if p.get("dates_raw") is not None
                        else p.get("dates"))
        if dr is None or np.ndim(dr) == 0 or len(dr) == 0:
            continue
        dates_int = np.asarray(
            [int(str(d)[:10].replace("-", "")) for d in dr], dtype=np.int32)
        f = p["feat"]
        cs = np.concatenate([[0.0], np.cumsum(f[:, 0])])
        # y5[p] = cs[p+5] - cs[p] for p in [0, T-5], else NaN
        T = f.shape[0]
        y5arr = np.full(T, np.nan)
        y5arr[:T - 4] = cs[5:T + 1] - cs[:T - 4]
        p["dates_int"] = dates_int
        p["y5arr"] = y5arr
        out[p["stock_uid"]] = p
    return out


def _y5_for(stock, position, feat=None):
    """sum of next-5 raw log returns starting at position p (row-aligned)."""
    f = stock["feat"]
    p = int(position)
    if p + 5 > f.shape[0]:
        return np.nan
    return float(np.sum(f[p:p + 5, 0]))


def _pos_for(stock, date_key):
    d_int = int(str(date_key)[:10].replace("-", ""))
    di = np.searchsorted(stock["dates_int"], d_int, side="left")
    if di >= len(stock["dates_int"]) or int(stock["dates_int"][di]) != d_int:
        return None
    return int(di)


def _daily_ic(rec, field, dense):
    score = np.asarray(rec[field], dtype=np.float64)
    true = np.asarray(rec["true_logret"], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(true) & np.asarray(rec["quality"]).astype(bool)
    order, bounds, uniq = _split_points(rec["date_key"])
    s, t, v = score[order], true[order], valid[order]
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        m = v[lo:hi]
        c = int(m.sum())
        if c < dense:
            continue
        sv, tv = s[lo:hi][m], t[lo:hi][m]
        out[uniq[i]] = float(spearmanr(sv, tv)[0]) if (c >= 2 and not np.all(sv == sv[0])) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=120)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    print("[f30] loading stocks...", flush=True)
    feats = _load_stock_feats()

    # ---- fit y5 labels ----
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    n_fit = len(tc["stock_uid"])
    y5 = np.full(n_fit, np.nan)
    hit = 0
    uids = np.asarray([str(u) for u in tc["stock_uid"]])
    dates = np.asarray([str(d)[:10] for d in tc["date_key"]])
    dint = np.asarray([int(d.replace("-", "")) for d in dates])
    uniq, inv = np.unique(uids, return_inverse=True)
    for ui, uid in enumerate(uniq):
        st = feats.get(str(uid))
        if st is None:
            continue
        ref = st["dates_int"]
        here = np.where(inv == ui)[0]
        didx = np.searchsorted(ref, dint[here], side="left")
        ok = (didx < len(ref)) & (ref[didx] == dint[here])
        pos = np.where(ok, didx, -1)
        y5arr = st["y5arr"]
        valid = ok & (pos < len(y5arr))
        y5[here[valid]] = y5arr[pos[valid]]
        hit += int(valid.sum())
    print(f"[f30] y5 fit hit={hit}/{n_fit}")

    # ---- train y5-direction head on BERT hidden ----
    rows = {k: tc[k] for k in ("stock_uid", "date_key")}
    b = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    rows["hidden"] = b["hidden"]
    y5_t = np.asarray(y5, dtype=np.float64)
    keep = np.isfinite(y5_t) & (np.asarray([str(d)[:10] for d in tc["date_key"]]) < "2023-02-01")
    train_idx = np.where(keep)[0]
    print(f"[f30] y5 train rows: {len(train_idx)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = DirHead(dim=256, hidden=64, dropout=0.1).to(dev)
    H = torch.from_numpy(rows["hidden"][train_idx].astype(np.float32)).to(dev)
    Y = torch.from_numpy((y5_t[train_idx] > 0).astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.0)
    rng = np.random.RandomState(args.seed)
    for ep in range(10):
        head.train()
        perm = rng.permutation(len(train_idx))
        tot = 0; ns = 0
        for s in range(0, len(train_idx), 4096):
            ids = perm[s:s + 4096]
            if len(ids) < 2:
                continue
            opt.zero_grad()
            out = head(H[ids])
            loss = nn.functional.binary_cross_entropy_with_logits(out, Y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += loss.item(); ns += 1
        print(f"[f30] ep{ep} loss={tot/max(1,ns):.4f}")
    torch.save({"head_state": head.state_dict(), "seed": args.seed},
               wr / f"head_BERT_y5dir_seed{args.seed}.pt")

    # ---- eval: y5 pred + consistency filter ----
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H_e = torch.from_numpy(de["hidden"].astype(np.float32)).to(dev)
    head.eval()
    with torch.no_grad():
        y5_logit = head(H_e).cpu().numpy().astype(np.float64)
    rec["y5_dir"] = (y5_logit > 0).astype(bool)

    # F score + direction
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval(); h = h.to(dev)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H_e).cpu().numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])
    rec["F_dir"] = rec["F"] > 0

    res = {"schema": "f30-multihorizon-v1", "seed": args.seed, "dense_threshold": dense}

    # consistency filter: keep rows where y1-dir == y5-dir
    agree = (rec["F_dir"] == rec["y5_dir"])
    rec["agree"] = agree
    print(f"[f30] eval agreement frac: {agree.mean():.3f}")
    res["agree_frac"] = float(agree.mean())

    for name, mask in (("all", np.ones(n, dtype=bool)),
                       ("agree", agree),
                       ("disagree", ~agree)):
        ra = slice_rec(rec, mask)
        ic_ = _daily_ic(ra, "F", dense)
        res[name] = {"ic": float(np.mean(list(ic_.values()))) if ic_ else None,
                     "n": int(mask.sum())}
        print(f"[f30] {name:10s} ic={res[name]['ic'] and round(res[name]['ic'],4)} n={mask.sum()}")

    # combined: agree AND |F| top-cv
    res["agree_cmag20"] = None
    for cv in (0.8, 0.6, 0.4, 0.2):
        cmag = _exact_frac(rec, "c_mag", cv, True)
        mask = agree & cmag
        ra = slice_rec(rec, mask)
        da20 = max(5, int(round(cv * dense)))
        ic_ = _daily_ic(ra, "F", da20)
        print(f"[f30] agree & |F|top{cv}: ic={np.mean(list(ic_.values())) if ic_ else None} n={mask.sum()}")

    out = rr / "f30_multihorizon.json"
    write_json_ledger(out, res, "f30_multihorizon", seed=args.seed)
    print(f"[f30] -> {out}")


def _exact_frac(rec, cf, cv, max_high=True):
    conf = np.asarray(rec[cf], dtype=np.float64)
    dates = np.asarray(rec["date_key"])
    acted = np.zeros(len(rec["stock_uid"]), dtype=bool)
    for d in np.unique(dates):
        dm = dates == d
        cvals = conf[dm]
        m = np.isfinite(cvals)
        if m.sum() == 0:
            continue
        keep_n = max(1, int(round(cv * m.sum())))
        ord_ = np.argsort(-cvals) if max_high else np.argsort(cvals)
        idx = np.where(dm)[0]
        acted[idx[ord_[:keep_n]]] = True
    return acted


if __name__ == "__main__":
    main()
