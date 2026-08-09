"""f29_quantile.py — quantile regression head on BERT hidden.

Genuinely new magnitude/uncertainty channel (06/07 never trained a quantile head
on BERT hidden):
  - q50 (median) head -> robust magnitude prediction (MAPE channel)
  - q10/q90 heads -> interval width = NEW confidence signal for abstention
Tests:
  A) q50 MAPE / DA vs iso(F) (0.0229 / 0.518) and J3 (0.0249 / 0.511)
  B) interval-width (q90-q10) abstention vs |F| abstention on the F score
Training: fit region, per-row pinball loss, R2 recipe sanity + final fit.
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
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_heads import fold_split, final_fit_split  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, build_rec, cand_path, slice_rec,
    _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


class QuantileHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.1, tau=0.5):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)

    def loss(self, pred, y):
        u = y - pred
        return (u * (self.tau - (u < 0).float())).mean()


def _rows(hidden_path):
    d = np.load(hidden_path, allow_pickle=True)
    rows = {k: d[k] for k in d.files}
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    for k in ("true_logret", "date_key", "stock_uid"):
        rows[k] = tc[k]
    return rows


def train_q(rows, fit_idx, val_idx, tau, lr=3e-4, epochs=12, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = QuantileHead(dim=256, hidden=64, dropout=0.1, tau=tau).to(device)
    H = torch.from_numpy(rows["hidden"].astype(np.float32)).to(device)
    Y = torch.from_numpy(rows["true_logret"].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.RandomState(seed)
    n = len(fit_idx)
    for ep in range(epochs):
        head.train()
        perm = rng.permutation(n)
        for s in range(0, n, 4096):
            ids = fit_idx[perm[s:s + 4096]]
            if len(ids) < 2:
                continue
            opt.zero_grad()
            loss = head.loss(head(H[ids]), Y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    return head


def _daily_mape(rec, field, dense):
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
        out[uniq[i]] = float(np.mean(np.abs(np.exp(s[lo:hi][m] - t[lo:hi][m]) - 1.0)))
    return out


def _daily_da(rec, field, dense):
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
        out[uniq[i]] = float(np.mean((np.sign(s[lo:hi][m]) > 0) == (t[lo:hi][m] > 0)))
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=110)
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()
    rows = _rows(wr / "bert_hidden_fit_w512_t2.npz")
    fit_idx = final_fit_split(rows)
    print(f"[f29] fit rows={len(fit_idx)}")

    # train q10/q50/q90
    heads = {}
    for tau in (0.1, 0.5, 0.9):
        head = train_q(rows, fit_idx, fit_idx, tau, lr=3e-4, epochs=12, seed=args.seed + int(tau * 100))
        torch.save({"head_state": head.state_dict(), "tau": tau},
                   wr / f"head_BERT_q{int(tau*100)}_seed{args.seed}.pt")
        heads[tau] = head
        print(f"[f29] trained q{int(tau*100)}")

    # eval
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    dev = next(iter(heads.values())).parameters().__next__().device
    H = H.to(dev)
    for tau, head in heads.items():
        head.eval()
        with torch.no_grad():
            rec[f"q{int(tau*100)}"] = head(H).cpu().numpy().astype(np.float64)
    rec["q_width"] = rec["q90"] - rec["q10"]

    res = {"schema": "f29-quantile-v1", "seed": args.seed, "dense_threshold": dense}
    # A) q50 magnitude
    res["mag_q50"] = {"mape": float(np.mean(list(_daily_mape(rec, "q50", dense).values()))),
                      "da": float(np.mean(list(_daily_da(rec, "q50", dense).values())))}
    res["mag_isoF"] = {"mape": 0.0229, "da": 0.5181}  # reference from f27
    print(f"[f29] q50 mape={res['mag_q50']['mape']:.4f} da={res['mag_q50']['da']:.4f} "
          f"(isoF 0.0229/0.518)")

    # B) interval-width abstention on F score (need F)
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval(); h = h.to(dev)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).cpu().numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)
    rec["c_mag"] = np.abs(rec["F"])

    res["abstention"] = {}
    for cf, max_high in (("c_mag", True), ("q_width", False)):
        row = []
        for cv in (1.0, 0.8, 0.6, 0.4, 0.2):
            acted = np.ones(n, dtype=bool) if cv == 1.0 else _exact_frac(rec, cf, cv, max_high)
            ra = slice_rec(rec, acted)
            da20 = max(5, int(round(cv * dense)))
            ic_ = _daily_ic(ra, "F", da20)
            row.append({"cov": cv, "ic": float(np.mean(list(ic_.values()))) if ic_ else None})
        res["abstention"][cf] = row
        print(f"[f29] {cf:8s}: " + " ".join(f"cov={r['cov']} ic={r['ic'] and round(r['ic'],4)}" for r in row))

    out = rr / "f29_quantile.json"
    write_json_ledger(out, res, "f29_quantile", seed=args.seed)
    print(f"[f29] -> {out}")


if __name__ == "__main__":
    main()
