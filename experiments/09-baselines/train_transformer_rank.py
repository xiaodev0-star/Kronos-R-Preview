"""train_transformer_rank.py — Transformer with per-date soft-Spearman RANK loss.

The original Transformer baseline used Huber regression on W=32 return windows →
predictions shrink toward zero → weak RankIC (0.009).  This version trains the
same sequence transformer with the Exp08-style per-date soft-Spearman rank loss
(one cross-section per step), directly optimizing daily RankIC — the same
objective that powers the Exp08 heads.

Architecture: reuse SequenceTransformer (2 layers, dim 96, heads 4) but with a
rank training loop that groups fit rows by date.

Usage:
    python train_transformer_rank.py
    python train_transformer_rank.py --dim 128 --epochs 12
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import common
import data
from common import load_eval_rows, full_metrics, save_predictions, save_json, RESULTS
from _exp07 import soft_spearman_loss


class SequenceTransformer(nn.Module):
    def __init__(self, window=32, dim=96, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, dim)
        self.pos = nn.Parameter(torch.randn(1, window, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.input_proj(x.unsqueeze(-1)) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_dates", type=int, default=0,
                    help="limit number of dates/epoch (smoke)")
    ap.add_argument("--apply_only", action="store_true")
    ap.add_argument("--out", default="transformer_rank")
    ap.add_argument("--max-stocks", type=int, default=0,
                    help="sample N stocks per date (bounds GPU memory; 0=all)")
    args = ap.parse_args()

    if args.apply_only:
        apply_eval(args)
        return

    fit = data.load_fit_sequences(window=32)
    X, y = fit["X"], fit["y"]
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[fin], y[fin]
    dates = np.asarray([str(d)[:10] for d in fit["date_key"]])[fin]
    # group by date
    groups = {}
    for i, d in enumerate(dates):
        groups.setdefault(d, []).append(i)
    date_list = sorted(groups.keys())
    if args.max_dates:
        date_list = date_list[: args.max_dates]
    print(f"[tr] fit rows={len(y)} dates={len(date_list)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = SequenceTransformer(window=32, dim=args.dim, layers=args.layers,
                                heads=4, dropout=0.1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = y.astype(np.float64)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0; ns = 0
        for d in date_list:
            idx = groups[d]
            if args.max_stocks > 0 and len(idx) > args.max_stocks:
                rng = np.random.RandomState(abs(hash(d)) % (2**31))
                idx = [idx[i] for i in rng.permutation(len(idx))[:args.max_stocks]]
            if len(idx) < 2:
                continue
            seqs = Xt[idx].to(dev)
            lab = yt[idx]
            opt.zero_grad()
            scores = model(seqs)
            ranks = torch.from_numpy(np.argsort(np.argsort(lab)).astype(np.float32)).to(dev)
            loss = soft_spearman_loss(scores, ranks, tau=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); ns += 1
            if ns % 500 == 0:
                print(f"[tr] ep{ep} step{ns} loss={loss.item():.4f}", flush=True)
        print(f"[tr] ep{ep} avg_loss={tot/max(1,ns):.4f}")
    print(f"[tr] trained in {time.time()-t0:.0f}s")
    torch.save({"model_state": model.cpu().state_dict(),
                "config": {"window": 32, "dim": args.dim, "layers": args.layers,
                           "heads": 4},
                "meta": {"features": "return-window(32)", "loss": "soft_spearman_per_date",
                         "epochs": args.epochs, "lr": args.lr, "seed": args.seed}},
               common.CHECKPOINTS / "transformer_rank.pt")
    print(f"[tr] model saved -> {common.CHECKPOINTS/'transformer_rank.pt'}")
    # free fit data before eval (system memory is tight)
    del X, y, dates, groups, date_list, Xt, yt
    import gc; gc.collect()

    # ---- eval ----
    apply_eval(args, model=model, dev=dev)


def apply_eval(args, model=None, dev=None):
    import gc
    common_module = common
    if model is None:
        ck = torch.load(common_module.CHECKPOINTS / "transformer_rank.pt",
                        map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = SequenceTransformer(window=cfg["window"], dim=cfg["dim"],
                                    layers=cfg["layers"], heads=cfg["heads"],
                                    dropout=0.1)
        model.load_state_dict(ck["model_state"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        args = type("A", (), {"out": ck.get("meta", {}).get("name", "transformer_rank")})()
        args.out = "transformer_rank"
        print("[tr-apply] loaded model")
    model = model.to(dev)
    model.eval()
    gc.collect()

    ev = data.load_eval_sequences(window=32)
    Xev = ev["X"]
    pred = np.full(len(Xev), np.nan)
    with torch.no_grad():
        for s in range(0, len(Xev), 65536):
            chunk = Xev[s:s + 65536]
            ok = np.isfinite(chunk).all(axis=1)
            if ok.any():
                z = torch.from_numpy(chunk[ok].astype(np.float32)).to(dev)
                pred[s:s + 65536][ok] = model(z).cpu().numpy()
    del Xev
    gc.collect()

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[tr] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions(args.out, pred, meta={
        "features": "return-window(32)", "arch": "TransformerEncoder",
        "loss": "soft_spearman_per_date"})
    save_json(RESULTS / f"{args.out}_metrics.json", m)
    print(f"[tr] saved -> {path}")


if __name__ == "__main__":
    main()
