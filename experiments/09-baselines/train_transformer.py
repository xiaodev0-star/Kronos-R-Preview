"""train_transformer.py — sequence Transformer baseline.

A small transformer encoder over a window of past daily log returns (W=32),
predicting the next-day raw log return.  The predicted return is the row score.

Architecture (maintainable):
  [B, W] returns
    -> linear per-timestep to dim + learned positional embedding
    -> N=2 transformer-encoder layers (heads=4, dim=64)
    -> use the LAST timestep representation -> MLP head -> scalar return

Training: Huber loss on the fit-region windows (NaN windows dropped).

Usage:
    python train_transformer.py
    python train_transformer.py --window 32 --dim 64 --layers 2 --epochs 15
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


class SequenceTransformer(nn.Module):
    def __init__(self, window=32, dim=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, dim)
        self.pos = nn.Parameter(torch.randn(1, window, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: [B, W] returns
        h = self.input_proj(x.unsqueeze(-1)) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def _train(X, y, *, window, dim, layers, heads, epochs, lr, batch, seed, device):
    torch.manual_seed(seed)
    model = SequenceTransformer(window=window, dim=dim, layers=layers,
                                heads=heads, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    rng = np.random.RandomState(seed)
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0; ns = 0
        for s in range(0, n, batch):
            ids = perm[s:s + batch]
            if len(ids) < 2:
                continue
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(Xt[ids]), yt[ids], beta=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); ns += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[trans] ep{ep} loss={tot/max(1,ns):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="small grid on R2 val slice")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fit = data.load_fit_sequences(window=args.window)
    X, y = fit["X"], fit["y"]
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[fin], y[fin]
    dates = np.asarray([str(d)[:10] for d in fit["date_key"]])[fin]
    print(f"[trans] fit rows: {len(y)} (of {len(fit['y'])})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- light tuning on the R2 val slice ----
    chosen = {"dim": args.dim, "layers": args.layers, "epochs": args.epochs}
    if args.tune:
        tr_mask = dates < "2022-02-01"
        va_mask = (dates >= "2022-02-01") & (dates < "2023-02-01")
        grid = [{"dim": 32, "layers": 1},
                {"dim": 64, "layers": 2}]
        best = None
        for cfg in grid:
            m_ = _train(X[tr_mask], y[tr_mask], window=args.window, dim=cfg["dim"],
                        layers=cfg["layers"], heads=args.heads, epochs=6, lr=args.lr,
                        batch=args.batch, seed=args.seed, device=dev)
            m_.eval()
            with torch.no_grad():
                vp = m_(torch.from_numpy(X[va_mask].astype(np.float32)).to(dev)) \
                    .cpu().numpy().astype(np.float64)
            vr = {"date_key": dates[va_mask], "true_logret": y[va_mask],
                  "quality": np.ones(va_mask.sum(), dtype=bool),
                  "dense_threshold": max(5, int(0.8 * 400))}
            ic = full_metrics(vp, vr)["avg_daily_rank_ic"]
            print(f"[trans] tune {cfg} val RankIC={ic:.4f}")
            if best is None or ic > best[1]:
                best = (cfg, ic)
        chosen.update(best[0])
        chosen["epochs"] = args.epochs
        print(f"[trans] tuned -> {chosen} (val RankIC={best[1]:.4f})")

    t0 = time.time()
    model = _train(X, y, window=args.window, dim=chosen["dim"], layers=chosen["layers"],
                   heads=args.heads, epochs=chosen["epochs"], lr=args.lr,
                   batch=args.batch, seed=args.seed, device=dev)
    print(f"[trans] trained in {time.time()-t0:.0f}s")

    ev = data.load_eval_sequences(window=args.window)
    Xev = ev["X"]
    pred = np.full(len(Xev), np.nan)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(Xev), 65536):
            chunk = Xev[s:s + 65536]
            ok = np.isfinite(chunk).all(axis=1)
            if ok.any():
                z = torch.from_numpy(chunk[ok].astype(np.float32)).to(dev)
                pred[s:s + 65536][ok] = model(z).cpu().numpy()
    del Xev

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[trans] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("transformer", pred, meta={
        "features": f"return-window({args.window})", "arch": "TransformerEncoder",
        "dim": chosen["dim"], "layers": chosen["layers"], "heads": args.heads,
        "epochs": chosen["epochs"], "lr": args.lr, "batch": args.batch, "seed": args.seed})
    save_json(RESULTS / "transformer_metrics.json", m)
    print(f"[trans] saved -> {path}")


if __name__ == "__main__":
    main()
