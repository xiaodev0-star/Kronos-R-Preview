"""train_mlp.py — MLP baseline on the shared c1_feats.

A small fully-connected network (5 -> 64 -> 1) regressing the next-day raw log
return, trained on the fit region.  The predicted return is the row score.

Maintainable / self-contained:
  - uses data.py + common.py
  - predictions saved to outputs/mlp_pred.npz
  - rolling validation slice for a quick recipe sanity, then final fit

Usage:
    python train_mlp.py
    python train_mlp.py --hidden 128 --epochs 20 --lr 3e-4
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


class MLP(nn.Module):
    def __init__(self, dim=5, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train(X, y, *, hidden, epochs, lr, batch, seed, device):
    torch.manual_seed(seed)
    model = MLP(dim=X.shape[1], hidden=hidden, dropout=0.1).to(device)
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
            print(f"[mlp] ep{ep} loss={tot/max(1,ns):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="small grid on R2 val slice")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fit = data.load_fit_features()
    X, y = fit["X"], fit["y"]
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[fin], y[fin]
    print(f"[mlp] fit rows: {len(y)}")

    dates = np.asarray([str(d)[:10] for d in fit["date_key"]])
    tr_mask = dates[fin] < "2022-02-01"
    va_mask = (dates[fin] >= "2022-02-01") & (dates[fin] < "2023-02-01")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- light tuning on the R2 val slice ----
    chosen = {"hidden": args.hidden, "epochs": args.epochs}
    if args.tune and tr_mask.sum() > 0 and va_mask.sum() > 0:
        grid = [{"hidden": 32, "epochs": 12},
                {"hidden": 64, "epochs": 20},
                {"hidden": 128, "epochs": 16}]
        best = None
        for cfg in grid:
            m_ = _train(X[tr_mask], y[tr_mask], hidden=cfg["hidden"], epochs=cfg["epochs"],
                        lr=args.lr, batch=args.batch, seed=args.seed, device=dev)
            m_.eval()
            with torch.no_grad():
                vp = m_(torch.from_numpy(X[va_mask].astype(np.float32)).to(dev)) \
                    .cpu().numpy().astype(np.float64)
            vr = {"date_key": fit["date_key"][fin][va_mask], "true_logret": y[va_mask],
                  "quality": np.ones(va_mask.sum(), dtype=bool),
                  "dense_threshold": max(5, int(0.8 * 400))}
            ic = full_metrics(vp, vr)["avg_daily_rank_ic"]
            print(f"[mlp] tune {cfg} val RankIC={ic:.4f}")
            if best is None or ic > best[1]:
                best = (cfg, ic)
        chosen.update(best[0])
        print(f"[mlp] tuned -> {chosen} (val RankIC={best[1]:.4f})")

    t0 = time.time()
    model = _train(X, y, hidden=chosen["hidden"], epochs=chosen["epochs"], lr=args.lr,
                   batch=args.batch, seed=args.seed, device=dev)
    print(f"[mlp] final trained in {time.time()-t0:.0f}s")

    ev = data.load_eval_features()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(ev["X"].astype(np.float32)).to(dev)) \
            .cpu().numpy().astype(np.float64)
    pred[~ev["valid"]] = np.nan

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[mlp] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("mlp", pred, meta={
        "features": "c1_feats(5)", "loss": "smooth_l1", "hidden": chosen["hidden"],
        "epochs": chosen["epochs"], "lr": args.lr, "batch": args.batch, "seed": args.seed})
    save_json(RESULTS / "mlp_metrics.json", m)
    print(f"[mlp] saved -> {path}")


if __name__ == "__main__":
    main()
