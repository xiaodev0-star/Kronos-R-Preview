"""train_tcn.py — TCN baseline (Temporal Convolutional Network).

A causal dilated convolutional network over a window of past daily log returns
(W=32), predicting the next-day raw log return.  Replaces the TimeFM baseline
(pretrained checkpoint unavailable in this environment; see train_timesfm.py).

Architecture (Bai et al. 2018-style causal dilated TCN):
  [B, W] returns
    -> 1D causal dilated conv blocks (kernel=3, dilations 1,2,4,...)
    -> last-timestep features -> MLP head -> scalar return

Light tuning (--tune): a small grid over channels/layers is scored on the
2022-02..2023-02 validation slice by daily RankIC, then the final model is fit
with the best config on the full fit region (no eval-window fitting).

Usage:
    python train_tcn.py            # default config
    python train_tcn.py --tune     # small grid on the R2 val slice
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import common
import data
from common import load_eval_rows, full_metrics, daily_rank_ic, save_predictions, save_json, RESULTS


class CausalConvBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1), nn.SiLU())

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, :x.shape[2]]          # causal crop


class TCN(nn.Module):
    def __init__(self, window=32, channels=32, layers=3, kernel=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            [CausalConvBlock(channels, kernel, 2 ** i, dropout)
             for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: [B, W]
        h = self.embed(x.unsqueeze(1))        # [B, C, W]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, :, -1]).squeeze(-1)


def _train(X, y, *, channels, layers, epochs, lr, batch, seed, device):
    torch.manual_seed(seed)
    model = TCN(window=X.shape[1], channels=channels, layers=layers,
                kernel=3, dropout=0.1).to(device)
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
            print(f"[tcn] ep{ep} loss={tot/max(1,ns):.4f}")
    return model


def _val_rankic(model, X, y, dates, mask, device):
    Xm, ym = X[mask], y[mask]
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xm.astype(np.float32)).to(device)).cpu().numpy()
    vr = {"date_key": dates[mask], "true_logret": ym, "quality": np.ones(mask.sum(), dtype=bool),
          "dense_threshold": max(5, int(0.8 * 400))}
    ic = daily_rank_ic(pred, vr)
    return float(np.mean(list(ic.values()))) if ic else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="small grid on R2 val slice")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.device == "cuda":
        dev0 = "cuda"
    elif args.device == "cpu":
        dev0 = "cpu"
    else:
        try:
            dev0 = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001  (broken CUDA driver -> CPU)
            dev0 = "cpu"
    print(f"[tcn] device: {dev0}")

    fit = data.load_fit_sequences(window=32)
    X, y = fit["X"], fit["y"]
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[fin], y[fin]
    dates = np.asarray([str(d)[:10] for d in fit["date_key"]])[fin]

    # ---- light tuning on the R2 val slice ----
    chosen = {"channels": args.channels, "layers": args.layers,
              "epochs": args.epochs, "lr": args.lr}
    if args.tune:
        dev = dev0
        tr_mask = dates < "2022-02-01"
        va_mask = (dates >= "2022-02-01") & (dates < "2023-02-01")
        grid = [
            {"channels": 16, "layers": 2},
            {"channels": 32, "layers": 3},
            {"channels": 64, "layers": 2},
        ]
        best = None
        for cfg in grid:
            m = _train(X[tr_mask], y[tr_mask], channels=cfg["channels"],
                       layers=cfg["layers"], epochs=8, lr=3e-4, batch=args.batch,
                       seed=args.seed, device=dev)
            ic = _val_rankic(m, X, y, dates, va_mask, dev)
            print(f"[tcn] tune {cfg} val RankIC={ic:.4f}")
            if best is None or ic > best[1]:
                best = (cfg, ic)
        chosen.update(best[0])
        chosen["epochs"] = args.epochs
        print(f"[tcn] tuned -> {chosen} (val RankIC={best[1]:.4f})")

    dev = dev0
    t0 = time.time()
    model = _train(X, y, channels=chosen["channels"], layers=chosen["layers"],
                   epochs=chosen["epochs"], lr=chosen["lr"], batch=args.batch,
                   seed=args.seed, device=dev)
    print(f"[tcn] final trained in {time.time()-t0:.0f}s")

    ev = data.load_eval_sequences(window=32)
    Xev = ev["X"]
    pred = np.full(len(Xev), np.nan)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(Xev), 65536):
            chunk = Xev[s:s + 65536]
            ok = np.isfinite(chunk).all(axis=1)
            if ok.any():
                pred[s:s + 65536][ok] = model(
                    torch.from_numpy(chunk[ok].astype(np.float32)).to(dev)).cpu().numpy()

    rows = load_eval_rows()
    assert len(pred) == rows["n_rows"]
    m = full_metrics(pred, rows)
    print(f"[tcn] eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("tcn", pred, meta={
        "features": "return-window(32)", "arch": "TCN(causal dilated conv)",
        "channels": chosen["channels"], "layers": chosen["layers"],
        "epochs": chosen["epochs"], "lr": chosen["lr"], "seed": args.seed})
    save_json(RESULTS / "tcn_metrics.json", m)
    print(f"[tcn] saved -> {path}")


if __name__ == "__main__":
    main()
