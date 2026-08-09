"""train_timesfm.py — TimeFM baseline.

Two modes:
  1) --use-real   Use the PRETRAINED TimesFM (Google TimesFM 2.5 200M torch).
                  This requires a downloaded checkpoint.  In this environment
                  HuggingFace is blocked and the Google Storage URL is gone, so
                  the checkpoint is NOT available here; the code path is kept
                  functional for when a checkpoint is provided locally
                  (--checkpoint <path>).
  2) default      TimesFM-STYLE fallback: a patch-based causal time-series
                  transformer (TimesFM's core idea: fixed-length patching +
                  transformer) trained from scratch on OUR fit region, so the
                  comparison is still meaningful.  Architecturally distinct
                  from the plain Transformer baseline (which uses per-timestep
                  embeddings; this one uses patching).

The predicted next-day log return is the row score (same contract as the
others).  Everything is documented in the output meta.

Usage:
    python train_timesfm.py                          # TimesFM-style (trained)
    python train_timesfm.py --use-real               # needs a pretrained ckpt
    python train_timesfm.py --use-real --checkpoint path/to/model.safetensors
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import common
import data
from common import load_eval_rows, full_metrics, save_predictions, save_json, RESULTS, CHECKPOINTS


# ---------------------------------------------------------------------------
# TimesFM-style patch transformer (fallback)
# ---------------------------------------------------------------------------
class PatchTimeSeriesTransformer(nn.Module):
    """TimesFM-style: patch the window, embed each patch, causal transformer,
    predict the next return from the final patch's representation."""

    def __init__(self, window=32, patch_len=4, dim=64, layers=2, heads=4,
                 dropout=0.1):
        super().__init__()
        assert window % patch_len == 0, "window must be a multiple of patch_len"
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.patch_embed = nn.Linear(patch_len, dim)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: [B, W]
        B, W = x.shape
        xp = x.view(B, self.n_patches, self.patch_len)   # [B, nP, patch_len]
        h = self.patch_embed(xp) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def _train_style(X, y, *, window, patch_len, dim, layers, heads, epochs,
                 lr, batch, seed, device):
    torch.manual_seed(seed)
    model = PatchTimeSeriesTransformer(window=window, patch_len=patch_len,
                                       dim=dim, layers=layers, heads=heads,
                                       dropout=0.1).to(device)
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
            print(f"[tfm] ep{ep} loss={tot/max(1,ns):.4f}")
    return model


def _run_real_timesfm(windows, checkpoint=None, context_len=512, batch=256):
    """Use the pretrained TimesFM (requires a local checkpoint or HF access)."""
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    model = TimesFM_2p5_200M_torch(
        torch_compile=False,
        config=ForecastConfig(max_context=context_len, max_horizon=4,
                              normalize_inputs=True, per_core_batch_size=batch))
    if checkpoint:
        model.load_checkpoint(str(checkpoint))
    else:
        model.from_pretrained("google/timesfm-2.5-200m-pytorch")
    # forecast horizon=1 for every window; chunked
    n = len(windows)
    out = np.full(n, np.nan)
    for s in range(0, n, 2000):
        chunk = [w.astype(np.float32) for w in windows[s:s + 2000]]
        pt, _qt = model.forecast(horizon=1, inputs=chunk)
        out[s:s + 2000] = np.asarray(pt)[:, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-real", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--patch-len", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ev = data.load_eval_sequences(window=args.window)
    rows = load_eval_rows()
    assert len(ev["X"]) == rows["n_rows"]

    if args.use_real:
        print("[tfm] using PRETRAINED TimesFM...")
        try:
            pred = _run_real_timesfm(ev["X"], checkpoint=args.checkpoint)
        except Exception as e:  # noqa: BLE001
            print(f"[tfm] real TimesFM unavailable ({e}); "
                  "falling back to TimesFM-style trained model")
            args.use_real = False
            pred = None
        if pred is not None:
            m = full_metrics(pred, rows)
            save_predictions("timesfm_real", pred, meta={"mode": "pretrained"})
            save_json(RESULTS / "timesfm_real_metrics.json", m)
            print(f"[tfm] real TimesFM eval RankIC={m['avg_daily_rank_ic']:.4f}")
            return

    print("[tfm] TimesFM-style fallback (patch transformer, trained from scratch)...")
    fit = data.load_fit_sequences(window=args.window)
    X, y = fit["X"], fit["y"]
    fin = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[fin], y[fin]
    print(f"[tfm] fit rows: {len(y)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = _train_style(X, y, window=args.window, patch_len=args.patch_len,
                         dim=args.dim, layers=args.layers, heads=4,
                         epochs=args.epochs, lr=args.lr, batch=args.batch,
                         seed=args.seed, device=dev)
    print(f"[tfm] trained in {time.time()-t0:.0f}s")

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

    m = full_metrics(pred, rows)
    print(f"[tfm] TimesFM-style eval RankIC={m['avg_daily_rank_ic']:.4f} "
          f"DA={m['avg_da_per_date']:.4f} MAPE={m['avg_mape']:.4f}")

    path = save_predictions("timesfm", pred, meta={
        "mode": "timesfm_style_patch_transformer_trained",
        "note": "pretrained TimesFM checkpoint unavailable (HF blocked); "
                "used TimesFM-style patch-transformer trained on fit region",
        "window": args.window, "patch_len": args.patch_len, "dim": args.dim,
        "layers": args.layers, "epochs": args.epochs, "lr": args.lr,
        "batch": args.batch, "seed": args.seed})
    save_json(RESULTS / "timesfm_metrics.json", m)
    print(f"[tfm] saved -> {path}")


if __name__ == "__main__":
    main()
