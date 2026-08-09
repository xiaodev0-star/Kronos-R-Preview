"""f43_rankft_v2.py — rank-aware BERT fine-tune v2 (calib-first validation).

f21/f26 showed a 150k-row / 2-epoch rank-aware fine-tune did not improve the
fresh-head RankIC.  This v2 uses a STRONGER recipe (300k rows, 3 epochs,
max_stocks=128 to bound GPU) and validates on the CALIB slice FIRST (audit
uids x 2023-02..2024-02, out-of-sample for all heads) — the same discipline
that rejected FUSION/f40/f42.

Pipeline:
  1. fine-tune BERT last block + norm with rank loss + MLM anchor
  2. cache the fine-tuned BERT's hidden for a fit subset (for a fresh head)
     and for the calib slice
  3. train a FRESH rank head on the fine-tuned fit hidden
  4. compare fresh-head RankIC on CALIB vs the frozen-T2 hidden head

Only if the fine-tuned representation improves calib RankIC is it pursued.
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

from posttrain_heads import soft_spearman_loss, MlpRankHead  # noqa: E402
from train_heads import final_fit_split, train_rank_per_date  # noqa: E402
from score_bert import load_bert, build_index  # noqa: E402
from critic_common import resolve_roots, upstream_paths  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, _split_points, write_json_ledger,
)
from f0_scores import rank_pct_per_date  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def daily_ic(rec, field, dense):
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


def _precompute_positions(index, stock_uid, date_key):
    uids = np.asarray(stock_uid)
    dint = np.asarray([int(str(d)[:10].replace("-", "")) for d in date_key])
    n = len(uids)
    positions = np.full(n, -1, dtype=np.int64)
    uniq, inv = np.unique(uids, return_inverse=True)
    for ui, uid in enumerate(uniq):
        p = index.by_uid.get(str(uid))
        if p is None:
            continue
        ref = np.asarray(p["dates_int"], dtype=np.int32)
        here = np.where(inv == ui)[0]
        didx = np.searchsorted(ref, dint[here], side="left")
        ok = (didx < len(ref)) & (ref[didx] == dint[here])
        positions[here] = np.where(ok, didx, -1)
    return positions


def _batch_from_rows(rows, idx_list, index, window, device):
    sel, y, yc = [], [], []
    for i in idx_list:
        uid = str(rows["stock_uid"][i]); date = str(rows["date_key"][i])[:10]
        pos = int(rows["position"][i])
        if pos < 0:
            continue
        b = index.bert_input(uid, date, window, position=pos)
        if b is None:
            continue
        sel.append(b); y.append(float(rows["true_logret"][i]))
        yc.append(int(rows["true_coarse_id"][i]))
    if not sel:
        return None, None, None, None
    n = len(sel)
    max_len = max(b[0].shape[0] for b in sel)
    inp = torch.zeros(n, max_len, dtype=torch.long, device=device)
    tids = torch.zeros(n, max_len, 3, dtype=torch.long, device=device)
    va = torch.zeros(n, max_len, 2, dtype=torch.float32, device=device)
    maskpos = torch.empty(n, dtype=torch.long, device=device)
    for j, (ids, ti, v, _) in enumerate(sel):
        L = ids.shape[0]
        inp[j, :L] = ids; tids[j, :L] = ti; va[j, :L] = v; maskpos[j] = L - 1
    return inp, tids, va, (np.asarray(y, np.float32),
                           torch.as_tensor(yc, device=device), maskpos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mlm_w", type=float, default=0.3)
    ap.add_argument("--lr_block", type=float, default=2e-4)
    ap.add_argument("--max_stocks", type=int, default=128)
    ap.add_argument("--window", type=int, default=512)
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    wr = weights_root()
    rr = results_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_bert(str(ROOT / "checkpoints" / "bert_critic_mlm_t2.pt"),
                              torch.device(device))
    model.train()
    for name, p in model.named_parameters():
        if (name.startswith("blocks.") and name.split(".")[1] == str(cfg.depth - 1)) \
                or name == "norm.weight":
            p.requires_grad = True
        else:
            p.requires_grad = False
    head = nn.Sequential(nn.Linear(int(cfg.dim), 64), nn.SiLU(), nn.Linear(64, 1)).to(device)
    trainable = [p for p in list(model.parameters()) + list(head.parameters())
                 if p.requires_grad]
    print(f"[f43] trainable: {sum(p.numel() for p in trainable)}")

    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret", "true_coarse_id")}
    _, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=roots.weights_root / "bert_input_index.pkl")
    rows["position"] = _precompute_positions(index, rows["stock_uid"], rows["date_key"])
    # subsample to first max_rows
    if args.max_rows:
        for k in rows:
            rows[k] = rows[k][: args.max_rows]
    groups = {}
    for i, d in enumerate(np.asarray([str(d)[:10] for d in rows["date_key"]])):
        groups.setdefault(d, []).append(i)
    dates = sorted(groups.keys())
    val_dates = set(d for d in dates if "2022-02-01" <= d < "2023-02-01")
    train_dates = [d for d in dates if d not in val_dates]
    print(f"[f43] train dates={len(train_dates)} val dates={len(val_dates)}")

    opt = torch.optim.AdamW(trainable, lr=args.lr_block, weight_decay=0.01)
    loss_f = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(dlist, train=True):
        model.train() if train else model.eval()
        head.train() if train else head.eval()
        tot_r = tot_m = 0.0; ns = 0
        for d in dlist:
            idx = groups[d]
            if args.max_stocks > 0 and len(idx) > args.max_stocks:
                rng = np.random.RandomState(abs(hash(d)) % (2**31))
                idx = [idx[i] for i in rng.permutation(len(idx))[:args.max_stocks]]
            inp, tids, va, extra = _batch_from_rows(rows, idx, index, args.window, device)
            if inp is None or inp.shape[0] < 2:
                continue
            y, yc, maskpos = extra
            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=(device == "cuda"),
                                        dtype=torch.bfloat16):
                    x = model._embed(inp, tids, va)
                    sin, cos = model.rotary(
                        torch.arange(inp.shape[1], device=device).unsqueeze(0)
                        .expand(inp.shape[0], -1))
                    x = model._run_blocks(x, sin, cos, attn_mask=None)
                    x = model.norm(x)
                    mp = maskpos.view(-1, 1, 1).expand(inp.shape[0], 1, x.shape[-1])
                    h = torch.gather(x, 1, mp).squeeze(1)
                    sc = head(h)
                t = torch.from_numpy(np.argsort(np.argsort(y)).astype(np.float32)).to(device)
                loss_rank = soft_spearman_loss(sc, t, tau=1.0)
                logits = model.head_coarse(h)
                loss_mlm = loss_f(logits.float(), yc)
                loss = loss_rank + args.mlm_w * loss_mlm
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            tot_r += loss_rank.item(); tot_m += loss_mlm.item(); ns += 1
        return tot_r / max(1, ns), tot_m / max(1, ns)

    for ep in range(args.epochs):
        tr_r, tr_m = run_epoch(train_dates, train=True)
        va_r, va_m = run_epoch(val_dates, train=False)
        print(f"[f43] ep{ep} train_rank={tr_r:.4f} mlm={tr_m:.3f} "
              f"val_rank={va_r:.4f} mlm={va_m:.3f}", flush=True)

    # save fine-tuned BERT
    ckpt = torch.load(str(ROOT / "checkpoints" / "bert_critic_mlm_t2.pt"),
                      map_location="cpu", weights_only=False)
    ckpt["model_state_dict"] = model.cpu().state_dict()
    torch.save(ckpt, ROOT / "checkpoints" / "bert_critic_rankft_v2.pt")
    print("[f43] saved fine-tuned BERT -> bert_critic_rankft_v2.pt")


if __name__ == "__main__":
    main()
