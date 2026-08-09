"""f21_rank_aware_bert.py — partial-unfreeze fine-tune of BERT for ranking.

The frozen-BERT hidden gives the dominant rank signal (ens6 ~0.0815).  This
fine-tunes the LAST transformer block + final norm (plus a new rank head) with a
soft-Spearman rank loss over per-date cross-sections, anchored by a small MLM
loss at the MASK position (predict the true next-day coarse token).  The rest of
BERT (embed, blocks[:-1], head_coarse) is frozen.

Gates:
  - val rank loss on a held-out slice (R2) must improve vs the frozen baseline;
  - val MLM loss must not explode (representation anchor).

Outputs:
  checkpoints/bert_critic_rankft.pt  (full BERT state, for hidden caching)
  weights_root/head_rankft.pt        (the trained rank head)
"""
from __future__ import annotations

import argparse
import sys
import time
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

from posttrain_heads import soft_spearman_loss  # noqa: E402
from score_bert import load_bert, build_index, VOCAB_BASE  # noqa: E402
from critic_common import resolve_roots, upstream_paths  # noqa: E402


class RankHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))
        self.loss = "soft_spearman"

    def forward(self, h):
        return self.net(h).squeeze(-1)


def _group_by_date(rows, max_rows=0):
    """Group fit-row indices by date_key (preserving row order within date)."""
    dates = np.asarray(rows["date_key"])
    groups = {}
    n = len(dates)
    stop = min(n, max_rows) if max_rows else n
    for i in range(stop):
        d = str(dates[i])[:10]
        groups.setdefault(d, []).append(i)
    return groups


def _precompute_positions(index, stock_uid, date_key):
    """Vectorized per-row position lookup (mirrors cache_bert_hidden)."""
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


def _batch_from_rows(rows, idx_list, index, window=512, device="cuda"):
    """Build a padded batch for one date; returns tensors, keep masks, y, yc."""
    sel = []
    y = []
    yc = []
    for i in idx_list:
        uid = str(rows["stock_uid"][i])
        date = str(rows["date_key"][i])[:10]
        pos = int(rows["position"][i])
        if pos < 0:
            continue
        b = index.bert_input(uid, date, window, position=pos)
        if b is None:
            continue
        sel.append(b)
        y.append(float(rows["true_logret"][i]))
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
        inp[j, :L] = ids
        tids[j, :L] = ti
        va[j, :L] = v
        maskpos[j] = L - 1
    y = np.asarray(y, dtype=np.float32)
    yc = torch.as_tensor(yc, device=device, dtype=torch.long)
    return inp, tids, va, (y, yc, maskpos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default=str(ROOT / "checkpoints" / "bert_critic_mlm_t2.pt"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--lr_block", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--mlm_w", type=float, default=0.1)
    ap.add_argument("--batch_dates", type=int, default=1)
    ap.add_argument("--max_stocks", type=int, default=0,
                    help="random-subset size per date (bounds batch; 0 = all)")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_bert(Path(args.bert), torch.device(device))
    model.train()
    print(f"[f21] BERT dim={cfg.dim} depth={cfg.depth} on {device}")

    # freeze everything except last block + norm
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and name.split(".")[1] == str(cfg.depth - 1):
            p.requires_grad = True
        elif name == "norm.weight":
            p.requires_grad = True
        else:
            p.requires_grad = False
    head = RankHead(dim=int(cfg.dim)).to(device)

    trainable = [p for p in list(model.parameters()) + list(head.parameters())
                 if p.requires_grad]
    print(f"[f21] trainable params: {sum(p.numel() for p in trainable)}")

    # data
    tc = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
                 / "training_cache.npz", allow_pickle=True)
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret", "true_coarse_id")}
    groups = _group_by_date(rows, max_rows=args.max_rows)
    dates = sorted(groups.keys())
    # split: R2-style val = dates in [2022-02-01, 2023-02-01)
    val_dates = set(d for d in dates if "2022-02-01" <= d < "2023-02-01")
    train_dates = [d for d in dates if d not in val_dates]
    print(f"[f21] train dates={len(train_dates)} val dates={len(val_dates)}")

    _, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=roots.weights_root / "bert_input_index.pkl")
    rows["position"] = _precompute_positions(index, rows["stock_uid"], rows["date_key"])
    print(f"[f21] positions: valid={int((rows['position'] >= 0).sum())}/{len(rows['position'])}")

    opt = torch.optim.AdamW(trainable, lr=args.lr_block, weight_decay=0.01)
    # separate LR for head via param groups is complex with frozen list; use one LR
    loss_f = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(date_list, train=True):
        model.train() if train else model.eval()
        head.train() if train else head.eval()
        tot_rank = 0.0
        tot_mlm = 0.0
        tot_steps = 0
        for d in date_list:
            idx = groups[d]
            if args.max_stocks > 0 and len(idx) > args.max_stocks:
                rng = np.random.RandomState(hash(d) % (2**32))
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
                    x = model.norm(x)                      # [B, L, dim]
                    mp = maskpos.view(-1, 1, 1).expand(inp.shape[0], 1, x.shape[-1])
                    h = torch.gather(x, 1, mp).squeeze(1)  # [B, dim]
                    sc = head(h)                            # [B]
                t = torch.from_numpy(np.argsort(np.argsort(y)).astype(np.float32)).to(device)
                loss_rank = soft_spearman_loss(sc, t, tau=1.0)
                # MLM anchor at the MASK position (predict true coarse token)
                logits = model.head_coarse(h)              # [B, vocab]
                loss_mlm = loss_f(logits.float(), yc)
                loss = loss_rank + args.mlm_w * loss_mlm
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
            tot_rank += loss_rank.item()
            tot_mlm += loss_mlm.item()
            tot_steps += 1
            if tot_steps % 200 == 0:
                print(f"[f21] {'tr' if train else 'va'} step {tot_steps} "
                      f"rank={loss_rank.item():.4f} mlm={loss_mlm.item():.3f}", flush=True)
        return tot_rank / max(1, tot_steps), tot_mlm / max(1, tot_steps), tot_steps

    for ep in range(args.epochs):
        t0 = time.time()
        tr_r, tr_m, _ = run_epoch(train_dates, train=True)
        va_r, va_m, va_n = run_epoch(val_dates, train=False)
        print(f"[f21] ep{ep}: train_rank={tr_r:.4f} train_mlm={tr_m:.3f} "
              f"val_rank={va_r:.4f} val_mlm={va_m:.3f} ({time.time()-t0:.0f}s)")

    # save
    ckpt = torch.load(str(Path(args.bert)), map_location="cpu", weights_only=False)
    ckpt["model_state_dict"] = model.cpu().state_dict()
    out_pt = ROOT / "checkpoints" / "bert_critic_rankft.pt"
    torch.save(ckpt, out_pt)
    torch.save({"head_state": head.cpu().state_dict(),
                "recipe": {"lr_block": args.lr_block, "lr_head": args.lr_head,
                           "epochs": args.epochs, "mlm_w": args.mlm_w}},
               ROOT / "server_runs" / "weights" / "07-bert-critic" / "seed42"
               / "head_rankft.pt")
    print(f"[f21] saved {out_pt} + head_rankft.pt")


if __name__ == "__main__":
    main()
