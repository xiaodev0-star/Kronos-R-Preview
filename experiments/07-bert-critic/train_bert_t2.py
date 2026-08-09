"""train_bert_t2.py — plan §4 T2: GPT-noise denoising fine-tune (user ask).

Exposes the frozen GPT's per-position proposals to the critic during training:
on top of the T1 scoring-aligned masking (final-position / va-dropout / recency),
~15% of non-labeled history day tokens are REPLACED by samples from the frozen
GPT at that position (``softmax(proposal/temp)``, temp=1.4 = the candidate
temperature).  The MLM labels still recover the REAL tokens at the T1-masked
positions, so the model learns to estimate distributions over contexts that
contain model-味 noise — the shape it faces at scoring time and, more
importantly, in multi-step rollouts.

Negatives come from a cached proposal bank (``cache_gpt_proposals.py``) —
one GPT snapshot per cache; for the anti-mirroring rule the plan §6.2 mixes ep1
+ ep100, so T2 may pass TWO caches and alternate per batch.

Continues from the T1 product (or mlm_v1) at low LR, <=3 epochs.

Usage:
    python experiments/07-bert-critic/cache_gpt_proposals.py --which ep100
    python experiments/07-bert-critic/train_bert_t2.py \
        --init_from checkpoints/bert_critic_mlm_t1_w512.pt \
        --proposals1 checkpoints/gpt_proposals_ep100.pt \
        --save_path checkpoints/bert_critic_mlm_t2.pt \
        --tag t2 --epochs 3 --lr 3e-5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from model import load_tokenizer
from model.kronos_bert import KronosBert
from training_utils import clip_grad_norm_

from t1_masking import make_t1_batch  # noqa: E402

SKELETON = dict(dim=256, depth=6, heads=4, num_kv_heads=1,
                ffn_multiplier=4, dropout=0.1)
NEG_TEMP = 1.4


def make_t2_batch(real_ids, time_id, va, vocab_base, mask_id, proposal,
                  window, mlm_prob, corrupt_fracs, final_pos_frac, va_zero_frac,
                  recency_window, gpt_replace_prob=0.15, temp=NEG_TEMP,
                  generator=None):
    """T1 masking + GPT-noise replacement on non-labeled history day tokens.

    ``proposal`` is [N, V] logits for the FULL sequence (cached).  ``offset``
    from make_t1_batch aligns the truncated window to the proposal slice.
    GPT-replaced rows get va=0 (the rollout shape: a model-generated token has
    unknown future volume/amount).  Returns (mid, mt, mv, ml, mp).
    """
    mid, mt, mv, ml, mp, off = make_t1_batch(
        real_ids, time_id, va, vocab_base, mask_id, window, mlm_prob,
        corrupt_fracs, final_pos_frac, va_zero_frac, recency_window, generator)
    L = mid.shape[0]
    if proposal is not None:
        noise = (ml == -100) & (mid >= 0) & (mid < vocab_base) & (mid != mask_id)
        if noise.any():
            # proposal[pos] (cached, unshifted GPT logits) predicts token pos+1;
            # the replacement for window position j (= full-sequence off+j) uses
            # proposal[off+j-1].  Same shift as ELECTRA's gpt_proposal_for_sequence.
            prop_pos = np.arange(off, off + L) - 1
            prop_pos[0] = off                      # BOS has no proposal -> reuse
            prop_slice = np.asarray(proposal[prop_pos], dtype=np.float32)
            gen_r = torch.rand(L, device=mid.device, generator=generator)
            do_replace = noise & (gen_r < gpt_replace_prob)
            if do_replace.any():
                idx = torch.nonzero(do_replace, as_tuple=False).squeeze(-1)
                prop = torch.from_numpy(prop_slice[idx.cpu().numpy()]).to(mid.device)
                p = torch.softmax(prop / temp, dim=-1)
                sample = torch.multinomial(p, 1).squeeze(-1)
                mid[idx] = sample
                mv[idx] = 0.0          # rollout shape: model token -> va unknown
    return mid, mt, mv, ml, mp


def main(args):
    set_global_seed(TrainingConfig.random_seed,
                    deterministic=getattr(args, "deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    ckpt_path = save_path + ".ckpt"

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks
    for k, v in SKELETON.items():
        setattr(ModelConfig, k, v)

    print(f"Device: {device}, tag={args.tag}")
    print(f"  init={args.init_from}, save={save_path}, ep={args.epochs}, lr={args.lr}")

    # ---- proposal banks ----
    def _load_proposals(path):
        if not Path(path).exists():
            raise RuntimeError(f"proposal cache missing: {path} "
                               "(run cache_gpt_proposals.py first)")
        c = torch.load(path, map_location="cpu", weights_only=False)
        return {"proposals": c["proposals"].numpy(),      # [n_seqs, max_len, V] fp16
                "lengths": c["lengths"].numpy(),
                "ckpt": c["ckpt"]}
    prop1 = _load_proposals(args.proposals1)
    prop2 = _load_proposals(args.proposals2) if args.proposals2 else None
    print(f"  proposals1: {prop1['ckpt']}")
    if prop2:
        print(f"  proposals2: {prop2['ckpt']} (anti-mirroring mix)")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=args.max_seq_len)
    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")
    # ensure the proposal bank matches the sequence order
    if prop1["lengths"].shape[0] != len(train_seqs):
        print(f"  WARNING: proposal bank ({prop1['lengths'].shape[0]}) != train seqs "
              f"({len(train_seqs)}) — assuming the same order from the same cache.")

    model = KronosBert().to(device)
    if args.init_from and Path(args.init_from).exists():
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model_state_dict"])
        print(f"  init weights <- {args.init_from} "
              f"(val_loss={init.get('val_loss'):.4f})")
    else:
        print(f"  WARNING: init_from {args.init_from} missing; from scratch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_updates = len(train_loader) * args.epochs
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16

    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_val = ck.get("best_val", float("inf"))
            global_step = ck.get("global_step", 0)
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume: {e}; fresh")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "mlm_acc": []}
    t0 = time.time()
    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))

    def run_one_batch(batch, seq_i):
        input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
        input_ids = input_ids.to(device); time_id = time_id.to(device)
        pos_id = pos_id.to(device); va_val = va_val.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0); time_id = time_id.unsqueeze(0)
            pos_id = pos_id.unsqueeze(0); va_val = va_val.unsqueeze(0)
        B, N = input_ids.shape
        out_ids, out_time, out_va, out_lab, out_pos = [], [], [], [], []
        for b in range(B):
            prop = prop1["proposals"][seq_i + b]
            if prop2 is not None and (seq_i + b) % 2 == 1:
                prop = prop2["proposals"][seq_i + b]
            mid, mt, mv, ml, mp = make_t2_batch(
                input_ids[b], time_id[b], va_val[b], vocab_base, mask_id, prop,
                window=args.window, mlm_prob=args.mlm_prob,
                corrupt_fracs=corrupt_fracs, final_pos_frac=args.final_pos_frac,
                va_zero_frac=args.va_zero_frac, recency_window=args.recency_window,
                gpt_replace_prob=args.gpt_replace_prob, temp=NEG_TEMP,
                generator=(torch.Generator().manual_seed(global_step + b)
                           if args.deterministic else None))
            out_ids.append(mid); out_time.append(mt); out_va.append(mv)
            out_lab.append(ml); out_pos.append(mp)
        Lmax = max(x.shape[0] for x in out_ids)
        pad_ids = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        pad_time = torch.zeros(B, Lmax, 3, dtype=torch.long, device=device)
        pad_va = torch.zeros(B, Lmax, 2, dtype=torch.float32, device=device)
        pad_lab = torch.full((B, Lmax), -100, dtype=torch.long, device=device)
        pad_pos = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        for b in range(B):
            L = out_ids[b].shape[0]
            pad_ids[b, :L] = out_ids[b]; pad_time[b, :L] = out_time[b]
            pad_va[b, :L] = out_va[b]; pad_lab[b, :L] = out_lab[b]
            pad_pos[b, :L] = out_pos[b]
        return pad_ids, pad_time, pad_va, pad_lab, pad_pos

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, accs = [], []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{args.epochs}")
        for bi, batch in enumerate(pbar):
            mid, mt, mv, ml, mp = run_one_batch(batch, bi)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mid, mt, mp, va_values=mv)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       ml.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = ml != -100
                    if valid.any():
                        accs.append((pred[valid] == ml[valid]).float().mean().item())
            if not torch.isfinite(loss):
                print(f"  [skip] non-finite at step {global_step}")
                continue
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            scheduler.step(); global_step += 1
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{np.mean(accs[-50:]) if accs else 0:.3f}"})

        avg_train = float(np.mean(losses))
        model.eval()
        vlosses, vaccs = [], []
        with torch.inference_mode():
            for bi, batch in enumerate(val_loader):
                mid, mt, mv, ml, mp = run_one_batch(batch, bi)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mid, mt, mp, va_values=mv)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            ml.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = ml != -100
                    if valid.any():
                        vaccs.append((pred[valid] == ml[valid]).float().mean().item())
                vlosses.append(vloss.item())
        avg_val = float(np.mean(vlosses))
        avg_val_acc = float(np.mean(vaccs)) if vaccs else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mlm_acc"].append(avg_val_acc)
        save_tag = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
                           "dropout": ModelConfig.dropout,
                           "vocab_size": ModelConfig.vocab_size,
                           "vocab_fine": ModelConfig.vocab_fine,
                           "mask_id": mask_id, "va_hidden_dim": ModelConfig.va_hidden_dim,
                           "rope_base": ModelConfig.rope_base, "arch": "kronos_bert"},
                "val_loss": best_val, "mlm_acc": avg_val_acc, "epoch": epoch,
                "completed": epoch == args.epochs - 1, "tag": args.tag,
                "t2": {"gpt_replace_prob": args.gpt_replace_prob, "temp": NEG_TEMP,
                       "window": args.window, "final_pos_frac": args.final_pos_frac,
                       "va_zero_frac": args.va_zero_frac,
                       "recency_window": args.recency_window},
                "proposals1": str(args.proposals1),
                "proposals2": str(args.proposals2),
                "init_from": str(args.init_from),
            }, save_path)
            save_tag = "  -> Saved best"
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "global_step": global_step,
                    "tag": args.tag}, ckpt_path)
        print(f"  [{epoch+1}/{args.epochs}] train={avg_train:.4f} val={avg_val:.4f} "
              f"best={best_val:.4f} mlm_acc={avg_val_acc:.3f} elapsed={time.time()-t0:.0f}s"
              f"{save_tag}", flush=True)

    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location="cpu", weights_only=False)
        ck["completed"] = True
        torch.save(ck, save_path)
    meta = {"tag": args.tag, "arch": "kronos_bert",
            "t2": {"gpt_replace_prob": args.gpt_replace_prob, "temp": NEG_TEMP},
            "init_from": str(args.init_from), "proposals1": str(args.proposals1),
            "proposals2": str(args.proposals2), "history": history,
            "result": {"best_val": best_val, "final_mlm_acc": avg_val_acc}}
    with open(os.path.join(os.path.dirname(save_path) or ".",
                           f"bert_mlm_{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2 GPT-noise denoising fine-tune")
    parser.add_argument("--init_from", type=str, default="checkpoints/bert_critic_mlm_t1_w512.pt")
    parser.add_argument("--proposals1", type=str, default="checkpoints/gpt_proposals_ep100.pt")
    parser.add_argument("--proposals2", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="checkpoints/bert_critic_mlm_t2.pt")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--final_pos_frac", type=float, default=0.5)
    parser.add_argument("--va_zero_frac", type=float, default=0.5)
    parser.add_argument("--recency_window", type=int, default=64)
    parser.add_argument("--gpt_replace_prob", type=float, default=0.15)
    parser.add_argument("--tag", type=str, default="t2")
    parser.add_argument("--deterministic", action="store_true", default=False)
    main(parser.parse_args())
