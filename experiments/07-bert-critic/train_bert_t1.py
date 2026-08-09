"""T1 (plan §4): scoring-aligned MLM fine-tune of the bidirectional BERT critic.

Continues from ``bert_critic_mlm_v1.pt`` at low LR with the three scoring-shape
alignments (``t1_masking.make_t1_batch``):
  1. final-position masking (50% scoring shape: MASK at the last position,
     no trailing EOS), 2. va-dropout (50%), 3. recency-weighted masking (2x
     over the last 64 positions).  ``--window`` drives the T6 ablation
     (W in {64,128,512}).

Acceptance (plan §4 T1): rerun C-a (must not regress) + rerun R1 (E_BERT[r]
RankIC vs mlm_v1 must improve).  Roll back to mlm_v1 if C-a degrades.

Usage:
    python experiments/07-bert-critic/train_bert_t1.py \
        --init_from checkpoints/bert_critic_mlm_v1.pt \
        --window 512 --epochs 3 --lr 3e-5 \
        --tag t1_w512 --save_path checkpoints/bert_critic_mlm_t1_w512.pt
"""
import argparse
import hashlib
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
from experiment_io import file_sha256
from model import load_tokenizer
from model.kronos_bert import KronosBert
from training_utils import clip_grad_norm_

from t1_masking import make_t1_batch  # noqa: E402

SKELETON = dict(dim=256, depth=6, heads=4, num_kv_heads=1,
                ffn_multiplier=4, dropout=0.1)


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
    print(f"  init={args.init_from}, save={save_path}, ep={args.epochs}, "
          f"lr={args.lr}, window={args.window}")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    print(f"Tokenizer. vocab_coarse={ModelConfig.vocab_size}, "
          f"vocab_fine={ModelConfig.vocab_fine}")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=args.max_seq_len)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)

    model = KronosBert().to(device)
    if args.init_from and Path(args.init_from).exists():
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model_state_dict"])
        print(f"  init weights <- {args.init_from} "
              f"(val_loss={init.get('val_loss'):.4f}, mlm_acc={init.get('mlm_acc'):.3f})")
    else:
        print(f"  WARNING: init_from {args.init_from} missing; training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from epoch {start_epoch}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume: {e}; fresh")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "mlm_acc": []}
    t0 = time.time()

    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))

    def run_one_batch(input_ids, time_id, pos_id, va_val):
        B, N = input_ids.shape
        out_ids, out_time, out_va, out_lab, out_pos = [], [], [], [], []
        for b in range(B):
            mid, mt, mv, ml, mp, _off = make_t1_batch(
                input_ids[b], time_id[b], va_val[b], vocab_base, mask_id,
                window=args.window, mlm_prob=args.mlm_prob,
                corrupt_fracs=corrupt_fracs, final_pos_frac=args.final_pos_frac,
                va_zero_frac=args.va_zero_frac, recency_window=args.recency_window,
                generator=(torch.Generator().manual_seed(global_step + b) if args.deterministic else None))
            out_ids.append(mid); out_time.append(mt); out_va.append(mv)
            out_lab.append(ml); out_pos.append(mp)
        Lmax = max(x.shape[0] for x in out_ids)
        pad_ids = torch.zeros(B, Lmax, dtype=torch.long, device=input_ids.device)
        pad_time = torch.zeros(B, Lmax, 3, dtype=torch.long, device=input_ids.device)
        pad_va = torch.zeros(B, Lmax, 2, dtype=torch.float32, device=input_ids.device)
        pad_lab = torch.full((B, Lmax), -100, dtype=torch.long, device=input_ids.device)
        pad_pos = torch.zeros(B, Lmax, dtype=torch.long, device=input_ids.device)
        for b in range(B):
            L = out_ids[b].shape[0]
            pad_ids[b, :L] = out_ids[b]
            pad_time[b, :L] = out_time[b]
            pad_va[b, :L] = out_va[b]
            pad_lab[b, :L] = out_lab[b]
            pad_pos[b, :L] = out_pos[b]
        return pad_ids, pad_time, pad_va, pad_lab, pad_pos

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, accs = [], []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{args.epochs}")
        for bi, batch in enumerate(pbar):
            input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
            input_ids = input_ids.to(device, non_blocking=True)
            time_id = time_id.to(device, non_blocking=True)
            pos_id = pos_id.to(device, non_blocking=True)
            va_val = va_val.to(device, non_blocking=True)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                time_id = time_id.unsqueeze(0)
                pos_id = pos_id.unsqueeze(0)
                va_val = va_val.unsqueeze(0)

            mlm_ids, mlm_time, mlm_va, mlm_labels, mlm_pos = run_one_batch(
                input_ids, time_id, pos_id, va_val)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mlm_ids, mlm_time, mlm_pos, va_values=mlm_va)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       mlm_labels.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = mlm_labels != -100
                    if valid.any():
                        accs.append((pred[valid] == mlm_labels[valid]).float().mean().item())

            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at step {global_step}")
                continue
            loss.backward()
            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{np.mean(accs[-50:]) if accs else 0:.3f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train = sum(losses) / max(len(losses), 1)
        avg_train_acc = sum(accs) / max(len(accs), 1) if accs else 0.0

        model.eval()
        vlosses, vaccs = [], []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
                input_ids = input_ids.to(device)
                time_id = time_id.to(device)
                pos_id = pos_id.to(device)
                va_val = va_val.to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    time_id = time_id.unsqueeze(0)
                    pos_id = pos_id.unsqueeze(0)
                    va_val = va_val.unsqueeze(0)
                mlm_ids, mlm_time, mlm_va, mlm_labels, mlm_pos = run_one_batch(
                    input_ids, time_id, pos_id, va_val)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mlm_ids, mlm_time, mlm_pos, va_values=mlm_va)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            mlm_labels.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = mlm_labels != -100
                    if valid.any():
                        vaccs.append((pred[valid] == mlm_labels[valid]).float().mean().item())
                vlosses.append(vloss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        avg_val_acc = sum(vaccs) / max(len(vaccs), 1) if vaccs else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mlm_acc"].append(avg_val_acc)
        history["lr"].append(cur_lr)

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
                "mlm_prob": args.mlm_prob, "corrupt_fracs": list(corrupt_fracs),
                "t1": {"window": args.window, "final_pos_frac": args.final_pos_frac,
                       "va_zero_frac": args.va_zero_frac,
                       "recency_window": args.recency_window},
                "init_from": str(args.init_from),
            }, save_path)
            save_tag = "  -> Saved best"

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "global_step": global_step,
                    "tag": args.tag}, ckpt_path)
        print(f"  [{epoch+1}/{args.epochs}] train={avg_train:.4f} val={avg_val:.4f} "
              f"best={best_val:.4f} mlm_acc={avg_val_acc:.3f} lr={cur_lr:.2e} "
              f"elapsed={elapsed:.0f}s{save_tag}", flush=True)

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    meta = {
        "tag": args.tag, "arch": "kronos_bert", "t1": {"window": args.window,
        "final_pos_frac": args.final_pos_frac, "va_zero_frac": args.va_zero_frac,
        "recency_window": args.recency_window},
        "init_from": str(args.init_from),
        "data": {"cutoff_date": DataConfig.cutoff_date,
                 "n_train_seqs": len(train_seqs), "n_val_seqs": len(val_seqs)},
        "train": {"mlm_prob": args.mlm_prob, "corrupt_fracs": list(corrupt_fracs),
                  "lr": args.lr, "weight_decay": args.weight_decay, "bf16": True},
        "result": {"best_val": best_val, "final_mlm_acc": avg_val_acc,
                   "random_baseline_acc": 1.0 / ModelConfig.vocab_size},
        "history": history,
    }
    with open(os.path.join(os.path.dirname(save_path) or ".", f"bert_mlm_{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 scoring-aligned MLM fine-tune")
    parser.add_argument("--init_from", type=str, default="checkpoints/bert_critic_mlm_v1.pt")
    parser.add_argument("--save_path", type=str, default="checkpoints/bert_critic_mlm_t1_w512.pt")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--final_pos_frac", type=float, default=0.5)
    parser.add_argument("--va_zero_frac", type=float, default=0.5)
    parser.add_argument("--recency_window", type=int, default=64)
    parser.add_argument("--tag", type=str, default="t1_w512")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=False)
    main(parser.parse_args())
