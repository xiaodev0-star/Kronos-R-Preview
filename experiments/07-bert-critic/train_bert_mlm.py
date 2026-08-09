"""Stage 1: BERT MLM pretraining over real pre-cutoff history (plan §6.1).

Uses the same packed v2 sequences as GPT CPT (pack_stocks_v2, split_stocks
train split, ALL pre-cutoff), with F1 vocab wiring and F2 80/10/10 corruption
(fixed in root train_bert.py / model/kronos_bert.py) plus ONE critic-specific
addition: the [MASK] rows get va_values forced to 0 during training, so the
model learns "masked position => today's volume/amount unknown => 0" — exactly
the input shape the critic faces at scoring time (§8.1 leakage red line).

Model = KronosBert with the GPT-identical skeleton (dim256/depth6/heads4/
GQA-1/ffn4/dropout0.1, ~6M params).  The critic's value is error decorrelation,
not capacity; the 16M big config is a v2 control arm only.

Usage:
    python experiments/07-bert-critic/train_bert_mlm.py \
        --save_path checkpoints/bert_critic_mlm_v1.pt \
        --mlm_prob 0.15 --epochs 6
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
from model.kronos_bert import KronosBert, make_mlm_batch
from training_utils import clip_grad_norm_


# GPT-identical skeleton (v1 default; big is the v2 control arm).
SKELETON = dict(dim=256, depth=6, heads=4, num_kv_heads=1,
                ffn_multiplier=4, dropout=0.1)


def data_fingerprint(cache_dir, tokenizer_path, mlm_prob, epochs, max_seq_len):
    """Short fingerprint of the Stage-1 data + masking configuration."""
    h = hashlib.sha256()
    for item in (str(cache_dir), str(tokenizer_path), str(mlm_prob),
                 str(epochs), str(max_seq_len), "80_10_10", "mask_va_zero"):
        h.update(item.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


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
    if args.mlm_prob > 0:
        pass  # mlm_prob passed explicitly to make_mlm_batch

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}, ep={args.epochs}, mlm_prob={args.mlm_prob}")
    print(f"  model: dim={ModelConfig.dim} depth={ModelConfig.depth} heads={ModelConfig.heads} "
          f"num_kv_heads={ModelConfig.num_kv_heads} ffn_mult={ModelConfig.ffn_multiplier} "
          f"dropout={ModelConfig.dropout} (GPT-identical skeleton, v1)")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    # F1: wire vocab from the tokenizer (must precede model construction).
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    print(f"Tokenizer loaded. vocab_coarse={ModelConfig.vocab_size}, "
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: enabled")

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

    # ---- resume ----
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
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}, step={global_step}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume (incompatible): {e}; starting fresh.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "mlm_acc": []}
    t0 = time.time()

    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    # F2 80/10/10 corruption; every selected position is a label.
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))
    # Stage-1 critic addition: [MASK] rows carry va=0 (leakage-safe scoring shape).
    zero_mask_va = not args.no_mask_va_zero

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

            B, N = input_ids.shape
            mlm_ids_list, mlm_labels_list = [], []
            for b in range(B):
                mlm_ids_b, mlm_labels_b = make_mlm_batch(
                    input_ids[b], vocab_base, mask_id, mlm_prob=args.mlm_prob,
                    corrupt_fracs=corrupt_fracs)
                mlm_ids_list.append(mlm_ids_b)
                mlm_labels_list.append(mlm_labels_b)
            mlm_ids = torch.stack(mlm_ids_list, dim=0)
            mlm_labels = torch.stack(mlm_labels_list, dim=0)

            va_mlm = va_val.clone()
            if zero_mask_va:
                va_mlm = va_mlm.masked_fill(
                    (mlm_ids == mask_id).unsqueeze(-1), 0.0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mlm_ids, time_id, pos_id, va_values=va_mlm)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       mlm_labels.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = (mlm_labels != -100)
                    if valid.any():
                        accs.append((pred[valid] == mlm_labels[valid]).float().mean().item())

            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at step {global_step}; continuing")
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

        # Validation
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
                B, N = input_ids.shape
                mlm_ids_list, mlm_labels_list = [], []
                for b in range(B):
                    mlm_ids_b, mlm_labels_b = make_mlm_batch(
                        input_ids[b], vocab_base, mask_id, mlm_prob=args.mlm_prob,
                        corrupt_fracs=corrupt_fracs)
                    mlm_ids_list.append(mlm_ids_b)
                    mlm_labels_list.append(mlm_labels_b)
                mlm_ids = torch.stack(mlm_ids_list, dim=0)
                mlm_labels = torch.stack(mlm_labels_list, dim=0)
                va_mlm = va_val.clone()
                if zero_mask_va:
                    va_mlm = va_mlm.masked_fill(
                        (mlm_ids == mask_id).unsqueeze(-1), 0.0)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mlm_ids, time_id, pos_id, va_values=va_mlm)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            mlm_labels.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = (mlm_labels != -100)
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
                           "mask_id": mask_id,
                           "va_hidden_dim": ModelConfig.va_hidden_dim,
                           "rope_base": ModelConfig.rope_base,
                           "arch": "kronos_bert"},
                "val_loss": best_val,
                "mlm_acc": avg_val_acc,
                "epoch": epoch,
                "completed": epoch == args.epochs - 1,
                "tag": args.tag,
                "mlm_prob": args.mlm_prob,
                "corrupt_fracs": list(corrupt_fracs),
                "mask_va_zero": zero_mask_va,
            }, save_path)
            save_tag = "  -> Saved best"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "best_val": best_val, "global_step": global_step,
            "tag": args.tag,
        }, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = args.epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        print(f"  [{epochs_done}/{args.epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"mlm_acc={avg_val_acc:.3f} lr={cur_lr:.2e} elapsed={elapsed:.0f}s "
              f"ETA={eta:.0f}s step={global_step}{save_tag}", flush=True)

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    # ---- sidecar fingerprint + history ----
    meta = {
        "tag": args.tag,
        "arch": "kronos_bert",
        "data": {
            "cache_dir": cache_dir,
            "tokenizer_path": str(args.tokenizer_path),
            "tokenizer_sha256": file_sha256(Path(args.tokenizer_path)),
            "cutoff_date": DataConfig.cutoff_date,
            "max_stocks": int(DataConfig.max_stocks),
            "max_seq_len": int(args.max_seq_len),
            "mode": "train",   # pack_stocks_v2 mode="train" => all pre-cutoff
            "n_train_seqs": len(train_seqs),
            "n_val_seqs": len(val_seqs),
            "fingerprint": data_fingerprint(cache_dir, args.tokenizer_path,
                                            args.mlm_prob, args.epochs, args.max_seq_len),
        },
        "model": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                  "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                  "ffn_multiplier": ModelConfig.ffn_multiplier,
                  "dropout": ModelConfig.dropout,
                  "vocab_size": ModelConfig.vocab_size,
                  "vocab_fine": ModelConfig.vocab_fine, "n_params": int(n_params)},
        "train": {"mlm_prob": args.mlm_prob, "corrupt_fracs": list(corrupt_fracs),
                  "mask_va_zero": zero_mask_va, "lr": args.lr,
                  "weight_decay": args.weight_decay, "warmup_ratio": 0.05,
                  "scheduler": "cosine", "grad_clip": 1.0, "bf16": True},
        "result": {"best_val": best_val, "final_val": avg_val,
                   "final_mlm_acc": avg_val_acc,
                   "random_baseline_acc": 1.0 / ModelConfig.vocab_size},
        "history": history,
    }
    with open(os.path.join(os.path.dirname(save_path) or ".",
                           f"bert_mlm_{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f} "
          f"(random baseline {1.0/ModelConfig.vocab_size:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: BERT MLM pretraining")
    parser.add_argument("--save_path", type=str, default="checkpoints/bert_critic_mlm_v1.pt")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max_stocks", type=int, default=0, help="0 = all 4695")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15,
                        help="main 0.15; control arm 0.25")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1",
                        help="F2 80/10/10 corruption split")
    parser.add_argument("--no_mask_va_zero", action="store_true",
                        help="disable zeroing va on [MASK] rows (NOT the default; "
                             "the default matches the scoring-time leakage-safe shape)")
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=False)
    main(parser.parse_args())
