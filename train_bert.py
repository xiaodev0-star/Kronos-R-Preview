"""Train KronosBert: bidirectional MLM calibrator for KronosPreview.

Reuses the same packed v2 sequences (BOS + tokens + EOS with va_values) but adds MLM masking.
Bidirectional attention → every position can see every other position → better for validating
GPT's next-token predictions in calibration (the predicted token is treated as "future" context).

Usage:
    python train_bert.py --epochs 5 --max_stocks 500 --max_seq_len 2048 --tag bert_v1
"""
import argparse
import math
import os
import time
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from model import load_tokenizer
from model.kronos_bert import KronosBert, make_mlm_batch


def main(args):
    set_global_seed(TrainingConfig.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    tok_path = args.tokenizer_path
    epochs = args.epochs
    max_seq_len = args.max_seq_len
    mlm_prob = args.mlm_prob
    lr = args.lr
    weight_decay = args.weight_decay
    ckpt_path = save_path + ".ckpt"

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks

    # Apply model size overrides (default: keep ModelConfig values)
    if args.dim > 0:
        ModelConfig.dim = args.dim
    if args.depth > 0:
        ModelConfig.depth = args.depth
    if args.heads > 0:
        ModelConfig.heads = args.heads
    if args.num_kv_heads > 0:
        ModelConfig.num_kv_heads = args.num_kv_heads
    if args.ffn_multiplier > 0:
        ModelConfig.ffn_multiplier = args.ffn_multiplier
    if args.dropout >= 0:
        ModelConfig.dropout = args.dropout

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}, tok={tok_path}, ep={epochs}")
    print(f"  max_stocks={args.max_stocks}, max_seq_len={max_seq_len}")
    print(f"  mlm_prob={mlm_prob}, lr={lr}, wd={weight_decay}")
    print(f"  model: dim={ModelConfig.dim} depth={ModelConfig.depth} heads={ModelConfig.heads} "
          f"num_kv_heads={ModelConfig.num_kv_heads} ffn_mult={ModelConfig.ffn_multiplier} "
          f"dropout={ModelConfig.dropout}")

    tokenizer = load_tokenizer(tok_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    # Reuse the existing token cache (same key as train_base.py).
    cache_tag = os.path.basename(tok_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=max_seq_len)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)

    # Model
    model = KronosBert().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: enabled")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    print(f"Optimizer: AdamW, lr={lr}, wd={weight_decay}")

    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16

    # ---- Resume ----
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
            print(f"  Cannot resume (incompatible): {e}")
            print(f"  Starting fresh training.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "mlm_acc": []}
    t0 = time.time()

    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        accs = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{epochs}")
        for bi, batch in enumerate(pbar):
            # dataloader returns 7-tuple (input_ids, targets, time_ids, pos, mask, va, reg_targets)
            input_ids, _, time_id, pos_id, _, va_val, _ = batch
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
            # Build MLM batch (no padding — single sequence at a time)
            mlm_ids_list = []
            mlm_labels_list = []
            for b in range(B):
                mlm_ids_b, mlm_labels_b = make_mlm_batch(
                    input_ids[b], vocab_base, mask_id, mlm_prob=mlm_prob)
                mlm_ids_list.append(mlm_ids_b)
                mlm_labels_list.append(mlm_labels_b)
            mlm_ids = torch.stack(mlm_ids_list, dim=0)
            mlm_labels = torch.stack(mlm_labels_list, dim=0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mlm_ids, time_id, pos_id, va_values=va_val)
                # CE loss only on masked positions
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )

                # MLM accuracy (informational)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = (mlm_labels != -100)
                    if valid.any():
                        acc = (pred[valid] == mlm_labels[valid]).float().mean().item()
                        accs.append(acc)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
        vlosses = []
        vaccs = []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, _, time_id, pos_id, _, va_val, _ = batch
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
                mlm_ids_list = []
                mlm_labels_list = []
                for b in range(B):
                    mlm_ids_b, mlm_labels_b = make_mlm_batch(
                        input_ids[b], vocab_base, mask_id, mlm_prob=mlm_prob)
                    mlm_ids_list.append(mlm_ids_b)
                    mlm_labels_list.append(mlm_labels_b)
                mlm_ids = torch.stack(mlm_ids_list, dim=0)
                mlm_labels = torch.stack(mlm_labels_list, dim=0)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mlm_ids, time_id, pos_id, va_values=va_val)
                    vloss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        mlm_labels.view(-1),
                        ignore_index=-100,
                    )
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
                           "vocab_size": ModelConfig.vocab_size,
                           "arch": "kronos_bert"},
                "val_loss": best_val,
                "mlm_acc": avg_val_acc,
                "epoch": epoch,
                "completed": epoch == epochs - 1,
                "tag": args.tag,
                "mlm_prob": mlm_prob,
            }, save_path)
            save_tag = "  -> Saved best"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "global_step": global_step,
            "tag": args.tag,
        }, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        print(f"  [{epochs_done}/{epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"mlm_acc={avg_val_acc:.3f} lr={cur_lr:.2e} elapsed={elapsed:.0f}s "
              f"ETA={eta:.0f}s step={global_step}{save_tag}", flush=True)

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    with open(os.path.join(os.path.dirname(save_path) or ".", f"history_{args.tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KronosBert (bidirectional MLM calibrator). Default: big BERT (16M, 4695 stocks).",
    )
    parser.add_argument("--save_path", type=str,
                        default="checkpoints/kronos_bert_big_v1.pt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="Subsample N stocks (0=all 4695).")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--tag", type=str, default="kronos_bert_big_v1")
    # Model size overrides. Defaults below are the big BERT (16M).
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--ffn_multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    main(parser.parse_args())
