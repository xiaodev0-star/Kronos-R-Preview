"""Continue training from checkpoint for remaining epochs."""
import math
import os
import time
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, pack_stocks, make_dataloader
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from reproducibility import set_global_seed

EXTRA_EPOCHS = 2


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad = False
    return tok


def main():
    set_global_seed(TrainingConfig.random_seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(TrainingConfig.tokenizer_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    train_seqs = pack_stocks(train_s, tokenizer, mode="train")
    val_seqs = pack_stocks(val_s, tokenizer, mode="train")
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}")

    train_loader = make_dataloader(train_seqs, batch_size=TrainingConfig.batch_size, shuffle=True)
    val_loader = make_dataloader(val_seqs, batch_size=TrainingConfig.batch_size, shuffle=False)

    # Load existing model
    model = KronosPreview().to(device)
    ckpt = torch.load(TrainingConfig.base_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("val_loss", float("inf"))
    print(f"Resumed from epoch {start_epoch}, val_loss={best_val:.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if TrainingConfig.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TrainingConfig.learning_rate,
        weight_decay=TrainingConfig.weight_decay)

    total_updates = len(train_loader) * EXTRA_EPOCHS
    warmup = max(1, int(total_updates * 0.1))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16
    global_step = 0

    for ep in range(EXTRA_EPOCHS):
        epoch = start_epoch + ep
        model.train()
        losses = []
        optimizer.zero_grad()
        t0 = time.time()

        for step, (input_ids, targets, time_ids, position_ids, attn_mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + EXTRA_EPOCHS}")
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            time_ids = time_ids.to(device, non_blocking=True)
            position_ids = position_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)

            if loss is None:
                continue

            (loss / TrainingConfig.accumulation_steps).backward()

            if (step + 1) % TrainingConfig.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            losses.append(loss.item())

        avg_train = sum(losses) / max(len(losses), 1)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for input_ids, targets, time_ids, position_ids, attn_mask in val_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                time_ids = time_ids.to(device)
                position_ids = position_ids.to(device)
                attn_mask = attn_mask.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)
                if loss is not None:
                    vlosses.append(loss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={cur_lr:.2e}  {elapsed:.0f}s  step={global_step}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
                "epoch": epoch, "val_loss": avg_val,
            }, TrainingConfig.base_model_path)
            print(f"  -> Saved best model")

    print(f"\nDone. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
