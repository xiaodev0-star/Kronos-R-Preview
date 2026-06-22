"""Train base model: KronosPreview on packed stock sequences.
Supports epoch-level checkpoint/resume via save_path + save_path.ckpt.
Supports: focal loss, weight_decay override, reasoning module."""
import argparse
import math
import os
import time
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from model import load_tokenizer
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from model.optimizer import build_muon_optimizers


def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0, entropy_alpha=0.0,
               ignore_index=-100):
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    mask = (targets != ignore_index).float()
    safe_targets = targets.clamp(min=0)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
    if entropy_alpha > 0:
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        ent_loss = (entropy * mask).sum() / mask.sum().clamp(min=1)
        loss = loss - entropy_alpha * ent_loss
    return loss


def _to_device(batch, device):
    """Move an 8-tuple batch to device and ensure batch dimension."""
    inp, tgt, ftgt, tids, pos, mask, va, rt = [x.to(device, non_blocking=True) for x in batch]
    if inp.dim() == 1:
        inp, tgt, ftgt, tids, pos, mask, va, rt = [x.unsqueeze(0) for x in (inp, tgt, ftgt, tids, pos, mask, va, rt)]
    return inp, tgt, ftgt, tids, pos, mask, va, rt


def _pad_batch(sequences, batch_size):
    batches = []
    for i in range(0, len(sequences), batch_size):
        group = sequences[i:i + batch_size]
        max_len = max(s["input_ids"].shape[0] for s in group)
        B = len(group)
        p_ids = torch.zeros(B, max_len, dtype=torch.long)
        p_tgt = torch.full((B, max_len - 1), -100, dtype=torch.long)
        p_ftgt = torch.zeros(B, max_len - 1, dtype=torch.long)  # fine targets
        p_time = torch.zeros(B, max_len, 3, dtype=torch.long)
        p_pos = torch.zeros(B, max_len, dtype=torch.long)
        p_mask = torch.zeros(B, max_len, max_len, dtype=torch.bool)
        p_va = torch.zeros(B, max_len, 2, dtype=torch.float32)
        p_rt = torch.full((B, max_len), -999.0, dtype=torch.float32)

        for j, s in enumerate(group):
            L = s["input_ids"].shape[0]
            Lt = s["targets"].shape[0]
            p_ids[j, :L] = s["input_ids"]
            p_tgt[j, :Lt] = s["targets"]
            if "fine_targets" in s:
                p_ftgt[j, :Lt] = s["fine_targets"]
            p_time[j, :L] = s["time_ids"]
            p_pos[j, :L] = s["position_ids"]
            p_va[j, :L] = s["va_values"]
            p_rt[j, :L] = s["reg_targets"]
            mask = torch.zeros(L, L, dtype=torch.bool)
            mask[:, 0] = True
            for start, end in s.get("boundaries", [(1, L)]):
                for pos in range(start, min(end, L)):
                    mask[pos, start:pos + 1] = True
            p_mask[j, :L, :L] = mask
            p_mask[j, L:, 0] = True

        batches.append((p_ids, p_tgt, p_ftgt, p_time, p_pos, p_mask, p_va, p_rt))
    return batches


class BatchedDataLoader:
    def __init__(self, sequences, batch_size, shuffle=True):
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(torch.randint(0, 2**31, (1,)).item())
            indices = torch.randperm(len(self.sequences), generator=rng).tolist()
        grouped = sorted(indices, key=lambda i: self.sequences[i]["input_ids"].shape[0])
        for i in range(0, len(grouped), self.batch_size):
            batch_idx = grouped[i:i + self.batch_size]
            group = [self.sequences[j] for j in batch_idx]
            yield _pad_batch(group, len(group))[0]

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size


def main(args):
    set_global_seed(TrainingConfig.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT standard architecture (HPO 2026-06-18 best: phase3_t000, DA 48.12% with V2).
    # Hardcode here so that any prior mutation of ModelConfig cannot leak into the GPT run.
    ModelConfig.dim = 256
    ModelConfig.depth = 2
    ModelConfig.heads = 4
    ModelConfig.num_kv_heads = 1
    ModelConfig.dropout = args.dropout
    ModelConfig.ffn_multiplier = 4
    ModelConfig.position_encoding = "rope"
    ModelConfig.rope_base = 10000.0
    ModelConfig.vocab_size = 1024
    ModelConfig.va_hidden_dim = 64

    save_path = args.save_path
    tok_path = args.tokenizer_path
    epochs = args.epochs
    ckpt_path = save_path + ".ckpt"
    effective_lr = args.lr

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}, tok={tok_path}, ep={epochs}")
    print(f"  loss={args.loss}, gamma={args.gamma}, wd={args.weight_decay}, lr={effective_lr}")
    if args.label_smoothing > 0:
        print(f"  label_smoothing={args.label_smoothing}")
    if args.entropy_alpha > 0:
        print(f"  entropy_alpha={args.entropy_alpha}")
    if args.dropout != 0.1:
        print(f"  dropout={args.dropout}")
    if args.reasoning:
        print(f"  reasoning=True, frozen={args.reasoning_frozen}")
        if args.base_checkpoint:
            print(f"  base_checkpoint={args.base_checkpoint}")
    if args.heteroscedastic:
        print(f"  heteroscedastic=True, het_weight={args.het_weight}")
    if args.max_stocks > 0:
        print(f"  [FAST] max_stocks={args.max_stocks} (subsampled for HPO screening)")
    if args.light_eval:
        print(f"  [LIGHT_EVAL] computing collapse_rate/token_diversity per epoch")

    tokenizer = load_tokenizer(tok_path, device)
    # Derive vocab_size from tokenizer (supports variable bit widths)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    print(f"Tokenizer loaded. vocab_coarse={ModelConfig.vocab_size}, vocab_fine={ModelConfig.vocab_fine} "
          f"(bits: L1={tokenizer.bits_l1}, L2={tokenizer.bits_l2})")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(tok_path).replace(".pt", "")
    cache_suffix = "_het_vol" if args.heteroscedastic else ""
    seq_suffix = f"_seq{args.max_seq_len}" if args.max_seq_len > 0 else ""
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}{cache_suffix}{seq_suffix}")
    if args.force_repack and os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"  [FORCE_REPACK] cleared cache: {cache_dir}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=args.max_seq_len)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    bs = TrainingConfig.batch_size
    if bs > 1:
        train_loader = BatchedDataLoader(train_seqs, bs, shuffle=True)
        val_loader = BatchedDataLoader(val_seqs, bs, shuffle=False)
        print(f"Loader: batched, bs={bs}, accum={TrainingConfig.accumulation_steps}")
    else:
        train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
        val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)
        print(f"Loader: single-seq, accum={TrainingConfig.accumulation_steps}")

    # Model
    if args.reasoning:
        base_state = None
        if args.base_checkpoint and os.path.exists(args.base_checkpoint):
            base_ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
            base_state = base_ckpt["model_state_dict"]
            print(f"  Loaded base checkpoint: {args.base_checkpoint} (val_loss={base_ckpt.get('val_loss', 'N/A')})")
        model = KronosPreviewWithReasoning(base_model_state=base_state).to(device)
        if args.reasoning_frozen:
            for name, param in model.named_parameters():
                if "reason" not in name:
                    param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Params: {total:,} (trainable: {trainable:,})")
        else:
            print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = KronosPreview().to(device)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if TrainingConfig.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Optimizer: Muon+AdamW (2D→Muon, 1D→AdamW) or standard AdamW
    if args.optimizer == "muon":
        optimizer, optimizer_adam = build_muon_optimizers(
            model, lr_muon=args.lr_muon, lr_adam=effective_lr,
            momentum=0.95, weight_decay_muon=0.0,
            weight_decay_adam=args.weight_decay)
        print(f"Optimizer: Muon(lr={args.lr_muon}) + AdamW(lr={effective_lr}, wd={args.weight_decay})")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr,
                                      weight_decay=args.weight_decay)
        optimizer_adam = None
        print(f"Optimizer: AdamW, lr={effective_lr}, wd={args.weight_decay}")

    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_adam = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda) if optimizer_adam else None
    amp_dtype = torch.bfloat16
    accum = TrainingConfig.accumulation_steps
    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

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
            if optimizer_adam and "optimizer_adam_state" in ckpt:
                optimizer_adam.load_state_dict(ckpt["optimizer_adam_state"])
                scheduler_adam.load_state_dict(ckpt["scheduler_adam_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}, step={global_step}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume from checkpoint (incompatible): {e}")
            print(f"  Starting fresh training.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_het_loss": [], "lr": []}
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_acc = torch.zeros((), device=device)
        n_loss = 0
        optimizer.zero_grad(set_to_none=True)
        if optimizer_adam:
            optimizer_adam.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{epochs}")
        for bi, batch in enumerate(pbar):
            input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target = _to_device(batch, device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if args.heteroscedastic:
                    coarse_logits, fine_logits, _, het_loss = model(
                        input_ids, time_id, pos_id, mask,
                        va_values=va_val, reg_targets=reg_target,
                        fine_targets=fine_target)
                else:
                    coarse_logits, fine_logits = model(
                        input_ids, time_id, pos_id, mask,
                        va_values=va_val, fine_targets=fine_target)
                    het_loss = None

                # Coarse loss (main)
                shift_coarse = coarse_logits[:, :-1, :].contiguous()
                shift_targets = target.contiguous()
                if (shift_targets == -100).all():
                    continue
                if args.loss == "focal":
                    loss = focal_loss(shift_coarse.view(-1, shift_coarse.size(-1)),
                                      shift_targets.view(-1), gamma=args.gamma,
                                      label_smoothing=args.label_smoothing,
                                      entropy_alpha=args.entropy_alpha)
                else:
                    loss = F.cross_entropy(shift_coarse.view(-1, shift_coarse.size(-1)),
                                           shift_targets.view(-1), ignore_index=-100,
                                           label_smoothing=args.label_smoothing)

                # Fine loss: f_logits is [B, S-1, V_fine], fine_target is [B, S-1]
                # f_logits[t] predicts fine_target[t], skip first position (no context)
                shift_fine = f_logits[:, 1:, :].contiguous()    # [B, S-2, V_fine]
                shift_fine_tgt = fine_target[:, 1:].contiguous() # [B, S-2]
                fine_mask = (shift_targets != -100)
                if fine_mask.any():
                    fine_loss = F.cross_entropy(
                        shift_fine.view(-1, shift_fine.size(-1)),
                        shift_fine_tgt.view(-1), ignore_index=0)
                    loss = loss + args.fine_weight * fine_loss

                if het_loss is not None and args.heteroscedastic:
                    loss = loss + args.het_weight * het_loss

            if loss is None:
                continue

            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
                optimizer.step()
                if optimizer_adam:
                    optimizer_adam.step()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_adam:
                    optimizer_adam.zero_grad(set_to_none=True)
                scheduler.step()
                if scheduler_adam:
                    scheduler_adam.step()
                global_step += 1

            loss_acc += loss.detach()
            n_loss += 1

            # Update pbar every 20 steps (avoids per-step .item() sync)
            if (bi + 1) % 20 == 0:
                pbar.set_postfix({"loss": f"{loss_acc.item() / n_loss:.4f}",
                                  "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        avg_train = (loss_acc / max(n_loss, 1)).item()

        # Validation (always CE for comparable val_loss)
        model.eval()
        vlosses = []
        v_het_losses = []
        val_pred_tokens = []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target = _to_device(batch, device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if args.heteroscedastic:
                        coarse_logits, fine_logits, _, val_het = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, reg_targets=reg_target,
                            fine_targets=fine_target)
                        if val_het is not None:
                            v_het_losses.append(val_het.item())
                    else:
                        coarse_logits, fine_logits = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, fine_targets=fine_target)
                    shift_logits = coarse_logits[:, :-1, :].contiguous()
                    shift_targets = target.contiguous()
                    if (shift_targets != -100).any():
                        vloss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_targets.view(-1), ignore_index=-100)
                        vlosses.append(vloss.item())
                    if args.light_eval:
                        valid_mask = (shift_targets != -100)
                        if valid_mask.any():
                            preds = shift_logits.argmax(dim=-1)
                            val_pred_tokens.append(preds[valid_mask].cpu())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        avg_val_het = sum(v_het_losses) / max(len(v_het_losses), 1) if v_het_losses else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Light eval: compute collapse rate and token diversity
        collapse_rate = 0.0
        n_unique_tokens = 0
        if args.light_eval and val_pred_tokens:
            all_preds = torch.cat(val_pred_tokens)
            total = all_preds.numel()
            if total > 0:
                unique, counts = torch.unique(all_preds, return_counts=True)
                collapse_rate = counts.max().item() / total
                n_unique_tokens = len(unique)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_het_loss"].append(avg_val_het)
        history["lr"].append(cur_lr)
        if args.light_eval:
            history.setdefault("collapse_rate", []).append(collapse_rate)
            history.setdefault("n_unique_tokens", []).append(n_unique_tokens)

        # Write per-epoch history for Optuna pruning
        if args.history_per_epoch:
            hp_path = os.path.join(os.path.dirname(save_path) or ".", f"history_{args.tag}.json")
            with open(hp_path, "w") as f:
                json.dump(history, f, indent=2)

        save_tag = ""
        if avg_val < best_val:
            best_val = avg_val
            sd = model.state_dict()
            torch.save({
                "model_state_dict": sd,
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
                "val_loss": best_val,
                "epoch": epoch,
                "completed": epoch == epochs - 1,
                "tag": args.tag,
                "loss_type": args.loss,
                "gamma": args.gamma,
                "weight_decay": args.weight_decay,
                "use_reasoning": args.reasoning,
                "reasoning_frozen": args.reasoning_frozen,
                "label_smoothing": args.label_smoothing,
                "entropy_alpha": args.entropy_alpha,
                "dropout": ModelConfig.dropout,
                "heteroscedastic": args.heteroscedastic,
                "het_weight": args.het_weight,
                "collapse_rate": collapse_rate,
                "n_unique_tokens": n_unique_tokens,
            }, save_path)
            save_tag = "  -> Saved best"

        ckpt_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "global_step": global_step,
            "tag": args.tag,
        }
        if optimizer_adam:
            ckpt_dict["optimizer_adam_state"] = optimizer_adam.state_dict()
            ckpt_dict["scheduler_adam_state"] = scheduler_adam.state_dict()
        torch.save(ckpt_dict, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        light_str = ""
        if args.light_eval and collapse_rate > 0:
            light_str = f" coll={collapse_rate*100:.1f}% uniq={n_unique_tokens}"
        print(f"  [{epochs_done}/{epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"lr={cur_lr:.2e} elapsed={elapsed:.0f}s ETA={eta:.0f}s step={global_step}"
              f"{light_str}{save_tag}", flush=True)

        if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
            break

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    with open(os.path.join(os.path.dirname(save_path) or ".", f"history_{args.tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kronos-Preview training. Default: focal γ=4 + heteroscedastic=ON (HPO 2026-06-18 best, phase3_t000: DA 48.12% with V2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HPO 2026-06-18 best (phase3_t000) — DA 48.12% with V2 calibration.
  # Just run with defaults:
  python train_base.py

  # 15-epoch / lower-dropout variant (phase4_t003) — DA 47.93% with V2, lower collapse (16.8%).
  python train_base.py --epochs 15 --dropout 0.05

  # Disable heteroscedastic head (reproduces pre-HPO behavior):
  python train_base.py --no-heteroscedastic

  # Standard CE baseline (no focal):
  python train_base.py --loss ce --weight_decay 0.001

  # Reasoning model (two-stage):
  python train_base.py --loss ce --reasoning --reasoning_frozen --base_checkpoint model.pt
  python train_base.py --loss focal --gamma 4.0 --reasoning
        """)
    # Core
    parser.add_argument("--save_path", type=str, default=TrainingConfig.base_model_path)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer: adamw (default) or muon+adamw (2D→Muon, 1D→AdamW)")
    parser.add_argument("--lr_muon", type=float, default=0.02,
                        help="Muon learning rate (only when --optimizer muon)")
    parser.add_argument("--light_eval", action="store_true",
                        help="Compute collapse_rate/token_diversity during validation (fast HPO proxy)")
    # Focal loss
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Focal loss gamma (HPO 2026-06-18 best: 4.0)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--entropy_alpha", type=float, default=0.0,
                        help="Entropy regularization weight for focal loss (0.0 = disabled)")
    # Heteroscedastic
    parser.add_argument("--heteroscedastic", dest="heteroscedastic", action="store_true", default=True)
    parser.add_argument("--no-heteroscedastic", dest="heteroscedastic", action="store_false")
    parser.add_argument("--het_weight", type=float, default=0.1,
                        help="Weight for heteroscedastic NLL loss (HPO 2026-06-18 best: 0.1)")
    parser.add_argument("--fine_weight", type=float, default=0.3,
                        help="Weight for fine-token auxiliary loss in dual-head prediction")
    # Reasoning
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--reasoning_frozen", action="store_true")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Pre-trained base model checkpoint for reasoning model")
    # Overrides
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="Subsample N stocks for fast HPO screening (0=all)")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Truncate sequences longer than this (0=no limit, recommended 2048 for HPO)")
    parser.add_argument("--force_repack", action="store_true",
                        help="Force re-tokenization (clear cache)")
    parser.add_argument("--history_per_epoch", action="store_true",
                        help="Write per-epoch val_loss to JSON (for Optuna pruning)")
    main(parser.parse_args())
