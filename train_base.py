"""Train base model: KronosPreview on packed stock sequences.
Supports epoch-level checkpoint/resume via save_path + save_path.ckpt.
Supports: focal loss, weight_decay override, reasoning module."""
import argparse
import math, os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from reproducibility import set_global_seed


def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0, entropy_alpha=0.0, ignore_index=-100):
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    mask = (targets != ignore_index).float()
    # Clamp targets for gather (avoid index -100)
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


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def _pad_batch(sequences, batch_size):
    batches = []
    for i in range(0, len(sequences), batch_size):
        group = sequences[i:i + batch_size]
        max_len = max(s["input_ids"].shape[0] for s in group)
        B = len(group)
        p_ids = torch.zeros(B, max_len, dtype=torch.long)
        p_tgt = torch.full((B, max_len - 1), -100, dtype=torch.long)
        p_time = torch.zeros(B, max_len, 3, dtype=torch.long)
        p_pos = torch.zeros(B, max_len, dtype=torch.long)
        p_mask = torch.zeros(B, max_len, max_len, dtype=torch.bool)
        # v2: va_values [B, max_len, 2]
        has_va = "va_values" in group[0]
        p_va = torch.zeros(B, max_len, 2, dtype=torch.float32) if has_va else None
        # v2: regression targets [B, max_len] (heteroscedastic head)
        has_rt = "reg_targets" in group[0]
        p_rt = torch.full((B, max_len), -999.0, dtype=torch.float32) if has_rt else None
        # v2 (Experiment A): fine token targets [B, max_len] (-100 = ignore in CE)
        has_ft = "fine_targets" in group[0]
        p_ft = torch.full((B, max_len), -100, dtype=torch.long) if has_ft else None

        for j, s in enumerate(group):
            L = s["input_ids"].shape[0]
            Lt = s["targets"].shape[0]
            p_ids[j, :L] = s["input_ids"]
            p_tgt[j, :Lt] = s["targets"]
            p_time[j, :L] = s["time_ids"]
            p_pos[j, :L] = s["position_ids"]
            if has_va:
                p_va[j, :L] = s["va_values"]
            if has_rt:
                p_rt[j, :L] = s["reg_targets"]
            if has_ft:
                p_ft[j, :L] = s["fine_targets"]
            # Causal mask: BOS visible to all, causal within each segment
            mask = torch.zeros(L, L, dtype=torch.bool)
            mask[:, 0] = True
            for start, end in s.get("boundaries", [(1, L)]):
                for pos in range(start, min(end, L)):
                    mask[pos, start:pos + 1] = True
            p_mask[j, :L, :L] = mask
            # Padded positions must attend to BOS to avoid NaN in SDPA
            p_mask[j, L:, 0] = True

        if has_va and has_rt and has_ft:
            batches.append((p_ids, p_tgt, p_time, p_pos, p_mask, p_va, p_rt, p_ft))
        elif has_va and has_rt:
            batches.append((p_ids, p_tgt, p_time, p_pos, p_mask, p_va, p_rt))
        elif has_va:
            batches.append((p_ids, p_tgt, p_time, p_pos, p_mask, p_va))
        else:
            batches.append((p_ids, p_tgt, p_time, p_pos, p_mask))
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


def main(args=None):
    set_global_seed(TrainingConfig.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT standard architecture (HPO 2026-06-18 best: phase3_t000, DA 48.12% with V2).
    # Hardcode here so that any prior mutation of ModelConfig (e.g. by train_bert.py
    # setting dim=512 for the big BERT) cannot leak into the GPT run.
    ModelConfig.dim = 256
    ModelConfig.depth = 2
    ModelConfig.heads = 4
    ModelConfig.num_kv_heads = 1
    ModelConfig.dropout = 0.1
    ModelConfig.ffn_multiplier = 4
    ModelConfig.position_encoding = "rope"
    ModelConfig.rope_base = 10000.0
    ModelConfig.vocab_size = 1024
    ModelConfig.va_hidden_dim = 64

    tag = args.tag if args else "default"
    save_path = args.save_path if args else TrainingConfig.base_model_path
    tok_path = args.tokenizer_path if args else TrainingConfig.tokenizer_path
    epochs = args.epochs if args else TrainingConfig.epochs
    ckpt_path = save_path + ".ckpt"

    # Config overrides from CLI
    loss_type = args.loss if args else "focal"
    gamma = args.gamma if args else 4.0
    weight_decay = args.weight_decay if args else TrainingConfig.weight_decay
    use_reasoning = args.reasoning if args else False
    reasoning_frozen = args.reasoning_frozen if args else False
    base_ckpt_path = args.base_checkpoint if args else None
    label_smoothing = args.label_smoothing if args else 0.0
    entropy_alpha = args.entropy_alpha if args else 0.0
    dropout_override = args.dropout if args else None
    use_heteroscedastic = args.heteroscedastic if args else True
    het_weight = args.het_weight if args else 0.1
    use_head_fine = args.use_head_fine if args else False
    fine_weight = args.fine_weight if args else 0.5
    history_per_epoch = args.history_per_epoch if args else False
    light_eval = args.light_eval if args else False
    max_stocks_override = args.max_stocks if args else 0
    force_repack = args.force_repack if args else False
    max_seq_len = args.max_seq_len if args else 0

    # Apply dropout override before model construction
    if dropout_override is not None:
        ModelConfig.dropout = dropout_override
    # Apply max_stocks override for fast HPO screening
    if max_stocks_override > 0:
        DataConfig.max_stocks = max_stocks_override

    print(f"Device: {device}, tag={tag}")
    print(f"  save={save_path}, tok={tok_path}, ep={epochs}")
    print(f"  loss={loss_type}, gamma={gamma}, wd={weight_decay}")
    if label_smoothing > 0:
        print(f"  label_smoothing={label_smoothing}")
    if entropy_alpha > 0:
        print(f"  entropy_alpha={entropy_alpha}")
    if dropout_override is not None:
        print(f"  dropout={dropout_override}")
    if use_reasoning:
        print(f"  reasoning=True, frozen={reasoning_frozen}")
        if base_ckpt_path:
            print(f"  base_checkpoint={base_ckpt_path}")
    if use_heteroscedastic:
        print(f"  heteroscedastic=True, het_weight={het_weight}")
    if use_head_fine:
        print(f"  use_head_fine=True, fine_weight={fine_weight}")
    if max_stocks_override > 0:
        print(f"  [FAST] max_stocks={max_stocks_override} (subsampled for HPO screening)")
    if light_eval:
        print(f"  [LIGHT_EVAL] computing collapse_rate/token_diversity per epoch")

    tokenizer = load_tokenizer(tok_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(tok_path).replace(".pt", "")
    cache_suffix = "_het_vol" if use_heteroscedastic else ""  # "vol" = volatility reg_target
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}{cache_suffix}")
    if force_repack and os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"  [FORCE_REPACK] cleared cache: {cache_dir}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=max_seq_len)
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
    if use_reasoning:
        base_state = None
        if base_ckpt_path and os.path.exists(base_ckpt_path):
            base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
            base_state = base_ckpt["model_state_dict"]
            print(f"  Loaded base checkpoint: {base_ckpt_path} (val_loss={base_ckpt.get('val_loss', 'N/A')})")
        model = KronosPreviewWithReasoning(base_model_state=base_state).to(device)
        if reasoning_frozen:
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

    # Optimizer (standard AdamW — fused has dtype issues with AMP + frozen params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=TrainingConfig.learning_rate,
                                  weight_decay=weight_decay)
    print(f"Optimizer: AdamW, wd={weight_decay}")

    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16
    accum = TrainingConfig.accumulation_steps

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
            print(f"  Cannot resume from checkpoint (incompatible): {e}")
            print(f"  Starting fresh training.")
            # Only delete resume checkpoint; preserve best model
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_het_loss": [], "lr": []}
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}")
        for bi, batch in enumerate(pbar):
            # v2 dataloader returns 6/7/8-tuple (8 = with reg_targets AND fine_targets)
            reg_tgt = None
            fine_tgt = None
            if len(batch) == 8:
                inp, tgt, tid, pos, mask, va, reg_tgt, fine_tgt = batch
            elif len(batch) == 7:
                inp, tgt, tid, pos, mask, va, reg_tgt = batch
            else:
                inp, tgt, tid, pos, mask, va = batch
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            va = va.to(device, non_blocking=True)
            if reg_tgt is not None:
                reg_tgt = reg_tgt.to(device, non_blocking=True)
            if fine_tgt is not None:
                fine_tgt = fine_tgt.to(device, non_blocking=True)

            # Ensure batch dimension (make_dataloader bs=1 strips it)
            if inp.dim() == 1:
                inp, tgt, tid, pos = inp.unsqueeze(0), tgt.unsqueeze(0), tid.unsqueeze(0), pos.unsqueeze(0)
                mask = mask.unsqueeze(0)
                va = va.unsqueeze(0)
                if reg_tgt is not None:
                    reg_tgt = reg_tgt.unsqueeze(0)
                if fine_tgt is not None:
                    fine_tgt = fine_tgt.unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if use_heteroscedastic and reg_tgt is not None:
                    logits_coarse, logits_fine, _, _, het_loss = model(
                        inp, tid, pos, mask, va_values=va, reg_targets=reg_tgt)
                else:
                    logits_coarse, logits_fine, _ = model(inp, tid, pos, mask, va_values=va)
                    het_loss = None
                shift_logits = logits_coarse[:, :-1, :].contiguous()
                shift_targets = tgt.contiguous()
                if (shift_targets == -100).all():
                    continue
                if loss_type == "focal":
                    loss = focal_loss(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_targets.view(-1), gamma=gamma,
                                      label_smoothing=label_smoothing,
                                      entropy_alpha=entropy_alpha)
                else:
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                           shift_targets.view(-1), ignore_index=-100,
                                           label_smoothing=label_smoothing)
                # Add heteroscedastic regression loss
                if het_loss is not None and use_heteroscedastic:
                    loss = loss + het_weight * het_loss
                # Experiment A: Add fine-head CE loss (activates head_fine)
                if use_head_fine and fine_tgt is not None:
                    shift_fine_logits = logits_fine[:, :-1, :].contiguous()
                    # fine_tgt is [B, S] (with -100 at BOS/EOS); we predict position t+1 at logit t
                    # so slice [:, 1:] to align with logits[:, :-1]
                    shift_fine_targets = fine_tgt[:, 1:].contiguous()  # [B, S-1], -100 = ignore
                    if (shift_fine_targets != -100).any():
                        loss_fine = F.cross_entropy(
                            shift_fine_logits.view(-1, shift_fine_logits.size(-1)),
                            shift_fine_targets.view(-1),
                            ignore_index=-100,
                            label_smoothing=label_smoothing,
                        )
                        loss = loss + fine_weight * loss_fine

            if loss is None:
                continue

            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        avg_train = sum(losses) / max(len(losses), 1)

        # Validation (always CE for comparable val_loss)
        model.eval()
        vlosses = []
        v_het_losses = []
        # Light eval: collect argmax predictions for collapse rate
        val_pred_tokens = []
        with torch.inference_mode():
            for batch in val_loader:
                reg_tgt = None
                fine_tgt = None
                if len(batch) == 8:
                    inp, tgt, tid, pos, mask, va, reg_tgt, fine_tgt = batch
                elif len(batch) == 7:
                    inp, tgt, tid, pos, mask, va, reg_tgt = batch
                else:
                    inp, tgt, tid, pos, mask, va = batch
                inp = inp.to(device); tgt = tgt.to(device)
                tid = tid.to(device); pos = pos.to(device); mask = mask.to(device)
                va = va.to(device)
                if reg_tgt is not None:
                    reg_tgt = reg_tgt.to(device)
                if fine_tgt is not None:
                    fine_tgt = fine_tgt.to(device)
                if inp.dim() == 1:
                    inp, tgt, tid, pos = inp.unsqueeze(0), tgt.unsqueeze(0), tid.unsqueeze(0), pos.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                    va = va.unsqueeze(0)
                    if reg_tgt is not None:
                        reg_tgt = reg_tgt.unsqueeze(0)
                    if fine_tgt is not None:
                        fine_tgt = fine_tgt.unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if use_heteroscedastic and reg_tgt is not None:
                        logits_coarse, _, _, _, val_het = model(
                            inp, tid, pos, mask, va_values=va, reg_targets=reg_tgt)
                        if val_het is not None:
                            v_het_losses.append(val_het.item())
                    else:
                        logits_coarse, _, _ = model(inp, tid, pos, mask, va_values=va)
                    shift_logits = logits_coarse[:, :-1, :].contiguous()
                    shift_targets = tgt.contiguous()
                    if (shift_targets != -100).any():
                        vloss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_targets.view(-1), ignore_index=-100)
                        vlosses.append(vloss.item())
                    # Light eval: collect argmax predictions (free with logits)
                    if light_eval:
                        valid_mask = (shift_targets != -100)
                        if valid_mask.any():
                            preds = shift_logits.argmax(dim=-1)  # [B, S-1]
                            val_pred_tokens.append(preds[valid_mask].cpu())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        avg_val_het = sum(v_het_losses) / max(len(v_het_losses), 1) if v_het_losses else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Light eval: compute collapse rate and token diversity
        collapse_rate = 0.0
        n_unique_tokens = 0
        if light_eval and val_pred_tokens:
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
        if light_eval:
            history.setdefault("collapse_rate", []).append(collapse_rate)
            history.setdefault("n_unique_tokens", []).append(n_unique_tokens)

        # Write per-epoch history for Optuna pruning
        if history_per_epoch:
            hp_path = os.path.join(os.path.dirname(save_path) or ".", f"history_{tag}.json")
            with open(hp_path, "w") as f:
                json.dump(history, f, indent=2)

        tag_s = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
                "val_loss": best_val,
                "epoch": epoch,
                "completed": epoch == epochs - 1,
                "tag": tag,
                "loss_type": loss_type,
                "gamma": gamma,
                "weight_decay": weight_decay,
                "use_reasoning": use_reasoning,
                "reasoning_frozen": reasoning_frozen,
                "label_smoothing": label_smoothing,
                "entropy_alpha": entropy_alpha,
                "dropout": ModelConfig.dropout,
                "heteroscedastic": use_heteroscedastic,
                "het_weight": het_weight,
                "collapse_rate": collapse_rate,
                "n_unique_tokens": n_unique_tokens,
            }, save_path)
            tag_s = "  -> Saved best"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "global_step": global_step,
            "tag": tag,
        }, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        mark = tag_s if tag_s else ""
        light_str = ""
        if light_eval and collapse_rate > 0:
            light_str = f" coll={collapse_rate*100:.1f}% uniq={n_unique_tokens}"
        print(f"  [{epochs_done}/{epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"lr={cur_lr:.2e} elapsed={elapsed:.0f}s ETA={eta:.0f}s step={global_step}{light_str}{mark}",
              flush=True)

        if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
            break

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    with open(os.path.join(os.path.dirname(save_path) or ".", f"history_{tag}.json"), "w") as f:
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
    parser.add_argument("--save_path", type=str, default=TrainingConfig.base_model_path)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Focal loss gamma (HPO 2026-06-18 best: 4.0)")
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--reasoning_frozen", action="store_true")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Pre-trained base model checkpoint for reasoning model")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CE/focal loss (0.0 = disabled, HPO 2026-06-18 best)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override ModelConfig.dropout (default: 0.1, HPO 2026-06-18 best)")
    parser.add_argument("--entropy_alpha", type=float, default=0.0,
                        help="Entropy regularization weight for focal loss (0.0 = disabled)")
    parser.add_argument("--heteroscedastic", dest="heteroscedastic", action="store_true", default=True,
                        help="Enable heteroscedastic regression head (default: ON, HPO 2026-06-18 best)")
    parser.add_argument("--no-heteroscedastic", dest="heteroscedastic", action="store_false",
                        help="Disable heteroscedastic regression head (reproduces pre-HPO behavior)")
    parser.add_argument("--het_weight", type=float, default=0.1,
                        help="Weight for heteroscedastic NLL loss (HPO 2026-06-18 best: 0.1)")
    parser.add_argument("--use_head_fine", action="store_true",
                        help="[Experiment A] Enable fine-head CE loss (activates the 2-level head_fine)")
    parser.add_argument("--fine_weight", type=float, default=0.5,
                        help="[Experiment A] Weight for fine-head CE loss (default: 0.5)")
    parser.add_argument("--history_per_epoch", action="store_true",
                        help="Write per-epoch val_loss to JSON (for Optuna pruning)")
    parser.add_argument("--light_eval", action="store_true",
                        help="Compute collapse_rate/token_diversity during validation (fast HPO proxy)")
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="Subsample N stocks for fast HPO screening (0=all)")
    parser.add_argument("--force_repack", action="store_true",
                        help="Force re-tokenization (clear cache, e.g. after reg_target change)")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Truncate sequences longer than this (0=no limit, recommended 2048 for HPO)")
    main(parser.parse_args())
