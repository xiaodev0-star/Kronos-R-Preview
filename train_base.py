"""Train base model: KronosPreview on packed stock sequences.
Supports epoch-level checkpoint/resume via save_path + save_path.ckpt.
Supports: focal loss, weight_decay override, reasoning module."""
import argparse
import math, os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, pack_stocks, make_dataloader
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from reproducibility import set_global_seed


def focal_loss(logits, targets, gamma=2.0, ignore_index=-100):
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    mask = (targets != ignore_index).float()
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
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

        for j, s in enumerate(group):
            L = s["input_ids"].shape[0]
            Lt = s["targets"].shape[0]
            p_ids[j, :L] = s["input_ids"]
            p_tgt[j, :Lt] = s["targets"]
            p_time[j, :L] = s["time_ids"]
            p_pos[j, :L] = s["position_ids"]
            p_mask[j, :L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))

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
    tag = args.tag if args else "default"
    save_path = args.save_path if args else TrainingConfig.base_model_path
    tok_path = args.tokenizer_path if args else TrainingConfig.tokenizer_path
    epochs = args.epochs if args else TrainingConfig.epochs
    ckpt_path = save_path + ".ckpt"

    # Config overrides from CLI
    loss_type = args.loss if args else "ce"
    gamma = args.gamma if args else 2.0
    weight_decay = args.weight_decay if args else TrainingConfig.weight_decay
    use_reasoning = args.reasoning if args else False
    reasoning_frozen = args.reasoning_frozen if args else False
    base_ckpt_path = args.base_checkpoint if args else None

    print(f"Device: {device}, tag={tag}")
    print(f"  save={save_path}, tok={tok_path}, ep={epochs}")
    print(f"  loss={loss_type}, gamma={gamma}, wd={weight_decay}")
    if use_reasoning:
        print(f"  reasoning=True, frozen={reasoning_frozen}")
        if base_ckpt_path:
            print(f"  base_checkpoint={base_ckpt_path}")

    tokenizer = load_tokenizer(tok_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(tok_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}")
    print(f"Encoding (cache: {cache_dir}) ...")
    train_seqs = pack_stocks(train_s, tokenizer, mode="train", cache_dir=cache_dir)
    val_seqs = pack_stocks(val_s, tokenizer, mode="train", cache_dir=cache_dir)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    bs = TrainingConfig.batch_size
    if bs > 1:
        train_loader = BatchedDataLoader(train_seqs, bs, shuffle=True)
        val_loader = BatchedDataLoader(val_seqs, bs, shuffle=False)
        print(f"Loader: batched, bs={bs}, accum={TrainingConfig.accumulation_steps}")
    else:
        train_loader = make_dataloader(train_seqs, batch_size=1, shuffle=True)
        val_loader = make_dataloader(val_seqs, batch_size=1, shuffle=False)
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
            for _ in range(global_step):
                scheduler.step()
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}, step={global_step}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume from checkpoint (incompatible): {e}")
            print(f"  Starting fresh training.")
            # Clean up incompatible checkpoint files
            os.remove(ckpt_path)
            if os.path.exists(save_path):
                os.remove(save_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}")
        for bi, (inp, tgt, tid, pos, mask) in enumerate(pbar):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Ensure batch dimension (make_dataloader bs=1 strips it)
            if inp.dim() == 1:
                inp, tgt, tid, pos = inp.unsqueeze(0), tgt.unsqueeze(0), tid.unsqueeze(0), pos.unsqueeze(0)
                mask = mask.unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits_coarse, _, _ = model(inp, tid, pos, mask)
                shift_logits = logits_coarse[:, :-1, :].contiguous()
                shift_targets = tgt.contiguous()
                if (shift_targets == -100).all():
                    continue
                if loss_type == "focal":
                    loss = focal_loss(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_targets.view(-1), gamma=gamma)
                else:
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                           shift_targets.view(-1), ignore_index=-100)

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
        with torch.inference_mode():
            for inp, tgt, tid, pos, mask in val_loader:
                inp = inp.to(device); tgt = tgt.to(device)
                tid = tid.to(device); pos = pos.to(device); mask = mask.to(device)
                if inp.dim() == 1:
                    inp, tgt, tid, pos = inp.unsqueeze(0), tgt.unsqueeze(0), tid.unsqueeze(0), pos.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits_coarse, _, _ = model(inp, tid, pos, mask)
                    shift_logits = logits_coarse[:, :-1, :].contiguous()
                    shift_targets = tgt.contiguous()
                    if (shift_targets != -100).any():
                        vloss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_targets.view(-1), ignore_index=-100)
                        vlosses.append(vloss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(cur_lr)

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
        print(f"  [{epochs_done}/{epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"lr={cur_lr:.2e} elapsed={elapsed:.0f}s ETA={eta:.0f}s step={global_step}{mark}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=TrainingConfig.base_model_path)
    parser.add_argument("--tokenizer_path", type=str, default=TrainingConfig.tokenizer_path)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--reasoning_frozen", action="store_true")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Pre-trained base model checkpoint for reasoning model")
    main(parser.parse_args())
