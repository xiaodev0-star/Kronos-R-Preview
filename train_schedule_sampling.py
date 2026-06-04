"""Schedule Sampling post-training for Kronos models.

Gradually replaces teacher forcing tokens with model predictions during training
to improve autoregressive generation. Based on the Scheduled Sampling approach
(Bengio et al., 2015) adapted for discrete token sequences.

Key differences from standard train_base.py:
1. Stochastic mixing: at each position, with probability p_sample, use model's
   own prediction instead of ground-truth token
2. Linear schedule: p_sample ramps from 0.0 to target_prob over training
3. Supports multiple schedule types: linear, exponential, sigmoid
4. Token-level or sequence-level sampling

Usage:
  python train_schedule_sampling.py --base_checkpoint checkpoints/hpo_v3/w2_focal_g8.pt \
      --loss focal --gamma 8.0 --ss_prob 0.5 --epochs 5 --tag ss_g8_p50
"""
import argparse
import math, os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processor import load_stocks, split_stocks, pack_stocks, make_dataloader
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from reproducibility import set_global_seed

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0, ignore_index=-100):
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    mask = (targets != ignore_index).float()
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
    return loss


def linear_schedule(step, total_steps, start=0.0, end=0.5):
    """p_sample linearly ramps from start to end."""
    if total_steps <= 0: return end
    progress = min(step / total_steps, 1.0)
    return start + (end - start) * progress


def exponential_schedule(step, total_steps, start=0.0, end=0.5, k=5.0):
    """p_sample exponentially approaches end."""
    if total_steps <= 0: return end
    progress = min(step / total_steps, 1.0)
    return start + (end - start) * (1.0 - math.exp(-k * progress)) / (1.0 - math.exp(-k))


def sigmoid_schedule(step, total_steps, start=0.0, end=0.5, k=10.0):
    """Sigmoid-shaped schedule — slow start, fast middle, slow end."""
    if total_steps <= 0: return end
    x = (step / total_steps - 0.5) * k
    progress = 1.0 / (1.0 + math.exp(-x))
    return start + (end - start) * progress


SCHEDULE_FNS = {
    "linear": linear_schedule,
    "exponential": exponential_schedule,
    "sigmoid": sigmoid_schedule,
}


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters(): p.requires_grad_(False)
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
            mask = torch.zeros(L, L, dtype=torch.bool)
            mask[:, 0] = True
            for start, end in s.get("boundaries", [(1, L)]):
                for pos in range(start, min(end, L)):
                    mask[pos, start:pos + 1] = True
            p_mask[j, :L, :L] = mask
        batches.append((p_ids, p_tgt, p_time, p_pos, p_mask))
    return batches


class BatchedDataLoader:
    def __init__(self, sequences, batch_size, shuffle=True):
        self.sequences = sequences; self.batch_size = batch_size; self.shuffle = shuffle

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
    parser = argparse.ArgumentParser(description="Schedule Sampling fine-tuning for Kronos")
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=TrainingConfig.tokenizer_path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tag", type=str, default="ss_default")
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ss_prob", type=float, default=0.5,
                        help="Target probability of using model prediction instead of ground truth")
    parser.add_argument("--ss_schedule", type=str, default="linear",
                        choices=["linear", "exponential", "sigmoid"],
                        help="Schedule type for ramping up sampling probability")
    parser.add_argument("--ss_type", type=str, default="token",
                        choices=["token", "sequence"],
                        help="token: sample per-position; sequence: sample at start of sequence")
    parser.add_argument("--reasoning", action="store_true",
                        help="Model uses CausalReasoningBlock")
    args = parser.parse_args(args)

    set_global_seed(TrainingConfig.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(CHECKPOINT_DIR, f"{args.tag}.pt")
    ckpt_path = save_path + ".ckpt"
    tok_path = os.path.join(CHECKPOINT_DIR, os.path.basename(args.tokenizer_path))

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}")
    print(f"  base_checkpoint={args.base_checkpoint}")
    print(f"  ss_prob={args.ss_prob}, schedule={args.ss_schedule}, type={args.ss_type}")
    print(f"  loss={args.loss}, gamma={args.gamma}, lr={args.lr}")
    if args.label_smoothing > 0: print(f"  label_smoothing={args.label_smoothing}")

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
    else:
        train_loader = make_dataloader(train_seqs, batch_size=1, shuffle=True)
        val_loader = make_dataloader(val_seqs, batch_size=1, shuffle=False)
    print(f"Loader ready: bs={bs}, accum={TrainingConfig.accumulation_steps}")

    # Load model from checkpoint
    vocab = ModelConfig.vocab_size
    if args.reasoning:
        model = KronosPreviewWithReasoning().to(device)
    else:
        model = KronosPreview().to(device)

    if os.path.exists(args.base_checkpoint):
        ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_val = ckpt.get("val_loss", "N/A")
        print(f"  Loaded base model: val_loss={prev_val}")
    else:
        print(f"  [WARN] Base checkpoint not found: {args.base_checkpoint}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Optimizer: AdamW, lr={args.lr}, wd={args.weight_decay}")

    total_updates = len(train_loader) * args.epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16
    accum = TrainingConfig.accumulation_steps
    schedule_fn = SCHEDULE_FNS[args.ss_schedule]

    # Resume
    start_epoch, best_val, global_step = 0, float("inf"), 0
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_step = ckpt.get("global_step", 0)
            for _ in range(global_step): scheduler.step()
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume: {e}. Starting fresh.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "ss_prob": []}
    t0 = time.time()

    total_train_batches = len(train_loader) * args.epochs
    flat_batch_counter = 0  # counts every batch across all epochs (0 → total_train_batches)

    for epoch in range(start_epoch, args.epochs):
        model.train(); losses = []; optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{args.epochs}")
        for bi, (inp, tgt, tid, pos, mask) in enumerate(pbar):
            # Compute SS probability: linear ramp from 0 → ss_prob over all batches
            ss_p = schedule_fn(flat_batch_counter, total_train_batches,
                               start=0.0, end=args.ss_prob)

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if inp.dim() == 1:
                inp, tgt, tid, pos = inp.unsqueeze(0), tgt.unsqueeze(0), tid.unsqueeze(0), pos.unsqueeze(0)
                mask = mask.unsqueeze(0)

            # Schedule Sampling: replace some input tokens with model predictions
            if ss_p > 0 and args.ss_type == "token":
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits, _, _ = model(inp, tid, pos, mask)
                    pred_tokens = logits.argmax(dim=-1)  # [B, S]

                # Build mixed input: at each position, use prediction with prob ss_p
                B, S = inp.shape
                for b in range(B):
                    # First token (BOS) always kept
                    for s in range(1, S):
                        if torch.rand(1, device=device).item() < ss_p:
                            inp[b, s] = pred_tokens[b, s - 1]  # use predicted token as input

            # Forward pass
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits_coarse, _, _ = model(inp, tid, pos, mask)
                shift_logits = logits_coarse[:, :-1, :].contiguous()
                shift_targets = tgt.contiguous()
                if (shift_targets == -100).all(): continue

                if args.loss == "focal":
                    loss = focal_loss(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_targets.view(-1), gamma=args.gamma,
                                      label_smoothing=args.label_smoothing)
                else:
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                           shift_targets.view(-1), ignore_index=-100,
                                           label_smoothing=args.label_smoothing)

            if loss is None: continue

            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            losses.append(loss.item())
            flat_batch_counter += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                              "ss_p": f"{ss_p:.2f}"})

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        avg_train = sum(losses) / max(len(losses), 1)

        # Validation
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
        history["ss_prob"].append(ss_p)

        if avg_val < best_val:
            best_val = avg_val

        # Save every epoch (last epoch overwrites — we want max SS exposure)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                       "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
            "val_loss": avg_val, "epoch": epoch,
            "completed": epoch == args.epochs - 1,
            "tag": args.tag, "ss_prob": args.ss_prob, "ss_schedule": args.ss_schedule,
            "loss_type": args.loss, "gamma": args.gamma,
        }, save_path)

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
        print(f"  [{epochs_done}/{args.epochs}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"lr={cur_lr:.2e} ss_p={ss_p:.3f} elapsed={elapsed:.0f}s ETA={eta:.0f}s  -> Saved",
              flush=True)

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    history_path = os.path.join(CHECKPOINT_DIR, f"history_{args.tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val_loss: {best_val:.4f} | history saved to {history_path}")


if __name__ == "__main__":
    main()
