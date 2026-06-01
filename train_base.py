"""Train base model: KronosPreview on packed stock sequences.  Optimized version."""
import math, os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig
from data_processor import load_stocks, split_stocks, pack_stocks, make_dataloader
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from reproducibility import set_global_seed


def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def _pad_batch(sequences, batch_size):
    """Pad variable-length sequences into fixed batch. targets padded to max_len-1
    to match model's shift_logits (logits[:, :-1, :])."""
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
        # sort by length to minimize padding
        grouped = sorted(indices, key=lambda i: self.sequences[i]["input_ids"].shape[0])
        for i in range(0, len(grouped), self.batch_size):
            batch_idx = grouped[i:i + self.batch_size]
            group = [self.sequences[j] for j in batch_idx]
            yield _pad_batch(group, len(group))[0]

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size


def main():
    set_global_seed(TrainingConfig.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(TrainingConfig.tokenizer_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    # Token cache
    cache_dir = TrainingConfig.token_cache_dir
    print(f"Encoding (cache: {cache_dir}) ...")
    train_seqs = pack_stocks(train_s, tokenizer, mode="train", cache_dir=cache_dir)
    val_seqs = pack_stocks(val_s, tokenizer, mode="train", cache_dir=cache_dir)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    # DataLoader
    bs = TrainingConfig.batch_size
    if bs > 1:
        train_loader = BatchedDataLoader(train_seqs, bs, shuffle=True)
        val_loader = BatchedDataLoader(val_seqs, bs, shuffle=False)
        print(f"Loader: batched, batch_size={bs}, accum={TrainingConfig.accumulation_steps}")
    else:
        train_loader = make_dataloader(train_seqs, batch_size=1, shuffle=True)
        val_loader = make_dataloader(val_seqs, batch_size=1, shuffle=False)
        print(f"Loader: single-seq, accum={TrainingConfig.accumulation_steps}")

    # Model
    model = KronosPreview().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    if TrainingConfig.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Optimizer (fused AdamW)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.learning_rate,
                                      weight_decay=TrainingConfig.weight_decay, fused=True)
        print("Optimizer: Fused AdamW")
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.learning_rate,
                                      weight_decay=TrainingConfig.weight_decay)
        print("Optimizer: AdamW (fused not supported)")

    total_updates = len(train_loader) * TrainingConfig.epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16
    accum = TrainingConfig.accumulation_steps

    os.makedirs(TrainingConfig.save_dir, exist_ok=True)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}
    global_step = 0
    t0 = time.time()

    for epoch in range(TrainingConfig.epochs):
        model.train()
        losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TrainingConfig.epochs}")
        for bi, (inp, tgt, tid, pos, mask) in enumerate(pbar):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tid = tid.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, _, loss = model(inp, tid, pos, mask, tgt)

            if loss is None:
                continue

            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        avg_train = sum(losses) / max(len(losses), 1)

        # Validation
        model.eval()
        vlosses = []
        with torch.no_grad():
            for inp, tgt, tid, pos, mask in val_loader:
                inp = inp.to(device); tgt = tgt.to(device)
                tid = tid.to(device); pos = pos.to(device); mask = mask.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, _, loss = model(inp, tid, pos, mask, tgt)
                if loss is not None:
                    vlosses.append(loss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(cur_lr)

        tag = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
            }, TrainingConfig.base_model_path)
            tag = "  -> Saved best"
        print(f"  Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={cur_lr:.2e}  {elapsed:.0f}s  step={global_step}{tag}")

        if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
            break

    with open(os.path.join(TrainingConfig.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
