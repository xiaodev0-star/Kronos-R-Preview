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


class EarlyStopping:
    """Early stopping with patience."""
    def __init__(self, patience=15, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, metric, epoch=0):
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0, entropy_alpha=0.0,
               ignore_index=-100):
    """Focal loss with optimized softmax computation (single log_softmax call)."""
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    mask = (targets != ignore_index).float()
    safe_targets = targets.clamp(min=0)
    with torch.no_grad():
        # Single log_softmax instead of separate softmax — avoids redundant computation
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).exp().clamp(1e-8, 1.0)
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
    if entropy_alpha > 0:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        ent_loss = (entropy * mask).sum() / mask.sum().clamp(min=1)
        loss = loss - entropy_alpha * ent_loss
    return loss


# ============================================================================
# Per-sequence loss aggregation for token-budget batching.
#
# When multiple stocks are packed into one right-padded batch (with is_causal=True
# so real tokens see identical logits to bs=1), we reduce the loss PER SEQUENCE
# and SUM over the batch. This reproduces the exact sequence-weighted gradient of
# the original bs=1 + accumulation loop (each stock contributes one mean loss),
# so loss curves and optimization dynamics are numerically identical to bs=1 --
# only the GPU efficiency changes.
# ============================================================================

def _per_seq_focal(logits, targets, gamma=2.0, label_smoothing=0.0,
                   entropy_alpha=0.0, ignore_index=-100):
    """Row-wise focal loss. logits [B,T,V], targets [B,T] -> [B] per-seq means.

    Algebraically identical to focal_loss() but reduced per row instead of
    globally, so summing the result over B equals the sum of per-sequence
    focal_loss() scalars.
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                         reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing).view(B, T)
    mask = (targets != ignore_index).float()
    safe_targets = targets.clamp(min=0)
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).exp().clamp(1e-8, 1.0)
    focal_weight = (1 - pt) ** gamma
    denom = mask.sum(1).clamp(min=1)
    per_seq = (focal_weight * ce * mask).sum(1) / denom
    if entropy_alpha > 0:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        per_seq = per_seq - entropy_alpha * ((entropy * mask).sum(1) / denom)
    return per_seq


def _per_seq_ce(logits, targets, ignore_index=-100, label_smoothing=0.0):
    """Row-wise cross-entropy. logits [B,T,V], targets [B,T] -> [B] per-seq means."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                         reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing).view(B, T)
    mask = (targets != ignore_index).float()
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


def _per_seq_het(reg_pred, reg_targets_shifted, ignore_val=-999.0):
    """Row-wise heteroscedastic NLL. reg_pred [B,T,2] (float), targets [B,T] -> [B].

    Mirrors heteroscedastic_nll_loss() reduced per row. Masked positions have
    their residual zeroed before squaring so sentinel targets (-999) never
    create large intermediate values.
    """
    mask = (reg_targets_shifted != ignore_val).float()
    mean = reg_pred[..., 0]
    log_var = reg_pred[..., 1].clamp(-5.0, 2.0)
    diff = (reg_targets_shifted - mean) * mask
    nll = 0.5 * (log_var + diff.pow(2) / log_var.exp())
    return (nll * mask).sum(1) / mask.sum(1).clamp(min=1)


def compute_batched_loss(coarse_logits, target, fine_logits, fine_target,
                         reg_pred, reg_target, args):
    """Sum-of-per-sequence total loss for a right-padded batch.

    Returns (loss_sum, coarse_sum, n_seq) where loss_sum aggregates
    focal(+fine+het) over all sequences in the batch, exactly matching what the
    bs=1 loop would accumulate for the same sequences.
    """
    shift_coarse = coarse_logits[:, :-1, :]
    if args.loss == "focal":
        coarse = _per_seq_focal(shift_coarse, target, gamma=args.gamma,
                                label_smoothing=args.label_smoothing,
                                entropy_alpha=args.entropy_alpha)
    else:
        coarse = _per_seq_ce(shift_coarse, target, ignore_index=-100,
                             label_smoothing=args.label_smoothing)
    total = coarse
    shift_fine = fine_logits[:, 1:, :]
    fine = _per_seq_ce(shift_fine, fine_target[:, 1:], ignore_index=0)
    total = total + args.fine_weight * fine
    if reg_pred is not None and args.heteroscedastic:
        het = _per_seq_het(reg_pred, reg_target[:, 1:])
        total = total + args.het_weight * het
    return total.sum(), coarse.sum().detach(), total.shape[0]


def build_wsd_scheduler(optimizer, total_updates, warmup_ratio=0.05, stable_ratio=0.75):
    """WSD (Warmup-Stable-Decay) scheduler — superior to cosine for short training runs.

    Phases:
      - Warmup (5%): linear ramp from 0 to peak lr
      - Stable (75%): constant peak lr
      - Decay (20%): linear decay to 0
    """
    warmup = max(1, int(total_updates * warmup_ratio))
    stable_end = warmup + int(total_updates * stable_ratio)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if step < stable_end:
            return 1.0
        # Linear decay from 1.0 to 0 over remaining steps
        decay_steps = total_updates - stable_end
        if decay_steps <= 0:
            return 1.0
        progress = (step - stable_end) / decay_steps
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _to_device(batch, device):
    """Move an 8-tuple batch to device and ensure batch dimension. mask may be None (is_causal)."""
    inp, tgt, ftgt, tids, pos, mask, va, rt = batch
    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    ftgt = ftgt.to(device, non_blocking=True)
    tids = tids.to(device, non_blocking=True)
    pos = pos.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True) if mask is not None else None
    va = va.to(device, non_blocking=True)
    rt = rt.to(device, non_blocking=True)
    if inp.dim() == 1:
        inp, tgt, ftgt, tids, pos, va, rt = [x.unsqueeze(0) for x in (inp, tgt, ftgt, tids, pos, va, rt)]
        if mask is not None:
            mask = mask.unsqueeze(0)
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
    """Length-sorted batched loader with tolerance-based bucketing and optional curriculum."""
    def __init__(self, sequences, batch_size, shuffle=True, tolerance=500,
                 curriculum_epoch=-1, total_epochs=30):
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tolerance = tolerance
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self._epoch = 0

    def set_epoch(self, epoch):
        """Set current epoch for curriculum filtering."""
        self._epoch = epoch

    def _get_curriculum_max_len(self):
        """Return max sequence length for current epoch (curriculum learning).

        Thresholds are absolute (designed for 30-epoch training):
          - Epoch 0-9:  max 2000 tokens (~70% stocks)
          - Epoch 10-19: max 5000 tokens
          - Epoch 20+:  no limit (all stocks)
        For shorter runs, proportionally compressed thresholds are used.
        """
        if self.curriculum_epoch < 0:
            return 0  # disabled
        epoch = self._epoch
        total = self.total_epochs
        # Compute phase boundaries proportionally, with minimum 1 epoch per phase
        phase1_end = max(1, total * 10 // 30)  # ~33% of training
        phase2_end = max(phase1_end + 1, total * 20 // 30)  # ~67% of training
        if epoch < phase1_end:
            return 2000
        elif epoch < phase2_end:
            return 5000
        return 0  # no limit

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(torch.randint(0, 2**31, (1,)).item())
            indices = torch.randperm(len(self.sequences), generator=rng).tolist()

        # Curriculum filtering
        max_len = self._get_curriculum_max_len()
        if max_len > 0:
            indices = [i for i in indices
                       if self.sequences[i]["input_ids"].shape[0] <= max_len]
            if not indices:
                indices = list(range(len(self.sequences)))  # fallback to all

        # Sort by length for efficient batching (tolerance-based bucketing)
        grouped = sorted(indices, key=lambda i: self.sequences[i]["input_ids"].shape[0])
        buckets = []
        bucket = [grouped[0]] if grouped else []
        for i in grouped[1:]:
            seq_len = self.sequences[i]["input_ids"].shape[0]
            bucket_max = max(self.sequences[j]["input_ids"].shape[0] for j in bucket)
            if seq_len - bucket_max <= self.tolerance:
                bucket.append(i)
            else:
                buckets.append(bucket)
                bucket = [i]
        if bucket:
            buckets.append(bucket)

        # Shuffle buckets for training diversity, then yield batches
        if self.shuffle:
            rng2 = torch.Generator()
            rng2.manual_seed(torch.randint(0, 2**31, (1,)).item())
            perm = torch.randperm(len(buckets), generator=rng2).tolist()
            buckets = [buckets[p] for p in perm]

        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch_idx = bucket[i:i + self.batch_size]
                group = [self.sequences[j] for j in batch_idx]
                yield _pad_batch(group, len(group))[0]

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size


def _pad_batch_causal(group):
    """Right-pad a group of variable-length stocks into one batch, mask=None.

    Returns the same 8-tuple layout as make_dataloader_v2 but with a real batch
    dimension and NO attention mask -- so the model uses SDPA is_causal=True.
    With right-padding + causal attention, every REAL query position i attends
    only to real keys 0..i, so its logits are identical to processing the stock
    alone (bs=1). Padded query rows are discarded by the loss (targets = -100 /
    fine 0 / reg -999), giving numerically identical training to bs=1.
    """
    Nmax = max(s["input_ids"].shape[0] for s in group)
    B = len(group)
    p_ids = torch.zeros(B, Nmax, dtype=torch.long)
    p_tgt = torch.full((B, Nmax - 1), -100, dtype=torch.long)
    p_ftgt = torch.zeros(B, Nmax - 1, dtype=torch.long)
    p_time = torch.zeros(B, Nmax, 3, dtype=torch.long)
    p_pos = torch.zeros(B, Nmax, dtype=torch.long)
    p_va = torch.zeros(B, Nmax, 2, dtype=torch.float32)
    p_rt = torch.full((B, Nmax), -999.0, dtype=torch.float32)
    for k, s in enumerate(group):
        L = s["input_ids"].shape[0]
        Lt = s["targets"].shape[0]
        p_ids[k, :L] = s["input_ids"]
        p_tgt[k, :Lt] = s["targets"]
        p_ftgt[k, :Lt] = s["fine_targets"]
        p_time[k, :L] = s["time_ids"]
        p_pos[k, :L] = s["position_ids"]
        p_va[k, :L] = s["va_values"]
        p_rt[k, :L] = s["reg_targets"]
    return (p_ids, p_tgt, p_ftgt, p_time, p_pos, None, p_va, p_rt)


class TokenBudgetLoader:
    """Adaptive-batch loader: packs stocks into right-padded batches under a token
    budget (B * max_len <= max_tokens), length-sorted to keep padding minimal.

    This is the throughput lever for the tiny (2.7M) GPT: a single stock barely
    occupies the GPU, so we process several per step. Because batches use
    is_causal=True (no explicit mask), real-token logits/losses/grads are
    identical to bs=1 (see compute_batched_loss); only GPU efficiency improves.

    Adaptive B (vs a fixed batch size) is essential: it uses large B for short
    stocks and B=1 for the longest ones, bounding per-step activation memory and
    avoiding the O(B*N^2) attention blow-up that fixed large batches hit.
    """
    def __init__(self, sequences, max_tokens, shuffle=True, cap_B=64,
                 curriculum_epoch=-1, total_epochs=30, band=64):
        self.sequences = sequences
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.cap_B = cap_B
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self.band = band
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _curriculum_max_len(self):
        if self.curriculum_epoch < 0:
            return 0
        epoch, total = self._epoch, self.total_epochs
        phase1_end = max(1, total * 10 // 30)
        phase2_end = max(phase1_end + 1, total * 20 // 30)
        if epoch < phase1_end:
            return 2000
        if epoch < phase2_end:
            return 5000
        return 0

    def _build_groups(self):
        idx = list(range(len(self.sequences)))
        max_len = self._curriculum_max_len()
        if max_len > 0:
            idx = [i for i in idx if self.sequences[i]["input_ids"].shape[0] <= max_len]
            if not idx:
                idx = list(range(len(self.sequences)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(torch.randint(0, 2**31, (1,)).item())
            perm = torch.randperm(len(idx), generator=g).tolist()
            idx = [idx[p] for p in perm]
            # stable sort by coarse length band -> keeps randomness within a band
            idx.sort(key=lambda i: self.sequences[i]["input_ids"].shape[0] // self.band)
        else:
            idx.sort(key=lambda i: self.sequences[i]["input_ids"].shape[0])
        groups, cur, cur_max = [], [], 0
        for i in idx:
            L = self.sequences[i]["input_ids"].shape[0]
            new_max = max(cur_max, L)
            if cur and ((len(cur) + 1) * new_max > self.max_tokens or len(cur) + 1 > self.cap_B):
                groups.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            groups.append(cur)
        return groups

    def __iter__(self):
        groups = self._build_groups()
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(torch.randint(0, 2**31, (1,)).item())
            perm = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[p] for p in perm]
        for grp in groups:
            yield _pad_batch_causal([self.sequences[i] for i in grp])

    def __len__(self):
        return len(self._build_groups())


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
    use_curriculum = getattr(args, "curriculum", False)
    # getattr keeps programmatic callers (sweep scripts that build a bare args
    # object) working; default is ON since it is provably identical to bs=1.
    batch_tokens = getattr(args, "batch_tokens", 12288)
    batch_cap = getattr(args, "batch_cap", 64)
    batched = batch_tokens > 0
    if batched:
        train_loader = TokenBudgetLoader(
            train_seqs, batch_tokens, shuffle=True, cap_B=batch_cap,
            curriculum_epoch=0 if use_curriculum else -1, total_epochs=epochs)
        val_loader = TokenBudgetLoader(
            val_seqs, batch_tokens, shuffle=False, cap_B=batch_cap)
        print(f"Loader: token-budget batched, max_tokens={batch_tokens}, "
              f"cap_B={batch_cap}, accum(seqs)={TrainingConfig.accumulation_steps} "
              f"(right-pad + is_causal; math-identical to bs=1)")
        if use_curriculum:
            print(f"  [CURRICULUM] Phase 1 (ep 1-{epochs//3}): max 2000 tokens, "
                  f"Phase 2 (ep {epochs//3+1}-{2*epochs//3}): max 5000, Phase 3: all")
    elif bs > 1:
        train_loader = BatchedDataLoader(train_seqs, bs, shuffle=True, tolerance=500,
                                          curriculum_epoch=0 if use_curriculum else -1,
                                          total_epochs=epochs)
        val_loader = BatchedDataLoader(val_seqs, bs, shuffle=False, tolerance=500)
        print(f"Loader: batched, bs={bs}, accum={TrainingConfig.accumulation_steps}, tolerance=500")
        if use_curriculum:
            print(f"  [CURRICULUM] Phase 1 (ep 1-{epochs//3}): max 2000 tokens, "
                  f"Phase 2 (ep {epochs//3+1}-{2*epochs//3}): max 5000, Phase 3: all")
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

    # ── Early stopping ──
    early_stop = None
    esp = getattr(args, "early_stop_patience", 0)
    if esp > 0:
        early_stop = EarlyStopping(patience=esp, min_delta=1e-4)
        print(f"  [early_stop] patience={esp}")

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

    amp_dtype = torch.bfloat16
    accum = TrainingConfig.accumulation_steps
    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

    # Compute actual total optimizer steps (accounts for curriculum + dynamic accum)
    n_train_seqs = len(train_seqs)
    if use_curriculum:
        phase1_end = max(1, epochs * 10 // 30)
        phase2_end = max(phase1_end + 1, epochs * 20 // 30)
        n_short = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 2000)
        n_med = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 5000)
        accum_late = accum * 2  # epoch >= 15
        actual_total = (
            phase1_end * max(1, n_short // accum)
            + min(max(0, phase2_end - phase1_end), max(0, 15 - phase1_end)) * max(1, n_med // accum)
            + max(0, min(phase2_end, epochs) - max(phase1_end, 15)) * max(1, n_med // accum_late)
            + max(0, epochs - max(phase2_end, 15)) * max(1, n_train_seqs // accum_late)
        )
    else:
        accum_late = accum * 2
        early_epochs = min(15, epochs)
        late_epochs = max(0, epochs - 15)
        steps_per_epoch = max(1, n_train_seqs // accum)
        steps_per_epoch_late = max(1, n_train_seqs // accum_late)
        actual_total = early_epochs * steps_per_epoch + late_epochs * steps_per_epoch_late

    total_updates = max(actual_total, 1)
    print(f"  Scheduler: total_updates={total_updates} (actual optimizer steps, "
          f"not naive {len(train_loader) * epochs})")

    # WSD scheduler (Warmup-Stable-Decay)
    scheduler = build_wsd_scheduler(optimizer, total_updates,
                                     warmup_ratio=TrainingConfig.warmup_ratio,
                                     stable_ratio=0.75)
    scheduler_adam = build_wsd_scheduler(optimizer_adam, total_updates,
                                          warmup_ratio=TrainingConfig.warmup_ratio,
                                          stable_ratio=0.75) if optimizer_adam else None

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

    # Profiler from args (passed by sweep_bits)
    profiler = getattr(args, "profiler", None)
    if profiler:
        profiler.start("gpt_train")

    epochs_done = 0
    for epoch in range(start_epoch, epochs):
        # Update curriculum epoch for length filtering
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        # For single-seq mode: apply curriculum filter by rebuilding loader each epoch
        # (batched mode filters internally via TokenBudgetLoader.set_epoch).
        if bs == 1 and use_curriculum and not batched:
            phase1_end = max(1, epochs * 10 // 30)
            phase2_end = max(phase1_end + 1, epochs * 20 // 30)
            if epoch < phase1_end:
                max_len = 2000
            elif epoch < phase2_end:
                max_len = 5000
            else:
                max_len = 0  # no limit
            if max_len > 0:
                filtered = [s for s in train_seqs if s["input_ids"].shape[0] <= max_len]
                if filtered:
                    train_loader = make_dataloader_v2(filtered, batch_size=1, shuffle=True)
                # else: keep full dataset if filter is too aggressive

        # Dynamic accumulation: double accum after epoch 15 for stronger gradient signal
        # epoch 0-14: accum=32 (effective batch=32)
        # epoch 15+:  accum=64 (effective batch=64, more stable gradients)
        epoch_accum = accum if epoch < 15 else accum * 2
        if epoch == 15:
            print(f"  [accum] Doubling accumulation: {accum} -> {epoch_accum} "
                  f"(effective batch {bs * epoch_accum})")

        model.train()
        loss_acc = torch.zeros((), device=device)
        n_loss = 0
        optimizer.zero_grad(set_to_none=True)
        if optimizer_adam:
            optimizer_adam.zero_grad(set_to_none=True)

        # Profiler sub-timing accumulators (sampled every 20 steps)
        t_forward_acc = 0.0
        t_backward_acc = 0.0

        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{epochs}",
                    ncols=80)
        seqs_in_accum = 0  # batched mode: sequences accumulated toward one opt step
        for bi, batch in enumerate(pbar):
            input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target = _to_device(batch, device)

            try:
                if batched:
                    # Token-budget batched path: per-sequence loss summed over the
                    # batch (identical gradient to bs=1), step every `epoch_accum`
                    # SEQUENCES so the effective batch matches the single-seq loop.
                    t_fwd_start = time.perf_counter() if (bi + 1) % 20 == 0 else 0
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        if args.heteroscedastic:
                            coarse_logits, fine_logits, reg_pred, _ = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, reg_targets=reg_target,
                                fine_targets=fine_target)
                        else:
                            coarse_logits, fine_logits = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, fine_targets=fine_target)
                            reg_pred = None
                        loss_sum, _, n_seq = compute_batched_loss(
                            coarse_logits, target, fine_logits, fine_target,
                            reg_pred, reg_target, args)
                    if t_fwd_start > 0:
                        t_forward_acc += time.perf_counter() - t_fwd_start
                    t_bwd_start = time.perf_counter() if (bi + 1) % 20 == 0 else 0
                    (loss_sum / epoch_accum).backward()
                    if t_bwd_start > 0:
                        t_backward_acc += time.perf_counter() - t_bwd_start
                    seqs_in_accum += n_seq
                    if seqs_in_accum >= epoch_accum:
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
                        seqs_in_accum = 0
                    loss_acc += loss_sum.detach()
                    n_loss += n_seq
                    if (bi + 1) % 20 == 0:
                        pbar.set_postfix({"loss": f"{loss_acc.item() / max(n_loss, 1):.4f}",
                                          "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                    if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                        break
                    continue

                # Forward pass (timed every 20 steps)
                t_fwd_start = time.perf_counter() if (bi + 1) % 20 == 0 else 0
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
                    shift_fine = fine_logits[:, 1:, :].contiguous()    # [B, S-2, V_fine]
                    shift_fine_tgt = fine_target[:, 1:].contiguous() # [B, S-2]
                    fine_mask = (shift_targets != -100)
                    if fine_mask.any():
                        fine_loss = F.cross_entropy(
                            shift_fine.view(-1, shift_fine.size(-1)),
                            shift_fine_tgt.view(-1), ignore_index=0)
                        loss = loss + args.fine_weight * fine_loss

                    if het_loss is not None and args.heteroscedastic:
                        loss = loss + args.het_weight * het_loss

                if t_fwd_start > 0:
                    t_forward_acc += time.perf_counter() - t_fwd_start

                if loss is None:
                    continue

                # Backward pass (timed every 20 steps)
                t_bwd_start = time.perf_counter() if (bi + 1) % 20 == 0 else 0
                (loss / epoch_accum).backward()
                if t_bwd_start > 0:
                    t_backward_acc += time.perf_counter() - t_bwd_start

                if (bi + 1) % epoch_accum == 0 or (bi + 1) == len(train_loader):
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

            except torch.cuda.OutOfMemoryError:
                # OOM fallback: skip this batch, clear cache
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_adam:
                    optimizer_adam.zero_grad(set_to_none=True)
                seqs_in_accum = 0  # discard the partial accumulation window
                print(f"  [OOM] Skipped batch {bi+1} (seq_len={input_ids.shape[-1]}, bs={input_ids.shape[0]})")
                continue

            # Update pbar every 20 steps (avoids per-step .item() sync)
            if (bi + 1) % 20 == 0:
                pbar.set_postfix({"loss": f"{loss_acc.item() / n_loss:.4f}",
                                  "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        # Flush any remaining accumulated gradient (batched mode leftover < accum)
        if batched and seqs_in_accum > 0:
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
            seqs_in_accum = 0

        # Record profiler sub-timings (sampled, not exact per-step)
        if profiler:
            if t_forward_acc > 0:
                profiler.record("gpt_forward", t_forward_acc)
            if t_backward_acc > 0:
                profiler.record("gpt_backward", t_backward_acc)

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
                        coarse_logits, fine_logits, reg_pred, val_het = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, reg_targets=reg_target,
                            fine_targets=fine_target)
                    else:
                        coarse_logits, fine_logits = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, fine_targets=fine_target)
                        reg_pred = None
                        val_het = None
                    shift_logits = coarse_logits[:, :-1, :].contiguous()
                    shift_targets = target.contiguous()
                    if batched:
                        # Per-sequence means -> val_loss stays sequence-averaged
                        # (identical metric to the bs=1 loop).
                        vlosses.extend(_per_seq_ce(shift_logits, shift_targets,
                                                   ignore_index=-100).tolist())
                        if args.heteroscedastic and reg_pred is not None:
                            v_het_losses.extend(
                                _per_seq_het(reg_pred, reg_target[:, 1:]).tolist())
                    else:
                        if val_het is not None:
                            v_het_losses.append(val_het.item())
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
        sd = model.state_dict()
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": sd,
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                           "vocab_size": ModelConfig.vocab_size, "vocab_fine": ModelConfig.vocab_fine},
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

        # Per-epoch checkpoint (for inference reuse of any epoch's weights)
        epoch_ckpt_path = save_path.replace(".pt", f"_ep{epoch+1}.pt")
        torch.save({
            "model_state_dict": sd,
            "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                       "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                       "vocab_size": ModelConfig.vocab_size, "vocab_fine": ModelConfig.vocab_fine},
            "val_loss": avg_val,
            "epoch": epoch,
            "tag": args.tag,
        }, epoch_ckpt_path)

        # Resume checkpoint (for training resume)
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

        # Early stopping check
        if early_stop and early_stop(avg_val, epoch):
            print(f"  [early_stop] Patience exhausted at epoch {epoch+1} "
                  f"(best={early_stop.best:.4f} at epoch {early_stop.best_epoch+1}). Stopping.")
            break

        if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
            break

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    with open(os.path.join(os.path.dirname(save_path) or ".", f"history_{args.tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    if profiler:
        profiler.end("gpt_train")

    print(f"\nDone. Best val_loss: {best_val:.4f} (after {epochs_done} epochs)")


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
    # ── Training improvement args ──
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience (0=disabled, recommended 3-5 for GPT)")
    # ── Speed: token-budget batching (right-pad + is_causal, math-identical to bs=1) ──
    parser.add_argument("--batch_tokens", type=int, default=12288,
                        help="Max tokens per batch (B*max_len). Adaptive batching for "
                             "~1.7x faster GPT training. 0 = legacy single-seq (bs=1).")
    parser.add_argument("--batch_cap", type=int, default=64,
                        help="Hard cap on sequences per batch (safety for very short stocks)")
    main(parser.parse_args())
