"""Train base model: KronosPreview on packed stock sequences.
Supports epoch-level checkpoint/resume via save_path + save_path.ckpt.
Supports: focal loss, weight_decay override, reasoning module."""
import argparse
import math
import os
import random
import time
import json

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from model import load_tokenizer
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from model.optimizer import build_muon_optimizers
from training_utils import clip_grad_norm_


def _atomic_torch_save(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _distribution_summary(counts):
    counts = counts.astype(np.int64, copy=False)
    total = int(counts.sum())
    used = counts[counts > 0]
    if total == 0:
        return {
            "total": 0,
            "n_unique": 0,
            "collapse_rate": 0.0,
            "entropy_bits": 0.0,
            "effective_tokens": 0.0,
        }
    probabilities = used.astype(np.float64) / total
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "total": total,
        "n_unique": int(used.size),
        "collapse_rate": float(used.max() / total),
        "entropy_bits": entropy,
        "effective_tokens": float(2.0 ** entropy),
    }


def _write_dataset_token_diagnostics(
    metrics_dir,
    train_sequences,
    val_sequences,
    vocab_coarse,
    vocab_fine,
):
    """Write exact train/validation target distributions for offline audit."""
    arrays = {
        "schema": np.asarray([1], dtype=np.int16),
        "vocab_coarse": np.asarray([vocab_coarse], dtype=np.int32),
        "vocab_fine": np.asarray([vocab_fine], dtype=np.int32),
        "vocab_joint": np.asarray(
            [vocab_coarse * vocab_fine], dtype=np.int32
        ),
    }
    summary = {"schema": 1, "splits": {}}
    for split_name, sequences in (
        ("train", train_sequences),
        ("validation", val_sequences),
    ):
        coarse_counts = np.zeros(vocab_coarse, dtype=np.int64)
        fine_counts = np.zeros(vocab_fine, dtype=np.int64)
        joint_counts = np.zeros(vocab_coarse * vocab_fine, dtype=np.int64)
        lengths = []
        for sequence in sequences:
            coarse = sequence["targets"].numpy().astype(
                np.int64, copy=False
            )
            fine = sequence["fine_targets"].numpy().astype(
                np.int64, copy=False
            )
            valid = (
                (coarse >= 0)
                & (coarse < vocab_coarse)
                & (fine >= 0)
                & (fine < vocab_fine)
            )
            coarse = coarse[valid]
            fine = fine[valid]
            joint = coarse * vocab_fine + fine
            coarse_counts += np.bincount(
                coarse, minlength=vocab_coarse
            )[:vocab_coarse]
            fine_counts += np.bincount(
                fine, minlength=vocab_fine
            )[:vocab_fine]
            joint_counts += np.bincount(
                joint, minlength=vocab_coarse * vocab_fine
            )[: vocab_coarse * vocab_fine]
            lengths.append(int(valid.sum()))
        arrays[f"{split_name}_coarse_counts"] = coarse_counts
        arrays[f"{split_name}_fine_counts"] = fine_counts
        arrays[f"{split_name}_joint_counts"] = joint_counts
        length_array = np.asarray(lengths, dtype=np.int64)
        summary["splits"][split_name] = {
            "n_sequences": len(sequences),
            "sequence_target_length": {
                "min": int(length_array.min()) if length_array.size else 0,
                "p10": (
                    float(np.quantile(length_array, 0.10))
                    if length_array.size
                    else 0.0
                ),
                "median": (
                    float(np.median(length_array))
                    if length_array.size
                    else 0.0
                ),
                "p90": (
                    float(np.quantile(length_array, 0.90))
                    if length_array.size
                    else 0.0
                ),
                "max": int(length_array.max()) if length_array.size else 0,
            },
            "coarse": _distribution_summary(coarse_counts),
            "fine": _distribution_summary(fine_counts),
            "joint": _distribution_summary(joint_counts),
        }

    distribution_path = os.path.join(
        metrics_dir, "dataset_token_distributions.npz"
    )
    temporary_distribution = distribution_path + ".tmp"
    with open(temporary_distribution, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_distribution, distribution_path)
    summary["distribution_sidecar"] = {
        "filename": os.path.basename(distribution_path),
        "arrays": sorted(arrays),
        "size_bytes": os.path.getsize(distribution_path),
    }
    summary_path = os.path.join(metrics_dir, "dataset_token_summary.json")
    temporary_summary = summary_path + ".tmp"
    with open(temporary_summary, "w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    os.replace(temporary_summary, summary_path)


def _cpu_state_dict(module):
    """Snapshot model tensors to CPU once for all per-epoch checkpoint files."""
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def _rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(checkpoint):
    restored = False
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
        restored = True
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
        restored = True
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        restored = True
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )
        restored = True
    return restored


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
    """Focal loss without a redundant softmax on the common configuration."""
    ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    mask = (targets != ignore_index).float()
    log_probs = None
    if label_smoothing == 0.0 and entropy_alpha == 0.0:
        # CE is -log(p_target) without label smoothing.  The focal weight is
        # intentionally detached, so this removes a full extra log_softmax
        # without changing the gradient path.
        with torch.no_grad():
            pt = (-ce).exp().clamp(1e-8, 1.0)
    else:
        safe_targets = targets.clamp(min=0)
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            pt = (
                log_probs.gather(-1, safe_targets.unsqueeze(-1))
                .squeeze(-1)
                .exp()
                .clamp(1e-8, 1.0)
            )
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
    if entropy_alpha > 0:
        if log_probs is None:
            log_probs = F.log_softmax(logits, dim=-1)
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
                         reduction="none", ignore_index=ignore_index,
                         label_smoothing=label_smoothing).view(B, T)
    mask = (targets != ignore_index).float()
    log_probs = None
    if label_smoothing == 0.0 and entropy_alpha == 0.0:
        with torch.no_grad():
            pt = (-ce).exp().clamp(1e-8, 1.0)
    else:
        safe_targets = targets.clamp(min=0)
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            pt = (
                log_probs.gather(-1, safe_targets.unsqueeze(-1))
                .squeeze(-1)
                .exp()
                .clamp(1e-8, 1.0)
            )
    focal_weight = (1 - pt) ** gamma
    denom = mask.sum(1).clamp(min=1)
    per_seq = (focal_weight * ce * mask).sum(1) / denom
    if entropy_alpha > 0:
        if log_probs is None:
            log_probs = F.log_softmax(logits, dim=-1)
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


def _mean_metric_chunks(chunks):
    """Average validation chunks with at most one device-to-host sync."""
    if not chunks:
        return 0.0
    if isinstance(chunks[0], torch.Tensor):
        values = torch.cat([chunk.reshape(-1) for chunk in chunks]).cpu().tolist()
    else:
        values = chunks
    return sum(values) / max(len(values), 1)


def compute_batched_loss(coarse_logits, target, fine_logits, fine_target,
                         reg_pred, reg_target, args):
    """Sum-of-per-sequence total loss for a right-padded batch.

    Returns ``(loss_sum, component_sums, n_seq)``. Component sums are detached
    unweighted per-sequence losses, making the downloadable training history
    sufficient to separate coarse, fine, and regression behaviour.
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
    fine = _per_seq_ce(
        fine_logits, fine_target, ignore_index=-100
    )
    total = total + args.fine_weight * fine
    het = torch.zeros_like(coarse)
    if reg_pred is not None and args.heteroscedastic:
        het = _per_seq_het(reg_pred, reg_target[:, 1:])
        total = total + args.het_weight * het
    components = {
        "coarse": coarse.sum().detach(),
        "fine": fine.sum().detach(),
        "heteroscedastic": het.sum().detach(),
    }
    return total.sum(), components, total.shape[0]


def build_wsd_scheduler(optimizer, total_updates, warmup_ratio=0.05, stable_ratio=0.0):
    """Warmup + Cosine Decay scheduler.

    Phases:
      - Warmup (5%): linear ramp from 0 to peak lr
      - Cosine Decay (95%): cosine anneal from peak to 0

    For fast-converging small models, a long stable phase wastes training budget
    at constant high LR. Cosine decay after warmup lets the LR decrease
    continuously, ensuring convergence even when early stopping triggers early.

    stable_ratio=0 (default) gives pure warmup+cosine.
    stable_ratio>0 inserts a constant-LR plateau before cosine decay starts.
    """
    warmup = max(1, int(total_updates * warmup_ratio))
    stable_end = warmup + int(total_updates * min(stable_ratio, 0.60))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if step < stable_end:
            return 1.0
        # Cosine decay from 1.0 to 0 over remaining steps
        decay_steps = max(1, total_updates - stable_end)
        progress = min(1.0, max(0.0, (step - stable_end) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

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
        p_ftgt = torch.full(
            (B, max_len - 1), -100, dtype=torch.long
        )
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
    fine -100 / reg -999), giving numerically identical training to bs=1.
    """
    Nmax = max(s["input_ids"].shape[0] for s in group)
    B = len(group)
    p_ids = torch.zeros(B, Nmax, dtype=torch.long)
    p_tgt = torch.full((B, Nmax - 1), -100, dtype=torch.long)
    p_ftgt = torch.full((B, Nmax - 1), -100, dtype=torch.long)
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
    def __init__(
        self,
        sequences,
        max_tokens,
        shuffle=True,
        cap_B=64,
        curriculum_epoch=-1,
        total_epochs=30,
        band=64,
        loader_seed=None,
        exact_accumulation=False,
    ):
        self.sequences = sequences
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.cap_B = cap_B
        self.curriculum_epoch = curriculum_epoch
        self.total_epochs = total_epochs
        self.band = band
        self._epoch = 0
        self.loader_seed = loader_seed
        self.exact_accumulation = exact_accumulation
        self.accumulation_boundary = 0
        self.last_iteration_stats = {}

    def set_epoch(self, epoch):
        self._epoch = epoch

    def set_accumulation_boundary(self, boundary):
        self.accumulation_boundary = int(boundary)

    def _generator(self, stream):
        generator = torch.Generator()
        if self.loader_seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        else:
            # Local, epoch-addressable RNG: model initialization/dropout can no
            # longer perturb data order in controlled architecture studies.
            seed = (
                int(self.loader_seed)
                + 1_000_003 * int(self._epoch)
                + int(stream)
            ) % (2**63 - 1)
        generator.manual_seed(seed)
        return generator

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
            g = self._generator(0)
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

    def _build_exact_group_blocks(self):
        """Pack deterministic blocks that end exactly on optimizer boundaries."""
        boundary = int(self.accumulation_boundary)
        if boundary <= 0:
            raise RuntimeError(
                "Exact accumulation requires a positive epoch boundary"
            )
        idx = list(range(len(self.sequences)))
        max_len = self._curriculum_max_len()
        if max_len > 0:
            idx = [
                i
                for i in idx
                if self.sequences[i]["input_ids"].shape[0] <= max_len
            ]
            if not idx:
                idx = list(range(len(self.sequences)))
        if self.shuffle:
            permutation = torch.randperm(
                len(idx), generator=self._generator(0)
            ).tolist()
            idx = [idx[position] for position in permutation]

        # The controlled schedule follows the floor-based optimizer budget used
        # by the scheduler and experiment protocol.  Drop the final incomplete
        # accumulation *before* length sorting so omitted sequences rotate
        # deterministically instead of always being the longest documents.
        usable = len(idx) // boundary * boundary
        if usable == 0:
            raise RuntimeError(
                "Exact accumulation has fewer usable sequences "
                f"({len(idx)}) than its boundary ({boundary})"
            )
        idx = idx[:usable]
        idx.sort(
            key=lambda i: self.sequences[i]["input_ids"].shape[0] // self.band
        )

        blocks = []
        for start in range(0, len(idx), boundary):
            block_indices = idx[start : start + boundary]
            groups, current, current_max = [], [], 0
            for sequence_index in block_indices:
                length = self.sequences[sequence_index]["input_ids"].shape[0]
                new_max = max(current_max, length)
                exceeds_tokens = (
                    current
                    and (len(current) + 1) * new_max > self.max_tokens
                )
                exceeds_cap = current and len(current) + 1 > self.cap_B
                if exceeds_tokens or exceeds_cap:
                    groups.append(current)
                    current, current_max = [sequence_index], length
                else:
                    current.append(sequence_index)
                    current_max = new_max
            if current:
                groups.append(current)
            if sum(len(group) for group in groups) != boundary:
                raise RuntimeError("Exact accumulation block was packed incorrectly")
            blocks.append(groups)
        if self.shuffle and blocks:
            permutation = torch.randperm(
                len(blocks), generator=self._generator(1)
            ).tolist()
            blocks = [blocks[position] for position in permutation]
        return blocks

    def _reset_iteration_stats(self):
        self.last_iteration_stats = {
            "microbatches": 0,
            "sequences": 0,
            "real_tokens": 0,
            "padded_tokens": 0,
            "max_sequences_per_microbatch": 0,
            "max_sequence_length": 0,
        }

    def _record_group(self, group):
        lengths = [
            int(self.sequences[index]["input_ids"].shape[0])
            for index in group
        ]
        batch_size = len(lengths)
        max_length = max(lengths)
        stats = self.last_iteration_stats
        stats["microbatches"] += 1
        stats["sequences"] += batch_size
        stats["real_tokens"] += sum(lengths)
        stats["padded_tokens"] += batch_size * max_length
        stats["max_sequences_per_microbatch"] = max(
            stats["max_sequences_per_microbatch"], batch_size
        )
        stats["max_sequence_length"] = max(
            stats["max_sequence_length"], max_length
        )

    def __iter__(self):
        self._reset_iteration_stats()
        if self.exact_accumulation:
            for block in self._build_exact_group_blocks():
                for group in block:
                    self._record_group(group)
                    yield _pad_batch_causal(
                        [self.sequences[i] for i in group]
                    )
            return
        groups = self._build_groups()
        if self.shuffle:
            g = self._generator(1)
            perm = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[p] for p in perm]
        for grp in groups:
            self._record_group(grp)
            yield _pad_batch_causal([self.sequences[i] for i in grp])

    def __len__(self):
        if self.exact_accumulation:
            return sum(
                len(block) for block in self._build_exact_group_blocks()
            )
        return len(self._build_groups())


def main(args):
    # Deterministic algorithms are off by default because they do not actually
    # make this trainer reproducible: the memory-efficient SDPA backward is
    # non-deterministic and `warn_only=True` lets it through, so two identical
    # passes already differ (18/38 gradient tensors, verified in Exp 05).  The
    # setting cost 4.3% of step time for nothing.  Seeding still fixes init,
    # data order and dropout.  Pass --deterministic to restore the old flags.
    set_global_seed(TrainingConfig.random_seed,
                    deterministic=getattr(args, "deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT standard architecture (HPO 2026-06-18 best: phase3_t000, DA 48.12% with V2).
    # Explicit CLI overrides are required by the architecture-scaling experiment;
    # otherwise retain the production baseline so prior in-process mutations cannot
    # leak into an ordinary run.
    ModelConfig.dim = args.dim or 256
    ModelConfig.depth = args.depth or 2
    ModelConfig.heads = args.heads or 4
    ModelConfig.num_kv_heads = args.num_kv_heads or 1
    ModelConfig.dropout = args.dropout
    ModelConfig.ffn_multiplier = args.ffn_multiplier or 4
    ModelConfig.position_encoding = "rope"
    ModelConfig.rope_base = 10000.0
    ModelConfig.vocab_size = 1024
    ModelConfig.va_hidden_dim = 64
    if ModelConfig.dim <= 0 or ModelConfig.depth <= 0:
        raise ValueError("Model dim and depth must be positive")
    if ModelConfig.heads <= 0 or ModelConfig.dim % ModelConfig.heads:
        raise ValueError(
            f"dim={ModelConfig.dim} must be divisible by heads={ModelConfig.heads}"
        )
    if (
        ModelConfig.num_kv_heads <= 0
        or ModelConfig.heads % ModelConfig.num_kv_heads
    ):
        raise ValueError(
            f"heads={ModelConfig.heads} must be divisible by "
            f"num_kv_heads={ModelConfig.num_kv_heads}"
        )
    if ModelConfig.ffn_multiplier <= 0:
        raise ValueError("ffn_multiplier must be positive")
    if args.gradient_checkpointing is not None:
        TrainingConfig.use_gradient_checkpointing = args.gradient_checkpointing

    save_path = args.save_path
    tok_path = args.tokenizer_path
    epochs = args.epochs
    ckpt_path = save_path + ".ckpt"
    save_stem, _ = os.path.splitext(save_path)
    metrics_dir = os.path.abspath(
        args.metrics_dir
        if getattr(args, "metrics_dir", "")
        else (os.path.dirname(save_path) or ".")
    )
    os.makedirs(metrics_dir, exist_ok=True)
    epoch_ckpt_index_path = os.path.join(
        metrics_dir, os.path.basename(save_stem) + "_checkpoints.json"
    )
    effective_lr = args.lr

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}, tok={tok_path}, ep={epochs}")
    print(f"  loss={args.loss}, gamma={args.gamma}, wd={args.weight_decay}, lr={effective_lr}")
    print(
        "  architecture="
        f"dim{ModelConfig.dim}/depth{ModelConfig.depth}/heads{ModelConfig.heads}/"
        f"kv{ModelConfig.num_kv_heads}/ffn{ModelConfig.ffn_multiplier}, "
        f"gradient_checkpointing={TrainingConfig.use_gradient_checkpointing}"
    )
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
    _write_dataset_token_diagnostics(
        metrics_dir,
        train_seqs,
        val_seqs,
        ModelConfig.vocab_size,
        ModelConfig.vocab_fine,
    )

    bs = TrainingConfig.batch_size
    use_curriculum = getattr(args, "curriculum", False)
    # getattr keeps programmatic callers (sweep scripts that build a bare args
    # object) working; default is ON since it is provably identical to bs=1.
    batch_tokens = getattr(args, "batch_tokens", 12288)
    batch_cap = getattr(args, "batch_cap", 64)
    controlled_loader_seed = getattr(args, "controlled_loader_seed", -1)
    exact_accumulation = getattr(
        args, "exact_accumulation_boundaries", False
    )
    batched = batch_tokens > 0
    if exact_accumulation and not batched:
        raise ValueError(
            "--exact_accumulation_boundaries requires --batch_tokens > 0"
        )
    if exact_accumulation and controlled_loader_seed < 0:
        raise ValueError(
            "--exact_accumulation_boundaries requires "
            "--controlled_loader_seed >= 0"
        )
    if batched:
        train_loader = TokenBudgetLoader(
            train_seqs, batch_tokens, shuffle=True, cap_B=batch_cap,
            curriculum_epoch=0 if use_curriculum else -1, total_epochs=epochs,
            loader_seed=(
                controlled_loader_seed
                if controlled_loader_seed >= 0
                else None
            ),
            exact_accumulation=exact_accumulation)
        val_loader = TokenBudgetLoader(
            val_seqs, batch_tokens, shuffle=False, cap_B=batch_cap)
        print(f"Loader: token-budget batched, max_tokens={batch_tokens}, "
              f"cap_B={batch_cap}, accum(seqs)={TrainingConfig.accumulation_steps} "
              f"(right-pad + is_causal; math-identical to bs=1)")
        if exact_accumulation:
            print(
                "  [CONTROLLED] loader_seed="
                f"{controlled_loader_seed}, exact accumulation boundaries"
            )
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
    constant_accumulation = getattr(args, "constant_accumulation", False)
    print(
        "  Accumulation policy: "
        + (
            f"constant {accum} sequences/update"
            if constant_accumulation
            else f"{accum} sequences/update, doubling at epoch 16"
        )
    )
    trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

    # Compute the optimizer-step budget from the exact accumulation policy.  The
    # historical default doubles accumulation at absolute epoch 15.  Exp 04 can
    # opt out with --constant_accumulation after the 2026-07-27 audit showed that
    # the larger batch removes updates without reducing forward/backward work.
    n_train_seqs = len(train_seqs)
    phase1_end = max(1, epochs * 10 // 30)
    phase2_end = max(phase1_end + 1, epochs * 20 // 30)
    n_short = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 2000)
    n_med = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 5000)

    def accumulation_for_epoch(epoch_index):
        if constant_accumulation or epoch_index < 15:
            return accum
        return accum * 2

    def sequences_for_epoch(epoch_index):
        if not use_curriculum:
            return n_train_seqs
        if epoch_index < phase1_end:
            return n_short
        if epoch_index < phase2_end:
            return n_med
        return n_train_seqs

    if exact_accumulation:
        train_loader.set_accumulation_boundary(accumulation_for_epoch(0))

    actual_total = sum(
        max(1, sequences_for_epoch(epoch_index)
            // accumulation_for_epoch(epoch_index))
        for epoch_index in range(epochs)
    )

    total_updates = max(actual_total, 1)
    print(f"  Scheduler: total_updates={total_updates} (actual optimizer steps, "
          f"not naive {len(train_loader) * epochs})")

    # Cosine scheduler: warmup → continuous cosine decay
    # No stable plateau — LR starts decreasing immediately after warmup,
    # ensuring convergence even when early stopping triggers early.
    scheduler = build_wsd_scheduler(optimizer, total_updates,
                                     warmup_ratio=TrainingConfig.warmup_ratio)
    scheduler_adam = build_wsd_scheduler(optimizer_adam, total_updates,
                                          warmup_ratio=TrainingConfig.warmup_ratio) if optimizer_adam else None

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
            restored_rng = _restore_rng_state(ckpt)
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}, step={global_step}")
            if not restored_rng:
                print("  [warning] Legacy checkpoint has no RNG state; resume is not bit-exact")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume from checkpoint (incompatible): {e}")
            print(f"  Starting fresh training.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    history = {
        "schema": 4,
        "epoch": [],
        "train_loss": [],
        "train_coarse_loss": [],
        "train_fine_loss": [],
        "train_het_loss": [],
        "val_loss": [],
        "val_coarse_loss": [],
        "val_fine_loss": [],
        "val_het_loss": [],
        "lr": [],
        "lr_adam": [],
        "optimizer_steps_this_epoch": [],
        "global_step": [],
        "accumulation_sequences": [],
        "scheduled_train_sequences": [],
        "train_microbatches": [],
        "mean_sequences_per_microbatch": [],
        "max_sequences_per_microbatch": [],
        "real_input_tokens": [],
        "padded_input_tokens": [],
        "padding_efficiency": [],
        "max_sequence_length": [],
        "peak_cuda_memory_allocated_gb": [],
        "peak_cuda_memory_reserved_gb": [],
        "epoch_time_s": [],
        "elapsed_time_s": [],
    }
    history_path = os.path.join(metrics_dir, f"history_{args.tag}.json")
    if args.history_per_epoch and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                stored_history = json.load(f)
            for key in history:
                if key == "schema":
                    continue
                stored_values = stored_history.get(key)
                if stored_values is None and key == "val_coarse_loss":
                    stored_values = stored_history.get("val_loss", [])
                if stored_values is None:
                    history[key] = [None] * start_epoch
                else:
                    history[key] = list(stored_values)[:start_epoch]
            for key in ("collapse_rate", "n_unique_tokens"):
                if key in stored_history:
                    history[key] = list(stored_history[key])[:start_epoch]
            print(f"  Restored history through epoch {start_epoch}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"  [warning] Could not restore history: {exc}")
    t0 = time.time()

    # Profiler from args (passed by sweep_bits)
    profiler = getattr(args, "profiler", None)
    if profiler:
        profiler.start("gpt_train")

    epochs_done = 0
    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.time()
        epoch_step_start = global_step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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

        # Historical default: double accumulation after epoch 15.  Formal Exp 04
        # passes --constant_accumulation and keeps the better-audited 32-sequence
        # update batch throughout.
        epoch_accum = accumulation_for_epoch(epoch)
        if hasattr(train_loader, "set_accumulation_boundary"):
            train_loader.set_accumulation_boundary(epoch_accum)
        if epoch == 15 and not constant_accumulation:
            print(f"  [accum] Doubling accumulation: {accum} -> {epoch_accum} "
                  f"(effective batch {bs * epoch_accum})")

        model.train()
        loss_acc = torch.zeros((), device=device)
        coarse_loss_acc = torch.zeros((), device=device)
        fine_loss_acc = torch.zeros((), device=device)
        het_loss_acc = torch.zeros((), device=device)
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
                    # batch (identical gradient to bs=1). Historical mode steps
                    # after the count reaches `epoch_accum` and may overshoot;
                    # exact mode packs blocks that end exactly on the boundary.
                    t_fwd_start = (
                        time.perf_counter()
                        if profiler and (bi + 1) % 20 == 0
                        else 0
                    )
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        if args.heteroscedastic:
                            coarse_logits, fine_logits, reg_pred, _ = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, reg_targets=reg_target,
                                fine_targets=fine_target,
                                compute_reg_loss=False)
                        else:
                            coarse_logits, fine_logits = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, fine_targets=fine_target)
                            reg_pred = None
                        loss_sum, component_sums, n_seq = compute_batched_loss(
                            coarse_logits, target, fine_logits, fine_target,
                            reg_pred, reg_target, args)
                    if t_fwd_start > 0:
                        t_forward_acc += time.perf_counter() - t_fwd_start
                    t_bwd_start = (
                        time.perf_counter()
                        if profiler and (bi + 1) % 20 == 0
                        else 0
                    )
                    (loss_sum / epoch_accum).backward()
                    if t_bwd_start > 0:
                        t_backward_acc += time.perf_counter() - t_bwd_start
                    seqs_in_accum += n_seq
                    if seqs_in_accum >= epoch_accum:
                        clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
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
                    coarse_loss_acc += component_sums["coarse"]
                    fine_loss_acc += component_sums["fine"]
                    het_loss_acc += component_sums["heteroscedastic"]
                    n_loss += n_seq
                    if (bi + 1) % 100 == 0:
                        pbar.set_postfix(
                            {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                        )
                    if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                        break
                    continue

                # Forward pass (timed every 20 steps)
                t_fwd_start = (
                    time.perf_counter()
                    if profiler and (bi + 1) % 20 == 0
                    else 0
                )
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
                    coarse_component = loss.detach()
                    fine_component = torch.zeros((), device=device)
                    het_component = torch.zeros((), device=device)

                    # Fine head predicts the same next-token positions as the
                    # coarse head. -100 marks EOS/padding; code 0 is valid.
                    fine_mask = fine_target != -100
                    if fine_mask.any():
                        fine_loss = F.cross_entropy(
                            fine_logits.reshape(-1, fine_logits.size(-1)),
                            fine_target.reshape(-1),
                            ignore_index=-100,
                        )
                        fine_component = fine_loss.detach()
                        loss = loss + args.fine_weight * fine_loss

                    if het_loss is not None and args.heteroscedastic:
                        het_component = het_loss.detach()
                        loss = loss + args.het_weight * het_loss

                if t_fwd_start > 0:
                    t_forward_acc += time.perf_counter() - t_fwd_start

                if loss is None:
                    continue

                # Backward pass (timed every 20 steps)
                t_bwd_start = (
                    time.perf_counter()
                    if profiler and (bi + 1) % 20 == 0
                    else 0
                )
                (loss / epoch_accum).backward()
                if t_bwd_start > 0:
                    t_backward_acc += time.perf_counter() - t_bwd_start

                if (bi + 1) % epoch_accum == 0 or (bi + 1) == len(train_loader):
                    clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
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
                coarse_loss_acc += coarse_component
                fine_loss_acc += fine_component
                het_loss_acc += het_component
                n_loss += 1

            except torch.cuda.OutOfMemoryError as error:
                if exact_accumulation:
                    raise RuntimeError(
                        "OOM in controlled exact-accumulation mode; aborting "
                        "instead of changing the optimizer-step/data schedule"
                    ) from error
                # OOM fallback: skip this batch, clear cache
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_adam:
                    optimizer_adam.zero_grad(set_to_none=True)
                seqs_in_accum = 0  # discard the partial accumulation window
                print(f"  [OOM] Skipped batch {bi+1} (seq_len={input_ids.shape[-1]}, bs={input_ids.shape[0]})")
                continue

            # Keep the CUDA stream asynchronous; loss is synchronized once at
            # epoch end instead of every progress-bar refresh.
            if (bi + 1) % 100 == 0:
                pbar.set_postfix(
                    {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                )

            if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
                break

        # Flush any remaining accumulated gradient (batched mode leftover < accum)
        if batched and seqs_in_accum > 0:
            clip_grad_norm_(trainable_params, TrainingConfig.grad_clip)
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

        epoch_optimizer_steps = global_step - epoch_step_start
        if exact_accumulation:
            expected_epoch_steps = (
                sequences_for_epoch(epoch) // epoch_accum
            )
            if epoch_optimizer_steps != expected_epoch_steps:
                raise RuntimeError(
                    "Controlled optimizer-step invariant failed at epoch "
                    f"{epoch + 1}: observed {epoch_optimizer_steps}, "
                    f"expected {expected_epoch_steps}"
                )
        batch_stats = (
            dict(train_loader.last_iteration_stats)
            if batched
            else {}
        )

        # Record profiler sub-timings (sampled, not exact per-step)
        if profiler:
            if t_forward_acc > 0:
                profiler.record("gpt_forward", t_forward_acc)
            if t_backward_acc > 0:
                profiler.record("gpt_backward", t_backward_acc)

        avg_train = (loss_acc / max(n_loss, 1)).item()
        avg_train_coarse = (coarse_loss_acc / max(n_loss, 1)).item()
        avg_train_fine = (fine_loss_acc / max(n_loss, 1)).item()
        avg_train_het = (het_loss_acc / max(n_loss, 1)).item()

        # Validation (always CE for comparable val_loss)
        model.eval()
        vlosses = []
        v_fine_losses = []
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
                            fine_targets=fine_target,
                            compute_reg_loss=not batched)
                    else:
                        coarse_logits, fine_logits = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, fine_targets=fine_target)
                        reg_pred = None
                        val_het = None
                    shift_logits = coarse_logits[:, :-1, :].contiguous()
                    shift_targets = target.contiguous()
                    v_fine_losses.append(
                        _per_seq_ce(
                            fine_logits,
                            fine_target,
                            ignore_index=-100,
                        ).detach()
                    )
                    if batched:
                        # Per-sequence means -> val_loss stays sequence-averaged
                        # (identical metric to the bs=1 loop).
                        vlosses.append(
                            _per_seq_ce(
                                shift_logits, shift_targets, ignore_index=-100
                            ).detach()
                        )
                        if args.heteroscedastic and reg_pred is not None:
                            v_het_losses.append(
                                _per_seq_het(
                                    reg_pred, reg_target[:, 1:]
                                ).detach()
                            )
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
                        preds = shift_logits.argmax(dim=-1)
                        val_pred_tokens.append(preds[valid_mask].detach())

        avg_val = _mean_metric_chunks(vlosses)
        avg_val_fine = _mean_metric_chunks(v_fine_losses)
        avg_val_het = _mean_metric_chunks(v_het_losses)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Light eval: compute collapse rate and token diversity
        collapse_rate = 0.0
        n_unique_tokens = 0
        if args.light_eval and val_pred_tokens:
            all_preds = torch.cat(val_pred_tokens).cpu()
            total = all_preds.numel()
            if total > 0:
                unique, counts = torch.unique(all_preds, return_counts=True)
                collapse_rate = counts.max().item() / total
                n_unique_tokens = len(unique)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train)
        history["train_coarse_loss"].append(avg_train_coarse)
        history["train_fine_loss"].append(avg_train_fine)
        history["train_het_loss"].append(avg_train_het)
        history["val_loss"].append(avg_val)
        # Keep val_loss for analyzer compatibility and expose the component
        # name explicitly in the downloadable audit trail.
        history["val_coarse_loss"].append(avg_val)
        history["val_fine_loss"].append(avg_val_fine)
        history["val_het_loss"].append(avg_val_het)
        history["lr"].append(cur_lr)
        history["lr_adam"].append(
            optimizer_adam.param_groups[0]["lr"]
            if optimizer_adam
            else None
        )
        history["optimizer_steps_this_epoch"].append(epoch_optimizer_steps)
        history["global_step"].append(global_step)
        history["accumulation_sequences"].append(epoch_accum)
        history["scheduled_train_sequences"].append(
            sequences_for_epoch(epoch)
        )
        train_microbatches = int(batch_stats.get("microbatches", 0))
        train_sequences = int(batch_stats.get("sequences", 0))
        real_input_tokens = int(batch_stats.get("real_tokens", 0))
        padded_input_tokens = int(batch_stats.get("padded_tokens", 0))
        mean_sequences_per_microbatch = (
            train_sequences / train_microbatches
            if train_microbatches
            else 1.0
        )
        padding_efficiency = (
            real_input_tokens / padded_input_tokens
            if padded_input_tokens
            else 1.0
        )
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        history["train_microbatches"].append(train_microbatches)
        history["mean_sequences_per_microbatch"].append(
            mean_sequences_per_microbatch
        )
        history["max_sequences_per_microbatch"].append(
            int(batch_stats.get("max_sequences_per_microbatch", 1))
        )
        history["real_input_tokens"].append(real_input_tokens)
        history["padded_input_tokens"].append(padded_input_tokens)
        history["padding_efficiency"].append(padding_efficiency)
        history["max_sequence_length"].append(
            int(batch_stats.get("max_sequence_length", 0))
        )
        history["peak_cuda_memory_allocated_gb"].append(peak_allocated_gb)
        history["peak_cuda_memory_reserved_gb"].append(peak_reserved_gb)
        history["epoch_time_s"].append(time.time() - epoch_t0)
        history["elapsed_time_s"].append(elapsed)
        if args.light_eval:
            history.setdefault("collapse_rate", []).append(collapse_rate)
            history.setdefault("n_unique_tokens", []).append(n_unique_tokens)

        # Write per-epoch history for Optuna pruning
        if args.history_per_epoch:
            temporary_history = history_path + ".tmp"
            with open(temporary_history, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            os.replace(temporary_history, history_path)

        save_tag = ""
        sd = _cpu_state_dict(model)
        if avg_val < best_val:
            best_val = avg_val
            _atomic_torch_save({
                "model_state_dict": sd,
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
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
                "constant_accumulation": constant_accumulation,
                "controlled_loader_seed": controlled_loader_seed,
                "exact_accumulation_boundaries": exact_accumulation,
                "collapse_rate": collapse_rate,
                "n_unique_tokens": n_unique_tokens,
            }, save_path)
            save_tag = "  -> Saved best"

        # Optional per-epoch inference checkpoint plus a compact index.  These
        # snapshots make it possible to revisit special points on the training
        # trajectory without retaining optimizer state for every epoch.
        if not getattr(args, "no_epoch_checkpoints", False):
            epoch_ckpt_path = f"{save_stem}_ep{epoch+1}.pt"
            _atomic_torch_save({
                "model_state_dict": sd,
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
                           "vocab_size": ModelConfig.vocab_size, "vocab_fine": ModelConfig.vocab_fine},
                "val_loss": avg_val,
                "epoch": epoch,
                "tag": args.tag,
                "constant_accumulation": constant_accumulation,
                "controlled_loader_seed": controlled_loader_seed,
                "exact_accumulation_boundaries": exact_accumulation,
            }, epoch_ckpt_path)
            try:
                if os.path.exists(epoch_ckpt_index_path):
                    with open(epoch_ckpt_index_path, "r", encoding="utf-8") as f:
                        checkpoint_index = json.load(f)
                else:
                    checkpoint_index = {"checkpoints": []}
                entries = [
                    item for item in checkpoint_index.get("checkpoints", [])
                    if int(item.get("epoch", 0)) < epoch + 1
                ]
                entries.append({
                    "epoch": epoch + 1,
                    "path": os.path.abspath(epoch_ckpt_path),
                    "size_bytes": os.path.getsize(epoch_ckpt_path),
                    "train_loss": avg_train,
                    "train_coarse_loss": avg_train_coarse,
                    "train_fine_loss": avg_train_fine,
                    "train_het_loss": avg_train_het,
                    "val_loss": avg_val,
                    "val_coarse_loss": avg_val,
                    "val_fine_loss": avg_val_fine,
                    "val_het_loss": avg_val_het,
                    "learning_rate": cur_lr,
                    "learning_rate_adam": (
                        optimizer_adam.param_groups[0]["lr"]
                        if optimizer_adam
                        else None
                    ),
                    "best_so_far": bool(save_tag),
                    "global_step": global_step,
                    "optimizer_steps_this_epoch": epoch_optimizer_steps,
                })
                checkpoint_index = {
                    "tag": args.tag,
                    "save_path": os.path.abspath(save_path),
                    "updated_epoch": epoch + 1,
                    "checkpoints": entries,
                }
                temporary_index = epoch_ckpt_index_path + ".tmp"
                with open(temporary_index, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_index, f, indent=2)
                os.replace(temporary_index, epoch_ckpt_index_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"  [warning] Could not update checkpoint index: {exc}")

        # Resume checkpoint (for training resume)
        ckpt_dict = {
            "model_state_dict": sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "global_step": global_step,
            "tag": args.tag,
            "constant_accumulation": constant_accumulation,
            "controlled_loader_seed": controlled_loader_seed,
            "exact_accumulation_boundaries": exact_accumulation,
            **_rng_state(),
        }
        if optimizer_adam:
            ckpt_dict["optimizer_adam_state"] = optimizer_adam.state_dict()
            ckpt_dict["scheduler_adam_state"] = scheduler_adam.state_dict()
        _atomic_torch_save(ckpt_dict, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        light_str = ""
        if args.light_eval and collapse_rate > 0:
            light_str = f" coll={collapse_rate*100:.1f}% uniq={n_unique_tokens}"
        print(f"  [{epochs_done}/{epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"lr={cur_lr:.2e} elapsed={elapsed:.0f}s ETA={eta:.0f}s step={global_step}"
              f" microbatch={mean_sequences_per_microbatch:.1f}/"
              f"{history['max_sequences_per_microbatch'][-1]}"
              f" pad={padding_efficiency:.1%}"
              f" peak={peak_allocated_gb:.1f}GiB"
              f"{light_str}{save_tag}", flush=True)

        # Early stopping check
        if early_stop and early_stop(avg_val, epoch):
            remaining = epochs - (epoch + 1)
            if remaining >= 3:
                # Instead of stopping, reset scheduler to cosine decay over remaining epochs.
                # This lets the LR decrease naturally while training continues.
                new_total = max(remaining * steps_per_epoch, 1)
                scheduler = build_wsd_scheduler(optimizer, new_total, warmup_ratio=0.0)
                if scheduler_adam:
                    scheduler_adam = build_wsd_scheduler(optimizer_adam, new_total, warmup_ratio=0.0)
                early_stop.counter = 0  # reset patience counter
                early_stop.patience = remaining  # remaining patience = remaining epochs
                print(f"  [scheduler_reset] Patience exhausted → cosine decay over {remaining} "
                      f"remaining epochs ({new_total} steps)")
            else:
                print(f"  [early_stop] Patience exhausted at epoch {epoch+1} "
                      f"(best={early_stop.best:.4f} at epoch {early_stop.best_epoch+1}). Stopping.")
                break

        if TrainingConfig.max_train_updates and global_step >= TrainingConfig.max_train_updates:
            break

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        _atomic_torch_save(ckpt, save_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if profiler:
        profiler.end("gpt_train")

    print(f"\nDone. Best val_loss: {best_val:.4f} (after {epochs_done} epochs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Kronos-Preview causal GPT model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configured defaults:
  python train_base.py

  # Short lower-dropout run:
  python train_base.py --epochs 15 --dropout 0.05

  # Disable the heteroscedastic head:
  python train_base.py --no-heteroscedastic

  # Standard CE baseline:
  python train_base.py --loss ce --weight_decay 0.001

  # Reasoning model (two-stage):
  python train_base.py --loss ce --reasoning --reasoning_frozen --base_checkpoint model.pt
  python train_base.py --loss focal --gamma 4.0 --reasoning
        """)
    # Core
    parser.add_argument("--save_path", type=str, default=TrainingConfig.base_model_path)
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="",
        help=(
            "Directory for downloadable history/checkpoint-index JSON; "
            "defaults to the checkpoint directory"
        ),
    )
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"],
                        help="Token loss; 'ce' is the reviewed Exp 04 line default")
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    # Defaults follow the reviewed Exp 04-A (muon) and Exp 04-B (lr_muon=0.005)
    # selections; override per experiment via CLI.
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"],
                        help="Optimizer: AdamW, or Muon for 2D weights plus AdamW for the rest")
    parser.add_argument("--lr_muon", type=float, default=0.005,
                        help="Muon learning rate (only when --optimizer muon)")
    parser.add_argument("--light_eval", action="store_true",
                        help="Compute collapse_rate/token_diversity during validation (fast HPO proxy)")
    # Focal loss
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Focal loss gamma")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--entropy_alpha", type=float, default=0.0,
                        help="Entropy regularization weight for focal loss (0.0 = disabled)")
    # Heteroscedastic
    parser.add_argument("--heteroscedastic", dest="heteroscedastic", action="store_true", default=True)
    parser.add_argument("--no-heteroscedastic", dest="heteroscedastic", action="store_false")
    parser.add_argument("--het_weight", type=float, default=0.1,
                        help="Weight for heteroscedastic NLL loss")
    parser.add_argument("--fine_weight", type=float, default=0.3,
                        help="Weight for fine-token auxiliary loss in dual-head prediction")
    # Reasoning
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--reasoning_frozen", action="store_true")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Pre-trained base model checkpoint for reasoning model")
    # Overrides
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=0,
                        help="Transformer width (0=production baseline 256)")
    parser.add_argument("--depth", type=int, default=0,
                        help="Transformer block count (0=production baseline 2)")
    parser.add_argument("--heads", type=int, default=0,
                        help="Attention head count (0=production baseline 4)")
    parser.add_argument("--num_kv_heads", type=int, default=0,
                        help="GQA key/value head count (0=production baseline 1)")
    parser.add_argument("--ffn_multiplier", type=int, default=0,
                        help="Feed-forward expansion multiplier (0=production baseline 4)")
    parser.add_argument("--gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_true",
                        default=None,
                        help="Enable per-block activation checkpointing")
    parser.add_argument("--no-gradient-checkpointing",
                        dest="gradient_checkpointing", action="store_false",
                        help="Explicitly disable activation checkpointing")
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="Subsample N stocks for fast HPO screening (0=all)")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Truncate sequences longer than this (0=no limit)")
    parser.add_argument("--force_repack", action="store_true",
                        help="Force re-tokenization (clear cache)")
    parser.add_argument("--history_per_epoch", action="store_true",
                        help="Write per-epoch val_loss to JSON (for Optuna pruning)")
    parser.add_argument("--no_epoch_checkpoints", action="store_true",
                        help="Do not save *_epN.pt inference snapshots; best .pt and resumable .ckpt are still kept")
    # ── Training improvement args ──
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience (0=disabled, recommended 3-5 for GPT)")
    # ── Speed: token-budget batching (right-pad + is_causal, math-identical to bs=1) ──
    parser.add_argument("--batch_tokens", type=int, default=12288,
                        help="Max tokens per batch (B*max_len). Adaptive batching for "
                             "~1.7x faster GPT training. 0 = legacy single-seq (bs=1).")
    parser.add_argument("--batch_cap", type=int, default=64,
                        help="Hard cap on sequences per batch (safety for very short stocks)")
    parser.add_argument(
        "--controlled_loader_seed",
        type=int,
        default=-1,
        help="Architecture-independent token-loader seed (-1 uses the "
             "process-global RNG).",
    )
    parser.add_argument(
        "--exact_accumulation_boundaries",
        action="store_true",
        default=False,
        help="Partition token-budget batches into deterministic blocks that "
             "end exactly at each optimizer update. Requires "
             "--controlled_loader_seed >= 0.",
    )
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Restore torch.use_deterministic_algorithms + cuBLAS "
                             "workspace pinning. Costs ~4%% and does NOT make this "
                             "trainer reproducible (SDPA backward stays "
                             "non-deterministic); kept for debugging only.")
    parser.add_argument(
        "--constant_accumulation",
        action="store_true",
        default=False,
        help="Keep TrainingConfig.accumulation_steps for every epoch instead of "
             "doubling it at absolute epoch 15.",
    )
    main(parser.parse_args())
