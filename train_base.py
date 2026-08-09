"""Train base model: KronosPreview on packed stock sequences.
Supports epoch-level checkpoint/resume via save_path + save_path.ckpt.
Supports: focal loss, weight_decay override, reasoning module."""
import argparse
import math
import os
import random
import time
import json
from pathlib import Path

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import (
    load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2,
    attach_sample_weights, _stock_cutoff_idx,
)
import regime
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


def _weighted_distribution_summary(counts):
    """Distribution summary over float (weighted) counts (Branch C)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    used = counts[counts > 0]
    if total <= 0 or used.size == 0:
        return {
            "total": 0.0,
            "n_unique": 0,
            "collapse_rate": 0.0,
            "entropy_bits": 0.0,
            "effective_tokens": 0.0,
        }
    probabilities = used / total
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "total": total,
        "n_unique": int(used.size),
        "collapse_rate": float(used.max() / total),
        "entropy_bits": entropy,
        "effective_tokens": float(2.0 ** entropy),
    }


def _write_weighted_token_diagnostics(metrics_dir, train_seqs, weight_cfg,
                                      vocab_coarse, vocab_fine):
    """Branch C: re-record train token distributions under the sample weights.

    Mirrors ``_write_dataset_token_diagnostics`` but aggregates WEIGHTED
    coarse/fine/joint counts (per-position ``sample_weights``, EOS position 0),
    plus per-regime and per-calendar-year valid-token quality slices, so the
    recency/regime re-weighting is fully auditable (ToDo §1.6 fingerprint
    re-recording).  Writes ``dataset_token_distributions_weighted.npz`` and
    ``weighted_summary.json``; returns the summary including a sha256
    fingerprint for cross-arm comparability.
    """
    import hashlib
    arrays = {
        "schema": np.asarray([2], dtype=np.int16),
        "vocab_coarse": np.asarray([vocab_coarse], dtype=np.int32),
        "vocab_fine": np.asarray([vocab_fine], dtype=np.int32),
        "vocab_joint": np.asarray([vocab_coarse * vocab_fine], dtype=np.int32),
    }
    coarse_counts = np.zeros(vocab_coarse, dtype=np.float64)
    fine_counts = np.zeros(vocab_fine, dtype=np.float64)
    joint_counts = np.zeros(vocab_coarse * vocab_fine, dtype=np.float64)
    regime_coarse = {}
    year_coarse = {}
    for sequence in train_seqs:
        coarse = sequence["targets"].numpy().astype(np.int64, copy=False)
        fine = sequence["fine_targets"].numpy().astype(np.int64, copy=False)
        sw = sequence.get("sample_weights")
        if sw is None:
            w = np.ones(len(coarse), dtype=np.float64)
        else:
            w = sw.numpy().astype(np.float64, copy=False)
        valid = (
            (coarse >= 0)
            & (coarse < vocab_coarse)
            & (fine >= 0)
            & (fine < vocab_fine)
        )
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            coarse_v = coarse[valid_idx]
            fine_v = fine[valid_idx]
            joint_v = coarse_v * vocab_fine + fine_v
            wv = w[valid_idx]
            coarse_counts += np.bincount(
                coarse_v, weights=wv, minlength=vocab_coarse
            )[:vocab_coarse]
            fine_counts += np.bincount(
                fine_v, weights=wv, minlength=vocab_fine
            )[:vocab_fine]
            joint_counts += np.bincount(
                joint_v, weights=wv,
                minlength=vocab_coarse * vocab_fine,
            )[: vocab_coarse * vocab_fine]
            reg_np = (
                sequence.get("regime_ids").numpy()
                if sequence.get("regime_ids") is not None
                else None
            )
            # time_ids[j, 2] = stored year-2010 of input position j; target
            # position p predicts input position p+1.
            year_np = sequence["time_ids"].numpy()[:, 2] + 2010
            for j in range(valid_idx.size):
                c = int(coarse_v[j])
                wj = float(wv[j])
                if reg_np is not None:
                    r = int(reg_np[valid_idx[j]])
                    if r not in regime_coarse:
                        regime_coarse[r] = np.zeros(vocab_coarse, dtype=np.float64)
                    regime_coarse[r][c] += wj
                y = int(year_np[valid_idx[j] + 1])
                if y not in year_coarse:
                    year_coarse[y] = np.zeros(vocab_coarse, dtype=np.float64)
                year_coarse[y][c] += wj
    arrays["train_coarse_counts_weighted"] = coarse_counts
    arrays["train_fine_counts_weighted"] = fine_counts
    arrays["train_joint_counts_weighted"] = joint_counts

    regime_labels = {
        -1: "unknown", 0: "low_vol", 1: "nan_insufficient",
        2: "mid_vol", 3: "high_vol",
    }
    summary = {
        "schema": 2,
        "split": "train",
        "weight_config": weight_cfg,
        "coarse": _weighted_distribution_summary(coarse_counts),
        "fine": _weighted_distribution_summary(fine_counts),
        "joint": _weighted_distribution_summary(joint_counts),
        "regime": {
            regime_labels.get(reg, str(reg)): {
                "regime_id": reg,
                "weighted_valid_coarse": _weighted_distribution_summary(
                    regime_coarse[reg]
                ),
            }
            for reg in sorted(regime_coarse)
        },
        "year": {
            str(year): {
                "weighted_valid_coarse": _weighted_distribution_summary(
                    year_coarse[year]
                ),
            }
            for year in sorted(year_coarse)
        },
    }
    fingerprint_payload = {
        "schema": 2,
        "weight_config": weight_cfg,
        "coarse_counts": coarse_counts.tolist(),
        "fine_counts": fine_counts.tolist(),
        "joint_counts": joint_counts.tolist(),
        "regime_total_weight": {
            str(reg): float(regime_coarse[reg].sum())
            for reg in sorted(regime_coarse)
        },
    }
    summary["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    distribution_path = os.path.join(
        metrics_dir, "dataset_token_distributions_weighted.npz"
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
    summary_path = os.path.join(metrics_dir, "weighted_summary.json")
    temporary_summary = summary_path + ".tmp"
    with open(temporary_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary_summary, summary_path)
    return summary


def _load_or_build_coarse_logret_centers(tokenizer, metrics_dir):
    """Branch B: coarse-codebook expected normalized-log_ret centers.

    For each coarse code c, decode [c, 0] (fine=0) with the frozen tokenizer and
    take reconstruction feature 0 (normalized log_ret).  score_i =
    softmax(coarse_logits)[..., :V_c] @ centers then recovers the model's
    expected signed normalized log_ret for the predicted day, in the same units
    as the realized ``reg_signed`` value used by the ListNet loss.

    Cached to ``<metrics_dir>/coarse_logret_centers.npy``; reused when present
    and the coarse vocabulary matches.
    """
    V_c = int(tokenizer.vocab_coarse)
    cache_path = os.path.join(metrics_dir, "coarse_logret_centers.npy")
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)
            if arr.shape == (V_c,):
                print(f"  [rank] Reused coarse_logret_centers.npy (V_c={V_c})")
                return torch.from_numpy(arr.astype(np.float32))
        except Exception as exc:
            print(f"  [rank] coarse_logret_centers.npy invalid, recomputing: {exc}")
    tok_device = next(tokenizer.parameters()).device
    tokenizer.eval()
    with torch.no_grad():
        coarse = torch.arange(V_c, dtype=torch.long, device=tok_device)
        fine = torch.zeros(V_c, dtype=torch.long, device=tok_device)
        all_idx = torch.stack([coarse, fine], dim=-1).unsqueeze(1)  # [V_c, 1, 2]
        decoded = tokenizer.decode_all(all_idx)                     # [V_c, 1, input_dim]
        centers = decoded[:, 0, 0].float().cpu().numpy()            # feature 0 = log_ret
    try:
        os.makedirs(metrics_dir, exist_ok=True)
        np.save(cache_path, centers)
        print(f"  [rank] Wrote coarse_logret_centers.npy (V_c={V_c})")
    except Exception as exc:
        print(f"  [rank] Could not write coarse_logret_centers.npy: {exc}")
    return torch.from_numpy(centers.astype(np.float32))


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
                   entropy_alpha=0.0, ignore_index=-100, sample_weights=None):
    """Row-wise focal loss. logits [B,T,V], targets [B,T] -> [B] per-seq means.

    Algebraically identical to focal_loss() but reduced per row instead of
    globally, so summing the result over B equals the sum of per-sequence
    focal_loss() scalars.  ``sample_weights`` ([B,T]) switches the per-seq
    reduction to a weight-and-mask weighted mean (Branch C); ``None`` keeps the
    historical mask-mean path bit-for-bit.
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
    if sample_weights is not None:
        # Branch C: weight-and-mask weighted mean over positions.
        w = sample_weights * mask
        denom_w = w.sum(1).clamp(min=1e-8)
        per_seq = (focal_weight * ce * w).sum(1) / denom_w
        if entropy_alpha > 0:
            if log_probs is None:
                log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            per_seq = per_seq - entropy_alpha * ((entropy * w).sum(1) / denom_w)
        return per_seq
    denom = mask.sum(1).clamp(min=1)
    per_seq = (focal_weight * ce * mask).sum(1) / denom
    if entropy_alpha > 0:
        if log_probs is None:
            log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        per_seq = per_seq - entropy_alpha * ((entropy * mask).sum(1) / denom)
    return per_seq


def _per_seq_ce(logits, targets, ignore_index=-100, label_smoothing=0.0,
                sample_weights=None):
    """Row-wise cross-entropy. logits [B,T,V], targets [B,T] -> [B] per-seq means.

    When ``sample_weights`` ([B,T] float) is given the reduction is a
    weight-and-mask weighted mean; ``None`` (default) keeps the historical
    mask-mean reduction bit-for-bit.
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                         reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing).view(B, T)
    mask = (targets != ignore_index).float()
    if sample_weights is not None:
        w = sample_weights * mask
        return (ce * w).sum(1) / w.sum(1).clamp(min=1e-8)
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


_TARGET_COARSE_DIST: torch.Tensor | None = None


def _load_target_coarse_dist(device) -> torch.Tensor:
    """Load the training-set coarse-token marginal distribution (Branch A).

    Branch A's L_dm aligns the model's predicted token histogram with the
    empirical target marginal distribution over the same codebook.  The target
    reference is the training-split coarse distribution from the tokenization
    diagnostics (dataset_token_summary.json).  Falls back to a uniform prior if
    unavailable (uniform KL still discourages collapse).
    """
    global _TARGET_COARSE_DIST
    if _TARGET_COARSE_DIST is not None:
        return _TARGET_COARSE_DIST
    counts = None
    summary_path = Path("checkpoints/dataset_token_summary.json")
    sidecar_path = Path("checkpoints/dataset_token_distributions.npz")
    if summary_path.exists():
        try:
            import json as _json
            summary = _json.loads(summary_path.read_text(encoding="utf-8"))
            train = summary.get("splits", {}).get("train", {}).get("coarse", {})
            n_unique = int(train.get("n_unique", 0))
            if n_unique > 0:
                counts = torch.ones(n_unique, dtype=torch.float32)
        except Exception:
            counts = None
    if counts is None and sidecar_path.exists():
        try:
            import numpy as _np
            data = _np.load(sidecar_path)
            train_counts = data["train_coarse_counts"]
            counts = torch.from_numpy(train_counts.sum(axis=0).astype(np.float32))
        except Exception:
            counts = None
    if counts is None:
        counts = torch.ones(128, dtype=torch.float32)
    # Pad to the full coarse vocabulary (ModelConfig.vocab_size) so the
    # distribution matches logits width; unused codes get zero probability.
    vocab = int(ModelConfig.vocab_size)
    full = torch.zeros(vocab, dtype=torch.float32)
    n = min(len(counts), vocab)
    full[:n] = counts[:n]
    dist = full / full.sum().clamp(min=1e-8)
    _TARGET_COARSE_DIST = dist.to(device)
    return _TARGET_COARSE_DIST


def distribution_match_loss(coarse_logits, targets, target_dist, ignore_index=-100,
                            sample_weights=None):
    """Branch A: symmetric KL between predicted soft-histogram and target marginal.

    ``coarse_logits`` [B, T, V], ``targets`` [B, T] (positions, -100 ignored).
    Builds the model's empirical prediction distribution over valid positions
    (softmax per position, mean over positions) and computes
    symmetric KL(target_dist || pred_dist) + KL(pred_dist || target_dist).
    The detached reference target distribution makes this a true distributional
    regularizer rather than a soft label loss.

    Branch C: when ``sample_weights`` ([B,T] float) is given the position
    histogram is a weight-and-mask weighted mean instead of a uniform one, so
    the dm term uses the same per-day weight grid as the token losses.
    """
    B, T, V = coarse_logits.shape
    # The coarse head may output vocab + 2 (BOS/EOS); clip to the codebook.
    V_code = min(V, int(target_dist.shape[0]))
    coarse_logits = coarse_logits[..., :V_code]
    V = V_code
    valid = targets != ignore_index
    if not valid.any():
        return torch.zeros((), device=coarse_logits.device)
    probs = torch.softmax(coarse_logits.float(), dim=-1)  # [B, T, V]
    if sample_weights is not None:
        w = sample_weights * valid.float()
        if w.sum() == 0:
            return torch.zeros((), device=coarse_logits.device)
        pred_dist = (probs * w.unsqueeze(-1)).sum(0) / w.sum().clamp(min=1e-8)
    else:
        pred_dist = probs[valid].mean(dim=0)              # [V]
    pred_dist = pred_dist.clamp(min=1e-8)
    pred_dist = pred_dist / pred_dist.sum()
    tgt = target_dist.float().to(coarse_logits.device).clamp(min=1e-8)
    tgt = tgt / tgt.sum()
    kl = (tgt * (tgt.log() - pred_dist.log())).sum() + (
        pred_dist * (pred_dist.log() - tgt.log())
    ).sum()
    return 0.5 * kl


def _listnet_loss(scores, realized, date_ids, temperature=1.0):
    """Branch B: ListNet top-one loss, grouped per trading date.

    Groups ``scores``/``realized`` by ``date_ids``; within each date's cross
    section the two vectors are centered (softmax is translation-invariant) and
    the loss is the ListNet top-one softmax CE:
    CE(softmax(realized/τ) ‖ softmax(scores/τ)) = -Σ softmax(y)·log_softmax(s).
    Averages over dates.  score and realized share normalized-log_ret units, so
    the softmaxes align directly with RankIC ordering.
    """
    total = torch.zeros((), device=scores.device)
    n = 0
    for d in torch.unique(date_ids):
        m = (date_ids == d) & (realized != -999.0)
        if m.sum() < 2:
            continue
        s = scores[m].float() / temperature
        y = realized[m].float() / temperature
        s = s - s.mean()
        y = y - y.mean()
        lps = torch.log_softmax(s, dim=0)     # predicted ranking distribution
        sy = torch.softmax(y, dim=0)          # realized ranking distribution
        total = total - (sy * lps).sum()      # CE(softmax(y) || softmax(s))
        n += 1
    return total / max(n, 1)


def _per_seq_het(reg_pred, reg_targets_shifted, ignore_val=-999.0, sample_weights=None):
    """Row-wise heteroscedastic NLL. reg_pred [B,T,2] (float), targets [B,T] -> [B].

    Mirrors heteroscedastic_nll_loss() reduced per row. Masked positions have
    their residual zeroed before squaring so sentinel targets (-999) never
    create large intermediate values.  ``sample_weights`` ([B,T]) switches the
    reduction to a weight-and-mask weighted mean (Branch C).
    """
    mask = (reg_targets_shifted != ignore_val).float()
    mean = reg_pred[..., 0]
    log_var = reg_pred[..., 1].clamp(-5.0, 2.0)
    diff = (reg_targets_shifted - mean) * mask
    nll = 0.5 * (log_var + diff.pow(2) / log_var.exp())
    if sample_weights is not None:
        w = sample_weights * mask
        return (nll * w).sum(1) / w.sum(1).clamp(min=1e-8)
    return (nll * mask).sum(1) / mask.sum(1).clamp(min=1)


def _per_pos_het(reg_pred, reg_targets_shifted, ignore_val=-999.0):
    """Per-position heteroscedastic NLL. reg_pred [B,T,2], targets [B,T] -> [B,T].

    Branch F regime-gated loss needs the per-token NLL so each regime's slice can
    be mean-reduced independently (``_per_seq_het`` collapses to per-row means).
    Masked positions are zeroed as in ``_per_seq_het`` (sentinel -999 -> 0).
    """
    mask = (reg_targets_shifted != ignore_val).float()
    mean = reg_pred[..., 0]
    log_var = reg_pred[..., 1].clamp(-5.0, 2.0)
    diff = (reg_targets_shifted - mean) * mask
    nll = 0.5 * (log_var + diff.pow(2) / log_var.exp())
    return nll * mask  # [B, T]


def _mean_metric_chunks(chunks):
    """Average validation chunks with at most one device-to-host sync."""
    if not chunks:
        return 0.0
    if isinstance(chunks[0], torch.Tensor):
        values = torch.cat([chunk.reshape(-1) for chunk in chunks]).cpu().tolist()
    else:
        values = chunks
    return sum(values) / max(len(values), 1)


def _future_targets(target, offsets, vocab):
    """Branch D (MTP): future coarse targets for offsets (2,3,4).

    head d at position t predicts input_ids[t+d] = targets[t+d-1].  Returns a
    list of [B, T] tensors with the aligned future targets (masked -100 where
    out of range or where the target is BOS/EOS).
    """
    B, T = target.shape
    outs = []
    for d in offsets:
        ft = torch.full_like(target, -100)
        shift = d - 1
        if shift < T:
            ft[:, : T - shift] = target[:, shift:]
        ft = ft.masked_fill((ft < 0) | (ft >= vocab), -100)
        outs.append(ft)
    return outs


def _compute_regime_gated_loss(coarse_logits, target, fine_logits, fine_target,
                               reg_pred, reg_target, args, regime_ids,
                               n_regimes=3):
    """Branch F: per-regime-gated token loss over a right-padded batch.

    ``regime_ids`` is [B, N] aligned to input_ids.  The loss grid is the shifted
    targets (target position p predicts input position p), so
    ``regime_for_loss[p] = regime_ids[p]`` for p in 0..T-1 (T = N-1) — i.e.
    ``rmask = regime_ids[:, :T]``.  Tokens with regime -1 (BOS, EOS, insufficient
    trailing window, padding) are excluded from every regime loss.

    The total is the SUM of per-regime means::

        total = Σ_r mean_{p : regime_for_loss[p]==r} (coarse_CE + 0.3*fine_CE
                + 0.1*het_NLL)

    so each regime contributes equally regardless of its token count.  Returns
    ``(total_scalar, components, B)`` where ``components`` carries the standard
    ``coarse``/``fine``/``heteroscedastic`` keys (per-regime sums, for history)
    plus a ``per_regime`` dict with per-regime totals for red-line monitoring.
    """
    B = coarse_logits.shape[0]
    T = target.shape[1]
    V = coarse_logits.shape[-1]
    shift = coarse_logits[:, :-1, :]                      # [B, T, V]
    ce = F.cross_entropy(shift.reshape(-1, V), target.reshape(-1),
                         reduction="none", ignore_index=-100,
                         label_smoothing=args.label_smoothing).view(B, T)
    Vf = fine_logits.shape[-1]
    fce = F.cross_entropy(fine_logits.reshape(-1, Vf), fine_target.reshape(-1),
                          reduction="none", ignore_index=-100).view(B, T)
    if reg_pred is not None and args.heteroscedastic:
        het = _per_pos_het(reg_pred, reg_target[:, 1:])   # [B, T]
    else:
        het = torch.zeros_like(ce)

    rmask = regime_ids[:, :T]                              # regime_for_loss grid
    total = torch.zeros((), device=ce.device)
    coarse_sum = torch.zeros((), device=ce.device)
    fine_sum = torch.zeros((), device=ce.device)
    het_sum = torch.zeros((), device=ce.device)
    comp = {}
    for r in range(n_regimes):
        m = (rmask == r) & (target != -100)
        entry = {"total": torch.zeros((), device=ce.device).detach(),
                 "coarse": torch.zeros((), device=ce.device).detach(),
                 "n_tokens": int(m.sum())}
        if not m.any():
            comp[r] = entry
            continue
        lr = ce[m].mean()
        coarse_sum = coarse_sum + lr
        entry["coarse"] = lr.detach()
        fm = m & (fine_target != -100)
        if fm.any():
            fl = fce[fm].mean()
            lr = lr + args.fine_weight * fl
            fine_sum = fine_sum + fl
            entry["fine"] = fl.detach()
        hm = m & (reg_target[:, 1:] != -999)
        if args.heteroscedastic and hm.any():
            hl = het[hm].mean()
            lr = lr + args.het_weight * hl
            het_sum = het_sum + hl
            entry["het"] = hl.detach()
        entry["total"] = lr.detach()
        comp[r] = entry
        total = total + lr
    components = {
        "coarse": coarse_sum.detach(),
        "fine": fine_sum.detach(),
        "heteroscedastic": het_sum.detach(),
        "per_regime": comp,
    }
    return total, components, B


def compute_batched_loss(coarse_logits, target, fine_logits, fine_target,
                         reg_pred, reg_target, args, rank_date_ids=None,
                         rank_centers=None, sample_weights=None,
                         future_logits=None, regime_ids=None):
    """Sum-of-per-sequence total loss for a right-padded batch.

    Returns ``(loss_sum, component_sums, n_seq)``. Component sums are detached
    unweighted per-sequence losses, making the downloadable training history
    sufficient to separate coarse, fine, and regression behaviour.

    Branch B (ListNet): when ``args.rank_weight > 0`` and both ``rank_date_ids``
    and ``rank_centers`` are given, a cross-sectional ranking loss is summed into
    the total.  ``scores`` = expected signed normalized log_ret of the coarse
    softmax at the window's last prediction position
    (``softmax(shift_coarse[:, -1, :V_c]) @ rank_centers``); ``realized`` = the
    target day's signed log_ret (``reg_target[:, -1]`` — the rank loader fills
    ``reg_target`` with the signed ``reg_signed`` window).  In rank mode the
    heteroscedastic term is skipped because ``reg_target`` holds signed values,
    not |z|.

    Branch C: ``sample_weights`` ([B,T] float, aligned to the ``targets`` grid)
    is threaded into every per-sequence token loss and, when ``dm_weight > 0``,
    into ``distribution_match_loss``.  ``None`` (default) keeps the historical
    mask-mean reduction bit-for-bit, so rank-mode and non-Branch-C runs are
    unchanged.

    Branch F (LoRA-per-regime): when ``regime_ids`` ([B,N], aligned to
    input_ids) is given, the whole loss is replaced by the per-regime-gated sum
    of ``_compute_regime_gated_loss``; ``None`` (default) keeps the dense path
    byte-for-byte.
    """
    if regime_ids is not None:
        return _compute_regime_gated_loss(
            coarse_logits, target, fine_logits, fine_target,
            reg_pred, reg_target, args, regime_ids)
    shift_coarse = coarse_logits[:, :-1, :]
    if args.loss == "focal":
        coarse = _per_seq_focal(shift_coarse, target, gamma=args.gamma,
                                label_smoothing=args.label_smoothing,
                                entropy_alpha=args.entropy_alpha,
                                sample_weights=sample_weights)
    else:
        coarse = _per_seq_ce(shift_coarse, target, ignore_index=-100,
                             label_smoothing=args.label_smoothing,
                             sample_weights=sample_weights)
    total = coarse
    fine = _per_seq_ce(
        fine_logits, fine_target, ignore_index=-100,
        sample_weights=sample_weights
    )
    total = total + args.fine_weight * fine
    rank_mode = (
        getattr(args, "rank_weight", 0.0) > 0
        and rank_date_ids is not None
        and rank_centers is not None
    )
    het = torch.zeros_like(coarse)
    if reg_pred is not None and args.heteroscedastic and not rank_mode:
        het = _per_seq_het(reg_pred, reg_target[:, 1:],
                           sample_weights=sample_weights)
        total = total + args.het_weight * het
    dm = torch.zeros_like(coarse)
    if getattr(args, "dm_weight", 0.0) > 0:
        target_dist = _load_target_coarse_dist(coarse_logits.device)
        dm_loss = distribution_match_loss(
            shift_coarse, target, target_dist, ignore_index=-100,
            sample_weights=sample_weights,
        )
        dm = dm_loss.expand_as(coarse).detach() if dm_loss.dim() == 0 else dm_loss
        total = total + args.dm_weight * dm_loss
    rank = torch.zeros_like(coarse)
    rank_scalar = torch.zeros((), device=coarse.device)
    if rank_mode:
        V = min(int(rank_centers.shape[0]), shift_coarse.shape[-1])
        centers = rank_centers.to(device=coarse_logits.device, dtype=torch.float32)
        score_col = torch.softmax(shift_coarse[:, -1, :V].float(), dim=-1)
        scores = score_col @ centers[:V]            # [B] expected log_ret
        realized = reg_target[:, -1]                # [B] signed log_ret (target day)
        rl = _listnet_loss(
            scores, realized, rank_date_ids,
            temperature=getattr(args, "rank_temperature", 1.0),
        )
        rank = rl.detach().expand_as(coarse)
        rank_scalar = args.rank_weight * rl
        total = total + rank_scalar
    mtp_scalar = torch.zeros((), device=coarse.device)
    if future_logits is not None and getattr(args, "mtp", False):
        vocab = coarse_logits.shape[-1] - 2
        fts = _future_targets(target, args.mtp_offsets, vocab)
        for fl, d, w in zip(future_logits, args.mtp_offsets, args.mtp_weights):
            ft = fts[args.mtp_offsets.index(d)]
            term = _per_seq_ce(fl[:, :-1, :], ft, ignore_index=-100)
            total = total + w * term
            mtp_scalar = mtp_scalar + (w * term.sum()).detach()
    components = {
        "coarse": coarse.sum().detach(),
        "fine": fine.sum().detach(),
        "heteroscedastic": het.sum().detach(),
    }
    if rank_mode:
        components["rank_listnet"] = rank.sum().detach()
    if getattr(args, "mtp", False) and future_logits is not None:
        components["mtp"] = mtp_scalar.detach()
    if getattr(args, "dm_weight", 0.0) > 0:
        components["distribution_match"] = (
            dm_loss.detach() if isinstance(dm_loss, torch.Tensor) and dm_loss.dim() == 0
            else dm.sum().detach()
        )
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
    """Move a 10-tuple batch to device and ensure batch dimension. mask may be None (is_causal).

    Branch C: the 9th element ``sw`` (sample_weights [B, max_len-1]) is moved
    with the batch; an absent weight channel defaults to all-ones so non-Branch-C
    runs are bit-identical to the historical 8-tuple path.

    Branch F: the 10th element ``regime_ids`` ([B, max_len], aligned to
    input_ids, -1 = no regime) is moved with the batch; ``None`` when the batch
    carries no input_ids-aligned regime channel (dense / Branch-C runs).
    """
    inp, tgt, ftgt, tids, pos, mask, va, rt = batch[:8]
    sw = batch[8] if len(batch) > 8 else None
    regime_ids = batch[9] if len(batch) > 9 else None
    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    ftgt = ftgt.to(device, non_blocking=True)
    tids = tids.to(device, non_blocking=True)
    pos = pos.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True) if mask is not None else None
    va = va.to(device, non_blocking=True)
    rt = rt.to(device, non_blocking=True)
    if sw is None:
        sw = torch.ones_like(tgt, dtype=torch.float32, device=tgt.device)
    else:
        sw = sw.to(device, non_blocking=True)
    if regime_ids is not None:
        regime_ids = regime_ids.to(device, non_blocking=True)
    if inp.dim() == 1:
        inp, tgt, ftgt, tids, pos, va, rt, sw = [
            x.unsqueeze(0) for x in (inp, tgt, ftgt, tids, pos, va, rt, sw)
        ]
        if mask is not None:
            mask = mask.unsqueeze(0)
        if regime_ids is not None:
            regime_ids = regime_ids.unsqueeze(0)
    return inp, tgt, ftgt, tids, pos, mask, va, rt, sw, regime_ids


def _to_device_rank(batch, device):
    """Move a Branch-B rank (8-tuple, date_id) batch to device.

    ``batch`` is ``((p_ids, p_tgt, p_ftgt, p_time, p_pos, None, p_va, p_rt),
    p_date)`` from CrossSectionalRankLoader.  The 8-tuple shares the right-padded
    layout of _pad_batch_causal (mask=None, is_causal).
    """
    (inp, tgt, ftgt, tids, pos, mask, va, rt), date_id = batch
    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    ftgt = ftgt.to(device, non_blocking=True)
    tids = tids.to(device, non_blocking=True)
    pos = pos.to(device, non_blocking=True)
    va = va.to(device, non_blocking=True)
    rt = rt.to(device, non_blocking=True)
    date_id = date_id.to(device, non_blocking=True)
    return inp, tgt, ftgt, tids, pos, mask, va, rt, date_id


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
        p_sw = torch.ones(B, max_len - 1, dtype=torch.float32)
        # Branch F: regime channel only when the group carries input_ids-aligned
        # regime_ids (len == len(input_ids)).  Branch C's targets-aligned
        # regime_ids (len == len(targets)) is a different grid and is not
        # surfaced into the batch (it never routes the model).
        has_regime = any(
            s.get("regime_ids") is not None
            and s["regime_ids"].shape[0] == s["input_ids"].shape[0]
            for s in group
        )
        p_regime = torch.full((B, max_len), -1, dtype=torch.long) if has_regime else None

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
            p_sw[j, :Lt] = s.get("sample_weights", 1.0)
            if p_regime is not None:
                p_regime[j, :L] = s.get("regime_ids", -1)
            mask = torch.zeros(L, L, dtype=torch.bool)
            mask[:, 0] = True
            for start, end in s.get("boundaries", [(1, L)]):
                for pos in range(start, min(end, L)):
                    mask[pos, start:pos + 1] = True
            p_mask[j, :L, :L] = mask
            p_mask[j, L:, 0] = True

        batches.append((p_ids, p_tgt, p_ftgt, p_time, p_pos, p_mask, p_va, p_rt, p_sw, p_regime))
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

    Returns the same 10-tuple layout as make_dataloader_v2 but with a real batch
    dimension and NO attention mask -- so the model uses SDPA is_causal=True.
    With right-padding + causal attention, every REAL query position i attends
    only to real keys 0..i, so its logits are identical to processing the stock
    alone (bs=1). Padded query rows are discarded by the loss (targets = -100 /
    fine -100 / reg -999), giving numerically identical training to bs=1.

    Branch F: the 10th element ``p_regime`` ([B, Nmax], aligned to input_ids,
    -1 = no regime) is present only when the group carries input_ids-aligned
    regime_ids (Branch C's targets-aligned regime_ids is not surfaced).
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
    p_sw = torch.ones(B, Nmax - 1, dtype=torch.float32)
    has_regime = any(
        s.get("regime_ids") is not None
        and s["regime_ids"].shape[0] == s["input_ids"].shape[0]
        for s in group
    )
    p_regime = torch.full((B, Nmax), -1, dtype=torch.long) if has_regime else None
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
        p_sw[k, :Lt] = s.get("sample_weights", 1.0)
        if p_regime is not None:
            p_regime[k, :L] = s.get("regime_ids", -1)
    return (p_ids, p_tgt, p_ftgt, p_time, p_pos, None, p_va, p_rt, p_sw, p_regime)


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


def _pad_cross_section_batch(group, date_key, context):
    """Pack one trading date's cross-section into a right-padded rank batch.

    Each row is [BOS] + ``context`` days (the ``context`` trading days ending on
    the target day).  With window input columns 0..W (W = context):
      - p_ids[k, 1:] = ids[lo_inp : day_idx+2]  (last col = target-day token)
      - p_rt[k, 1:]  = reg_signed[lo_inp : day_idx+2] (last col = target-day
        signed normalized log_ret == realized)
      - p_tgt[k, :]  = next-token targets over window positions 1..W
      - p_pos renumbered 0..W, mask=None (is_causal)
    ``date_key`` is a packed (year*10000 + month*100 + day) integer broadcast to
    a [B] tensor for per-date grouping in _listnet_loss.
    """
    W = int(context)
    B = len(group)
    L = W + 1
    p_ids = torch.zeros(B, L, dtype=torch.long)
    p_tgt = torch.full((B, W), -100, dtype=torch.long)
    p_ftgt = torch.full((B, W), -100, dtype=torch.long)
    p_time = torch.zeros(B, L, 3, dtype=torch.long)
    p_pos = torch.zeros(B, L, dtype=torch.long)
    p_va = torch.zeros(B, L, 2, dtype=torch.float32)
    p_rt = torch.full((B, L), -999.0, dtype=torch.float32)
    p_date = torch.full((B,), date_key, dtype=torch.long)
    for k, (seq, day_idx) in enumerate(group):
        lo_inp = day_idx - W + 2
        hi_inp = day_idx + 1
        p_ids[k, 0] = seq["input_ids"][0]
        p_ids[k, 1:] = seq["input_ids"][lo_inp:hi_inp + 1]
        p_tgt[k, :] = seq["targets"][lo_inp - 1:hi_inp]
        p_ftgt[k, :] = seq["fine_targets"][lo_inp - 1:hi_inp]
        p_time[k, 0] = seq["time_ids"][lo_inp]
        p_time[k, 1:] = seq["time_ids"][lo_inp:hi_inp + 1]
        p_pos[k, :] = torch.arange(L)
        p_va[k, 0] = seq["va_values"][lo_inp]
        p_va[k, 1:] = seq["va_values"][lo_inp:hi_inp + 1]
        p_rt[k, 0] = -999.0
        p_rt[k, 1:] = seq["reg_signed"][lo_inp:hi_inp + 1]
    return (p_ids, p_tgt, p_ftgt, p_time, p_pos, None, p_va, p_rt), p_date


class CrossSectionalRankLoader:
    """Branch B: date-cross-section loader for ListNet ranking fine-tuning.

    Built from the same tokenized ``train_seqs`` as the standard loaders.  An
    inverted index maps date -> [(seq_idx, day_idx)] where ``day_idx`` is a
    sequence position whose target day (day ``day_idx``) has a valid signed
    realized log_ret (``reg_signed[day_idx+1] != -999``).  Each microbatch is ONE
    date's cross-section of up to ``cross_section`` stocks; window rows are
    [BOS] + ``context`` days ending on the target day (see
    ``_pad_cross_section_batch``).  ``loader_seed`` fixes the epoch's date shuffle
    and intra-date sampling via a local torch.Generator.
    """

    def __init__(self, sequences, context=64, cross_section=64, min_stocks=30,
                 loader_seed=42, max_dates_per_epoch=0, cap_per_date=0):
        self.sequences = sequences
        self.context = int(context)
        self.cross_section = int(cross_section)
        self.min_stocks = int(min_stocks)
        self.loader_seed = int(loader_seed)
        self.max_dates_per_epoch = int(max_dates_per_epoch)
        self.cap_per_date = int(cap_per_date)
        self._epoch = 0
        self.last_iteration_stats = {}
        self._build_date_index()

    def _build_date_index(self):
        """Vectorized inverted date index: date_key -> [(seq_idx, day_idx), ...].

        ``day_idx`` is a valid sequence position p whose target day (day p) has
        ``reg_signed[p+1] != -999``.  date_key packs the calendar date as
        ``(year) * 10000 + month * 100 + day`` with ``year`` = stored
        (year - 2010) + 2010.
        """
        date_index = {}
        for si, seq in enumerate(self.sequences):
            S = int(seq["input_ids"].shape[0])
            lo, hi = self.context, S - 1
            if hi <= lo:
                continue
            rts = seq["reg_signed"].numpy()
            tids = seq["time_ids"].numpy()          # [S, 3] (day, month, year-2010)
            idxs = np.arange(lo, hi)                # candidate day_idx positions
            idxs = idxs[rts[idxs + 1] != -999.0]    # valid target-day realized
            if idxs.size == 0:
                continue
            keys = ((tids[idxs + 1, 2] + 2010) * 10000
                    + tids[idxs + 1, 1] * 100
                    + tids[idxs + 1, 0])
            for key, day_idx in zip(keys.astype(np.int64).tolist(), idxs.tolist()):
                bucket = date_index.get(key)
                if bucket is None:
                    date_index[key] = [(si, day_idx)]
                elif not self.cap_per_date or len(bucket) < self.cap_per_date:
                    bucket.append((si, day_idx))
        eligible = {
            key: rows for key, rows in date_index.items()
            if len(rows) >= self.min_stocks
        }
        self.date_index = eligible
        self._dates = sorted(eligible)
        self.n_dates = len(self._dates)
        self.n_pairs = sum(len(rows) for rows in eligible.values())
        self._stocks_per_date = sorted(len(rows) for rows in eligible.values())

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __iter__(self):
        self.last_iteration_stats = {
            "microbatches": 0,
            "sequences": 0,
            "real_tokens": 0,
            "padded_tokens": 0,
            "max_sequences_per_microbatch": 0,
            "max_sequence_length": 0,
        }
        generator = torch.Generator().manual_seed(
            (int(self.loader_seed) + 1_000_003 * int(self._epoch)) % (2 ** 63 - 1)
        )
        order = torch.randperm(len(self._dates), generator=generator).tolist()
        if self.max_dates_per_epoch > 0:
            order = order[: self.max_dates_per_epoch]
        stats = self.last_iteration_stats
        for position in order:
            key = self._dates[position]
            rows = self.date_index[key]
            if self.cross_section > 0 and len(rows) > self.cross_section:
                chosen_idx = torch.randperm(len(rows), generator=generator).tolist()
                chosen = [rows[i] for i in chosen_idx[: self.cross_section]]
            else:
                chosen = rows
            B = len(chosen)
            stats["microbatches"] += 1
            stats["sequences"] += B
            stats["real_tokens"] += B * (self.context + 1)
            stats["padded_tokens"] += B * (self.context + 1)
            stats["max_sequences_per_microbatch"] = max(
                stats["max_sequences_per_microbatch"], B
            )
            stats["max_sequence_length"] = self.context + 1
            yield _pad_cross_section_batch(
                [(self.sequences[si], d) for si, d in chosen],
                key, self.context)

    def __len__(self):
        if self.max_dates_per_epoch > 0:
            return min(self.max_dates_per_epoch, self.n_dates)
        return self.n_dates


def _write_collation_fingerprint(metrics_dir, loader, args):
    """Branch B: fingerprint the cross-sectional collation (ToDo §1.6).

    The cross-sectional collation reweights (stock, day) pairs relative to the
    per-stock token collation (uniform over dates x uniform over stocks within a
    date ~ weight 1/n_stocks(date)), so the collation configuration is recorded
    alongside the token distribution diagnostics.
    """
    spd = sorted(loader._stocks_per_date)
    n = len(spd)
    summary = {
        "schema": 1,
        "collation": "cross_section",
        "data_source": "same train_seqs (token cache)",
        "sampling": ("uniform over dates x uniform over stocks within date; "
                     "effective (stock, day) weight ~ 1/n_stocks(date), which "
                     "differs from uniform-over-(stock, day)"),
        "rank_weight": getattr(args, "rank_weight", 0.0),
        "rank_context": getattr(args, "rank_context", 64),
        "rank_cross_section": getattr(args, "rank_cross_section", 64),
        "rank_min_stocks": getattr(args, "rank_min_stocks", 30),
        "rank_temperature": getattr(args, "rank_temperature", 1.0),
        "loader_seed": loader.loader_seed,
        "max_dates_per_epoch": getattr(args, "rank_max_dates_per_epoch", 0),
        "n_eligible_dates": loader.n_dates,
        "n_stock_day_pairs": loader.n_pairs,
        "stocks_per_date": {
            "min": int(spd[0]) if n else 0,
            "median": float(np.median(spd)) if n else 0.0,
            "max": int(spd[-1]) if n else 0,
        },
        "window": "BOS + last context days before target day; target day's signed log_ret is the last p_rt column",
        "score": "softmax(shift_coarse[:, -1, :V_c]) @ coarse_logret_centers (expected signed normalized log_ret)",
        "realized": "reg_signed[:, -1] (target-day signed normalized log_ret)",
    }
    path = os.path.join(metrics_dir, "collation_fingerprint.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)
    return summary


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

    # GPT architecture: Exp 03 reviewed selection (depth6, 256d/6L/4h/GQA-1).
    # Explicit CLI overrides are required by the architecture-scaling experiment;
    # otherwise retain the reviewed baseline so prior in-process mutations cannot
    # leak into an ordinary run.
    ModelConfig.dim = args.dim or ModelConfig.dim
    ModelConfig.depth = args.depth or ModelConfig.depth
    ModelConfig.heads = args.heads or ModelConfig.heads
    ModelConfig.num_kv_heads = args.num_kv_heads or ModelConfig.num_kv_heads
    ModelConfig.dropout = args.dropout
    ModelConfig.ffn_multiplier = args.ffn_multiplier or ModelConfig.ffn_multiplier
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

    # Branch F (LoRA-per-regime): mode flag + incompatibility guards.  lora_r>0
    # is an independent mode (forces adamw + token-budget batched path; freezes
    # everything except the LoRA A/B matrices).  Rank / sample-weight / MTP runs
    # are untouched when lora_r=0.
    lora_mode = args.lora_r > 0
    regime_thresholds = None
    regime_quantiles = None
    lora_config = None
    if lora_mode:
        if args.optimizer != "adamw":
            raise ValueError("Branch F (--lora_r) requires --optimizer adamw")
        if args.reasoning:
            raise ValueError("Branch F (--lora_r) is incompatible with --reasoning")
        if getattr(args, "mtp", False):
            raise ValueError("Branch F (--lora_r) is incompatible with Branch D (--mtp)")
        if getattr(args, "sample_weight_mode", "none") != "none":
            raise ValueError("Branch F (--lora_r) is incompatible with Branch C "
                             "(--sample_weight_mode)")
        if args.loss != "ce":
            raise ValueError("Branch F (--lora_r) requires --loss ce "
                             "(the per-regime loss gate reduces CE)")
        if getattr(args, "dm_weight", 0.0) > 0:
            raise ValueError("Branch F (--lora_r) is incompatible with Branch A "
                             "(--dm_weight > 0)")
        regime_quantiles = tuple(
            float(x) for x in args.regime_quantiles.split(",")
        )
        if len(regime_quantiles) != 2:
            raise ValueError("--regime_quantiles must have exactly two "
                             "comma-separated values")
        # Cross-sectional terciles over the TRAIN split only (pre-cutoff rows),
        # exactly matching the single source of truth in regime.py.
        train_logrets = [
            s["features_raw"][:_stock_cutoff_idx(s, DataConfig.cutoff_date), 0]
            for s in train_s
        ]
        regime_thresholds = regime.compute_regime_thresholds(
            train_logrets, window=args.regime_window, quantiles=regime_quantiles)
        lora_config = {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "n_regimes": 3,
            "regime_window": args.regime_window,
            "regime_quantiles": list(regime_quantiles),
            "regime_thresholds": list(regime_thresholds),
        }
        print(f"  [BranchF] LoRA-per-regime: r={args.lora_r}, alpha={args.lora_alpha}, "
              f"n_regimes=3, window={args.regime_window}, "
              f"thresholds={tuple(round(t, 6) for t in regime_thresholds)}")

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
                                max_seq_len=args.max_seq_len,
                                regime_thresholds=regime_thresholds,
                                regime_window=args.regime_window)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=args.max_seq_len,
                              regime_thresholds=regime_thresholds,
                              regime_window=args.regime_window)
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
    # Branch F requires the token-budget batched path (right-pad + is_causal),
    # which is where the regime_ids channel flows into the model / loss.
    if lora_mode and not batched:
        raise ValueError("Branch F (--lora_r) requires the token-budget batched "
                         "path (--batch_tokens > 0)")
    # Branch B: cross-sectional ListNet fine-tuning replaces the standard loader.
    rank_mode = getattr(args, "rank_weight", 0.0) > 0
    if lora_mode and rank_mode:
        raise ValueError("Branch F (--lora_r) is incompatible with Branch B "
                         "(--rank_weight > 0)")
    rank_loader = None
    rank_centers = None
    coll_fp = None
    if rank_mode:
        # Exact-accumulation and the batched path both assume the token-budget
        # loader; the rank loader has its own sequence accounting.
        exact_accumulation = False
        batched = False

    # Branch C: non-stationarity adaptation via per-position loss re-weighting.
    # recency decays exponentially toward the cutoff; regime balances the
    # trailing-20d realized-vol terciles.  Weights are a loss-level overlay that
    # is bit-identical to no weighting when sample_weight_mode == "none", so the
    # loader/optimizer-step schedule is unchanged.
    sample_weight_mode = getattr(args, "sample_weight_mode", "none")
    sample_weight_cfg = None
    if sample_weight_mode != "none":
        if rank_mode:
            raise ValueError(
                "--sample_weight_mode requires the standard per-stock loaders; "
                "it is not supported with the Branch-B rank loader"
            )
        stocks_by_symbol = {s["symbol"]: s for s in train_s}
        sample_weight_cfg = attach_sample_weights(
            train_seqs,
            stocks_by_symbol,
            mode=sample_weight_mode,
            recency_tau_days=args.recency_tau_days,
            weight_clip=(args.weight_clip_lo, args.weight_clip_hi),
        )
        _write_weighted_token_diagnostics(
            metrics_dir,
            train_seqs,
            sample_weight_cfg,
            ModelConfig.vocab_size,
            ModelConfig.vocab_fine,
        )
        print(
            f"  [BranchC] sample_weight_mode={sample_weight_mode}, "
            f"tau_days={args.recency_tau_days}, "
            f"weight_clip=({args.weight_clip_lo}, {args.weight_clip_hi})"
        )
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

    if rank_mode:
        rank_loader = CrossSectionalRankLoader(
            train_seqs,
            context=getattr(args, "rank_context", 64),
            cross_section=getattr(args, "rank_cross_section", 64),
            min_stocks=getattr(args, "rank_min_stocks", 30),
            loader_seed=(
                controlled_loader_seed
                if controlled_loader_seed >= 0
                else 42
            ),
            max_dates_per_epoch=getattr(args, "rank_max_dates_per_epoch", 0),
            cap_per_date=getattr(args, "rank_cap_per_date", 0),
        )
        train_loader = rank_loader
        val_loader = TokenBudgetLoader(
            val_seqs, batch_tokens, shuffle=False, cap_B=batch_cap)
        rank_centers = _load_or_build_coarse_logret_centers(
            tokenizer, metrics_dir).to(device)
        coll_fp = _write_collation_fingerprint(metrics_dir, rank_loader, args)
        print(f"Loader: CrossSectionalRankLoader (Branch B), "
              f"n_dates={rank_loader.n_dates}, n_pairs={rank_loader.n_pairs}, "
              f"context={getattr(args, 'rank_context', 64)}, "
              f"cross_section={getattr(args, 'rank_cross_section', 64)}")

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
        # Branch F: attach zero-init per-regime LoRA adapters BEFORE loading the
        # dense base weights (LoRA keys are absent from dense checkpoints and
        # stay at their zero init).  A ~ N(0, 0.02), B = 0 => the attached model
        # is bit-identical to the dense baseline at epoch 0.
        lora_params = None
        if lora_mode:
            lora_params = model.attach_lora(args.lora_r, args.lora_alpha, n_regimes=3)
            # attach_lora runs after the model was moved to device, so move the
            # newly created LoRA A/B matrices to the device too.
            model = model.to(device)
        if args.base_checkpoint and os.path.exists(args.base_checkpoint):
            base_ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
            # Branch D adds head_future.* unconditionally; Branch F adds
            # blocks.*.lora.*.  Older checkpoints lack them, so load non-strict
            # and verify only those prefixes are missing.
            res = model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
            missing = list(res.missing_keys)
            unexpected = list(res.unexpected_keys)
            if missing and any(
                not (k.startswith("head_future.") or ".lora." in k) for k in missing
            ):
                raise RuntimeError(
                    f"Unexpected missing state_dict keys (not head_future/.lora): {missing}")
            if unexpected:
                raise RuntimeError(f"Unexpected state_dict keys in checkpoint: {unexpected}")
            miss_note = ""
            if missing:
                miss_note = " [missing: "
                if any(k.startswith("head_future.") for k in missing):
                    miss_note += "head_future.* "
                if any(".lora." in k for k in missing):
                    miss_note += "lora.* "
                miss_note = miss_note.rstrip() + ", random init]"
            print(f"  Loaded base checkpoint: {args.base_checkpoint} "
                  f"(val_loss={base_ckpt.get('val_loss', 'N/A')}){miss_note}")
        if lora_params is not None:
            # Freeze every non-LoRA parameter (embeddings, blocks, norms, heads);
            # only the LoRA A/B matrices train, isolating the regime-conditioning
            # effect from head/backbone retuning (Branch F).
            for p in model.parameters():
                p.requires_grad_(False)
            for p in lora_params:
                p.requires_grad_(True)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  [BranchF] Frozen backbone; trainable LoRA params: "
                  f"{trainable:,} / {total:,}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if TrainingConfig.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # ── Early stopping ──
    early_stop = None
    esp = getattr(args, "early_stop_patience", 0)
    if esp > 0:
        early_stop = EarlyStopping(patience=esp, min_delta=1e-4)
        print(f"  [early_stop] patience={esp}")

    # Branch D (MTP): parse offsets/weights; force AdamW; freeze the backbone
    # (keep only the future heads trainable) during phase 1.
    mtp_unfrozen = False
    if lora_mode:
        # Branch F: the backbone was frozen at model build; only the LoRA A/B
        # parameters have requires_grad=True, so AdamW on the trainable subset
        # is exactly AdamW on the LoRA params.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr,
                                      weight_decay=args.weight_decay)
        optimizer_adam = None
        print(f"Optimizer: AdamW (LoRA params only, backbone frozen), "
              f"lr={effective_lr}, wd={args.weight_decay}")
    elif getattr(args, "mtp", False):
        args.mtp_offsets = [int(v) for v in args.mtp_offsets.split(",")]
        args.mtp_weights = [float(v) for v in args.mtp_weights.split(",")]
        if len(args.mtp_offsets) != len(args.mtp_weights):
            raise ValueError("--mtp_offsets and --mtp_weights must have equal length")
        if args.optimizer != "adamw":
            raise ValueError("Branch D (--mtp) requires --optimizer adamw "
                             "(freezing the backbone is incompatible with "
                             "Muon's 2D parameter grouping)")
        if not batched:
            raise ValueError("Branch D (--mtp) requires the token-budget batched "
                             "path (--batch_tokens > 0)")
        for name, p in model.named_parameters():
            p.requires_grad = bool(name.startswith("head_future"))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.mtp_head_lr,
                                      weight_decay=args.weight_decay)
        optimizer_adam = None
        print(f"Optimizer: AdamW (MTP phase 1, frozen backbone), "
              f"lr={args.mtp_head_lr}, wd={args.weight_decay}")
    elif args.optimizer == "muon":
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
    n_rank_batches = len(rank_loader) if rank_loader is not None else 0
    phase1_end = max(1, epochs * 10 // 30)
    phase2_end = max(phase1_end + 1, epochs * 20 // 30)
    n_short = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 2000)
    n_med = sum(1 for s in train_seqs if s["input_ids"].shape[0] <= 5000)

    def accumulation_for_epoch(epoch_index):
        if rank_mode:
            # Each rank microbatch holds ~rank_cross_section sequences, so the
            # optimizer-step budget scales the standard accumulation accordingly.
            return accum * getattr(args, "rank_cross_section", 64)
        if constant_accumulation or epoch_index < 15:
            return accum
        return accum * 2

    def sequences_for_epoch(epoch_index):
        if rank_mode:
            return n_rank_batches * getattr(args, "rank_cross_section", 64)
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
        "sampling_config": sample_weight_cfg,
        "lora_config": lora_config,
        "epoch": [],
        "train_loss": [],
        "train_coarse_loss": [],
        "train_fine_loss": [],
        "train_het_loss": [],
        "train_rank_listnet": [],
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
    if lora_mode:
        # Branch F per-regime val coarse CE red-line series (populated per epoch
        # during validation; absent in dense runs so the schema is unchanged).
        history["val_regime_ce_0"] = []
        history["val_regime_ce_1"] = []
        history["val_regime_ce_2"] = []
    history_path = os.path.join(metrics_dir, f"history_{args.tag}.json")
    if args.history_per_epoch and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                stored_history = json.load(f)
            for key in history:
                if key in ("schema", "sampling_config", "lora_config"):
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
        # Branch D (MTP) phase 2: unfreeze the backbone once, rebuild the
        # optimizer (low LR) and the WSD scheduler over the remaining updates.
        if (getattr(args, "mtp", False) and not mtp_unfrozen
                and epoch == args.mtp_freeze_epochs):
            for p in model.parameters():
                p.requires_grad = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr,
                                          weight_decay=args.weight_decay)
            remaining = sum(
                max(1, sequences_for_epoch(e) // accumulation_for_epoch(e))
                for e in range(epoch, epochs))
            scheduler = build_wsd_scheduler(optimizer, max(remaining, 1),
                                            warmup_ratio=0.0)
            mtp_unfrozen = True
            print(f"  [MTP] Phase 2: backbone unfrozen, AdamW lr={effective_lr}, "
                  f"remaining updates={remaining}")
        epoch_t0 = time.time()
        epoch_step_start = global_step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Update curriculum epoch for length filtering
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        # For single-seq mode: apply curriculum filter by rebuilding loader each epoch
        # (batched mode filters internally via TokenBudgetLoader.set_epoch).
        if bs == 1 and use_curriculum and not batched and not rank_mode:
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
        rank_loss_acc = torch.zeros((), device=device)
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
            if rank_mode:
                input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target, rank_date_id = _to_device_rank(batch, device)
                sample_weights = None
                regime_ids = None
            else:
                input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target, sample_weights, regime_ids = _to_device(batch, device)
                rank_date_id = None

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
                        if getattr(args, "mtp", False):
                            fts = _future_targets(target, args.mtp_offsets,
                                                  ModelConfig.vocab_size)
                            coarse_logits, fine_logits, reg_pred, _, future_logits = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, reg_targets=reg_target,
                                fine_targets=fine_target, future_targets=fts,
                                compute_reg_loss=False,
                                regime_ids=regime_ids)
                            loss_sum, component_sums, n_seq = compute_batched_loss(
                                coarse_logits, target, fine_logits, fine_target,
                                reg_pred, reg_target, args,
                                sample_weights=sample_weights,
                                future_logits=future_logits,
                                regime_ids=regime_ids)
                        elif args.heteroscedastic:
                            coarse_logits, fine_logits, reg_pred, _ = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, reg_targets=reg_target,
                                fine_targets=fine_target,
                                compute_reg_loss=False,
                                regime_ids=regime_ids)
                            loss_sum, component_sums, n_seq = compute_batched_loss(
                                coarse_logits, target, fine_logits, fine_target,
                                reg_pred, reg_target, args,
                                sample_weights=sample_weights,
                                regime_ids=regime_ids)
                        else:
                            coarse_logits, fine_logits = model(
                                input_ids, time_id, pos_id, mask,
                                va_values=va_val, fine_targets=fine_target,
                                regime_ids=regime_ids)
                            reg_pred = None
                            loss_sum, component_sums, n_seq = compute_batched_loss(
                                coarse_logits, target, fine_logits, fine_target,
                                reg_pred, reg_target, args,
                                sample_weights=sample_weights,
                                regime_ids=regime_ids)
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

                if rank_mode:
                    # Branch B: one microbatch = one date's cross-section.
                    # forward uses compute_reg_loss=False; the het NLL is skipped
                    # inside compute_batched_loss in rank mode because reg_target
                    # holds signed values, not |z|.
                    t_fwd_start = (
                        time.perf_counter()
                        if profiler and (bi + 1) % 20 == 0
                        else 0
                    )
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        coarse_logits, fine_logits, reg_pred, _ = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, reg_targets=reg_target,
                            fine_targets=fine_target,
                            compute_reg_loss=False)
                        loss_sum, component_sums, n_seq = compute_batched_loss(
                            coarse_logits, target, fine_logits, fine_target,
                            reg_pred, reg_target, args,
                            rank_date_ids=rank_date_id,
                            rank_centers=rank_centers)
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
                    rank_loss_acc += component_sums["rank_listnet"]
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
                            fine_targets=fine_target,
                            regime_ids=regime_ids)
                    else:
                        coarse_logits, fine_logits = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, fine_targets=fine_target,
                            regime_ids=regime_ids)
                        het_loss = None

                    # Coarse loss (main)
                    shift_coarse = coarse_logits[:, :-1, :].contiguous()
                    shift_targets = target.contiguous()
                    if (shift_targets == -100).all():
                        continue
                    if sample_weights is not None:
                        # Branch C: per-position weighted reduction (bs=1).
                        if args.loss == "focal":
                            loss = _per_seq_focal(shift_coarse, shift_targets, gamma=args.gamma,
                                                  label_smoothing=args.label_smoothing,
                                                  entropy_alpha=args.entropy_alpha,
                                                  sample_weights=sample_weights)
                        else:
                            loss = _per_seq_ce(shift_coarse, shift_targets, ignore_index=-100,
                                               label_smoothing=args.label_smoothing,
                                               sample_weights=sample_weights)
                    elif args.loss == "focal":
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
                    # Note: bs=1 het is the model's scalar NLL (unweighted); the
                    # batched Branch C path weights het via _per_seq_het.
                    fine_mask = fine_target != -100
                    if fine_mask.any():
                        if sample_weights is not None:
                            fine_loss = _per_seq_ce(fine_logits, fine_target,
                                                    ignore_index=-100,
                                                    sample_weights=sample_weights)
                            fine_component = fine_loss.detach()
                            loss = loss + args.fine_weight * fine_loss
                        else:
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

        # Flush any remaining accumulated gradient (batched/rank leftover < accum)
        if (batched or rank_mode) and seqs_in_accum > 0:
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
            if (batched or rank_mode)
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
        avg_train_rank = (rank_loss_acc / max(n_loss, 1)).item() if rank_mode else None

        # Validation (always CE for comparable val_loss)
        model.eval()
        vlosses = []
        v_fine_losses = []
        v_het_losses = []
        val_pred_tokens = []
        # Branch F: per-regime val coarse CE for red-line monitoring (each regime
        # should track the dense baseline's same-regime value).
        regime_val_ce = {r: [] for r in range(3)} if lora_mode else None
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, target, fine_target, time_id, pos_id, mask, va_val, reg_target, _sample_weights, _regime_ids = _to_device(batch, device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    if args.heteroscedastic:
                        coarse_logits, fine_logits, reg_pred, val_het = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, reg_targets=reg_target,
                            fine_targets=fine_target,
                            compute_reg_loss=not batched,
                            regime_ids=_regime_ids if lora_mode else None)
                    else:
                        coarse_logits, fine_logits = model(
                            input_ids, time_id, pos_id, mask,
                            va_values=va_val, fine_targets=fine_target,
                            regime_ids=_regime_ids if lora_mode else None)
                        reg_pred = None
                        val_het = None
                    shift_logits = coarse_logits[:, :-1, :].contiguous()
                    shift_targets = target.contiguous()
                    if lora_mode and _regime_ids is not None:
                        ce_pos = F.cross_entropy(
                            shift_logits.reshape(-1, shift_logits.size(-1)),
                            shift_targets.reshape(-1), reduction="none",
                            ignore_index=-100,
                        ).view_as(shift_targets)
                        rmask = _regime_ids[:, : shift_targets.shape[1]]
                        for r in range(3):
                            m = (rmask == r) & (shift_targets != -100)
                            if m.any():
                                regime_val_ce[r].append(ce_pos[m].mean().item())
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

        # Branch F: per-regime val coarse CE averages (red-line monitoring).
        if lora_mode and regime_val_ce is not None:
            for r in range(3):
                vals = regime_val_ce[r]
                avg = (sum(vals) / len(vals)) if vals else float("nan")
                history[f"val_regime_ce_{r}"].append(avg)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train)
        history["train_coarse_loss"].append(avg_train_coarse)
        history["train_fine_loss"].append(avg_train_fine)
        history["train_het_loss"].append(avg_train_het)
        history["train_rank_listnet"].append(avg_train_rank)
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
                "rank_weight": args.rank_weight,
                "rank_context": args.rank_context,
                "rank_cross_section": args.rank_cross_section,
                "rank_min_stocks": args.rank_min_stocks,
                "rank_temperature": args.rank_temperature,
                "collation_fingerprint": coll_fp,
                "lora_config": lora_config,
                # Top-level keys consumed by eval_helpers.load_gpt_lora
                # (checked after the "config" dict).
                "lora_r": args.lora_r if lora_mode else None,
                "lora_alpha": args.lora_alpha if lora_mode else None,
                "n_regimes": 3 if lora_mode else None,
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
                "rank_weight": args.rank_weight,
                "rank_context": args.rank_context,
                "rank_cross_section": args.rank_cross_section,
                "rank_min_stocks": args.rank_min_stocks,
                "rank_temperature": args.rank_temperature,
                "collation_fingerprint": coll_fp,
                "lora_config": lora_config,
                "lora_r": args.lora_r if lora_mode else None,
                "lora_alpha": args.lora_alpha if lora_mode else None,
                "n_regimes": 3 if lora_mode else None,
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
                    "train_rank_listnet": avg_train_rank,
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
                    "sampling_config": sample_weight_cfg,
                    "lora_config": lora_config,
                    "val_regime_ce_0": (
                        history["val_regime_ce_0"][-1] if lora_mode else None
                    ),
                    "val_regime_ce_1": (
                        history["val_regime_ce_1"][-1] if lora_mode else None
                    ),
                    "val_regime_ce_2": (
                        history["val_regime_ce_2"][-1] if lora_mode else None
                    ),
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
            "rank_weight": args.rank_weight,
            "rank_context": args.rank_context,
            "rank_cross_section": args.rank_cross_section,
            "rank_min_stocks": args.rank_min_stocks,
            "rank_temperature": args.rank_temperature,
            "collation_fingerprint": coll_fp,
            "lora_config": lora_config,
            "lora_r": args.lora_r if lora_mode else None,
            "lora_alpha": args.lora_alpha if lora_mode else None,
            "n_regimes": 3 if lora_mode else None,
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
    # Branch A: daily marginal distribution matching (SFT).  Default 0.0 keeps
    # the production CPT recipe unchanged; Branch A passes --dm_weight > 0.
    parser.add_argument("--dm_weight", type=float, default=0.0,
                        help="Branch A: weight for symmetric-KL distribution-matching "
                             "loss aligning predicted vs target coarse-token histogram "
                             "(0.0 = disabled, keeps reviewed CPT recipe)")
    # Branch B: cross-sectional ListNet ranking fine-tuning.  Default 0.0 keeps
    # the production CPT recipe unchanged; > 0 enables the rank mode (see
    # CrossSectionalRankLoader / _listnet_loss).
    parser.add_argument("--rank_weight", type=float, default=0.0,
                        help="Branch B: weight for ListNet cross-sectional ranking "
                             "loss (0.0 = disabled, keeps reviewed CPT recipe)")
    parser.add_argument("--rank_context", type=int, default=64,
                        help="Branch B: trading days before the target day in each "
                             "rank window (row = [BOS] + context days)")
    parser.add_argument("--rank_cross_section", type=int, default=64,
                        help="Branch B: max stocks sampled per date cross-section")
    parser.add_argument("--rank_min_stocks", type=int, default=30,
                        help="Branch B: drop dates with fewer stocks than this")
    parser.add_argument("--rank_temperature", type=float, default=1.0,
                        help="Branch B: ListNet softmax temperature")
    parser.add_argument("--rank_max_dates_per_epoch", type=int, default=0,
                        help="Branch B: cap on dates per epoch (0 = all eligible dates)")
    parser.add_argument("--rank_cap_per_date", type=int, default=0,
                        help="Branch B: max (stock, day) pairs retained per date in "
                             "the inverted index (0 = unlimited; bounds memory on "
                             "full-data runs)")
    # Branch C: non-stationarity adaptation (recency / regime per-position
    # sample weighting).  Default "none" keeps the production CPT recipe
    # unchanged; weights are a loss-level overlay so the loader/optimizer-step
    # schedule is unchanged (identity weighting = bit-identical to today).
    parser.add_argument("--sample_weight_mode", type=str, default="none",
                        choices=["none", "recency", "regime", "combined"],
                        help="Branch C: per-position loss re-weighting mode "
                             "(none = disabled, keeps reviewed CPT recipe)")
    parser.add_argument("--recency_tau_days", type=int, default=504,
                        help="Branch C: recency exponential-decay time constant "
                             "in trading days")
    parser.add_argument("--weight_clip_lo", type=float, default=0.05,
                        help="Branch C: lower clip of per-position sample weight")
    parser.add_argument("--weight_clip_hi", type=float, default=20.0,
                        help="Branch C: upper clip of per-position sample weight")
    # Branch D: multi-token prediction heads (MTP).  Default off keeps the
    # reviewed CPT recipe unchanged; --mtp forces --optimizer adamw and the
    # token-budget batched path.
    parser.add_argument("--mtp", action="store_true",
                        help="Branch D: enable future coarse heads (t+2/t+3/t+4)")
    parser.add_argument("--mtp_offsets", type=str, default="2,3,4",
                        help="Branch D: comma-separated future offsets in days")
    parser.add_argument("--mtp_weights", type=str, default="0.5,0.25,0.125",
                        help="Branch D: comma-separated loss weights per future head")
    parser.add_argument("--mtp_freeze_epochs", type=int, default=1,
                        help="Branch D: freeze backbone for this many epochs while "
                             "training only the future heads")
    parser.add_argument("--mtp_head_lr", type=float, default=1e-3,
                        help="Branch D: AdamW LR for the future heads in phase 1")
    # Branch F: LoRA-per-regime.  Default lora_r=0 keeps the reviewed CPT recipe
    # unchanged; --lora_r > 0 forces --optimizer adamw, the token-budget batched
    # path, and freezes everything except the LoRA A/B matrices (which are
    # conditioned on trailing-20d realized-vol tercile regime ids per token).
    parser.add_argument("--lora_r", type=int, default=0,
                        help="Branch F: LoRA rank (0 = disabled, keeps dense recipe)")
    parser.add_argument("--lora_alpha", type=float, default=8.0,
                        help="Branch F: LoRA alpha scaling (alpha/r) applied to the "
                             "low-rank correction")
    parser.add_argument("--regime_window", type=int, default=20,
                        help="Branch F: trailing realized-vol window (days) used for "
                             "regime labels (inclusive, strict point-in-time)")
    parser.add_argument("--regime_quantiles", type=str, default="0.333,0.667",
                        help="Branch F: comma-separated tercile quantiles for the "
                             "train-split regime cutoffs")
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
    # Exp 03/04-A/04-B controlled schedule uses 6144 tokens per microbatch
    # with accumulation_steps=32 for an effective batch of ~196k tokens.
    parser.add_argument("--batch_tokens", type=int, default=6144,
                        help="Max tokens per batch (B*max_len). Adaptive batching for "
                             "GPU occupancy. 0 = legacy single-seq (bs=1).")
    parser.add_argument("--batch_cap", type=int, default=64,
                        help="Hard cap on sequences per batch (safety for very short stocks)")
    # Exp 03/04-A/04-B use a fixed architecture-independent loader seed so
    # data order is identical across arms and reproducible across reruns.
    parser.add_argument(
        "--controlled_loader_seed",
        type=int,
        default=42,
        help="Architecture-independent token-loader seed (-1 uses the "
             "process-global RNG).",
    )
    parser.add_argument(
        "--exact_accumulation_boundaries",
        action="store_true",
        default=True,
        help="Partition token-budget batches into deterministic blocks that "
             "end exactly at each optimizer update. Requires "
             "--controlled_loader_seed >= 0.",
    )
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Restore torch.use_deterministic_algorithms + cuBLAS "
                             "workspace pinning. Costs ~4%% and does NOT make this "
                             "trainer reproducible (SDPA backward stays "
                             "non-deterministic); kept for debugging only.")
    # Exp 04-A/04-B audit showed the historical epoch-15 doubling removes
    # optimizer steps without reducing forward/backward work; keep constant.
    parser.add_argument(
        "--constant_accumulation",
        action="store_true",
        default=True,
        help="Keep TrainingConfig.accumulation_steps for every epoch instead of "
             "doubling it at absolute epoch 15.",
    )
    main(parser.parse_args())
