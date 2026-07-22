"""ELECTRA-style Binary Discriminator for Kronos BERT backbone.

Key idea (ELECTRA, Clark et al. 2020):
  - Instead of MLM (predict what the masked token was), train a DISCRIMINATOR
    to detect whether each token in the input is ORIGINAL or REPLACED.
  - This gives a DENSE binary classification signal over ALL positions
    (not just the 15% masked ones in MLM).
  - At inference: the discriminator's confidence P(original) at each position
    measures how "natural" the sequence is — directly usable for coherence scoring.

Training:
  - Take input sequence
  - Randomly replace ~15% of non-special tokens with random alternatives
  - Train binary classifier head (sigmoid) to predict original(1) vs replaced(0)
  - Loss = binary cross-entropy over ALL positions

Inference for GPT+BERT fusion:
  - Build candidate sequence: [BOS, tok_0, ..., tok_{p-1}, candidate_k]
  - Run discriminator → get P(original) for every token
  - Score = mean log P(original) across history tokens
  - Higher score = more coherent sequence = better candidate

Reference: Clark et al. "ELECTRA: Pre-training Text Encoders as Discriminators
           Rather Than Generators" (ICLR 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import _BaseTransformer
from config import ModelConfig


class KronosElectraDiscriminator(_BaseTransformer):
    """ELECTRA-style discriminator built on the Kronos BERT backbone.

    Same architecture as KronosBert (bidirectional, same shared backbone) but:
      - Output head: 1-dim sigmoid (binary: original vs replaced) instead of vocab logits
      - Training: binary cross-entropy over ALL tokens, not just masked positions
      - Inference: P(original) at each position = sequence "naturalness" signal

    Special tokens: BOS=vocab_size, EOS=vocab_size+1 (no MASK needed)
    """
    def __init__(self, cfg=None):
        cfg = cfg or ModelConfig
        super().__init__(cfg, n_special=3)  # BOS + EOS + MASK (use same embed dim as BERT)
        self.vocab_base = cfg.vocab_size
        # Binary classification head: 1 output with sigmoid
        # Predicts P(original) ∈ [0,1] for each position
        self.disc_head = nn.Linear(cfg.dim, 1, bias=True)

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None,
                va_values=None, return_logits=False):
        """Forward pass returning per-position binary logits.

        Args:
            input_ids: [B, N] long (may contain replaced tokens)
            time_ids:  [B, N, 3] long
            position_ids: [B, N] long
            attn_mask: [B, N, N] bool (True=attend), None for full bidirectional
            va_values: [B, N, 2] float
            return_logits: if True, return raw logits (before sigmoid)

        Returns:
            logits: [B, N, 1] — raw logits for binary CE loss
            or
            probs: [B, N, 1] — sigmoid probabilities P(original)
        """
        no_batch = input_ids.dim() == 1
        extra = {"va_values": va_values}
        input_ids, time_ids, position_ids, attn_mask, extra = self._prepare_inputs(
            input_ids, time_ids, position_ids, attn_mask, **extra)
        va_values = extra["va_values"]

        x = self._embed(input_ids, time_ids, va_values)
        sin, cos = self.rotary(position_ids)
        x = self._run_blocks(x, sin, cos, attn_mask)
        x = self.norm(x)
        logits = self.disc_head(x)  # [B, N, 1]

        if no_batch:
            logits = logits.squeeze(0)

        if return_logits:
            return logits
        return torch.sigmoid(logits)


def make_replaced_batch(input_ids, vocab_base, replace_prob=0.15,
                        ignore_index=-100, generator=None):
    """Build an ELECTRA-style replaced-token detection batch.

    Randomly replaces ~replace_prob of non-special tokens with random alternatives.
    Produces binary labels: 1=original, 0=replaced.

    This is a SIMPLIFIED version — the original ELECTRA uses a small MLM generator
    to produce "plausible" replacements. We use uniform random sampling for
    simplicity (similar to RTS in Liello et al. 2024).

    Args:
        input_ids: [N] long (original token ids, including BOS/EOS)
        vocab_base: int — tokens >= this are special (BOS/EOS) and never replaced
        replace_prob: fraction of non-special tokens to replace (default 0.15)
        ignore_index: positions with this label are ignored in loss (we don't ignore any: all positions get supervision)

    Returns:
        replaced_ids: [N] long — input with some tokens replaced
        labels: [N] long — 1=original, 0=replaced (for ALL positions)
    """
    N = input_ids.shape[0]
    device = input_ids.device
    rand = torch.rand(N, device=device)

    # Non-special tokens are candidates for replacement
    is_special = (input_ids >= vocab_base)
    can_replace = (~is_special) & (rand < replace_prob)

    replaced_ids = input_ids.clone()
    # Replace with random token from vocabulary
    n_replace = can_replace.sum().item()
    if n_replace > 0:
        random_tokens = torch.randint(0, vocab_base, (n_replace,),
                                       device=device, dtype=torch.long)
        replaced_ids[can_replace] = random_tokens

    # Labels: 1=original, 0=replaced — ALL positions get supervision
    labels = torch.ones(N, dtype=torch.float32, device=device)
    labels[can_replace] = 0.0

    return replaced_ids, labels


def discriminator_loss(logits, labels, ignore_special=True, vocab_base=None,
                       input_ids=None):
    """Binary cross-entropy loss for ELECTRA discriminator.

    Args:
        logits: [B, N, 1] or [B, N] — raw logits from disc_head
        labels: [B, N] — 1=original, 0=replaced
        ignore_special: if True, ignore special token positions (BOS/EOS)
        vocab_base: needed if ignore_special=True
        input_ids: needed if ignore_special=True

    Returns:
        scalar loss averaged over valid positions
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-1)  # [B, N]

    # BCE with logits
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    if ignore_special and vocab_base is not None and input_ids is not None:
        mask = (input_ids < vocab_base).float()  # only score non-special tokens
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    return loss


def discriminator_accuracy(logits, labels, vocab_base=None, input_ids=None):
    """Compute binary classification accuracy of the discriminator.

    Returns: (accuracy_overall, accuracy_on_replaced, accuracy_on_original)
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-1)

    preds = (logits > 0).float()  # logit > 0 → predict original
    correct = (preds == labels).float()

    # Overall accuracy (ignoring special tokens)
    if vocab_base is not None and input_ids is not None:
        mask = (input_ids < vocab_base).float()
        overall = (correct * mask).sum() / mask.sum().clamp(min=1)

        # Accuracy on replaced tokens only
        replaced_mask = mask * (labels == 0).float()
        acc_replaced = (correct * replaced_mask).sum() / replaced_mask.sum().clamp(min=1)

        # Accuracy on original tokens
        original_mask = mask * (labels == 1).float()
        acc_original = (correct * original_mask).sum() / original_mask.sum().clamp(min=1)
    else:
        overall = correct.mean()
        replaced_mask = (labels == 0).float()
        acc_replaced = (correct * replaced_mask).sum() / replaced_mask.sum().clamp(min=1)
        acc_original = (correct * (labels == 1).float()).sum() / (labels == 1).float().sum().clamp(min=1)

    return overall, acc_replaced, acc_original


# =============================================================================
# Coherence scoring for GPT+BERT fusion
# =============================================================================

@torch.no_grad()
def electra_coherence_score(discriminator, input_ids, time_ids, position_ids,
                             va_values=None):
    """Score a sequence's coherence using ELECTRA discriminator.

    Returns the mean P(original) across all history positions.
    Higher = discriminator thinks the sequence is "natural" = more coherent.

    This is what replaces the BERT mask-predict scoring in the fusion pipeline.
    """
    probs = discriminator(input_ids, time_ids, position_ids, va_values=va_values)
    # probs: [B, N, 1] or [N, 1]
    if probs.dim() == 3:
        probs = probs.squeeze(-1)  # [B, N]

    # Mean log-probability as coherence score
    # Using log-prob for numerical stability (sum instead of product)
    eps = 1e-8
    log_probs = torch.log(probs + eps)

    # Average across positions and batch
    score = log_probs.mean(dim=-1)  # [B] or scalar
    return score
