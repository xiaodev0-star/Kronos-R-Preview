"""KronosBert: Bidirectional Transformer for calibration of KronosPreview predictions.

Architecture mirrors KronosPreview but uses FULL attention (no causal mask) and adds
a [MASK] token for MLM-style training. At inference time, the BERT model is used to
re-score / re-sample tokens that the GPT model has predicted, by checking bidirectional
context consistency.

All shared building blocks (RMSNorm, Attention, FeedForward, etc.) live in model/layers.py.

Special tokens:
  BOS  = vocab_size       (input_id)
  EOS  = vocab_size + 1
  MASK = vocab_size + 2   (new for BERT)

Embedding vocab size = vocab_size + 3.
"""
import torch
import torch.nn as nn

from config import ModelConfig
from model.layers import _BaseTransformer


class KronosBert(_BaseTransformer):
    """Bidirectional Transformer for stock sequence understanding.

    Same backbone as KronosPreview but:
      - attention is full (no causal mask) when no attn_mask is provided
      - embedding vocab is +3 (adds [MASK] special token)
      - outputs logits over the *coarse* vocabulary at every position
    """
    def __init__(self, cfg=None):
        cfg = cfg or ModelConfig
        # causal=False => full bidirectional attention (the whole point of the
        # BERT critic); SDPA uses is_causal=False on the efficient backend.
        super().__init__(cfg, n_special=3, causal=False)  # BOS + EOS + MASK
        self.vocab_base = cfg.vocab_size
        self.mask_id = cfg.vocab_size + 2
        self.head_coarse = nn.Linear(cfg.dim, cfg.vocab_size, bias=True)

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None,
                va_values=None):
        """Forward pass.

        Args:
            input_ids: [B, N] long (may contain MASK tokens)
            time_ids:  [B, N, 3] long
            position_ids: [B, N] long
            attn_mask: [B, N, N] bool (True=attend, False=block) — for padding/external use.
                        If None, FULL BIDIRECTIONAL attention (this is the whole
                        point of the BERT critic — do NOT let the shared
                        Attention fall through to its causal default).
            va_values: [B, N, 2] float
        Returns:
            logits: [B, N, vocab_base] — probability over coarse tokens at every position.
        """
        no_batch = input_ids.dim() == 1
        extra = {"va_values": va_values}
        input_ids, time_ids, position_ids, attn_mask, extra = self._prepare_inputs(
            input_ids, time_ids, position_ids, attn_mask, **extra)
        va_values = extra["va_values"]

        # Bidirectional default is handled by ``causal=False`` in __init__:
        # the shared Attention applies is_causal=False when attn_mask is None,
        # keeping SDPA on its efficient backend (no explicit [N,N] mask).

        x = self._embed(input_ids, time_ids, va_values)
        sin, cos = self.rotary(position_ids)
        x = self._run_blocks(x, sin, cos, attn_mask)
        x = self.norm(x)
        logits = self.head_coarse(x)  # [B, N, vocab_base]

        if no_batch:
            logits = logits.squeeze(0)
        return logits


def make_mlm_batch(input_ids, vocab_base, mask_id,
                   mlm_prob=0.15, ignore_index=-100, generator=None,
                   corrupt_fracs=(0.8, 0.1, 0.1)):
    """Build an MLM batch with BERT-style 80/10/10 corruption (F2).

    For every non-special position selected at ``mlm_prob``: 80% → [MASK],
    10% → a random ordinary token, 10% → left unchanged.  The position is a
    label in ALL three cases (the model must recover the original token), the
    standard BERT procedure.  The 10% random-replacement branch is the key
    distribution-robustness device for the critic: it shows the model
    "an unreasonable token inside an otherwise real sequence", the exact input
    shape it must handle at inference when GPT candidates are inserted.

    Args:
        input_ids: [N] long (the *original* token ids, including BOS/EOS)
        vocab_base: int — tokens >= this are special (BOS/EOS/MASK) and never masked
        mask_id: int — the [MASK] token id
        mlm_prob: masking probability (default 0.15)
        ignore_index: CE ignore value
        generator: optional torch.Generator for reproducible corruption
        corrupt_fracs: (mask_frac, random_frac, keep_frac), sums to 1
    Returns:
        mlm_ids: [N] long — corrupted input (MASK / random / unchanged)
        mlm_labels: [N] long — original ids at selected positions, else ignore_index
    """
    N = input_ids.shape[0]
    device = input_ids.device
    if generator is not None:
        rand = torch.rand(N, device=device, generator=generator)
    else:
        rand = torch.rand(N, device=device)
    # Build mask: candidates = non-special (id < vocab_base)
    is_special = (input_ids >= vocab_base)
    can_mask = (~is_special) & (rand < mlm_prob)

    f_mask, f_rnd, f_keep = corrupt_fracs
    if generator is not None:
        pick = torch.rand(N, device=device, generator=generator)
    else:
        pick = torch.rand(N, device=device)
    do_mask = can_mask & (pick < f_mask)
    do_random = can_mask & (pick >= f_mask) & (pick < f_mask + f_rnd)
    # remainder (pick >= f_mask + f_rnd) => keep original, but still a label

    mlm_ids = input_ids.clone()
    mlm_ids[do_mask] = mask_id
    if bool(do_random.any()):
        n_rnd = int(do_random.sum())
        random_tokens = torch.randint(0, vocab_base, (n_rnd,),
                                      device=device, dtype=torch.long)
        mlm_ids[do_random] = random_tokens
    labels = torch.full((N,), ignore_index, dtype=torch.long, device=device)
    labels[can_mask] = input_ids[can_mask]
    return mlm_ids, labels
