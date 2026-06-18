"""KronosBert: Bidirectional Transformer for calibration of KronosPreview (GPT-style) predictions.

Architecture mirrors KronosPreview but uses FULL attention (no causal mask) and adds
a [MASK] token for MLM-style training. At inference time, the BERT model is used to
re-score / re-sample tokens that the GPT model has predicted, by checking bidirectional
context consistency.

Key idea (the "BERT calibration" hypothesis):
  1. GPT (causal) predicts next-1-token: P_GPT(y_hat | history)
  2. BERT (bidirectional) sees history + [MASK] (and optionally future if known)
     and produces P_BERT(.|history,future)
  3. If y_hat is consistent with the bidirectional context, P_BERT(y_hat) is high
  4. Combine or resample based on P_BERT

Special tokens:
  - BOS = vocab_size       (input_id)
  - EOS = vocab_size + 1
  - MASK = vocab_size + 2  (new for BERT, used to mask positions during MLM)

Embedding vocab size is therefore vocab_size + 3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        freqs = torch.einsum("bi,d->bid", position_ids.float(), self.inv_freq)
        return torch.sin(freqs), torch.cos(freqs)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, sin, cos):
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    cos2 = torch.cat([cos, cos], dim=-1)
    sin2 = torch.cat([sin, sin], dim=-1)
    q_out = q * cos2 + _rotate_half(q) * sin2
    k_out = k * cos2 + _rotate_half(k) * sin2
    return q_out, k_out


class BertAttention(nn.Module):
    """Bidirectional multi-head attention (no causal mask by default).

    Accepts an optional attn_mask (True = attend, False = block) for padding.
    For pure bidirectional attention, pass an all-True mask or None.
    """
    def __init__(self, dim, heads, num_kv_heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // heads
        self.kv_groups = heads // num_kv_heads
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.dropout_p = dropout if dropout > 0.0 else 0.0

    def forward(self, x, sin, cos, attn_mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, sin, cos)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        dp = self.dropout_p if self.training else 0.0
        # SDPA: if attn_mask is None, full attention (no causal mask).
        # If provided, must be [B, 1, N, N] boolean — True=attend, False=block.
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, multiplier=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * multiplier)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class BertBlock(nn.Module):
    """Pre-norm bidirectional transformer block (no causal mask)."""
    def __init__(self, dim, heads, num_kv_heads, ffn_multiplier, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = BertAttention(dim, heads, num_kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_multiplier, dropout)

    def forward(self, x, sin, cos, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), sin, cos, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class KronosBert(nn.Module):
    """Bidirectional Transformer for stock sequence understanding.

    Same backbone as KronosPreview but:
      - attention is full (no causal mask) when no attn_mask is provided
      - embedding vocab is +3 (adds [MASK] special token)
      - outputs logits over the *coarse* vocabulary at every position

    Special tokens:
      BOS  = ModelConfig.vocab_size       (=1024)
      EOS  = ModelConfig.vocab_size + 1   (=1025)
      MASK = ModelConfig.vocab_size + 2   (=1026)  [new]
    Embedding size = vocab_size + 3 = 1027.
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig
        self.dim = cfg.dim
        self.vocab_base = cfg.vocab_size
        self.bos_id = cfg.vocab_size
        self.eos_id = cfg.vocab_size + 1
        self.mask_id = cfg.vocab_size + 2
        vocab_full = cfg.vocab_size + 3  # BOS, EOS, MASK

        self.token_emb = nn.Embedding(vocab_full, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)
        self.va_proj = nn.Sequential(
            nn.Linear(2, cfg.va_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(cfg.va_hidden_dim, cfg.dim, bias=True),
        )

        self.blocks = nn.ModuleList([
            BertBlock(cfg.dim, cfg.heads, cfg.num_kv_heads,
                      cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.dim)
        # Output head: predict coarse vocabulary (BOS/EOS/MASK excluded by training mask).
        self.head_coarse = nn.Linear(cfg.dim, cfg.vocab_size, bias=True)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None,
                va_values=None):
        """Forward pass.

        Args:
            input_ids: [B, N] long (may contain MASK tokens)
            time_ids:  [B, N, 3] long
            position_ids: [B, N] long
            attn_mask: [B, N, N] bool (True=attend, False=block) — for padding/external use.
                        If None, full bidirectional attention.
            va_values: [B, N, 2] float
        Returns:
            logits: [B, N, vocab_base] — probability over coarse tokens at every position.
        """
        no_batch = input_ids.dim() == 1
        if no_batch:
            input_ids = input_ids.unsqueeze(0)
            time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
            if va_values is not None:
                va_values = va_values.unsqueeze(0)

        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])
        if va_values is not None:
            x = x + self.va_proj(va_values)

        sin, cos = self.rotary(position_ids)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attn_mask, use_reentrant=False)
            else:
                x = block(x, sin, cos, attn_mask)

        x = self.norm(x)
        logits = self.head_coarse(x)  # [B, N, vocab_base]

        if no_batch:
            logits = logits.squeeze(0)
        return logits


def make_mlm_batch(input_ids, vocab_base, mask_id,
                   mlm_prob=0.15, ignore_index=-100, generator=None):
    """Build an MLM batch: randomly mask non-special tokens and produce labels.

    Args:
        input_ids: [N] long (the *original* token ids, including BOS/EOS)
        vocab_base: int — tokens >= this are special (BOS/EOS/MASK) and never masked
        mask_id: int — the [MASK] token id
        mlm_prob: masking probability (default 0.15)
        ignore_index: CE ignore value
    Returns:
        mlm_ids: [N] long — input with [MASK] replacing ~mlm_prob of non-special positions
        mlm_labels: [N] long — original ids at masked positions, else ignore_index
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

    mlm_ids = input_ids.clone()
    mlm_ids[can_mask] = mask_id
    labels = torch.full((N,), ignore_index, dtype=torch.long, device=device)
    labels[can_mask] = input_ids[can_mask]
    return mlm_ids, labels
