"""Shared transformer building blocks for KronosPreview (GPT) and KronosBert.

Extracted to avoid ~120 lines of verbatim duplication between the two model files.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # RoPE angles must be constructed in fp32. CUDA autocast treats einsum
        # as a low-precision op and otherwise rounds integer positions to bf16
        # before the angles are formed (positions 3000 and 3001 then collide).
        with torch.amp.autocast(position_ids.device.type, enabled=False):
            freqs = torch.einsum(
                "bi,d->bid", position_ids.float(), self.inv_freq.float()
            )
            return torch.sin(freqs), torch.cos(freqs)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, sin, cos):
    # sin, cos: [B, N, d//2] -> [B, 1, N, d//2]
    # Keep angle generation accurate, then match Q/K precision for the rotary
    # multiply so mixed-precision attention retains its original memory cost.
    sin = sin.to(dtype=q.dtype).unsqueeze(1)
    cos = cos.to(dtype=q.dtype).unsqueeze(1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    # Algebraically and bitwise identical to concatenating sin/cos to head_dim,
    # but avoids two full-size temporary tensors and the rotate-half temporary.
    q_out = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
    k_out = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
    return q_out, k_out


class Attention(nn.Module):
    """Multi-head attention with GQA support. Causal mask is optional."""
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
        # One fused QKV matmul instead of three. Concatenating the weights costs a
        # sub-megabyte copy and measured 1.024x on the real training step; the
        # parameters stay separate so existing checkpoints load unchanged.
        qkv = F.linear(x, torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0))
        q, k, v = qkv.split(
            (self.heads * self.head_dim,
             self.num_kv_heads * self.head_dim,
             self.num_kv_heads * self.head_dim), dim=-1)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, sin, cos)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        drop_rate = self.dropout_p if self.training else 0.0
        # Use is_causal=True when no explicit mask is provided — avoids allocating [N,N] mask
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_rate)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SiLU-gated feed-forward network."""
    def __init__(self, dim, multiplier=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * multiplier)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (RMSNorm + Attention + FFN)."""
    def __init__(self, dim, heads, num_kv_heads, ffn_multiplier, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, heads, num_kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_multiplier, dropout)

    def forward(self, x, sin, cos, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), sin, cos, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _BaseTransformer(nn.Module):
    """Shared backbone for KronosPreview (GPT/causal) and KronosBert (bidirectional).

    Subclasses only need to override `_run_blocks()` to insert extra processing
    (e.g. reasoning blocks) between the transformer stack and the final norm.
    """
    def __init__(self, cfg, n_special=3):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size + n_special, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)
        self.va_proj = nn.Sequential(
            nn.Linear(2, cfg.va_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(cfg.va_hidden_dim, cfg.dim, bias=True),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.num_kv_heads,
                             cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def _prepare_inputs(self, input_ids, time_ids, position_ids,
                        attn_mask=None, **kwargs):
        """Ensure [B, N] format and expand 3D mask to 4D for SDPA.

        Returns (input_ids, time_ids, position_ids, attn_mask) with batch dim.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
            for k, v in kwargs.items():
                if v is not None:
                    kwargs[k] = v.unsqueeze(0)

        # SDPA requires [B, 1, N, N] when B > 1
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        return input_ids, time_ids, position_ids, attn_mask, kwargs

    def _embed(self, input_ids, time_ids, va_values=None):
        """Build input embeddings from tokens, time, and VA features."""
        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])
        if va_values is not None:
            x = x + self.va_proj(va_values)
        return x

    def _run_blocks(self, x, sin, cos, attn_mask=None):
        """Run the transformer stack. Override in subclasses for extra processing."""
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attn_mask, use_reentrant=False)
            else:
                x = block(x, sin, cos, attn_mask)
        return x

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None, **kwargs):
        raise NotImplementedError
