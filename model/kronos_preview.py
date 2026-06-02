"""Kronos-Preview: SDPA + RMSNorm + SiLU-gated FFN + RoPE。"""
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
    # sin, cos: [B, N, d//2] -> [B, 1, N, d//2]
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    # Expand to full head_dim by concatenating
    cos2 = torch.cat([cos, cos], dim=-1)  # [B, 1, N, head_dim]
    sin2 = torch.cat([sin, sin], dim=-1)
    q_out = q * cos2 + _rotate_half(q) * sin2
    k_out = k * cos2 + _rotate_half(k) * sin2
    return q_out, k_out


class Attention(nn.Module):
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


class TransformerBlock(nn.Module):
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


class KronosPreview(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig
        vocab_full = cfg.vocab_size + 2  # +2 for BOS/EOS
        self.token_emb = nn.Embedding(vocab_full, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.num_kv_heads,
                             cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.head_coarse = nn.Linear(cfg.dim, vocab_full, bias=True)
        self.head_fine = nn.Linear(cfg.dim, vocab_full, bias=True)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None, targets=None):
        # Normalize: ensure [B, N] format
        no_batch = input_ids.dim() == 1
        if no_batch:
            input_ids = input_ids.unsqueeze(0)
            time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
            if targets is not None:
                targets = targets.unsqueeze(0)

        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])

        sin, cos = self.rotary(position_ids)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attn_mask, use_reentrant=False)
            else:
                x = block(x, sin, cos, attn_mask)

        x = self.norm(x)
        logits_coarse = self.head_coarse(x)
        logits_fine = self.head_fine(x)

        loss = None
        if targets is not None:
            shift_logits = logits_coarse[:, :-1, :].contiguous()
            shift_targets = targets.contiguous()
            if (shift_targets != -100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

        if no_batch:
            logits_coarse = logits_coarse.squeeze(0)
            logits_fine = logits_fine.squeeze(0)

        return logits_coarse, logits_fine, loss


class CausalReasoningBlock(nn.Module):
    """Lightweight causal reasoning: cross-attention to learned memory tokens."""
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False), nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, memory):
        h = self.norm1(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + self.gate.tanh() * h
        x = x + self.ffn(self.norm2(x))
        return x


class KronosPreviewWithReasoning(nn.Module):
    """KronosPreview + CausalReasoningBlock after transformer blocks."""
    def __init__(self, base_model_state=None, n_reason_tokens=8, n_reason_layers=1):
        super().__init__()
        cfg = ModelConfig
        vocab_full = cfg.vocab_size + 2
        self.token_emb = nn.Embedding(vocab_full, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.num_kv_heads,
                             cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.head_coarse = nn.Linear(cfg.dim, vocab_full, bias=True)
        self.head_fine = nn.Linear(cfg.dim, vocab_full, bias=True)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)
        self.reason_tokens = nn.Parameter(torch.randn(1, n_reason_tokens, cfg.dim) * 0.02)
        self.reason_blocks = nn.ModuleList([
            CausalReasoningBlock(cfg.dim, heads=cfg.heads) for _ in range(n_reason_layers)
        ])
        self._gradient_checkpointing = False
        if base_model_state is not None:
            self.load_state_dict(base_model_state, strict=False)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None, targets=None):
        no_batch = input_ids.dim() == 1
        if no_batch:
            input_ids = input_ids.unsqueeze(0)
            time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
            if targets is not None:
                targets = targets.unsqueeze(0)

        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])
        sin, cos = self.rotary(position_ids)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attn_mask, use_reentrant=False)
            else:
                x = block(x, sin, cos, attn_mask)

        B = x.size(0)
        memory = self.reason_tokens.expand(B, -1, -1)
        for rblock in self.reason_blocks:
            x = rblock(x, memory)

        x = self.norm(x)
        logits_coarse = self.head_coarse(x)
        logits_fine = self.head_fine(x)

        loss = None
        if targets is not None:
            shift_logits = logits_coarse[:, :-1, :].contiguous()
            shift_targets = targets.contiguous()
            if (shift_targets != -100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1), ignore_index=-100)

        if no_batch:
            logits_coarse = logits_coarse.squeeze(0)
            logits_fine = logits_fine.squeeze(0)
        return logits_coarse, logits_fine, loss
