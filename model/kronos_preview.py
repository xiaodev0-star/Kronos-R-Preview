"""KronosPreview: GPT-style causal transformer for stock next-token prediction.

Architecture: SDPA + RMSNorm + SiLU-gated FFN + RoPE + Heteroscedastic regression head.
All shared building blocks (RMSNorm, Attention, FeedForward, etc.) live in model/layers.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model.layers import _BaseTransformer


def heteroscedastic_nll_loss(pred, target, ignore_val=-999.0):
    """Heteroscedastic Gaussian NLL loss for regression.

    Args:
        pred: [N, 2] tensor (mean, log_var)
        target: [N] tensor of regression targets
        ignore_val: sentinel for masked positions (BOS/EOS / padding)
    Returns:
        scalar loss, averaged over valid positions
    """
    mask = (target != ignore_val)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    mean = pred[mask, 0]
    log_var = pred[mask, 1]
    tgt = target[mask]
    # Clamp log_var for numerical stability (σ ∈ [e^{-5}, e^{2}])
    log_var = log_var.clamp(-5.0, 2.0)
    # Gaussian NLL: 0.5 * (log_var + (target - mean)^2 / exp(log_var))
    nll = 0.5 * (log_var + (tgt - mean).pow(2) / log_var.exp())
    return nll.mean()


class KronosPreview(_BaseTransformer):
    """GPT with dual-head prediction: coarse (macro pattern) + fine (micro detail).

    head_coarse predicts among V1 coarse codes (the main autoregressive target).
    head_fine predicts among V2 fine codes, conditioned on coarse embedding.
    Total head params: O(V1 + V2), not O(V1 * V2).
    """
    def __init__(self, cfg=None):
        cfg = cfg or ModelConfig
        super().__init__(cfg, n_special=2)
        self._vocab_l1 = cfg.vocab_size
        self._vocab_l2 = getattr(cfg, "vocab_fine", 256)
        self.head_coarse = nn.Linear(cfg.dim, cfg.vocab_size + 2, bias=True)
        # Fine head: conditions on [hidden_state, coarse_embedding]
        self._fine_emb = nn.Embedding(cfg.vocab_size + 2, cfg.dim)  # compact coarse repr
        self.head_fine = nn.Sequential(
            nn.Linear(cfg.dim * 2, cfg.dim, bias=True),
            nn.SiLU(),
            nn.Linear(cfg.dim, self._vocab_l2, bias=True),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=True),
            nn.SiLU(),
            nn.Linear(cfg.dim, 2, bias=True),
        )

    def _predict_reg(self, x, reg_targets):
        with torch.amp.autocast("cuda", enabled=False):
            shift_hidden = x[:, :-1, :].float().contiguous()
            reg_pred = self.head_reg(shift_hidden)
            shift_reg_targets = reg_targets[:, 1:].float().contiguous()
            het_loss = heteroscedastic_nll_loss(
                reg_pred.reshape(-1, 2), shift_reg_targets.reshape(-1), ignore_val=-999.0)
        return reg_pred, het_loss

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None,
                va_values=None, reg_targets=None, fine_targets=None,
                return_hidden=False):
        no_batch = input_ids.dim() == 1
        extra = {"va_values": va_values, "reg_targets": reg_targets, "fine_targets": fine_targets}
        input_ids, time_ids, position_ids, attn_mask, extra = self._prepare_inputs(
            input_ids, time_ids, position_ids, attn_mask, **extra)
        va_values, reg_targets, fine_targets = extra["va_values"], extra["reg_targets"], extra["fine_targets"]

        x = self._embed(input_ids, time_ids, va_values)
        sin, cos = self.rotary(position_ids)
        x = self._run_blocks(x, sin, cos, attn_mask)
        x = self.norm(x)
        coarse_logits = self.head_coarse(x)

        # Fine logits: conditioned on coarse embedding
        if fine_targets is not None:
            coarse_emb = self._fine_emb(input_ids[:, :fine_targets.shape[1]])
        else:
            coarse_pred = coarse_logits[:, :-1, :self._vocab_l1].argmax(dim=-1)
            coarse_emb = self._fine_emb(coarse_pred)
        T = coarse_emb.shape[1]
        fine_input = torch.cat([x[:, :T, :], coarse_emb], dim=-1)
        fine_logits = self.head_fine(fine_input)

        if reg_targets is not None:
            reg_pred, het_loss = self._predict_reg(x, reg_targets)
            if no_batch:
                coarse_logits = coarse_logits.squeeze(0)
                fine_logits = fine_logits.squeeze(0)
                if return_hidden:
                    return coarse_logits, fine_logits, reg_pred, het_loss, x.squeeze(0)
            if return_hidden:
                return coarse_logits, fine_logits, reg_pred, het_loss, x
            return coarse_logits, fine_logits, reg_pred, het_loss

        if no_batch:
            coarse_logits = coarse_logits.squeeze(0)
            fine_logits = fine_logits.squeeze(0)
            if return_hidden:
                return coarse_logits, fine_logits, x.squeeze(0)
        if return_hidden:
            return coarse_logits, fine_logits, x
        return coarse_logits, fine_logits


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


class KronosPreviewWithReasoning(KronosPreview):
    """KronosPreview + CausalReasoningBlock inserted after transformer stack.

    Inherits everything from KronosPreview; only overrides `_run_blocks` to
    inject reasoning cross-attention between the transformer blocks and the
    final norm.
    """
    def __init__(self, base_model_state=None, n_reason_tokens=8, n_reason_layers=1):
        super().__init__()
        cfg = ModelConfig
        self.reason_tokens = nn.Parameter(
            torch.randn(1, n_reason_tokens, cfg.dim) * 0.02)
        self.reason_blocks = nn.ModuleList([
            CausalReasoningBlock(cfg.dim, heads=cfg.heads)
            for _ in range(n_reason_layers)
        ])
        if base_model_state is not None:
            self.load_state_dict(base_model_state, strict=False)

    def _run_blocks(self, x, sin, cos, attn_mask=None):
        """Transformer stack + reasoning cross-attention."""
        x = super()._run_blocks(x, sin, cos, attn_mask)
        B = x.size(0)
        memory = self.reason_tokens.expand(B, -1, -1)
        for rblock in self.reason_blocks:
            x = rblock(x, memory)
        return x
