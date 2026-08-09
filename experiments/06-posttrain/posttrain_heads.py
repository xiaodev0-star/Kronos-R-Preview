"""PT-03/PT-04/PT-05 head architectures and losses for frozen-backbone probes.

All heads consume the frozen hidden state ``h`` ([B, dim]) and are trained as
independent readouts.  Loss reductions must be invariant to microbatch packing.

Rank losses operate on a single date's full legal cross-section (one batch =
one date, no duplicate stock_uid, enforced by the loader).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Losses
# ============================================================================

def soft_rank(scores, tau=1.0):
    """Differentiable soft rank (sigmoid trick): r_i = sum_j sigmoid((s_i - s_j)/tau).

    Monotone in ``scores``, matches order exactly as tau -> 0.  O(n^2).
    """
    d = scores.unsqueeze(-1) - scores.unsqueeze(-2)   # [B, B]
    return torch.sigmoid(d / tau).sum(-1)


def soft_spearman_loss(scores, true_ranks, tau=1.0):
    """1 - Pearson(sigmoid soft-rank, true rank percentile), per date batch."""
    r = soft_rank(scores, tau).float()
    t = true_ranks.float()
    r = r - r.mean()
    t = t - t.mean()
    denom = (r * r).sum().clamp_min(1e-9) * (t * t).sum().clamp_min(1e-9)
    corr = (r * t).sum() / denom.sqrt()
    return (1.0 - corr).clamp_min(0.0)


def pairwise_logistic_loss(scores, true_logret, tau=1.0, dead_zone=None,
                           max_pairs=100_000, seed=42):
    """Pairwise logistic rank loss within one date batch, with near-tie dead zone.

    Pairs are sampled within the batch (one date).  ``dead_zone`` (absolute raw
    log-return gap) pre-registered from the pre-cutoff train quantile; pairs
    with |y_i - y_j| < dead_zone are skipped.
    """
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    if dead_zone is None:
        dead_zone = 0.0
    rng = torch.Generator(device=scores.device).manual_seed(seed)
    idx_i = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    idx_j = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    # exclude self-pairs and near-ties
    keep = (idx_i != idx_j) & ((true_logret[idx_i] - true_logret[idx_j]).abs() >= dead_zone)
    if keep.sum() < 1:
        return torch.tensor(0.0, device=scores.device)
    i, j = idx_i[keep], idx_j[keep]
    target = (true_logret[i] > true_logret[j]).float()
    logit = (scores[i] - scores[j]) / tau
    loss = F.binary_cross_entropy_with_logits(logit, target)
    return loss


def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta)


# ============================================================================
# Return / direction heads (PT-03)
# ============================================================================

class LinearReturnHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = "huber"

    def forward(self, h):
        return self.fc(h).squeeze(-1)          # [B]


class MlpReturnHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.loss = "huber"

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LinearDirectionHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h).squeeze(-1))   # direction_prob in [0,1]


class MlpDirectionHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        return torch.sigmoid(self.net(h).squeeze(-1))


# ============================================================================
# Rank heads (PT-03 / PT-04)
# ============================================================================

class LinearRankHead(nn.Module):
    """Rank score head; loss is pairwise logistic (fallback) or soft-Spearman."""
    def __init__(self, dim=256, loss="soft_spearman"):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = loss

    def forward(self, h):
        return self.fc(h).squeeze(-1)          # rank_score (any monotone scale)


class MlpRankHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0, loss="soft_spearman"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.loss = loss

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ============================================================================
# Cross-sectional heads (PT-04)
# ============================================================================

class IndependentMLP(nn.Module):
    """score_i = g(h_i): per-stock capacity control (no cross-stock context)."""
    def __init__(self, dim=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class DeepSetsHead(nn.Module):
    """score_i = psi([h_i, mean_pool_phi(h)]): permutation-equivariant context.

    ``phi`` pools the same-date hidden states; ``psi`` conditions each stock on
    the pooled context.  Invariant to input permutation (see test_10).
    """
    def __init__(self, dim=256, latent=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, latent), nn.SiLU())
        self.psi = nn.Sequential(
            nn.Linear(dim + latent, latent), nn.SiLU(), nn.Linear(latent, 1))

    def forward(self, h):
        pooled = self.phi(h).mean(dim=0, keepdim=True)   # [1, latent]
        ctx = pooled.expand(h.shape[0], -1)
        return self.psi(torch.cat([h, ctx], dim=-1)).squeeze(-1)


class ISABSetTransformer(nn.Module):
    """Inducing-point Set Transformer (Lee et al., ICML 2019), pilot config.

    One ISAB block: multihead attention from stocks to inducing points and back,
    then a per-stock score head.  Uses genuine (inducing-point) cross-stock
    attention, NOT independent chunks.  permutation equivariant.
    """
    def __init__(self, dim=256, latent=64, n_inducing=32, heads=4):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, latent) * 0.02)
        self.to_latent = nn.Linear(dim, latent)
        self.attn = nn.MultiheadAttention(latent, heads, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, 1))

    def forward(self, h):
        # The batch IS one date's set: B stocks each contribute one [latent] token.
        x = self.to_latent(h)                      # [B, latent]
        x = x.unsqueeze(0)                         # [1, B, latent] (one set)
        # stocks attend to inducing points (pooling), then stocks attend back
        pooled, _ = self.attn(self.inducing, x, x)  # [1, n_inducing, latent]
        back, _ = self.attn(x, pooled, pooled)      # [1, B, latent]
        return self.score(back.squeeze(0)).squeeze(-1)


# ============================================================================
# Controls (PT-03 C0/C1/C2)
# ============================================================================

class C1FeatureHead(nn.Module):
    """Feature-only control head: last-return, 5/20-day momentum, 20-day vol/vol.

    Input features are point-in-time (visible at position p-1).  Same capacity
    as the hidden MLP probe (dim -> 64 -> 1).
    """
    def __init__(self, n_features=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class C2PosteriorHead(nn.Module):
    """Posterior-statistics control: mean/std/entropy/P(up) same-capacity head."""
    def __init__(self, n_stats=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, stats):
        return self.net(stats).squeeze(-1)
