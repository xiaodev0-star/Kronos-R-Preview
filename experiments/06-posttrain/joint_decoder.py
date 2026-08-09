"""PT-01: exact coarse/fine joint posterior decoder.

The joint posterior factorizes as

    p(c, f | h) = p(c | h) * p(f | h, c),

where EVERY candidate coarse code ``c`` forms its own fine-head conditioning
``head_fine([h, fineEmb(c)])``.  The legacy greedy path (``forward_selected``)
only returns ``p(f | h, argmax_c)`` and must not feed expectation / sampling /
reranking consumers.

Formal results are exact over the full ``V_c x V_f`` support (read from the
reviewed tokenizer at runtime; 7+7 bits => 128 x 128 = 16,384 here, not
hard-coded).  All decoded-normalized values are recovered to raw price space
with each sample's own ``p_mean[0]`` / ``p_std[0]``; ``p_std[0]`` must be finite
and positive, otherwise the sample is a data-quality failure.

Quantile definition (PostTrain-ToDo §4.3): sort by decoded raw return, merge
duplicate-return probability mass, take the left quantile where the CDF first
reaches q.  Because ``p_std[0] > 0`` the raw-space ordering is a positive affine
transform of the normalized ordering, so one shared sort permutation over the
decode table suffices for every stock.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12


class DecodeTable:
    """Precomputed ``[V_c, V_f, feature_dim]`` tokenizer decode table.

    ``decode_all`` is a batched decoder, not a lookup table, so we enumerate the
    joint support once at construction.
    """

    def __init__(self, tokenizer, device="cpu"):
        self.v_c = int(tokenizer.vocab_coarse)
        self.v_f = int(tokenizer.bsq_fine.vocab_size)
        if self.v_c * self.v_f > 2_000_000:
            raise ValueError(f"joint support too large: {self.v_c}x{self.v_f}")
        self.device = torch.device(device)

        cc = torch.arange(self.v_c, dtype=torch.long).repeat_interleave(self.v_f)
        ff = torch.arange(self.v_f, dtype=torch.long).repeat(self.v_c)
        idx = torch.stack([cc, ff], dim=-1).unsqueeze(0)  # [1, V_c*V_f, 2]
        # Build on the tokenizer's own device, then move the fixed lookup.
        tok_device = next(tokenizer.parameters()).device
        with torch.no_grad():
            decoded = tokenizer.decode_all(idx.to(tok_device)).squeeze(0)
        decoded = decoded.to(self.device)
        # The decode table is a fixed lookup; never let the tokenizer's weights
        # drag it into autograd graphs (breaks .numpy() in downstream code).
        self.table = decoded.detach().view(self.v_c, self.v_f, decoded.shape[-1])
        # Feature 0 is normalized log_ret.
        self.norm_logret = self.table[..., 0].detach()  # [V_c, V_f]

        # Shared sort permutation over flattened [V_c*V_f] normalized log-returns.
        self.flat_norm = self.norm_logret.reshape(-1)
        self.order = torch.argsort(self.flat_norm, stable=True)

    def raw_logret(self, p_mean0, p_std0):
        """Raw-space return table ``[..., V_c, V_f]`` for per-sample scalars.

        ``p_mean0`` / ``p_std0`` broadcast from the right-most dims.
        """
        return self.norm_logret * p_std0 + p_mean0


class JointStats:
    """Mutable container for per-row joint posterior statistics (tensors [K, ...])."""

    def __init__(self, k, device):
        self.device = torch.device(device)
        self.k = k
        # MAP (J1)
        self.map_c = None
        self.map_f = None
        self.map_return = None
        self.map_joint_prob = None
        # Posterior moments
        self.mean = None
        self.median = None
        self.std = None
        self.q10 = None
        self.q90 = None
        self.p_up = None
        # Entropies
        self.coarse_entropy = None
        self.cond_fine_entropy = None
        self.joint_entropy = None
        # Greedy baseline (J0)
        self.greedy_c = None
        self.greedy_f = None
        self.greedy_return = None
        # NLLs (need true coarse/fine ids)
        self.full_joint_nll = None
        self.ordinary_joint_nll = None
        # Marginal diagnostic
        self.special_mass = None

    def as_dict(self):
        out = {}
        for k, v in vars(self).items():
            if k in ("device", "k"):
                continue
            if v is not None:
                out[k] = v
        return out


@torch.no_grad()
def decode_joint(
    model,
    decode_table,
    hidden,
    p_mean0,
    p_std0,
    true_coarse_ids=None,
    true_fine_ids=None,
    true_logret=None,
    t_c=1.0,
    t_f=1.0,
    chunk=512,
):
    """Compute exact joint posterior statistics for ``hidden [K, dim]``.

    Args:
        model: KronosPreview exposing ``coarse_logits_from_hidden`` and
            ``fine_logits_for_coarse``.
        decode_table: precomputed DecodeTable.
        hidden: [K, dim] final-norm hidden states.
        p_mean0 / p_std0: [K] per-sample raw-space normalization scalars for
            feature 0 (log_ret).  ``p_std0`` must be finite and > 0.
        true_coarse_ids / true_fine_ids: [K] optional true joint token for NLLs.
        t_c / t_f: coarse / conditional-fine softmax temperatures.
        chunk: number of rows to process per fine-logits expansion pass.

    Returns:
        (JointStats, quality_flags).  ``quality_flags`` is [K] bool: True where
        p_std0 is finite and positive (valid for raw-space recovery).
    """
    device = hidden.device
    k = hidden.shape[0]
    stats = JointStats(k, device)

    p_mean0 = torch.as_tensor(p_mean0, dtype=torch.float32, device=device).reshape(-1)
    p_std0 = torch.as_tensor(p_std0, dtype=torch.float32, device=device).reshape(-1)
    if p_mean0.numel() == 1:
        p_mean0 = p_mean0.expand(k)
    if p_std0.numel() == 1:
        p_std0 = p_std0.expand(k)
    quality = torch.isfinite(p_mean0) & torch.isfinite(p_std0) & (p_std0 > 0)

    v_c = decode_table.v_c
    v_f = decode_table.v_f
    dim = hidden.shape[-1]

    # ---- coarse posterior -------------------------------------------------
    coarse_logits = model.coarse_logits_from_hidden(hidden)  # [K, V_c+2]
    p_full = F.softmax(coarse_logits / t_c, dim=-1)          # [K, V_c+2]
    special_mass = p_full[:, v_c:].sum(dim=-1)               # [K]
    # q(c) = P(c | ordinary) = p_full(c) / (1 - special_mass).  When the model
    # is (near-)certain about BOS/EOS (special_mass ~ 1), the float division
    # underflows, so q is renormalized to a proper distribution over ordinary
    # codes.  full_joint_nll still uses p_full (spec §4.1); ordinary_nll uses q.
    q = p_full[:, :v_c] / (1.0 - special_mass).clamp_min(EPS).unsqueeze(-1)  # [K, V_c]
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)

    # ---- fine posterior for every candidate coarse ------------------------
    p_f_given_c = torch.empty(k, v_c, v_f, dtype=hidden.dtype, device=device)
    for start in range(0, k, chunk):
        stop = min(start + chunk, k)
        hid_exp = hidden[start:stop].unsqueeze(1).expand(stop - start, v_c, dim)
        hid_exp = hid_exp.reshape(-1, dim)
        coarse_ids = torch.arange(v_c, device=device).repeat(stop - start)
        fl = model.fine_logits_for_coarse(hid_exp, coarse_ids)
        p_f_given_c[start:stop] = F.softmax(
            fl / t_f, dim=-1).view(stop - start, v_c, v_f)

    joint = q.unsqueeze(-1) * p_f_given_c                    # [K, V_c, V_f]

    # ---- raw-space returns ------------------------------------------------
    flat_norm = decode_table.flat_norm.to(device)            # [M]
    order = decode_table.order.to(device)                    # [M]
    M = v_c * v_f
    raw_flat = flat_norm * p_std0.unsqueeze(-1) + p_mean0.unsqueeze(-1)  # [K, M]

    # ---- statistics that don't need the flattened sort ---------------------
    stats.special_mass = special_mass
    stats.coarse_entropy = -(q * (q + EPS).log()).sum(-1)
    cond_fine = -(joint * (p_f_given_c + EPS).log()).sum(dim=(-1, -2))
    stats.cond_fine_entropy = cond_fine
    stats.joint_entropy = -(joint * (joint + EPS).log()).sum(dim=(-1, -2))

    # joint MAP (J1): argmax over full (c, f) grid of q(c)*p(f|c)
    joint_flat = joint.reshape(k, M)
    map_idx = joint_flat.argmax(-1)
    stats.map_c = map_idx // v_f
    stats.map_f = map_idx % v_f
    stats.map_return = raw_flat[torch.arange(k, device=device), map_idx]
    stats.map_joint_prob = joint_flat[torch.arange(k, device=device), map_idx]

    # posterior mean / variance (J2)
    stats.mean = (joint_flat * raw_flat).sum(-1)
    mean_sq = (joint_flat * raw_flat.square()).sum(-1)
    var = (mean_sq - stats.mean.square()).clamp_min(0.0)
    stats.std = var.sqrt()

    # P(raw_logret > 0) in RAW space (J4).  With p_std>0 the threshold on the
    # normalized table is norm > -p_mean/p_std.
    with torch.no_grad():
        thr = (-p_mean0 / p_std0.clamp_min(EPS))               # [K]
        norm_sorted = flat_norm[order]                          # [M] shared
        probs_sorted = joint_flat[:, order]                     # [K, M]
        idx_up = torch.searchsorted(
            norm_sorted.unsqueeze(0).expand(k, M).contiguous(),
            thr.unsqueeze(-1), right=True,
        )[:, 0]                                                # [K]
        up_mask = torch.arange(M, device=device).unsqueeze(0) >= idx_up.unsqueeze(-1)
        stats.p_up = (probs_sorted * up_mask).sum(-1)

        # median / quantiles: left quantile where CDF first reaches q.
        # torch.searchsorted needs values shaped [B, ...] when boundaries are 2D.
        cdf = torch.cumsum(probs_sorted, dim=-1)               # [K, M]
        rows_idx = torch.arange(k, device=device)
        for qval, attr in ((0.10, "q10"), (0.50, "median"), (0.90, "q90")):
            qi = torch.searchsorted(
                cdf, torch.full((k, 1), qval, device=device), right=False
            )[:, 0].clamp_max(M - 1)
            stats.__dict__[attr] = raw_flat[rows_idx, order[qi]]

        # CRPS of the discrete posterior vs the true raw return (if provided).
        # CRPS = E|X-y| - 0.5 E|X-X'|; the E|X-X'| term uses the O(M) sorted form
        # term_i = x_i*(2*F_i - P_total) - (2*PX_i - PX_total).
        if true_logret is not None:
            y = torch.as_tensor(true_logret, dtype=torch.float32,
                                device=device).reshape(-1)
            raw_sorted = raw_flat[rows_idx[:, None], order[None, :]]  # [K, M] ascending
            P = torch.cumsum(probs_sorted, dim=-1)
            PX = torch.cumsum(probs_sorted * raw_sorted, dim=-1)
            total_px = PX[:, -1:]
            term = raw_sorted * (2.0 * P - 1.0) - (2.0 * PX - total_px)
            e_abs = (probs_sorted * (raw_sorted - y.unsqueeze(-1)).abs()).sum(-1)
            e_pair = (probs_sorted * term).sum(-1)
            stats.crps = e_abs - 0.5 * e_pair

    # greedy baseline (J0): argmax coarse over ordinary codes, then conditional
    # fine argmax, matching the legacy forward_selected decode path.
    stats.greedy_c = p_full[:, :v_c].argmax(-1)
    g_idx = torch.arange(k, device=device), stats.greedy_c
    stats.greedy_f = p_f_given_c[g_idx].argmax(-1)
    stats.greedy_return = raw_flat[
        torch.arange(k, device=device), stats.greedy_c * v_f + stats.greedy_f]

    # ---- NLLs (require true ids) -------------------------------------------
    if true_coarse_ids is not None and true_fine_ids is not None:
        tc = torch.as_tensor(true_coarse_ids, dtype=torch.long, device=device).reshape(-1)
        tf = torch.as_tensor(true_fine_ids, dtype=torch.long, device=device).reshape(-1)
        valid = (tc >= 0) & (tc < v_c) & (tf >= 0) & (tf < v_f)
        full_c_prob = p_full[torch.arange(k, device=device), tc]
        fine_prob = p_f_given_c[torch.arange(k, device=device), tc, tf]
        q_c_prob = q[torch.arange(k, device=device), tc]
        full_joint_nll = torch.full((k,), float("nan"), device=device)
        ordinary_nll = torch.full((k,), float("nan"), device=device)
        full_joint_nll[valid] = -torch.log((full_c_prob * fine_prob).clamp_min(EPS))[valid]
        ordinary_nll[valid] = -torch.log((q_c_prob * fine_prob).clamp_min(EPS))[valid]
        stats.full_joint_nll = full_joint_nll
        stats.ordinary_joint_nll = ordinary_nll

    return stats, quality


def greedy_logits_from_hidden(model, hidden):
    """Legacy greedy decode logits for parity checks (returns coarse, fine).

    Mirrors ``forward_selected``: fine conditioned on argmax coarse over ordinary
    codes.  Only for the J0 compatibility baseline / guardrail checks.
    """
    coarse_logits = model.coarse_logits_from_hidden(hidden)
    coarse_pred = coarse_logits[:, : model._vocab_l1].argmax(dim=-1)
    fine_logits = model.fine_logits_for_coarse(hidden, coarse_pred)
    return coarse_logits, fine_logits
