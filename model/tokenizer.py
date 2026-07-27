import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TokenizerConfig


class BSQQuantizer(nn.Module):
    """Binary Spherical Quantization — implicit-codebook quantizer.

    Projects a latent vector onto the unit sphere, then through learnable
    hyperplanes to produce a k-bit binary code b in {-1, 1}^k. The
    vocabulary of 2^k codes is implicit — every code is reachable.
    """

    def __init__(self, embedding_dim, bits, commitment_cost, entropy_weight):
        super().__init__()
        self.bits = int(bits)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.entropy_weight = float(entropy_weight)
        self.project = nn.Linear(self.embedding_dim, self.bits, bias=False)
        self.decode_proj = nn.Linear(self.bits, self.embedding_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.project.weight)
        nn.init.orthogonal_(self.decode_proj.weight)

    @staticmethod
    def _bits_to_int(bits_01):
        # bits_01: [*, k] in {0, 1}
        k = bits_01.shape[-1]
        powers = torch.arange(k - 1, -1, -1, device=bits_01.device, dtype=bits_01.dtype)
        return (bits_01 * (2 ** powers)).sum(dim=-1)

    @staticmethod
    def _int_to_bits(indices, bits):
        mask = torch.arange(bits - 1, -1, -1, device=indices.device).unsqueeze(0)
        return ((indices.unsqueeze(-1) >> mask) & 1).to(torch.float32) * 2.0 - 1.0

    def forward(self, z):
        z_norm = F.normalize(z, dim=-1)
        logits = self.project(z_norm)

        b_soft = torch.tanh(logits)                         # smooth approx
        b_hard = torch.sign(logits)                         # {-1, +1}
        b = b_soft + (b_hard - b_soft).detach()             # STE (reordered: 1 sub vs 2)

        bits_01 = ((b + 1) * 0.5).long().clamp_(0, 1)      # in-place clamp
        indices = self._bits_to_int(bits_01)

        # Entropy regularization (only regularizer with gradient to project)
        ent_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        if self.training:
            prob = (b_soft * 0.5 + 0.5).mean(0).clamp(1e-10, 1 - 1e-10)  # reuse tanh→sigmoid
            ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log()).mean()
            ent_loss = -ent * self.entropy_weight

        return b, bits_01, indices, ent_loss

    def training_bits(self, z):
        """Return reconstruction bits and entropy loss without ID round-trips.

        ``HierarchicalQuantizer.forward`` only needs the hard {-1, +1} bits,
        while the public ``forward`` method also exposes token IDs.  Converting
        bits to integer IDs and immediately converting them back costs several
        small CUDA kernels.  The ``logits > 0`` rule exactly matches the old
        sign -> long -> bit-unpack path, including mapping an exact zero to -1.
        """
        z_norm = F.normalize(z, dim=-1)
        logits = self.project(z_norm)
        hard_bits = (
            (logits > 0)
            .to(dtype=logits.dtype)
            .mul_(2.0)
            .sub_(1.0)
        )

        ent_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        if self.training:
            b_soft = torch.tanh(logits)
            prob = (b_soft * 0.5 + 0.5).mean(0).clamp(1e-10, 1 - 1e-10)
            ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log()).mean()
            ent_loss = -ent * self.entropy_weight
        return hard_bits, ent_loss

    def quantize(self, z):
        z_norm = F.normalize(z, dim=-1)
        logits = self.project(z_norm)
        b_hard = torch.sign(logits)
        bits_01 = ((b_hard + 1) / 2).long().clamp(0, 1)
        indices = self._bits_to_int(bits_01)
        return b_hard, bits_01, indices

    def decode_ids(self, indices):
        b = self._int_to_bits(indices, self.bits).to(self.decode_proj.weight.dtype)
        return self.decode_proj(b)

    @property
    def vocab_size(self):
        return 2 ** self.bits


class HierarchicalQuantizer(nn.Module):
    """BSQ-based hierarchical tokenizer for financial K-line sequences.

    2-level coarse->fine quantization:
      - encoder: MLP (input_dim -> hidden_dim -> embedding_dim)
      - BSQ coarse: k1 bits -> captures principal structure
      - BSQ fine:   k2 bits -> encodes residual detail
      - decoder: MLP (embedding_dim -> hidden_dim -> input_dim)

    Loss = L_coarse(recon from coarse-only) + L_fine(recon from full) + quant_loss
    """
    input_dim = TokenizerConfig.input_dim
    hidden_dim = TokenizerConfig.hidden_dim
    embedding_dim = TokenizerConfig.embedding_dim
    num_quantizers = TokenizerConfig.num_quantizers

    def __init__(self, input_dim=None, hidden_dim=None, embedding_dim=None,
                 num_quantizers=None, bits_per_quantizer=10,
                 commitment_cost=0.05, entropy_weight=0.05):
        super().__init__()
        self.num_quantizers = max(1, int(num_quantizers or self.num_quantizers))
        self._embedding_dim = int(embedding_dim or self.embedding_dim)
        _commit = getattr(TokenizerConfig, "bsq_commitment_cost", commitment_cost)
        _ent = getattr(TokenizerConfig, "bsq_entropy_weight", entropy_weight)

        # bits_per_quantizer can be int (all layers same) or list (per-layer)
        if isinstance(bits_per_quantizer, (list, tuple)):
            _bits_list = [int(b) for b in bits_per_quantizer]
        else:
            _bits_list = [int(bits_per_quantizer)] * self.num_quantizers
        assert len(_bits_list) == self.num_quantizers, \
            f"bits_per_quantizer length {len(_bits_list)} != num_quantizers {self.num_quantizers}"

        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim or self.input_dim), int(hidden_dim or self.hidden_dim)),
            nn.LayerNorm(int(hidden_dim or self.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim or self.hidden_dim), self._embedding_dim),
            nn.LayerNorm(self._embedding_dim),
        )
        self.bsq_quantizers = nn.ModuleList([
            BSQQuantizer(self._embedding_dim, _bits_list[i], _commit, _ent)
            for i in range(self.num_quantizers)
        ])
        self.bsq_coarse = self.bsq_quantizers[0]
        self.bsq_fine = self.bsq_quantizers[1] if self.num_quantizers > 1 else self.bsq_quantizers[0]
        self.decoder = nn.Sequential(
            nn.Linear(self._embedding_dim, int(hidden_dim or self.hidden_dim)),
            nn.LayerNorm(int(hidden_dim or self.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim or self.hidden_dim), int(input_dim or self.input_dim)),
        )

    def forward(self, x, return_all=False):
        z = self.encoder(x)
        residual = z
        z_q = torch.zeros_like(z)
        total_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        for bsq in self.bsq_quantizers:
            hard_bits, q_loss = bsq.training_bits(residual)
            z_q_i = bsq.decode_proj(hard_bits)
            residual = residual - z_q_i.detach()
            z_q.add_(z_q_i)
            total_loss.add_(q_loss)

        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)

        return recon_loss + total_loss

    def encode_all(self, x):
        z = self.encoder(x)
        residual = z
        indices = []
        for bsq in self.bsq_quantizers:
            _, _, idx = bsq.quantize(residual)
            z_q = bsq.decode_proj(bsq._int_to_bits(idx, bsq.bits).to(bsq.decode_proj.weight.dtype))
            residual = residual - z_q.detach()
            indices.append(idx)
        return torch.stack(indices, dim=-1)  # [B, N, num_quantizers]

    def encode(self, x):
        """Return coarse token IDs for GPT/BERT input.

        The fine layer only improves tokenizer reconstruction — GPT learns
        to predict fine tokens separately via a dual-head architecture.
        """
        all_indices = self.encode_all(x)
        idx_coarse = all_indices[..., 0]
        return idx_coarse, None

    @property
    def vocab_size(self):
        """Joint vocabulary size (for tokenizer internal use)."""
        v = 1
        for bsq in self.bsq_quantizers:
            v *= bsq.vocab_size
        return v

    @property
    def vocab_coarse(self):
        """Coarse vocabulary size — the GPT/BERT prediction target."""
        return self.bsq_coarse.vocab_size

    @property
    def bits_l1(self):
        return self.bsq_coarse.bits

    @property
    def bits_l2(self):
        return self.bsq_fine.bits

    def decode_all(self, all_indices):
        if isinstance(all_indices, torch.Tensor):
            if all_indices.dim() == 3 and all_indices.size(-1) == self.num_quantizers:
                pass
            else:
                raise ValueError(f"Expected indices tensor of shape [B, N, {self.num_quantizers}], got {all_indices.shape}")
        else:
            raise ValueError(f"Expected {self.num_quantizers} levels, got {tuple(all_indices)}")

        B, N, _ = all_indices.shape
        z_q = torch.zeros(B, N, self._embedding_dim, device=all_indices.device)
        for i, bsq in enumerate(self.bsq_quantizers):
            indices = all_indices[:, :, i]
            z_q = z_q + bsq.decode_ids(indices)
        return self.decoder(z_q)

    def decode(self, idx_coarse, idx_fine):
        all_indices = torch.stack([idx_coarse, idx_fine], dim=-1)
        return self.decode_all(all_indices)


def build_tokenizer_kwargs(config_dict=None):
    """Build kwargs for HierarchicalQuantizer from an optional config dict.

    Supports per-layer bits via TokenizerConfig.bits_l1/bits_l2.
    """
    cfg = config_dict or {}
    # Per-layer bits: bits_l1/bits_l2 override bits_per_quantizer
    b1 = cfg.get("bits_l1", getattr(TokenizerConfig, "bits_l1", 0))
    b2 = cfg.get("bits_l2", getattr(TokenizerConfig, "bits_l2", 0))
    default_bits = cfg.get(
        "bits_per_quantizer", getattr(TokenizerConfig, "bits_per_quantizer", 10))
    if b1 > 0 and b2 > 0:
        bits = [int(b1), int(b2)]
    elif isinstance(default_bits, (list, tuple)):
        bits = [int(b) for b in default_bits]
    else:
        bits = int(default_bits)
    return {
        "input_dim": cfg.get("input_dim", TokenizerConfig.input_dim),
        "hidden_dim": cfg.get("hidden_dim", TokenizerConfig.hidden_dim),
        "embedding_dim": cfg.get("embedding_dim", TokenizerConfig.embedding_dim),
        "num_quantizers": cfg.get("num_quantizers", TokenizerConfig.num_quantizers),
        "bits_per_quantizer": bits,
        "commitment_cost": cfg.get(
            "bsq_commitment_cost", getattr(TokenizerConfig, "bsq_commitment_cost", 0.05)),
        "entropy_weight": cfg.get(
            "bsq_entropy_weight", getattr(TokenizerConfig, "bsq_entropy_weight", 0.05)),
    }


def export_tokenizer_config():
    """Export current TokenizerConfig as a dict (for checkpoint saving)."""
    b1 = getattr(TokenizerConfig, "bits_l1", 0)
    b2 = getattr(TokenizerConfig, "bits_l2", 0)
    default_bits = getattr(TokenizerConfig, "bits_per_quantizer", 10)
    bits = [int(b1), int(b2)] if b1 > 0 and b2 > 0 else int(default_bits)
    return {
        "input_dim": TokenizerConfig.input_dim,
        "hidden_dim": TokenizerConfig.hidden_dim,
        "embedding_dim": TokenizerConfig.embedding_dim,
        "num_quantizers": TokenizerConfig.num_quantizers,
        "bits_per_quantizer": bits,
        "bsq_commitment_cost": getattr(TokenizerConfig, "bsq_commitment_cost", 0.05),
        "bsq_entropy_weight": getattr(TokenizerConfig, "bsq_entropy_weight", 0.05),
    }
