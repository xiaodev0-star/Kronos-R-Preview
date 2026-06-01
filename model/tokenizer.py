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

        b_hard = torch.sign(logits)  # {-1, +1}
        b_soft = torch.tanh(logits)  # smooth approx

        # Straight-through estimator
        b = (b_hard - b_soft).detach() + b_soft

        bits_01 = ((b + 1) / 2).long().clamp(0, 1)  # {0, 1}
        indices = self._bits_to_int(bits_01)

        # Commitment loss
        commit_loss = F.mse_loss(b_soft.detach(), b_hard) * self.commitment_cost

        # Entropy regularization
        codebook_loss = torch.tensor(0.0, device=z.device)
        ent_loss = torch.tensor(0.0, device=z.device)
        if self.training:
            prob = torch.sigmoid(logits).mean(dim=0).clamp(1e-10, 1 - 1e-10)
            ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log()).mean()
            ent_loss = -ent * self.entropy_weight

        quant_loss = commit_loss + codebook_loss + ent_loss
        return b, bits_01, indices, quant_loss

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
        _bits = getattr(TokenizerConfig, "bits_per_quantizer", bits_per_quantizer)
        _commit = getattr(TokenizerConfig, "bsq_commitment_cost", commitment_cost)
        _ent = getattr(TokenizerConfig, "bsq_entropy_weight", entropy_weight)

        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim or self.input_dim), int(hidden_dim or self.hidden_dim)),
            nn.LayerNorm(int(hidden_dim or self.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim or self.hidden_dim), self._embedding_dim),
            nn.LayerNorm(self._embedding_dim),
        )
        self.bsq_quantizers = nn.ModuleList([
            BSQQuantizer(self._embedding_dim, _bits, _commit, _ent)
            for _ in range(self.num_quantizers)
        ])
        self.bsq_coarse = self.bsq_quantizers[0]
        self.bsq_fine = self.bsq_quantizers[1] if self.num_quantizers > 1 else self.bsq_quantizers[0]
        self.decoder = nn.Sequential(
            nn.Linear(self._embedding_dim, int(hidden_dim or self.hidden_dim)),
            nn.LayerNorm(int(hidden_dim or self.hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim or self.hidden_dim), int(input_dim or self.input_dim)),
        )

    def _bsq_quantize_latent(self, z):
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device)
        indices = []
        z_q_coarse = None
        for i, bsq in enumerate(self.bsq_quantizers):
            b, idx, q_loss = bsq.quantize(residual)[:2] + (bsq.quantize(residual)[2],)
            b, _, idx = bsq.quantize(residual)
            q_loss = 0  # just quantize, no loss in inference
            z_q = bsq.decode_proj(bsq._int_to_bits(idx, bsq.bits).to(bsq.decode_proj.weight.dtype))
            if i == 0:
                z_q_coarse = z_q
            residual = residual - z_q.detach()
            z_q_total = z_q_total + z_q
            indices.append(idx)
        return z_q_total, indices, z_q_coarse

    def forward(self, x, return_all=False):
        z = self.encoder(x)
        # Quantize
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device)
        indices = []
        z_q_coarse = None
        for i, bsq in enumerate(self.bsq_quantizers):
            b, bits_01, idx, q_loss = bsq(residual)
            z_q = bsq.decode_proj(bsq._int_to_bits(idx, bsq.bits).to(bsq.decode_proj.weight.dtype))
            if i == 0:
                z_q_coarse = z_q
            residual = residual - z_q.detach()
            z_q_total = z_q_total + z_q
            indices.append(idx)
            total_loss = total_loss + q_loss

        x_recon_full = self.decoder(z_q_total)
        recon_loss = F.mse_loss(x_recon_full, x)

        if self.num_quantizers > 1 and z_q_coarse is not None:
            x_recon_coarse = self.decoder(z_q_coarse)
            recon_loss = recon_loss + F.mse_loss(x_recon_coarse, x)

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
        all_indices = self.encode_all(x)
        idx_coarse = all_indices[..., 0]
        idx_fine = all_indices[..., 1] if self.num_quantizers > 1 else all_indices[..., 0]
        return idx_coarse, idx_fine

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

    def codebook_stats(self):
        stats = {}
        for i, bsq in enumerate(self.bsq_quantizers):
            stats[f"BSQ level_{i}"] = bsq.bits
        if len(self.bsq_quantizers) >= 1:
            stats["level_0"] = stats.get("BSQ level_0", "coarse")
        if len(self.bsq_quantizers) >= 2:
            stats["level_1"] = stats.get("BSQ level_1", "fine")
        return stats
