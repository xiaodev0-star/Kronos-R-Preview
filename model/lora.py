"""LoRA-per-regime adapters for Branch F (model side).

Each injected ``nn.Linear`` (attn q/k/v/out, ffn gate/up/down) gets a
``LoRAAdapter`` holding ``n_regimes`` independent low-rank pairs ``(A_r, B_r)``.
A ~ N(0, 0.02), B = 0, so at init the adapter outputs exactly zero and the
attached model is bit-identical to the dense CPT baseline.  Per token, the active
regime's pair is selected by hard routing from ``regime_ids``; regime -1 fires no
adapter (pure base projection).

Branch F freezes the dense backbone and trains only the ``A/B`` matrices, so the
regime-conditioning effect is isolated from head / backbone retuning.
"""
import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """One per-regime low-rank (A, B) adapter for a single injected Linear.

    ``forward(x, regime_ids)`` returns the low-rank correction to add on top of
    the base projection.  The correction is ``0`` when ``regime_ids`` is None or
    a token's regime is -1 (never fires), which keeps the dense model unchanged.
    """

    def __init__(self, in_features, out_features, r, alpha, n_regimes=3):
        super().__init__()
        if r <= 0 or alpha <= 0:
            raise ValueError("r and alpha must be > 0")
        if n_regimes <= 0:
            raise ValueError("n_regimes must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.n_regimes = n_regimes
        self.scale = alpha / r
        self.loras = nn.ModuleList()
        for _ in range(n_regimes):
            a = nn.Linear(in_features, r, bias=False)
            b = nn.Linear(r, out_features, bias=False)
            nn.init.normal_(a.weight, std=0.02)
            nn.init.zeros_(b.weight)   # zero output => dense bit-identical at init
            self.loras.append(nn.Sequential(a, b))

    def forward(self, x, regime_ids):
        """Low-rank correction, shape ``[B, N, out_features]``.

        Args:
            x: ``[B, N, in_features]`` activation (pre-projection).
            regime_ids: ``[B, N]`` int64 per-token regime in
                ``{0..n_regimes-1}``; ``-1`` fires no adapter.  None -> zero
                correction (dense path).
        """
        B, N, _ = x.shape
        if regime_ids is None:
            return x.new_zeros(B, N, self.out_features)
        if regime_ids.dim() == 1:
            regime_ids = regime_ids.unsqueeze(0)
        out = x.new_zeros(B, N, self.out_features)
        for r in range(self.n_regimes):
            # Hard per-token routing mask.  Cast to x.dtype BEFORE the masked add
            # so bf16/float mismatch under autocast is avoided.
            m = (regime_ids == r).unsqueeze(-1).to(x.dtype)  # [B, N, 1]
            if m.any():
                out = out + m * self.loras[r](x)
        return out * self.scale


class LoRAModule(nn.Module):
    """Container managing the LoRA adapter set attached to a model.

    ``attach_lora`` registers each adapter both on its host Linear (``linear.lora``
    — which the layer forward paths check) and here.  To keep the adapter params
    in the state_dict exactly once (they are already owned by the host Linears),
    this container keeps *plain references* to the adapters rather than
    re-registering them as submodules.
    """

    def __init__(self):
        super().__init__()
        self._adapters = []   # plain list of LoRAAdapter (owned by host Linears)

    def add(self, adapter: LoRAAdapter) -> LoRAAdapter:
        self._adapters.append(adapter)
        return adapter

    def __len__(self) -> int:
        return len(self._adapters)

    def lora_parameters(self):
        """Iterate over all attached LoRA parameters (A and B matrices)."""
        for ad in self._adapters:
            yield from ad.parameters()
