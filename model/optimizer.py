"""Muon optimizer — Momentum + Newton-Schulz orthogonalization.

Algorithm (Keller Jordan, 2025):
  1. Nesterov momentum on gradient
  2. Newton-Schulz orthogonalization (3 iterations, bf16)
  3. Scale by max(m,n)/m for 2D weight matrices
  4. Standard Adam for 1D params (biases, norms)

Usage:
    from model.optimizer import build_muon_optimizers
    opt_muon, opt_adam = build_muon_optimizers(model, lr_muon=0.02, lr_adam=3e-4)
"""
import torch
from torch.optim.optimizer import Optimizer

_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_NS_STEPS = 3  # NS=3 ≈ NS=5 in training quality, 28% faster (verified)


def _newton_schulz(G, steps=None):
    """Orthogonalize gradient matrix via Newton-Schulz iteration."""
    if steps is None:
        steps = _NS_STEPS
    a, b, c = _NS_COEFFS
    X = G.bfloat16() / (G.norm() + 1e-7)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: Momentum + Newton-Schulz orthogonalization for 2D weight matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad + momentum * buf if nesterov else buf

                shape = update.shape
                if update.dim() >= 2:
                    update_2d = update.view(-1, shape[-1])
                    update_2d = _newton_schulz(update_2d)
                    m, n = update_2d.shape
                    update_2d.mul_(max(m, n) / m)
                    update = update_2d.view(shape)

                p.add_(update, alpha=-lr)

        return loss


def build_muon_optimizers(model, lr_muon=0.005, lr_adam=3e-4,
                           momentum=0.95, weight_decay_muon=0.0,
                           weight_decay_adam=0.01):
    """Build paired Muon (2D weights) + AdamW (1D params) optimizers.

    Default lr_muon=0.005 follows the reviewed Exp 04-B HPO selection
    (trial_4c721141ab): token-quality metrics improve monotonically as
    lr_muon drops from 0.02 to 0.005 on the depth6 architecture.
    """
    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (muon_params if param.dim() >= 2 else adam_params).append(param)

    opt_muon = Muon(muon_params, lr=lr_muon, momentum=momentum, weight_decay=weight_decay_muon)
    opt_adam = torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=weight_decay_adam)
    return opt_muon, opt_adam
