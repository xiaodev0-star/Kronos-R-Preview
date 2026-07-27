"""Shared, numerically conservative training-runtime helpers.

The project runs primarily on Windows, where ``torch.compile``/Triton is not
available.  These helpers use native PyTorch CUDA paths only:

* foreach gradient clipping (same reduction/update semantics);
* a manually captured fixed-shape forward/backward/clip step;
* capture warm-up that restores parameters before the first counted update.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
):
    """Use the native CUDA foreach path without changing clipping semantics."""
    params = list(parameters)
    if not params:
        return torch.tensor(0.0)
    use_foreach = params[0].device.type == "cuda"
    return torch.nn.utils.clip_grad_norm_(
        params,
        max_norm,
        norm_type=norm_type,
        foreach=True if use_foreach else None,
    )


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    lr = optimizer.param_groups[0]["lr"]
    return float(lr.detach().cpu()) if isinstance(lr, torch.Tensor) else float(lr)


@dataclass
class FixedBatchCudaGraphStep:
    """Captured forward/backward/clip with the historical optimizer outside."""

    graph: torch.cuda.CUDAGraph
    loss: torch.Tensor
    optimizer: torch.optim.Optimizer

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        self.optimizer.step()
        return self.loss

    @classmethod
    def capture(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        loss_closure: Callable[[], torch.Tensor],
        grad_clip: float,
        loss_accumulator: torch.Tensor | None = None,
        warmup_steps: int = 3,
    ) -> "FixedBatchCudaGraphStep":
        params = list(parameters)
        if not params or params[0].device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA parameters")
        parameter_snapshot = [
            parameter.detach().clone() for parameter in params
        ]
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(params[0].device)

        # The graph must replay actual zero kernels. Allocate gradient buffers
        # before warmup; host-only ``grad = None`` assignments are not captured.
        for parameter in params:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)

        def full_step():
            # set_to_none=False records real zeroing kernels in the graph.  A
            # host-only ``grad = None`` assignment would not replay.
            optimizer.zero_grad(set_to_none=False)
            loss = loss_closure()
            loss.backward()
            clip_grad_norm_(params, grad_clip)
            if loss_accumulator is not None:
                loss_accumulator.add_(loss.detach())
            return loss

        warmup_stream = torch.cuda.Stream(device=params[0].device)
        warmup_stream.wait_stream(torch.cuda.current_stream(params[0].device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(1, int(warmup_steps))):
                full_step()
        torch.cuda.current_stream(params[0].device).wait_stream(warmup_stream)
        torch.cuda.synchronize(params[0].device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_loss = full_step()
        torch.cuda.synchronize(params[0].device)

        # Capture executes the graph once. Restore the exact pre-capture state
        # without replacing any tensor referenced by the captured graph.
        with torch.no_grad():
            for parameter, saved in zip(params, parameter_snapshot):
                parameter.copy_(saved)
            if loss_accumulator is not None:
                loss_accumulator.zero_()
        optimizer.zero_grad(set_to_none=False)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, params[0].device)
        torch.cuda.synchronize(params[0].device)
        return cls(
            graph=graph,
            loss=static_loss,
            optimizer=optimizer,
        )
