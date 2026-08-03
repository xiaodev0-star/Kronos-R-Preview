"""Portable experiment-output and provenance helpers.

Every formal study writes into two independent trees:

``weights``
    Large server-only artifacts: model/tokenizer checkpoints and reusable
    caches.  This tree is not required for offline result analysis.

``results``
    Downloadable artifacts: JSON/CSV/NPZ diagnostics, plots, manifests, and
    text logs.  Files in this tree never contain model state dictionaries.

The base directories can be moved outside the repository on an Ubuntu server
with ``KRONOS_WEIGHTS_ROOT`` and ``KRONOS_RESULTS_ROOT``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
HEAVY_SUFFIXES = {".ckpt", ".pkl", ".pt", ".pth"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_base(env_name: str, fallback: Path) -> Path:
    raw = os.environ.get(env_name, "").strip()
    return Path(raw).expanduser().resolve() if raw else fallback.resolve()


def default_study_roots(
    experiment_key: str,
    *,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Return ``(weights_root, results_root)`` for one formal study."""
    run_name = f"seed{seed}"
    weights_base = _resolved_base(
        "KRONOS_WEIGHTS_ROOT", ROOT / "server_runs" / "weights"
    )
    results_base = _resolved_base(
        "KRONOS_RESULTS_ROOT", ROOT / "server_runs" / "results"
    )
    return (
        weights_base / experiment_key / run_name,
        results_base / experiment_key / run_name,
    )


@dataclass(frozen=True)
class StudyLayout:
    """Resolved heavy/downloadable roots for one study."""

    weights_root: Path
    results_root: Path

    @classmethod
    def create(cls, weights_root: Path, results_root: Path) -> "StudyLayout":
        weights = weights_root.expanduser().resolve()
        results = results_root.expanduser().resolve()
        if weights == results:
            raise ValueError("weights_root and results_root must be different")
        if weights in results.parents or results in weights.parents:
            raise ValueError(
                "weights_root and results_root must be independent trees, "
                "not parent/child paths"
            )
        weights.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        return cls(weights_root=weights, results_root=results)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "weights_root": str(self.weights_root),
            "results_root": str(self.results_root),
            "download_only_results_root": True,
            "results_must_not_contain_model_weights": True,
            "weights_tree_role": "server-only checkpoints and reusable caches",
            "results_tree_role": (
                "downloadable JSON/CSV/NPZ diagnostics, plots, and logs"
            ),
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def numerical_preflight(*, require_cuda: bool = False) -> dict[str, Any]:
    """Verify the numerical invariants required by every formal experiment.

    This is production preflight code, not a dependency on a separate test
    tree.  ``runtime_environment`` runs it while creating every study
    manifest, and this module can also be executed directly before upload.
    """
    from types import SimpleNamespace

    import torch

    from model.kronos_preview import KronosPreview
    from model.layers import RotaryEmbedding, _apply_rope
    from train_base import compute_batched_loss

    checks: dict[str, Any] = {}

    rotary = RotaryEmbedding(head_dim=64)
    positions = torch.tensor(
        [[0, 1, 255, 256, 257, 1023, 2048, 3000, 3001, 4095, 8191]]
    )
    sin, cos = rotary(positions)
    reference_freqs = (
        positions.float().unsqueeze(-1)
        * rotary.inv_freq.float().view(1, 1, -1)
    )
    if not torch.equal(sin, torch.sin(reference_freqs)):
        raise RuntimeError("RoPE preflight failed: sin table is not FP32-exact")
    if not torch.equal(cos, torch.cos(reference_freqs)):
        raise RuntimeError("RoPE preflight failed: cos table is not FP32-exact")
    if torch.equal(sin[:, 7], sin[:, 8]) or torch.equal(
        cos[:, 7], cos[:, 8]
    ):
        raise RuntimeError(
            "RoPE preflight failed: positions 3000 and 3001 collide"
        )
    checks["rope_cpu_fp32_exact"] = True
    checks["rope_long_adjacent_positions_distinct"] = True

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for formal server experiments")
    if cuda_available:
        rotary_cuda = RotaryEmbedding(head_dim=64).cuda()
        positions_cuda = positions.cuda()
        expected_sin, expected_cos = rotary_cuda(positions_cuda)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            actual_sin, actual_cos = rotary_cuda(positions_cuda)
        if not torch.equal(actual_sin, expected_sin) or not torch.equal(
            actual_cos, expected_cos
        ):
            raise RuntimeError(
                "RoPE preflight failed under CUDA bf16 autocast"
            )
        if actual_sin.dtype != torch.float32 or actual_cos.dtype != torch.float32:
            raise RuntimeError("RoPE table was not retained in FP32")
        checks["rope_cuda_bf16_autocast_exact"] = True
    else:
        checks["rope_cuda_bf16_autocast_exact"] = None

    q = torch.randn(1, 2, 3, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 64, dtype=torch.bfloat16)
    rope_sin, rope_cos = rotary(torch.arange(3).unsqueeze(0))
    q_out, k_out = _apply_rope(q, k, rope_sin, rope_cos)
    if q_out.dtype != q.dtype or k_out.dtype != k.dtype:
        raise RuntimeError("RoPE application did not preserve Q/K dtype")
    checks["rope_application_preserves_qk_dtype"] = True

    tiny_config = SimpleNamespace(
        vocab_size=8,
        vocab_fine=4,
        dim=16,
        depth=0,
        heads=2,
        num_kv_heads=1,
        ffn_multiplier=2,
        dropout=0.0,
        va_hidden_dim=8,
        rope_base=10000.0,
    )
    model = KronosPreview(tiny_config).eval()
    input_ids = torch.tensor([[8, 1, 2, 9]])
    time_ids = torch.zeros(1, 4, 3, dtype=torch.long)
    position_ids = torch.arange(4).unsqueeze(0)
    fine_targets = torch.tensor([[0, 3, -100]])
    captured: dict[str, Any] = {}

    def capture_fine_input(_module: Any, args: tuple[Any, ...]) -> None:
        captured["value"] = args[0].detach()

    hook = model.head_fine[0].register_forward_pre_hook(capture_fine_input)
    try:
        model(
            input_ids,
            time_ids,
            position_ids,
            fine_targets=fine_targets,
        )
    finally:
        hook.remove()
    conditioned = captured["value"][..., model._fine_emb.embedding_dim :]
    expected_conditioning = model._fine_emb(input_ids[:, 1:])
    if not torch.equal(conditioned, expected_conditioning):
        raise RuntimeError(
            "Fine-head preflight failed: training is not conditioned on "
            "the current target coarse token"
        )
    checks["fine_head_current_coarse_alignment"] = True

    coarse_logits = torch.zeros(1, 3, 10, requires_grad=True)
    coarse_targets = torch.tensor([[1, 2]])
    fine_logits = torch.zeros(1, 2, 4, requires_grad=True)
    loss_args = SimpleNamespace(
        loss="ce",
        gamma=0.0,
        label_smoothing=0.0,
        entropy_alpha=0.0,
        fine_weight=1.0,
        heteroscedastic=False,
        het_weight=0.0,
    )
    loss, components, _ = compute_batched_loss(
        coarse_logits,
        coarse_targets,
        fine_logits,
        torch.tensor([[0, -100]]),
        None,
        None,
        loss_args,
    )
    loss.backward()
    if components["fine"].item() <= 0:
        raise RuntimeError("Fine-head preflight failed: fine code 0 was ignored")
    if fine_logits.grad is None or fine_logits.grad[0, 0, 0].item() == 0:
        raise RuntimeError(
            "Fine-head preflight failed: fine code 0 has no gradient"
        )
    if not torch.equal(
        fine_logits.grad[0, 1],
        torch.zeros_like(fine_logits.grad[0, 1]),
    ):
        raise RuntimeError(
            "Fine-head preflight failed: -100 padding produced a gradient"
        )
    checks["fine_code_zero_trained_and_minus100_ignored"] = True

    return {
        "schema": 1,
        "passed": True,
        "cuda_required": require_cuda,
        "cuda_available": cuda_available,
        "checks": checks,
    }


def runtime_environment(root: Path = ROOT) -> dict[str, Any]:
    """Capture reproducibility metadata without copying secret environment data."""
    memory_bytes = None
    try:
        import psutil  # type: ignore[import-not-found]

        memory_bytes = int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        pass

    packages: dict[str, str] = {}
    for name in (
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "tqdm",
    ):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = "unavailable"

    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "bf16_supported": (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except (ImportError, RuntimeError):
        torch_info = {"available": False}

    tracked_env = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "KRONOS_RESULTS_ROOT",
            "KRONOS_WEIGHTS_ROOT",
            "OMP_NUM_THREADS",
            "PYTORCH_CUDA_ALLOC_CONF",
        )
        if name in os.environ
    }
    git_commit = _command_output(["git", "rev-parse", "HEAD"], root)
    git_status = _command_output(["git", "status", "--short"], root)
    nvidia_smi = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "captured_at_utc": utc_now(),
        "command": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "packages": packages,
        "torch": torch_info,
        "numerical_preflight": numerical_preflight(),
        "nvidia_smi": nvidia_smi,
        "tracked_environment": tracked_env,
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "status_short": git_status,
        },
    }


def _inventory(
    root: Path,
    *,
    hash_files: bool,
    excluded_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    excluded = set(excluded_names)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or path.name.endswith(".tmp"):
            continue
        row: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        if hash_files:
            row["sha256"] = file_sha256(path)
        rows.append(row)
    return rows


def write_download_manifest(layout: StudyLayout) -> Path:
    """Inventory the result bundle and assert that no checkpoint leaked into it."""
    manifest_name = "download_manifest.json"
    result_files = _inventory(
        layout.results_root,
        hash_files=True,
        excluded_names=(manifest_name,),
    )
    leaked = [
        row["path"]
        for row in result_files
        if row["suffix"] in HEAVY_SUFFIXES
    ]
    if leaked:
        raise RuntimeError(
            "Large/checkpoint artifacts leaked into results_root: "
            + ", ".join(leaked)
        )
    heavy_files = _inventory(layout.weights_root, hash_files=False)
    payload = {
        "schema": 1,
        "created_at_utc": utc_now(),
        "layout": layout.metadata(),
        "download_bundle": {
            "root": str(layout.results_root),
            "file_count": len(result_files),
            "total_bytes": sum(row["size_bytes"] for row in result_files),
            "sha256_included": True,
            "files": result_files,
        },
        "server_only_bundle": {
            "root": str(layout.weights_root),
            "file_count": len(heavy_files),
            "total_bytes": sum(row["size_bytes"] for row in heavy_files),
            "sha256_included": False,
            "files": heavy_files,
        },
    }
    output = layout.results_root / manifest_name
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, output)
    return output


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Kronos numerical experiment preflight"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when CUDA is unavailable",
    )
    args = parser.parse_args()
    result = numerical_preflight(require_cuda=args.require_cuda)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
