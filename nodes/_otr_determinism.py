"""Determinism scaffolding for the audio forward (C-1, C-2, C-3, I-2).

Two layers, deliberately separated:

  * ``assert_determinism_env_ready()`` -- a pure ``os.environ`` check (no CUDA,
    never ``cuda.is_initialized()`` per C-1) confirming the run_comfy_otr
    launcher exported the determinism env BEFORE python/torch started.
  * ``deterministic_inference(seed)`` -- a SCOPED context manager that pins
    strict determinism (``use_deterministic_algorithms(True, warn_only=False)``
    + SDPA MATH backend) and seeds every RNG around ONE audio forward, then
    restores all prior flags + RNG state in ``finally``. The PROCESS default
    stays non-strict so the default video render does not crash on sm_120
    (C-2/C-3).

Import-time is side-effect-free (C-5): the lightweight module defaults are an
explicit call, not an import side effect.
"""
from __future__ import annotations

import contextlib
import os
import random

import torch

# The env the launcher must export BEFORE python/torch starts (C-1).
REQUIRED_DETERMINISM_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "NVIDIA_TF32_OVERRIDE": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


def determinism_env_status() -> dict:
    """Map each required env var -> ``(ok: bool, actual: str | None)``.

    Pure ``os.environ`` inspection; touches no torch/CUDA state (C-1).
    """
    return {
        key: (os.environ.get(key) == expected, os.environ.get(key))
        for key, expected in REQUIRED_DETERMINISM_ENV.items()
    }


def assert_determinism_env_ready(strict: bool = True) -> bool:
    """Return True iff every required determinism env var has its expected value.

    With ``strict=True`` raise ``RuntimeError`` naming the offending vars so a
    determinism run fails loudly at queue time instead of degrading silently.
    Never calls ``torch.cuda.is_initialized()`` (C-1).
    """
    status = determinism_env_status()
    bad = {k: v[1] for k, v in status.items() if not v[0]}
    if bad and strict:
        want = ", ".join(f"{k}={REQUIRED_DETERMINISM_ENV[k]!r}" for k in bad)
        got = ", ".join(f"{k}={bad[k]!r}" for k in bad)
        raise RuntimeError(
            "Determinism env not ready -- launch via "
            "scripts/run_comfy_otr.{bat,ps1}. "
            f"Expected {want}; got {got}. (See C-1.)"
        )
    return not bad


def apply_module_determinism_defaults() -> None:
    """Post-import-torch defaults safe for the whole process: TF32 off, cuDNN
    benchmark off, cuDNN deterministic on. Does NOT enable
    ``use_deterministic_algorithms`` -- that is scoped to the forward (C-2).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _enter_sdpa_math(stack: contextlib.ExitStack) -> None:
    """Pin the SDPA MATH backend for the scope, across torch API versions."""
    try:  # torch >= 2.3 preferred API
        from torch.nn.attention import SDPBackend, sdpa_kernel
        stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
        return
    except Exception:
        pass
    try:  # legacy fallback
        stack.enter_context(
            torch.backends.cuda.sdp_kernel(
                enable_math=True, enable_flash=False, enable_mem_efficient=False
            )
        )
    except Exception:
        pass  # CPU-only / unsupported -> MATH is already the only path


@contextlib.contextmanager
def deterministic_inference(seed: int, *, warn_only: bool = False):
    """Scoped strict determinism around a SINGLE audio forward.

    Seeds python / numpy / torch / cuda from ``seed`` (numpy masked to 32 bits),
    enables ``use_deterministic_algorithms(True, warn_only=warn_only)`` and pins
    the SDPA MATH backend, then restores every prior flag + RNG state on exit.
    Safe on CPU-only hosts (CUDA branches guard on availability).
    """
    seed = int(seed)
    have_cuda = torch.cuda.is_available()

    # --- save prior state ---
    prev_det = torch.are_deterministic_algorithms_enabled()
    try:
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    except Exception:
        prev_warn = False
    prev_bench = torch.backends.cudnn.benchmark
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_mm_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_py = random.getstate()
    prev_torch = torch.get_rng_state()
    prev_cuda = torch.cuda.get_rng_state_all() if have_cuda else None
    try:
        import numpy as _np
        prev_np = _np.random.get_state()
    except Exception:
        _np = None
        prev_np = None

    stack = contextlib.ExitStack()
    try:
        # --- seed every RNG ---
        random.seed(seed)
        if _np is not None:
            _np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if have_cuda:
            torch.cuda.manual_seed_all(seed)

        # --- strict flags (scoped) ---
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        _enter_sdpa_math(stack)

        yield
    finally:
        stack.close()
        try:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        except Exception:
            torch.use_deterministic_algorithms(prev_det)
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cuda.matmul.allow_tf32 = prev_mm_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        random.setstate(prev_py)
        torch.set_rng_state(prev_torch)
        if have_cuda and prev_cuda is not None:
            torch.cuda.set_rng_state_all(prev_cuda)
        if _np is not None and prev_np is not None:
            _np.random.set_state(prev_np)
