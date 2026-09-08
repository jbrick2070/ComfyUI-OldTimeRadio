"""Determinism scaffolding for the audio forward (C-1, C-2, C-3, I-2).

Two layers, deliberately separated:

  * ``assert_determinism_env_ready()`` -- a pure ``os.environ`` check (no CUDA,
    never ``cuda.is_initialized()`` per C-1) confirming the headless launcher
    exported the determinism env BEFORE python/torch started.
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
import random

import torch

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

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
        key: (otr_env.get(key) == expected, otr_env.get(key))
        for key, expected in REQUIRED_DETERMINISM_ENV.items()
    }


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
    # Probed per call rather than at import: torch is imported lazily in places
    # here, and a module-level probe would fix the answer before the backend is
    # necessarily ready. Cheap -- it is a cached attribute read.
    try:
        _MPS_AVAILABLE = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 -- older torch without the mps namespace
        _MPS_AVAILABLE = False

    # --- save prior state ---
    prev_det = torch.are_deterministic_algorithms_enabled()
    try:
        prev_fill = bool(torch.utils.deterministic.fill_uninitialized_memory)
    except Exception:  # noqa: BLE001 -- torch < 2.1 has no such flag
        prev_fill = True
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
        # APPLE SILICON: turn OFF the NaN-poisoning of uninitialized memory.
        #
        # `use_deterministic_algorithms(True)` also sets
        # `torch.utils.deterministic.fill_uninitialized_memory = True` (default
        # since torch 2.1), which fills every `torch.empty()` with NaN so that
        # reading uninitialized memory is loud instead of silent. That is a
        # DEBUGGING aid; it is not what makes results deterministic.
        #
        # On MPS it is actively destructive, because ComfyUI selects
        # sub-quadratic attention on Mac and that path calls
        #     torch.baddbmm(<uninitialized empty buffer>, q, k, beta=0)
        # `beta=0` means "ignore the input buffer entirely" -- and CPU and CUDA
        # honour that, so the NaN never matters there. The MPS kernel does NOT:
        # it propagates the buffer regardless of beta, so the NaN enters at
        # attention layer 0 and poisons everything downstream.
        #
        # MEASURED on a Mac mini M4, torch 2.12.1 (PBUG-20260907-09):
        #     deterministic OFF -> torch.empty(mps)=0    baddbmm NaNs    0/1024
        #     deterministic ON  -> torch.empty(mps)=nan  baddbmm NaNs 1024/1024
        #     same call on CPU with the flag ON          baddbmm NaNs    0/1024
        # End to end this made Stable Audio 3 return 100.0000% non-finite
        # samples on every cue -- silently, with the sampler completing all 100
        # steps and raising nothing.
        #
        # Scoped to MPS and restored in `finally`, so CUDA behaviour -- including
        # the byte-identical golden determinism runs -- is untouched. Determinism
        # itself is unaffected: a buffer that `beta=0` never reads cannot change
        # a result, whatever it is filled with.
        if _MPS_AVAILABLE:
            try:
                torch.utils.deterministic.fill_uninitialized_memory = False
            except Exception:  # noqa: BLE001 -- never break generation over a flag
                pass
        _enter_sdpa_math(stack)

        yield
    finally:
        stack.close()
        if _MPS_AVAILABLE:
            try:
                torch.utils.deterministic.fill_uninitialized_memory = prev_fill
            except Exception:  # noqa: BLE001
                pass
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
