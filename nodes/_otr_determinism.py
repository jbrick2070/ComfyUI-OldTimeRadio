"""Determinism scaffolding for the audio forward (C-1, C-2, C-3, I-2).

``REQUIRED_DETERMINISM_ENV`` is the env the headless launcher must export
BEFORE python/torch starts (C-1); a test pins ``scripts/_otr_soak_server_launch.cmd``
against it.

``deterministic_inference(seed)`` is a SCOPED context manager that pins strict
determinism (``use_deterministic_algorithms(True, warn_only=False)`` + SDPA MATH
backend) and seeds every RNG around ONE audio forward, then restores all prior
flags + RNG state in ``finally``. The PROCESS default stays non-strict so the
default video render does not crash on sm_120 (C-2/C-3).

Import-time is side-effect-free (C-5).
"""
from __future__ import annotations

import contextlib
import random
import threading

import torch

# The env the launcher must export BEFORE python/torch starts (C-1).
REQUIRED_DETERMINISM_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "NVIDIA_TF32_OVERRIDE": "0",
    "TOKENIZERS_PARALLELISM": "false",
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


#: ONE OUTERMOST SCOPE OWNS THE PROCESS STATE (audit B9, 2026-11-19).
#: Every flag this context sets is PROCESS-GLOBAL -- the deterministic
#: algorithms switch, cuDNN, TF32, the SDPA backend pin, MPS memory filling
#: and the RNGs -- and the cloud fan-out enters it from up to four worker
#: threads at once. When each entry saved and restored its own snapshot, two
#: overlapping scopes exiting out of order restored each other's STRICT state:
#: deterministic mode, cuDNN-deterministic and math-only SDPA stayed on for
#: the rest of the server's life, and on Apple Silicon a switch between
#: ``use_deterministic_algorithms(True)`` and the MPS fill-off line left NaN
#: memory filling on, which re-arms the baddbmm NaN of PBUG-20260907-09b
#: (measured on the M6, torch 2.12.1: 1024/1024 NaN). Now the first entry
#: snapshots and applies the strict state under a lock, nested or concurrent
#: entries only re-seed, and the LAST exit restores the snapshot.
_SCOPE_LOCK = threading.Lock()
_SCOPE: dict = {"depth": 0, "saved": None, "stack": None}


def _mps_available() -> bool:
    # Probed per call rather than at import: torch is imported lazily in places
    # here, and a module-level probe would fix the answer before the backend is
    # necessarily ready. Cheap -- it is a cached attribute read.
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 -- older torch without the mps namespace
        return False


def _numpy():
    try:
        import numpy as _np
        return _np
    except Exception:  # noqa: BLE001
        return None


def _snapshot(have_cuda: bool) -> dict:
    """Every process-global flag and RNG state this context touches."""
    try:
        prev_fill = bool(torch.utils.deterministic.fill_uninitialized_memory)
    except Exception:  # noqa: BLE001 -- torch < 2.1 has no such flag
        prev_fill = True
    try:
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    except Exception:
        prev_warn = False
    _np = _numpy()
    return {
        "det": torch.are_deterministic_algorithms_enabled(),
        "fill": prev_fill,
        "warn": prev_warn,
        "bench": torch.backends.cudnn.benchmark,
        "cudnn_det": torch.backends.cudnn.deterministic,
        "mm_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "py": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if have_cuda else None,
        "np": _np.random.get_state() if _np is not None else None,
    }


def _apply_strict(warn_only: bool, mps: bool) -> None:
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
    if mps:
        try:
            torch.utils.deterministic.fill_uninitialized_memory = False
        except Exception:  # noqa: BLE001 -- never break generation over a flag
            pass


def _restore(saved: dict, have_cuda: bool, mps: bool) -> None:
    if mps:
        try:
            torch.utils.deterministic.fill_uninitialized_memory = saved["fill"]
        except Exception:  # noqa: BLE001
            pass
    try:
        torch.use_deterministic_algorithms(saved["det"], warn_only=saved["warn"])
    except Exception:
        torch.use_deterministic_algorithms(saved["det"])
    torch.backends.cudnn.benchmark = saved["bench"]
    torch.backends.cudnn.deterministic = saved["cudnn_det"]
    torch.backends.cuda.matmul.allow_tf32 = saved["mm_tf32"]
    torch.backends.cudnn.allow_tf32 = saved["cudnn_tf32"]
    random.setstate(saved["py"])
    torch.set_rng_state(saved["torch"])
    if have_cuda and saved["cuda"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda"])
    _np = _numpy()
    if _np is not None and saved["np"] is not None:
        _np.random.set_state(saved["np"])


def _seed_all(seed: int, have_cuda: bool) -> None:
    random.seed(seed)
    _np = _numpy()
    if _np is not None:
        _np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if have_cuda:
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def deterministic_inference(seed: int, *, warn_only: bool = False):
    """Scoped strict determinism around a SINGLE audio forward.

    Seeds python / numpy / torch / cuda from ``seed`` (numpy masked to 32 bits),
    enables ``use_deterministic_algorithms(True, warn_only=warn_only)`` and pins
    the SDPA MATH backend, then restores every prior flag + RNG state on exit.
    Safe on CPU-only hosts (CUDA branches guard on availability).

    THREAD-SAFE BY OWNERSHIP, NOT BY SERIALIZING. The outermost entry saves
    and applies; an entry while another scope is open (a nested call, or a
    cloud fan-out worker) only re-seeds, and its exit restores nothing; the
    last exit restores the outermost snapshot. The forwards themselves still
    run concurrently. ``warn_only`` is the outermost entry's.
    """
    seed = int(seed)
    have_cuda = torch.cuda.is_available()
    mps = _mps_available()
    with _SCOPE_LOCK:
        if _SCOPE["depth"] == 0:
            _SCOPE["saved"] = _snapshot(have_cuda)
            try:
                _apply_strict(warn_only, mps)
                stack = contextlib.ExitStack()
                _enter_sdpa_math(stack)
            except BaseException:
                _restore(_SCOPE["saved"], have_cuda, mps)
                _SCOPE["saved"] = None
                raise
            _SCOPE["stack"] = stack
        _SCOPE["depth"] += 1
        try:
            _seed_all(seed, have_cuda)
        except BaseException:
            # A seed torch refuses (outside 64 bits; the seed comes from a
            # widget) must not leave this entry counted: a depth that never
            # returns to zero would keep the strict state on for good
            # (Sonnet QA 2026-11-19). Roll the entry back, restore if it was
            # the only one, and re-raise.
            _SCOPE["depth"] -= 1
            if _SCOPE["depth"] == 0:
                stack, saved = _SCOPE["stack"], _SCOPE["saved"]
                _SCOPE["stack"] = _SCOPE["saved"] = None
                try:
                    if stack is not None:
                        stack.close()
                finally:
                    _restore(saved, have_cuda, mps)
            raise
    try:
        yield
    finally:
        with _SCOPE_LOCK:
            _SCOPE["depth"] -= 1
            if _SCOPE["depth"] == 0:
                stack, saved = _SCOPE["stack"], _SCOPE["saved"]
                _SCOPE["stack"] = _SCOPE["saved"] = None
                try:
                    if stack is not None:
                        stack.close()
                finally:
                    _restore(saved, have_cuda, mps)
