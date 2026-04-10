"""
_vram_log.py

Theme C / v1.4 — Per-phase VRAM snapshot logging.

Lightweight telemetry for the Gemma 4 orchestrator and any future node that
wants to record its VRAM high-water mark. The snapshot lines are written to
the shared `otr_runtime.log` in a structured format that
`tests/vram_profile_test.py` can parse later.

Design rules
------------
- CUDA-absent safe. Every public function is a no-op when torch.cuda is not
  available. This keeps the orchestrator importable on CI machines and on
  the sandbox used for AST regression tests.
- No heavy imports at module load time. torch is imported lazily inside
  each function so importing this module costs nothing.
- Peak counter reset is opt-in. The caller decides when a new phase begins;
  we never reset implicitly because that would destroy overlapping peaks
  across nested callers.
- Log format is a single line, greppable, machine-parseable:
      VRAM_SNAPSHOT phase=<label> current_gb=<float> peak_gb=<float>
  Ceiling cross events also emit a second warning line:
      VRAM_CEILING_EXCEEDED phase=<label> peak_gb=<float> ceiling_gb=14.5
- Ceiling is the same 14.5 GB real-world target documented in ROADMAP and
  enforced in tests/vram_profile_test.py.

Usage (from a node method)
--------------------------
    from ._vram_log import vram_snapshot, vram_reset_peak

    vram_reset_peak("script_writer_entry")
    # ... heavy work ...
    vram_snapshot("script_writer_after_model_load")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

log = logging.getLogger("OTR")

VRAM_CEILING_GB: float = 14.5

# Runtime log path — same file the orchestrator uses. Kept local to this
# module so callers do not have to thread a path through.
_RUNTIME_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "otr_runtime.log",
)


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _write_runtime_log(line: str) -> None:
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        with open(_RUNTIME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        # Never let telemetry take down a generation run.
        pass


def vram_reset_peak(label: str = "") -> None:
    """Reset the CUDA peak memory counter so the next snapshot is per-phase."""
    if not _cuda_available():
        return
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        if label:
            _write_runtime_log(f"VRAM_RESET phase={label}")
    except Exception as exc:
        log.debug("vram_reset_peak(%s) failed: %s", label, exc)


def vram_snapshot(label: str) -> dict:
    """Record current and peak VRAM to the runtime log. Returns the numbers.

    Returns a dict with current_gb and peak_gb (both 0.0 when CUDA is absent)
    so callers that want to react programmatically can do so without parsing
    the log file.
    """
    result = {"phase": label, "current_gb": 0.0, "peak_gb": 0.0}
    if not _cuda_available():
        return result
    try:
        import torch
        current = int(torch.cuda.memory_allocated())
        peak = int(torch.cuda.max_memory_allocated())
        current_gb = current / (1024.0 ** 3)
        peak_gb = peak / (1024.0 ** 3)
        result["current_gb"] = round(current_gb, 3)
        result["peak_gb"] = round(peak_gb, 3)
        _write_runtime_log(
            f"VRAM_SNAPSHOT phase={label} "
            f"current_gb={current_gb:.3f} peak_gb={peak_gb:.3f}"
        )
        if peak_gb > VRAM_CEILING_GB:
            _write_runtime_log(
                f"VRAM_CEILING_EXCEEDED phase={label} "
                f"peak_gb={peak_gb:.3f} ceiling_gb={VRAM_CEILING_GB:.1f}"
            )
    except Exception as exc:
        log.debug("vram_snapshot(%s) failed: %s", label, exc)
    return result

_CLEANUP_CALLBACKS = []

def register_vram_cleanup(callback):
    """Register a custom cleanup function (e.g. to clear LLM caches)."""
    if callback not in _CLEANUP_CALLBACKS:
        _CLEANUP_CALLBACKS.append(callback)

def force_vram_offload():
    """Aggressively clear VRAM of non-OTR models and internal fragmentation."""
    # Step 1: Run registered OTR-specific cleanups (e.g. clear LLM cache)
    for callback in _CLEANUP_CALLBACKS:
        try:
            callback()
        except Exception:
            pass

    # Step 2: Tell ComfyUI to kick out its models (Diffusion, VAE, etc.)
    try:
        import comfy.model_management
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    
    # Step 3: Explicit Python and PyTorch garbage collection
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
