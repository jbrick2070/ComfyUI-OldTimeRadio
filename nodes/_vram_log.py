"""
_vram_log.py

Theme C / v1.4 - Per-phase VRAM snapshot logging.

Lightweight telemetry for the Gemma 4 orchestrator and any future node that
wants to record its VRAM high-water mark. The snapshot lines are written to
the shared `otr_runtime.log` in a structured format that
`tests/vram_profile_test.py` can parse later.

Design rules
------------
- CUDA-absent safe. Legacy CUDA snapshots retain their numeric-zero contract;
  memory_snapshot independently observes process RSS and optional MPS counters.
- No heavy imports at module load time. torch is imported lazily inside
  each function so importing this module costs nothing.
- Peak counter reset is opt-in. The caller decides when a new phase begins;
  we never reset implicitly because that would destroy overlapping peaks
  across nested callers.
- Log format is a single line, greppable, machine-parseable:
      VRAM_SNAPSHOT phase=<label> current_gb=<float> peak_gb=<float>
  Pure telemetry -- no VRAM policy authority. The OOM budget is owned by the
  operator's per-hardware tier JSON, so this module only records the
  high-water mark; it never enforces a ceiling.

Usage (from a node method)
--------------------------
    from ._vram_log import vram_snapshot, vram_reset_peak

    vram_reset_peak("script_writer_entry")
    # ... heavy work ...
    vram_snapshot("script_writer_after_model_load")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("OTR")

def _runtime_log_path() -> str:
    try:
        from ._otr_paths import otr_runtime_log_path
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_paths import otr_runtime_log_path  # type: ignore
    return str(otr_runtime_log_path())


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _write_runtime_log(line: str) -> None:
    try:
        log_path = _runtime_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
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
    except Exception as exc:
        log.debug("vram_snapshot(%s) failed: %s", label, exc)
    return result


def memory_snapshot(label: str, *, model_id: str | None = None) -> dict:
    """Observe RSS and MPS allocations without changing allocation or policy.

    Missing counters are unknown, not zero. These snapshots mark actual native
    generation returns and model-retirement stages; they do not prove that all
    references were released or diagnose an OS kill. Imports remain lazy.
    """
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "phase": label, "pid": os.getpid(),
              "model_id": model_id if isinstance(model_id, str) else None,
              "process_rss_bytes": None, "mps_current_bytes": None,
              "mps_driver_bytes": None}
    try:
        import psutil
        result["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        import torch
        if getattr(torch, "mps", None) and torch.backends.mps.is_available():
            for key, counter in (("mps_current_bytes", "current_allocated_memory"),
                                 ("mps_driver_bytes", "driver_allocated_memory")):
                try:
                    result[key] = int(getattr(torch.mps, counter)())
                except Exception:
                    pass
    except Exception:
        pass
    try:
        line = "MEMORY_SNAPSHOT " + json.dumps(result, ensure_ascii=True, sort_keys=True)
        _write_runtime_log(line)
        log.info(line)
    except Exception:
        pass  # Observational logging must never break the owning operation.
    return result
