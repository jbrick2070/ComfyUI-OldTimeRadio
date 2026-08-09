"""
_otr_memory.py  --  shared DRAM/VRAM hygiene helpers for OTR nodes
==================================================================

BUG-LOCAL-030 long-form hardening (2026-05-03 EVENING, post round-robin
risk-#10 review). Two helpers used at phase boundaries inside the
composite pipeline:

  * ``phase_gc(label)`` -- gc.collect() + torch.cuda.empty_cache() +
    torch.cuda.ipc_collect(). Idempotent, never raises. Run at a phase
    boundary so PyTorch returns RAM/VRAM to the OS instead of holding
    it across phases.

  * ``dram_canary(min_free_gb=6.0, label="")`` -- best-effort psutil
    check. Returns (ok: bool, reason: str). ``ok=True`` on any error
    (psutil missing, syscall fails, etc.) so the canary degrades open,
    never blocks a render that would otherwise complete. Caller decides
    whether to log + abort or just log + continue when ``ok=False``.

Why a shared module: it was written for three phase barriers, so
putting the logic in one place kept the invariants identical and made
adding a call site cheap.

CURRENT REALITY (grep before trusting this paragraph): there is
exactly ONE live ``phase_gc`` call site in production --
``otr_post_upscale_procgen_blend.py`` at its entry. The other two
barriers are gone, not merely quiet: ``OTR_BatchHumoRender`` was
removed (it is in ``DELETED_NODE_TYPES``; HuMo is an in-process
adapter now), and queue item 8 (2026-08-08) ripped the standalone
RTXUpscale stage, folding per-clip model enhancement INSIDE
SilentComposite (``nodes/_otr_upscale_engines/``) where it needs no
handoff of its own. The helpers stay because the remaining barrier is
real and a future phase may want one.

Both helpers are best-effort: any internal exception is caught and
logged at WARNING. Callers should NEVER let memory hygiene abort
their main work.
"""
from __future__ import annotations

import logging

log = logging.getLogger("OTR")


# Default ceiling: 6 GB free DRAM. Below this, the canary trips. Picked
# from Gemini's round-robin recommendation (risk-#10 review). On
# Jeffrey's 32 GB workstation this leaves ~5x headroom for the OS +
# active apps; trip-on-empty means we still have enough RAM to abort
# cleanly instead of taking the workstation down with us.
DEFAULT_MIN_FREE_DRAM_GB = 6.0


def phase_gc(label: str = "") -> None:
    """Force gc.collect() + torch.cuda.empty_cache() + ipc_collect.

    Run at phase boundaries. PyTorch is lazy about returning RAM/VRAM
    to the OS after heavy generation (HuMo, LTX, FLUX); this nudges
    it. Cheap (single-digit ms even after GB-scale activity) so safe
    to call generously. Logs at INFO so it shows up in the runtime
    log without spamming.
    """
    import gc
    try:
        gc.collect()
    except Exception as exc:
        log.warning("[OTR_Memory] gc.collect() failed (%s): %s", label, exc)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                log.warning(
                    "[OTR_Memory] torch.cuda.empty_cache() failed (%s): %s",
                    label, exc,
                )
            try:
                torch.cuda.ipc_collect()
            except Exception as exc:
                log.warning(
                    "[OTR_Memory] torch.cuda.ipc_collect() failed (%s): %s",
                    label, exc,
                )
    except Exception:
        # torch not importable in some test contexts -- silent skip.
        pass
    if label:
        log.info("[OTR_Memory] phase_gc complete: %s", label)


def dram_canary(
    min_free_gb: float = DEFAULT_MIN_FREE_DRAM_GB,
    label: str = "",
) -> tuple[bool, str]:
    """Check available system RAM. Returns (ok, reason).

    ``ok=True`` if at least ``min_free_gb`` GB of system RAM is
    available, OR if psutil is unavailable / the syscall fails. The
    canary is best-effort and degrades OPEN (never blocks a render
    that would otherwise complete) -- callers must be willing to
    accept ``ok=True`` when the helper cannot actually verify.

    ``ok=False`` only when psutil successfully reported < min_free_gb.
    Callers should log + abort gracefully (return a fallback path,
    raise a caught exception) rather than hard-crashing the
    workstation.

    Reason string is human-readable: ``"3.42 GB free, < 6.00 GB
    threshold"`` or ``"psutil unavailable"`` etc.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return True, "psutil unavailable (canary degrades open)"
    try:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024.0 ** 3)
    except Exception as exc:  # noqa: BLE001
        return True, f"psutil.virtual_memory() failed: {exc}"
    if avail_gb < float(min_free_gb):
        msg = (
            f"DRAM canary TRIPPED at {label or 'unnamed phase'}: "
            f"{avail_gb:.2f} GB free, < {min_free_gb:.2f} GB threshold. "
            "Caller should abort the current phase and degrade to "
            "fallback rather than risking workstation OOM."
        )
        log.warning("[OTR_Memory] %s", msg)
        return False, msg
    return True, f"{avail_gb:.2f} GB free, >= {min_free_gb:.2f} GB threshold"


__all__ = [
    "DEFAULT_MIN_FREE_DRAM_GB",
    "phase_gc",
    "dram_canary",
]
