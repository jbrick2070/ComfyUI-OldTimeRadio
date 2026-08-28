"""nodes/_otr_vram_levers.py -- pipeline VRAM-residue freeing (Lever 1).

BUG-LOCAL-265 root cause: when the HuMo video phase loads, the OTR
pipeline still holds VRAM in models loaded earlier in the run. The
dominant pin -- proven on the wan_i2v 14B OOM (2026-06-15) -- is the
writer LLM (Mistral-Nemo, ~7 GB) and Bark, loaded through OTR's OWN
loaders as raw ``transformers`` objects cached in module-level dicts
(``_otr_model_loader.LLM_CACHE``, ``_otr_bark_lib._BARK_CACHE``).
``comfy.model_management`` CANNOT see those (its detach loop /
``unload_all_models`` / ``free_memory`` all report "0 models unloaded"
against them), so they stay resident and starve the next heavy engine's
VRAM budget. Any ComfyUI-tracked ModelPatcher (FLUX) is the lesser,
already-dynamically-offloaded part.

"Lever 1" is the fix: drain the OTR-owned caches (unload_llm + bark)
AND surgically DETACH any tracked ComfyUI patcher, in one pass, before
the heavy engine loads. V-4 / V-5: it NEVER calls ``unload_all_models``
-- the targeted detach is the actual VRAM move.

``free_otr_pipeline_residue()`` is the single canonical residue-freer.

CALLER LIST CORRECTED 2026-08-28. This paragraph named exactly two sites --
``OTR_HuMoTierLoader.load()`` before the tier VRAM check, and
``OTR_BatchHumoRender``'s inter-phase cleanup before Phase C -- and neither
calls it any more; ``eng_humo.py`` now only MENTIONS it in a comment
describing what its own detach-reclaim replaced. The function did not become
less important, it became more general, and the docstring simply never caught
up. Live callers today:

  * ``_otr_gguf_backend.py`` -- around GGUF model handoffs.
  * ``_otr_video_engines/render_driver.py`` -- pre-render, and the
    inter-engine reclaim between two beats on different engines, which is what
    keeps two heavy engines from co-residing on a 16 GB card.
  * ``_otr_video_engines/wrapper_bridge.py``.

Verify with a grep before trusting this list too -- it has been wrong once.

Every step is best-effort: a missing API on an older torch / Comfy
combo, or a model that was never loaded, is a no-op, never a raise.
``unload_llm()`` and ``_unload_bark()`` are both idempotent when their
cache is empty, so calling this when nothing is resident is harmless.

This module touches only MODEL weights that have already finished their
job earlier in the run -- it never reads or mutates audio data, so the
audio path stays byte-identical (CLAUDE.md Prime Directive 1).
"""
from __future__ import annotations

import logging
import sys as _sys
from pathlib import Path

# Sibling-module bootstrap: in production ComfyUI imports this as
# ``nodes._otr_vram_levers``; tests load node files standalone with no
# parent package. Prepending the nodes dir makes ``_otr_model_loader``
# and ``_otr_bark_lib`` reachable as top-level modules in both contexts.
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))

log = logging.getLogger("OTR.nodes._otr_vram_levers")


def _vram_snapshot() -> dict:
    """Best-effort ``(allocated, reserved, free, total)`` in MB.

    Every field is ``float('nan')`` when torch / CUDA is unavailable
    (headless test runs). Pure telemetry -- never raises.
    """
    snap = {
        "alloc_mb": float("nan"),
        "reserved_mb": float("nan"),
        "free_mb": float("nan"),
        "total_mb": float("nan"),
    }
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            snap["alloc_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            snap["reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            free_b, total_b = torch.cuda.mem_get_info()
            snap["free_mb"] = free_b / (1024 ** 2)
            snap["total_mb"] = total_b / (1024 ** 2)
    except Exception:  # noqa: BLE001
        pass
    return snap


def _free_writer_llm(steps_run: list, steps_failed: list) -> None:
    """Drop the writer LLM cache (``_otr_model_loader.LLM_CACHE``).

    The writer LLM should already be evicted by OTR_LedgerFreezeCascade's
    finally-block ``unload_llm()`` -- this call is the safety net for the
    case where the cascade node is absent from the graph or its unload
    raised. ``unload_llm()`` is idempotent when the cache is empty.
    """
    try:
        try:
            from . import _otr_model_loader as _ml  # type: ignore
        except ImportError:
            import _otr_model_loader as _ml  # type: ignore
        _ml.unload_llm()
        steps_run.append("unload_llm")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"unload_llm ({exc})")


def _free_bark(steps_run: list, steps_failed: list) -> None:
    """Drop the Bark TTS cache (``_otr_bark_lib._BARK_CACHE``).

    Bark finished generating speech in the audio phase; the audio data
    is already mixed and assembled downstream, so freeing the Bark model
    weights here does not touch audio output. Nothing in the HuMo path
    evicts Bark otherwise -- it is the residue ``unload_all_models()``
    most reliably misses.
    """
    try:
        try:
            from . import _otr_bark_lib as _bk  # type: ignore
        except ImportError:
            import _otr_bark_lib as _bk  # type: ignore
        _bk._unload_bark()
        steps_run.append("_unload_bark")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"_unload_bark ({exc})")


def free_otr_pipeline_residue(*, reason: str = "") -> dict:
    """Free every model the OTR pipeline leaves resident before HuMo.

    Order matters: the out-of-band transformers caches (writer LLM,
    Bark) -- the REAL residue, invisible to ``comfy.model_management`` --
    are released FIRST so their weights leave the GPU before the CUDA
    allocator's free-block coalesce runs; then any ComfyUI-tracked
    patchers (FLUX) are DETACHED surgically (weights to the offload
    device); then the cache flush hands pages back to the driver.

    V-4 / V-5: this is the single-resident-engine discipline -- it NEVER
    calls ``unload_all_models`` (the blanket registry nuke). The targeted
    detach is the actual VRAM move; the registry clear is not needed.

    Steps:
      1. ``unload_llm()``           -- writer LLM (out-of-band cache)
      2. ``_unload_bark()``         -- Bark TTS (out-of-band cache)
      3. detach tracked patchers    -- ComfyUI FLUX (weights -> offload dev)
      4. ``gc.collect()``
      5. ``mm.soft_empty_cache(force=True)``
      6. ``torch.cuda.synchronize() / empty_cache() / ipc_collect()``

    Returns a report dict with before/after VRAM telemetry and the list
    of steps that ran / failed. ``free_gb_after`` is the post-clean free
    VRAM in GB (``float('nan')`` headless) -- OTR_HuMoTierLoader's
    auto-downgrade rule reads it.

    Best-effort: never raises. A step that fails is recorded in
    ``steps_failed`` and the rest still run.
    """
    import gc

    steps_run: list = []
    steps_failed: list = []
    before = _vram_snapshot()

    # 1 + 2: OTR out-of-band caches that unload_all_models() cannot see.
    _free_writer_llm(steps_run, steps_failed)
    _free_bark(steps_run, steps_failed)

    # 3: ComfyUI-tracked models (FLUX checkpoint, any HuMo patcher).
    mm = None
    try:
        import comfy.model_management as mm  # type: ignore
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"import comfy.model_management ({exc})")
        mm = None
    if mm is not None:
        # 3a: BUG-LOCAL-291 -- detach still-tracked patchers BEFORE the
        # registry clear. ``unload_all_models()`` clears the model-
        # management registry but, on its own, does not guarantee the
        # underlying ModelPatcher weights leave the GPU when another
        # strong reference survives (e.g. the ComfyUI executor's cached
        # node outputs). Walking ``current_loaded_models`` and calling
        # ``detach(unpatch_all=True)`` mutates the SHARED patcher object,
        # which moves its weights to the offload device even though the
        # Python wrapper stays alive elsewhere. This is the same
        # mechanism the FLUX portrait node's EXIT eviction uses; doing
        # it here too means the HuMoTierLoader call site (and any smoke
        # workflow without the portrait node) gets the same reclaim.
        # Best-effort per-patcher; a failure on one never blocks the
        # rest or the unload_all_models() below.
        try:
            _tracked = list(getattr(mm, "current_loaded_models", None) or [])
        except Exception:  # noqa: BLE001
            _tracked = []
        _detached = 0
        for _lm in _tracked:
            # current_loaded_models holds LoadedModel wrappers; the
            # ModelPatcher is on .model (fall back to the entry itself).
            _patcher = getattr(_lm, "model", _lm)
            try:
                _detach = getattr(_patcher, "detach", None)
                if callable(_detach):
                    try:
                        _detach(unpatch_all=True)
                    except TypeError:
                        _detach()  # older signature: no kwargs
                    _detached += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
            # Fallback: move the underlying nn.Module weights to CPU.
            try:
                _inner = getattr(_patcher, "model", None)
                if _inner is not None and hasattr(_inner, "to"):
                    _inner.to("cpu")
                    _detached += 1
            except Exception:  # noqa: BLE001
                pass
        if _detached:
            steps_run.append(f"detach_patchers({_detached})")
        # V-4/V-5: NO ``unload_all_models``. The detach loop above already moved
        # each tracked patcher's weights to the offload device (the actual VRAM
        # move); clearing ComfyUI's model registry is not needed to free VRAM, and
        # the blanket nuke is forbidden by the single-resident-engine invariant.
        # The real residue the live data exposed -- the OUT-OF-BAND writer LLM +
        # Bark, invisible to comfy.model_management -- is freed by steps 1-2, which
        # ``unload_all_models`` never touched (it reported "0 models unloaded" on
        # the wan_i2v OOM where the writer LLM was the ~7 GB pin).

    # 4: drop python refs so the allocator coalesce sees freed pages.
    try:
        gc.collect()
        steps_run.append("gc.collect")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"gc.collect ({exc})")

    # 5: hand pages back so a VRAM probe reflects the real drop.
    if mm is not None:
        try:
            mm.soft_empty_cache(force=True)
            steps_run.append("soft_empty_cache")
        except Exception as exc:  # noqa: BLE001
            steps_failed.append(f"soft_empty_cache ({exc})")

    # 6: torch-side allocator flush.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            steps_run.append("cuda.synchronize")
            torch.cuda.empty_cache()
            steps_run.append("cuda.empty_cache")
            torch.cuda.ipc_collect()
            steps_run.append("cuda.ipc_collect")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"cuda flush ({exc})")

    after = _vram_snapshot()
    free_mb_after = after["free_mb"]
    free_gb_after = (
        free_mb_after / 1024.0 if free_mb_after == free_mb_after else float("nan")
    )

    report = {
        "reason": reason,
        "steps_run": steps_run,
        "steps_failed": steps_failed,
        "before_alloc_mb": before["alloc_mb"],
        "after_alloc_mb": after["alloc_mb"],
        "before_reserved_mb": before["reserved_mb"],
        "after_reserved_mb": after["reserved_mb"],
        "free_gb_after": free_gb_after,
        "total_gb": (
            after["total_mb"] / 1024.0
            if after["total_mb"] == after["total_mb"] else float("nan")
        ),
    }

    tag = f" ({reason})" if reason else ""
    if steps_failed:
        log.warning(
            "[VRAMLevers] free_otr_pipeline_residue%s partial: ran=%s "
            "failed=%s | allocated %.0f -> %.0f MB, reserved %.0f -> %.0f MB, "
            "free=%.0f MB",
            tag, steps_run, steps_failed,
            before["alloc_mb"], after["alloc_mb"],
            before["reserved_mb"], after["reserved_mb"],
            after["free_mb"],
        )
    else:
        log.info(
            "[VRAMLevers] free_otr_pipeline_residue%s OK: %s | "
            "allocated %.0f -> %.0f MB, reserved %.0f -> %.0f MB, free=%.0f MB",
            tag, ", ".join(steps_run),
            before["alloc_mb"], after["alloc_mb"],
            before["reserved_mb"], after["reserved_mb"],
            after["free_mb"],
        )
    return report


__all__ = ["free_otr_pipeline_residue"]
