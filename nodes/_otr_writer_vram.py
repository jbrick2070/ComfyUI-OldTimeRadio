"""nodes/_otr_writer_vram.py -- S3 post-script LLM VRAM reclaim.

The writer holds a multi-GB creative/technical LLM resident across all of its
LLM phases (style, news, cast, outline, dialogue, title, story-brief). The LAST
phase is story_brief_reflection; after it the writer only assembles + saves the
ledger. Evicting the LLM there -- after the script, before the downstream TTS /
render phase -- keeps it from being co-resident with Bark / Kokoro / HuMo / FLUX
during render.

This is NOT a between-phases offload (CLAUDE.md "never force_vram_offload
between LLM phases"): every writer LLM phase has already run by the call site,
so a full teardown is correct here.

Pure helper, no widget / INPUT_TYPES / socket -> no workflow-JSON wiring
(PD3 N/A). Never raises (PD1, audio is king): a teardown failure must not break
the writer's output.

NOTE: the actual VRAM-envelope benefit (LLM gone before render) can only be
confirmed by an operator GPU smoke -- this path does nothing observable headless.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def unload_writer_llm_after_script(unload_fn=None) -> str:
    """Evict the writer LLM after the last LLM phase. Returns a status string
    for testability: "unloaded" | "skipped_env" | "skipped_no_local" |
    "skipped_error".

    Gated by OTR_WRITER_UNLOAD_AFTER_SCRIPT (default "1" = on). Set it to "0"
    to disable without a code change (operator escape hatch for the unvalidated
    path). `unload_fn` is injectable for tests; production uses
    _otr_model_loader.unload_llm (the canonical full teardown) only when a
    local writer LLM is actually resident. A fully cloud LLM run never stores
    OpenRouter / Comfy Credits entries in the local singleton cache, so it must
    not import torch just to empty an already-empty CUDA allocator.
    """
    if os.environ.get("OTR_WRITER_UNLOAD_AFTER_SCRIPT", "1").strip() == "0":
        return "skipped_env"
    try:
        if unload_fn is None:
            try:
                from . import _otr_model_loader as _OTRML
            except ImportError:  # pragma: no cover - standalone / test load
                import _otr_model_loader as _OTRML  # type: ignore
            if not _OTRML.unload_llm_if_local_resident():
                return "skipped_no_local"
            return "unloaded"
        unload_fn()
        return "unloaded"
    except Exception as exc:  # noqa: BLE001 -- never break the writer (PD1)
        log.warning("[OTR writer] post-script LLM unload skipped: %s", exc)
        return "skipped_error"
