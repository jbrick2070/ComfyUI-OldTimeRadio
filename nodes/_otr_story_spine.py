"""Mandatory post-script handoff for shared inline banks.

The handoff unloads the writer model and runs one deterministic structural,
transport, and narrow-safety scrub. It has no word-fit, critic, quality,
anti-loop, candidate, or reauthoring branch.
"""
from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger("OTR.story_spine")


def _unload_writer_llm(meta: dict) -> None:
    try:
        from . import _otr_writer_vram as writer_vram
        meta["writer_llm_unload"] = (
            writer_vram.unload_writer_llm_after_script()
        )
    except Exception as exc:
        meta["writer_llm_unload"] = f"error:{type(exc).__name__}"
        log.warning("writer LLM unload failed: %r", exc)


def run_post_script_spine(led: Any, meta: dict) -> None:
    """Unload the writer and scrub the accepted ledger without reauthoring."""
    _unload_writer_llm(meta)
    try:
        from ._otr_ledger_scrub import scrub_ledger
        ledger_data = getattr(led, "data", led)
        result = scrub_ledger(ledger_data)
        meta["ledger_scrub"] = result.to_dict()
        meta["story_spine_status"] = (
            "clean" if result.ok else "structural_or_safety_failure"
        )
        if result.normalized:
            save = getattr(led, "save", None)
            if callable(save):
                save()
    except Exception as exc:
        meta["story_spine_status"] = f"error:{type(exc).__name__}"
        log.warning("post-script scrub failed: %r", exc)


__all__ = ["run_post_script_spine"]
