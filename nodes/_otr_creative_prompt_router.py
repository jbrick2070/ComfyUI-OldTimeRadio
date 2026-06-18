"""Sprint D D2a -- creative-phase prompt resolver.

When a writer-side creative-phase prompt assembly looks up its
system-prompt string, it goes through this resolver instead of
referencing the local `_SYSTEM_PROMPT` constant directly. The
resolver dispatches on the catalog row's `prompt_profile`:

  prompt_profile = "modern"        -> return the phase-specific
                                       modern prompt from the
                                       per-phase module
  prompt_profile = "otr_1940s_v1"  -> return OTR_PERIOD_SYSTEM_PROMPT
                                       from _otr_period_prompts for
                                       every creative phase

The four phases the writer's creative slot covers:
  outline
  line_composer_system
  polish_character
  polish_announcer

Sprint D D2a: helper defined, NOT wired. Caller-count tests enforce
zero production callers at the D2a boundary. Sprint D D2b wires the
resolver into the 4 phase sites; caller-count flips to exactly 4
at the D2b boundary.

Sprint D D2c few-shot decision: `render_few_shot_block` from
`_otr_period_prompts` is documented as OMIT for v1 (saves ~600
tokens of context budget). The caller-count test pins
render_few_shot_block at 0 production callers; re-introducing it
fires the test and forces a deliberate scope decision.

Audio C7 contract: at default config (both writer slots on
Mistral-Nemo, prompt_profile = "modern" everywhere) the resolver
returns the EXACT SAME object references as the pre-D2b direct
constant lookups. Byte identity holds through D2a (no consumers
yet) and D2b (consumers wired, but default path bit-identical).
"""
from __future__ import annotations

from typing import Literal

from . import _otr_model_catalog
from ._otr_line_composer import (
    _SYSTEM_PROMPT as _MODERN_LINE_COMPOSER_SYSTEM,
)
from ._otr_outline import _SYSTEM_PROMPT as _MODERN_OUTLINE_SYSTEM
from ._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT


Phase = Literal[
    "outline",
    "line_composer_system",
]


# Frozen mapping from phase identifier to the corresponding modern
# system-prompt string. Built at module-import time from the four
# per-phase constants so the returned references are object-identity
# stable across calls (preserves the Sprint D audio C7 contract under
# default config).
_MODERN_BY_PHASE: dict[str, str] = {
    "outline":              _MODERN_OUTLINE_SYSTEM,
    "line_composer_system": _MODERN_LINE_COMPOSER_SYSTEM,
}


def resolve_creative_system_prompt(repo_id: str, phase: Phase) -> str:
    """Return the system-prompt string for a creative-phase call.

    Args:
        repo_id: the canonical HF repo_id assigned to the writer's
            creative slot (the same value the writer stamps at
            `meta.creative_model` in D2b).
        phase: one of the four creative-phase identifiers in `Phase`.

    Returns:
        The system-prompt string for that (model, phase) pair.

    Raises:
        ValueError: if `phase` is not one of the four declared
            identifiers. Catches typos at the call site.

    A repo_id that is NOT a curated local model -- a remote slot handle
    (``openrouter:slot-a/b``, ``comfy:slot-a/b``) or any other non-curated id --
    has no per-model ``prompt_profile``, so it resolves to the MODERN prompt
    (the default). The period prompt (``otr_1940s_v1``) is an opt-in property of
    specific CURATED local models only; it must never KeyError-crash an episode
    just because the creative slot is a remote model (BUG: a remote creative
    writer aborted the whole run here, 2026-06-18).
    """
    if phase not in _MODERN_BY_PHASE:
        raise ValueError(
            f"unknown creative phase {phase!r}; expected one of "
            f"{sorted(_MODERN_BY_PHASE)}"
        )
    rows = {m.repo_id: m for m in _otr_model_catalog.CURATED_LLM_MODELS}
    row = rows.get(repo_id)
    if row is not None and row.prompt_profile == "otr_1940s_v1":
        return OTR_PERIOD_SYSTEM_PROMPT
    return _MODERN_BY_PHASE[phase]


__all__ = [
    "Phase",
    "resolve_creative_system_prompt",
]
