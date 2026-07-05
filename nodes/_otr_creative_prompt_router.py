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

The creative phases the resolver currently covers (see `Phase`):
  outline               -- returns the modern constant, OBJECT-IDENTICAL
                           (_otr_outline's `resolved is _SYSTEM_PROMPT`
                           sentinel depends on it)
  line_composer_system  -- Stage 1b (multi-modal schema): sourced from the
                           science JSON story pack, byte-identical VALUE
                           (not object-identity)

Wired at exactly 2 production call sites (_otr_outline, _otr_line_composer);
a caller-count test pins that count.

Sprint D D2c few-shot decision: `render_few_shot_block` from
`_otr_period_prompts` is documented as OMIT for v1 (saves ~600
tokens of context budget). The caller-count test pins
render_few_shot_block at 0 production callers; re-introducing it
fires the test and forces a deliberate scope decision.

Audio C7 contract: at default config (writer on Mistral-Nemo,
prompt_profile = "modern") the resolver returns prompts BYTE-IDENTICAL
to the legacy direct-constant lookups. `outline` is object-identical;
`line_composer_system` is value-identical (sourced from the JSON pack,
pinned byte-for-byte by tests/test_story_pack_stage1.py). The audio
baseline is unchanged either way.
"""
from __future__ import annotations

from typing import Literal

from . import _otr_model_catalog
from ._otr_line_composer import (
    _SYSTEM_PROMPT as _MODERN_LINE_COMPOSER_SYSTEM,
)
from ._otr_outline import _SYSTEM_PROMPT as _MODERN_OUTLINE_SYSTEM
# Lane-enablement chunk 1 (2026-07-06): the three outline STAGE prompts are
# pack-routed; the constants below remain as the science extraction fixture
# (byte-identity pinned by tests).
from ._otr_outline import (
    _BEAT_SYSTEM_PROMPT as _MODERN_OUTLINE_BEAT_SYSTEM,
    _MACRO_SYSTEM_PROMPT as _MODERN_OUTLINE_MACRO_SYSTEM,
    _PHASE_SYSTEM_PROMPT as _MODERN_OUTLINE_PHASE_SYSTEM,
)
from ._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT
from ._otr_story_pack import get_pack_prompt
from ._otr_story_routing import resolve_story_pack


Phase = Literal[
    "outline",
    "line_composer_system",
    # Lane chunk 1: the three outline STAGE prompts (pack-routed; the plain
    # "outline" phase above stays the period-overlay probe, object-identity).
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
]


# Frozen mapping from phase identifier to the corresponding modern
# system-prompt string. Built at module-import time from the four
# per-phase constants so the returned references are object-identity
# stable across calls (preserves the Sprint D audio C7 contract under
# default config).
_MODERN_BY_PHASE: dict[str, str] = {
    "outline":              _MODERN_OUTLINE_SYSTEM,
    "line_composer_system": _MODERN_LINE_COMPOSER_SYSTEM,
    "outline_macro_system": _MODERN_OUTLINE_MACRO_SYSTEM,
    "outline_phase_system": _MODERN_OUTLINE_PHASE_SYSTEM,
    "outline_beat_system":  _MODERN_OUTLINE_BEAT_SYSTEM,
}


# Stage 1b (multi-modal story schema): the science lane sources selected creative
# seams from its JSON story pack instead of the local Python constant. The pack
# value is BYTE-IDENTICAL to the constant (pinned by tests/test_story_pack_stage1.py);
# object-identity is intentionally not preserved for a pack-sourced phase, so the
# audio C7 contract now holds by VALUE, not object reference.
#
# Only `line_composer_system` is migrated. `outline` stays on its constant (its
# downstream `resolved is _SYSTEM_PROMPT` sentinel in _otr_outline depends on
# object identity).
#
# Stage 2: the pack is resolved through the ROUTING layer
# (resolve_story_pack via banks.json), no fixed path. Stage 2C: the workflow
# `source_bank` widget selection threads through `source_bank_id`; this
# literal is now ONLY the parameter default (byte-identical science lane).
_SCIENCE_BANK_ID = "science_news"
_PHASE_TO_PACK_SEAM: dict[str, str] = {
    "line_composer_system": "line_composer_system",
    # Lane chunk 1 (2026-07-06): the outline stage seams. A bank whose pack
    # lacks them fails LOUD at get_pack_prompt (lane-enablement item).
    "outline_macro_system": "outline_macro_system",
    "outline_phase_system": "outline_phase_system",
    "outline_beat_system":  "outline_beat_system",
}


def resolve_creative_system_prompt(
    repo_id: "str | None",
    phase: Phase,
    source_bank_id: str = _SCIENCE_BANK_ID,
) -> str:
    """Return the system-prompt string for a creative-phase call.

    Args:
        repo_id: the canonical HF repo_id assigned to the writer's
            creative slot (the same value the writer stamps at
            `meta.creative_model` in D2b).
        phase: one of the four creative-phase identifiers in `Phase`.
        source_bank_id: the story-path bank whose pack supplies any
            pack-routed seam (Stage 2C widget selection). Default is the
            science lane -- all pre-2C callers stay byte-identical.

    Returns:
        The system-prompt string for that (model, phase) pair.

    Raises:
        ValueError: if `phase` is not one of the four declared
            identifiers. Catches typos at the call site.

    A repo_id that is NOT a curated local model -- a remote slot handle
    (``openrouter:slot-a/b``, ``comfy:slot-a/b``), ``None`` (a legacy/
    reroll caller with no writer slot in scope; BUG-LOCAL-417 routes those
    through here for NON-science banks), or any other non-curated id --
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
    seam = _PHASE_TO_PACK_SEAM.get(phase)
    if seam is not None:
        # Byte-identical to _MODERN_BY_PHASE[phase] under the default
        # (science_news) bank; routed through banks.json so JSON owns the
        # prompt content. Fail-loud on unknown bank/model/seam. Stage 2C:
        # the workflow `source_bank` widget selection threads here via
        # `source_bank_id`; the science literal survives ONLY as the default.
        return get_pack_prompt(resolve_story_pack(source_bank_id), seam)
    return _MODERN_BY_PHASE[phase]


__all__ = [
    "Phase",
    "resolve_creative_system_prompt",
]
