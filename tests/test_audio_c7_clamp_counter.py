"""Sprint D D2b -- audio C7 byte-identity at default config plus clamp counter.

Two assertions per v3 plan G4 gate. Both protect Prime Directive 1
at the D2b boundary:

  test_default_config_both_slots_mistral_nemo_returns_modern_prompts_byte_stable
      Calling the resolver with the default config (Mistral-Nemo,
      modern profile) returns the EXACT SAME object the legacy
      direct constant references return. Object identity, not
      equality -- string interning quirks cannot mask a drift.

  test_clamp_counter_zero_during_basic_resolver_calls
      The D0c clamp log surface is not exercised by the resolver's
      modern path. Confirms the resolver does not accidentally
      invoke the reflection repair pass.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_creative_prompt_router as router  # noqa: E402
from nodes import _otr_line_composer  # noqa: E402
from nodes import _otr_outline  # noqa: E402


MISTRAL_NEMO = "mistralai/Mistral-Nemo-Instruct-2407"
PHASES = (
    "outline",
    "line_composer_system",
)


def test_default_config_both_slots_mistral_nemo_returns_modern_prompts_byte_stable() -> None:
    """At default config the resolver returns the legacy modern
    phase prompts by object identity. This is the Prime Directive 1
    audio C7 contract at the D2b boundary -- the LLM call sees the
    EXACT SAME byte sequence pre/post wiring.
    """
    expected = {
        "outline":              _otr_outline._SYSTEM_PROMPT,
        "line_composer_system": _otr_line_composer._SYSTEM_PROMPT,
    }
    for phase in PHASES:
        actual = router.resolve_creative_system_prompt(MISTRAL_NEMO, phase)
        assert actual is expected[phase], (
            f"D2b audio C7 boundary failure at phase {phase!r}: "
            f"resolver returned a different object than the legacy "
            f"_SYSTEM_PROMPT direct reference. Object identity must "
            f"hold for Prime Directive 1. If a future commit rebuilds "
            f"these strings the audio baseline DRIFTS -- halt and "
            f"investigate before shipping."
        )


def test_clamp_counter_zero_during_basic_resolver_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The D0c clamp log line ("[OTR_StoryBrief] repair pass clamped")
    must NOT fire while the resolver runs at default config.

    Sprint 2A/2D removed the hand-rolled story-brief repair pass (and
    its clamp log) when `run_story_brief_reflection` was converted onto
    the shared `structured_call` ladder, so this count is now a
    structural zero. The assertion stands as a regression guard that
    no future change re-introduces a story-brief repair-clamp log on
    the resolver's modern path.
    """
    caplog.set_level(logging.INFO, logger="OTR")
    for phase in PHASES:
        router.resolve_creative_system_prompt(MISTRAL_NEMO, phase)
    clamp_records = [
        r for r in caplog.records
        if "[OTR_StoryBrief] repair pass clamped" in r.getMessage()
    ]
    assert len(clamp_records) == 0, (
        f"D0c clamp log fired during resolver calls (expected 0); "
        f"got {len(clamp_records)} records. Resolver is not supposed "
        f"to invoke the reflection repair pass."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
