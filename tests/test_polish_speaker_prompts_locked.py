"""tests/test_polish_speaker_prompts_locked.py — S33 B5 design lock.

S33 B5 (2026-05-15) renamed `_POLISH_SYSTEM_PROMPT` ->
`_POLISH_SYSTEM_PROMPT_CHARACTER` for symmetric naming with
`_POLISH_SYSTEM_PROMPT_ANNOUNCER`, and added a design-lock comment
block locking the two prompts against accidental collapse into a
single unified prompt.

The behavior contract is:
  * Character beats: forbid narration (it's a leak)
  * Announcer beats: allow narration (it IS the announcer voice)
  * `polish_line()` dispatches by `speaker_role` to pick the right
    prompt at runtime.
  * No `_POLISH_SYSTEM_PROMPT_UNIFIED` / `_UNIFIED_POLISH_PROMPT`
    module attribute (locked via forbidden-sweep markers + this
    test).

This file proves all six properties hold post-B5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_line_composer  # noqa: E402
from nodes._otr_line_composer import (  # noqa: E402
    _POLISH_SYSTEM_PROMPT_ANNOUNCER,
    _POLISH_SYSTEM_PROMPT_CHARACTER,
    polish_line,
)


# ---------------------------------------------------------------------------
# 1. Both module-level prompt constants exist and are distinct
# ---------------------------------------------------------------------------


class TestPromptConstantsDistinct:
    def test_polish_prompts_are_distinct(self):
        # Both constants exist as module-level attributes; their
        # text contents are NOT identical.
        assert hasattr(_otr_line_composer, "_POLISH_SYSTEM_PROMPT_CHARACTER")
        assert hasattr(_otr_line_composer, "_POLISH_SYSTEM_PROMPT_ANNOUNCER")
        assert (
            _POLISH_SYSTEM_PROMPT_CHARACTER
            != _POLISH_SYSTEM_PROMPT_ANNOUNCER
        ), (
            "S33 B5 design lock: the two polish prompts MUST stay "
            "distinct. Collapsing them regresses one role or the "
            "other (see design-lock comment in _otr_line_composer)."
        )


# ---------------------------------------------------------------------------
# 2 + 3. Behavior-based assertions on each prompt's text
# ---------------------------------------------------------------------------


class TestPromptSemantics:
    def test_polish_character_prompt_forbids_narration(self):
        # The character prompt must contain a narration-forbidden
        # marker. Behavior-based check, not name-based: confirms the
        # constant carries the semantic the design lock relies on.
        text = _POLISH_SYSTEM_PROMPT_CHARACTER
        text_lower = text.lower()
        # Accept either explicit "no he said / she replied" or any
        # "narration of any kind" / "no narration" phrasing.
        narration_forbidden = (
            "narration of any kind" in text_lower
            or 'no "he said"' in text_lower
            or "no he said" in text_lower
            or ("no " in text_lower and "narration" in text_lower)
        )
        assert narration_forbidden, (
            "S33 B5 design lock: the CHARACTER polish prompt MUST "
            "explicitly forbid narration. Removing the no-narration "
            "marker would let character beats keep narration leaks "
            "(the failure mode this prompt exists to catch)."
        )

    def test_polish_announcer_prompt_allows_narration(self):
        # The announcer prompt must explicitly allow third-person
        # narration -- it IS the announcer voice.
        text = _POLISH_SYSTEM_PROMPT_ANNOUNCER
        text_lower = text.lower()
        narration_allowed = (
            "third-person narration is ok" in text_lower
            or "third-person narration is fine" in text_lower
            or (
                "third-person" in text_lower
                and ("narration" in text_lower or "storyteller" in text_lower)
            )
        )
        assert narration_allowed, (
            "S33 B5 design lock: the ANNOUNCER polish prompt MUST "
            "explicitly allow third-person narration. Removing the "
            "narration-allowed marker would let polish strip the "
            "announcer's own voice."
        )


# ---------------------------------------------------------------------------
# 4. polish_line() dispatches different prompts by speaker_role
# ---------------------------------------------------------------------------


class TestDispatchByRole:
    def test_polish_line_dispatches_different_prompts_by_role(self):
        # Capture the system prompt at each call. Two invocations
        # with different roles must produce different system prompts.
        seen: dict[str, str] = {}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            role = next(
                m["content"] for m in messages if m["role"] == "system"
            )
            # Stamp the LAST captured system per invocation; the
            # outer code reads `seen` between calls.
            seen["last_system"] = role
            return "polished output"

        # Character invocation.
        polish_line(
            capture,
            leaked_line='"Sample text," she said.',
            speaker_voice_card="ALICE",
            speaker_role="character",
            polish_generate_fn=capture,
        )
        character_system = seen["last_system"]

        # Announcer invocation.
        polish_line(
            capture,
            leaked_line="(beat) The signal returned [whispers]",
            speaker_voice_card="ANNOUNCER",
            speaker_role="announcer",
            polish_generate_fn=capture,
        )
        announcer_system = seen["last_system"]

        assert character_system != announcer_system, (
            "S33 B5 design lock: polish_line must dispatch DIFFERENT "
            "system prompts for speaker_role='character' vs "
            "'announcer'. Same prompt for both = design collapse."
        )
        assert character_system == _POLISH_SYSTEM_PROMPT_CHARACTER
        assert announcer_system == _POLISH_SYSTEM_PROMPT_ANNOUNCER


# ---------------------------------------------------------------------------
# 5. No unified-prompt constant exists
# ---------------------------------------------------------------------------


class TestNoUnifiedPromptConstant:
    @pytest.mark.parametrize("bad_name", [
        "_POLISH_SYSTEM_PROMPT_UNIFIED",
        "_UNIFIED_POLISH_PROMPT",
    ])
    def test_no_unified_polish_prompt_constant(self, bad_name):
        assert not hasattr(_otr_line_composer, bad_name), (
            f"S33 B5 design lock regressed: `{bad_name}` exists on "
            "_otr_line_composer. The two-prompt split must be "
            "preserved (see design-lock comment)."
        )


# ---------------------------------------------------------------------------
# 6. Design-lock comment block present in source
# ---------------------------------------------------------------------------


class TestDesignLockCommentPresent:
    def test_design_lock_comment_present_in_line_composer(self):
        # Read the source file and confirm the design-lock callout is
        # in place (not just a memo elsewhere). Future contributors
        # who try to collapse the two prompts will hit the comment
        # before they edit.
        repo_root = Path(__file__).resolve().parents[1]
        src_path = repo_root / "nodes" / "_otr_line_composer.py"
        src = src_path.read_text(encoding="utf-8")
        # Look for the load-bearing phrases of the design-lock block.
        assert "S33 B5 design lock" in src, (
            "S33 B5 design lock comment block missing from "
            "nodes/_otr_line_composer.py"
        )
        assert "DO NOT collapse" in src, (
            "S33 B5 design lock callout phrase 'DO NOT collapse' "
            "missing from nodes/_otr_line_composer.py"
        )
        assert "Character beats: forbid narration" in src, (
            "S33 B5 design lock rationale ('Character beats: forbid "
            "narration') missing from nodes/_otr_line_composer.py"
        )
        assert "Announcer beats: allow narration" in src, (
            "S33 B5 design lock rationale ('Announcer beats: allow "
            "narration') missing from nodes/_otr_line_composer.py"
        )
