"""tests/test_period_prompts.py

Phase 0+ old-timey LLM period-prompt module. Pure stdlib unit tests:
verifies prompt shape, exemplar conformance to the period rules, and
the render_prompt assembly contract.
"""
from __future__ import annotations

import pytest

from nodes._otr_period_prompts import (
    OTR_PERIOD_SYSTEM_PROMPT,
    PERIOD_EXEMPLARS,
    PeriodExemplar,
    render_few_shot_block,
    render_prompt,
)


# ---------- system prompt shape -----------------------------------------


def test_system_prompt_is_non_empty_string():
    assert isinstance(OTR_PERIOD_SYSTEM_PROMPT, str)
    assert len(OTR_PERIOD_SYSTEM_PROMPT) > 200


def test_system_prompt_mentions_core_period_anchors():
    """If any of these anchor terms drift out of the system prompt, the
    LLM loses its tonal target. Test guards against accidental edits."""
    must_have = [
        "1940s",
        "Suspense",
        "NARRATOR",
        "CHARACTER:dialogue",
        "Family-broadcast safe",
    ]
    for token in must_have:
        assert token in OTR_PERIOD_SYSTEM_PROMPT, f"missing anchor token: {token!r}"


def test_system_prompt_forbids_modern_words():
    """The prompt must explicitly forbid the most common modern slang
    that breaks the period."""
    for forbidden in ("okay", "guys", "cool", "hey"):
        assert forbidden in OTR_PERIOD_SYSTEM_PROMPT, (
            f"forbidden-word callout missing for: {forbidden!r}"
        )


# ---------- exemplar shape ----------------------------------------------


def test_period_exemplars_are_frozen_dataclasses():
    for ex in PERIOD_EXEMPLARS:
        with pytest.raises(Exception):
            ex.title = "OTHER"  # type: ignore[misc]


def test_at_least_three_period_exemplars_shipped():
    """Three exemplars cover the small-cast / sci-fi / suspense range
    OTR most commonly produces."""
    assert len(PERIOD_EXEMPLARS) >= 3


def test_each_exemplar_has_narrator_open():
    """Period broadcast convention: every exemplar opens on NARRATOR."""
    for ex in PERIOD_EXEMPLARS:
        first_line = ex.body.lstrip().splitlines()[0]
        assert first_line.startswith("NARRATOR:"), (
            f"exemplar {ex.title!r} does not open on NARRATOR: {first_line!r}"
        )


def test_each_exemplar_uses_uppercase_dialogue_tags():
    """No lowercase tags, no bracketed tags. Loose check: every line
    that contains a colon either starts with [SOUND/MUSIC/SFX, or has
    its colon-prefixed token in upper case."""
    for ex in PERIOD_EXEMPLARS:
        for raw in ex.body.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("["):
                continue
            colon = line.find(":")
            if colon <= 0:
                continue
            tag = line[:colon].strip()
            letters = [c for c in tag if c.isalpha()]
            if not letters:
                continue
            ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            assert ratio >= 0.8, (
                f"exemplar {ex.title!r} has non-uppercase tag {tag!r}"
            )


def test_no_exemplar_uses_forbidden_modern_words():
    """The exemplars must walk the talk. Catches accidental drift if
    someone edits an exemplar without re-checking against the system
    prompt rules."""
    forbidden_substrings = [
        " okay",  # space-prefix to avoid catching "Tokyo" etc.
        " guys",
        " cool",
        "smartphone",
        "internet",
    ]
    for ex in PERIOD_EXEMPLARS:
        body_lower = ex.body.lower()
        for f in forbidden_substrings:
            assert f not in body_lower, (
                f"exemplar {ex.title!r} contains forbidden modern word: {f!r}"
            )


# ---------- render_few_shot_block ---------------------------------------


def test_render_few_shot_block_default_returns_two_exemplars():
    block = render_few_shot_block()
    # Two exemplar markers (one per block).
    assert block.count("--- Exemplar:") == 2


def test_render_few_shot_block_respects_max():
    block = render_few_shot_block(max_exemplars=1)
    assert block.count("--- Exemplar:") == 1


def test_render_few_shot_block_accepts_custom_list():
    custom = [
        PeriodExemplar(
            title="Custom",
            show_inspiration="Test",
            body="NARRATOR: hello.\n",
        )
    ]
    block = render_few_shot_block(exemplars=custom)
    assert "Custom" in block
    assert "Test" in block


def test_render_few_shot_block_max_zero_returns_empty_string():
    """A caller dialing the few-shot off via max=0 should get an empty
    string, not a NoneType crash."""
    assert render_few_shot_block(max_exemplars=0) == ""


# ---------- render_prompt -----------------------------------------------


def test_render_prompt_returns_two_strings():
    sys, user = render_prompt("write episode 1")
    assert isinstance(sys, str)
    assert isinstance(user, str)


def test_render_prompt_uses_period_system_by_default():
    sys, _ = render_prompt("anything")
    assert sys == OTR_PERIOD_SYSTEM_PROMPT


def test_render_prompt_includes_few_shot_by_default():
    _, user = render_prompt("write a brief episode")
    assert "--- Exemplar:" in user
    assert "Now write the requested episode" in user
    assert "write a brief episode" in user


def test_render_prompt_few_shot_can_be_skipped():
    _, user = render_prompt("write a brief episode", include_few_shot=False)
    assert "--- Exemplar:" not in user
    assert user == "write a brief episode"


def test_render_prompt_strips_user_instruction_whitespace():
    _, user = render_prompt("   write a brief episode   ", include_few_shot=False)
    assert user == "write a brief episode"


def test_render_prompt_custom_system_prompt_override():
    """A caller may swap the system prompt entirely (advanced use)."""
    sys, _ = render_prompt(
        "instruction",
        system_prompt="custom test system prompt",
        include_few_shot=False,
    )
    assert sys == "custom test system prompt"
