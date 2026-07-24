"""Positive contract tests for the shared inline line composer."""

from __future__ import annotations

import pytest

from nodes._otr_line_composer import (
    LineCompositionFailedError,
    LineRequest,
    _build_user_prompt,
    compose_line,
    render_current_beat,
    render_outline_spine,
    strip_line_formatting,
)


def _req(**updates) -> LineRequest:
    values = {
        "speaker": "ALICE VALE",
        "intent": "Answer the signal.",
        "mood": "alert",
        "canon_header": "The relay stands beside a flooded square.",
        "last_lines": [],
        "allowed_people": frozenset({"ALICE VALE", "BOB REED"}),
        "allowed_things": frozenset({"velvet chair", "smoking room"}),
    }
    values.update(updates)
    return LineRequest(**values)


def _creative_sequence(*texts):
    state = {"index": 0, "calls": 0}

    def generate(messages, *, temperature, max_new_tokens, stop=None):
        state["calls"] += 1
        index = min(state["index"], len(texts) - 1)
        state["index"] += 1
        return texts[index]

    generate.state = state
    return generate


def test_prompt_transports_exact_story_context_and_source_terms():
    prompt = _build_user_prompt(_req(
        style_descriptor="quiet documentary",
        theme="trust under pressure",
        outline_spine="OUTLINE\n- b001: the signal arrives",
        current_beat_block="CURRENT BEAT\n-> b001",
        position="setup, beat 1 of 4",
    ))

    assert "The relay stands beside a flooded square." in prompt
    assert "ALICE VALE, BOB REED" in prompt
    assert "smoking room, velvet chair" in prompt
    assert "quiet documentary" in prompt
    assert "trust under pressure" in prompt
    assert "Answer the signal." in prompt


@pytest.mark.parametrize(
    "text",
    [
        "Go.",
        " ".join(["lantern"] * 700),
        "WARNING: The transmission has begun.",
        "[REDACTED] stays in the station record.",
        "She smoked a cigarette beside the velvet chair.",
        "A new silver tram crossed the flooded square.",
    ],
)
def test_length_vocabulary_and_style_never_recall_or_replace(text):
    creative = _creative_sequence(text, "replacement must never be requested")
    result = compose_line(
        creative_fn=creative,
        req=_req(),
        max_attempts=2,
    )

    assert result.text == text
    assert creative.state["calls"] == 1


def test_empty_transport_response_can_retry_once():
    creative = _creative_sequence("   ", "The signal is steady.")
    result = compose_line(
        creative_fn=creative,
        req=_req(),
        max_attempts=2,
    )

    assert result.text == "The signal is steady."
    assert creative.state["calls"] == 2


def test_all_empty_responses_fail_as_missing_spoken_structure():
    creative = _creative_sequence("", " ")
    with pytest.raises(LineCompositionFailedError):
        compose_line(
            creative_fn=creative,
            req=_req(),
            max_attempts=2,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ALICE VALE: The signal is steady.", "The signal is steady."),
        ("[ALICE VALE] The signal is steady.", "The signal is steady."),
        ("[VOICE: SOMEONE] The signal is steady.", "The signal is steady."),
        ("WARNING: The signal is steady.", "WARNING: The signal is steady."),
        ("[REDACTED] The signal is steady.", "[REDACTED] The signal is steady."),
    ],
)
def test_transport_cleanup_uses_explicit_or_authoritative_labels(raw, expected):
    assert strip_line_formatting(
        raw, {"ALICE VALE", "ANNOUNCER"}
    ) == expected


def test_outline_helpers_preserve_authored_beat_text():
    outline = [
        {"beat_id": "b001", "speaker": "ALICE VALE", "intent": "Begin."},
        {"beat_id": "b002", "speaker": "BOB REED", "intent": "Answer."},
    ]

    spine = render_outline_spine(outline)
    current = render_current_beat(outline, "b002")

    assert "Begin." in spine and "Answer." in spine
    assert "b002" in current and "Answer." in current
