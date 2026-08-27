"""Positive contract tests for the shared inline line composer."""

from __future__ import annotations

import pytest

from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE
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


def _recording_creative(*texts):
    """A creative_fn that records the exact messages of every call."""
    state = {"index": 0, "calls": [], "temperatures": []}

    def generate(messages, *, temperature, max_new_tokens, stop=None):
        state["calls"].append([dict(m) for m in messages])
        state["temperatures"].append(temperature)
        index = min(state["index"], len(texts) - 1)
        state["index"] += 1
        return texts[index]

    generate.state = state
    return generate


def test_the_retry_after_an_empty_line_is_told_what_was_wrong():
    """The second ask is a CORRECTION, not the same dice thrown warmer.

    This call site used to re-send the byte-identical `messages` with the
    temperature raised 0.1, and the model was never told that its reply had
    cleaned away to nothing. A model that is not told cannot correct.
    """
    creative = _recording_creative("ALICE VALE:", "The signal is steady.")
    result = compose_line(creative_fn=creative, req=_req(), max_attempts=2)

    assert result.text == "The signal is steady."
    first, second = creative.state["calls"]

    # The first ask is the plain request; the second carries the model's own
    # rejected reply plus the complaint.
    assert len(second) == len(first) + 2
    assert second[:len(first)] == first
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"] == "ALICE VALE:"
    assert second[-1]["role"] == "user"
    assert "no words remaining" in second[-1]["content"]

    # Cooler on the correction, not hotter. Raising the temperature on a
    # reply that produced no words is asking the same question louder.
    assert creative.state["temperatures"][1] < creative.state["temperatures"][0]


def test_a_third_ask_shows_one_rejected_reply_not_a_growing_pile():
    creative = _recording_creative("", "  ", "The signal is steady.")
    result = compose_line(creative_fn=creative, req=_req(), max_attempts=3)

    assert result.text == "The signal is steady."
    first, _second, third = creative.state["calls"]
    assert len(third) == len(first) + 2
    assert third[-2]["content"] == "  "


def test_a_transport_failure_gets_a_reroll_and_no_invented_complaint():
    """There is no model turn to correct when the call itself raised."""
    state = {"calls": [], "raised": False}

    def generate(messages, *, temperature, max_new_tokens, stop=None):
        state["calls"].append([dict(m) for m in messages])
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("provider exploded")
        return "The signal is steady."

    result = compose_line(creative_fn=generate, req=_req(), max_attempts=2)

    assert result.text == "The signal is steady."
    first, second = state["calls"]
    assert second == first, "a transport failure must not fabricate a complaint"


# ---------------------------------------------------------------------------
# Cockney policy scope -- the rule follows the ACTIVE SPEAKER, not the roster.
#
# `allowed_people` is the whole episode cast. It earns its keep as named-entity
# grounding and transport cleanup, and it used to decide dialogue style as
# well: with LEMMY anywhere in that cast, every line's system prompt carried
# the Cockney rule, so the writer re-registered the entire ensemble. These
# tests pin the boundary at the one speaker the call is actually writing.
# ---------------------------------------------------------------------------


def _system_of(creative, call_index=0):
    """The system message of one recorded creative call."""
    message = creative.state["calls"][call_index][0]
    assert message["role"] == "system"
    return message["content"]


def test_a_non_lemmy_line_has_no_cockney_policy_with_lemmy_in_the_cast():
    creative = _recording_creative("The signal is steady.")
    compose_line(
        creative_fn=creative,
        req=_req(
            speaker="ALICE VALE",
            allowed_people=frozenset({"ALICE VALE", "LEMMY"}),
        ),
        max_attempts=1,
    )

    assert _COCKNEY_ORTHOGRAPHY_RULE not in _system_of(creative)


def test_a_lemmy_line_gets_the_lemmy_scoped_rule_and_the_spelling_clause():
    creative = _recording_creative("Right you are, then.")
    compose_line(
        creative_fn=creative,
        req=_req(
            speaker="LEMMY",
            allowed_people=frozenset({"ALICE VALE", "LEMMY"}),
        ),
        max_attempts=1,
    )

    system = _system_of(creative)
    assert _COCKNEY_ORTHOGRAPHY_RULE in system
    assert "For LEMMY's spoken lines only" in system
    assert "standard English spelling" in system


def test_the_empty_line_retry_carries_the_same_speaker_scope():
    """A correction retry must not quietly re-scope the accent policy."""
    creative = _recording_creative("", "Right you are, then.")
    result = compose_line(
        creative_fn=creative,
        req=_req(
            speaker="LEMMY",
            allowed_people=frozenset({"ALICE VALE", "LEMMY"}),
        ),
        max_attempts=2,
    )

    assert result.text == "Right you are, then."
    first, second = _system_of(creative, 0), _system_of(creative, 1)
    assert first == second
    assert "For LEMMY's spoken lines only" in second
