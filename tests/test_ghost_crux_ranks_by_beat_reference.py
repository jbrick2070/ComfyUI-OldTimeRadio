"""Ghost v3 Half B, deterministic tier: the beat's own words RANK its subject.

Operator ruling 2026-09-03 (hard). `resolve_crux_kernel` chose the beat's subject by
the beat's POSITION in the episode -- `key_objects[ordinal % len(objects)]` -- so the
beat about the ledger drew `pen` purely because it was the third row, and swapping two
beats swapped the pictures with them.

THREE RULES FROM THE RULING, and they are not the same rule:

1. The subject is a PHYSICAL ARTIFACT of the story world -- a thing that could be
   photographed. That is already guaranteed here, because the candidate pool is
   `meta.key_objects` and nothing else.
2. **Beat reference RANKS candidates; it is NOT the source.** "Especially if referred
   to in the beat" is a strong preference and does not license lifting whatever noun
   the beat happens to contain.
3. **No fluff, and the worked case is a NEGATION.** A noun the dialogue mentions only
   to say it is ABSENT is not an artifact of the scene.

THE CASE THAT MOTIVATED THE ITEM IS THE CASE THAT MUST NOT BE DRAWN. On
`signal_lost_the_municipal_ledger_20260721_020231` the operator said *"I don't see any
trucks though, it does mention a truck once"*. The line:

    "Look at these numbers, Maurice. This isn't just some dusty list of truck routes
     and coal tonnages; it's the heartbeat of an entire community."

Ellie is holding a LEDGER in an ARCHIVE. There is no truck in the scene -- it exists
only inside a rhetorical negative. A dialogue-noun extractor draws a coal truck into
an archive drama, and radio dialogue is full of the same shape.

KNOWN CEILING, measured in the ruling and NOT a defect in this tier: across 339
episodes and 3,643 beats with text, the dialogue names one of the episode's own
key_objects on 26.3% of beats. This deterministic tier cannot reach a beat whose
dialogue names no listed object -- which is why the operator also chose to extend the
batched author. On those beats the odometer below still runs, byte-identical to today.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines.ghost_signal_author import (
    _beat_mentions_object,
    resolve_crux_kernel,
)


META = {
    "key_objects": ["a brass pen", "a handwritten municipal ledger", "a coal truck"],
    "story_brief_terms": {"setting": ["the high-security archive"]},
}

#: The exact line from the ruling.
NEGATION_LINE = (
    "Look at these numbers, Maurice. This isn't just some dusty list of truck "
    "routes and coal tonnages; it's the heartbeat of an entire community."
)


# --------------------------------------------------------------------------- #
# The defect: position chose the subject
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ordinal", [0, 1, 2, 3, 7])
def test_a_named_object_wins_regardless_of_the_beats_position(ordinal):
    """THE FIX: swapping two beats must no longer swap their pictures."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=ordinal, role="character_video",
        beat_text="She opened the municipal ledger and traced the column.")
    assert "ledger" in kernel, (ordinal, kernel)
    assert source == "key_object_in_beat"


def test_without_beat_text_the_odometer_is_unchanged():
    """The ~74% of beats that name no listed object must behave exactly as before."""
    seen = [resolve_crux_kernel(META, ordinal=o, role="character_video")[0]
            for o in range(3)]
    assert "pen" in seen[0] and "ledger" in seen[1] and "truck" in seen[2]


# --------------------------------------------------------------------------- #
# Rule 3: the negation. This is the case the operator actually complained about.
# --------------------------------------------------------------------------- #
def test_a_noun_named_only_inside_a_negation_is_not_the_subject():
    """No truck is in the scene -- it exists only in a rhetorical negative."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", beat_text=NEGATION_LINE)
    assert "truck" not in kernel, kernel
    assert source == "key_object", "it must fall back to the odometer, not rank"


def test_the_negation_does_not_reach_across_a_sentence_boundary():
    """A cue in the PREVIOUS sentence must not negate a clean later mention.

    Measured while writing this: an unbounded 8-word lookback read
    "It isn't a ledger. Then she opened the ledger." as fully negated and threw
    the real subject away.
    """
    assert _beat_mentions_object(
        "It isn't a ledger. Then she opened the ledger.", "ledger") is True


@pytest.mark.parametrize("line", [
    "It isn't a ledger at all.",
    "That was never a ledger.",
    "This is not just a ledger.",
    "Far from a ledger, it is a map.",
])
def test_negation_cues_are_honoured(line):
    assert _beat_mentions_object(line, "municipal ledger") is False


# --------------------------------------------------------------------------- #
# Rule 2: reference RANKS, it never SOURCES
# --------------------------------------------------------------------------- #
def test_a_noun_in_the_beat_that_is_not_a_key_object_is_never_drawn():
    """The whole point: the candidate pool stays `meta.key_objects`.

    A beat full of vivid nouns that the episode never listed must not introduce
    any of them -- that is the dialogue-noun extractor the ruling forbids.
    """
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video",
        beat_text="The lighthouse and the harbour and the storm were everywhere.")
    for stray in ("lighthouse", "harbour", "storm"):
        assert stray not in kernel, (stray, kernel)
    assert source == "key_object"


def test_an_empty_or_missing_beat_text_is_safe():
    for txt in ("", None, "   "):
        kernel, _ = resolve_crux_kernel(
            META, ordinal=0, role="character_video", beat_text=txt)
        assert kernel


@pytest.mark.parametrize("role,mode", [
    ("announcer_visual", "object"),
    ("announcer_visual", "signal"),
    ("music_visual", "object"),
    ("music_visual", "signal"),
])
def test_the_bookend_radio_still_wins_on_a_non_character_beat(role, mode):
    """A bookend takes its subject from the radio motifs, NOT from dialogue.

    The ruling is explicit: "radio objects stay on the announcer and music beds,
    but placed in the setting". Beat text must not be able to pull a bookend off
    the radio -- without this, a music bed whose neighbouring line mentions the
    ledger would stop drawing the programme's own object.

    Parametrized over the real GHOST_BOOKEND_MOTIFS keys on purpose: a first
    draft of this test passed mode="" , which matches no motif, so the branch
    never fired and the test proved nothing.
    """
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role=role, mode=mode,
        beat_text="She opened the municipal ledger and traced the column.")
    assert source == "bookend_radio", (kernel, source)
    assert "ledger" not in kernel
