"""The carried repair draft can overflow the prompt -- and that used to CRASH.

`_run_markup_ladder` carries the rejected draft into the repair turn so the model
can see what it is fixing. Whether the draft FITS is a prediction:
`_draft_fits_repair_turn` estimates chars/4 against a flat VRAM-shaped cap. When
that estimate is wrong in the generous direction the draft rides anyway, and the
transport refuses at the door with `PromptContextOverflowError` (phase
`prompt_no_room`) BEFORE spending a single token.

Nothing in `_otr_scifi_news_pro` caught it. The exception left the ladder, left the
node, and surfaced as a failed prompt with nothing in `otr/obs/` -- i.e. the episode
crashed when it was not supposed to, which is the operator's stated bar.

The remedy is the one the ladder ALREADY performs when the prediction says no: drop
the draft and continue without it. These tests pin the three behaviours that matter,
including the two where the guard must NOT fire.
"""
from __future__ import annotations

import pytest

from nodes._otr_generation_budget import (
    CAPACITY_PHASE_DECODE_DEGENERACY,
    CAPACITY_PHASE_PROMPT_NO_ROOM,
    GenerationDegeneracyError,
    PromptContextOverflowError,
)
from nodes._otr_scifi_news_pro_markup import ANNOUNCER_NAME
from nodes import _otr_scifi_news_pro as scifi_news_pro


CAST = ["Ada", "Bo"]

#: The marker the repair turn wraps the carried draft in. If the ladder stops
#: carrying drafts this constant stops matching and these tests fail loudly,
#: which is correct -- they would no longer be testing what they claim.
DRAFT_MARKER = "-----BEGIN REJECTED DRAFT-----"


def play(*body_lines):
    return "\n".join((
        "TITLE: The Test",
        "MUSIC: theme up",
        f"{ANNOUNCER_NAME}: Tonight, a test.",
        "SCENE 1: a room",
        "Ada: We begin the work.",
        *body_lines,
        "Bo: And we end it.",
        f"{ANNOUNCER_NAME}: That was a test.",
        "CODA: The end.",
        "MUSIC: theme down",
        "END.",
    ))


class ScriptedWriter:
    """Answers with a queued reply per call; an Exception instance is RAISED."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.prompts.append(list(messages)[-1]["content"])
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def run_ladder(writer):
    return scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=CAST,
        initial_temperature=0.7,
    )


def _overflow():
    return PromptContextOverflowError(
        "prompt requires 9001 input tokens, context_cap=8192 leaves 0",
        phase=CAPACITY_PHASE_PROMPT_NO_ROOM,
    )


# --------------------------------------------------------------------------- #
# 1. THE DEFECT: an overflow on the repair turn must not kill the episode
# --------------------------------------------------------------------------- #
def test_a_draft_that_overflows_the_repair_turn_is_dropped_not_fatal():
    """Attempt 1 is malformed, so attempt 2 carries the draft and overflows.

    The ladder must drop the draft, retry that rung once WITHOUT it, and deliver
    the clean play. Before the guard this raised out of the node.
    """
    writer = ScriptedWriter([play("*SFX: a door slams"), _overflow(), play()])

    raw, parsed, diag = run_ladder(writer)

    assert parsed is not None and raw, "the episode must still be delivered"
    assert len(writer.prompts) == 3, writer.prompts
    # the prompt that overflowed carried the draft...
    assert DRAFT_MARKER in writer.prompts[1]
    # ...and the retry of that same rung did not.
    assert DRAFT_MARKER not in writer.prompts[2], (
        "the retry resent the draft that just overflowed -- that is the banned "
        "re-roll of a deterministic prompt_no_room refusal")
    # the trace must SHOW the cold regeneration, not hide it
    assert diag["cold_regenerations"] >= 1, diag


def test_the_dropped_draft_retry_holds_its_rung_temperature():
    """Dropping the draft must not silently escalate the ladder's temperature.

    `_MARKUP_LADDER_TEMPS` documents that the markup ladder NEVER raises
    temperature. The guard retries the SAME rung, so the temperature it retries
    at is the temperature it just failed at.
    """
    seen = []

    class TempRecordingWriter(ScriptedWriter):
        def __call__(self, messages, *, temperature, max_new_tokens):
            seen.append(temperature)
            return super().__call__(
                messages, temperature=temperature,
                max_new_tokens=max_new_tokens)

    writer = TempRecordingWriter(
        [play("*SFX: a door slams"), _overflow(), play()])
    run_ladder(writer)

    assert len(seen) == 3, seen
    assert seen[2] == seen[1], (
        "the retry after dropping the draft changed rung temperature: %r" % (seen,))


# --------------------------------------------------------------------------- #
# 2. WHERE THE GUARD MUST NOT FIRE
# --------------------------------------------------------------------------- #
def test_an_overflow_with_no_draft_to_drop_still_raises():
    """A bare prompt that does not fit will not fit on a retry.

    `prompt_no_room` is absent from REROLLABLE_PHASES forever because the
    arithmetic that refused it is deterministic. The guard is legitimate only
    because dropping the draft asks a DIFFERENT question; with no draft to drop
    there is no different question, so this must stay fatal rather than loop.
    """
    writer = ScriptedWriter([_overflow()])
    with pytest.raises(PromptContextOverflowError):
        run_ladder(writer)
    assert len(writer.prompts) == 1, "it must not retry a bare over-long prompt"


def test_a_degenerate_decode_is_not_swallowed_by_the_overflow_guard():
    """`GenerationDegeneracyError` SUBCLASSES PromptContextOverflowError.

    It means the transport halted a decode that had stopped steering -- a
    different condition entirely, and not this guard's to absorb. A guard that
    catches the base class without checking the phase would silently turn a
    halted decode into a cold regeneration.
    """
    writer = ScriptedWriter([
        play("*SFX: a door slams"),
        GenerationDegeneracyError(
            "decode stopped steering",
            phase=CAPACITY_PHASE_DECODE_DEGENERACY,
        ),
        play(),
    ])
    with pytest.raises(GenerationDegeneracyError):
        run_ladder(writer)


def test_the_happy_path_is_untouched():
    """No overflow, no behaviour change: the guard is exception-only."""
    writer = ScriptedWriter([play()])
    raw, parsed, _diag = run_ladder(writer)
    assert parsed is not None and raw
    assert len(writer.prompts) == 1
    assert DRAFT_MARKER not in writer.prompts[0]
