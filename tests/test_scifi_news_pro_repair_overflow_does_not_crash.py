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
the draft and continue without it. The overflow CONSUMES its rung -- one counted
call, one trace row -- because two layers up the runner asserts
``box["calls"] == len(attempt_trace)`` and raises on drift; see the last test here,
which is the one that would have caught the first, broken version of this guard.

These tests pin the behaviours that matter, including the two where the guard must
NOT fire.
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

    The overflow CONSUMES its rung -- one counted call, one trace row -- and the
    NEXT rung regenerates cold without the draft, delivering the clean play.
    Before the guard this raised straight out of the node.
    """
    writer = ScriptedWriter([play("*SFX: a door slams"), _overflow(), play()])

    raw, parsed, diag = run_ladder(writer)

    assert parsed is not None and raw, "the episode must still be delivered"
    assert len(writer.prompts) == 3, writer.prompts
    # the prompt that overflowed carried the draft...
    assert DRAFT_MARKER in writer.prompts[1]
    # ...and the rung after it did not.
    assert DRAFT_MARKER not in writer.prompts[2], (
        "the next rung resent the draft that just overflowed -- that is the "
        "banned re-roll of a deterministic prompt_no_room refusal")
    # the trace must SHOW the cold regeneration, not hide it
    # EXACTLY one, not ">= 1". The loose form passed while the overflow
    # handler and the next rung were BOTH counting the same cold regeneration
    # -- a weak assertion is how a miscount survives its own test.
    assert diag["cold_regenerations"] == 1, diag


def test_the_rung_after_an_overflow_does_not_raise_temperature():
    """Dropping the draft must not silently escalate the ladder's temperature.

    `_MARKUP_LADDER_TEMPS` documents that the markup ladder NEVER raises
    temperature. The rung after an overflow carries no draft, so it takes the
    ladder's existing cold-regeneration branch -- `temp = max(temp, last_temp)`
    -- which HOLDS the previous rung rather than resetting to the opening
    temperature. This asserts the observed effect, not the mechanism.
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
        "the rung after the overflow raised temperature: %r -- the markup "
        "ladder never raises" % (seen,))


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


# --------------------------------------------------------------------------- #
# THE INVARIANT MY FIRST FIX BROKE, and which these tests did not catch.
#
# The first version of the overflow guard retried in place -- a SECOND
# creative_fn call inside the same rung. Every test above still passed, because
# they drive _run_markup_ladder directly. But two layers up the runner wraps the
# creative slot in `_counting` (which increments per INVOCATION, before the
# underlying call) and then asserts:
#
#     if box["calls"] != len(p3_attempts):
#         raise NewsProScriptError("script", "P3 attempt/call count drift")
#
# Two calls against one trace row is exactly that drift, so the "fix" turned one
# crash into a different, later crash. The trace cannot absorb an extra row
# either: PassAttemptTrace.outcome is Literal["parse_rejected","accepted"] and
# _validate_attempt_sequence demands attempts contiguous from 1.
#
# So the rule is ONE COUNTED CALL PER TRACE ROW, and this is the test that
# would have caught it.
# --------------------------------------------------------------------------- #
def test_every_counted_call_has_exactly_one_trace_row(tmp_path):
    """Drive the ladder through the same counting wrapper the runner uses."""
    counted = {"calls": 0}
    inner = ScriptedWriter([play("*SFX: a door slams"), _overflow(), play()])

    def counting_fn(messages, *, temperature, max_new_tokens):
        counted["calls"] += 1          # increments on INVOCATION, like _counting
        return inner(messages, temperature=temperature,
                     max_new_tokens=max_new_tokens)

    _raw, parsed, diag = scifi_news_pro._run_markup_ladder(
        counting_fn,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=CAST,
        initial_temperature=0.7,
    )

    assert parsed is not None
    traces = diag["attempt_trace"]
    assert counted["calls"] == len(traces), (
        "P3 attempt/call count drift: %d counted calls against %d trace rows. "
        "The runner raises NewsProScriptError on exactly this mismatch, so a "
        "second in-rung call is a crash, not a fix."
        % (counted["calls"], len(traces)))
    # and the trace stays well-formed for _validate_attempt_sequence
    assert [t.attempt for t in traces] == list(range(1, len(traces) + 1))
    assert sum(1 for t in traces if t.selected) == 1
    assert traces[-1].selected

