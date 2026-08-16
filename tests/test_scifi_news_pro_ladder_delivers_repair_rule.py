"""PBUG-20260812-03 -- prove the repair rule REACHES THE MODEL, end to end.

Every other test around this defect checks what
`_standalone_stage_direction_repair_note` RETURNS. None of them proves the
returned string is ever put in front of the model, and that is the only property
the live failure actually turned on: the note was computed, returned "", and the
writer burned four attempts. A note that returns perfect text into a prompt
nobody sends is the same outage.

So this drives the real `_run_markup_ladder` with a scripted `creative_fn`:

* attempt 1 answers with the exact shape that killed the `viz_green` leg
  (`*SFX: a door slams`);
* attempt 2 answers with a clean play.

Then it asserts on the SECOND prompt -- the one the repair rung actually built.

This is the test the kibitz panel asked for (Codex MUST-FIX 5,
`kibitz-runs/2026-08-12-writer-stage-direction-note/r2/`), and it is the answer
to lesson L26: it is behavioural, it would fail if the note were computed and
dropped, and it would fail if the codes were compared as strings against a plain
enum -- the silent never-fires trap that a returns-the-right-string test cannot
see.
"""
from __future__ import annotations

import pytest

from nodes._otr_scifi_news_pro_markup import ANNOUNCER_NAME
from nodes import _otr_scifi_news_pro as scifi_news_pro


CAST = ["Ada", "Bo"]


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
    """Answers with a queued reply per call and records every prompt sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        # The user turn is the last message; that is what the ladder assembles.
        self.prompts.append(list(messages)[-1]["content"])
        return self.replies.pop(0)


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


# ---------------------------------------------------------------------------
# The live shape
# ---------------------------------------------------------------------------
def test_the_stage_direction_rule_REACHES_the_second_prompt():
    writer = ScriptedWriter([play("*SFX: a door slams"), play()])
    raw, parsed, _diag = run_ladder(writer)

    assert len(writer.prompts) == 2, "the first attempt should have been rejected"
    first, second = writer.prompts

    assert "FORMAT REPAIR RULE" not in first, (
        "the rule leaked into the FIRST prompt -- it is a repair instruction")
    assert "FORMAT REPAIR RULE" in second, (
        "THE LIVE DEFECT: the repair rung never received the targeted rule, so "
        "the model gets only the generic instruction and re-emits the same "
        "shape until the ladder exhausts")
    assert "*SFX" in second, "the rule must quote the parser's own evidence"
    assert "sound-effect speaker" in second
    assert parsed is not None and raw


def test_the_ladder_ACCEPTS_the_repaired_second_answer():
    """The rule is only worth delivering if a repaired reply is then taken."""
    writer = ScriptedWriter([play("*SFX: a door slams"), play()])
    _raw, parsed, diag = run_ladder(writer)
    assert parsed is not None
    assert [ln.speaker for ln in parsed.scenes[0].lines] == ["Ada", "Bo"]
    assert len(diag["defects_by_attempt"]) == 1


def test_telemetry_still_carries_IMMUTABLE_STRING_defects():
    """The note now takes typed defects, but `PassAttemptTrace.parse_defects`
    is declared `tuple[str, ...]` and its `__post_init__` raises otherwise. This
    pins that the two representations stayed separate."""
    writer = ScriptedWriter([play("*SFX: a door slams"), play()])
    _raw, _parsed, diag = run_ladder(writer)
    trace = diag["attempt_trace"][0]
    assert isinstance(trace.parse_defects, tuple)
    assert trace.parse_defects, "the rejected attempt recorded no defects"
    assert all(isinstance(row, str) for row in trace.parse_defects)
    assert any("*SFX" in row for row in trace.parse_defects)


# ---------------------------------------------------------------------------
# The decorated-real-name shape gets the OTHER rule, also end to end
# ---------------------------------------------------------------------------
def test_a_decorated_CAST_NAME_reaches_the_model_as_a_RESTORE_instruction():
    writer = ScriptedWriter([play("*Ada: We continue the work."), play()])
    _raw, parsed, _diag = run_ladder(writer)

    assert len(writer.prompts) == 2
    second = writer.prompts[1]
    assert "Restore the plain canonical label" in second
    assert "KEEP THE DIALOGUE EXACTLY AS WRITTEN" in second
    # The delete-a-character advice must not be what a real cast member gets.
    assert "omit it when nonessential" not in second
    assert parsed is not None


def test_an_ordinary_unknown_name_gets_only_the_GENERIC_instruction():
    """The note stays out of the way when it has nothing safe to say."""
    writer = ScriptedWriter([play("Adda: We continue the work."), play()])
    _raw, parsed, _diag = run_ladder(writer)

    second = writer.prompts[1]
    assert "Repair only the malformed FORMAT defects below" in second
    assert "FORMAT REPAIR RULE" not in second
    assert "Restore the plain canonical label" not in second
    assert parsed is not None


# ---------------------------------------------------------------------------
# The failure the whole bug consisted of
# ---------------------------------------------------------------------------
def test_a_model_that_never_repairs_exhausts_the_ladder_and_RAISES():
    """Four identical bad answers must still fail closed -- the fix makes the
    rung better informed, it does not make an invalid script valid."""
    writer = ScriptedWriter([play("*SFX: a door slams")] * 8)
    with pytest.raises(scifi_news_pro.NewsProScriptError) as caught:
        run_ladder(writer)
    assert "markup ladder exhausted" in str(caught.value)
    # ...and every retry after the first carried the rule, so the exhaustion is
    # the model's, not a missing instruction's.
    for prompt in writer.prompts[1:]:
        assert "FORMAT REPAIR RULE" in prompt
