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
    # RE-POINTED 2026-08-24. This asserted the phrase "sound-effect speaker",
    # which sat beside an invented worked example -- "a row like '*SFX: a door
    # slams'" -- inside a string RETURNED INTO THE PROMPT. Operator: "there
    # should be no SFX"; the subsystem was ripped twice over, so naming its
    # token to a model taught it a form nothing can render (Bug Bible 12.132).
    # The rule now says the same thing without inventing a token, and the
    # model's OWN row above is the evidence, which teaches the shape better.
    assert "only the cast and the announcer can speak" in second
    # NOTE: no "a door slams" assertion here on purpose. The second PROMPT
    # legitimately contains that string -- it carries the rejected draft, and
    # the draft is the model's own text. The claim that the RULE invents no
    # cue example is asserted where it can actually be isolated, against the
    # note itself, in test_scifi_news_pro_speaker_resolution.py.
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
def test_a_decorated_CAST_NAME_now_COSTS_NO_REPAIR_ATTEMPT_AT_ALL():
    """RE-DERIVED 2026-08-24 (PBUG-20260824-01) and strictly better.

    This used to prove the SECOND prompt carried a restore-the-label rule --
    i.e. that a real cast member wearing a stray marker cost a whole repair
    attempt and then needed the model to comply. The shared speaker resolver
    reads `*Ada:` as Ada on the FIRST pass, so there is no second prompt to
    inspect: the attempt is not spent, and no model compliance is required.

    A rule that never has to fire beats a rule that fires correctly.
    """
    writer = ScriptedWriter([play("*Ada: We continue the work."), play()])
    _raw, parsed, _diag = run_ladder(writer)

    assert len(writer.prompts) == 1, (
        "a decorated real cast name must now parse on the first attempt")
    assert parsed is not None
    spoken = [ln.speaker for scene in parsed.scenes for ln in scene.lines]
    assert "Ada" in spoken


def test_an_ordinary_unknown_name_IS_TOLD_THE_LEGAL_LABELS():
    """CHANGED DELIBERATELY 2026-08-24. It used to assert the note stays
    silent for an undecorated unknown name, on the reasoning that inventing
    advice could lose a line.

    Silence is not neutral: it hands back the bare generic instruction, which
    is the exact input measured burning four attempts on live legs twice
    (PBUG-20260812-03, then PBUG-20260824-01). The advice now names the legal
    labels, and the guard the old rule actually cared about -- never tell the
    model to delete or fold a character's line -- is asserted here explicitly.
    """
    writer = ScriptedWriter([play("Adda: We continue the work."), play()])
    _raw, parsed, _diag = run_ladder(writer)

    second = writer.prompts[1]
    assert "Repair only the malformed FORMAT defects below" in second
    assert "is not one of this episode's characters" in second
    assert "'Adda'" in second
    assert "Ada" in second and "Bo" in second     # the exact legal labels
    # THE GUARD THAT MATTERED: never the advice that deletes a character.
    assert "omit it when nonessential" not in second
    assert "Do not delete the line" in second
    assert parsed is not None


# ---------------------------------------------------------------------------
# The failure the whole bug consisted of
# ---------------------------------------------------------------------------
def test_a_model_that_never_repairs_exhausts_the_ladder_then_SALVAGES():
    """POLICY CHANGED 2026-08-24 BY THE OPERATOR: *"accepts sometimes a wrong
    name populated but shouldn't kill the whole episode."*

    The old rule -- fail closed after four bad answers -- was written when
    refusing looked free. The overnight measurement priced it: this lane
    refused an episode on 6 of 10 passes. So the ladder still spends every
    attempt trying for a clean parse, and then delivers instead of dying.

    THE SOUND CUE IS NOT CAST AS A PERSON, and that is the part worth
    pinning. `*SFX` resolves to nobody and is decorated, so salvage reads it
    as a stage direction and DROPS the row. Adopting it would have handed a
    door slam a speaking voice.
    """
    writer = ScriptedWriter([play("*SFX: a door slams")] * 8)
    _raw, parsed, diag = run_ladder(writer)

    # Exhaustion still happened -- and every retry carried a real rule, so it
    # was the model's failure and not a missing instruction's.
    assert len(writer.prompts) == 4
    for prompt in writer.prompts[1:]:
        assert "FORMAT REPAIR RULE" in prompt
    # Then salvage delivered, and said so.
    assert diag["salvaged"] is True
    assert parsed is not None
    assert "*SFX" not in diag["salvage_adopted_speakers"], (
        "a sound cue must never be adopted as a speaking character")
    assert any("*SFX" in row for row in diag["salvage_dropped_rows"])
    spoken = {ln.speaker for scene in parsed.scenes for ln in scene.lines}
    assert spoken == {"Ada", "Bo"}, (
        f"only the real cast may speak in a salvaged episode; got {spoken}")
