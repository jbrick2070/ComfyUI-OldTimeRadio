"""A repair turn must carry the draft it is repairing.

FOUND BY THE r1 REVIEW ARC (2026-08-12, Fable + Codex + Antigravity, driver
judged). The `scifi_news_pro` leg had died three times running, each time with a
DIFFERENT malformed shape -- invented speakers, then prose-as-dialogue, then
screenplay action lines with abbreviated cues.

THE ROOT. The retry turn was `base_user` + *"Repair only the malformed FORMAT
defects below... Keep the same story, cast, events, and wording wherever the
format is already valid"* + the defect list. **The rejected draft was never
included.** The model was ordered to preserve wording it had never been shown,
and handed only a list of complaints about an invisible text. So every attempt
after the first was a COLD REGENERATION -- which is exactly why four attempts
diverged into four different failures instead of converging on one repaired
script. A model failing to obey a rule repeats itself; a model regenerating
blind wanders.

THE PROOF THAT THIS WAS NEVER INTENDED is in the module's own docstring:
`_MARKUP_LADDER_TEMPS` decays to 0.30 and describes that rung as *"repeats 0.30
WITH THE DEFECT QUOTE"*. The temperature was lowered toward determinism FOR a
repair context the code never supplied. Near-deterministic sampling is right for
surgical repair and actively harmful for regeneration: a cold rung at 0.30
re-derives the same wrong structure it just produced.

So two things change together, and the second is not decoration: the draft rides
along, and the temperature decays ONLY when it did.
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
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.temps = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.prompts.append(list(messages)[-1]["content"])
        self.temps.append(temperature)
        return self.replies.pop(0)


def run_ladder(writer, temperature=0.75):
    return scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=CAST,
        initial_temperature=temperature,
    )


#: The exact shape that killed the live leg.
BAD = play("Johannes Lachner enters, determination in his eyes.")


# ---------------------------------------------------------------------------
# THE FIX
# ---------------------------------------------------------------------------
def test_the_second_prompt_CONTAINS_the_rejected_draft():
    """THE DEFECT, inverted. Without this the model is told to preserve wording
    it cannot see."""
    writer = ScriptedWriter([BAD, play()])
    run_ladder(writer)
    second = writer.prompts[1]
    assert "REJECTED DRAFT" in second
    assert "Johannes Lachner enters, determination in his eyes." in second, (
        "the repair turn does not carry the draft it is asking the model to "
        "repair -- every retry is a cold regeneration")


def test_the_FIRST_prompt_carries_no_draft():
    """There is nothing to repair on attempt 1, and a stray marker there would
    imply the model had already written something."""
    writer = ScriptedWriter([BAD, play()])
    run_ladder(writer)
    assert "REJECTED DRAFT" not in writer.prompts[0]


def test_the_draft_is_delimited_so_the_model_can_tell_it_from_the_assignment():
    writer = ScriptedWriter([BAD, play()])
    run_ladder(writer)
    second = writer.prompts[1]
    assert "-----BEGIN REJECTED DRAFT-----" in second
    assert "-----END REJECTED DRAFT-----" in second
    assert second.index("BEGIN REJECTED DRAFT") < second.index("END REJECTED DRAFT")


def test_each_retry_carries_the_MOST_RECENT_draft_not_the_first():
    """Attempt 3 must repair attempt 2's text. Carrying a stale draft would ask
    the model to fix something it has already moved past."""
    first_bad = play("Johannes Lachner enters, determination in his eyes.")
    second_bad = play("(a door slams)")
    writer = ScriptedWriter([first_bad, second_bad, play()])
    run_ladder(writer)
    third = writer.prompts[2]
    assert "(a door slams)" in third
    assert "determination in his eyes" not in third


def test_the_defect_list_still_travels_with_the_draft():
    """The draft alone does not say what is wrong with it."""
    writer = ScriptedWriter([BAD, play()])
    run_ladder(writer)
    second = writer.prompts[1]
    assert "Repair only the malformed FORMAT defects below" in second
    assert "BAD_LINE_SHAPE" in second


# ---------------------------------------------------------------------------
# The temperature corollary -- decay only when the draft went with it
# ---------------------------------------------------------------------------
def test_temperature_DECAYS_when_the_draft_was_carried():
    """Near-deterministic sampling is correct for surgical repair."""
    writer = ScriptedWriter([BAD, BAD, play()])
    run_ladder(writer, temperature=0.75)
    assert writer.temps[0] == 0.75
    assert writer.temps[1] < writer.temps[0], writer.temps


def test_temperature_HOLDS_when_the_draft_could_not_be_carried(monkeypatch):
    """A cold rung at 0.30 re-derives the structure it just produced -- the
    current failure mode. If the draft cannot ride, the retry must at least be
    free to explore."""
    monkeypatch.setattr(scifi_news_pro, "_draft_fits_repair_turn",
                        lambda base_user, draft, system="": False)
    writer = ScriptedWriter([BAD, BAD, play()])
    run_ladder(writer, temperature=0.75)
    assert writer.temps[1] == 0.75, (
        "temperature decayed on a COLD regeneration: %s" % (writer.temps,))
    assert "REJECTED DRAFT" not in writer.prompts[1]


# ---------------------------------------------------------------------------
# The budget guard
# ---------------------------------------------------------------------------
def realistic_prompt_and_draft(words):
    """A `base_user`-sized prompt and a draft of `words` spoken words.

    SIZED FROM THE REAL THING, because the first version of this test was the
    L26 trap in its purest form: it asserted a 400-character toy play fitted
    beside a 17-character prompt, passed, and hid a guard that dropped the
    draft for every full-length episode. Measured shapes: `_script_user_prompt`
    runs ~5-6k characters with the digest at its `_DIGEST_CHAR_CAP` of 3600,
    and a 1520-word draft (the documented structural ceiling) is ~9.2k.
    """
    base_user = "x" * 6112
    line = "Ada: We measured it twice and the second reading was worse.\n"
    per_line_words = 10
    draft = line * max(1, words // per_line_words)
    return base_user, draft


@pytest.mark.parametrize("words", [45, 400, 800, 1200, 1520])
def test_a_REAL_episode_draft_fits_at_every_length_we_ship(words):
    """THE TEST THAT SHOULD HAVE CAUGHT THE FIRST GUARD. 1520 words is the
    documented structural ceiling -- the top of the normal operating range, not
    an edge case. Dropping the draft there silently restores the exact
    cold-regeneration bug this change exists to fix."""
    base_user, draft = realistic_prompt_and_draft(words)
    assert scifi_news_pro._draft_fits_repair_turn(base_user, draft), (
        "a %d-word episode draft (%d chars) was DROPPED beside a %d-char "
        "prompt -- the ladder silently reverts to cold regeneration"
        % (words, len(draft), len(base_user)))


def test_the_guard_still_budgets_for_the_REPLY_not_just_the_prompt():
    """A repair turn that fits the prompt but leaves no room to write the
    corrected episode back has not actually helped."""
    base_user, draft = realistic_prompt_and_draft(1520)
    cap = 8192
    prompt_tokens = (len(base_user) + len(draft)) / scifi_news_pro._CHARS_PER_TOKEN
    reply_tokens = (len(draft) / scifi_news_pro._CHARS_PER_TOKEN) * scifi_news_pro._REPAIR_REPLY_MARGIN
    assert (prompt_tokens + reply_tokens) < cap, (
        "prompt+reply do not fit the window at the structural ceiling")


def test_an_absurdly_long_draft_is_DROPPED():
    """Better one cold retry than a truncated prompt."""
    assert not scifi_news_pro._draft_fits_repair_turn("base", "x" * 500_000)


def test_an_empty_draft_is_not_carried():
    assert not scifi_news_pro._draft_fits_repair_turn("base", "")


def test_the_temperature_sequence_is_NON_INCREASING_even_when_a_draft_drops():
    """`_MARKUP_LADDER_TEMPS` documents "the markup ladder NEVER raises
    temperature". The first fix broke that: dropping the draft on attempt 2
    reset to the opening temperature and produced 0.75, 0.49, 0.75, 0.30 -- a
    rise, and a last-chance attempt at full exploration."""
    calls = {"n": 0}

    def only_second_drops(base_user, draft, system=""):
        calls["n"] += 1
        return calls["n"] != 1          # attempt 2's draft is dropped

    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    monkey.setattr(scifi_news_pro, "_draft_fits_repair_turn", only_second_drops)
    try:
        writer = ScriptedWriter([BAD, BAD, BAD, play()])
        run_ladder(writer, temperature=0.75)
    finally:
        monkey.undo()

    temps = writer.temps
    assert temps == sorted(temps, reverse=True), (
        "temperature rose across the ladder: %s" % (temps,))
    assert temps[-1] <= temps[0]


# ---------------------------------------------------------------------------
# Nothing else regressed
# ---------------------------------------------------------------------------
def test_a_clean_first_attempt_still_costs_exactly_one_call():
    writer = ScriptedWriter([play()])
    _raw, parsed, _diag = run_ladder(writer)
    assert parsed is not None
    assert len(writer.prompts) == 1


def test_an_unrepairable_model_still_fails_CLOSED():
    """Carrying the draft must not make an invalid script valid."""
    writer = ScriptedWriter([BAD] * 8)
    with pytest.raises(scifi_news_pro.NewsProScriptError):
        run_ladder(writer)
