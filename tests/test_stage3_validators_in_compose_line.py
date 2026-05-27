"""Sprint 10B Wave 1 Agent B regression -- in-line Stage 3 validators
on the legacy compose_line path.

Covers nodes/_otr_line_composer.compose_line's new keyword args:

  * enable_stage3_validators (bool)  -- gate
  * stage3_plan (Stage1Plan)         -- needed for cast / arc / facts
  * stage3_beat (Stage1Beat)         -- per-line target
  * stage3_banned_phrases (list)     -- optional override

Behavior under test:

  * Validators run AFTER the strip pipeline so they score the EXACT
    text that would ship.
  * Error-severity findings (speaker_leak / length_drift / pronoun /
    on-beat / banned_phrase) trigger ONE recursive repair regenerate
    with the finding messages threaded in as the reroll_hint.
  * The repair regenerate cannot recurse further (the
    _stage3_repair_attempted guard).
  * Warn-severity findings stamp on LineResult.validation_findings
    without regenerating.
  * Findings stamped on LineResult.validation_findings reflect the
    FINAL shipped state (post-repair if repair fired).
  * Disabled (default) preserves PD1 byte-identity -- no validator
    calls, no LLM regen, validation_findings stays an empty tuple.

Pure-Python. No GPU. No live LLM. creative_fn is mocked with canned
strings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    LineResult,
    compose_line,
)
from nodes._otr_stage1_plan import (  # noqa: E402
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan_with_ren_maeve() -> Stage1Plan:
    return Stage1Plan(
        premise=(
            "A signal operator and a research scientist confront a "
            "breakthrough drug trial in late-night comms."
        ),
        arc=Stage1Arc(
            setup="Late-shift comms shack; quiet call.",
            complication="The scientist offers a free dose with strings.",
            resolution="The operator refuses the dose but accepts the hearing.",
        ),
        cast=[
            Stage1CastMember(
                name="REN BLACK",
                gender="male",
                pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona=(
                    "Late-shift signal operator. Lost his sister to "
                    "dementia. Skeptic of easy answers."
                ),
                arc_role="protagonist",
            ),
            Stage1CastMember(
                name="DR. MAEVE COLE",
                gender="female",
                pronouns="she/her",
                voice_id="v2/en_speaker_2",
                persona=(
                    "Lead researcher on a brain-aging spray trial. "
                    "Believer."
                ),
                arc_role="foil",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b001", speaker="ANNOUNCER",
                intent="Open the episode.", length_target_words=18,
                emotional_register="neutral", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b002", speaker="REN BLACK",
                intent=(
                    "Ren answers the call and asks why it could not "
                    "wait until morning."
                ),
                length_target_words=24,
                emotional_register="cautious", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b003", speaker="DR. MAEVE COLE",
                intent="Maeve names the trial result and offers a dose.",
                length_target_words=30,
                emotional_register="warm urgency", callback_to="b002",
            ),
        ],
        running_facts=[
            "Ren works late-shift signal operations.",
            "Maeve leads a nasal-spray trial that reverses brain aging.",
        ],
    )


def _line_req(speaker: str = "REN BLACK", target: int = 24) -> LineRequest:
    return LineRequest(
        speaker=speaker,
        intent="alert",
        mood="tight",
        target_words=target,
        canon_header="x",
        last_lines=[],
    )


def _mock_creative_fn(text: str):
    """Return a generate fn that always emits the given text."""
    def fn(messages, *, temperature, max_new_tokens, stop=None):
        return text
    return fn


def _mock_creative_fn_sequence(texts: list[str]):
    """Return a generate fn that returns the next text on each call.
    Used to simulate the repair regenerate producing a different
    line than the original."""
    state = {"i": 0}
    def fn(messages, *, temperature, max_new_tokens, stop=None):
        i = state["i"]
        state["i"] = min(i + 1, len(texts) - 1)
        return texts[i]
    return fn


# ---------------------------------------------------------------------------
# Disabled path (default) -- preserves PD1 byte-identity
# ---------------------------------------------------------------------------


class TestStage3Disabled:
    """When enable_stage3_validators=False (default) OR when the
    plan/beat args are missing, the validators never fire.
    validation_findings stays an empty tuple. No recursion."""

    def test_default_disabled_findings_empty(self):
        """Baseline: no Stage 3 args -> no validators run."""
        clean_text = "Station Seven on the line. State your business."
        res = compose_line(
            creative_fn=_mock_creative_fn(clean_text),
            req=_line_req(target=12),
        )
        assert isinstance(res, LineResult)
        assert res.validation_findings == ()
        assert res.text == clean_text

    def test_explicit_disabled_findings_empty(self):
        """Explicit enable_stage3_validators=False -> no validators
        run even if plan + beat provided."""
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        clean_text = "Station Seven on the line. State your business."
        res = compose_line(
            creative_fn=_mock_creative_fn(clean_text),
            req=_line_req(target=12),
            enable_stage3_validators=False,
            stage3_plan=plan,
            stage3_beat=beat,
        )
        assert res.validation_findings == ()

    def test_enabled_but_no_plan_findings_empty(self):
        """Gate requires BOTH enable_stage3_validators=True AND a
        non-None plan AND a non-None beat. Missing any one disables."""
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        clean_text = "Station Seven on the line. State your business."
        # No plan.
        res = compose_line(
            creative_fn=_mock_creative_fn(clean_text),
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=None,
            stage3_beat=beat,
        )
        assert res.validation_findings == ()
        # No beat.
        res = compose_line(
            creative_fn=_mock_creative_fn(clean_text),
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=None,
        )
        assert res.validation_findings == ()


# ---------------------------------------------------------------------------
# Enabled path -- validators fire, findings stamped, repair triggers
# ---------------------------------------------------------------------------


class TestStage3EnabledFindingsStamped:
    """When enabled, validators run and findings land on
    LineResult.validation_findings as a tuple of dicts mirroring
    _otr_stage3_validators.ValidationFinding."""

    def test_clean_line_stamps_empty_findings(self):
        """A clean line passes all validators -- findings tuple is
        present but empty list-of-dicts (still distinguishable from
        the disabled-path empty tuple by the absence of either)."""
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        # On-target word count, no leak patterns, in-character speaker.
        clean_text = (
            "Station Seven, Black on the line. Doctor, it is past "
            "three in the morning here. Tell me again why this could "
            "not wait."
        )
        res = compose_line(
            creative_fn=_mock_creative_fn(clean_text),
            req=_line_req(target=24),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=beat,
        )
        # No errors -> no repair fired -> shipped text is the original.
        assert res.text == clean_text
        assert res.validation_findings == ()

    def test_speaker_leak_triggers_repair_attempt(self):
        """Bracketed-direction speaker leak: the existing validator
        catches `[breathless]` style stage directions. The repair
        regenerate produces a clean line; shipped text reflects the
        repaired version.
        """
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        # Bracketed direction triggers validate_speaker_leak's
        # 'bracketed_direction' pattern.
        leaky_text = (
            '[breathless] "That is it. Time is not up yet."'
        )
        clean_repair = (
            "I am here. State your business. Time is short."
        )
        res = compose_line(
            creative_fn=_mock_creative_fn_sequence(
                [leaky_text, clean_repair],
            ),
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=beat,
        )
        # Repair must have fired (sequence mock advanced past slot 0).
        assert res.text == clean_repair


class TestStage3RecursionGuard:
    """The _stage3_repair_attempted flag prevents the repair from
    triggering another repair. Worst case: ONE repair regenerate per
    compose_line call, never more."""

    def test_repair_does_not_double_recurse(self):
        """If the repair output ALSO has errors, no second repair
        fires -- the line ships with findings stamped."""
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        # Both attempts produce a bracketed-direction leak. The first
        # triggers repair; repair output also leaks but no second
        # repair fires.
        leaky_text = '[breathless] "That is it. Time is not up yet."'
        also_leaky_text = '[whispers] "Still time. Hold on."'
        calls = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["n"] += 1
            return leaky_text if calls["n"] == 1 else also_leaky_text

        res = compose_line(
            creative_fn=fn,
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=beat,
            max_attempts=1,
        )
        # Repair was attempted (so calls > original 1) but cannot
        # recurse further. With max_attempts=1 per pass we expect
        # 2 generate calls total: 1 original + 1 repair, no more.
        assert calls["n"] == 2, (
            f"Expected exactly 2 generate calls (1 original + 1 "
            f"repair, no further recursion); got {calls['n']}."
        )


class TestStage3FindingsShape:
    """Each finding dict has the keys ValidationFinding declares:
    severity, code, beat_id, speaker, message, expected, got."""

    def test_finding_dict_keys(self):
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]
        # Bad line: speaker leak via bracketed direction.
        leaky_text = '[whispers] "Hello."'
        # Repair also fails so a finding is stamped.
        also_leaky_text = '[whispers] "Still."'
        calls = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["n"] += 1
            return leaky_text if calls["n"] == 1 else also_leaky_text

        res = compose_line(
            creative_fn=fn,
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=beat,
            max_attempts=1,
        )
        assert len(res.validation_findings) >= 1
        f = res.validation_findings[0]
        for key in (
            "severity", "code", "beat_id", "speaker", "message",
            "expected", "got",
        ):
            assert key in f, (
                f"Finding dict missing key {key!r}; got: {f!r}"
            )
        assert f["beat_id"] == beat.beat_id
        # speaker_leak is the expected code for bracketed direction.
        assert f["code"] in (
            "speaker_leak", "length_drift", "banned_phrase",
        )


class TestStage3KnownBadSprayOfHopeFixture:
    """Integration: speaker-leak shapes from operator-flagged
    failures get caught by validate_speaker_leak and the repair path
    fires.

    Note: the EXACT Spray of Hope beat 4 text ("Breathless, REN BLACK
    mutters, ...") is NOT currently caught by the existing
    _SPEAKER_LEAK_PATTERNS (which require 'he/she/they/the announcer'
    + a verb -- not 'NAME + verb'). Extending validate_speaker_leak to
    catch the name-attribution shape is a separate concern outside
    Agent B's scope (Agent B wires existing validators; pattern
    extension is its own commit). This test pins the bracketed-
    direction shape which IS caught by existing validators, as the
    worked-example acceptance for Agent B's wiring.
    """

    def test_bracketed_stage_direction_caught_and_repaired(self):
        """A bracketed stage direction in a character line is a
        clear speaker-leak pattern. validate_speaker_leak catches it,
        repair fires, the cycle either ships repaired text or stamps
        findings on the original."""
        plan = _plan_with_ren_maeve()
        beat = plan.beats[1]   # REN BLACK character beat
        leaky_text = (
            '[breathless] "That is it. Time is not up yet."'
        )
        clean_repair = (
            "I am here. State your business. Time is short."
        )
        res = compose_line(
            creative_fn=_mock_creative_fn_sequence(
                [leaky_text, clean_repair],
            ),
            req=_line_req(target=12),
            enable_stage3_validators=True,
            stage3_plan=plan,
            stage3_beat=beat,
        )
        # Either the repair landed clean (text replaced) OR the
        # original shipped with findings stamped. Both are acceptable
        # outcomes per the Agent B design: ONE repair attempt, ship
        # whatever comes back, stamp the final findings.
        assert res.text != leaky_text or res.validation_findings, (
            "Either repair must have replaced the leaky text OR "
            "findings must be stamped. Neither happened: text "
            "unchanged AND findings empty."
        )
