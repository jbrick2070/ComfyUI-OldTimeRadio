"""Best-of-N structural story-refine selector (2026-06-23).

Local-only (default), opt-in remote, deterministic best-of-N OUTLINE selector.
This file grows one chunk at a time:

  Chunk 1 -- OutlineRequest.diversity_hint + _build_user_prompt render
             (flag-off / empty-hint => byte-identical prompt).
  Chunk 2 -- score_outline pure scorer + StoryScore (raw-intent metrics).
  Chunk 3 -- select_best_outline selector + flag parse + provider gate.
  Chunk 4 -- optional remote best-of-N + fail-closed cost guard.

Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_outline as OUT  # noqa: E402
from nodes import _otr_story_select as SEL  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a valid OutlineRequest (budget is required by the v2.0 contract).
# ---------------------------------------------------------------------------
def _req(diversity_hint=None):
    from nodes import _otr_episode_budget as EB
    budget = EB.compute_episode_budget(
        target_words=400,
        act_count=EB.default_act_count(400),
        include_act_breaks=True,
        num_characters=2,
    )
    kwargs = dict(
        news_seed="A deep-space signal is detected near a dying star.",
        style="hard sci-fi procedural",
        target_words=400,
        character_cast=("MALI", "MANFRED"),
        budget=budget,
    )
    if diversity_hint is not None:
        kwargs["diversity_hint"] = diversity_hint
    return OUT.OutlineRequest(**kwargs)


# ---------------------------------------------------------------------------
# Chunk 1 -- diversity_hint field + prompt render
# ---------------------------------------------------------------------------
class TestDiversityHint:
    def test_field_defaults_to_empty(self):
        assert _req().diversity_hint == ""

    def test_empty_hint_prompt_is_byte_identical_to_default(self):
        # Field defaulted vs explicitly "" must produce the SAME prompt, and
        # neither may carry the structural-variation overlay. This is the
        # byte-identical guarantee for candidate 0 / every non-selector call.
        defaulted = OUT._build_user_prompt(_req())
        explicit_empty = OUT._build_user_prompt(_req(""))
        assert defaulted == explicit_empty
        assert "Structural variation" not in defaulted

    def test_whitespace_only_hint_is_treated_as_empty(self):
        # The render strips; a whitespace-only hint must not perturb the prompt.
        assert OUT._build_user_prompt(_req("   ")) == OUT._build_user_prompt(_req(""))

    def test_nonempty_hint_is_rendered_verbatim(self):
        hint = "open on the personal stake, not the institutional threat"
        prompt = OUT._build_user_prompt(_req(hint))
        assert "Structural variation" in prompt
        assert hint in prompt

    def test_nonempty_hint_only_appends(self):
        # The hinted prompt is the empty prompt plus the appended overlay block:
        # every line of the empty prompt is still present, in order.
        empty = OUT._build_user_prompt(_req(""))
        hinted = OUT._build_user_prompt(_req("vary which stake opens the story"))
        assert empty != hinted
        assert hinted.startswith(empty.split("\nBuild a dramatic outline")[0])
        assert "vary which stake opens the story" in hinted

    def test_different_hints_produce_different_prompts(self):
        a = OUT._build_user_prompt(_req("open on the turn"))
        b = OUT._build_user_prompt(_req("open on the consequence"))
        assert a != b


# ---------------------------------------------------------------------------
# Chunk 2 -- score_outline pure scorer
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc  # noqa: E402


@_dc
class _FakeBeat:
    speaker_role: str
    intent: str = ""


@_dc
class _FakeOutline:
    premise: str
    beats: list


_ROSTER = ("MALI", "MANFRED")


def _generic_outline():
    # A "console standoff": intents stuffed with GENERIC crisis nouns that are
    # NOT in the premise palette; nothing references the premise.
    return _FakeOutline(
        premise="Two engineers stand in a control room.",
        beats=[
            _FakeBeat("music_open", "theme plays"),
            _FakeBeat("character", "the reactor console overloads as the countdown begins"),
            _FakeBeat("character", "they grab the lever and the failsafe switch"),
            _FakeBeat("character", "the gauge climbs toward meltdown"),
            _FakeBeat("music_close", "theme fades"),
        ],
    )


def _grounded_outline():
    # Premise-anchored: intents reference premise/roster nouns; no generic
    # crisis nouns at all.
    return _FakeOutline(
        premise="A district adopts a tutoring algorithm whose grading weights are disputed.",
        beats=[
            _FakeBeat("music_open", "theme plays"),
            _FakeBeat("character", "Mali questions the tutoring algorithm grading weights"),
            _FakeBeat("character", "Manfred defends the district adoption decision"),
            _FakeBeat("character", "the disputed weights reshape a flagged transcript"),
            _FakeBeat("music_close", "theme fades"),
        ],
    )


class TestScoreOutline:
    def test_returns_storyscore(self):
        s = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        assert isinstance(s, SEL.StoryScore)

    def test_deterministic(self):
        o = _grounded_outline()
        assert SEL.score_outline(o, {}, _ROSTER) == SEL.score_outline(o, {}, _ROSTER)

    def test_ungrounded_crisis_nonzero_on_raw_intents(self):
        # Verify-at-build #2: the metric MUST discriminate on raw intents.
        s = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        assert s.ungrounded_crisis_density > 0.0

    def test_grounded_beats_generic_on_every_axis(self):
        g = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        p = SEL.score_outline(_grounded_outline(), {}, _ROSTER)
        assert p.ungrounded_crisis_density < g.ungrounded_crisis_density
        assert p.premise_grounding > g.premise_grounding
        assert p.distinct_conflict_nouns > g.distinct_conflict_nouns

    def test_grounded_outline_density_zero_grounding_full(self):
        p = SEL.score_outline(_grounded_outline(), {}, _ROSTER)
        assert p.ungrounded_crisis_density == 0.0
        assert p.premise_grounding == pytest.approx(1.0)

    def test_announcer_counts_as_voiced(self):
        # An announcer beat with a generic crisis noun must contribute (proves
        # _is_voiced includes announcer, mirroring build_sq_data's scope).
        o = _FakeOutline(
            premise="A quiet town.",
            beats=[_FakeBeat("announcer", "a reactor meltdown looms")],
        )
        s = SEL.score_outline(o, {}, ())
        assert s.ungrounded_crisis_density > 0.0

    def test_no_voiced_beats_is_zero_not_crash(self):
        # Division-by-zero guards: only non-voiced beats => clean zeros.
        o = _FakeOutline(
            premise="A countdown reactor meltdown lever.",
            beats=[_FakeBeat("music_open", "x"), _FakeBeat("sfx", "y")],
        )
        s = SEL.score_outline(o, {}, ())
        assert s.ungrounded_crisis_density == 0.0
        assert s.premise_grounding == 0.0
        assert s.distinct_conflict_nouns == 0

    def test_pure_no_mutation_of_intents(self):
        o = _generic_outline()
        before = [b.intent for b in o.beats]
        SEL.score_outline(o, {}, _ROSTER)
        assert [b.intent for b in o.beats] == before

    def test_empty_outline_no_crash(self):
        s = SEL.score_outline(_FakeOutline(premise="", beats=[]), {}, ())
        assert s.ungrounded_crisis_density == 0.0
        assert s.distinct_conflict_nouns == 0
        assert s.premise_grounding == 0.0
