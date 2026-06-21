"""Bare leading stage-direction scrub + detector (roundtable-converged).

Pins the FULL panel test corpus (3-pass roundtable, docs/2026-06-22-stage-
direction-leak/): the destructive floor strips ONLY the high-confidence
unambiguous cases and KEEPS every false-positive counterexample the panel
raised. Pure / CPU. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import (  # noqa: E402
    detect_leading_stage_business,
    scrub_leading_stage_direction,
)

# (input, expected) -- the destructive floor STRIPS these to the dialogue.
_STRIP = [
    ("twirls his pen nervously Look, Pinky, we've been through this.",
     "Look, Pinky, we've been through this."),
    ("pauses, sets pen down Alright, Pinky. We've had a breach.",
     "Alright, Pinky. We've had a breach."),
    ("clenches jaw You mean like the one you assured me couldn't happen?",
     "You mean like the one you assured me couldn't happen?"),
    ("glances at Pinky We need a plan.", "We need a plan."),
    ("looks at the Map We should go.", "We should go."),   # article-object skip
    ("sighs No.", "No."),                                   # short-utterance
]

# Lines the floor MUST leave UNCHANGED (every panel false-positive guard).
_KEEP = [
    "looks can be deceiving, John.",            # 2nd token copula -> subject
    "pauses are evidence, Brain, not proof.",   # 2nd token copula
    "look, Pinky, we've been through this",     # dialogue-starter "look"
    "maybe we should ask John and Mary.",       # pronoun "we" + starter "maybe"
    "looks like rain. We should go.",           # terminal punct in lead
    "looks at Pinky and Brain We need a plan.",  # conjunction-object -> abort
    "sighs No",                                  # short utt, no terminal punct
    "You mean like the one you assured me couldn't happen?",  # already clean
    "We need a plan.",                           # already capitalized
    "Alright, Pinky. We've had a breach.",
    "I'll take responsibility, Pinky.",
    "",                                          # empty
    "(sighs)",                                   # delimited-only -> not our job
]


class TestScrubStrips:
    @pytest.mark.parametrize("raw,expected", _STRIP)
    def test_strips_to_dialogue(self, raw, expected):
        assert scrub_leading_stage_direction(raw) == expected


class TestScrubKeeps:
    @pytest.mark.parametrize("raw", _KEEP)
    def test_keeps_unchanged(self, raw):
        assert scrub_leading_stage_direction(raw) == raw


class TestIdempotent:
    @pytest.mark.parametrize("raw", [r for r, _ in _STRIP] + _KEEP)
    def test_idempotent(self, raw):
        once = scrub_leading_stage_direction(raw)
        assert scrub_leading_stage_direction(once) == once


class TestNeverRaises:
    @pytest.mark.parametrize("bad", [None, 123, [], {"a": 1}])
    def test_defensive_inputs(self, bad):
        # never raises; returns a string
        out = scrub_leading_stage_direction(bad)
        assert isinstance(out, str)


class TestDetector:
    @pytest.mark.parametrize("raw", [r for r, _ in _STRIP])
    def test_detects_strip_cases(self, raw):
        hit, hint = detect_leading_stage_business(raw)
        assert hit is True
        assert hint  # non-empty reroll directive

    @pytest.mark.parametrize("raw", _KEEP)
    def test_does_not_flag_clean_dialogue(self, raw):
        hit, hint = detect_leading_stage_business(raw)
        assert hit is False
        assert hint == ""

    def test_detector_broader_than_floor_on_long_lead(self):
        # a 7-word impersonal lead: the floor (max 6) keeps it, the detector
        # (max 10) flags it for reroll.
        raw = "twirls his pen and taps the desk twice You mean it?"
        assert scrub_leading_stage_direction(raw) == raw     # floor: too long
        hit, _ = detect_leading_stage_business(raw)
        assert hit is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
