"""Story-Quality 3.2/3.3/3.6 gate-seam cluster (2026-06-27).

Covers the new _otr_line_hygiene detectors (anchor-stuffing, one-breath,
whole-line stage-action with the BN-1 guard, personal-cost boilerplate) and the
shared _quality_flags_for_line scorer wired into the compose_line clean-quality
gate -- v2 + character gated, with MF-1 guard threading. Pure / CPU (mock
creative_fn). UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import (  # noqa: E402
    extract_specificity_anchors_from_header,
    flag_anchor_stuffing,
    flag_one_breath,
    flag_personal_cost_boilerplate,
    is_whole_line_stage_action,
)
from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _quality_flags_for_line,
    _quality_reroll_hint,
    compose_line,
)

_HEADER_WITH_ANCHORS = (
    "TITLE: x\nSETTING: y\nTIME: z\n\n"
    "Specificity anchors (when natural, ground a line in one of these concrete "
    "details; do not force them into every line):\n"
    "- 41.3 degrees C\n- 837/835 form\n- the rail death\n- Swift Observatory\n\n"
    "PREMISE: w"
)
_ANCHORS = ["41.3 degrees C", "837/835 form", "the rail death", "Swift Observatory"]


class TestExtractAnchors:
    def test_parses_block_in_header_order(self):
        assert extract_specificity_anchors_from_header(_HEADER_WITH_ANCHORS) == _ANCHORS

    def test_no_block_returns_empty(self):
        assert extract_specificity_anchors_from_header("TITLE: x\nSETTING: y") == []

    def test_bad_input_safe(self):
        assert extract_specificity_anchors_from_header(None) == []
        assert extract_specificity_anchors_from_header(123) == []


class TestFlagAnchorStuffing:
    def test_three_distinct_anchors_flags(self):
        line = ("the 837/835 form logged the rail death right as it hit "
                "41.3 degrees C outside the gate")
        hit, hint = flag_anchor_stuffing(line, _ANCHORS)
        assert hit is True
        assert "41.3 degrees C" in hint

    def test_two_anchors_passes(self):
        line = "the 837/835 form is wrong about the rail death, plain and simple"
        assert flag_anchor_stuffing(line, _ANCHORS)[0] is False

    def test_no_anchors_passes(self):
        assert flag_anchor_stuffing("Name's on the chip, Steiner.", _ANCHORS)[0] is False

    def test_threshold_tunable(self):
        line = "the 837/835 form is wrong about the rail death"
        assert flag_anchor_stuffing(line, _ANCHORS, threshold=2)[0] is True


class TestFlagOneBreath:
    def test_over_hard_ceiling_flags(self):
        line = " ".join(["word"] * 29)
        assert flag_one_breath(line)[0] is True

    def test_soft_ceiling_with_nesting_flags(self):
        # 27 words (over the 22 soft ceiling, under the 28 hard ceiling) with
        # heavy comma/conjunction nesting -> trips the soft path.
        line = ("the record now includes a partial echo of the rail death, and "
                "the discrimination, and the harassment, while nobody here ever "
                "wrote any of it all down")
        assert flag_one_breath(line)[0] is True

    def test_short_line_passes(self):
        assert flag_one_breath("Name's on the chip, Steiner. Let's see it.")[0] is False


class TestWholeLineStageAction:
    @pytest.mark.parametrize("line", [
        "snaps off pen's tip, jams it into the decryption machine's port, "
        "turning it into scrap metal",
        "steps forward, revealing a keycard from the drawer",
    ])
    def test_action_chain_flagged(self, line):
        assert is_whole_line_stage_action(line) is True

    @pytest.mark.parametrize("line", [
        "My thumb keeps catching on the log's corner, just there.",
        "I snap off the pen's tip.",
        "Looks like rain, Watson.",
        "She knows the code.",
        '"Get the rig running," she said, flat.',
    ])
    def test_dialogue_left_alone(self, line):
        assert is_whole_line_stage_action(line) is False


class TestFlagPersonalCostBoilerplate:
    @pytest.mark.parametrize("line", [
        "the trust they will lose either way",
        "what it costs them to be the one who decides",
    ])
    def test_boilerplate_flagged(self, line):
        assert flag_personal_cost_boilerplate(line)[0] is True

    def test_concrete_consequence_left_alone(self):
        assert flag_personal_cost_boilerplate(
            "She loses access to the observatory archive.")[0] is False


def _req(**over):
    base = dict(
        speaker="EDNA", intent="confront", mood="tense", target_words=15,
        canon_header=_HEADER_WITH_ANCHORS,
        last_lines=[("MARLOWE", "You wanted to see me?")],
        speaker_role="character",
    )
    base.update(over)
    return LineRequest(**base)


_STUFFED = ("the 837/835 form logged the rail death right as it hit "
            "41.3 degrees C near the Swift Observatory gate")
_CLEAN = "Name's on the chip, Steiner. Let's see it before dawn."


class TestScorer:
    def test_v2_off_skips_anchor_and_one_breath(self):
        # v2 OFF => only cliche/stage/nose run; a stuffed line scores 0 flags.
        flags = _quality_flags_for_line(_STUFFED, _req(story_quality_v2_enabled=False))
        assert [c for c, _, _ in flags] == []

    def test_v2_on_character_flags_anchor_stuffing(self):
        flags = _quality_flags_for_line(_STUFFED, _req(story_quality_v2_enabled=True))
        assert "anchor_stuffing" in {c for c, _, _ in flags}

    def test_announcer_role_skips_v2_subset(self):
        flags = _quality_flags_for_line(
            _STUFFED, _req(story_quality_v2_enabled=True, speaker_role="announcer"))
        assert "anchor_stuffing" not in {c for c, _, _ in flags}


class TestHintComposition:
    def test_collapse_when_one_breath_and_anchor(self):
        flags = [("one_breath", "a", "one_breath"),
                 ("anchor_stuffing", "b", "anchor_stuffing")]
        assert "one spoken beat" in _quality_reroll_hint(flags)

    def test_top1_priority_one_breath_over_cliche(self):
        flags = [("cliche", "cliche-reason", "cliche"),
                 ("one_breath", "one-breath-reason", "one_breath")]
        assert _quality_reroll_hint(flags) == "one-breath-reason"

    def test_caps_at_240(self):
        flags = [("cliche", "x" * 500, "cliche")]
        assert len(_quality_reroll_hint(flags)) == 240


class TestComposerQualityGate:
    def test_v2_on_stuffed_draft_rerolls_and_stamps(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _STUFFED if calls["n"] == 1 else _CLEAN

        res = compose_line(creative_fn=mock, req=_req(story_quality_v2_enabled=True))
        assert calls["n"] == 2                       # exactly one reroll
        assert res.text == _CLEAN
        assert "anchor_stuffing_retry" in res.compose_flags

    def test_v2_off_stuffed_draft_no_reroll(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _STUFFED

        res = compose_line(creative_fn=mock, req=_req(story_quality_v2_enabled=False))
        assert calls["n"] == 1                       # byte-identical, no reroll
        assert res.text == _STUFFED
        assert not any(f.endswith("_retry") for f in res.compose_flags)

    def test_reroll_budget_bounded(self):
        """A persistently stuffed draft rerolls at most once (the guard caps it);
        the composer must not recurse without bound."""
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _STUFFED                          # never improves

        res = compose_line(creative_fn=mock, req=_req(story_quality_v2_enabled=True))
        assert calls["n"] == 2                       # initial + one quality reroll
        assert res.text == _STUFFED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
