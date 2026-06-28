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
    detect_stage_business_for_reroll,
    extract_specificity_anchors_from_header,
    flag_anchor_stuffing,
    flag_cliche,
    flag_one_breath,
    flag_personal_cost_boilerplate,
    is_whole_line_stage_action,
)
from nodes._otr_line_composer import (  # noqa: E402
    _CODA_FACT_MAX,
    LineRequest,
    _news_coda_fact_flags,
    _quality_flags_for_line,
    _quality_reroll_hint,
    compose_line,
    compose_news_coda,
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


class TestWholeLineWiredIntoReroll:
    """3.3: is_whole_line_stage_action feeds the existing draft-stage detector."""

    def test_all_nonwhitelisted_action_chain_drives_reroll(self):
        # No chunk leads with a whitelisted _NARRATION_VERBS verb, so the
        # per-chunk undelimited check misses it; the whole-line detector catches
        # the impersonal comma chain (the 3.3 value-add).
        hit, hint, reason = detect_stage_business_for_reroll(
            "snaps off the cap, cracks the seal, drains the vial", "MARLOWE")
        assert hit is True
        assert reason == "whole_line_action"
        assert hint

    def test_quoted_dialogue_not_a_whole_line_hit(self):
        hit, _hint, reason = detect_stage_business_for_reroll(
            '"My thumb keeps catching on the log\'s corner," I said.', "ALICE")
        assert not (hit and reason == "whole_line_action")


class TestClicheExpansion:
    @pytest.mark.parametrize("line", [
        "You're playing with fire, Watson.",
        "We're playing with fire here.",
        "Everything hangs in the balance now.",
        "Over my dead body.",
        "Not on my watch.",
        "That secret is best left buried.",
        "We're running out of time.",
        "Move, before it's too late.",
        "Shut down the lab. Safety first.",
        "The whole plan could go up in smoke.",
    ])
    def test_new_cliches_flagged(self, line):
        assert flag_cliche(line)[0] is True

    @pytest.mark.parametrize("line", [
        "The fuse is on fire.",
        "The balance arm is stuck.",
        "I checked my watch.",
    ])
    def test_near_misses_left_alone(self, line):
        assert flag_cliche(line)[0] is False


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

    def test_reroll_not_better_keeps_original_and_stamps_degraded(self):
        """3.4 re-verify: a reroll that swaps one cliche for another (same defect
        count) is NOT kept -- the original draft ships, with quality_reroll_degraded."""
        draft = "You're playing with fire, Watson."
        worse = "Not on my watch, Watson."          # still one cliche
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return draft if calls["n"] == 1 else worse

        res = compose_line(creative_fn=mock, req=_req())
        assert calls["n"] == 2                       # one reroll attempted
        assert res.text == draft                     # original kept (tie -> original)
        assert "quality_reroll_degraded" in res.compose_flags
        assert "cliche_retry" in res.compose_flags

    def test_reroll_better_is_kept(self):
        draft = "You're playing with fire, Watson."
        clean = "The seal cracked sometime before midnight."
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return draft if calls["n"] == 1 else clean

        res = compose_line(creative_fn=mock, req=_req())
        assert res.text == clean
        assert "cliche_retry" in res.compose_flags
        assert "quality_reroll_degraded" not in res.compose_flags

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


class TestCodaTruncationFlag:
    """3.5: measurement-only. The coda fact is never trimmed by the new flag; it
    only records when the existing _CODA_FACT_MAX cap bit (MF-2)."""

    def test_short_fact_no_flag(self):
        assert _news_coda_fact_flags("NASA confirmed the probe landed.",
                                     "NASA confirmed the probe landed.") == ()

    def test_capped_fact_flags_truncated(self):
        raw = "word " * 60                      # ~300 chars before cleaning
        from nodes._otr_line_composer import clean_one_line
        capped = clean_one_line(raw, _CODA_FACT_MAX)
        assert len(capped) <= _CODA_FACT_MAX
        assert _news_coda_fact_flags(raw, capped) == ("news_coda_truncated",)

    def test_compose_news_coda_short_brief_unchanged(self):
        def fn(messages, **kw):
            return "A tale of a keeper and a signal in the dark"
        res = compose_news_coda(
            creative_fn=fn, news_close_brief="The dam held through the night.",
            premise="A keeper waits for a ship", cast_seed=42)
        assert "news_coda_truncated" not in res.compose_flags

    def test_compose_news_coda_long_brief_stamps_truncated(self):
        long_brief = ("Investigators confirmed that the failed launch traces to a "
                      "cracked seal in the upper stage, a part flagged twice in "
                      "prior reviews and cleared each time despite the written "
                      "objections of two senior engineers who later resigned over "
                      "the decision and the way it was recorded.")
        assert len(long_brief) > _CODA_FACT_MAX

        def fn(messages, **kw):
            return "A tale of a keeper and a signal in the dark"
        res = compose_news_coda(
            creative_fn=fn, news_close_brief=long_brief,
            premise="A keeper waits for a ship", cast_seed=42)
        assert "news_coda_truncated" in res.compose_flags


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
