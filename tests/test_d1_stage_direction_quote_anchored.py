"""D1 (2026-06-22, story-quality lift) -- bare stage-direction leak AFTER /
BETWEEN / WITHOUT quotes.

Covers the shared quote helper, the 3rd-person action-clause classifier, the
Tier-2 composer-reroll detector, the Tier-3 quote-anchored freeze floor, the
per-line compose_flags breadcrumb, the negative fixtures, and the byte-identical
golden no-op gate over the hand-authored clean fixture. Pure / CPU. UTF-8 no BOM,
SFW.

Corpus = the real frozen "Chandra's Echo" ledger (grounding_pack.md sec A):
  b005/b010/b012 trailing-after-quote (balanced-quote class -> floor strips),
  b015 embedded-malformed + b017 undelimited (Tier-2 reroll owns them; the floor
  deliberately leaves them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_line_hygiene as H  # noqa: E402
from nodes._otr_ledger_scrub import (  # noqa: E402
    CODE_STAGE_DIRECTION,
    _strip_stage_directions,
    scrub_ledger,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

B005 = '"Not before I amplify it. The world deserves to hear this." adjusts dials on the console'
B010 = ('"I assure you, Manfred, my work is entirely my own. My husband\'s '
        'theories have no bearing here." clutches her wedding ring tightly')
B012 = ('"Gentlemen, let\'s not jump to conclusions. Mali\'s work, while '
        'fascinating, is purely theoretical." taps his cane impatiently')
B015 = ('Well, Manfred, it seems I\'ll be presenting my findings to the UN '
        'sooner than expected." tightens her scarf, a nervous gesture "I do '
        'hope they find my \'theoretical\' work as... persuasive as you seem to.')
B017 = ('Sherlock, stop this at once! overrides systems, fingers dancing on '
        'the console I won\'t let you jeopardize Mali\'s life\'s work over a... '
        'difference in interpretation.')


class TestIsThirdPersonActionClause:
    def test_locked_assertions(self):
        assert H.is_third_person_action_clause("clutches her wedding ring tightly") is True
        assert H.is_third_person_action_clause("taps his cane impatiently") is True
        assert H.is_third_person_action_clause("adjusts dials on the console") is True
        # first person -> dialogue, not a direction
        assert H.is_third_person_action_clause("I adjust the dial as I speak") is False

    def test_second_person_rejected(self):
        assert H.is_third_person_action_clause("adjusts your dial slowly") is False

    def test_non_verb_lead_rejected(self):
        # a real sentence with a subject before the verb is not a direction
        assert H.is_third_person_action_clause("she walks home every night") is False

    def test_non_verb_dialogue_lead_rejected(self):
        # a dialogue-starter / non-verb lead is not a verb-led direction
        assert H.is_third_person_action_clause("look at the readings now") is False
        assert H.is_third_person_action_clause("well, that settles it") is False

    def test_overlong_rejected(self):
        long = "adjusts " + " ".join(["the"] * 20) + " dial"
        assert H.is_third_person_action_clause(long) is False


class TestSegmentDoubleQuotes:
    def test_curly_and_straight_parity(self):
        straight = '"hello there." adjusts the dial'
        curly = '“hello there.” adjusts the dial'
        ns, segs_s = H.segment_double_quotes(straight)
        nc, segs_c = H.segment_double_quotes(curly)
        assert ns == nc  # curly folded to straight
        assert segs_s == segs_c

    def test_apostrophes_ignored(self):
        norm, _ = H.segment_double_quotes("They liked your 'alive' frequency.")
        assert norm.count('"') == 0  # single quotes never counted

    def test_balanced_alternation(self):
        _, segs = H.segment_double_quotes(B005)
        assert segs[0][1] is False  # starts outside
        assert segs[1][1] is True   # in quote
        assert segs[2][1] is False  # trailing outside (the leak)


class TestFloorQuoteAnchored:
    def test_b005_stripped_wellformed(self):
        reasons = []
        out, changed = _strip_stage_directions(B005, reasons)
        assert changed is True
        assert out == '"Not before I amplify it. The world deserves to hear this."'
        assert "trailing_after_quote" in reasons

    def test_b010_stripped(self):
        out, changed = _strip_stage_directions(B010)
        assert changed is True
        assert out.endswith('here."')
        assert "clutches" not in out

    def test_b012_stripped(self):
        out, changed = _strip_stage_directions(B012)
        assert changed is True
        assert out.endswith('theoretical."')
        assert "taps his cane" not in out

    def test_floor_idempotent(self):
        for raw in (B005, B010, B012):
            once, c1 = _strip_stage_directions(raw)
            twice, c2 = _strip_stage_directions(once)
            assert once == twice
            assert c2 is False

    def test_b015_left_for_tier2(self):
        # malformed embedded -> floor leaves it (1st-person outside spans)
        out, changed = _strip_stage_directions(B015)
        assert changed is False

    def test_b017_left_for_tier2(self):
        # undelimited no-quote -> floor excludes it
        out, changed = _strip_stage_directions(B017)
        assert changed is False

    def test_unbalanced_quote_not_scrubbed(self):
        # odd quote count -> floor leaves unscrubbed (cannot route to reroll)
        bad = 'I never said that." adjusts the dial slowly'
        out, changed, reason = H.strip_quote_anchored_stage_direction(bad)
        assert changed is False
        assert out == bad


class TestTier2Detector:
    def test_reason_codes(self):
        assert H.detect_stage_business_for_reroll(B005)[2] == "trailing_after_quote"
        assert H.detect_stage_business_for_reroll(B010)[2] == "trailing_after_quote"
        assert H.detect_stage_business_for_reroll(B012)[2] == "trailing_after_quote"
        assert H.detect_stage_business_for_reroll(B015)[2] == "embedded_between_quotes"
        assert H.detect_stage_business_for_reroll(B017)[2] == "undelimited_action_clause"

    def test_all_corpus_hits(self):
        for raw in (B005, B010, B012, B015, B017):
            hit, hint, _ = H.detect_stage_business_for_reroll(raw)
            assert hit is True
            assert hint


class TestNegatives:
    NEG = [
        "I adjust the dial as I speak.",                        # 1st-person action
        "They were very interested in your 'alive' frequency.",  # scare quotes
        '"He walks home every night," I told them.',            # 3rd-person re OTHER
        "As the observatory's gaze turns away, the tapestry bears the mark.",  # announcer
        "Keep it running, Mali. Whatever you need, get it done.",  # plain dialogue
    ]

    def test_floor_noop(self):
        for raw in self.NEG:
            _out, changed = _strip_stage_directions(raw)
            assert changed is False, raw

    def test_detector_no_hit(self):
        for raw in self.NEG:
            hit, _, _ = H.detect_stage_business_for_reroll(raw)
            assert hit is False, raw


class TestScrubIntegrationBreadcrumb:
    def _ledger(self, text):
        return {
            "schema": "l3-2026-05-14",
            "cast": [{"char_id": "c03", "name": "MALI"}],
            "lines": [{
                "line_id": "b005", "beat_id": "b005", "char_id": "c03",
                "speaker_role": "character", "text": text,
                "word_count": 99, "char_count": len(text),
            }],
            "meta": {},
        }

    def test_breadcrumb_on_compose_flags(self):
        led = self._ledger(B005)
        res = scrub_ledger(led, repair_available=True)
        row = led["lines"][0]
        # story-quality v2 (baked in): the L7 dialogue|action split now OWNS the
        # trailing-after-quote class -- it runs first, keeps ONLY the spoken
        # dialogue (unquoted) and records the leaked action as an action_split
        # breadcrumb, so the later stage-direction floor has nothing left to strip.
        assert "Not before I amplify it. The world deserves to hear this." in row["text"]
        assert "adjusts dials" not in row["text"]
        flags = list(row.get("compose_flags") or [])
        # the leaked trailing action is recorded as a breadcrumb (action_split is
        # the v2 owner; the legacy stage_dir_stripped breadcrumb covered the same
        # case pre-bake-in).
        assert any(
            f.startswith("action_split:") or f.startswith("stage_dir_stripped:")
            for f in flags
        )
        assert any("trailing_after_quote" in f for f in flags)
        assert CODE_STAGE_DIRECTION in [f.code for f in res.findings]


class TestGoldenNoOpGate:
    """The clean strong fixture must produce ZERO floor strips (D1 golden gate)
    -- a non-zero strip is the operator-gated recapture trigger, surfaced LOUD,
    not discovered by a red byte-identical test."""

    def test_clean_fixture_zero_strips(self):
        led = json.loads((_FIXTURES / "clean_strong_ledger.json").read_text(encoding="utf-8"))
        before = [dict(r) for r in led["lines"]]
        res = scrub_ledger(led, repair_available=True)
        # no stage-direction finding anywhere
        assert CODE_STAGE_DIRECTION not in [f.code for f in res.findings]
        # no spoken row gained a stage_dir_stripped breadcrumb
        for row in led["lines"]:
            flags = list(row.get("compose_flags") or [])
            assert not any(str(f).startswith("stage_dir_stripped:") for f in flags)

    def test_floor_direct_noop_on_clean_lines(self):
        led = json.loads((_FIXTURES / "clean_strong_ledger.json").read_text(encoding="utf-8"))
        for row in led["lines"]:
            txt = row.get("text") or ""
            if not txt:
                continue
            out, changed = _strip_stage_directions(txt)
            assert changed is False, txt


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
