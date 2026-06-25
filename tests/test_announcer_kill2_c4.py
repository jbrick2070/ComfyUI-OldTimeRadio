"""KILL 4 -- C4: role-keyed enrichment + the truncation reserve/clamp.

build_sq_data now appends a concrete on-stage clause for EVERY dramatic-function
role (setup / pressure / personal_stake + each climax class), not just the two it
covered before -- so the body is un-starved. CONSEQUENCE is omitted (unreachable
under climax-last today). The truncation now RESERVES room for the enrichment, so
a long intent can no longer cut the concrete climax clause off the end (the old
[:_INTENT_MAX] did). The two pre-KILL-4 roles enrich byte-identically.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_quality_l12 as L12  # noqa: E402


class _Beat:
    def __init__(self, beat_id, speaker_role, intent="they argue over the matter",
                 mood="tense", speaker="ALEX"):
        self.beat_id = beat_id
        self.speaker_role = speaker_role
        self.intent = intent
        self.mood = mood
        self.speaker = speaker


def _char_beats(n, intent="they argue over the matter"):
    return [_Beat(f"b{i:03d}", "character", intent=intent) for i in range(1, n + 1)]


class TestEnrichRoles:
    def test_covers_setup_pressure_personal_stake_and_all_climax(self):
        assert L12.BEAT_ROLE_SETUP in L12._ENRICH_ROLES
        assert L12.BEAT_ROLE_PRESSURE in L12._ENRICH_ROLES
        assert L12.BEAT_ROLE_PERSONAL_STAKE in L12._ENRICH_ROLES
        assert L12.CLIMAX_CLASS_ROLES <= L12._ENRICH_ROLES

    def test_consequence_is_omitted(self):
        assert L12.BEAT_ROLE_CONSEQUENCE not in L12._ENRICH_ROLES

    def test_every_enriched_role_has_a_tail(self):
        for role in L12._ENRICH_ROLES:
            assert role in L12._ENRICH_TAILS, role


class TestEnrichTail:
    SLOT = {"conflict_object": "the red ledger"}
    FC = {"personal_cost": "her good name"}

    @pytest.mark.parametrize("role,needle", [
        ("setup", "the scene is set"),
        ("pressure", "the pressure tightens"),
        ("revelation", "the truth about the red ledger comes to light"),
        ("reversal", "turns against what was expected"),
        ("unresolved_final_sound", "left hanging"),
        ("reconciliation", "make their peace"),
        ("bittersweet_parting", "gaining and losing"),
        ("ironic_twist", "ironic turn"),
        ("quiet_acceptance", "quietly accept"),
        ("confession", "confessed aloud"),
    ])
    def test_thin_intent_gets_class_specific_tail(self, role, needle):
        tail = L12._enrich_tail("thin", role, self.SLOT, self.FC)
        assert needle in tail

    def test_substantive_intent_with_object_gets_no_tail(self):
        intent = "she opens the red ledger and reads the damning entry aloud to all"
        assert L12._enrich_tail(intent, "revelation", self.SLOT, self.FC) == ""

    def test_consequence_role_gets_no_tail(self):
        assert L12._enrich_tail("thin", L12.BEAT_ROLE_CONSEQUENCE, self.SLOT, self.FC) == ""


class TestEnrichIntentByteIdentical:
    """The two pre-KILL-4 roles must still produce the EXACT prior enrichment."""

    def test_irreversible_choice_unchanged(self):
        out = L12._enrich_intent(
            "short", L12.BEAT_ROLE_IRREVERSIBLE_CHOICE,
            {"conflict_object": "the red ledger"}, {"personal_cost": "her good name"},
        )
        assert out == (
            "short; on-stage, the decision about the red ledger is made now, "
            "costing her good name"
        )

    def test_personal_stake_unchanged(self):
        out = L12._enrich_intent(
            "short", L12.BEAT_ROLE_PERSONAL_STAKE,
            {"conflict_object": "the key"}, {"personal_cost": "his freedom"},
        )
        assert out == "short; what is personally at stake: his freedom"


class TestTruncationClamp:
    def test_long_intent_preserves_the_tail(self):
        # original + tail > 200 -> the OLD code cut the tail; the clamp keeps it.
        beats = _char_beats(2, intent="setup")
        beats[1].intent = ("padding word " * 30).strip()  # ~390 chars, not thin
        L12.build_sq_data(
            beats, {}, "two people fight over a relic", 7, climax_role="revelation",
        )
        final = beats[-1].intent
        assert len(final) <= 200
        assert "on-stage, the truth about" in final  # the concrete clause survived

    def test_all_intents_within_max_length(self):
        beats = _char_beats(5, intent=("x " * 90).strip())  # long, not thin
        L12.build_sq_data(
            beats, {}, "energy grid coal fight", 2, climax_role="reversal",
        )
        for b in beats:
            assert len(b.intent) <= 200

    def test_setup_beat_now_enriched_when_thin(self):
        beats = _char_beats(3, intent="x")  # thin -> earns enrichment
        sq = L12.build_sq_data(beats, {}, "a relic dispute in a lab", 9)
        setup_ids = [bid for bid, e in sq.items() if e["beat_role"] == "setup"]
        assert setup_ids  # there is a setup beat
        for bid in setup_ids:
            b = next(b for b in beats if b.beat_id == bid)
            assert "the scene is set" in b.intent  # was un-enriched pre-KILL-4

    def test_determinism_holds(self):
        a = _char_beats(4, intent="x")
        b = _char_beats(4, intent="x")
        sqa = L12.build_sq_data(a, {}, "a relic dispute", 3, climax_role="confession")
        sqb = L12.build_sq_data(b, {}, "a relic dispute", 3, climax_role="confession")
        assert sqa == sqb
        assert [x.intent for x in a] == [x.intent for x in b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
