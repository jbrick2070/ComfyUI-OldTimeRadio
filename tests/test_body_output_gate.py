"""Story-engine assumption-audit KILL 1 (2026-06-24) -- the deterministic
BODY-OUTPUT gate.

The L1 grounding only edits ``beat.intent``; the SHIPPED dialogue was ungated,
so the weak local writer still collapsed every premise into the same "console
standoff". KILL 1 validates the COMPOSED line text and rerolls ONCE with a
split, targeted hint.

Covers:
  l12       ``ungrounded_crisis_tokens`` (+ ``count_ungrounded_crisis``
            delegation); ``_strip_possessive`` / ``_content_tokens`` /
            ``line_references_object`` (head-noun + possessive); and
            ``validate_composed_grounding`` pass/fail with SPLIT reasons,
            including a conflict object that lives ONLY in ``outline.premise``.
  composer  the de-license of the NAMED-ENTITIES license + the grounding rider,
            gated on ``req.conflict_object`` (byte-identical when empty); the new
            ``grounded_nouns`` field default.
  reroll    ``build_reroll_line_request`` threads ``grounded_nouns`` from meta.
  writer    ``_otr_body_gate_hint`` splits reasons into a concrete reroll note.

Pure Python; no GPU, no LLM, no network. Where the lever is off (no
conflict_object), each assertion proves the byte-identical baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_quality_l12 as L12  # noqa: E402
from nodes import _otr_line_composer as LC  # noqa: E402
from nodes import _otr_reroll as RR  # noqa: E402

try:  # the heavy writer node imports torch; guard so CI without torch still runs
    from nodes import OTR_LedgerScriptWriter as WRITER  # noqa: E402
except Exception:  # pragma: no cover - env-dependent
    WRITER = None


def _palette(*texts, roster=()):
    return L12.premise_noun_palette(roster, *texts)


# ---------------------------------------------------------------------------
# ungrounded_crisis_tokens + count delegation
# ---------------------------------------------------------------------------
class TestUngroundedCrisisTokens:
    def test_flags_generic_machinery_in_order(self):
        grounded = _palette("a dispute at the orbital relay station")
        text = "press the red lever, then hit the console override"
        toks = L12.ungrounded_crisis_tokens(text, grounded)
        assert toks == ["lever", "console", "override"]

    def test_count_delegates_to_tokens(self):
        grounded = _palette("the relay")
        text = "the lever and the gauge and the lever again"
        assert L12.count_ungrounded_crisis(text, grounded) == len(
            L12.ungrounded_crisis_tokens(text, grounded)
        ) == 3

    def test_premise_noun_is_not_flagged(self):
        # A word that is ALSO a generic crisis noun but appears in the premise is
        # grounded -> not an offender (e.g. a story literally about a "panel").
        grounded = _palette("a fight over the ethics review panel")
        assert "panel" not in L12.ungrounded_crisis_tokens(
            "they argue across the panel", grounded
        )

    def test_clean_line_has_no_offenders(self):
        grounded = _palette("a disputed fossil dig")
        assert L12.ungrounded_crisis_tokens(
            "you signed off on the provenance, not me", grounded
        ) == []


# ---------------------------------------------------------------------------
# object-reference matching (head-noun / possessive / token overlap)
# ---------------------------------------------------------------------------
class TestLineReferencesObject:
    def test_head_noun_overlap(self):
        assert L12.line_references_object(
            "the ward is where this ends", "the quarantine ward"
        )

    def test_possessive_is_stripped(self):
        # "ward's" -> "ward" matches the object token "ward".
        assert L12.line_references_object(
            "the ward's lights flicker", "the quarantine ward"
        )

    def test_no_overlap_is_false(self):
        assert not L12.line_references_object(
            "we are out of time", "the disputed fossil's provenance"
        )

    def test_empty_object_is_vacuously_true(self):
        assert L12.line_references_object("anything at all", "")

    def test_stopwords_do_not_match(self):
        # "the" is a stopword in both the object and the line -> must NOT count
        # as a reference on its own.
        assert not L12.line_references_object("the the the", "the reactor")


# ---------------------------------------------------------------------------
# validate_composed_grounding -- the gate decision + SPLIT reasons
# ---------------------------------------------------------------------------
class TestValidateComposedGrounding:
    ROLES = L12.CLIMAX_CLASS_ROLES | {L12.BEAT_ROLE_PRESSURE}

    def test_clean_grounded_line_passes(self):
        grounded = _palette("a dispute over the dig site's permit")
        entry = {
            "beat_role": L12.BEAT_ROLE_PRESSURE,
            "conflict_object": "the dig site's permit",
        }
        ok, reasons = L12.validate_composed_grounding(
            "your name is not on the permit, Dana", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is True
        assert reasons == []

    def test_ungrounded_crisis_fails_with_offending_tokens(self):
        grounded = _palette("a dispute over the dig site's permit")
        entry = {"beat_role": L12.BEAT_ROLE_SETUP, "conflict_object": "the permit"}
        ok, reasons = L12.validate_composed_grounding(
            "pull the lever and start the countdown", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is False
        assert reasons == ["ungrounded_crisis:lever,countdown"]

    def test_missing_conflict_object_on_required_role(self):
        grounded = _palette("a dispute over the dig site's permit")
        entry = {
            "beat_role": L12.BEAT_ROLE_IRREVERSIBLE_CHOICE,
            "conflict_object": "the dig site's permit",
        }
        ok, reasons = L12.validate_composed_grounding(
            "I am done arguing with you about this", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is False
        assert reasons == ["missing_conflict_object:the dig site's permit"]

    def test_object_not_required_off_role(self):
        # SETUP is not in the required-object set, so a missing object is fine.
        grounded = _palette("a dispute over the permit")
        entry = {"beat_role": L12.BEAT_ROLE_SETUP, "conflict_object": "the permit"}
        ok, reasons = L12.validate_composed_grounding(
            "good morning, everyone", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is True

    def test_both_failures_split_into_two_reasons(self):
        grounded = _palette("a dispute over the permit")
        entry = {
            "beat_role": L12.BEAT_ROLE_REVERSAL,
            "conflict_object": "the permit",
        }
        ok, reasons = L12.validate_composed_grounding(
            "trip the failsafe now", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is False
        assert "ungrounded_crisis:failsafe" in reasons
        assert "missing_conflict_object:the permit" in reasons

    def test_object_only_in_premise_palette_grounds_the_line(self):
        # The conflict object's head noun appears ONLY in outline.premise; the
        # palette must include premise nouns so the line counts as grounded.
        grounded = _palette("an inquest into the lighthouse keeper's logbook")
        entry = {
            "beat_role": L12.BEAT_ROLE_CONFESSION,
            "conflict_object": "the keeper's logbook",
        }
        ok, reasons = L12.validate_composed_grounding(
            "the logbook never left my hands", entry, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is True
        assert reasons == []

    def test_none_entry_is_safe(self):
        grounded = _palette("the permit")
        ok, reasons = L12.validate_composed_grounding(
            "a calm grounded line", None, grounded,
            require_conflict_object_on_roles=self.ROLES,
        )
        assert ok is True

    def test_max_ungrounded_threshold(self):
        grounded = _palette("the permit")
        entry = {"beat_role": L12.BEAT_ROLE_SETUP, "conflict_object": ""}
        # one offender allowed when max_ungrounded=1
        ok, _ = L12.validate_composed_grounding(
            "one lever only", entry, grounded, max_ungrounded=1,
        )
        assert ok is True
        ok2, _ = L12.validate_composed_grounding(
            "lever and gauge", entry, grounded, max_ungrounded=1,
        )
        assert ok2 is False


# ---------------------------------------------------------------------------
# composer: grounded_nouns field + de-license gated on conflict_object
# ---------------------------------------------------------------------------
class TestComposerDeLicense:
    def _req(self, **kw):
        base = dict(
            speaker="ALEX", intent="say something", mood="tense",
            target_words=20, speaker_role="character",
            canon_header="", last_lines=[],
            allowed_people=frozenset({"Dana"}),
            allowed_things=frozenset({"the lab"}),
        )
        base.update(kw)
        return LC.LineRequest(**base)

    def test_grounded_nouns_defaults_empty(self):
        assert self._req().grounded_nouns == frozenset()

    def test_license_present_when_no_conflict_object(self):
        prompt = LC._build_user_prompt(self._req())
        assert 'Generic roles ("the tech", "the lab", "mission control")' in prompt
        assert "Ground this line in the news facts" in prompt

    def test_license_dropped_when_conflict_object_set(self):
        prompt = LC._build_user_prompt(
            self._req(conflict_object="the dig site's permit")
        )
        # the permissive "generic roles are fine" license is gone
        assert "are fine. Do not invent any other proper name." not in prompt
        # and the news-facts rider is replaced by premise+conflict grounding
        assert "Ground this line in the news facts" not in prompt
        assert "the dig site's permit" in prompt
        assert "do not retreat to generic control-room machinery" in prompt

    def test_named_entities_block_still_forbids_invention(self):
        prompt = LC._build_user_prompt(
            self._req(conflict_object="the permit")
        )
        assert "do not invent any other proper name" in prompt


# ---------------------------------------------------------------------------
# reroll rebuilder threads grounded_nouns from meta
# ---------------------------------------------------------------------------
class TestRerollThreadsGroundedNouns:
    def _ledger(self, meta):
        return {
            "meta": meta,
            "cast": [{"char_id": "c01", "name": "ALEX"}],
            "lines": [
                {
                    "line_id": "b001", "beat_id": "b001", "char_id": "c01",
                    "beat_intent": "they argue", "traits": "tense",
                    "target_words": 20, "arc_phase": "rising",
                },
            ],
        }

    def test_grounded_nouns_round_trips(self):
        meta = {"grounded_nouns": ["permit", "dana", "logbook"]}
        led = self._ledger(meta)
        req = RR.build_reroll_line_request(led, led["lines"][0], led["cast"])
        assert req.grounded_nouns == frozenset({"permit", "dana", "logbook"})

    def test_absent_key_is_empty(self):
        led = self._ledger({})
        req = RR.build_reroll_line_request(led, led["lines"][0], led["cast"])
        assert req.grounded_nouns == frozenset()


# ---------------------------------------------------------------------------
# writer hint builder -- SPLIT reasons -> one concrete note
# ---------------------------------------------------------------------------
@pytest.mark.skipif(WRITER is None, reason="heavy writer node import unavailable")
class TestBodyGateHint:
    def test_ungrounded_crisis_hint_names_offenders(self):
        hint = WRITER._otr_body_gate_hint(
            ["ungrounded_crisis:lever,countdown"], {},
        )
        assert "lever" in hint and "countdown" in hint
        assert "generic crisis machinery" in hint

    def test_missing_object_hint_names_object(self):
        hint = WRITER._otr_body_gate_hint(
            ["missing_conflict_object:the dig site's permit"], {},
        )
        assert "the dig site's permit" in hint
        assert hint.startswith("Ground the line")

    def test_both_reasons_combined(self):
        hint = WRITER._otr_body_gate_hint(
            ["ungrounded_crisis:lever", "missing_conflict_object:the permit"],
            {},
        )
        assert "lever" in hint
        assert "the permit" in hint

    def test_fallback_to_entry_object_when_no_reasons(self):
        hint = WRITER._otr_body_gate_hint([], {"conflict_object": "the permit"})
        assert "the permit" in hint

    def test_empty_when_nothing_actionable(self):
        assert WRITER._otr_body_gate_hint([], {}) == ""
