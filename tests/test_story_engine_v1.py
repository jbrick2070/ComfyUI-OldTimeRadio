"""tests/test_story_engine_v1.py -- story-engine v1 sprint (F1..F8).

Per-feature unit coverage for the content-only story-quality fixes. Pure
Python, no GPU, no LLM (mock generate_fns only). One file, grouped by feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    compose_line_draft,
)
from nodes._otr_dramatic_state import pick_costly_choice_slot  # noqa: E402
from nodes._otr_slot_drama_contract import (  # noqa: E402
    SlotDramaContract,
    derive_contract_skeleton,
    validate_episode_contracts,
)


def _req(**over):
    base = dict(
        speaker="ALICE",
        intent="reveal the signal",
        mood="tense",
        target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )
    base.update(over)
    return LineRequest(**base)


# ===========================================================================
# F1 -- length tail no longer hard-caps every line at 20-30 words
# ===========================================================================

class TestF1LengthTail:

    def test_tail_drops_literal_word_cap(self):
        prompt = _build_user_prompt(_req(target_words=120))
        assert "20-30 words" not in prompt
        assert "about 20-30" not in prompt

    def test_tail_keeps_spoken_cadence_rider(self):
        prompt = _build_user_prompt(_req())
        assert "spoken-length -- one breath" in prompt
        assert "Ground this line in the news facts" in prompt

    def test_word_count_target_still_present(self):
        # the per-line target is still communicated via WRITE LINE
        prompt = _build_user_prompt(_req(target_words=42))
        assert "Word count target: 42." in prompt

    def test_token_budget_scales_to_beat_target(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # small beat target -> attempt-1 budget scales (15*4=60 < cap 200)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=15))
        assert captured["mnt"] == 60

    def test_token_budget_capped_for_long_beat(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # large beat target -> capped at 200 (864*4 would be 3456)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=864))
        assert captured["mnt"] == 200

    def test_token_budget_zero_target_uses_full_cap(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # zero/falsy target -> full cap, never the 40 floor starving it
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=0))
        assert captured["mnt"] == 200


# ===========================================================================
# F6 -- split rider: indirect-performance unconditional, situation-change gated
# ===========================================================================

class TestF6SplitRider:

    def test_indirect_rider_unconditional_on_plain_beat(self):
        # no dramatic fields at all -> the indirect rider is STILL present
        prompt = _build_user_prompt(_req())
        assert "Do not summarize the objective" in prompt
        assert "Perform the objective indirectly" in prompt

    def test_situation_change_absent_on_plain_beat(self):
        prompt = _build_user_prompt(_req())
        assert "The situation must be different after this line." not in prompt

    def test_situation_change_present_on_turn_beat(self):
        prompt = _build_user_prompt(_req(beat_turn="she names the omission aloud"))
        assert "Perform the objective indirectly" in prompt
        assert "The situation must be different after this line." in prompt

    def test_rider_lands_before_speak_now(self):
        prompt = _build_user_prompt(_req(beat_turn="the lie collapses"))
        assert prompt.index("Perform the objective indirectly") < prompt.index("Speak now.")


# ===========================================================================
# F2 -- costly-choice binding: must_turn only on a character slot, audit-safe
# ===========================================================================

_DS = {
    "dramatic_question": "Will the vial be opened before the audit closes?",
    "character_a_wants": "broadcast the strain to the whole network",
    "character_b_wants": "keep the vial sealed and unrecorded",
    "ending_change": "the vial is opened and the record cannot be undone",
}


def _skeleton(slot_id, speaker, slot_index, costly):
    ds = dict(_DS)
    ds["costly_choice_beat"] = costly
    return derive_contract_skeleton(
        slot_row={"dialogue_slot_id": slot_id, "speaker": speaker},
        slot_index=slot_index,
        dramatic_state=ds,
        active_props=[],
        key_terms=["the vial", "the audit"],
    )


def _full_contract(slot_id, speaker, slot_index, costly):
    sk = _skeleton(slot_id, speaker, slot_index, costly)
    sk["line_job"] = "press the point without naming it"
    sk["hidden_pressure"] = "the audit clock is running out"
    return SlotDramaContract.model_validate(sk)


class TestF2CostlyBinding:

    def test_pick_costly_from_character_only_returns_character_slot(self):
        char_slots = ["d002", "d004", "d006"]
        assert pick_costly_choice_slot(char_slots) in char_slots

    def test_must_turn_lands_on_costly_character_slot(self):
        # costly = d004 (a character slot) -> only d004 gets must_turn
        s2 = _skeleton("d002", "EDNA", 0, "d004")
        s4 = _skeleton("d004", "PETER", 1, "d004")
        s5 = _skeleton("d005", "ANNOUNCER", 2, "d004")
        assert s2["must_turn"] is False
        assert s4["must_turn"] is True
        assert s5["must_turn"] is False

    def test_cleared_costly_yields_no_turn_and_invalid_audit(self):
        # all-announcer / empty-cast path: costly cleared -> no must_turn
        c1 = _full_contract("d001", "ANNOUNCER", 0, "")
        c2 = _full_contract("d002", "ANNOUNCER", 1, "")
        assert c1.must_turn is False and c2.must_turn is False
        ok, reasons = validate_episode_contracts(
            [c1, c2], [], ["the vial", "the audit"])
        assert ok is False
        assert any("no slot carries" in r for r in reasons)

    def test_exactly_one_character_turn_passes_audit(self):
        c_open = _full_contract("d001", "ANNOUNCER", 0, "d004")
        c_mid = _full_contract("d002", "EDNA", 1, "d004")
        c_turn = _full_contract("d004", "PETER", 2, "d004")
        ok, reasons = validate_episode_contracts(
            [c_open, c_mid, c_turn], [], ["the vial", "the audit"])
        assert ok is True, reasons
        turn_slots = [c.dialogue_slot_id for c in (c_open, c_mid, c_turn) if c.must_turn]
        assert turn_slots == ["d004"]

    def test_two_turns_fail_audit(self):
        c_a = _full_contract("d002", "EDNA", 1, "d002")
        c_b = _full_contract("d004", "PETER", 2, "d004")
        ok, reasons = validate_episode_contracts(
            [c_a, c_b], [], ["the vial", "the audit"])
        assert ok is False
        assert any("more than one" in r for r in reasons)
