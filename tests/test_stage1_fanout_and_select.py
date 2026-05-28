"""Sprint 4.4 (2026-05-28) -- fan_out_and_select_stage1_plan
composition helper.

Pinned invariants:
  * Composes Sprint 4.3 fan_out_stage1_plans + Sprint 4
    select_winning_beat_sheet into one call.
  * Returns FanOutAndSelectResult with winner_plan, the per-knob
    FanOutResult list, the BeatSelectorAudit, and an error string
    on failure.
  * All-knob-failure path -> winner_plan=None +
    no_valid_error_msg populated (no exception bubbles).
  * No-valid-validate path -> winner_plan=None + audit attached
    (still no exception).
  * Happy path -> the highest-total eligible candidate wins.
"""
from __future__ import annotations

import json

from nodes._otr_dramatic_state import DramaticState
from nodes._otr_stage1_fanout import (
    DiversityKnob,
    FanOutAndSelectResult,
    fan_out_and_select_stage1_plan,
)
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
    parse_and_validate_plan,
    stamp_dialogue_slot_ids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ds(costly: str = "d003") -> DramaticState:
    return DramaticState(
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        character_a_wants="hide the falsified ledger before the audit",
        character_b_wants="get Maeve's signature on the record tonight",
        costly_choice_beat=costly,
        ending_change="the falsified ledger becomes public record",
    )


def _clean_beats() -> list[Stage1Beat]:
    return [
        Stage1Beat(
            beat_id="b001", speaker="ANNOUNCER",
            intent="open the episode",
            length_target_words=15, emotional_register="welcoming",
            state_before="audit has not begun",
            state_after="audit window is open", tension=2,
        ),
        Stage1Beat(
            beat_id="b002", speaker="ALICE",
            intent="discover the missing ledger",
            length_target_words=20, emotional_register="tight unease",
            state_before="ledger somewhere",
            state_after="ledger missing", tension=3,
        ),
        Stage1Beat(
            beat_id="b003", speaker="ALICE",
            intent="make the costly choice to sign",
            length_target_words=30, emotional_register="grim resolve",
            state_before="alice undecided",
            state_after="alice has signed", tension=5,
        ),
        Stage1Beat(
            beat_id="b004", speaker="ANNOUNCER",
            intent="close the episode and tag the broadcast",
            length_target_words=15, emotional_register="quiet aftermath",
            state_before="signature is real",
            state_after="record is public", tension=3,
        ),
    ]


def _clean_plan_dict() -> dict:
    plan = Stage1Plan(
        premise="A short test premise long enough to pass schema.",
        arc=Stage1Arc(
            setup="setup statement long enough to validate.",
            complication="complication statement long enough to validate.",
            resolution="resolution statement long enough to validate.",
        ),
        cast=[
            Stage1CastMember(
                name="ALICE", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_3",
                persona="weary forensic engineer with dry humor.",
                arc_role="reluctant insider",
            ),
            Stage1CastMember(
                name="BOB", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="ambitious grant officer evasive.",
                arc_role="pressure source",
            ),
        ],
        beats=_clean_beats(),
        running_facts=["the audit closes at midnight"],
        dramatic_state=_ds(),
    )
    stamp_dialogue_slot_ids(plan)
    return plan.model_dump()


def _parse_to_plan(raw: str) -> Stage1Plan:
    return parse_and_validate_plan(json.loads(raw))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_winner_plus_audit():
    payload = json.dumps(_clean_plan_dict())

    def gen_fn(messages, *, temperature, max_new_tokens):
        return payload

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE_STAGE1_SYSTEM",
        user_prompt="USER",
    )
    assert isinstance(res, FanOutAndSelectResult)
    assert res.ok is True
    assert res.winner_plan is not None
    assert len(res.fan_out_results) == 3
    assert all(r.ok for r in res.fan_out_results)
    assert res.selector_audit is not None
    assert res.no_valid_error_msg == ""
    # All three candidates identical -> winner_index 0 (tie-break).
    assert res.selector_audit.winner_index == 0


def test_happy_path_single_knob_mode():
    payload = json.dumps(_clean_plan_dict())

    def gen_fn(messages, *, temperature, max_new_tokens):
        return payload

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE",
        user_prompt="USER",
        knobs=[DiversityKnob.MORAL_DILEMMA],
    )
    assert res.ok is True
    assert len(res.fan_out_results) == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_all_knobs_fail_generate_returns_no_winner():
    """generate_fn raises on every knob -> winner_plan=None +
    detailed error message; no exception bubbles."""
    def gen_fn(messages, *, temperature, max_new_tokens):
        raise RuntimeError("LLM timeout")

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    assert res.ok is False
    assert res.winner_plan is None
    assert res.selector_audit is None
    assert "no eligible candidates" in res.no_valid_error_msg
    assert "LLM timeout" in res.no_valid_error_msg
    assert all(not r.ok for r in res.fan_out_results)


def test_all_candidates_fail_validate_returns_no_winner_with_audit():
    """Every candidate parses OK but fails Sprint 2 validators ->
    winner_plan=None + audit captured (NoValidBeatSheetError
    surfaces through the helper but does NOT raise)."""
    bad_dict = _clean_plan_dict()
    # Break the costly_choice_beat slot -> validate_beat_sheet
    # rejects.
    bad_dict["dramatic_state"]["costly_choice_beat"] = "d099"
    payload = json.dumps(bad_dict)

    def gen_fn(messages, *, temperature, max_new_tokens):
        return payload

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    assert res.ok is False
    assert res.winner_plan is None
    assert res.selector_audit is not None
    # Audit captures the per-candidate defect lists.
    assert all(d for d in res.selector_audit.defects_per_candidate)
    assert "validate_beat_sheet" in res.no_valid_error_msg


def test_partial_failure_winner_picked_from_eligible_subset():
    """Two of three knobs fail; the surviving candidate wins."""
    payload = json.dumps(_clean_plan_dict())
    call_idx = {"n": 0}

    def gen_fn(messages, *, temperature, max_new_tokens):
        call_idx["n"] += 1
        if call_idx["n"] == 2:  # second knob fails
            raise RuntimeError("transient LLM hiccup")
        return payload

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    assert res.ok is True
    assert res.selector_audit is not None
    # 2 eligible candidates (knob 0 and knob 2; knob 1 failed).
    assert len(res.selector_audit.eligible_indices) == 2


# ---------------------------------------------------------------------------
# Composition fidelity
# ---------------------------------------------------------------------------


def test_fan_out_results_carry_knob_labels():
    """The composition preserves the knob labels for forensic
    inspection -- the operator sees which knob produced which
    candidate."""
    payload = json.dumps(_clean_plan_dict())

    def gen_fn(messages, *, temperature, max_new_tokens):
        return payload

    res = fan_out_and_select_stage1_plan(
        generate_fn=gen_fn,
        parse_fn=_parse_to_plan,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    labels = [r.knob for r in res.fan_out_results]
    assert labels == [
        DiversityKnob.MORAL_DILEMMA,
        DiversityKnob.BUREAUCRATIC_ABSURD,
        DiversityKnob.INTIMATE_PERSONAL_COST,
    ]
