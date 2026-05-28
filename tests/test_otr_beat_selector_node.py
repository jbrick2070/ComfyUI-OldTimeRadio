"""Sprint 4.1 (2026-05-28) -- OTR_BeatSelector ComfyUI node wrapper.

The node is a thin wrapper around
`_otr_beat_selector.select_winning_beat_sheet`; the pure-Python
selector itself is covered by tests/test_beat_selector.py. This
file pins the node-level surface: input shape, output shape,
JSON parsing, empty-input handling, registration.
"""
from __future__ import annotations

import json

import pytest

from nodes.OTR_BeatSelector import OTR_BeatSelector
from nodes._otr_dramatic_state import DramaticState
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
    stamp_dialogue_slot_ids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ds() -> DramaticState:
    return DramaticState(
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        character_a_wants="hide the falsified ledger before the audit",
        character_b_wants="get Maeve's signature on the record tonight",
        costly_choice_beat="d003",
        ending_change="the falsified ledger is now part of the public record",
    )


def _clean_beats() -> list[Stage1Beat]:
    return [
        Stage1Beat(
            beat_id="b001", speaker="ANNOUNCER",
            intent="open the episode",
            length_target_words=15, emotional_register="welcoming",
            state_before="audit has not begun",
            state_after="audit window is open",
            tension=2,
        ),
        Stage1Beat(
            beat_id="b002", speaker="ALICE",
            intent="discover the missing ledger",
            length_target_words=20, emotional_register="tight unease",
            state_before="ledger somewhere",
            state_after="ledger missing",
            tension=3,
        ),
        Stage1Beat(
            beat_id="b003", speaker="ALICE",
            intent="make the costly choice to sign",
            length_target_words=30, emotional_register="grim resolve",
            state_before="alice undecided",
            state_after="alice has signed",
            tension=5,
        ),
        Stage1Beat(
            beat_id="b004", speaker="ANNOUNCER",
            intent="close the episode and tag the broadcast",
            length_target_words=15, emotional_register="quiet aftermath",
            state_before="signature is real",
            state_after="record is public",
            tension=3,
        ),
    ]


def _plan_dict() -> dict:
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


# ---------------------------------------------------------------------------
# Surface invariants
# ---------------------------------------------------------------------------


def test_node_inputs_required_a_optional_bc():
    schema = OTR_BeatSelector.INPUT_TYPES()
    assert "candidate_a_json" in schema["required"]
    assert "candidate_b_json" in schema["optional"]
    assert "candidate_c_json" in schema["optional"]
    # PD6 (no model_id widget): the selector has no LLM call so the
    # surface must not carry one.
    all_keys = list(schema["required"].keys()) + list(schema["optional"].keys())
    assert "model_id" not in all_keys
    assert "creative_model" not in all_keys
    assert "technical_model" not in all_keys


def test_node_returns_two_strings():
    assert OTR_BeatSelector.RETURN_TYPES == ("STRING", "STRING")
    assert OTR_BeatSelector.RETURN_NAMES == (
        "winning_plan_json", "selector_audit",
    )


def test_node_is_registered_in_package():
    """Catches the regression where a new node lands in nodes/ but
    is forgotten in __init__.py's _NODE_MODULES dict."""
    import nodes  # the package side import
    # The package __init__ scans modules and registers them on the
    # repo-root __init__.py NODE_CLASS_MAPPINGS dict; the nodes/
    # package itself doesn't carry the mapping but importing
    # OTR_BeatSelector at module level proves the class is reachable.
    import nodes.OTR_BeatSelector as bs_mod
    assert hasattr(bs_mod, "OTR_BeatSelector")


# ---------------------------------------------------------------------------
# Happy path: 3 valid candidates
# ---------------------------------------------------------------------------


def test_node_picks_winner_among_three():
    node = OTR_BeatSelector()
    payload = json.dumps(_plan_dict())
    winning_json, audit_json = node.run(
        candidate_a_json=payload,
        candidate_b_json=payload,
        candidate_c_json=payload,
    )
    # All three identical -> winner index 0 by tie break.
    winner = json.loads(winning_json)
    audit = json.loads(audit_json)
    assert winner["premise"].startswith("A short test premise")
    assert audit["winner_index"] == 0
    assert audit["eligible_indices"] == [0, 1, 2]
    assert all(s["total"] == 5 for s in audit["scores_per_candidate"])


def test_node_handles_single_candidate():
    node = OTR_BeatSelector()
    payload = json.dumps(_plan_dict())
    winning_json, audit_json = node.run(
        candidate_a_json=payload,
        candidate_b_json="",
        candidate_c_json="",
    )
    audit = json.loads(audit_json)
    assert audit["winner_index"] == 0
    assert audit["eligible_indices"] == [0]


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_node_empty_inputs_emit_empty_plan_and_audit():
    node = OTR_BeatSelector()
    winning_json, audit_json = node.run(
        candidate_a_json="",
        candidate_b_json="",
        candidate_c_json="",
    )
    assert winning_json == ""
    audit = json.loads(audit_json)
    assert audit["winner_index"] == -1
    assert "no usable" in audit["reason"]


def test_node_malformed_json_skips_candidate_does_not_crash():
    node = OTR_BeatSelector()
    good = json.dumps(_plan_dict())
    winning_json, audit_json = node.run(
        candidate_a_json="not json at all {",
        candidate_b_json=good,
        candidate_c_json="",
    )
    # Selector saw one good candidate; index 0 in the filtered list.
    audit = json.loads(audit_json)
    assert audit["winner_index"] == 0
    assert audit["eligible_indices"] == [0]


def test_node_all_candidates_invalid_emits_empty_plan_with_audit():
    """All three candidates fail Sprint 2 validate_beat_sheet ->
    no valid winner -> empty winning_plan + audit."""
    node = OTR_BeatSelector()
    bad = _plan_dict()
    # Break the costly_choice_beat slot on every candidate.
    bad["dramatic_state"]["costly_choice_beat"] = "d099"
    payload = json.dumps(bad)
    winning_json, audit_json = node.run(
        candidate_a_json=payload,
        candidate_b_json=payload,
        candidate_c_json=payload,
    )
    assert winning_json == ""
    audit = json.loads(audit_json)
    assert audit["winner_index"] == -1
