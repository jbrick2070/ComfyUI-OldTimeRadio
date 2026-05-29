"""docs/sprint_drafts/build3/test_slot_drama_contract.py -- Build 3 DRAFT tests.

Covers:
  * deterministic derivation correctness (speaker, must_turn,
    state_before/after, concrete-detail rotation);
  * the per-slot validator catching EACH failure mode;
  * the episode-level "exactly one costly-choice turn" rule;
  * the LLM job-field pass via a FAKE generate_fn (no torch, no model);
  * the regenerate-once-then-deterministic-fallback path.

No torch, no live model, no I/O. Pure pydantic + stdlib.

Run (Windows venv):
  C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe -m pytest \\
    "docs/sprint_drafts/build3/test_slot_drama_contract.py" -v
"""
from __future__ import annotations

import json

import pytest

# Promoted (Build 3, 2026-05-28): the module now lives in the nodes/
# package, so import it the normal way (the staged sys.path/path-loader
# block was removed on promotion per build3/WIRING_SPEC.md).
from nodes import _otr_slot_drama_contract as sdc


# ---------------------------------------------------------------------------
# Fixtures -- minimal shapes matching the real surfaces
# ---------------------------------------------------------------------------


DRAMATIC_STATE = {
    "dramatic_question": "Will Maeve sign the confession before the audit closes?",
    "character_a_wants": "Maeve: protect her brother by staying silent",
    "character_b_wants": "Doyle: extract the signature before midnight",
    "costly_choice_beat": "d004",
    "ending_change": "Maeve signs and surrenders the only leverage she had",
}

ACTIVE_PROPS = ["the confession form", "a desk lamp", "the office clock"]
KEY_TERMS = ["audit", "midnight deadline"]

# Five voiced slots; d004 is the costly-choice turn.
SLOT_ROWS = [
    {"dialogue_slot_id": "d001", "speaker": "Maeve", "intent": "stall the interview"},
    {"dialogue_slot_id": "d002", "speaker": "Doyle", "intent": "press for the signature"},
    {"dialogue_slot_id": "d003", "speaker": "Maeve", "intent": "deflect with a question"},
    {"dialogue_slot_id": "d004", "speaker": "Maeve", "intent": "face the choice"},
    {"dialogue_slot_id": "d005", "speaker": "Doyle", "intent": "close the scene"},
]


def _episode_skeletons():
    return [
        sdc.derive_contract_skeleton(
            slot_row=row,
            slot_index=i,
            dramatic_state=DRAMATIC_STATE,
            active_props=ACTIVE_PROPS,
            key_terms=KEY_TERMS,
        )
        for i, row in enumerate(SLOT_ROWS)
    ]


# ---------------------------------------------------------------------------
# Deterministic derivation
# ---------------------------------------------------------------------------


def test_speaker_derived_from_slot_row():
    sk = sdc.derive_contract_skeleton(
        slot_row=SLOT_ROWS[1], slot_index=1, dramatic_state=DRAMATIC_STATE,
        active_props=ACTIVE_PROPS, key_terms=KEY_TERMS,
    )
    assert sk["speaker"] == "Doyle"
    assert sk["dialogue_slot_id"] == "d002"


def test_speaker_falls_back_to_speaker_name_then_name():
    sk = sdc.derive_contract_skeleton(
        slot_row={"dialogue_slot_id": "d001", "speaker_name": "Ada"},
        slot_index=0, dramatic_state=DRAMATIC_STATE,
    )
    assert sk["speaker"] == "Ada"
    sk2 = sdc.derive_contract_skeleton(
        slot_row={"dialogue_slot_id": "d001", "name": "Ben"},
        slot_index=0, dramatic_state=DRAMATIC_STATE,
    )
    assert sk2["speaker"] == "Ben"


def test_must_turn_only_on_costly_choice_slot():
    sks = _episode_skeletons()
    turns = [s["dialogue_slot_id"] for s in sks if s["must_turn"]]
    assert turns == ["d004"]


def test_turning_slot_state_changes_non_turning_holds():
    sks = _episode_skeletons()
    by_id = {s["dialogue_slot_id"]: s for s in sks}
    turn = by_id["d004"]
    assert turn["state_before"] != turn["state_after"]
    # the ending-change drives the after-state on the turn
    assert "signs" in turn["state_after"] or "surrenders" in turn["state_after"]
    # a non-turning slot holds its state
    hold = by_id["d002"]
    assert hold["state_before"] == hold["state_after"]


def test_concrete_details_are_from_pool_and_rotate():
    sks = _episode_skeletons()
    pool = {t.lower() for t in (ACTIVE_PROPS + KEY_TERMS)}
    for s in sks:
        for d in s["concrete_detail_required"]:
            assert d.lower() in pool
    # the turning slot gets two details; others get one
    by_id = {s["dialogue_slot_id"]: s for s in sks}
    assert len(by_id["d004"]["concrete_detail_required"]) == 2
    assert len(by_id["d001"]["concrete_detail_required"]) == 1


def test_derivation_is_deterministic():
    a = _episode_skeletons()
    b = _episode_skeletons()
    assert a == b


def test_empty_pool_yields_no_details():
    sk = sdc.derive_contract_skeleton(
        slot_row=SLOT_ROWS[0], slot_index=0, dramatic_state=DRAMATIC_STATE,
        active_props=[], key_terms=[],
    )
    assert sk["concrete_detail_required"] == []


# ---------------------------------------------------------------------------
# Validator -- catches each failure mode
# ---------------------------------------------------------------------------


def _good_contract(**overrides):
    base = dict(
        dialogue_slot_id="d004",
        speaker="Maeve",
        line_job="force the signature",
        hidden_pressure="the brother she is protecting",
        concrete_detail_required=["the confession form"],
        state_before="unresolved standoff over the confession",
        state_after="Maeve signs and surrenders the only leverage she had",
        must_turn=True,
    )
    base.update(overrides)
    return sdc.SlotDramaContract(**base)


def test_validator_passes_good_contract():
    ok, reasons = sdc.validate_contract(_good_contract(), ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


def test_validator_rejects_bad_slot_id():
    ok, reasons = sdc.validate_contract(
        {
            "dialogue_slot_id": "X9",
            "speaker": "Maeve",
            "line_job": "x",
            "hidden_pressure": "y",
            "concrete_detail_required": [],
            "state_before": "a",
            "state_after": "b",
            "must_turn": False,
        },
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert not ok
    assert any("schema_invalid" in r for r in reasons)


def test_validator_rejects_empty_field():
    # line_job blank -> schema rejects (min_length=1) -> schema_invalid
    ok, reasons = sdc.validate_contract(
        {
            "dialogue_slot_id": "d001",
            "speaker": "Maeve",
            "line_job": "  ",
            "hidden_pressure": "y",
            "concrete_detail_required": [],
            "state_before": "a",
            "state_after": "a",
            "must_turn": False,
        },
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert not ok
    assert any("schema_invalid" in r or "empty_field" in r for r in reasons)


def test_validator_rejects_detail_not_in_pool():
    ok, reasons = sdc.validate_contract(
        _good_contract(concrete_detail_required=["a smoking pistol"]),
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert not ok
    assert any("concrete_detail_not_in_pool" in r for r in reasons)


def test_validator_rejects_turn_without_state_change():
    ok, reasons = sdc.validate_contract(
        _good_contract(state_before="same", state_after="same"),
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert not ok
    assert any("must_turn_but_state_unchanged" in r for r in reasons)


def test_validator_rejects_turn_without_concrete_detail():
    ok, reasons = sdc.validate_contract(
        _good_contract(concrete_detail_required=[]),
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert not ok
    assert any("must_turn_but_no_concrete_detail" in r for r in reasons)


def test_validator_allows_non_turn_equal_state():
    ok, reasons = sdc.validate_contract(
        _good_contract(
            dialogue_slot_id="d002",
            must_turn=False,
            state_before="held",
            state_after="held",
        ),
        ACTIVE_PROPS, KEY_TERMS,
    )
    assert ok, reasons


# ---------------------------------------------------------------------------
# Episode-level rule -- exactly one costly-choice turn
# ---------------------------------------------------------------------------


def test_episode_requires_exactly_one_turn():
    good = [_good_contract(dialogue_slot_id="d004", must_turn=True)]
    good.append(
        _good_contract(
            dialogue_slot_id="d001", must_turn=False,
            state_before="held", state_after="held",
            concrete_detail_required=["audit"],
        )
    )
    ok, reasons = sdc.validate_episode_contracts(good, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


def test_episode_rejects_zero_turns():
    contracts = [
        _good_contract(
            dialogue_slot_id="d001", must_turn=False,
            state_before="held", state_after="held",
        )
    ]
    ok, reasons = sdc.validate_episode_contracts(contracts, ACTIVE_PROPS, KEY_TERMS)
    assert not ok
    assert any("no slot carries the costly-choice turn" in r for r in reasons)


def test_episode_rejects_two_turns():
    contracts = [
        _good_contract(dialogue_slot_id="d003", must_turn=True),
        _good_contract(dialogue_slot_id="d004", must_turn=True),
    ]
    ok, reasons = sdc.validate_episode_contracts(contracts, ACTIVE_PROPS, KEY_TERMS)
    assert not ok
    assert any("more than one costly-choice turn" in r for r in reasons)


# ---------------------------------------------------------------------------
# LLM pass with a fake generate_fn
# ---------------------------------------------------------------------------


def _fake_generate_ok(messages, *, temperature, max_new_tokens):
    return json.dumps(
        {
            "line_job": "back Maeve into naming the cost out loud",
            "hidden_pressure": "the brother whose freedom hangs on her silence",
        }
    )


def test_generate_slot_job_fields_parses_fake():
    jobs = sdc.generate_slot_job_fields(
        _fake_generate_ok,
        speaker="Doyle",
        dramatic_state=DRAMATIC_STATE,
        state_before="standoff",
        state_after="standoff",
        must_turn=False,
        beat_intent="press for the signature",
        concrete_detail_required=["the confession form"],
    )
    assert jobs.line_job.startswith("back Maeve")
    assert "brother" in jobs.hidden_pressure


def test_build_contract_llm_path_first_try():
    contract, source = sdc.build_slot_drama_contract(
        _fake_generate_ok,
        slot_row=SLOT_ROWS[3],   # d004, the turn
        slot_index=3,
        dramatic_state=DRAMATIC_STATE,
        beat_intent="face the choice",
        active_props=ACTIVE_PROPS,
        key_terms=KEY_TERMS,
    )
    assert source == "llm"
    ok, reasons = sdc.validate_contract(contract, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons
    assert contract.must_turn is True


# ---------------------------------------------------------------------------
# Regenerate-once-then-fallback path
# ---------------------------------------------------------------------------


class _FlakeyGen:
    """Fails the first call, succeeds the second."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.calls += 1
        if self.calls == 1:
            return "this is not json"
        return json.dumps(
            {"line_job": "name the cost", "hidden_pressure": "the brother's freedom"}
        )


def test_regenerate_once_recovers():
    gen = _FlakeyGen()
    contract, source = sdc.build_slot_drama_contract(
        gen,
        slot_row=SLOT_ROWS[3],
        slot_index=3,
        dramatic_state=DRAMATIC_STATE,
        beat_intent="face the choice",
        active_props=ACTIVE_PROPS,
        key_terms=KEY_TERMS,
    )
    assert gen.calls == 2
    assert source == "llm_regenerate"
    ok, _ = sdc.validate_contract(contract, ACTIVE_PROPS, KEY_TERMS)
    assert ok


def _always_bad(messages, *, temperature, max_new_tokens):
    return "never valid json"


def test_falls_back_to_minimal_after_two_failures():
    contract, source = sdc.build_slot_drama_contract(
        _always_bad,
        slot_row=SLOT_ROWS[3],
        slot_index=3,
        dramatic_state=DRAMATIC_STATE,
        beat_intent="face the choice",
        active_props=ACTIVE_PROPS,
        key_terms=KEY_TERMS,
    )
    assert source == "minimal"
    ok, reasons = sdc.validate_contract(contract, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


def test_none_generate_fn_uses_minimal():
    contract, source = sdc.build_slot_drama_contract(
        None,
        slot_row=SLOT_ROWS[0],
        slot_index=0,
        dramatic_state=DRAMATIC_STATE,
        beat_intent="stall the interview",
        active_props=ACTIVE_PROPS,
        key_terms=KEY_TERMS,
    )
    assert source == "minimal"
    ok, reasons = sdc.validate_contract(contract, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


def test_minimal_contract_for_turn_is_valid_and_grounds():
    contract = sdc.build_minimal_contract(
        slot_row=SLOT_ROWS[3],
        slot_index=3,
        dramatic_state=DRAMATIC_STATE,
        beat_intent="face the choice",
        active_props=ACTIVE_PROPS,
        key_terms=KEY_TERMS,
    )
    assert contract.must_turn is True
    assert contract.state_before != contract.state_after
    assert len(contract.concrete_detail_required) >= 1
    ok, reasons = sdc.validate_contract(contract, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


def test_full_episode_via_builder_passes_episode_check():
    contracts = []
    for i, row in enumerate(SLOT_ROWS):
        c, _ = sdc.build_slot_drama_contract(
            _fake_generate_ok,
            slot_row=row,
            slot_index=i,
            dramatic_state=DRAMATIC_STATE,
            beat_intent=row["intent"],
            active_props=ACTIVE_PROPS,
            key_terms=KEY_TERMS,
        )
        contracts.append(c)
    ok, reasons = sdc.validate_episode_contracts(contracts, ACTIVE_PROPS, KEY_TERMS)
    assert ok, reasons


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
