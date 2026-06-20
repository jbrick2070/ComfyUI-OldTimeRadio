"""Tests for build_line_dramatic_fields -- story-quality Phase 1, A5
(delivering the news-driven slot contract onto the line composer's per-line
dramatic fields)."""

from nodes._otr_slot_drama_contract import build_line_dramatic_fields as F


_DS = {
    "character_a_wants": "expose the reactor leak to the town",
    "character_b_wants": "keep the reactor leak quiet to avoid panic",
    "dramatic_question": "Will the reactor leak reach the public?",
    "ending_change": "The reactor leak is made public.",
    "costly_choice_beat": "d003",
}


def _contract(**over):
    base = {
        "dialogue_slot_id": "d002",
        "speaker": "EDNA",
        "line_job": "press Marcus to admit the leak",
        "hidden_pressure": "she already knows he signed off on it",
        "concrete_detail_required": ["reactor leak"],
        "state_before": "guarded",
        "state_after": "guarded",
        "must_turn": False,
    }
    base.update(over)
    return base


def test_objective_and_subtext_mapping():
    out = F(_contract(), _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert "press Marcus to admit the leak" in out["beat_objective"]
    assert out["beat_subtext"] == "she already knows he signed off on it"


def test_concrete_detail_folded_into_objective():
    out = F(_contract(concrete_detail_required=["reactor leak", "signed memo"]),
            _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert "reactor leak" in out["beat_objective"]
    assert "signed memo" in out["beat_objective"]


def test_obstacle_is_the_opposing_leads_want():
    # speaker == A -> obstacle is B's want
    out_a = F(_contract(), _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert out_a["beat_obstacle"] == _DS["character_b_wants"]
    # speaker == B -> obstacle is A's want
    out_b = F(_contract(speaker="MARCUS"), _DS,
              speaker="MARCUS", a_name="EDNA", b_name="MARCUS")
    assert out_b["beat_obstacle"] == _DS["character_a_wants"]


def test_non_lead_speaker_gets_antagonist_pressure():
    out = F(_contract(speaker="DEPUTY"), _DS,
            speaker="DEPUTY", a_name="EDNA", b_name="MARCUS")
    assert out["beat_obstacle"] == _DS["character_b_wants"]


def test_turn_only_on_must_turn_with_real_shift():
    no_turn = F(_contract(must_turn=False, state_before="x", state_after="y"),
                _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert no_turn["beat_turn"] == ""
    same = F(_contract(must_turn=True, state_before="x", state_after="x"),
             _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert same["beat_turn"] == ""
    turn = F(_contract(must_turn=True, state_before="hiding it",
                       state_after="forced to choose"),
             _DS, speaker="EDNA", a_name="EDNA", b_name="MARCUS")
    assert "hiding it" in turn["beat_turn"] and "forced to choose" in turn["beat_turn"]


def test_bad_input_never_raises():
    out = F(None, None, speaker="X")
    assert out == {"beat_objective": "", "beat_obstacle": "",
                   "beat_turn": "", "beat_subtext": ""}
