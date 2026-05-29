"""Sprint 1 keystone (2026-05-28) -- dialogue_slot_id integrity tests.

Covers the three live touch surfaces (Story Room extract/commit removed
in the 2026-05-29 lean-down):
  1. _otr_outline.Beat + stamp_dialogue_slot_ids
  2. _otr_stage1_plan.Stage1Beat + stamp_dialogue_slot_ids
  3. production_ledger.init_lines_from_outline + set_lines

The 5/5 episode soak gate post-commit verifies live integrity; this
file pins the deterministic invariants subagents can run before push.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from nodes._otr_outline import (
    Beat as OutlineBeat,
    Outline,
    stamp_dialogue_slot_ids as stamp_outline_slot_ids,
)
from nodes._otr_stage1_plan import (
    Stage1Beat,
    Stage1CastMember,
    Stage1Arc,
    Stage1Plan,
    parse_and_validate_plan,
    stamp_dialogue_slot_ids as stamp_stage1_slot_ids,
)
from nodes.production_ledger import Ledger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _outline_with_mixed_beats() -> Outline:
    """Outline with announcer / character / music_inter / sfx / announcer
    so stamping has every shape to test."""
    beats = [
        OutlineBeat(
            beat_id="b001", speaker="ANNOUNCER", speaker_role="announcer",
            intent="open the episode", target_words=15, mood="welcoming",
            arc_phase="setup",
        ),
        OutlineBeat(
            beat_id="b002", speaker="ALICE", speaker_role="character",
            intent="raise the question", target_words=20, mood="wry",
            arc_phase="setup",
        ),
        OutlineBeat(
            beat_id="b003", speaker="NARRATOR", speaker_role="music_inter",
            intent="bridge to the next phase", target_words=5,
            mood="transitional", arc_phase="setup",
        ),
        OutlineBeat(
            beat_id="b004", speaker="BOB", speaker_role="character",
            intent="reveal the obstacle", target_words=25, mood="grave",
            arc_phase="complication",
        ),
        OutlineBeat(
            beat_id="b005", speaker="NARRATOR", speaker_role="sfx",
            intent="footsteps approach off-stage", target_words=5,
            mood="tense", arc_phase="complication",
            sfx_cue="footsteps approach",
        ),
        OutlineBeat(
            beat_id="b006", speaker="ALICE", speaker_role="character",
            intent="make the costly choice", target_words=30,
            mood="resolute", arc_phase="resolution",
        ),
        OutlineBeat(
            beat_id="b007", speaker="ANNOUNCER", speaker_role="announcer",
            intent="close the episode", target_words=15,
            mood="reflective", arc_phase="resolution",
        ),
    ]
    return Outline(
        title="Test Episode",
        premise="A short test premise for slot id stamping verification.",
        setting="Test setting",
        time_of_day="midnight",
        beats=beats,
    )


def _stage1_plan_with_mixed_beats() -> Stage1Plan:
    return Stage1Plan(
        premise="A short test premise for Stage 1 slot id stamping.",
        arc=Stage1Arc(
            setup="setup statement long enough to validate.",
            complication="complication statement long enough to validate.",
            resolution="resolution statement long enough to validate.",
        ),
        cast=[
            Stage1CastMember(
                name="ALICE", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_3",
                persona="weary forensic engineer with dry humor and "
                        "twenty years on the night shift.",
                arc_role="reluctant insider",
            ),
            Stage1CastMember(
                name="BOB", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="ambitious grant officer evasive about funding "
                        "and prone to deflection.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(beat_id="b001", speaker="ANNOUNCER",
                       intent="opening bookend",
                       length_target_words=15,
                       emotional_register="welcoming"),
            Stage1Beat(beat_id="b002", speaker="ALICE",
                       intent="set up the question",
                       length_target_words=20,
                       emotional_register="curious dread"),
            Stage1Beat(beat_id="b003", speaker="MUSIC",
                       intent="bridge",
                       length_target_words=0,
                       emotional_register="transitional"),
            Stage1Beat(beat_id="b004", speaker="BOB",
                       intent="apply pressure",
                       length_target_words=25,
                       emotional_register="tight evasion"),
            Stage1Beat(beat_id="b005", speaker="ALICE",
                       intent="costly choice",
                       length_target_words=30,
                       emotional_register="grim resolve"),
            Stage1Beat(beat_id="b006", speaker="ANNOUNCER",
                       intent="closing bookend",
                       length_target_words=15,
                       emotional_register="quiet aftermath"),
        ],
        running_facts=["the device is in the basement"],
    )


# ---------------------------------------------------------------------------
# 1. _otr_outline.Beat + stamp_dialogue_slot_ids
# ---------------------------------------------------------------------------


def test_outline_beat_accepts_well_formed_slot_id():
    """Beat schema accepts dXXX format and None."""
    b = OutlineBeat(
        beat_id="b001", speaker="ALICE", speaker_role="character",
        intent="open the line", target_words=20, mood="wry",
        dialogue_slot_id="d001",
    )
    assert b.dialogue_slot_id == "d001"

    b2 = OutlineBeat(
        beat_id="b002", speaker="NARRATOR", speaker_role="music_inter",
        intent="bridge into next", target_words=5, mood="transitional",
        dialogue_slot_id=None,
    )
    assert b2.dialogue_slot_id is None


@pytest.mark.parametrize("bad", ["d1", "D001", "d0001", "001", "x001"])
def test_outline_beat_rejects_malformed_slot_id(bad):
    """Pattern validator catches malformed ids."""
    with pytest.raises(ValidationError):
        OutlineBeat(
            beat_id="b001", speaker="ALICE", speaker_role="character",
            intent="open the line", target_words=20, mood="wry",
            dialogue_slot_id=bad,
        )


def test_stamp_outline_slot_ids_assigns_in_voiced_order():
    """d001..dN assigned to character + announcer beats only, in order."""
    out = _outline_with_mixed_beats()
    stamp_outline_slot_ids(out)
    assigned = [(b.beat_id, b.speaker_role, b.dialogue_slot_id)
                for b in out.beats]
    # Voiced beats in declaration order: b001, b002, b004, b006, b007.
    assert assigned == [
        ("b001", "announcer", "d001"),
        ("b002", "character", "d002"),
        ("b003", "music_inter", None),
        ("b004", "character", "d003"),
        ("b005", "sfx", None),
        ("b006", "character", "d004"),
        ("b007", "announcer", "d005"),
    ]


def test_stamp_outline_slot_ids_is_idempotent():
    """Second stamping pass re-stamps from d001 and converges identically."""
    out = _outline_with_mixed_beats()
    stamp_outline_slot_ids(out)
    snapshot = [b.dialogue_slot_id for b in out.beats]
    stamp_outline_slot_ids(out)
    assert [b.dialogue_slot_id for b in out.beats] == snapshot


# ---------------------------------------------------------------------------
# 2. _otr_stage1_plan.Stage1Beat + stamp_dialogue_slot_ids
# ---------------------------------------------------------------------------


def test_stage1beat_accepts_well_formed_slot_id():
    b = Stage1Beat(
        beat_id="b001", speaker="ALICE",
        intent="set up the question",
        length_target_words=20,
        emotional_register="curious dread",
        dialogue_slot_id="d001",
    )
    assert b.dialogue_slot_id == "d001"


def test_stage1_stamp_voiced_predicate_is_speaker_not_music():
    """MUSIC beats stay None; ANNOUNCER + cast names get a slot id."""
    plan = _stage1_plan_with_mixed_beats()
    stamp_stage1_slot_ids(plan)
    pairs = [(b.beat_id, b.speaker, b.dialogue_slot_id) for b in plan.beats]
    # Voiced (speaker != MUSIC): b001/ANN, b002/ALICE, b004/BOB,
    # b005/ALICE, b006/ANN. b003/MUSIC stays None.
    assert pairs == [
        ("b001", "ANNOUNCER", "d001"),
        ("b002", "ALICE",     "d002"),
        ("b003", "MUSIC",     None),
        ("b004", "BOB",       "d003"),
        ("b005", "ALICE",     "d004"),
        ("b006", "ANNOUNCER", "d005"),
    ]


def test_parse_and_validate_plan_stamps_slot_ids():
    """The canonical parse entry point runs the stamping helper."""
    plan = _stage1_plan_with_mixed_beats()
    plan_dict = plan.model_dump()
    parsed = parse_and_validate_plan(plan_dict)
    voiced = [b for b in parsed.beats if b.speaker != "MUSIC"]
    music = [b for b in parsed.beats if b.speaker == "MUSIC"]
    assert all(b.dialogue_slot_id is not None for b in voiced)
    assert all(b.dialogue_slot_id is None for b in music)
    # Ensure d-id sequence is dense, monotonic.
    seq = [b.dialogue_slot_id for b in voiced]
    assert seq == [f"d{i:03d}" for i in range(1, len(seq) + 1)]


# ---------------------------------------------------------------------------
# 3. production_ledger.init_lines_from_outline / set_lines
# ---------------------------------------------------------------------------


def test_init_lines_from_outline_copies_dialogue_slot_id(tmp_path):
    """Slot ids on outline.Beat propagate to ledger lines."""
    out = _outline_with_mixed_beats()
    stamp_outline_slot_ids(out)
    led = Ledger(episode_id="EP-TEST", out_dir=str(tmp_path))
    led.init_lines_from_outline(out, char_id_by_name={
        "ALICE": "alice", "BOB": "bob",
    })
    lines = led.data["lines"]
    by_beat = {ln["beat_id"]: ln for ln in lines}
    assert by_beat["b001"]["dialogue_slot_id"] == "d001"
    assert by_beat["b002"]["dialogue_slot_id"] == "d002"
    assert by_beat["b003"]["dialogue_slot_id"] is None  # music_inter
    assert by_beat["b004"]["dialogue_slot_id"] == "d003"
    assert by_beat["b005"]["dialogue_slot_id"] is None  # sfx
    assert by_beat["b006"]["dialogue_slot_id"] == "d004"
    assert by_beat["b007"]["dialogue_slot_id"] == "d005"


def test_set_lines_preserves_dialogue_slot_id(tmp_path):
    """set_lines is schema-uniform with init_lines_from_outline."""
    led = Ledger(episode_id="EP-TEST", out_dir=str(tmp_path))
    led.set_lines([
        {"line_id": "b001", "beat_id": "b001", "char_id": "announcer",
         "text": "open", "dialogue_slot_id": "d001"},
        {"line_id": "b002", "beat_id": "b002", "char_id": "alice",
         "text": "line", "dialogue_slot_id": "d002"},
    ])
    lines = led.data["lines"]
    assert [ln["dialogue_slot_id"] for ln in lines] == ["d001", "d002"]
