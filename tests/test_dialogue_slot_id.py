"""Sprint 1 keystone (2026-05-28) -- dialogue_slot_id integrity tests.

Covers the three live touch surfaces (Story Room extract/commit removed
in the 2026-05-29 lean-down):
  1. _otr_outline.Beat + stamp_dialogue_slot_ids
  2. production_ledger.init_lines_from_outline + set_lines

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
from nodes.production_ledger import Ledger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _outline_with_mixed_beats() -> Outline:
    """Outline with announcer / character / music_inter / music_close /
    announcer so stamping has every shape to test. (rip-sfx-broll
    2026-07-01: the sfx beat became a second non-voiced music beat.)"""
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
            beat_id="b005", speaker="NARRATOR", speaker_role="music_close",
            intent="closing sting under the resolution", target_words=5,
            mood="tense", arc_phase="complication",
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
        ("b005", "music_close", None),
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
# 2. production_ledger.init_lines_from_outline / set_lines
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
    assert by_beat["b005"]["dialogue_slot_id"] is None  # music_close
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
