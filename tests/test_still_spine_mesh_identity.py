"""A mesh row is chosen by IDENTITY, never by absence.

PBUG-20260811-02, prerequisite half. ``_still_spine_row_for_mesh`` built its
candidate key set from four possibly-empty row fields and then asked
``str(char_id or "") in keys``. A music beat probes with ``char_id == ""`` and a
mesh row stores ``char_id: ""``, so ``"" in keys`` was True for EVERY mesh row
and the loop returned whichever one it reached first -- the last row, by beat
order -- regardless of which beat was asked for.

WHY THIS IS A PREREQUISITE AND NOT A CLEANUP. The still spine's real defect is
that the consumer rewrote its lookup id to ``b000_music_open`` while the
post-audio producer keys rows ``music_opening_001``. The SCENE half fails loud
on that mismatch. The MESH half could not: the empty-string match handed back a
plausible row anyway, so a same-id mesh regression test stays GREEN while the
alias is still wrong. Until this is fixed there is no red state to prove the
mesh half with, and a green there would be read as "the mesh half is fine".

Found by the Antigravity lane of the 2026-08-12 kibitz panel, confirmed
independently by Codex, and proved by execution against the failing episode's
own row shape before any code changed.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import render_driver as rd


def _mesh(beat_id, *, subject="", object_id="", char_id=""):
    return {"kind": "mesh_fodder", "beat_id": beat_id,
            "mesh_subject_id": subject, "object_id": object_id,
            "char_id": char_id, "path": "%s.png" % (object_id or beat_id)}


def test_an_empty_char_id_does_NOT_match_a_row_it_does_not_name():
    """The exact live shape: two music mesh rows, both with empty char_id, and a
    music beat probing with an empty char_id. Before the fix this returned the
    LAST row by the empty string, so asking for a beat that has no row at all
    still produced one."""
    rows = [_mesh("music_closing_001", subject="radio_host",
                  object_id="meshfodder_music_closing_001"),
            _mesh("music_opening_001", subject="radio_host",
                  object_id="meshfodder_music_opening_001")]

    assert rd._still_spine_row_for_mesh(rows, "b000_music_open", "") is None


def test_the_beat_that_IS_named_still_resolves():
    """The fix must not cost the ordinary case."""
    rows = [_mesh("music_closing_001", subject="radio_host",
                  object_id="meshfodder_music_closing_001"),
            _mesh("music_opening_001", subject="radio_host",
                  object_id="meshfodder_music_opening_001")]

    row = rd._still_spine_row_for_mesh(rows, "music_opening_001", "")
    assert row is not None
    assert row["beat_id"] == "music_opening_001"


def test_a_probe_with_NO_truthy_identity_at_all_returns_none():
    """No beat and no character is not a question; it must not be answered."""
    rows = [_mesh("l001", subject="radio_host", object_id="meshfodder_l001")]

    assert rd._still_spine_row_for_mesh(rows, "", "") is None
    assert rd._still_spine_row_for_mesh(rows, None, None) is None


def test_a_shared_SUBJECT_id_still_matches_because_that_keying_is_deliberate():
    """Subject keying is not the bug and must survive. The recurring radio mesh
    deliberately shares one subject identity across beats, so a character probe
    resolves through ``mesh_subject_id`` even when the beat ids differ."""
    rows = [_mesh("l001", subject="radio_host", object_id="meshfodder_l001")]

    row = rd._still_spine_row_for_mesh(rows, "l999", "radio_host")
    assert row is not None
    assert row["mesh_subject_id"] == "radio_host"


def test_a_decoy_row_LAST_cannot_win_by_ordering():
    """The loop walks rows in reverse, so a wrong row placed last was the one an
    empty probe used to return. Position must not decide identity."""
    rows = [_mesh("music_opening_001", subject="radio_host",
                  object_id="meshfodder_music_opening_001"),
            _mesh("", subject="", object_id="", char_id="")]

    assert rd._still_spine_row_for_mesh(rows, "b000_music_open", "") is None
    got = rd._still_spine_row_for_mesh(rows, "music_opening_001", "")
    assert got is not None and got["beat_id"] == "music_opening_001"


@pytest.mark.parametrize("field", ["mesh_subject_id", "object_id", "char_id"])
def test_an_empty_ROW_field_is_never_a_match_key(field):
    """Emptiness in the ROW is as dangerous as emptiness in the probe: either
    side alone was enough to make ``"" in keys`` true."""
    row = _mesh("l001", subject="radio_host", object_id="meshfodder_l001")
    row[field] = ""

    assert rd._still_spine_row_for_mesh([row], "", "") is None
