# -*- coding: utf-8 -*-
"""A workflow picks one LTX 2.5 lane, not one per role.

WHY THIS TEST EXISTS. `otr_8gb_ltx25_audio_in` shipped with
`character_visual` pinned to the foley lane while `announcer_visual` and
`music_visual` were pinned to the audio-in lane -- three roles, two
different engines, in a workflow whose entire point is "every beat rides
the audio-in lane". Nothing caught it: `build_variants.py --check` only
proves the committed JSON matches what the matrix would regenerate, and
`test_workflow_matrix_is_lossless.py` only proves the matrix matches the
retired per-machine config files it replaced -- neither compares a row
against itself. Fixed 2026-09-25 by pointing `character_visual` at the
same engine as the other two roles; this test is what stops it silently
drifting back apart, since the suite would otherwise stay green whichever
engine `character_visual` names.

SCOPE. Only the LTX 2.5 foley/mime/audio-in family (`_ENGINES` below) is
checked, because that family is the one that promises "one lane, every
role" as its whole reason to exist (foley, mime and audio-in are otherwise
interchangeable weight-sharing siblings on the same tier). A workflow whose
roles legitimately use different kinds of video engine for different beats
-- the audio-reactive `viz_*` engines on the canonical and the low-tier
workflows, for instance -- is a different, deliberate design and is not
what this test is about. The engine ids were renamed 2026-09-25 (dropped
"native" -- 0b2 step 2); this file names them by their current ids.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO / "config" / "workflow_matrix.json"

_VISUAL_ROLE_KEYS = (
    "role_overrides.announcer_visual",
    "role_overrides.character_visual",
    "role_overrides.music_visual",
)

_ENGINES = (
    "ltx25_foley_16gb",
    "ltx25_foley_24gb",
    "ltx25_foley_blackwell",
    "ltx25_mime_16gb",
    "ltx25_mime_24gb",
    "ltx25_audio_in_16gb",
    "ltx25_audio_in_24gb",
)


@pytest.fixture(scope="module")
def matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _family_visual_roles(row: dict) -> dict:
    deltas = row.get("deltas", {})
    return {
        key: deltas[key]
        for key in _VISUAL_ROLE_KEYS
        if key in deltas and deltas[key] in _ENGINES
    }


@pytest.mark.parametrize("row_id", [
    row["id"] for row in json.loads(MATRIX_PATH.read_text(encoding="utf-8"))["rows"]
])
def test_a_row_on_the_ltx25_family_uses_it_for_every_visual_role(row_id, matrix):
    row = next(r for r in matrix["rows"] if r["id"] == row_id)
    roles = _family_visual_roles(row)
    if not roles:
        pytest.skip("row does not select an ltx25 foley/mime/audio-in engine for any visual role")
    engines = set(roles.values())
    assert len(engines) == 1, (
        "%s pins the ltx25 foley/mime/audio-in family to more than one "
        "engine across its visual roles: %r -- this family is one engine "
        "for every role, not one per role." % (row_id, roles)
    )
