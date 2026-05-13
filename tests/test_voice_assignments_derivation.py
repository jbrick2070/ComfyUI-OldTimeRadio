"""tests/test_voice_assignments_derivation.py

Voice-path-cleanbreak Sprint 6.2 (2026-05-12). Per-character voice
data is derived at render time from led["cast"]; meta.voice_assignments
must NOT be persisted by the writer.

Contract under test:
  - voice_assignments_from_cast() returns the {<name>: {voice_preset}}
    shape, ANNOUNCER excluded.
  - notes field is absent from the derived shape.
  - Empty cast -> empty dict (graceful).
  - Non-dict cast rows are skipped silently.
"""
from __future__ import annotations

import pytest

from nodes import _otr_ledger_consumers as _OTRLC


def test_voice_assignments_from_cast_basic():
    led = {
        "cast": [
            {"char_id": "c01", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"},
            {"char_id": "c02", "name": "ASTRA", "voice_preset": "v2/en_speaker_4"},
        ],
    }
    result = _OTRLC.voice_assignments_from_cast(led)
    assert result == {
        "LEMMY": {"voice_preset": "v2/en_speaker_8"},
        "ASTRA": {"voice_preset": "v2/en_speaker_4"},
    }


def test_voice_assignments_excludes_announcer():
    """ANNOUNCER lives in the Kokoro namespace; not part of the Bark
    voice_assignments surface."""
    led = {
        "cast": [
            {"char_id": "announcer", "name": "ANNOUNCER", "voice_preset": "bm_george"},
            {"char_id": "c01", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"},
        ],
    }
    result = _OTRLC.voice_assignments_from_cast(led)
    assert "ANNOUNCER" not in result
    assert "LEMMY" in result


def test_voice_assignments_no_notes_field():
    """Sprint 6.2 retired the notes field. character_description lives
    on visual_plan.characters[name].portrait_prompt only."""
    led = {
        "cast": [
            {
                "char_id": "c01",
                "name": "LEMMY",
                "voice_preset": "v2/en_speaker_8",
                "character_description": "Worn space-trucker, baritone",
            },
        ],
    }
    result = _OTRLC.voice_assignments_from_cast(led)
    assert result["LEMMY"] == {"voice_preset": "v2/en_speaker_8"}
    assert "notes" not in result["LEMMY"]


def test_voice_assignments_empty_cast():
    """No cast rows -> empty dict, not an error."""
    assert _OTRLC.voice_assignments_from_cast({}) == {}
    assert _OTRLC.voice_assignments_from_cast({"cast": []}) == {}
    assert _OTRLC.voice_assignments_from_cast({"cast": None}) == {}


def test_voice_assignments_skips_malformed_rows():
    """Non-dict rows are skipped silently (don't crash)."""
    led = {
        "cast": [
            {"char_id": "c01", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"},
            "not a dict",
            None,
            {"char_id": "c02"},  # no name
            {"name": "BABA", "voice_preset": "v2/en_speaker_3"},
        ],
    }
    result = _OTRLC.voice_assignments_from_cast(led)
    assert result == {
        "LEMMY": {"voice_preset": "v2/en_speaker_8"},
        "BABA":  {"voice_preset": "v2/en_speaker_3"},
    }


def test_voice_assignments_missing_voice_preset_yields_empty_string():
    """Cast row without voice_preset gets empty string. The cast contract
    Gate 1 (writer cast-lock exit) catches missing presets at the source;
    this helper doesn't re-validate."""
    led = {
        "cast": [
            {"char_id": "c01", "name": "GHOST"},  # no voice_preset
        ],
    }
    result = _OTRLC.voice_assignments_from_cast(led)
    assert result == {"GHOST": {"voice_preset": ""}}


def test_meta_voice_assignments_not_persisted_by_writer():
    """Sprint 6.2 contract: the writer's K.5 stamp must NOT write
    voice_assignments to meta. Cast is the canonical source.

    This test exercises the writer's K.5 logic by inspecting source
    text (a future writer change that re-introduces persisted
    meta.voice_assignments would surface here). The shape-level
    integration test lives in tests/test_lfc_*.py against the
    real writer fixtures.
    """
    import pathlib
    src_path = (
        pathlib.Path(__file__).parent.parent
        / "nodes" / "OTR_LedgerScriptWriter.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # The K.5 block (Sprint 6.2) must NOT contain meta["voice_assignments"] = ...
    # Search for the assignment pattern, not just the string (the explanatory
    # comments name the field).
    import re
    assignments = re.findall(r'meta\[\s*"voice_assignments"\s*\]\s*=', src)
    assert not assignments, (
        f"writer K.5 must not persist meta.voice_assignments "
        f"(Sprint 6.2 contract); found {len(assignments)} assignment(s)"
    )


def test_meta_visual_plan_characters_has_no_notes_field():
    """Sprint 6.2 contract: character entries in visual_plan have
    only portrait_prompt -- no notes. Source-level guard."""
    import pathlib, re
    src_path = (
        pathlib.Path(__file__).parent.parent
        / "nodes" / "OTR_LedgerScriptWriter.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # Find the K.5 block and look for "notes":
    k5_start = src.find("# K.5 -- voice-path-cleanbreak")
    k5_end_marker = "# --- L. Assemble return values"
    k5_end = src.find(k5_end_marker, k5_start)
    if k5_start == -1 or k5_end == -1:
        pytest.skip("K.5 block landmark not found; skip source-level guard")
    k5_block = src[k5_start:k5_end]
    # The "notes" key must not appear as a dict assignment in the K.5 block.
    notes_assignments = re.findall(r'"notes"\s*:', k5_block)
    assert not notes_assignments, (
        f"writer K.5 must not stamp 'notes' field on cast entries "
        f"(Sprint 6.2 contract); found {len(notes_assignments)} occurrence(s)"
    )
