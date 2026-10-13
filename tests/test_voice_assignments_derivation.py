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
    """This surface is character-only by design, regardless of which engine
    delivers the announcer (Kokoro, or -- since 2026-08-24 -- Bark)."""
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


#: The writer's K.5 stamp lives in the writer TAIL since the writer split;
#: the node file is read too, so the stamp cannot move back unwatched.
_WRITER_FILES = ("OTR_LedgerScriptWriter.py", "_otr_writer_tail.py")


def _writer_src(name):
    import pathlib
    return (pathlib.Path(__file__).parent.parent / "nodes" / name).read_text(
        encoding="utf-8")


def test_meta_voice_assignments_not_persisted_by_writer():
    """Sprint 6.2 contract: the writer's K.5 stamp must NOT write
    voice_assignments to meta. Cast is the canonical source.

    Source-level guard over BOTH writer files: this used to read only the node
    file, which stopped holding K.5 when the writer was split, so it watched
    nothing. Searches for the assignment, not the string (the explanatory
    comments name the field).
    """
    import re
    for name in _WRITER_FILES:
        assignments = re.findall(r'meta\[\s*"voice_assignments"\s*\]\s*=',
                                 _writer_src(name))
        assert not assignments, (
            f"{name}: writer K.5 must not persist meta.voice_assignments "
            f"(Sprint 6.2 contract); found {len(assignments)} assignment(s)"
        )


def test_meta_visual_plan_characters_has_no_notes_field():
    """Sprint 6.2 contract: character entries in visual_plan have
    only portrait_prompt -- no notes. Source-level guard.

    The landmark is REQUIRED: this used to skip when it was missing, and it
    skipped on every run once K.5 moved to the writer tail."""
    import re
    src = _writer_src("_otr_writer_tail.py")
    k5_start = src.find("# K.5 -- voice-path-cleanbreak")
    k5_end = src.find("# --- L. Assemble return values", k5_start)
    assert k5_start != -1 and k5_end != -1, (
        "K.5 block landmarks not found in nodes/_otr_writer_tail.py -- "
        "point this guard at wherever the block now lives")
    k5_block = src[k5_start:k5_end]
    # The "notes" key must not appear as a dict assignment in the K.5 block.
    notes_assignments = re.findall(r'"notes"\s*:', k5_block)
    assert not notes_assignments, (
        f"writer K.5 must not stamp 'notes' field on cast entries "
        f"(Sprint 6.2 contract); found {len(notes_assignments)} occurrence(s)"
    )
