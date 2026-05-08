"""tests/test_canon.py — character canon §3 tests."""
from __future__ import annotations

from pathlib import Path

from nodes._otr_canon import (
    CANON_FILENAME,
    CharacterCanonEntry,
    load_canon,
    render_canon_markdown,
    write_canon,
)


def test_render_full_entry():
    e = CharacterCanonEntry(
        character_id="c02",
        canonical_name="AEGEUS",
        voice="bark:v2/en_speaker_5",
        tics=["clipped", "no contractions", "marine metaphors"],
        forbidden=["military slang"],
        phrase_pattern="[Noun] is [verb-ing] back through [place]",
    )
    md = render_canon_markdown([e])
    assert "## c02 - AEGEUS" in md
    assert "- Voice: bark:v2/en_speaker_5" in md
    assert "- Tics: clipped, no contractions, marine metaphors" in md
    assert "- Forbidden: military slang" in md
    assert "- Phrase pattern: [Noun] is [verb-ing] back through [place]" in md


def test_render_omits_empty_fields():
    e = CharacterCanonEntry(character_id="c01", canonical_name="MONTY")
    md = render_canon_markdown([e])
    assert "## c01 - MONTY" in md
    assert "- Voice:" not in md
    assert "- Tics:" not in md
    assert "- Forbidden:" not in md
    assert "- Phrase pattern:" not in md


def test_render_multiple_entries_ordered():
    a = CharacterCanonEntry("c01", "MONTY", voice="bark:v2/en_speaker_3")
    b = CharacterCanonEntry("c02", "AEGEUS", voice="bark:v2/en_speaker_5")
    md = render_canon_markdown([a, b])
    assert md.find("c01 - MONTY") < md.find("c02 - AEGEUS")


def test_write_canon_creates_file(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    e = CharacterCanonEntry("c01", "MONTY", voice="bark:v2/en_speaker_3")
    out = write_canon(episode_dir, [e])
    assert out.is_file()
    assert out.name == CANON_FILENAME
    assert "MONTY" in out.read_text(encoding="utf-8")


def test_load_canon_returns_empty_when_missing(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    assert load_canon(episode_dir) == []


def test_round_trip_full_canon(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    original = [
        CharacterCanonEntry(
            character_id="c02",
            canonical_name="AEGEUS",
            voice="bark:v2/en_speaker_5",
            tics=["clipped", "no contractions"],
            forbidden=["military slang"],
            phrase_pattern="[Noun] is [verb-ing] back through [place]",
        ),
    ]
    write_canon(episode_dir, original)
    loaded = load_canon(episode_dir)
    assert len(loaded) == 1
    assert loaded[0].character_id == "c02"
    assert loaded[0].canonical_name == "AEGEUS"
    assert loaded[0].voice == "bark:v2/en_speaker_5"
    assert loaded[0].tics == ["clipped", "no contractions"]
    assert loaded[0].forbidden == ["military slang"]
    assert loaded[0].phrase_pattern == "[Noun] is [verb-ing] back through [place]"


def test_round_trip_multiple_entries(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    original = [
        CharacterCanonEntry("c01", "MONTY", voice="bark:v2/en_speaker_3"),
        CharacterCanonEntry("c02", "SAILOR", voice="bark:v2/en_speaker_9"),
        CharacterCanonEntry("c03", "AEGEUS", voice="bark:v2/en_speaker_5"),
    ]
    write_canon(episode_dir, original)
    loaded = load_canon(episode_dir)
    assert [e.character_id for e in loaded] == ["c01", "c02", "c03"]
    assert [e.canonical_name for e in loaded] == ["MONTY", "SAILOR", "AEGEUS"]
