"""nodes/_otr_canon.py

Character canon layer (Phase 0+ Cast Contract Extensions §3).

The canon describes WHO each character IS (tics, forbidden register, phrase
patterns) -- distinct from the cast contract which describes ROUTING (id,
name, aliases, voice spec). Hard contract = routing; canon = identity.

Status: data model + markdown rendering + file I/O round-trip.

Public surface:
    CharacterCanonEntry — dataclass per character
    render_canon_markdown(entries) -> str
    write_canon(episode_dir, entries) -> Path
    load_canon(episode_dir) -> list[CharacterCanonEntry]   ([] if no file)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


__all__ = [
    # character canon (existing)
    "CharacterCanonEntry",
    "CANON_FILENAME",
    "render_canon_markdown",
    "write_canon",
    "load_canon",
    # episode canon (Task 2 / v2.0 LedgerScriptWriter)
    "EpisodeCanon",
    "EPISODE_CANON_FILENAME",
    "episode_canon_from_outline_dict",
    "render_episode_canon_header",
    "write_episode_canon",
    "load_episode_canon",
]


CANON_FILENAME = "character_canon.md"
EPISODE_CANON_FILENAME = "episode_canon.json"


@dataclass
class CharacterCanonEntry:
    """One character's identity card. character_id pairs with the cast
    contract's CharacterEntry of the same id.
    """

    character_id: str  # "c02"
    canonical_name: str  # "AEGEUS"
    voice: str = ""  # voice_spec form, e.g. "bark:v2/en_speaker_5"
    tics: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    phrase_pattern: Optional[str] = None


def render_canon_markdown(entries: list[CharacterCanonEntry]) -> str:
    """Render character canon as markdown for ScriptWriter prompt injection.

    Format example:
        ## c02 - AEGEUS
        - Voice: bark:v2/en_speaker_5
        - Tics: clipped, no contractions, marine metaphors
        - Forbidden: military slang
        - Phrase pattern: [Noun] is [verb-ing] back through [place]

    Empty fields are omitted from the output (terser prompt = lower token
    cost = better LLM compliance).
    """
    lines: list[str] = []
    for entry in entries:
        lines.append(f"## {entry.character_id} - {entry.canonical_name}")
        if entry.voice:
            lines.append(f"- Voice: {entry.voice}")
        if entry.tics:
            lines.append(f"- Tics: {', '.join(entry.tics)}")
        if entry.forbidden:
            lines.append(f"- Forbidden: {', '.join(entry.forbidden)}")
        if entry.phrase_pattern:
            lines.append(f"- Phrase pattern: {entry.phrase_pattern}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_canon(episode_dir: Path, entries: list[CharacterCanonEntry]) -> Path:
    """Write character_canon.md to <episode_dir>/character_canon.md."""
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir does not exist: {episode_dir}")
    out = episode_dir / CANON_FILENAME
    out.write_text(render_canon_markdown(entries), encoding="utf-8")
    return out


def load_canon(episode_dir: Path) -> list[CharacterCanonEntry]:
    """Parse character_canon.md back into entries.

    Round-trips with render_canon_markdown for entries that go through
    write_canon. Hand-edited canons are best-effort; unrecognized lines
    are ignored. Returns [] if the file doesn't exist.
    """
    episode_dir = Path(episode_dir)
    in_path = episode_dir / CANON_FILENAME
    if not in_path.is_file():
        return []
    text = in_path.read_text(encoding="utf-8")
    entries: list[CharacterCanonEntry] = []
    cur: Optional[CharacterCanonEntry] = None

    header_re = re.compile(r"^##\s+(\S+)\s+-\s+(.+)$")
    field_re = re.compile(r"^-\s+([A-Za-z][A-Za-z _]*?):\s+(.+)$")

    for raw in text.splitlines():
        line = raw.rstrip()
        m = header_re.match(line)
        if m:
            if cur is not None:
                entries.append(cur)
            cur = CharacterCanonEntry(
                character_id=m.group(1).strip(),
                canonical_name=m.group(2).strip(),
            )
            continue
        if cur is None:
            continue
        m = field_re.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key == "voice":
            cur.voice = val
        elif key == "tics":
            cur.tics = [t.strip() for t in val.split(",") if t.strip()]
        elif key == "forbidden":
            cur.forbidden = [t.strip() for t in val.split(",") if t.strip()]
        elif key == "phrase pattern":
            cur.phrase_pattern = val
    if cur is not None:
        entries.append(cur)
    return entries


# ---------------------------------------------------------------------------
# Episode canon (Task 2 / v2.0 LedgerScriptWriter)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCanon:
    """World-facts canon for a single episode. Sibling to character_canon.md.

    Identity (who characters are) lives in CharacterCanonEntry.
    World facts (what the episode is about) live here. Both write into the
    same episode dir; the line composer reads both at compose time.
    """

    title: str
    premise: str
    setting: str
    time_of_day: str
    sound_palette: list[str] = field(default_factory=list)


def episode_canon_from_outline_dict(outline_data: dict) -> EpisodeCanon:
    """Build EpisodeCanon from a validated outline dict.

    Accepts the dict form of _otr_outline.Outline (after model_dump()) or
    any dict with the required keys. Decoupled from _otr_outline by design
    so this module stays import-clean.

    Required keys: title, premise, setting, time_of_day.
    Optional key:  sound_palette (defaults to []).

    Raises KeyError if a required key is missing.
    """
    return EpisodeCanon(
        title=outline_data["title"],
        premise=outline_data["premise"],
        setting=outline_data["setting"],
        time_of_day=outline_data["time_of_day"],
        sound_palette=list(outline_data.get("sound_palette", [])),
    )


def render_episode_canon_header(canon: EpisodeCanon) -> str:
    """Render EpisodeCanon as a compact prompt header for line composer.

    Compact intentionally: this gets injected into every per-line LLM call,
    so token cost compounds. Sound palette only included when non-empty.
    """
    lines = [
        f"TITLE: {canon.title}",
        f"SETTING: {canon.setting}",
        f"TIME: {canon.time_of_day}",
        f"PREMISE: {canon.premise}",
    ]
    if canon.sound_palette:
        lines.append(f"SOUND PALETTE: {', '.join(canon.sound_palette)}")
    return "\n".join(lines)


def write_episode_canon(episode_dir: Path, canon: EpisodeCanon) -> Path:
    """Write episode_canon.json to <episode_dir>/episode_canon.json.

    Atomic via write-tmp-then-rename pattern. Overwrites existing.
    """
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir does not exist: {episode_dir}")
    out = episode_dir / EPISODE_CANON_FILENAME
    tmp = out.with_suffix(out.suffix + ".tmp")
    payload = {
        "title": canon.title,
        "premise": canon.premise,
        "setting": canon.setting,
        "time_of_day": canon.time_of_day,
        "sound_palette": list(canon.sound_palette),
    }
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    return out


def load_episode_canon(episode_dir: Path) -> Optional[EpisodeCanon]:
    """Load episode_canon.json from episode dir. Returns None if missing.

    Raises ValueError if the file is malformed or missing required keys.
    """
    episode_dir = Path(episode_dir)
    in_path = episode_dir / EPISODE_CANON_FILENAME
    if not in_path.is_file():
        return None
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"episode_canon.json malformed: {exc}") from exc
    try:
        return EpisodeCanon(
            title=data["title"],
            premise=data["premise"],
            setting=data["setting"],
            time_of_day=data["time_of_day"],
            sound_palette=list(data.get("sound_palette", [])),
        )
    except KeyError as exc:
        raise ValueError(f"episode_canon.json missing key: {exc}") from exc


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_canon.py` or `python -m nodes._otr_canon`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tempfile

    print("=== _otr_canon.py self-test (episode canon extensions) ===")

    # Test 1: episode_canon_from_outline_dict happy path.
    print("\n[Test 1] episode_canon_from_outline_dict")
    outline_data = {
        "title": "The Echo Below",
        "premise": "A signal that should not exist starts repeating itself.",
        "setting": "An abandoned observatory",
        "time_of_day": "3 a.m.",
    }
    canon = episode_canon_from_outline_dict(outline_data)
    assert canon.title == "The Echo Below"
    assert canon.sound_palette == []
    print("  PASS")

    # Test 2: missing required key raises KeyError.
    print("\n[Test 2] missing required key")
    try:
        episode_canon_from_outline_dict({"title": "x", "premise": "y", "setting": "z"})
        print("  FAIL: missing time_of_day was accepted")
    except KeyError:
        print("  PASS: KeyError raised")

    # Test 3: write/load round-trip.
    print("\n[Test 3] write_episode_canon / load_episode_canon round-trip")
    with tempfile.TemporaryDirectory() as td:
        ep_dir = Path(td)
        out_path = write_episode_canon(ep_dir, canon)
        assert out_path.is_file()
        loaded = load_episode_canon(ep_dir)
        assert loaded is not None
        assert loaded == canon
        print("  PASS")

    # Test 4: load returns None when file absent.
    print("\n[Test 4] load_episode_canon returns None when absent")
    with tempfile.TemporaryDirectory() as td:
        result = load_episode_canon(Path(td))
        assert result is None
        print("  PASS")

    # Test 5: load raises on malformed JSON.
    print("\n[Test 5] malformed JSON raises ValueError")
    with tempfile.TemporaryDirectory() as td:
        ep_dir = Path(td)
        (ep_dir / EPISODE_CANON_FILENAME).write_text("{not json", encoding="utf-8")
        try:
            load_episode_canon(ep_dir)
            print("  FAIL: malformed JSON accepted")
        except ValueError:
            print("  PASS: ValueError raised")

    # Test 6: render_episode_canon_header shape.
    print("\n[Test 6] render_episode_canon_header")
    header = render_episode_canon_header(canon)
    for required in ("TITLE:", "SETTING:", "TIME:", "PREMISE:"):
        assert required in header, f"missing {required} in header"
    assert "SOUND PALETTE:" not in header  # empty palette omitted
    canon_with_sound = EpisodeCanon(
        title="x", premise="y", setting="z", time_of_day="midnight",
        sound_palette=["wind", "static", "morse code"],
    )
    header2 = render_episode_canon_header(canon_with_sound)
    assert "SOUND PALETTE: wind, static, morse code" in header2
    print("  PASS")

    # Test 7: existing CharacterCanonEntry API untouched.
    print("\n[Test 7] existing API still works")
    entry = CharacterCanonEntry(character_id="c01", canonical_name="AEGEUS")
    md = render_canon_markdown([entry])
    assert "AEGEUS" in md
    print("  PASS")

    print("\n=== Task 2 self-tests passed ===")
