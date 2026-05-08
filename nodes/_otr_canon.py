"""nodes/_otr_canon.py

Character canon layer (Phase 0+ Cast Contract Extensions §3).

The canon describes WHO each character IS (tics, forbidden register, phrase
patterns) -- distinct from the cast contract which describes ROUTING (id,
name, aliases, voice spec). Hard contract = routing; canon = identity.

Status: data model + markdown rendering + file I/O round-trip.
ScriptWriter prompt injection and _check_voice_consistency rubric
integration are DEFERRED to a follow-up session (those touch
story_orchestrator.py which is load-bearing during the in-flight FULL
acceptance run).

Public surface:
    CharacterCanonEntry — dataclass per character
    render_canon_markdown(entries) -> str
    write_canon(episode_dir, entries) -> Path
    load_canon(episode_dir) -> list[CharacterCanonEntry]   ([] if no file)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CANON_FILENAME = "character_canon.md"


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
