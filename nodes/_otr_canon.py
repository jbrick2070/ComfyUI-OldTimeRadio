"""nodes/_otr_canon.py

Episode canon: the world facts (title, premise, setting, time of day, sound
palette) the writer pulls from the outline, writes beside the episode, and
injects as a compact header into each per-line prompt.

Public surface:
    EpisodeCanon -- dataclass per episode
    episode_canon_from_outline_dict(outline_data) -> EpisodeCanon
    render_episode_canon_header(canon) -> str
    write_episode_canon(episode_dir, canon) -> Path
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


__all__ = [
    "EpisodeCanon",
    "EPISODE_CANON_FILENAME",
    "episode_canon_from_outline_dict",
    "render_episode_canon_header",
    "write_episode_canon",
]


EPISODE_CANON_FILENAME = "episode_canon.json"


# ---------------------------------------------------------------------------
# Episode canon (Task 2 / v2.0 LedgerScriptWriter)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCanon:
    """World-facts canon for a single episode (what the episode is about)."""

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
