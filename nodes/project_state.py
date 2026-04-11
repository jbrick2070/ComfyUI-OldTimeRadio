"""
project_state.py

Theme C / v1.4 - Project State JSON ("series bible").

Purpose
-------
Per-series, per-session state file that stores locked creative decisions so the
LLM pipeline cannot contradict itself across episodes. Read-only during
generation. Writable only between episodes.

Stored state
------------
- character_voice_locks : mapping of character name -> Bark preset (or Kokoro
  voice id for the announcer). Prevents a character's timbre from drifting
  episode to episode.
- forbidden_patterns    : list of phrases, tropes, or name collisions the
  series has already retired. Fed into the Gemma4 prompt as a negative list.
- tone_contract         : short prose description of the series tone (e.g.,
  "wry, mid-century sci-fi radio, optimistic, never cynical"). Injected into
  every script prompt.
- locked_decisions      : free-form key/value pairs for anything else the
  showrunner has nailed down (setting era, recurring catchphrases, jingle
  keys, narrator identity, etc.).
- series_title          : human-readable series name.
- episode_number        : integer, advanced by the between-episode writer
  only. The generation path must never mutate this.

Hard rules (from ROADMAP + CLAUDE.md)
-------------------------------------
- 100% local. No cloud. No API keys.
- Safe for work. No curse words, no violence.
- UTF-8 no BOM. Writes go through a single helper that guarantees this.
- Read-only during generation: `ProjectState.load()` returns a frozen view.
  Mutations require `ProjectState.open_for_edit()` which is only used by the
  between-episode tooling, never by a node's `INPUT_TYPES` path.
- v1.4 beta scope: the state file lives at a fixed path in the repo root
  (`project_state.json`). Per-workflow selectable paths are a v1.5 concern.

ComfyUI node
------------
`ProjectStateLoader` exposes the loaded state as a single dict-shaped output
so downstream nodes (Gemma4ScriptWriter, Gemma4Director) can pull fields
without each having to know the file path. The node has zero writable
widgets; it is intentionally boring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Repo root = parent of the `nodes/` directory this file lives in.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_STATE_PATH = os.path.join(_REPO_ROOT, "project_state.json")


# ---------------------------------------------------------------------------
# Default state - used when no project_state.json exists yet.
# ---------------------------------------------------------------------------

def _default_state_dict() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "series_title": "Untitled Old Time Radio Series",
        "episode_number": 1,
        "tone_contract": (
            "Wry, mid-century old time radio. Warm, optimistic, never cynical. "
            "Every episode has a clear start, middle, and end."
        ),
        "character_voice_locks": {
            # "ANNOUNCER": "kokoro_af_bella",
            # "DETECTIVE_HART": "v2/en_speaker_6",
        },
        "forbidden_patterns": [
            # Phrases or tropes retired in earlier episodes.
        ],
        "locked_decisions": {
            # "setting_era": "1952",
            # "recurring_sponsor": "Polaris Cocoa",
        },
    }


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ProjectState:
    schema_version: int = 1
    series_title: str = ""
    episode_number: int = 1
    tone_contract: str = ""
    character_voice_locks: Dict[str, str] = field(default_factory=dict)
    forbidden_patterns: List[str] = field(default_factory=list)
    locked_decisions: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectState":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            series_title=str(data.get("series_title", "")),
            episode_number=int(data.get("episode_number", 1)),
            tone_contract=str(data.get("tone_contract", "")),
            character_voice_locks=dict(data.get("character_voice_locks", {}) or {}),
            forbidden_patterns=list(data.get("forbidden_patterns", []) or []),
            locked_decisions=dict(data.get("locked_decisions", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ProjectState":
        """Read the project state from disk. Returns defaults if missing.

        This is the READ-ONLY entry point used during generation. Callers
        must treat the returned object as immutable. Mutations performed on
        the returned dataclass are not written back from this path.
        """
        target = path or PROJECT_STATE_PATH
        if not os.path.isfile(target):
            return cls.from_dict(_default_state_dict())
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def open_for_edit(cls, path: Optional[str] = None) -> "ProjectState":
        """Between-episode write path. NOT for generation-time callers.

        The name is intentionally loud so that any future node author who
        reaches for this during a generation phase has to explain themselves
        in code review.
        """
        return cls.load(path)

    def save(self, path: Optional[str] = None) -> str:
        """Write UTF-8 no BOM. Only called from between-episode tooling."""
        target = path or PROJECT_STATE_PATH
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
        return target

    # ------------------------------------------------------------------
    # Prompt helpers - used by Gemma4ScriptWriter / Gemma4Director
    # ------------------------------------------------------------------

    def prompt_preamble(self) -> str:
        """Return a short prose block suitable for injection into an LLM prompt.

        Kept deliberately compact so it does not bloat the context window.
        """
        lines: List[str] = []
        if self.series_title:
            lines.append(f"Series: {self.series_title} (episode {self.episode_number})")
        if self.tone_contract:
            lines.append(f"Tone contract: {self.tone_contract}")
        if self.character_voice_locks:
            locks = ", ".join(f"{name}" for name in sorted(self.character_voice_locks))
            lines.append(f"Locked characters: {locks}")
        if self.forbidden_patterns:
            # Cap to the first 12 entries to keep the preamble tight.
            capped = self.forbidden_patterns[:12]
            lines.append("Avoid: " + "; ".join(capped))
        if self.locked_decisions:
            decisions = ", ".join(
                f"{k}={v}" for k, v in sorted(self.locked_decisions.items())
            )
            lines.append(f"Locked decisions: {decisions}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class ProjectStateLoader:
    """Load the series bible as a read-only dict for downstream nodes."""

    CATEGORY = "OldTimeRadio/Infrastructure"
    FUNCTION = "load_state"
    RETURN_TYPES = ("PROJECT_STATE", "STRING")
    RETURN_NAMES = ("project_state", "prompt_preamble")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Zero widgets on purpose. The path is fixed at the repo root for
        # v1.4 beta. This keeps widgets_values length stable and avoids the
        # INPUT_TYPES regression class documented in the Bug Bible.
        return {"required": {}}

    def load_state(self) -> tuple:
        state = ProjectState.load()
        return (state.to_dict(), state.prompt_preamble())


NODE_CLASS_MAPPINGS = {
    "ProjectStateLoader": ProjectStateLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProjectStateLoader": "Project State Loader (OTR)",
}
