"""nodes/_otr_cast_contract.py

Cast Contract data model + serialization.
Phase 0+ Cast Contract Extensions §1 (versioning) + §2 (episode lock).

See ROADMAP.md "Phase 0+ candidates" -> "Cast Contract Extensions" for the
full design. Source patterns: NousResearch/autonovel state.json.

Status: §1 + §2 implementation. Not yet wired into story_orchestrator.py
or production_ledger.py — that's the integration step (deferred per the
in-flight FULL acceptance run discipline).

Public surface:
    CharacterEntry — single character (id, canonical name, aliases, voice spec)
    CastContract  — collection + content-addressed sha version
    lock_to_episode(contract, episode_dir) -> Path  (§2)
    load_locked(episode_dir) -> CastContract | None  (§2)

Module deliberately has zero dependencies beyond stdlib so it can be
imported from any node without VRAM/torch coupling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CharacterEntry:
    """Single character within a CastContract.

    voice_spec uses 'engine:preset' form -- see _otr_voice_resolver.py for
    parsing. Examples: 'bark:v2/en_speaker_5', 'kokoro:bm_fable'.
    """

    character_id: str  # "c01", "c02", ...
    canonical_name: str  # "MONTY"
    aliases: list[str] = field(default_factory=list)  # ["MONTGOMERY"]
    voice_spec: str = ""  # "bark:v2/en_speaker_5"

    def matches(self, name: str) -> bool:
        """True if name matches canonical_name or any alias (case-insensitive)."""
        u = name.upper()
        if u == self.canonical_name.upper():
            return True
        return any(u == a.upper() for a in self.aliases)


@dataclass
class CastContract:
    """A versioned cast roster.

    The version field is content-addressed: same characters -> same version
    (regardless of insertion order, alias order, or whitespace). Callers can
    compare cast_contract_version across ledger entries to detect drift in
    O(1) string comparison.
    """

    characters: list[CharacterEntry] = field(default_factory=list)
    version: str = ""  # "sha:HEX..."

    def stamp_version(self) -> str:
        """Recompute version from characters and assign. Returns the new version.

        Order-independent: characters sorted by character_id, aliases sorted
        alphabetically before hashing.
        """
        normalized = sorted(
            (
                {
                    "character_id": c.character_id,
                    "canonical_name": c.canonical_name,
                    "aliases": sorted(c.aliases),
                    "voice_spec": c.voice_spec,
                }
                for c in self.characters
            ),
            key=lambda d: d["character_id"],
        )
        blob = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
        sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
        self.version = f"sha:{sha}"
        return self.version

    def lookup(self, name: str) -> Optional[CharacterEntry]:
        """Return the first character whose canonical_name or aliases match.

        Case-insensitive. Returns None if no match.
        """
        for c in self.characters:
            if c.matches(name):
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "characters": [asdict(c) for c in self.characters],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CastContract":
        chars = [CharacterEntry(**c) for c in data.get("characters", [])]
        return cls(characters=chars, version=data.get("version", ""))


# ---------- §2: lock to disk ---------------------------------------------

LOCKED_FILENAME = "cast_contract.locked.json"


def lock_to_episode(contract: CastContract, episode_dir: Path) -> Path:
    """§2: Freeze contract into the episode workspace. Immutable.

    Writes <episode_dir>/cast_contract.locked.json. Refuses to overwrite an
    existing locked file (raises RuntimeError) -- once locked, the episode's
    contract is final. Stamps version if the contract doesn't already have
    one.
    """
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir does not exist: {episode_dir}")
    locked_path = episode_dir / LOCKED_FILENAME
    if locked_path.exists():
        raise RuntimeError(
            f"cast contract already locked at {locked_path}; "
            "refusing to overwrite (immutable per §2)"
        )
    if not contract.version:
        contract.stamp_version()
    locked_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return locked_path


def load_locked(episode_dir: Path) -> Optional[CastContract]:
    """Load the locked cast contract for an episode, or None if not locked."""
    episode_dir = Path(episode_dir)
    locked_path = episode_dir / LOCKED_FILENAME
    if not locked_path.is_file():
        return None
    data = json.loads(locked_path.read_text(encoding="utf-8"))
    return CastContract.from_dict(data)
