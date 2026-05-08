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


# ---------- §1 helpers: build + alias detection --------------------------

DEFAULT_VOICE_ENGINE = "bark"
ALIAS_PREFIX_LEN = 4  # min shared prefix to consider name an alias of canonical


def _coerce_voice_spec(raw) -> str:
    """Normalize a voice-assignment value into 'engine:preset' form.

    Accepts:
      - str with colon: "bark:v2/en_speaker_3" -> returned as-is (trimmed)
      - str without colon: "v2/en_speaker_3" -> "bark:v2/en_speaker_3"
      - dict {"engine": "bark", "preset": "..."}: assembled into "bark:..."
      - dict {"preset": "..."} (no engine): defaults to bark
      - empty/None -> "" (caller decides whether to fail)
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        engine = (raw.get("engine") or DEFAULT_VOICE_ENGINE).strip().lower()
        preset = (raw.get("preset") or "").strip()
        if not preset:
            return ""
        return f"{engine}:{preset}"
    s = str(raw).strip()
    if not s:
        return ""
    if ":" in s:
        return s
    return f"{DEFAULT_VOICE_ENGINE}:{s}"


def build_contract_from_director_plan(director_plan: dict) -> CastContract:
    """Build a versioned CastContract from a Director plan.

    Reads ``director_plan["voice_assignments"]`` (a mapping from canonical
    character name to a voice spec). Characters are emitted in
    canonical-name alphabetical order, each assigned a stable character_id
    of ``c01``, ``c02``, ... in that sort order. Voice specs without an
    ``engine:`` prefix default to ``bark:`` (current OTR baseline). The
    resulting contract is version-stamped before return.

    Empty / missing ``voice_assignments`` returns an empty CastContract
    (caller's responsibility to decide whether that's a hard failure).
    """
    plan = director_plan or {}
    assignments = plan.get("voice_assignments") or {}
    if not isinstance(assignments, dict):
        raise TypeError(
            "director_plan['voice_assignments'] must be a dict; got "
            f"{type(assignments).__name__}"
        )

    # Normalize keys ONCE and rebuild a clean lookup. Indexing the original
    # dict with a stripped key would KeyError on padded inputs like
    # ``{" MONTY ": ...}`` -- caught by 2026-05-08 round-robin code review.
    clean_assignments: dict[str, object] = {}
    for raw_name, raw_value in assignments.items():
        stripped = str(raw_name).strip()
        if not stripped:
            continue
        # First occurrence wins on collisions ("MONTY" + "  MONTY  " ->
        # the first one keeps its value); the alternative would be silently
        # overwriting, which is worse.
        clean_assignments.setdefault(stripped, raw_value)

    sorted_names = sorted(clean_assignments.keys(), key=str.upper)

    characters: list[CharacterEntry] = []
    for idx, name in enumerate(sorted_names, start=1):
        cid = f"c{idx:02d}"
        voice_spec = _coerce_voice_spec(clean_assignments[name])
        characters.append(
            CharacterEntry(
                character_id=cid,
                canonical_name=name.upper(),
                aliases=[],
                voice_spec=voice_spec,
            )
        )

    contract = CastContract(characters=characters)
    contract.stamp_version()
    return contract


def _extract_dialogue_tags(script: str) -> list[str]:
    """Scan a script string and return every uppercase dialogue tag found.

    Matches ``CHARACTER:`` at the start of a line (after optional
    whitespace), where CHARACTER is a run of letters (and ``.`` / ``'`` /
    space / ``-``). Returns the unique tag strings preserved in
    first-seen order so the caller can iterate deterministically.

    The OTR convention is ``MONTY:``, ``SAILOR:``, etc. -- this matches
    that and the slightly noisier ``MONTY T.:`` form.
    """
    if not script:
        return []
    seen: dict[str, None] = {}
    for raw_line in script.splitlines():
        line = raw_line.lstrip()
        if not line:
            continue
        # Find the first colon, take what's before it as the candidate tag.
        colon = line.find(":")
        if colon <= 0 or colon > 60:
            continue
        candidate = line[:colon].strip()
        if not candidate:
            continue
        # Tag must be majority uppercase letters; allows MONTY, MONTY T.,
        # AEGEUS, etc. Reject narration headers like 'Scene 1' or 'Act II'.
        letters = [c for c in candidate if c.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio < 0.8:
            continue
        # Skip a small set of known structural tags. Match against the
        # first whitespace-delimited token so "SCENE 1" is filtered just
        # like "SCENE".
        first_token = candidate.split()[0].upper() if candidate.split() else ""
        if first_token in {"SCENE", "ACT", "FADE", "INT", "EXT", "NARRATOR"}:
            continue
        seen.setdefault(candidate.upper(), None)
    return list(seen.keys())


def detect_aliases(script: str, contract: CastContract) -> dict[str, str]:
    """Heuristic alias detector.

    Scans every uppercase dialogue tag in ``script``. For each tag that is
    not already a canonical name or registered alias in ``contract``,
    checks whether it shares a leading prefix of at least
    ``ALIAS_PREFIX_LEN`` characters with any canonical name (in either
    direction -- handles both truncation, e.g. MONTGOMERY -> MONTY, and
    expansion, e.g. MONTY -> MONTGOMERY).

    Returns a dict mapping ``{alias_seen_in_script: character_id}``.
    Does NOT mutate the contract; the caller decides whether to apply
    each alias (and re-stamp the version) or escalate to §4 adversarial
    classification for ambiguous tags.

    Pure heuristic only -- no LLM call. False positives are possible
    when two canonical names share a long prefix (e.g. MARLA / MARLON);
    in that case the FIRST canonical match in iteration order wins, and
    §4 is the canonical place to disambiguate.
    """
    if not isinstance(contract, CastContract):
        raise TypeError("contract must be a CastContract")
    aliases_found: dict[str, str] = {}
    if not script:
        return aliases_found

    for tag in _extract_dialogue_tags(script):
        # Already known? Skip.
        if contract.lookup(tag) is not None:
            continue
        tag_u = tag.upper()
        for character in contract.characters:
            cn = character.canonical_name.upper()
            if not cn:
                continue
            # Either direction: tag startswith cn[:N] or cn startswith tag[:N]
            n = ALIAS_PREFIX_LEN
            if len(tag_u) >= n and len(cn) >= n and (
                tag_u[:n] == cn[:n]
            ):
                aliases_found[tag] = character.character_id
                break
    return aliases_found
