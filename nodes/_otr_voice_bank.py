"""Voice reference bank + deterministic caster (plan E.1 / E.2, Wave 1 / 1d).

Two surfaces live here, both BESIDE the frozen legacy caster (never replacing
it -- I-4 keeps the two casters on disjoint RNGs):

  * **The bank** -- ``config/voice_reference_bank.json`` is a set of castable
    voice references, each coding to ``config/voice_bank_entry_schema.json``
    (the Wave-0 E.1 schema). ``load_voice_bank`` validates every entry against
    that schema (dependency-free), rejects duplicate ``voice_ref_id`` values,
    and caches by content sha (the ``voice_bank_id+sha`` re-baseline trigger).
    Identity is ``voice_ref_id`` -- never the character name (I-9).

  * **The caster** -- ``assign_voice_for_slot`` scores the bank for one slot by
    gender(100) / timbre(40) / role(20) / age(10), stable-sorts, and makes ONE
    seeded ``random.Random`` choice keyed on a ``stable_cast_seed`` derived from
    the slot. It walks a match ladder (g+t+r+age -> drop age -> drop role ->
    gender-only) and raises unless ``allow_voice_reuse``. The announcer voice is
    pinned per engine via ``announcer_voice_ref`` (raises if the active
    announcer engine has no reference).

Import-time is side-effect-free (the bank + schema are read lazily on first
use). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger("OTR")

CASTING_POLICY_VERSION = "1"
VOICE_BANK_SCHEMA_VERSION = "1"

_BANK_FILENAME = "voice_reference_bank.json"
_SCHEMA_FILENAME = "voice_bank_entry_schema.json"

# Caster scoring weights (plan E.2).
_W_GENDER = 100
_W_TIMBRE = 40
_W_ROLE = 20
_W_AGE = 10

# Match-ladder: each tier is the set of required match dimensions. The first
# tier with an available (non-reused) candidate wins.
_LADDER = (
    frozenset({"gender", "timbre", "role", "age"}),
    frozenset({"gender", "timbre", "role"}),
    frozenset({"gender", "timbre"}),
    frozenset({"gender"}),
)


class VoiceBankError(ValueError):
    """Raised when the voice reference bank is malformed or fails its schema."""


class VoiceCastingError(RuntimeError):
    """Raised when no castable voice can be assigned to a slot (fail-closed)."""


@dataclass(frozen=True)
class VoiceBankEntry:
    """One castable voice reference (frozen; mirrors the E.1 schema)."""

    voice_ref_id: str
    engine: str
    gender: str
    timbre: Tuple[str, ...]
    roles: Tuple[str, ...]
    age_band: str
    ref_path: str
    ref_sha256: str
    commercial_clean: bool


# --------------------------------------------------------------------------- #
# Dependency-free schema validation (same JSON-Schema subset the Wave-0 tests
# use: required keys + property types + minLength + minItems + array item type).
# --------------------------------------------------------------------------- #
def _type_ok(value, decl) -> bool:
    types = decl if isinstance(decl, list) else [decl]
    for t in types:
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "null" and value is None:
            return True
    return False


def _validate_entry(entry: dict, schema: dict) -> None:
    if not isinstance(entry, dict):
        raise VoiceBankError(f"voice entry must be an object, got {type(entry).__name__}")
    for req in schema.get("required", []):
        if req not in entry:
            raise VoiceBankError(f"voice entry missing required key {req!r}")
    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in entry:
            continue
        val = entry[key]
        decl = spec.get("type")
        if decl is not None and not _type_ok(val, decl):
            raise VoiceBankError(f"voice entry {key}={val!r} not of type {decl}")
        if spec.get("minLength") and isinstance(val, str) and len(val) < spec["minLength"]:
            raise VoiceBankError(f"voice entry {key} is shorter than minLength")
        if spec.get("minItems") and isinstance(val, list) and len(val) < spec["minItems"]:
            raise VoiceBankError(f"voice entry {key} needs >= {spec['minItems']} items")
        if decl == "array" and isinstance(val, list):
            item_spec = spec.get("items", {})
            if item_spec.get("type"):
                for it in val:
                    if not _type_ok(it, item_spec["type"]):
                        raise VoiceBankError(f"voice entry {key} item {it!r} bad type")


# --------------------------------------------------------------------------- #
# Lazy, content-sha-cached bank loading
# --------------------------------------------------------------------------- #
def _config_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "config", filename)


_BANK_CACHE: dict = {}


def _entry_from_dict(d: dict) -> VoiceBankEntry:
    return VoiceBankEntry(
        voice_ref_id=str(d["voice_ref_id"]),
        engine=str(d["engine"]),
        gender=str(d["gender"]),
        timbre=tuple(str(t) for t in (d.get("timbre") or [])),
        roles=tuple(str(r) for r in (d.get("roles") or [])),
        age_band=str(d["age_band"]),
        ref_path=str(d["ref_path"]),
        ref_sha256=str(d["ref_sha256"]),
        commercial_clean=bool(d["commercial_clean"]),
    )


def load_voice_bank(path: Optional[str] = None) -> Tuple[Tuple[VoiceBankEntry, ...], str]:
    """Load + validate the voice reference bank. Returns (entries, source_sha256).

    Validates every entry against ``voice_bank_entry_schema.json``, rejects
    duplicate ``voice_ref_id`` values, and caches by content sha. Raises
    :class:`VoiceBankError` on any malformed entry or duplicate id.
    """
    bank_path = path or _config_path(_BANK_FILENAME)
    with open(bank_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _BANK_CACHE.get(sha)
    if cached is not None:
        return cached
    with open(_config_path(_SCHEMA_FILENAME), "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    data = json.loads(text)
    rows = data.get("voices") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise VoiceBankError("voice_reference_bank: 'voices' must be a non-empty list")
    entries: List[VoiceBankEntry] = []
    seen = set()
    for row in rows:
        _validate_entry(row, schema)
        entry = _entry_from_dict(row)
        if entry.voice_ref_id in seen:
            raise VoiceBankError(f"duplicate voice_ref_id {entry.voice_ref_id!r}")
        seen.add(entry.voice_ref_id)
        entries.append(entry)
    result = (tuple(entries), sha)
    _BANK_CACHE[sha] = result
    return result


def get_all_registered_voices(bank: Optional[Tuple[VoiceBankEntry, ...]] = None) -> List[VoiceBankEntry]:
    """All bank entries, stable-sorted by ``voice_ref_id`` (E.1)."""
    entries = bank if bank is not None else load_voice_bank()[0]
    return sorted(entries, key=lambda e: e.voice_ref_id)


# --------------------------------------------------------------------------- #
# Deterministic caster (E.2) -- disjoint RNG from the legacy caster (I-4)
# --------------------------------------------------------------------------- #
def stable_cast_seed(*, episode_seed, casting_policy_version, char_id, gender,
                     timbre, role, age_band) -> int:
    """Reduce a slot's identity to a stable int seed (null -> "")."""
    from ._otr_resolved_request import _seed_to_int64

    canon = {
        "episode_seed": episode_seed if episode_seed is not None else "",
        "casting_policy_version": casting_policy_version or "",
        "char_id": char_id or "",
        "gender": gender or "",
        "timbre": [str(t) for t in (timbre or [])],
        "role": role or "",
        "age_band": age_band or "",
    }
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _seed_to_int64("stable_cast_seed_v1", payload)


def _score(entry: VoiceBankEntry, *, gender, timbre, role, age_band) -> int:
    s = 0
    if gender and entry.gender == gender:
        s += _W_GENDER
    if timbre and set(timbre) & set(entry.timbre):
        s += _W_TIMBRE
    if role and role in entry.roles:
        s += _W_ROLE
    if age_band and entry.age_band == age_band:
        s += _W_AGE
    return s


def _matches(entry: VoiceBankEntry, dims, *, gender, timbre, role, age_band) -> bool:
    if "gender" in dims and not (gender and entry.gender == gender):
        return False
    if "timbre" in dims and not (timbre and set(timbre) & set(entry.timbre)):
        return False
    if "role" in dims and not (role and role in entry.roles):
        return False
    if "age" in dims and not (age_band and entry.age_band == age_band):
        return False
    return True


def assign_voice_for_slot(
    *,
    role: str,
    engine: str,
    char_id: str,
    gender: str,
    timbre=(),
    age_band: str = "",
    episode_seed=0,
    casting_policy_version: str = CASTING_POLICY_VERSION,
    allow_voice_reuse: bool = False,
    used_voice_ref_ids=None,
    require_commercial_clean: bool = False,
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
) -> VoiceBankEntry:
    """Assign exactly one voice reference to a slot (deterministic, fail-closed).

    Candidates are the bank entries for ``engine`` (optionally restricted to
    commercially-clean ones). They are scored gender/timbre/role/age, then a
    match ladder (g+t+r+age -> drop age -> drop role -> gender-only) selects the
    first tier with an available (non-reused) candidate; within that tier the
    pool is stable-sorted by score then ``voice_ref_id`` and ONE seeded choice
    is made. Raises :class:`VoiceCastingError` when nothing is castable unless
    ``allow_voice_reuse`` lets a previously-used reference be reused.
    """
    entries = bank if bank is not None else load_voice_bank()[0]
    used = set(used_voice_ref_ids or ())
    candidates = [e for e in entries if e.engine == engine]
    if require_commercial_clean:
        candidates = [e for e in candidates if e.commercial_clean is True]
    if not candidates:
        raise VoiceCastingError(
            f"no voice references for engine {engine!r} "
            f"(role {role!r}, char_id {char_id!r})"
        )

    seed = stable_cast_seed(
        episode_seed=episode_seed,
        casting_policy_version=casting_policy_version,
        char_id=char_id,
        gender=gender,
        timbre=timbre,
        role=role,
        age_band=age_band,
    )

    def _pick(pool: List[VoiceBankEntry]) -> VoiceBankEntry:
        pool = sorted(
            pool,
            key=lambda e: (
                -_score(e, gender=gender, timbre=timbre, role=role, age_band=age_band),
                e.voice_ref_id,
            ),
        )
        return random.Random(seed).choice(pool)

    def _ladder_pick(exclude_used: bool) -> Optional[VoiceBankEntry]:
        for dims in _LADDER:
            pool = [
                e for e in candidates
                if _matches(e, dims, gender=gender, timbre=timbre,
                            role=role, age_band=age_band)
                and (not exclude_used or e.voice_ref_id not in used)
            ]
            if pool:
                return _pick(pool)
        return None

    # Normal path: each slot gets a unique reference. If the ladder is exhausted
    # only because every gender-matching reference is already used, allow_reuse
    # re-walks the SAME ladder permitting a reused reference -- but the gender
    # floor (the last tier) still holds, so a male slot never gets a female
    # voice. No gender match at all -> fail closed.
    chosen = _ladder_pick(exclude_used=True)
    if chosen is None and allow_voice_reuse:
        chosen = _ladder_pick(exclude_used=False)
    if chosen is None:
        detail = (
            "all matching references are already used"
            if not allow_voice_reuse else "no gender-matching reference exists"
        )
        raise VoiceCastingError(
            f"no castable voice reference for char_id {char_id!r} "
            f"(engine {engine!r}, role {role!r}, gender {gender!r}); {detail}"
        )
    return chosen


def announcer_voice_ref(
    engine: str, bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
) -> VoiceBankEntry:
    """The pinned announcer reference for ``engine`` (E.1). Raise if none.

    Deterministic pin: the lowest ``voice_ref_id`` among the engine's
    announcer-role references.
    """
    entries = bank if bank is not None else load_voice_bank()[0]
    cands = sorted(
        [e for e in entries if e.engine == engine and "announcer_voice" in e.roles],
        key=lambda e: e.voice_ref_id,
    )
    if not cands:
        raise VoiceCastingError(
            f"no announcer voice reference for engine {engine!r}"
        )
    return cands[0]
