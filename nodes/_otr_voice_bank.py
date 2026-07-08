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

# v2 (2026-06-11, whiny-fix, operator-directed): scores now WEIGHT the in-tier
# lottery instead of only sorting it (G4 -- the pick was uniform across the
# tier, so a perfectly-scored ref and a barely-matching one drew equally).
# Version bump = deliberate C7 re-baseline of all future casting draws.
# OTR_CAST_WEIGHTED=0 restores the uniform v1 draw for A/B comparison.
CASTING_POLICY_VERSION = "2"
VOICE_BANK_SCHEMA_VERSION = "1"
# VC chunk 4 (2026-06-22): the HYBRID LLM voice-fit. The LLM PROPOSES a
# voice_ref_id from the engine's gender-slot cards; Python VALIDATES it and
# falls closed to the deterministic assign_voice_for_slot scorer. Version-bump
# this when the card schema or validation contract changes (re-baseline trigger).
VOICE_FIT_POLICY_VERSION = "1"

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
    # Whiny-fix P2 ADDITIVE curation fields (default-empty = bank not yet
    # audited; behavior unchanged until the operator's audition reel stamps
    # them). quality_tier: "a" | "b" | "reject" | ""; style_tags e.g.
    # ("lead_safe", "nasal_risk").
    quality_tier: str = ""
    style_tags: Tuple[str, ...] = ()
    # Cloud-audio campaign 2026-07-03 (C3): the provider's own voice identifier
    # (e.g. ElevenLabs voice_id). Threaded end-to-end so cloud casting resolves
    # from adapter metadata, NOT a disk sentinel. Empty for every local
    # (ref-clip / preset) engine -- behavior unchanged until a cloud bank ships.
    provider_voice_id: str = ""


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
        quality_tier=str(d.get("quality_tier") or ""),
        style_tags=tuple(str(t) for t in (d.get("style_tags") or [])),
        provider_voice_id=str(d.get("provider_voice_id") or ""),   # C3
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


# Bank-backed voice engines (bark uses v2/en_speaker_* presets, not bank refs);
# the library-coverage gate over config/voice_reference_bank.json applies to these.
APPROVED_VOICE_ENGINES: Tuple[str, ...] = ("indextts2", "chatterbox", "dia", "kokoro")


def compute_bank_coverage(
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    *,
    approved_engines: Tuple[str, ...] = APPROVED_VOICE_ENGINES,
    cast_size: int = 5,
) -> dict:
    """Voice-casting goal (B): each approved engine has a SOLID library. PURE.

    Returns ``{engine: {total, by_gender, by_gender_age, adult_male, adult_female,
    meets_floor, gaps}}``.

    ``meets_floor`` is the HARD bar a test asserts: an engine must have at least
    ``cast_size`` ADULT voices for EACH of male/female, so a worst-case same-gender
    ``cast_size``-character cast can be cast with NO reuse. ``gaps`` lists
    aspirational thin spots (male-light count, no non-adult coverage, no female
    elder, no `other`/androgynous voices) for operator remediation -- surfaced, not
    failed (the library is solid for the common case but should grow).
    """
    if bank is None:
        bank, _ = load_voice_bank()
    out: dict = {}
    for eng in approved_engines:
        rows = [e for e in bank if e.engine == eng]
        by_g: dict = {}
        by_ga: dict = {}
        for e in rows:
            by_g[e.gender] = by_g.get(e.gender, 0) + 1
            by_ga[(e.gender, e.age_band)] = by_ga.get((e.gender, e.age_band), 0) + 1
        adult_m = by_ga.get(("male", "adult"), 0)
        adult_f = by_ga.get(("female", "adult"), 0)
        gaps: List[str] = []
        if adult_m < adult_f:
            gaps.append(
                f"male-light: {adult_m} adult male vs {adult_f} adult female")
        if not any(ab != "adult" for (_g, ab) in by_ga):
            gaps.append("no non-adult age coverage (no child/teen/elder mix)")
        if (by_ga.get(("female", "elder"), 0) == 0
                and by_ga.get(("male", "elder"), 0) > 0):
            gaps.append("elder coverage is male-only (no female elder)")
        if by_g.get("other", 0) == 0:
            gaps.append("no 'other'/androgynous voices")
        out[eng] = {
            "total": len(rows),
            "by_gender": dict(by_g),
            "by_gender_age": {f"{g}/{a}": n for (g, a), n in sorted(by_ga.items())},
            "adult_male": adult_m,
            "adult_female": adult_f,
            "meets_floor": adult_m >= cast_size and adult_f >= cast_size,
            "gaps": gaps,
        }
    return out


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
    # Whiny-fix P2c stage 1: audited rejects NEVER cast (deterministic pool
    # pre-filter -- never a score reweight). Un-audited entries (empty tier)
    # pass untouched, so the un-audited bank behaves exactly as before.
    candidates = [e for e in entries
                  if e.engine == engine and e.quality_tier != "reject"]
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
        # Casting v2 (whiny-fix, operator-directed): scores WEIGHT the lottery
        # (score+1 so a zero-score candidate keeps a small chance instead of
        # vanishing); deterministic in the same stable_cast_seed. The G4 v1
        # uniform draw stays reachable via OTR_CAST_WEIGHTED=0 for A/B.
        if os.getenv("OTR_CAST_WEIGHTED", "1") == "0":
            return random.Random(seed).choice(pool)
        weights = [
            _score(e, gender=gender, timbre=timbre, role=role, age_band=age_band) + 1
            for e in pool
        ]
        return random.Random(seed).choices(pool, weights=weights, k=1)[0]

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


def filter_by_quality_tier(entries, *, lead: bool = False):
    """Whiny-fix P2c: the deterministic two-stage POOL PRE-FILTER (shared by
    CastLock's caster AND the render-time ``_resolve_clone_ref_path`` so
    tier-reject refs cannot leak through the fallback route -- G16).

    * ``reject`` entries are dropped ALWAYS.
    * ``lead=True``: prefer tier-a ``lead_safe`` if any survive; else tier-a;
      else the full non-reject pool (the configured degrade).
    * Un-audited banks (no tiers stamped) pass through unchanged.

    NEVER reweights scores -- pre-filter only.
    """
    pool = [e for e in entries if getattr(e, "quality_tier", "") != "reject"]
    if not lead:
        return pool
    tier_a = [e for e in pool if getattr(e, "quality_tier", "") == "a"]
    if tier_a:
        lead_safe = [e for e in tier_a if "lead_safe" in getattr(e, "style_tags", ())]
        return lead_safe or tier_a
    return pool


# --------------------------------------------------------------------------- #
# VC chunk 4 (2026-06-22) -- HYBRID LLM voice-fit support: build the engine's
# gender-slot voice CARDS for the LLM, validate its proposal, look an entry up.
# All pure + fail-soft. Identity is voice_ref_id (I-9); cards carry NO ref_path
# and NO character name.
# --------------------------------------------------------------------------- #
def build_voice_cards(
    engine: str,
    gender: str,
    *,
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    max_cards: int = 12,
) -> List[dict]:
    """The voice CARDS the LLM picks from for one character's PRECOMPUTED gender
    slot. Deterministically ordered (by ``voice_ref_id``), reject-filtered,
    capped at ``max_cards``. Each card: ``voice_ref_id`` / ``age_band`` /
    ``timbre`` / ``roles`` / ``style_tags`` / ``commercial_clean`` + a compact
    human-readable ``descriptor`` (the bank has no curated prose field, so it is
    synthesized from age/timbre/style). NO ref_path, NO character name (I-9)."""
    gnorm = (gender or "").strip().lower()
    if not engine or not gnorm:
        return []
    try:
        entries = bank if bank is not None else load_voice_bank()[0]
    except Exception:  # noqa: BLE001 -- no bank -> no cards -> caller falls closed
        return []
    pool = sorted(
        (e for e in entries
         if e.engine == engine and e.gender == gnorm
         and getattr(e, "quality_tier", "") != "reject"),
        key=lambda e: e.voice_ref_id,
    )[: max(0, int(max_cards))]
    cards: List[dict] = []
    for e in pool:
        descriptor = ", ".join(
            x for x in ([e.age_band] + list(e.timbre) + list(e.style_tags)) if x
        )
        cards.append({
            "voice_ref_id": e.voice_ref_id,
            "age_band": e.age_band,
            "timbre": list(e.timbre),
            "roles": list(e.roles),
            "style_tags": list(e.style_tags),
            "commercial_clean": bool(e.commercial_clean),
            "descriptor": descriptor,
        })
    return cards


def default_char_engine(
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
) -> str:
    """The default bank-backed char_voice engine (the ``APPROVED_VOICE_ENGINES``
    legacy-first order, first one with char_voice refs). Matches what CastLock
    resolves for the canonical ``default`` voice_bank; '' when none. The writer
    builds voice cards against this; a CastLock voice_bank that resolves a
    DIFFERENT engine just makes the proposal fail validation -> fall closed."""
    try:
        entries = bank if bank is not None else load_voice_bank()[0]
    except Exception:  # noqa: BLE001
        return ""
    with_refs = {e.engine for e in entries if "char_voice" in e.roles}
    for eng in APPROVED_VOICE_ENGINES:
        if eng in with_refs:
            return eng
    return ""


def voice_ref_entry(
    voice_ref_id: str, engine: str,
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
) -> Optional[VoiceBankEntry]:
    """The bank entry for (voice_ref_id, engine), or None. Pure + fail-soft."""
    if not voice_ref_id or not engine:
        return None
    try:
        entries = bank if bank is not None else load_voice_bank()[0]
    except Exception:  # noqa: BLE001
        return None
    return next(
        (e for e in entries
         if e.voice_ref_id == voice_ref_id and e.engine == engine), None
    )


def validate_voice_proposal(
    proposed_id: str,
    engine: str,
    gender: str,
    *,
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    used_ids=(),
) -> str:
    """Validate an LLM-proposed ``voice_ref_id`` for one slot. Returns the id iff
    it is in-library + engine-correct + gender-consistent + not a reject + not
    already used (no-collision); else '' (the caller falls closed to the
    deterministic scorer). Pure; never raises."""
    pid = str(proposed_id or "").strip()
    gnorm = (gender or "").strip().lower()
    if not pid or not engine or not gnorm:
        return ""
    used = set(used_ids or ())
    if pid in used:
        return ""
    entry = voice_ref_entry(pid, engine, bank)
    if entry is None:
        return ""
    if entry.gender != gnorm or getattr(entry, "quality_tier", "") == "reject":
        return ""
    return pid


# --------------------------------------------------------------------------- #
# VC chunk 2 (2026-06-22) -- two-lane identity: deterministic bark v2/* preset
# <-> same-gender clone voice_ref_id map.
#
# voice_preset (bark v2/en_speaker_*) is the UNIVERSAL fallback identity; a
# cloner engine (indextts2 / chatterbox / dia / kokoro) wants a real bank
# voice_ref_id. This map lets a bark-cast identity resolve to a SAME-GENDER
# clone reference at the contract level so the fallback never silently degrades
# a cloner render to bark. Pure + fail-soft (no bank / unknown preset -> "").
# --------------------------------------------------------------------------- #
def bark_preset_gender(preset: str) -> str:
    """Gender of a bark ``v2/en_speaker_*`` preset, read from
    ``config/cast_pools.VOICE_PROFILES`` (the single source of truth -- never a
    hand-kept copy). Returns 'male' / 'female', or '' when the preset is unknown
    or cast_pools is unavailable. Pure; never raises."""
    p = str(preset or "").strip()
    if not p:
        return ""
    try:
        try:
            from ..config import cast_pools as _POOLS  # type: ignore
        except (ImportError, ValueError):
            import sys
            here = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(here)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from config import cast_pools as _POOLS  # type: ignore
        for entry in getattr(_POOLS, "VOICE_PROFILES", ()):
            # entry: (preset, gender, lang_code, quality_tags)
            if entry and str(entry[0]).strip() == p:
                return str(entry[1]).strip().lower()
    except Exception:  # noqa: BLE001 -- cast_pools optional; fail-soft
        return ""
    return ""


def same_gender_voice_ref_for_preset(
    preset: str,
    engine: str,
    *,
    bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    gender_hint: str = "",
) -> str:
    """Deterministic ``v2/en_speaker_* -> same-gender voice_ref_id`` for
    ``engine``. Returns the LOWEST ``voice_ref_id`` among the engine's
    same-gender, non-reject references (stable -> C7), or '' when no such ref
    exists. ``gender_hint`` overrides the preset-derived gender (use the cast
    row's gender when known). Pure + fail-soft; never raises."""
    gender = (gender_hint or bark_preset_gender(preset) or "").strip().lower()
    if not gender or not engine:
        return ""
    try:
        entries = bank if bank is not None else load_voice_bank()[0]
    except Exception:  # noqa: BLE001 -- no/broken bank -> bark stays the identity
        return ""
    cands = sorted(
        (e for e in entries
         if e.engine == engine and e.gender == gender
         and getattr(e, "quality_tier", "") != "reject"),
        key=lambda e: e.voice_ref_id,
    )
    return cands[0].voice_ref_id if cands else ""


def _google_announcer_gender(episode_seed) -> str:
    digest = hashlib.sha1(
        ("google_tts_announcer:%s" % (episode_seed if episode_seed is not None else ""))
        .encode("utf-8")
    ).hexdigest()
    return "male" if int(digest, 16) % 2 == 0 else "female"


def _google_announcer_voice_ref(
    bank: Tuple[VoiceBankEntry, ...], episode_seed=0,
) -> VoiceBankEntry:
    """Deterministic Google announcer selection.

    When both male- and female-coded preferred announcers are available, select
    the gender from the episode seed for a stable roughly 50/50 mix. Within the
    gender, prefer entries tagged preferred_announcer/british_leaning.
    """
    cands = [
        e for e in bank
        if e.engine == "google_tts"
        and "announcer_voice" in e.roles
        and e.quality_tier != "reject"
    ]
    if not cands:
        raise VoiceCastingError("no announcer voice reference for engine 'google_tts'")
    preferred = [e for e in cands if "preferred_announcer" in e.style_tags]
    pool = preferred or cands
    by_gender = {}
    for entry in pool:
        by_gender.setdefault(entry.gender, []).append(entry)
    wanted = _google_announcer_gender(episode_seed)
    if "male" in by_gender and "female" in by_gender:
        pool = by_gender[wanted]
    elif wanted in by_gender:
        pool = by_gender[wanted]
    def _key(entry):
        tags = set(entry.style_tags)
        return (
            "preferred_announcer" not in tags,
            "british_leaning" not in tags,
            entry.voice_ref_id,
        )
    return sorted(pool, key=_key)[0]


def announcer_voice_ref(
    engine: str, bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    episode_seed=0,
) -> VoiceBankEntry:
    """The announcer reference for ``engine`` (E.1). Raise if none.

    Legacy engines keep their deterministic pin: the lowest ``voice_ref_id``
    among announcer-role references. Google TTS mixes male/female preferred
    announcer candidates by episode seed when both are available.
    """
    entries = bank if bank is not None else load_voice_bank()[0]
    if engine == "google_tts":
        return _google_announcer_voice_ref(entries, episode_seed=episode_seed)
    cands = sorted(
        [e for e in entries if e.engine == engine and "announcer_voice" in e.roles],
        key=lambda e: e.voice_ref_id,
    )
    if not cands:
        raise VoiceCastingError(
            f"no announcer voice reference for engine {engine!r}"
        )
    return cands[0]
