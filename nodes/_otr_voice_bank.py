"""Voice reference bank + deterministic caster (plan E.1 / E.2, Wave 1 / 1d).

Two surfaces live here, both BESIDE the frozen legacy caster (never replacing
it -- I-4 keeps the two casters on disjoint RNGs):

  * **The bank** -- ``config/voice_reference_bank.json`` is a set of castable
    voice references, each coding to ``config/voice_bank_entry_schema.json``
    (the Wave-0 E.1 schema). ``load_voice_bank`` validates every entry against
    that schema (dependency-free), rejects duplicate ``voice_ref_id`` values,
    and caches by content sha (the ``voice_bank_id+sha`` re-baseline trigger).
    Durable identity is ``voice_ref_id`` -- never the character name (I-9);
    no-reuse casting also treats same reference assets as collisions so alias ids
    cannot sneak the same voice into a cast.

  * **The caster** -- ``assign_voice_for_slot`` scores the bank for one slot by
    gender(100) / timbre(40) / role(20) / age(10), stable-sorts, and makes ONE
    seeded ``random.Random`` choice keyed on a ``stable_cast_seed`` derived from
    the slot. It walks a match ladder (g+t+r+age -> drop age -> drop role ->
    gender-only) and raises unless ``allow_voice_reuse``. The announcer voice is
    resolved per engine via ``announcer_voice_ref``; curated announcer pools can
    mix gender by episode seed, while legacy single-ref engines stay pinned.

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

import os as _os_boot
import sys as _sys_boot

# STANDALONE LOAD. `scripts/otr_provision.py`, `otr_make_portable_voice_bank.py`
# and friends load this file by PATH under a non-package name, deliberately --
# "without importing the ComfyUI node package". Neither ladder rung below can
# resolve then: the relative one has no parent package and the flat one needs
# `nodes/` on sys.path, which those loaders do not add. Same insert, and the
# same reason, as `otr_post_upscale_procgen_blend.py`.
_NODES_DIR = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
if _NODES_DIR not in _sys_boot.path:
    _sys_boot.path.insert(0, _NODES_DIR)

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat / standalone load
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

# v2 (2026-06-11, whiny-fix, operator-directed): scores now WEIGHT the in-tier
# lottery instead of only sorting it (G4 -- the pick was uniform across the
# tier, so a perfectly-scored ref and a barely-matching one drew equally).
# Version bump = deliberate C7 re-baseline of all future casting draws.
# OTR_CAST_WEIGHTED=0 restores the uniform v1 draw for A/B comparison.
# v3 (2026-08-05): _ladder_pick no longer accepts a candidate tier of ONE, which
# was a 100% pin on three (engine, gender, timbre) combos. Deliberate C7
# re-baseline of all future casting draws. The effective floor is folded into
# stable_cast_seed as "+mtp{n}" so two different settings can never both claim
# to be policy 3. OTR_CAST_MIN_TIER_POOL makes floor=3 a one-run live A/B.
CASTING_POLICY_VERSION = "3"

# 2 rather than 3: measured against the shipped bank, floor 2 removes every
# size-1 tier while 5 of 24 combos still honour the requested timbre; floor 3
# also removes them but leaves only 2 of 24 honouring timbre at all -- it buys
# spread by deleting the dimension rather than fixing it.
_MIN_TIER_POOL_DEFAULT = 2


def _min_tier_pool() -> int:
    """Smallest candidate tier the ladder will accept (env-overridable A/B)."""
    try:
        return max(1, int(otr_env.get("OTR_CAST_MIN_TIER_POOL",
                                    _MIN_TIER_POOL_DEFAULT)))
    except (TypeError, ValueError):
        return _MIN_TIER_POOL_DEFAULT
VOICE_BANK_SCHEMA_VERSION = "1"
# VOICE_FIT_POLICY_VERSION IS RETIRED (2026-08-18). It versioned the hybrid LLM
# voice-fit's card schema and validation contract, and both were ripped along
# with the pass -- a constant that versions nothing is worse than no constant,
# because the next reader assumes something still honours it.

# Engines whose announcer comes from a CURATED pool and is drawn per episode
# rather than pinned to the lowest voice_ref_id. kokoro joined 2026-08-05: it
# had exactly one announcer-role row (bm_george), so every published episode
# opened with the same voice. cast_lock.py reads this same tuple -- the two
# must never drift, which is why it lives here and not as a second literal.
_SEEDED_ANNOUNCER_ENGINES = ("google_tts", "chatterbox", "dia", "kokoro")

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


def reserved_voice_ref_ids() -> frozenset:
    """Voice references that belong to ONE named character and nobody else.

    DERIVED FROM THE POLICY, never a hand-kept list. Every `voice_ref_id` named
    by a Lemmy route -- qualified or provisional, on any engine -- is reserved by
    the act of being named there. Add a route and the reservation follows for
    free; that is the whole reason this reads the policy instead of restating it,
    because a second list is a list that drifts.

    Fail-SOFT: an unimportable pack reserves nothing and casting behaves exactly
    as it did before this existed. The strictness lives where a route is claimed,
    not in a helper that would otherwise break every render on an import error.

    FAIL-SOFT ON ANY EXCEPTION, not just ImportError (widened 2026-08-18 after a
    QA pass). Its callers are ``assign_voice_for_slot`` and
    ``gender_agnostic_fallback_ref`` -- the latter promises in its own docstring
    to be pure and never raise. (It had four callers until the hybrid LLM
    voice-fit was ripped the same day; ``build_voice_cards`` and
    ``validate_voice_proposal`` went with it.) A malformed policy (say
    ``LEMMY_VOICE_POLICY`` set to a truthy non-dict, so ``.get`` raises
    ``AttributeError``) would otherwise propagate straight through those
    promises and out of ``cast_lock.py``, turning a cosmetic config error into a
    dead render. Reserving nothing is the correct degradation: it restores
    exactly the pre-reservation behaviour rather than failing an episode.
    """
    try:
        from ..config import cast_pools as _POOLS  # type: ignore
    except ImportError:
        try:
            from config import cast_pools as _POOLS  # type: ignore
        except ImportError:
            return frozenset()
    try:
        policy = getattr(_POOLS, "LEMMY_VOICE_POLICY", None) or {}
        return _reserved_ids_from_policy(policy)
    except Exception:  # noqa: BLE001 -- a broken policy reserves nothing
        log.warning("[OTR voice] reserved-id policy unreadable; reserving none",
                    exc_info=True)
        return frozenset()


def _reserved_ids_from_policy(policy) -> frozenset:
    """The reserved-id walk itself, split out so the fail-soft wrapper above
    covers every step of it rather than only the import."""
    ids = set()

    # ONLY A CLONE OF HIS OWN RECORDING IS HIS. The first cut of this reserved
    # every voice any Lemmy route named, and that was too broad in a way that
    # would have caused its own regression: `bm_george` is a shared kokoro
    # catalogue voice tagged `preferred_announcer`, and `el_daniel` / `gt_algenib`
    # are shared cloud voices. Lemmy is PROVISIONALLY POINTED AT those; he does
    # not own them, and pulling the preferred announcer out of the pool to
    # protect a cameo would trade one defect for another.
    #
    # A `local_wav` route is different in kind: those rows are clones of the
    # operator-approved recording of Lemmy himself, so hearing that timbre IS
    # hearing Lemmy. Those are reserved; the borrowed catalogue voices are not.
    for key in ("approved_native_routes", "provisional_native_routes"):
        routes = policy.get(key)
        if not isinstance(routes, dict):
            continue
        for record in routes.values():
            if not isinstance(record, dict):
                continue
            receipt = record.get("provisional_receipt")
            if isinstance(receipt, dict):
                if receipt.get("identity_kind") != "local_wav":
                    continue
                ref = str(record.get("voice_ref_id") or "").strip()
                if ref:
                    ids.add(ref)
                continue
            qual = record.get("qualification_record")
            if isinstance(qual, dict):
                ref_block = qual.get("reference")
                if not (isinstance(ref_block, dict)
                        and ref_block.get("kind") == "local_wav"):
                    continue
                ref = str(qual.get("voice_ref_id") or "").strip()
                if ref:
                    ids.add(ref)
    return frozenset(ids)


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
    # The real HUMAN behind this reference, when two rows are two recordings of
    # one person. ref_path collision cannot catch that case: LibriVox's Mark F.
    # Smith has a plain and a grandfatherly take in two different files, so
    # without this one narrator can be cast as two characters in one episode.
    speaker_id: str = ""


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
_BANK_UNAVAILABLE_ROUTE_IDS_CACHE: dict = {}


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
        speaker_id=str(d.get("speaker_id") or ""),
    )


def load_voice_bank(path: Optional[str] = None) -> Tuple[Tuple[VoiceBankEntry, ...], str]:
    """Load + validate the voice reference bank. Returns (entries, source_sha256).

    Validates every entry against ``voice_bank_entry_schema.json``, rejects
    duplicate ``voice_ref_id`` values, and caches by content sha. Raises
    :class:`VoiceBankError` on any malformed entry or duplicate id.

    ALSO on an unreadable or unparseable file. That used to leak raw
    ``FileNotFoundError`` / ``JSONDecodeError`` past this docstring's promise,
    so a caller that correctly caught the documented ``VoiceBankError`` still
    died on the two most likely real-world failures -- a missing bank and a
    corrupt one. Owning the contract here means every consumer gets it, rather
    than each one widening its own except clause and guessing at the list.
    """
    # Portable machines may carry their own authorized reference recordings.
    # Provisioning and runtime share this one authority; without it a setup
    # could verify a two-voice portable bank and then cast from the unrelated
    # historical bank at render time.
    bank_path = path or otr_env.get("OTR_VOICE_REFERENCE_BANK") or _config_path(
        _BANK_FILENAME)
    try:
        with open(bank_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise VoiceBankError(
            f"voice_reference_bank unreadable at {bank_path}: {exc}") from exc
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _BANK_CACHE.get(sha)
    if cached is not None:
        return cached
    try:
        with open(_config_path(_SCHEMA_FILENAME), "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        data = json.loads(text)
    except (OSError, ValueError) as exc:   # ValueError covers JSONDecodeError
        raise VoiceBankError(
            f"voice_reference_bank or its schema is unparseable: {exc}") from exc
    unavailable_route_ids = frozenset()
    if isinstance(data, dict):
        raw_unavailable = data.get("unavailable_qualified_route_ids", [])
        if (not isinstance(raw_unavailable, list)
                or any(not isinstance(value, str) or not value.strip()
                       or value != value.strip()
                       for value in raw_unavailable)
                or len(raw_unavailable) != len(set(raw_unavailable))):
            raise VoiceBankError(
                "voice_reference_bank: 'unavailable_qualified_route_ids' must "
                "be a duplicate-free list of non-empty route-id strings")
        unavailable_route_ids = frozenset(raw_unavailable)
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
    _BANK_UNAVAILABLE_ROUTE_IDS_CACHE[sha] = unavailable_route_ids
    return result


def unavailable_qualified_route_ids(
        path: Optional[str] = None, *, source_sha256: Optional[str] = None
) -> frozenset:
    """Qualified routes this bank explicitly and intentionally cannot carry.

    This is a narrow exception list, not a portable-mode switch. CastLock only
    honours an id after that exact route is selected for the active engine and
    its exact reference row is absent. Every other qualification check remains
    fail-closed.
    """
    if source_sha256 is not None:
        try:
            return _BANK_UNAVAILABLE_ROUTE_IDS_CACHE[source_sha256]
        except KeyError as exc:
            raise VoiceBankError(
                "voice bank metadata requested for an unloaded source SHA-256 "
                "%r" % source_sha256) from exc
    _entries, sha = load_voice_bank(path)
    return _BANK_UNAVAILABLE_ROUTE_IDS_CACHE[sha]


def get_all_registered_voices(bank: Optional[Tuple[VoiceBankEntry, ...]] = None) -> List[VoiceBankEntry]:
    """All bank entries, stable-sorted by ``voice_ref_id`` (E.1)."""
    entries = bank if bank is not None else load_voice_bank()[0]
    return sorted(entries, key=lambda e: e.voice_ref_id)


# Bank-backed voice engines (bark uses v2/en_speaker_* presets, not bank refs);
# the library-coverage gate over config/voice_reference_bank.json applies to these.
#
# ``google_tts`` appended 2026-08-08. It was ABSENT, so the coverage floor --
# the gate whose entire job is to stop an engine going thin -- never looked at
# the cloud lane, and google_tts sat at FOUR castable char voices until a
# nine-speaker roster failed casting outright (CastLock re-raises for
# google_tts, no fallback). Order matters and this must stay LAST:
# ``default_char_engine`` returns the first approved engine that has char_voice
# refs, so appending leaves that resolution byte-identical.
APPROVED_VOICE_ENGINES: Tuple[str, ...] = (
    "indextts2", "chatterbox", "dia", "kokoro", "google_tts")


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


def _norm_ref_identity(ref_path: str) -> str:
    p = str(ref_path or "").strip().replace("\\", "/").lower()
    while "//" in p:
        p = p.replace("//", "/")
    return p


def voice_ref_usage_keys(entry: VoiceBankEntry) -> set:
    """Collision keys for no-reuse casting.

    ``voice_ref_id`` remains the durable ledger identity; these keys only guard
    runtime assignment so alias ids that share a WAV/provider voice cannot both
    be selected when reuse is disabled.
    """
    keys = {str(entry.voice_ref_id or "")}
    ref_path = _norm_ref_identity(getattr(entry, "ref_path", "") or "")
    if ref_path:
        keys.add(f"ref_path:{ref_path}")
    provider_voice_id = str(getattr(entry, "provider_voice_id", "") or "").strip().lower()
    if provider_voice_id:
        keys.add(f"provider:{entry.engine}:{provider_voice_id}")
    speaker_id = str(getattr(entry, "speaker_id", "") or "").strip().lower()
    if speaker_id:
        # NOT engine-scoped: two takes of one human collide within an engine,
        # and casting is single-engine anyway, so the bare id is the identity.
        keys.add(f"speaker:{speaker_id}")
    return {k for k in keys if k}


def _entry_is_used(entry: VoiceBankEntry, used: set) -> bool:
    return bool(voice_ref_usage_keys(entry) & used)


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
    # A RESERVED IDENTITY IS NOT IN THE POOL (operator 2026-08-17: "Lemmy should
    # only be Lemmy, not some random character -- Lemmy stays in the box unless
    # Lemmy is called upon").
    #
    # Measured on a real published episode: MOE GORDON, a character who is not
    # Lemmy, drew `idx_lemmy_algenib_cockney_v1` -- the reference a blinded
    # operator audition qualified as LEMMY'S voice. Nothing was broken; the row
    # simply sits in the char_voice pool like any other, so the seeded draw may
    # hand his signature Cockney to anybody. That quietly undoes the identity the
    # audition established: a voice can only mean "this is Lemmy" if it is the
    # only place it is ever heard.
    #
    # THE POLICY PATH IS UNAFFECTED, which is what makes this safe. A qualified
    # or provisional route stamps its bank entry DIRECTLY in CastLock and never
    # comes through this selector, so reserving these ids cannot starve Lemmy of
    # his own voice -- it only stops everyone else borrowing it.
    #
    # Same shape as the audited-reject filter above: a deterministic pool
    # pre-filter, never a score reweight.
    reserved = reserved_voice_ref_ids()
    if reserved:
        candidates = [e for e in candidates if e.voice_ref_id not in reserved]
    if require_commercial_clean:
        candidates = [e for e in candidates if e.commercial_clean is True]
    if not candidates:
        raise VoiceCastingError(
            f"no voice references for engine {engine!r} "
            f"(role {role!r}, char_id {char_id!r})"
        )

    # Bound HERE, before the seed, not inside _ladder_pick: the floor is folded
    # into the cast seed below, and a name local to that closure would not exist
    # yet at this line.
    floor = _min_tier_pool()

    seed = stable_cast_seed(
        episode_seed=episode_seed,
        # The floor changes WHICH voice a given (episode_seed, char_id) draws, so
        # it belongs in the seed's identity. Without it two different byte
        # outputs would both claim policy '3'.
        casting_policy_version=f"{casting_policy_version}+mtp{floor}",
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
        if otr_env.get("OTR_CAST_WEIGHTED", "1") == "0":
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
                and (not exclude_used or not _entry_is_used(e, used))
            ]
            # A tier of ONE is not a choice, it is a pin: every episode asking
            # for that (engine, gender, timbre) got the same voice, 100% of the
            # time. Measured against the shipped bank, three combos did exactly
            # that -- (indextts2, female, warm), (kokoro, male, warm) and
            # (kokoro, male, deep). Skip a tier thinner than the floor and let
            # the ladder drop a dimension instead.
            #
            # The gender-only last tier is ALWAYS accepted regardless of size,
            # so the gender floor and the fail-closed behaviour below are
            # untouched and no new VoiceCastingError becomes reachable.
            if pool and (len(pool) >= floor or dims == _LADDER[-1]):
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


def _seeded_announcer_gender(engine: str, episode_seed) -> str:
    digest = hashlib.sha1(
        ("%s_announcer:%s" % (
            str(engine or ""),
            episode_seed if episode_seed is not None else "",
        ))
        .encode("utf-8")
    ).hexdigest()
    return "male" if int(digest, 16) % 2 == 0 else "female"


def gender_agnostic_fallback_ref(
    bank: Tuple[VoiceBankEntry, ...], *, engine: str, char_id: str,
    episode_seed=0, role: str = "char_voice", used=None,
) -> Optional[VoiceBankEntry]:
    """The reference an UNCASTABLE row actually renders with, or None.

    A row whose gender the bank cannot serve -- `other` is 20% of every roll and
    the bank has zero rows carrying it -- makes `assign_voice_for_slot` raise. The
    render path must still hand a cloning engine a real voice rather than
    silently dropping to bark, so it falls back to a uniform draw over the
    engine's refs, deterministically keyed on char_id so C7 holds.

    This lives here, and is called from BOTH CastLock (to stamp the ledger) and
    the render-time resolver (to open the file), because those two must agree.
    A second copy of this draw in the caster would let the ledger name one voice
    while the render opened another -- which is the exact defect the CastLock
    stamp exists to close.

    Role-matching refs are preferred so an announcer render gets an announcer
    ref, then ANY ref for the engine.

    ``used`` excludes references another character already holds, using the same
    two-pass shape as ``_ladder_pick``: prefer an unused reference, and fall back
    to the whole pool only when every candidate is taken (a voice is always
    better than none). The caster passes it because it tracks the episode's
    used-set; the render-time resolver has no cross-character state and passes
    nothing, which is harmless -- once the caster stamps an id, the resolver
    matches on that id and never reaches this draw at all.

    RESERVED AND REJECTED REFERENCES ARE NOT IN THIS POOL EITHER, and this was
    the THIRD place the reservation had to be taught (2026-08-18). It is the
    easiest one to miss because it is not a "caster" by name -- but it draws a
    real voice that both the ledger stamp and the render path use, so a reserved
    id reached from here is heard exactly like one reached from the selector.
    It is not a rare branch: `other` is 20% of every gender roll and the bank
    carries zero rows for it, so every such row lands here, and a uniform draw
    over the engine's refs handed out Lemmy's clone at roughly the same odds as
    any other voice (measured: 7-9 of 200 draws per engine before this filter).
    The reject filter matches ``assign_voice_for_slot``'s pool pre-filter; it is
    a no-op against the shipped bank (zero rows are tiered `reject` today) and
    exists so an audited reject cannot be rendered by the one path that used to
    skip the audit.
    """
    reserved = reserved_voice_ref_ids()

    def _eligible(entry) -> bool:
        return (entry.engine == engine
                and entry.voice_ref_id not in reserved
                and getattr(entry, "quality_tier", "") != "reject")

    role_cands = [e for e in bank if _eligible(e) and role in e.roles]
    cands = sorted(
        role_cands or [e for e in bank if _eligible(e)],
        key=lambda e: e.voice_ref_id,
    )
    if not cands:
        return None
    rng = random.Random(f"{episode_seed}_{char_id}_anyref")
    if used:
        unused = [e for e in cands if not _entry_is_used(e, set(used))]
        if unused:
            return rng.choice(unused)
    return rng.choice(cands)


def _seeded_preferred_announcer_voice_ref(
    bank: Tuple[VoiceBankEntry, ...], *, engine: str, episode_seed=0,
) -> VoiceBankEntry:
    """Deterministic preferred-announcer selection for engines with a curated
    announcer pool.

    When both male- and female-coded preferred announcers are available, select
    the gender from the episode seed for a stable roughly 50/50 mix. Within the
    gender, prefer entries tagged preferred_announcer/british_leaning.
    """
    cands = [
        e for e in bank
        if e.engine == engine
        and "announcer_voice" in e.roles
        and e.quality_tier != "reject"
    ]
    if not cands:
        raise VoiceCastingError(
            f"no announcer voice reference for engine {engine!r}")
    preferred = [e for e in cands if "preferred_announcer" in e.style_tags]
    pool = preferred or cands
    by_gender = {}
    for entry in pool:
        by_gender.setdefault(entry.gender, []).append(entry)
    wanted = _seeded_announcer_gender(engine, episode_seed)
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
    # Draw WITHIN the selected gender rather than taking the head of the sorted
    # pool. Sorting first keeps the tie-break order stable, so an engine that
    # offers exactly one preferred announcer per gender (google_tts, chatterbox,
    # dia) draws from a one-element list and is byte-identical to the old pin.
    # Engines with a real pool (kokoro) now reach every voice in it.
    ordered = sorted(pool, key=_key)
    pick_seed = hashlib.sha1(
        ("%s_announcer_pick:%s" % (
            engine, "" if episode_seed is None else episode_seed,
        )).encode("utf-8")
    ).hexdigest()
    return random.Random(pick_seed).choice(ordered)


def announcer_voice_ref(
    engine: str, bank: Optional[Tuple[VoiceBankEntry, ...]] = None,
    episode_seed=0,
) -> VoiceBankEntry:
    """The announcer reference for ``engine`` (E.1). Raise if none.

    Legacy engines keep their deterministic pin: the lowest ``voice_ref_id``
    among announcer-role references. Curated modern announcer pools mix
    male/female preferred announcer candidates by episode seed when both are
    available.
    """
    entries = bank if bank is not None else load_voice_bank()[0]
    if engine in _SEEDED_ANNOUNCER_ENGINES:
        return _seeded_preferred_announcer_voice_ref(
            entries, engine=engine, episode_seed=episode_seed)
    cands = sorted(
        [e for e in entries if e.engine == engine and "announcer_voice" in e.roles],
        key=lambda e: e.voice_ref_id,
    )
    if not cands:
        raise VoiceCastingError(
            f"no announcer voice reference for engine {engine!r}"
        )
    return cands[0]
