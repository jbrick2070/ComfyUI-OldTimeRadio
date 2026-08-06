"""Join a cast slot name to the gender the SOURCE actually records.

Why this exists
---------------
The row gender for an adaptation character was decided by a 40/40/20 roll, not by
the play. Measured across every published adaptation ledger, 44 non-announcer rows
carry a gender that contradicts the shipped provenance sidecar -- MALVOLIO and LEAR
cast female, MIRANDA and CORDELIA cast male. The roster truth was already on disk
in 14 provenance sidecars and simply never reached the writer.

`slot.gender` is not a voice field. It feeds the description LLM, the outline
prompt, the dialogue cast block and the image prompt's gender anchor, so this join
corrects the script and the portrait as well as the voice.

What this module will and will not do
-------------------------------------
It reads a confirmed record and abstains honestly. A slot that no tier resolves,
or whose candidates disagree, comes back "unknown" and is left to the existing
roll -- no worse than today. It never guesses from a name, never calls an LLM, and
never overwrites a confirmed sidecar fact.

Deliberately a separate module from `_otr_character_roster`, whose docstring
states the render path reads a confirmed record and never infers. That module runs
at VENDOR time; this one runs on the render path. Keeping them apart makes the
boundary structural instead of conventional.

Everything returned into ledger meta is a plain dict: `_copy_sidecar`
(`_otr_source_payload.py:195`) is a shallow `dict(sidecar)`, so these objects alias
straight into durable meta and are json-dumped. A MappingProxyType or a frozen
dataclass there is a serialization failure waiting for a render.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

log = logging.getLogger("OTR")

# Honorifics a speech prefix drops: the sidecar records "SIR TOBY", the manifest
# hint says "Toby". Mirrors the shape at _otr_character_roster.py:176-192.
_HONORIFICS = frozenset({
    "sir", "lady", "lord", "king", "queen", "duke", "duchess", "prince",
    "princess", "don", "friar", "master", "mistress", "doctor", "captain",
    "saint", "st", "mr", "mrs", "ms",
})

_BINARY = ("male", "female")

# --------------------------------------------------------------------------- #
# ONE gender normalization boundary (item 8, converged 2026-08-06).
#
# The published corpus carries EIGHTEEN distinct gender strings across 5,123 cast
# rows -- `male`, `female`, `other`, `non-binary`, `unspecified`, `neutral`,
# `woman`, `man`, `any`, `artificial`, `ai`, `synthetic`, `genderfluid`, `n/a`,
# `unknown`, `various`, `child-like`, plus absent. `_VALID_GENDERS` in
# `_otr_casting.py` is enforced only on the CastingResponse path; every other
# producer route bypasses it, which is how the rest got in.
#
# This is the single owner. It lives HERE because this module is a stdlib-only
# LEAF (json / dataclasses / pathlib / typing) that imports none of its consumers
# -- `_otr_casting`, `cast_lock`, `_otr_voice_bank`, `otr_meta_brief_image_prompt`,
# `_otr_voice_node_common`, `story_orchestrator` and the content-owned lane
# runners can all call it with no import cycle in either direction.
#
# READ-THROUGH, never write-back: OTR_CastLock sits AFTER the freeze cascade and
# its whole contract is byte-safety, so normalization is applied at every READ
# site and a frozen row's stored value is left exactly as written. Only a
# producer minting a fresh row stores the normalized form.
_GENDER_TOKENS: dict = {
    "male": "male", "m": "male", "man": "male",
    "female": "female", "f": "female", "woman": "female",
    # Everything the corpus actually carries that is neither binary. Listed
    # EXPLICITLY rather than caught by an else-branch, because a closed table is
    # what makes strict mode able to reject an unlisted value at all -- an
    # "anything else -> other" rule would make strict mode unreachable.
    "other": "other",
    "non-binary": "other", "nonbinary": "other", "non binary": "other",
    "neutral": "other", "unspecified": "other", "any": "other",
    "unknown": "other", "n/a": "other", "na": "other", "none": "other",
    "various": "other", "artificial": "other", "ai": "other",
    "synthetic": "other", "genderfluid": "other", "gender-fluid": "other",
    "child-like": "other", "childlike": "other",
}

#: Bumped when the voice/portrait consistency CONTRACT changes -- never per run.
#: `cast_lock_revision` cannot serve this: it increments every time the node
#: executes, so it cannot tell a policy-era ledger from a pre-policy one. Nor can
#: "is `presentation_gender` present?", which cannot distinguish POLICY ABSENT
#: from FIELD DROPPED -- exactly the ambiguity that let the spoken-citation
#: receipt regress unnoticed. Absence of the stamp means legacy, always.
VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION = 1

#: The durable ledger key carrying the revision above.
VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY = "voice_portrait_consistency_policy_revision"


def normalize_gender(raw, *, strict: bool = False) -> str:
    """Map any recorded gender string to exactly ``male`` / ``female`` / ``other``.

    ``strict=True`` (policy-revision rows) RAISES ``ValueError`` on a blank or
    unlisted value -- a new producer writing a gender nobody can resolve is a
    defect, not a shrug. ``strict=False`` (legacy read-through) warns once per
    distinct value and maps the unlisted to ``other`` so historical ledgers stay
    readable.

    Pure; no I/O. The returned value is what every consumer compares against --
    never the raw string, because `woman` is not `female` to an equality test and
    that is how a correctly-gendered row reached the gender-agnostic voice path.
    """
    token = str(raw or "").strip().lower()
    mapped = _GENDER_TOKENS.get(token)
    if mapped is not None:
        return mapped
    if strict:
        raise ValueError(
            "unmapped gender %r under policy revision %d; add it to "
            "_GENDER_TOKENS or fix the producer"
            % (raw, VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION)
        )
    if token not in _UNMAPPED_SEEN:
        _UNMAPPED_SEEN.add(token)
        log.warning(
            "[roster_gender] unmapped gender %r -> 'other' (legacy read-through)",
            raw,
        )
    return "other"


#: Distinct unmapped values already warned about, so a 1,595-ledger scan does not
#: emit the same warning thousands of times. Diagnosis lives in the audit, which
#: reports counted and path-attributed findings; this is only noise control.
_UNMAPPED_SEEN: set = set()


_SUPPLEMENT_FILENAME = "roster_gender_supplement.json"

# Cache keyed by resolved path + mtime so an edited supplement is picked up
# without a process restart, and a hot render path does not re-read the file.
_SUPPLEMENT_CACHE: dict = {}


class RosterGenderError(RuntimeError):
    """A committed supplement contradicts a confirmed sidecar fact."""


@dataclass(frozen=True)
class RosterGenderVerdict:
    """What the join concluded, and why -- the 'why' is stamped in the ledger."""

    gender: str          # "male" | "female" | "unknown"
    evidence: str        # a quotable reason, or the abstention code
    tier: str            # exact | alias | qualified | contains | supplement | none
    matched: tuple       # the roster names that produced this verdict

    @property
    def is_pinned(self) -> bool:
        return self.gender in _BINARY


def sidecar_path_for_text(text_path) -> Path:
    """The provenance sidecar beside a vendored source text.

    Matches the writer at scripts/otr_fetch_public_domain.py:244-246 -- the
    sidecar is the text's stem plus '.provenance.json', in the text's own
    directory.
    """
    p = Path(text_path)
    return p.parent / (p.stem + ".provenance.json")


def load_roster_characters(text_path) -> tuple:
    """The sidecar's `characters` list as PLAIN dicts, or () if unavailable.

    Absence is not an error: fixture texts and freshly added sources may have no
    sidecar, and the caller falls back to today's roll.
    """
    path = sidecar_path_for_text(text_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ()
    rows = data.get("characters") if isinstance(data, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    out = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append(dict(row))
    return tuple(out)


def _norm(value) -> str:
    return str(value or "").strip().upper()


def _strip_honorifics(name: str) -> str:
    words = [w for w in str(name or "").split() if w]
    kept = [w for w in words if w.lower().strip(".,'") not in _HONORIFICS]
    return " ".join(kept) if kept else " ".join(words)


def _candidate_names(row: Mapping) -> tuple:
    """The names a slot might legitimately match this roster row by."""
    names = []
    for key in ("name", "roster_name"):
        n = _norm(row.get(key))
        if n and n not in names:
            names.append(n)
    return tuple(names)


def _verdict_from(rows: Sequence, tier: str) -> RosterGenderVerdict:
    """Collapse a tier's candidate rows into one honest verdict."""
    matched = tuple(_norm(r.get("name")) or _norm(r.get("roster_name")) for r in rows)
    genders = {str(r.get("gender") or "").strip().lower() for r in rows}
    binary = {g for g in genders if g in _BINARY}
    if len(binary) > 1:
        # Two source people share this prefix and disagree. Abstain -- picking
        # one would be a coin flip wearing a citation.
        return RosterGenderVerdict("unknown", "ambiguous_join", tier, matched)
    if len(binary) == 1:
        gender = binary.pop()
        reason = ""
        for r in rows:
            if str(r.get("gender") or "").strip().lower() == gender:
                reason = str(r.get("description") or "").strip()
                if not reason:
                    reason = "sidecar gender_source=%s" % (
                        str(r.get("gender_source") or "unknown"),
                    )
                break
        return RosterGenderVerdict(gender, reason, tier, matched)
    return RosterGenderVerdict("unknown", "roster_unknown", tier, matched)


def resolve_roster_gender(slot_name, characters: Iterable) -> RosterGenderVerdict:
    """Resolve one cast slot name against a sidecar `characters` list.

    Tier ladder; the FIRST tier that yields any candidate wins, and the whole
    winning tier must agree:

      exact      slot == name or roster_name
      short_form slot == the honorific-stripped form (TOBY for SIR TOBY)
      qualified  a roster name starts with 'slot ' -- ANTIPHOLUS matching
                 'ANTIPHOLUS OF EPHESUS' (and both DROMIOs, which is why that
                 case must abstain rather than agree)
      contains   slot starts with 'name ' -- the inverse shape

    Ambiguity abstains. A tier that matches only unknown-gender rows abstains too,
    and does NOT fall through to a looser tier: the source named this person and
    declined to gender them, which is an answer.
    """
    slot = _norm(slot_name)
    rows = [dict(r) for r in (characters or []) if isinstance(r, Mapping)]
    if not slot or not rows:
        return RosterGenderVerdict("unknown", "no_roster", "none", ())

    exact = [r for r in rows if slot in _candidate_names(r)]
    if exact:
        return _verdict_from(exact, "exact")

    slot_short = _strip_honorifics(slot)
    short_form = [
        r for r in rows
        if slot_short and slot_short in tuple(
            _strip_honorifics(n) for n in _candidate_names(r)
        )
    ]
    if short_form:
        return _verdict_from(short_form, "short_form")

    qualified = [
        r for r in rows
        if any(n.startswith(slot + " ") for n in _candidate_names(r))
    ]
    if qualified:
        return _verdict_from(qualified, "qualified")

    contains = [
        r for r in rows
        if any(n and slot.startswith(n + " ") for n in _candidate_names(r))
    ]
    if contains:
        return _verdict_from(contains, "contains")

    return RosterGenderVerdict("unknown", "no_match", "none", ())


_REPO_ROOT = Path(__file__).resolve().parent.parent


def supplement_dir_for_bank(bank_row) -> Optional[Path]:
    """Where a source bank's curated supplement lives: beside its manifest.

    Returns None when the bank declares no manifest, which is every invention
    lane -- they never reach the join at all.
    """
    defaults = getattr(bank_row, "defaults", None) or {}
    if not isinstance(defaults, Mapping):
        return None
    raw = str(defaults.get("manifest_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.parent


def _supplement_path(bank_dir) -> Path:
    return Path(bank_dir) / _SUPPLEMENT_FILENAME


def load_gender_supplement(bank_dir) -> dict:
    """The committed curated supplement for a source bank, or {} if absent."""
    if bank_dir is None:
        return {}
    path = _supplement_path(bank_dir)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return {}
    cached = _SUPPLEMENT_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    entries = data.get("entries") if isinstance(data, Mapping) else None
    if not isinstance(entries, Mapping):
        return {}
    normalized = {}
    for play_code, names in entries.items():
        if not isinstance(names, Mapping):
            continue
        bucket = {}
        for name, spec in names.items():
            if not isinstance(spec, Mapping):
                continue
            gender = str(spec.get("gender") or "").strip().lower()
            evidence = str(spec.get("evidence") or "").strip()
            if gender not in _BINARY:
                raise RosterGenderError(
                    "supplement %s/%s: gender must be male or female, got %r"
                    % (play_code, name, spec.get("gender"))
                )
            if not evidence:
                raise RosterGenderError(
                    "supplement %s/%s: a non-empty evidence string is required"
                    % (play_code, name)
                )
            bucket[_norm(name)] = {"gender": gender, "evidence": evidence}
        normalized[str(play_code)] = bucket
    _SUPPLEMENT_CACHE[str(path)] = (stamp, normalized)
    return normalized


def resolve_with_supplement(
    slot_name,
    characters: Iterable,
    *,
    play_code: str = "",
    supplement: Optional[Mapping] = None,
) -> RosterGenderVerdict:
    """The roster ladder, then the committed supplement for what it could not reach.

    MERGE RULE, enforced rather than documented: the supplement may only fill a
    slot the roster left unknown. A supplement entry that contradicts a CONFIRMED
    sidecar gender raises -- a curated convenience file must never be able to
    overrule the source itself.
    """
    verdict = resolve_roster_gender(slot_name, characters)
    bucket = (supplement or {}).get(str(play_code or "")) or {}
    entry = bucket.get(_norm(slot_name))
    if not entry:
        return verdict
    if verdict.is_pinned:
        if entry["gender"] != verdict.gender:
            raise RosterGenderError(
                "supplement contradicts the sidecar for %r: sidecar says %s, "
                "supplement says %s" % (slot_name, verdict.gender, entry["gender"])
            )
        return verdict
    if verdict.evidence == "ambiguous_join":
        # Two named source people disagree. A curated line cannot adjudicate
        # which one this slot is; leave it to the roll.
        return verdict
    return RosterGenderVerdict(
        entry["gender"], entry["evidence"], "supplement", verdict.matched,
    )


def gender_map_for_names(
    names: Iterable,
    characters: Iterable,
    *,
    play_code: str = "",
    supplement: Optional[Mapping] = None,
) -> dict:
    """UPPER-CASED name -> verdict dict, for every name that RESOLVED.

    Unresolved names are omitted, never recorded as 'unknown': the caller pins
    what it knows and leaves the rest to the existing allocator untouched.
    """
    out = {}
    for name in names or []:
        key = _norm(name)
        if not key or key in out:
            continue
        verdict = resolve_with_supplement(
            name, characters, play_code=play_code, supplement=supplement,
        )
        if verdict.is_pinned:
            out[key] = {
                "gender": verdict.gender,
                "evidence": verdict.evidence,
                "tier": verdict.tier,
                "roster_name": verdict.matched[0] if verdict.matched else key,
            }
    return out
