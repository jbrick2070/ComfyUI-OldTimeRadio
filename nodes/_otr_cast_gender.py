"""Every cast row ends male or female -- the one ladder, for every lane.

OPERATOR RULING (2026-10-17): "unisex names is ok but we need to clamp male or
female in the ledger and voice and descriptions for rendering prompts, stills
and video". A name may stay unisex; the GENDER on the ledger cast row may not.
Voice casting picks from that stamped gender, and the still and video prompts
anchor on it, so nothing downstream ever meets an unresolved "other".

WHERE IT RUNS. The producers resolve gender BEFORE any model writes a bio or a
line (`_otr_casting` no longer deals an "other" share; My Story settles every
role before its acts; Sci-Fi News Pro's casting schema is already binary).
`production_ledger.Ledger.set_cast` calls :func:`clamp_cast_rows` as the
safety net every lane passes through, and says so LOUDLY when it fires,
because a clamp there means a producer let a non-binary gender through.

THE LADDER, strongest evidence first (each rung declines rather than guess):
  1. a courtesy title the name LEADS with ("Mrs. Vance", "Sir Toby");
  2. a relation word in the role ("her brother"), possessives skipped;
  3. our name pool, honorifics stripped (unisex / unknown names decline);
  4. subject/reflexive pronouns in the description -- only when the
     description names no other cast member and they all agree;
  4b. a known character: a name whose first word (when no honorific leads)
      our vendored source rosters record with one gender ("JULIET" female,
      "LEAR" male), never one the operator locked to stay on the roll;
  5. a coin seeded on the normalized name, so the same name casts the same way.

Pure, deterministic, stdlib + repo modules only.
"""
from __future__ import annotations

try:
    from ._otr_shared import cast_text as _CAST_TEXT
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import cast_text as _CAST_TEXT  # type: ignore

import logging
import random
import re
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from . import _otr_roster_gender as _RG
    from ._otr_character_roster import _FEMALE_RELATION, _MALE_RELATION, _scan
except ImportError:  # pragma: no cover -- flat load
    import _otr_roster_gender as _RG  # type: ignore
    from _otr_character_roster import (  # type: ignore
        _FEMALE_RELATION, _MALE_RELATION, _scan)

try:
    from ..config import cast_pools as _POOLS
except ImportError:  # pragma: no cover -- flat load
    from config import cast_pools as _POOLS  # type: ignore

log = logging.getLogger("OTR")

BINARY = ("male", "female")

_SUBJECT_PRONOUNS = {
    "she": "female", "herself": "female", "he": "male", "himself": "male",
}
#: A pronoun IMMEDIATELY after one of these opens a reported clause, usually
#: about someone else ("doubts that she saw lights").
_SUBORDINATORS = frozenset({"that", "whether"})
#: Words a cast name can contain that are not that person ("The Doctor").
_NAME_ARTICLES = frozenset({"the", "a", "an", "of", "and", "de", "la", "le",
                            "el", "los", "las", "du", "von", "van"})


def binary(raw) -> str:
    """``male``/``female`` for any spelling of them, else ``""``."""
    g = _RG.canonical_bank_gender(raw)
    return g if g in BINARY else ""


def name_forms(name: str) -> "tuple[tuple[str, ...], ...]":
    """A name's word forms: honorifics dropped, and as typed."""
    words = tuple(re.findall(r"[^\W_]+(?:['\-][^\W_]+)*",
                             _RG.fold_accents(str(name or "")).casefold()))
    kept = tuple(w for w in words if w not in _RG._HONORIFICS)
    return tuple(dict.fromkeys(f for f in (kept, words) if f))


def name_tokens(name: str) -> "tuple[str, ...]":
    """Casefolded words of a name, honorifics dropped (as typed if that
    leaves nothing)."""
    forms = name_forms(name)
    return forms[0] if forms else ()


def description_gender(description: str, others: Sequence[str] = ()) -> str:
    """The gender a description's own subject/reflexive pronouns state, or "".

    Object and possessive pronouns are no evidence ("her brother"). A pronoun
    right after "that"/"whether" opens a reported clause and is skipped. A
    description that names another cast member (articles aside) gives no
    pronoun evidence at all -- no word window can tell whose pronoun it is.
    """
    words = re.findall(r"[^\W_]+", str(description or "").lower())
    names = {w for name in others for form in name_forms(name)
             for w in form} - _NAME_ARTICLES
    if names.intersection(words):
        return ""
    seen = {_SUBJECT_PRONOUNS[w] for i, w in enumerate(words)
            if w in _SUBJECT_PRONOUNS
            and not (i and words[i - 1] in _SUBORDINATORS)}
    return next(iter(seen)) if len(seen) == 1 else ""


_KNOWN_CHARACTERS: "dict[str, str] | None" = None


def known_character_gender(name: str) -> str:
    """The gender our vendored sources' rosters record for a one-word
    character name ("JULIET", "LEAR", "PROSPERO"), or "".

    Built once from ``config/source_banks/*/sources/*.provenance.json`` --
    the gender-stamped rosters the adaptation lanes already pin from -- so the
    curated first-name pool, which knows modern names, is not the only
    registry (operator, 2026-10-17: "Juliet should be female"). Only names
    recorded as a single word, with one gender everywhere they appear, and
    never a rank or relation word ("NURSE", "FOOL", "PRINCE").
    """
    global _KNOWN_CHARACTERS
    if _KNOWN_CHARACTERS is None:
        import glob
        import json
        import os

        try:
            from ._otr_character_roster import _FEMALE_TITLE, _MALE_TITLE
        except ImportError:  # pragma: no cover -- flat load
            from _otr_character_roster import _FEMALE_TITLE, _MALE_TITLE  # type: ignore
        vocab = (set(_MALE_TITLE) | set(_FEMALE_TITLE) | set(_MALE_RELATION)
                 | set(_FEMALE_RELATION) | set(_RG._HONORIFICS))
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config", "source_banks")
        seen: "dict[str, set]" = {}
        for path in glob.glob(os.path.join(root, "*", "sources",
                                           "*.provenance.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    chars = json.load(fh).get("characters") or []
            except (OSError, ValueError):
                continue
            for c in chars:
                g = binary((c or {}).get("gender"))
                words = re.findall(r"[^\W_]+(?:['\-][^\W_]+)*",
                                   _RG.fold_accents(str((c or {}).get("name") or "")).casefold())
                if g and len(words) == 1 and len(words[0]) >= 3 \
                        and words[0] not in vocab:
                    seen.setdefault(words[0], set()).add(g)
        # An operator ruling "stays on the roll" (a locked entry with no
        # gender in any bank's character_gender_index.json -- ARIEL, PUCK)
        # is never overridden by another play's roster.
        on_the_roll = set()
        for path in glob.glob(os.path.join(root, "*",
                                           "character_gender_index.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    entries = json.load(fh).get("entries") or {}
            except (OSError, ValueError):
                continue
            for key, entry in entries.items():
                if (entry or {}).get("locked") and not (entry or {}).get("gender"):
                    on_the_roll.add(_RG.fold_accents(str(key).rsplit("|", 1)[-1]).casefold())
        _KNOWN_CHARACTERS = {k: next(iter(v)) for k, v in seen.items()
                             if len(v) == 1 and k not in on_the_roll}
    # The name's FIRST word, unless the name opens with an honorific: in "Dr.
    # Harris" or "Mrs. Poole" the first real word is a surname, not the
    # character the roster means. "Juliet Capulet" reads "juliet".
    words = re.findall(r"[^\W_]+(?:['\-][^\W_]+)*", _RG.fold_accents(str(name or "")).casefold())
    if not words or words[0] in _RG._HONORIFICS:
        return ""
    return _KNOWN_CHARACTERS.get(words[0], "")


def settle_binary_gender(name: str, role: str = "", description: str = "",
                         others: Sequence[str] = ()) -> "tuple[str, str]":
    """``(gender, source)`` -- always ``male`` or ``female``. See the ladder
    in the module docstring."""
    gender = _RG.slot_honorific_gender(name)
    if gender in BINARY:
        return gender, "courtesy_title"
    role_text = re.sub(r"\b[a-z]+'s\b", " ", str(role or "").lower())
    gender = _scan(re.findall(r"[a-z]+", role_text),
                   _MALE_RELATION, _FEMALE_RELATION)
    if gender in BINARY:
        return gender, "role_relation"
    gender = _POOLS.gender_of_first_name(" ".join(name_tokens(name)))
    if gender in BINARY:
        return gender, "name_pool"
    gender = description_gender(description, others)
    if gender:
        return gender, "description_pronouns"
    # Below the description: the index holds some surnames and place names
    # ("Harris", "France"), so a description that says "she" must win.
    gender = known_character_gender(name)
    if gender:
        return gender, "known_character"
    key = " ".join(str(name or "").split()).casefold()
    coin = random.Random("otr_cast_gender:%s" % key)
    return ("male" if coin.random() < 0.5 else "female"), "seeded_coin"


def clamp_cast_rows(rows: Sequence[MutableMapping]) -> "list[dict]":
    """Clamp every row's gender to male/female IN PLACE; return receipts.

    A row already male/female (any spelling) is canonicalized and otherwise
    untouched. Mutating the caller's dicts is deliberate: a writer that keeps
    reading its own rows after `set_cast` must see the same gender the ledger
    holds.
    """
    receipts: List[dict] = []
    names = [str(r.get("name") or "") for r in rows]
    for i, row in enumerate(rows):
        before = row.get("gender")
        g = binary(before)
        if g:
            if before != g:
                row["gender"] = g
            continue
        others = [n for j, n in enumerate(names) if j != i]
        g, source = settle_binary_gender(
            names[i], str(row.get("role") or ""),
            " ".join(_CAST_TEXT.identity_fields(row).values()), others)
        row["gender"] = g
        receipts.append({"char_id": str(row.get("char_id") or ""),
                         "name": names[i], "gender": g, "source": source,
                         "was": before})
        log.warning(
            "[cast_gender] CLAMPED %r (%s) from %r to %s (%s) -- a producer "
            "let a non-binary gender reach the ledger", names[i],
            row.get("char_id"), before, g, source)
    return receipts


__all__ = [
    "BINARY", "binary", "clamp_cast_rows", "description_gender",
    "known_character_gender",
    "name_forms", "name_tokens", "settle_binary_gender",
]
