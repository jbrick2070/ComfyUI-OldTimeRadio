"""ONE naming authority, enforced at the boundary -- Bug Bible ``11.61``.

An upstream creative pass invents entity names; a downstream deterministic
assigner overrides them with its own; and the per-record prompt is handed BOTH
with no precedence stated. The model then fills its free-text
``<story-linked role>`` slot with the upstream name, and the record describes
somebody else. Nothing errors -- the field is non-empty, on-format, schema-valid
and well written.

``11.61`` forbids two reflexes by name, and this module implements neither:

* **NOT prompt persuasion.** *"DO NOT fix it by instructing the model harder ...
  the guarantee has to be structural."* So the conflicting name is removed from
  the context BEFORE generation (:func:`reconcile_text`).
* **NOT fuzzy repair of the output.** *"it rewrites the foreign name to the
  record's own and leaves that other person's face, bearing and delivery prose
  in place."* So :func:`find_foreign_identities` REPORTS; it never rewrites a
  generated string. A contaminated response is discarded whole and regenerated,
  never laundered.

**WHY NOT ``11.61``'s PREFERRED "REWRITTEN to the assigned ones"?** That needs a
superseded-identity -> roster-slot mapping and the only available candidate is
ordinal position. It does not hold: when LEMMY is cast,
``assemble_pre_locked_rows`` sets ``remaining_open = num_characters - 1``, so
there are N upstream identities and N-1 open slots and every pairing shifts by
one -- silently giving a row the wrong person's traits, which IS the defect. The
upstream cast entry also carries no role field to match on. So this module takes
the entry's explicit ``(or removed)`` branch, with a DISTINCT neutral label per
identity.

**THE LABEL IS DISTINCT PER IDENTITY ON PURPOSE.** Collapsing several people to
one shared token ("this character") destroys which traits belong to whom and
invites a blended description -- a different wrong-person bug wearing new clothes.

**LANE-NEUTRAL BY CONSTRUCTION.** Nothing here knows what a pitch, a news
article or a play is. Callers classify their own names; adaptation lanes whose
roster IS the source's cast simply report no superseded identities, so
``shakespeare`` and ``public_domain`` are untouched and their fidelity is never
attacked.

This module is imported by BOTH the runtime guard and the offline archive sweep
(``scripts/audit_wrong_person_census.py``). One implementation, so the sweep can
never certify something different from what runtime enforces.

WHAT THIS COVERS, AND WHAT IT DOES NOT -- stated because a guard whose reach is
assumed rather than known is how the last census reported a blind spot as a
clean bill of health:

* **Covered.** Lanes that build their cast through ``lock_cast`` AND hand it
  structured upstream identities. Today that is the ``original`` family, which
  records ``selected_concept.cast`` -- and which carried every hit the
  structured census could see.
* **Layer 2 only.** ``media_archive`` exhibits the identical defect (verified:
  ``ADRIAN CARRUTHERS`` carrying *"Dr. Amelia Hartley"*, ``DALE SPENDER``
  carrying *"Dr. Amelia Hartfield"*) but names its people only in free-text
  prose, so there is no structured surface to reconcile. Harvesting names from
  the prose was measured and REJECTED: it fires on the healthy Title-Case
  occupation head (``"30s, skeptical Film Historian"``), which is the dominant
  correct style. Giving that lane structured identities is its own item.
* **Not reached at all.** ``scifi_news_pro`` derives its cast from the script
  the model already wrote (``_speakers_in_order`` -> ``_assign_voices``) and
  never calls ``lock_cast``.
* **Correctly inert.** ``shakespeare`` and ``public_domain``: the roster IS the
  source's cast, so :func:`superseded_identities` returns nothing and their
  prompts stay byte-identical. Their fidelity is never attacked.
* **Not guaranteed.** ``OTR_NAME_MODE=llm_slot_fill`` renames rows AFTER their
  descriptions are written, so reconciliation computed against the pre-fill
  roster does not describe the final names. ``lock_cast`` records this rather
  than pretending otherwise. Pool mode is the production default.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence

# Typographic apostrophes/quotes are the live hazard: an upstream identity often
# carries a quoted nickname ("HENRY 'HANK' GRISWOLD") and any pass through an
# LLM, an editor or a JSON round-trip can swap the glyph. Unify before comparing.
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "ʼ": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
}
_DASHES = "‐‑‒–—―−"
_STRIP_EDGE = ".,;:!?()[]{}'\"`"

#: Titles and honorifics are ROLE words, not identity. They must never be
#: redacted on their own ("Captain" is legitimate colour) and must never be
#: treated as a name token, or "Professor & Lab Director" yields a false
#: "Professor Lab" -- an honorific-adjacency class measured on this corpus.
_TITLE_WORDS = frozenset({
    "dr", "mr", "mrs", "ms", "miss", "sir", "lady", "lord", "professor",
    "captain", "sergeant", "officer", "father", "mother", "sister", "brother",
    "rev", "reverend", "madame", "madam", "colonel", "major", "general",
    "detective", "inspector", "nurse", "aunt", "uncle",
})

#: Ordinary English words that are ALSO common surnames in the name pool. A
#: short form matching one of these cannot be admitted on its own: upstream
#: ``EDWARD STONE`` would otherwise rewrite *"A stone wall frames the harbour"*
#: into *"A CHARACTER A wall frames the harbour"*, mangling correct prose to
#: remove a name that was never there. The full name is still matched.
_ORDINARY_WORDS = frozenset({
    "stone", "gray", "grey", "reed", "reeds", "brooks", "rivers", "river",
    "hill", "hills", "banks", "bank", "fields", "field", "moss", "frost",
    "snow", "rain", "storm", "wolf", "fox", "bell", "bells", "cross", "king",
    "price", "young", "sharp", "swift", "brown", "black", "white", "green",
    "long", "short", "small", "little", "best", "love", "hope", "grace",
    "rose", "dawn", "day", "night", "summer", "winter", "west", "north",
    "south", "east", "church", "wood", "woods", "park", "lane", "ford",
    "marsh", "glass", "steel", "iron", "flint", "sands", "shore", "still",
})

#: Relational connectors. A superseded name AFTER one of these is a MENTION of
#: somebody else, not a claim to be them -- ``"foil to Hiram's obsession"``
#: describes THIS row's relationship and is exactly the legitimate prose Bug
#: Bible 11.61 warns must not be flagged.
_RELATIONAL_ROLES = (
    r"foil|rival|counterpart|opposite|answer|shadow|match|nemesis|"
    r"confidante|confidant|friend|enemy|partner|assistant|deputy|second|aide|"
    r"ally|daughter|son|child|brother|sister|wife|husband|widow|mother|father|"
    r"protege|apprentice|student|teacher|mentor|heir|servant|employer|boss"
)
#: A superseded name AFTER a relational connector is a MENTION of somebody
#: else, not a claim to be them. The connector may be separated from the
#: matched surface by the rest of that person's name -- "rival to Hiram Bleek"
#: puts two words between "to" and the matched surname -- so intervening
#: capitalised name words are allowed before the match position.
_RELATIONAL_LEAD = re.compile(
    r"(?:(?:" + _RELATIONAL_ROLES + r")\s+(?:to|of|for|with)"
    r"|against|beside|alongside|versus|unlike|opposite)"
    r"(?:\s+[A-Z][\w''’-]*)*\s*$",
    re.IGNORECASE,
)

#: A single token shorter than this is too collision-prone to redact or flag on
#: its own -- the name pool contains ordinary words ("Stone", "Gray"), so a
#: 3-letter fragment would maul unrelated prose. Measured false-positive class.
_MIN_TOKEN_LEN = 4


class ForeignIdentity(NamedTuple):
    """One superseded identity found in a generated field."""

    identity: str      # the upstream identity as the producer supplied it
    matched: str       # the exact surface form that was found
    field: str         # which generated field it was found in
    label: str         # the neutral label this identity reconciles to


def normalize_text(value: Any) -> str:
    """Case-folded, quote-unified, whitespace-collapsed comparison form.

    NFKC first so composed and decomposed accents compare equal; then the glyph
    maps; then whitespace collapse, because a brief may wrap a name across a
    line break; then casefold. **The casefold is not cosmetic** -- the first
    census of this defect compared case-sensitively, so a pitch of
    ``ELIZABETH 'LIZZIE' WALSH`` never matched a row reading
    ``Elizabeth 'Lizzie' Walsh`` and an episode with BOTH dramatic rows
    contaminated was scored clean.
    """
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = "".join(_QUOTE_MAP.get(ch, ch) for ch in text)
    text = "".join("-" if ch in _DASHES else ch for ch in text)
    return " ".join(text.split()).casefold()


def name_tokens(normalized: str) -> frozenset:
    """Identity tokens of an already-normalised name.

    Titles are dropped: they are role words shared across unrelated people, so
    keeping them would make every "Dr." collide with every other "Dr.".
    """
    out = set()
    for word in normalized.split():
        token = word.strip(_STRIP_EDGE)
        if token and token not in _TITLE_WORDS:
            out.add(token)
    return frozenset(out)


def roster_owns(identity: str, roster_names: Sequence[str]) -> bool:
    """True when the roster genuinely owns this name, so it is NOT superseded.

    Ownership is normalised EQUALITY, or an identical token set (which absorbs
    word order and title differences: ``DR. JONAS REED`` and ``Reed, Jonas``).

    **SUBSET CONTAINMENT WAS REMOVED, and that is a correction.** It counted a
    single shared token as ownership of a whole person, so a roster row
    ``SOM STONE`` silently claimed the foreign identity ``DR. STONE`` and the
    guard stopped considering it -- a false NEGATIVE wearing the costume of
    caution. It was justified as protecting the adaptation shape (``MACBETH``
    against a roster ``LADY MACBETH``), but that protection is not needed:
    adaptation lanes record no ``selected_concept.cast``, so they supply NO
    upstream identities and never reach this function with the source's names
    at all. Where one ever did, the names ARE the roster rows and equality
    already covers it.
    """
    ident_norm = normalize_text(identity)
    if not ident_norm:
        return True
    ident_tokens = name_tokens(ident_norm)
    if not ident_tokens:
        return True
    for roster_name in roster_names:
        roster_norm = normalize_text(roster_name)
        if ident_norm == roster_norm:
            return True
        if ident_tokens == name_tokens(roster_norm):
            return True
    return False


def superseded_identities(
    upstream_names: Iterable[str],
    roster_names: Sequence[str],
) -> List[str]:
    """The upstream names the roster does NOT own, in input order, deduped.

    This is ``11.61``'s ensemble scope applied to the INPUT. An adaptation lane
    whose roster is the source's own cast returns an empty list here, which is
    why this module never attacks source fidelity.
    """
    out: List[str] = []
    seen = set()
    for raw in upstream_names or ():
        name = str(raw or "").strip()
        if not name or roster_owns(name, roster_names):
            continue
        key = normalize_text(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def identity_aliases(identity: str, roster_names: Sequence[str]) -> List[str]:
    """Every deterministic surface form of one identity, longest first.

    A brief rarely repeats the canonical string. ``ELIZABETH 'LIZZIE' WALSH``
    appears as ``Lizzie Gray``, ``'Eddie'`` or a bare surname, and matching only
    the full form leaves the intruder sitting in the prompt. Longest-first so a
    full name is consumed before its own parts.

    Tokens the ROSTER owns are never returned -- if the roster has a MING and the
    upstream identity is "Ming Chao", redacting "Ming" would attack the row's own
    name.
    """
    roster_tokens = set()
    for roster_name in roster_names:
        roster_tokens |= name_tokens(normalize_text(roster_name))

    forms = {identity.strip()}
    ident_norm = normalize_text(identity)

    # The name with its quoted nickname removed: "Edward 'Eddie' Stone" ->
    # "Edward Stone", which is how briefs usually write it in prose.
    stripped = re.sub(r"['\"][^'\"]{1,20}['\"]", " ", identity)
    stripped = " ".join(stripped.split())
    if stripped:
        forms.add(stripped)

    # Individual identity tokens, including the nickname itself.
    for token in name_tokens(ident_norm):
        if len(token) < _MIN_TOKEN_LEN or token in roster_tokens:
            continue
        for raw in re.split(r"[^A-Za-z']+", identity):
            if raw and normalize_text(raw).strip(_STRIP_EDGE) == token:
                forms.add(raw.strip(_STRIP_EDGE))

    cleaned = [f for f in (s.strip(_STRIP_EDGE + " ") for s in forms) if len(f) >= 2]
    return sorted(set(cleaned), key=len, reverse=True)


_APOSTROPHES = "'‘’‚‛ʼ′´`"
_HYPHENS = "-‐‑‒–—―−"


def _tolerant_fragment(fragment: str) -> str:
    """Regex source for one surface, tolerant of the glyphs we claim to unify.

    ``normalize_text`` unifies curly apostrophes, dash variants and whitespace
    runs -- but reconciliation and detection match against the LIVE text, so
    matching on the raw surface silently ignored all of it: ``ANA O'NEIL`` did
    not match *"Ana O’Neil"* and ``MARY-JANE DOE`` did not match
    *"Mary–Jane Doe"*, so wrong-person prose could ship past both layers.
    The pattern itself has to be as tolerant as the comparison claims to be.
    """
    out = []
    for ch in unicodedata.normalize("NFKC", fragment):
        if ch in _APOSTROPHES:
            out.append("[" + re.escape(_APOSTROPHES) + "]")
        elif ch in _HYPHENS:
            out.append("[" + re.escape(_HYPHENS) + "]")
        elif ch.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _boundary_pattern(fragment: str) -> "re.Pattern":
    """Match a fragment on letter boundaries so it cannot eat a longer word.

    Plain substring matching turns "Reed" into a hit inside "Reedy" and, worse,
    lets a redaction chew a word it does not own. Boundaries are letter-class
    rather than a word-boundary escape so an apostrophe or hyphen inside a name does not end
    the token.
    """
    return re.compile(
        r"(?<![^\W\d_])" + _tolerant_fragment(fragment) + r"(?![^\W\d_])",
        re.IGNORECASE | re.UNICODE,
    )


def default_label(index: int) -> str:
    """The neutral, DISTINCT stand-in for the Nth superseded identity.

    Not a person's name (so it cannot become a wrong-person description) and not
    shared between identities (so the brief keeps telling the model which traits
    belong to whom).
    """
    return f"CHARACTER {chr(ord('A') + index)}" if index < 26 else f"CHARACTER {index + 1}"


def build_alias_plan(
    identities: Sequence[str],
    roster_names: Sequence[str],
) -> "tuple[List[tuple], Dict[str, str]]":
    """Resolve ALL identities together into one unambiguous surface->label plan.

    Returns ``([(surface_form, label, identity)], {identity: label})`` sorted
    longest-first.

    **WHY THIS IS BUILT ACROSS THE WHOLE SET AND NOT PER IDENTITY.** Resolving
    one person at a time and substituting sequentially lets an early identity's
    SHORT form eat part of a later identity's full name. Measured: upstream
    ``["JONAS REED", "MARTHA REED"]`` over *"JONAS REED argues with MARTHA
    REED."* produced *"CHARACTER A argues with CHARACTER B CHARACTER A."* --
    the shared surname was consumed by the first person, so the second was
    rendered as TWO people. That is the same harm as giving everyone one shared
    label, arrived at from the opposite direction.

    So: a short form (bare token or nickname) is admitted ONLY when exactly one
    superseded identity claims it. Canonical full forms are always kept, because
    they are unambiguous by construction and are what detection depends on.
    """
    labels: Dict[str, str] = {}
    # EVERY surface competes in the same ambiguity check, canonical forms
    # included. Exempting canonical forms let one identity's canonical name
    # collide with another's nickname and the last writer won:
    # ``["ELIZABETH 'LIZZIE' WALSH", "LIZZIE"]`` collapsed BOTH people into
    # CHARACTER A, which is precisely the distinct-identity guarantee this
    # module exists to keep.
    claims: Dict[str, set] = {}
    surface_text: Dict[str, str] = {}
    is_canonical: Dict[str, bool] = {}

    for index, identity in enumerate(identities):
        label = default_label(index)
        labels[identity] = label
        ident_tokens = name_tokens(normalize_text(identity))
        for form in identity_aliases(identity, roster_names):
            form_norm = normalize_text(form)
            if not form_norm:
                continue
            claims.setdefault(form_norm, set()).add(identity)
            surface_text.setdefault(form_norm, form)
            canonical = name_tokens(form_norm) == ident_tokens
            is_canonical[form_norm] = is_canonical.get(form_norm, False) or canonical

    canonical_claims: Dict[str, set] = {}
    for index, identity in enumerate(identities):
        ident_tokens = name_tokens(normalize_text(identity))
        for form in identity_aliases(identity, roster_names):
            form_norm = normalize_text(form)
            if form_norm and name_tokens(form_norm) == ident_tokens:
                canonical_claims.setdefault(form_norm, set()).add(identity)

    plan: List[tuple] = []
    for form_norm, claimants in claims.items():
        if len(claimants) != 1:
            # Contested. If exactly ONE identity claims it as its own canonical
            # name, that identity wins -- otherwise a person whose whole name is
            # another person's nickname would lose their only surface and go
            # unreconciled entirely.
            owners = canonical_claims.get(form_norm) or set()
            if len(owners) != 1:
                continue    # genuinely ambiguous: it cannot say WHICH person
            claimants = owners
        only = next(iter(claimants))
        if not is_canonical[form_norm]:
            # A short form must also not be an ordinary English word, or
            # reconciliation mangles innocent prose to remove a name that was
            # never in it.
            tokens = form_norm.split()
            if len(tokens) == 1 and tokens[0].strip(_STRIP_EDGE) in _ORDINARY_WORDS:
                continue
        plan.append((surface_text[form_norm], labels[only], only))
    # Longest first so a full name is always consumed before any of its parts.
    plan.sort(key=lambda row: len(row[0]), reverse=True)
    return plan, labels


def reconcile_text(
    text: str,
    identities: Sequence[str],
    roster_names: Sequence[str],
) -> "tuple[str, Dict[str, str]]":
    """Replace every superseded identity with its neutral label.

    Returns ``(reconciled_text, {identity: label})``. Pure and total: it never
    raises, and with no identities it returns the input unchanged, so every
    lane that supplies none keeps byte-identical prompts.

    Substitution is ONE pass over a combined longest-first matcher, not a
    sequence of per-identity passes. A single pass consumes each position once,
    so a replacement can never be re-matched or partially overwritten by a
    later identity's shorter form.
    """
    if not text or not identities:
        return text, {}
    plan, labels = build_alias_plan(identities, roster_names)
    if not plan:
        return text, labels

    # Same tolerance as detection, or reconciliation would leave behind exactly
    # the glyph variants detection then fires on.
    by_norm = {normalize_text(surface): label for surface, label, _ident in plan}
    combined = re.compile(
        r"(?<![^\W\d_])(?:"
        + "|".join(_tolerant_fragment(surface) for surface, _l, _i in plan)
        + r")(?![^\W\d_])",
        re.IGNORECASE | re.UNICODE,
    )

    def _swap(match: "re.Match") -> str:
        return by_norm.get(normalize_text(match.group(0)), match.group(0))

    out = combined.sub(_swap, text)

    # Adjacent repeats of ONE label ("CHARACTER A CHARACTER A") read as two
    # people. This fires for real -- "Lizzie Walsh" is two separate admitted
    # short forms of one identity -- so the collapse is load-bearing, not
    # defensive decoration.
    for label in set(labels.values()):
        out = re.sub(
            r"(?:" + re.escape(label) + r")(?:[\s,']+" + re.escape(label) + r")+",
            label, out,
        )
    return " ".join(out.split()), labels


def _claims_the_identity(text: str, match: "re.Match") -> bool:
    """Is this occurrence the row CLAIMING to be that person, or MENTIONING them?

    **THE DEFECT IS "THIS ROW IS THE WRONG PERSON", NOT "THIS ROW SAID A NAME",
    and Bug Bible 11.61 is explicit that conflating the two is a mistake:** a
    sibling-name check *"flags correct relational prose ('foil to the Time
    Traveler', 'Rosalind's confidante'), which is legitimate and desirable"*.

    Live proof from the archive: ``ELLIE TERWILLIGER`` reads *"40s, foil to
    Hiram's meticulous obsession."* That describes ELLIE -- her relationship to
    somebody else -- and is good writing. A context-free match discards it and
    replaces a healthy row with a template, making the output WORSE than doing
    nothing. That is the one failure mode a guard must never have.

    Two signals separate a claim from a mention, and both are structural:

    * a POSSESSIVE (``Hiram's``) is always about somebody else's attribute;
    * a RELATIONAL CONNECTOR immediately before the name (``foil to``,
      ``daughter of``, ``rival to``) announces a relationship, not an identity.

    Everything else counts as a claim, which keeps the measured true positives:
    ``"30s, Henry 'Hank' Griswold."`` and ``"40s, Lizzie Gray - The Timekeeper"``
    are bare identity slots with no connector and no possessive.
    """
    tail = text[match.end():match.end() + 3]
    if re.match(r"[" + re.escape(_APOSTROPHES) + r"]s(?![A-Za-z])", tail):
        return False
    lead = text[max(0, match.start() - 40):match.start()]
    if _RELATIONAL_LEAD.search(lead):
        return False
    return True


def find_leaked_labels(fields: Dict[str, str], labels: Iterable[str]) -> List[str]:
    """Neutral labels that leaked OUT of the prompt and INTO generated prose.

    Reconciliation puts ``CHARACTER A`` in front of the model, so the model can
    copy it -- exactly as it copies a real name into the same free-text slot.
    ``"30s, CHARACTER A."`` is not a wrong-person description, but it is
    obviously broken text that would be spoken, printed in the credits and
    painted into a portrait, so it must be caught by the same pass.

    Detection for the names cannot find this: it looks for the identities that
    were REMOVED, and the label is the thing that replaced them.
    """
    leaked: List[str] = []
    for label in labels or ():
        if not label:
            continue
        pattern = _boundary_pattern(label)
        for value in (fields or {}).values():
            if value and pattern.search(str(value)):
                leaked.append(label)
                break
    return leaked


def find_foreign_identities(
    fields: Dict[str, str],
    identities: Sequence[str],
    roster_names: Sequence[str],
) -> List[ForeignIdentity]:
    """REPORT superseded identities present in generated fields. Never rewrites.

    ``fields`` maps a field name to its generated text -- pass every model-owned
    prose field, not just the headline one. ``speech_signature`` is demonstrably
    contaminated in the archive alongside ``character_description``, and a guard
    that checks one field certifies half a row.
    """
    found: List[ForeignIdentity] = []
    if not identities:
        return found
    # THE SAME resolved plan reconciliation uses. If detection admitted surfaces
    # reconciliation does not remove, the guard would fire on text the boundary
    # was never going to clean -- and the archive sweep would measure a contract
    # nobody ships.
    # DEFENCE IN DEPTH: drop anything the roster owns before searching. The
    # primary protection is that callers pass the output of
    # `superseded_identities`, but a caller who forgets would otherwise get the
    # row flagged for its OWN name -- and a detector that fires on the correct
    # case is worse than none, because it teaches everyone to ignore it.
    identities = [i for i in identities if not roster_owns(i, roster_names)]
    if not identities:
        return found
    plan, _labels = build_alias_plan(identities, roster_names)
    by_identity: Dict[str, List[tuple]] = {}
    for surface, label, identity in plan:
        by_identity.setdefault(identity, []).append((surface, label))

    for identity in identities:
        for field_name, value in (fields or {}).items():
            text = str(value or "")
            if not text:
                continue
            for surface_form, label in by_identity.get(identity, ()):
                if len(surface_form) < _MIN_TOKEN_LEN:
                    continue
                # EVERY occurrence, not just the first. `search` returns only
                # the earliest match, so a legitimate MENTION appearing before a
                # real CLAIM would shadow it and the row would ship:
                # "foil to Jonas's obsession. But I am Jonas!" was missed
                # entirely, because the possessive mention satisfied the check
                # and the claim after it was never examined.
                if any(_claims_the_identity(text, m)
                       for m in _boundary_pattern(surface_form).finditer(text)):
                    found.append(ForeignIdentity(
                        identity=identity, matched=surface_form,
                        field=str(field_name), label=label,
                    ))
                    break   # one finding per (identity, field); no double count
    return found


__all__ = [
    "ForeignIdentity",
    "build_alias_plan",
    "find_leaked_labels",
    "normalize_text",
    "name_tokens",
    "roster_owns",
    "superseded_identities",
    "identity_aliases",
    "default_label",
    "reconcile_text",
    "find_foreign_identities",
]
