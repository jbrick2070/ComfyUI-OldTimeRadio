"""nodes/_otr_story_quality_l12.py -- Story-Quality LIFT L1 + L2 (2026-06-23).

DETERMINISTIC, UPSTREAM beat-plan shaping. A dependency-free leaf (stdlib
only: hashlib / re / unicodedata) so it can be imported by both the outline
writer and the line composer without forming an import cycle. It operates on
duck-typed beat objects (anything exposing ``beat_id`` / ``speaker`` /
``speaker_role`` / ``intent`` / ``mood``) -- it never imports ``_otr_outline``.

THE PROBLEM (4-frontier-model + operator unanimous, 2026-06-23): every episode
collapses into the SAME "console standoff" -- whatever the premise, 2-3 people
fight over a lever/key/console while a gauge climbs and a countdown runs, and
the decisive moment happens OFF-stage (announcer fiat). The pipeline already
instructs against this in soft prose; the weak local writer ignores it. The
only effective fix is DETERMINISTIC + UPSTREAM (Python that builds + fills the
beat skeleton), NOT another QA reroll gate.

THE LEVERS
  L1  premise-anchored conflict: each voiced beat gets a Python-chosen,
      premise-specific ``conflict_object`` + ``conflict_type`` from a domain
      palette (seed-keyed), and the GENERIC crisis nouns the weak model
      defaults to are deterministically substituted (whole-token, in
      ``beat.intent`` ONLY) with the beat's grounded conflict object.
  L2  phase = dramatic FUNCTION: each voiced CHARACTER beat gets a
      ``beat_role`` from a real sequence (setup -> pressure -> personal_stake
      -> irreversible_choice-on-stage-as-the-last-beat -> consequence), filled
      deterministically with a fallback when the model under-delivers; a new
      validator enforces the sequence LAST.

DATA PLACEMENT: ``Beat`` is strict Pydantic with NO ``meta`` field -- adding
defaulted Beat fields would leak into ``model_dump()`` and drift the JSON. So
the per-beat SQ data is held in a WRITER-SIDE ``dict[beat_id -> sq]`` (built
here, consumed when constructing ``LineRequest``). The only mutation to a Beat
is to ``beat.intent`` (crisis-noun substitution + minimal fallback enrichment).

FLAG OFF (the default): ``build_sq_data`` is never called, the sq dict is
empty, the LineRequest SQ fields stay "", and every render is byte-identical.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# beat_role -- the dramatic FUNCTION sequence (L2)
# ---------------------------------------------------------------------------
# The CLIMAX lands on the last voiced character beat (on-stage, never narrated
# after the fact). Its TYPE is a CLIMAX-CLASS role (2026-06-24, operator
# directive): `irreversible_choice` is no longer the universal climax -- it is
# ONE climax class among the ending taxonomy. The structural invariant is
# "exactly one climax-class beat, and it is LAST"; WHICH climax class is
# style-selected (from `_otr_style_catalog` ending_tags) when the style-grammar
# lever is on, and defaults to `irreversible_choice` when off (byte-identical).
BEAT_ROLE_SETUP = "setup"
BEAT_ROLE_PRESSURE = "pressure"
BEAT_ROLE_PERSONAL_STAKE = "personal_stake"
BEAT_ROLE_CONSEQUENCE = "consequence"

# Climax-class roles (the ending taxonomy). irreversible_choice is the default;
# the other 8 are the non-doomsday endings the style grammar selects from.
BEAT_ROLE_IRREVERSIBLE_CHOICE = "irreversible_choice"
BEAT_ROLE_REVELATION = "revelation"
BEAT_ROLE_REVERSAL = "reversal"
BEAT_ROLE_UNRESOLVED_FINAL_SOUND = "unresolved_final_sound"
BEAT_ROLE_RECONCILIATION = "reconciliation"
BEAT_ROLE_BITTERSWEET_PARTING = "bittersweet_parting"
BEAT_ROLE_IRONIC_TWIST = "ironic_twist"
BEAT_ROLE_QUIET_ACCEPTANCE = "quiet_acceptance"
BEAT_ROLE_CONFESSION = "confession"

CLIMAX_CLASS_ROLES = frozenset({
    BEAT_ROLE_IRREVERSIBLE_CHOICE,
    BEAT_ROLE_REVELATION,
    BEAT_ROLE_REVERSAL,
    BEAT_ROLE_UNRESOLVED_FINAL_SOUND,
    BEAT_ROLE_RECONCILIATION,
    BEAT_ROLE_BITTERSWEET_PARTING,
    BEAT_ROLE_IRONIC_TWIST,
    BEAT_ROLE_QUIET_ACCEPTANCE,
    BEAT_ROLE_CONFESSION,
})

BEAT_ROLES = (
    BEAT_ROLE_SETUP,
    BEAT_ROLE_PRESSURE,
    BEAT_ROLE_PERSONAL_STAKE,
    BEAT_ROLE_IRREVERSIBLE_CHOICE,
    BEAT_ROLE_CONSEQUENCE,
    BEAT_ROLE_REVELATION,
    BEAT_ROLE_REVERSAL,
    BEAT_ROLE_UNRESOLVED_FINAL_SOUND,
    BEAT_ROLE_RECONCILIATION,
    BEAT_ROLE_BITTERSWEET_PARTING,
    BEAT_ROLE_IRONIC_TWIST,
    BEAT_ROLE_QUIET_ACCEPTANCE,
    BEAT_ROLE_CONFESSION,
)

# ---------------------------------------------------------------------------
# GENERIC crisis nouns -- the weak-model "console standoff" vocabulary (L1)
# ---------------------------------------------------------------------------
# Whole-token matches in ``beat.intent`` that are NOT grounded in the premise
# roster get substituted with the beat's premise-anchored conflict object. Kept
# casefolded; matched as whole word tokens only.
GENERIC_CRISIS_NOUNS = frozenset({
    "override", "overrides", "purge", "lever", "levers", "console", "consoles",
    "protocol", "protocols", "countdown", "self-destruct", "selfdestruct",
    "reactor", "core", "sequence", "lockdown", "terminal", "terminals",
    "mainframe", "switch", "switches", "button", "buttons", "keycard",
    "keycards", "failsafe", "fail-safe", "killswitch", "kill-switch",
    "detonator", "detonators", "deadman", "override-key", "control-panel",
    "panel", "panels", "gauge", "gauges", "meltdown", "containment",
})

# ---------------------------------------------------------------------------
# Domain palette (L1) -- premise -> {conflict_objects, conflict_types}
# ---------------------------------------------------------------------------
# Each domain offers CONCRETE, premise-appropriate contested things (objects)
# and the NATURE of the fight (types). "general" is a generic institutional-
# power palette so no premise is ever unserved. UTF-8, ASCII content.
DOMAIN_PALETTE: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "education": {
        "conflict_objects": (
            "the tutoring algorithm's grading weights",
            "a student's flagged transcript",
            "the district's adoption vote",
            "the classroom pilot's consent forms",
            "the assessment dataset",
        ),
        "conflict_types": (
            "a fight over who is accountable when the system is wrong",
            "a clash between measured results and lived experience",
            "a struggle to be believed by the people in charge",
        ),
    },
    "paleontology": {
        "conflict_objects": (
            "the disputed fossil's provenance",
            "the dig site's permit",
            "the specimen's catalog number",
            "the field notes",
            "the museum's loan agreement",
        ),
        "conflict_types": (
            "a fight over credit and the record",
            "a clash between haste and care for the evidence",
            "a struggle over who owns the find",
        ),
    },
    "energy": {
        "conflict_objects": (
            "the coal-seam variance",
            "the plant's emissions permit",
            "the rate-hike filing",
            "the shutdown timeline",
            "the workers' transition fund",
        ),
        "conflict_types": (
            "a fight over jobs versus the law",
            "a clash between the deadline and what is fair",
            "a struggle over who pays for the change",
        ),
    },
    "astronomy": {
        "conflict_objects": (
            "the telescope's observing time",
            "the disputed signal's data",
            "the survey's release embargo",
            "the array's calibration log",
            "the funding renewal",
        ),
        "conflict_types": (
            "a fight over whose interpretation gets published",
            "a clash between certainty and the data",
            "a struggle for the credit of a discovery",
        ),
    },
    "medicine": {
        "conflict_objects": (
            "the trial's enrollment list",
            "a patient's records",
            "the recall decision",
            "the dosage protocol",
            "the consent paperwork",
        ),
        "conflict_types": (
            "a fight over who is harmed by the delay",
            "a clash between the rules and one life",
            "a struggle to be heard by the board",
        ),
    },
    "agriculture": {
        "conflict_objects": (
            "the seed-patent claim",
            "the water-allocation order",
            "the harvest contract",
            "the soil-test results",
            "the cooperative's vote",
        ),
        "conflict_types": (
            "a fight over a season's livelihood",
            "a clash between the market and the land",
            "a struggle over who controls the supply",
        ),
    },
    "law": {
        "conflict_objects": (
            "the sealed deposition",
            "the disputed clause",
            "the settlement offer",
            "the evidence log",
            "the appeal deadline",
        ),
        "conflict_types": (
            "a fight over what the record will say",
            "a clash between the letter and the spirit of the rule",
            "a struggle to keep a promise under pressure",
        ),
    },
    "finance": {
        "conflict_objects": (
            "the audit's flagged ledger",
            "the merger's disclosure",
            "the pension's allocation",
            "the loan covenant",
            "the trading log",
        ),
        "conflict_types": (
            "a fight over whose money is at risk",
            "a clash between the quarter and the truth",
            "a struggle over who takes the blame",
        ),
    },
    "environment": {
        "conflict_objects": (
            "the wetland's survey",
            "the cleanup's funding",
            "the species listing",
            "the discharge permit",
            "the monitoring data",
        ),
        "conflict_types": (
            "a fight over the cost of doing it right",
            "a clash between the deadline and the evidence",
            "a struggle over who lives with the result",
        ),
    },
    "labor": {
        "conflict_objects": (
            "the grievance filing",
            "the schedule change",
            "the safety report",
            "the contract vote",
            "the layoff list",
        ),
        "conflict_types": (
            "a fight over whose work counts",
            "a clash between the floor and the office",
            "a struggle to be treated fairly",
        ),
    },
    "space": {
        "conflict_objects": (
            "the mission's go/no-go call",
            "the disputed telemetry",
            "the crew-rotation order",
            "the abort criteria",
            "the comms log",
        ),
        "conflict_types": (
            "a fight over an acceptable risk",
            "a clash between the schedule and the crew",
            "a struggle over who makes the call",
        ),
    },
    "communications": {
        "conflict_objects": (
            "the broadcast's retraction",
            "the leaked transcript",
            "the source's identity",
            "the airtime decision",
            "the correction notice",
        ),
        "conflict_types": (
            "a fight over what the public is told",
            "a clash between the story and the source",
            "a struggle to keep a confidence",
        ),
    },
    "general": {
        "conflict_objects": (
            "the contested decision on the table",
            "the disputed record",
            "the authorization everyone needs",
            "the one signature that settles it",
            "the deadline that cannot move",
        ),
        "conflict_types": (
            "a fight over who gets to decide",
            "a clash between the rule and the right thing",
            "a struggle to be believed in time",
        ),
    },
}

# Ordered keyword -> domain map (first match wins). Keys are casefolded.
_DOMAIN_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("classroom", "education"), ("teacher", "education"), ("student", "education"),
    ("school", "education"), ("tutor", "education"), ("curriculum", "education"),
    ("fossil", "paleontology"), ("dinosaur", "paleontology"),
    ("dig site", "paleontology"), ("paleontolog", "paleontology"),
    ("specimen", "paleontology"), ("excavation", "paleontology"),
    ("coal", "energy"), ("power plant", "energy"), ("emission", "energy"),
    ("grid", "energy"), ("reactor", "energy"), ("pipeline", "energy"),
    ("utility", "energy"),
    ("telescope", "astronomy"), ("astronom", "astronomy"),
    ("observatory", "astronomy"), ("galaxy", "astronomy"),
    ("comet", "astronomy"), ("supernova", "astronomy"),
    ("patient", "medicine"), ("hospital", "medicine"), ("clinical", "medicine"),
    ("doctor", "medicine"), ("vaccine", "medicine"), ("trial", "medicine"),
    ("disease", "medicine"), ("drug", "medicine"),
    ("farm", "agriculture"), ("crop", "agriculture"), ("harvest", "agriculture"),
    ("seed", "agriculture"), ("irrigation", "agriculture"),
    ("court", "law"), ("trial", "law"), ("lawyer", "law"), ("judge", "law"),
    ("statute", "law"), ("verdict", "law"), ("deposition", "law"),
    ("bank", "finance"), ("audit", "finance"), ("merger", "finance"),
    ("pension", "finance"), ("market", "finance"), ("invest", "finance"),
    ("wetland", "environment"), ("pollution", "environment"),
    ("climate", "environment"), ("species", "environment"),
    ("cleanup", "environment"), ("habitat", "environment"),
    ("union", "labor"), ("strike", "labor"), ("worker", "labor"),
    ("grievance", "labor"), ("layoff", "labor"), ("wage", "labor"),
    ("mission", "space"), ("astronaut", "space"), ("orbit", "space"),
    ("launch", "space"), ("spacecraft", "space"), ("rover", "space"),
    ("broadcast", "communications"), ("newsroom", "communications"),
    ("journalist", "communications"), ("transmission", "communications"),
    ("airtime", "communications"), ("reporter", "communications"),
)

# Stopwords for the premise noun-token extractor (the "grounded" palette).
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "under", "their", "they", "them", "what", "when", "where", "which",
    "while", "would", "could", "should", "about", "after", "before",
    "between", "against", "your", "yours", "ours", "have", "has", "had",
    "will", "shall", "must", "been", "were", "are", "was", "but", "not",
    "all", "any", "one", "two", "out", "off", "its", "his", "her",
    "she", "him", "who", "how", "why", "now", "then", "than", "too",
    "very", "just", "more", "most", "some", "such", "only", "own",
    "same", "each", "both", "few", "nor", "yet", "via", "per", "amid",
})


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
def _norm(s: Any) -> str:
    """NFC-normalize + casefold a value to a stable comparison string."""
    return unicodedata.normalize("NFC", str(s or "")).casefold()


def _seeded_pick(candidates: Tuple[str, ...], key: str) -> str:
    """Deterministic modulo selection from SORTED candidates (never hash())."""
    if not candidates:
        return ""
    ordered = sorted(candidates)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return ordered[int(digest, 16) % len(ordered)]


# ---------------------------------------------------------------------------
# L1 -- domain selection + conflict slots
# ---------------------------------------------------------------------------
def select_domain(meta: Any, premise: str) -> str:
    """Pick a palette domain from the premise + meta, ordered keyword map.

    Inspects, in order: the explicit premise string, then a handful of meta
    news fields (title / logline / script_brief / theme). First keyword hit
    wins; default ``"general"`` so no premise is unserved. Pure + deterministic
    (no LLM, no RNG)."""
    haystack_parts: List[str] = [premise or ""]
    if isinstance(meta, Mapping):
        news = meta.get("news") if isinstance(meta.get("news"), Mapping) else {}
        for fld in ("title", "logline", "script_brief", "theme", "headline"):
            for src in (meta, news):
                if isinstance(src, Mapping) and src.get(fld):
                    haystack_parts.append(str(src.get(fld)))
    hay = _norm(" \n ".join(haystack_parts))
    for kw, domain in _DOMAIN_KEYWORDS:
        if kw in hay:
            return domain
    return "general"


def conflict_palette(domain: str) -> Dict[str, Tuple[str, ...]]:
    """Return the {conflict_objects, conflict_types} palette for a domain."""
    return DOMAIN_PALETTE.get(domain) or DOMAIN_PALETTE["general"]


def assign_conflict_slot(domain: str, beat_index: int, seed: Any) -> Dict[str, str]:
    """Seed-keyed conflict_object + conflict_type for one beat (distinct keys)."""
    palette = conflict_palette(domain)
    obj = _seeded_pick(
        palette["conflict_objects"],
        f"{seed}:{beat_index}:{domain}:object",
    )
    typ = _seeded_pick(
        palette["conflict_types"],
        f"{seed}:{beat_index}:{domain}:type",
    )
    return {"conflict_object": obj, "conflict_type": typ}


# ---------------------------------------------------------------------------
# L1 -- crisis-noun grounding (mutates beat.intent ONLY)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def premise_noun_palette(roster: Any, *texts: str) -> frozenset:
    """Grounded token palette = roster names + premise/title/logline nouns.

    Tokens are ``[A-Za-z][A-Za-z'-]{2,}`` minus stopwords and ALL-CAPS tokens
    (those are speaker/announcer labels, not premise nouns). Casefolded."""
    allowed: set = set()
    try:
        for nm in (roster or ()):
            for tok in _TOKEN_RE.findall(str(nm or "")):
                allowed.add(tok.casefold())
    except TypeError:
        pass
    for txt in texts:
        for tok in _TOKEN_RE.findall(str(txt or "")):
            if tok.isupper():
                continue
            low = tok.casefold()
            if low in _STOPWORDS:
                continue
            allowed.add(low)
    return frozenset(allowed)


def ungrounded_crisis_tokens(intent: str, grounded: frozenset) -> List[str]:
    """Return the whole-token GENERIC crisis nouns in ``intent`` that are NOT in
    the grounded premise palette -- casefolded, in order of appearance (the
    sameness signal, itemised so a reroll hint can name the offenders). Pure +
    deterministic."""
    out: List[str] = []
    for tok in _TOKEN_RE.findall(str(intent or "")):
        low = tok.casefold()
        if low in GENERIC_CRISIS_NOUNS and low not in grounded:
            out.append(low)
    return out


def count_ungrounded_crisis(intent: str, grounded: frozenset) -> int:
    """Count whole-token GENERIC crisis nouns in ``intent`` that are NOT in the
    grounded premise palette (the sameness signal)."""
    return len(ungrounded_crisis_tokens(intent, grounded))


def ground_crisis_nouns(
    intent: str, conflict_object: str, grounded: frozenset,
) -> Tuple[str, int]:
    """Substitute every ungrounded GENERIC crisis noun with ``conflict_object``.

    Whole-token, case-insensitive; first replacement uses the full object, any
    further ones use a short back-reference ("it") so the sentence stays
    readable. Returns ``(new_intent, n_substituted)``. Deterministic."""
    if not conflict_object:
        return intent, 0
    n = [0]

    def _sub(m: "re.Match") -> str:
        tok = m.group(0)
        low = tok.casefold()
        if low in GENERIC_CRISIS_NOUNS and low not in grounded:
            n[0] += 1
            return conflict_object if n[0] == 1 else "it"
        return tok

    new_intent = _TOKEN_RE.sub(_sub, str(intent or ""))
    return new_intent, n[0]


# ---------------------------------------------------------------------------
# KILL 1 (2026-06-24 assumption-audit) -- the deterministic BODY-OUTPUT gate.
# The L1 grounding above only edits ``beat.intent``; the SHIPPED dialogue was
# ungated, so the weak local writer still collapsed every premise into the same
# "console standoff" (proven live: "press this red lever", "blowing the fuel
# cells"). These validators run on the COMPOSED line text in the writer loop:
# it must not lean on ungrounded generic crisis machinery, and a climax-class /
# pressure beat must REFERENCE its premise-anchored ``conflict_object``. Pure
# leaf; never raises.
# ---------------------------------------------------------------------------
def _strip_possessive(tok: str) -> str:
    """Casefold ``tok`` and drop a trailing possessive (``Ward's`` -> ``ward``).

    ``_TOKEN_RE`` only admits the ASCII apostrophe, so the curly U+2019 variant
    never reaches a token (the regex stops at it, yielding the bare stem)."""
    t = tok.casefold()
    for suf in ("'s", "'"):
        if t.endswith(suf):
            return t[: -len(suf)]
    return t


def _content_tokens(text: str) -> frozenset:
    """Casefolded, possessive-stripped content tokens of ``text`` minus stopwords."""
    out: set = set()
    for tok in _TOKEN_RE.findall(str(text or "")):
        low = _strip_possessive(tok)
        if low and low not in _STOPWORDS:
            out.add(low)
    return frozenset(out)


def line_references_object(text: str, conflict_object: str) -> bool:
    """True iff ``text`` overlaps the conflict object on any content token
    (head-noun / ``_TOKEN_RE`` token overlap, casefold, possessive-stripped). An
    empty object is vacuously satisfied (nothing to require)."""
    obj_tokens = _content_tokens(conflict_object)
    if not obj_tokens:
        return True
    return bool(obj_tokens & _content_tokens(text))


def validate_composed_grounding(
    text: str,
    sq_entry: Optional[Mapping[str, Any]],
    grounded: frozenset,
    *,
    max_ungrounded: int = 0,
    require_conflict_object_on_roles: frozenset = frozenset(),
) -> Tuple[bool, List[str]]:
    """Validate a SHIPPED character line against the deterministic grounding.

    Two checks on the composed dialogue ``text`` (NOT ``beat.intent``):
      1. ungrounded_crisis -- generic crisis nouns not in the grounded premise
         palette (reuses ``ungrounded_crisis_tokens`` /
         ``count_ungrounded_crisis``); fails when the count exceeds
         ``max_ungrounded``.
      2. missing_conflict_object -- when this beat's ``beat_role`` is in
         ``require_conflict_object_on_roles`` AND it carries a
         ``conflict_object``, the line must reference that object (token
         overlap).

    Returns ``(ok, reasons)`` where each reason is a machine hint the writer
    SPLITS into a targeted reroll instruction:
      ``"ungrounded_crisis:<tok>,<tok>"`` -- the offending tokens (de-duped), and
      ``"missing_conflict_object:<object>"`` -- the grounded object to inject.
    Pure + deterministic; never raises."""
    reasons: List[str] = []
    toks = ungrounded_crisis_tokens(text, grounded)
    if len(toks) > max_ungrounded:
        seen: List[str] = []
        for t in toks:
            if t not in seen:
                seen.append(t)
        reasons.append("ungrounded_crisis:" + ",".join(seen))
    entry = sq_entry if isinstance(sq_entry, Mapping) else {}
    role = str(entry.get("beat_role", "") or "")
    conflict_object = str(entry.get("conflict_object", "") or "")
    if (
        role in require_conflict_object_on_roles
        and conflict_object
        and not line_references_object(text, conflict_object)
    ):
        reasons.append("missing_conflict_object:" + conflict_object)
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# L2 -- beat_role assignment + validator
# ---------------------------------------------------------------------------
def assign_beat_roles(
    ordered_char_beat_ids: List[str],
    *,
    climax_role: str = BEAT_ROLE_IRREVERSIBLE_CHOICE,
) -> Dict[str, str]:
    """Assign a dramatic-function role to each voiced CHARACTER beat, in order.

    Sequence guarantees the CLIMAX is the LAST voiced character beat (on-stage)
    and exactly one ``personal_stake`` precedes it:
      n==1: [<climax>]
      n==2: [setup, <climax>]   (no room for personal_stake)
      n>=3: [setup, pressure*, personal_stake, <climax>]
    The climax's TYPE is ``climax_role`` -- one of CLIMAX_CLASS_ROLES; defaults to
    ``irreversible_choice`` (so an unkeyed call is byte-identical to the pre-2026-
    06-24 behavior). An invalid climax_role falls back to irreversible_choice
    (fail-soft, never raises). Pure + deterministic."""
    if climax_role not in CLIMAX_CLASS_ROLES:
        climax_role = BEAT_ROLE_IRREVERSIBLE_CHOICE
    n = len(ordered_char_beat_ids)
    roles: Dict[str, str] = {}
    if n == 0:
        return roles
    if n == 1:
        roles[ordered_char_beat_ids[0]] = climax_role
        return roles
    if n == 2:
        roles[ordered_char_beat_ids[0]] = BEAT_ROLE_SETUP
        roles[ordered_char_beat_ids[1]] = climax_role
        return roles
    for i, bid in enumerate(ordered_char_beat_ids):
        if i == 0:
            roles[bid] = BEAT_ROLE_SETUP
        elif i == n - 1:
            roles[bid] = climax_role
        elif i == n - 2:
            roles[bid] = BEAT_ROLE_PERSONAL_STAKE
        else:
            roles[bid] = BEAT_ROLE_PRESSURE
    return roles


class BeatRoleViolation(ValueError):
    """Raised when the deterministic beat_role sequence is internally
    inconsistent (a self-check on our own assignment; should never fire in
    production -- a fire means a bug in assign_beat_roles)."""


def validate_beat_roles(
    roles_by_beat: Mapping[str, str], ordered_char_beat_ids: List[str],
) -> None:
    """Enforce the dramatic-function contract LAST, first-failure style.

    Contract (n = number of voiced character beats):
      * every role is a known BEAT_ROLE;
      * exactly one CLIMAX-CLASS role (CLIMAX_CLASS_ROLES -- irreversible_choice
        OR one of the 8 ending archetypes) and it is the LAST voiced beat;
      * for n >= 3, exactly one ``personal_stake`` and it precedes the climax
        (tiny n<3 episodes have no room and are exempt).
    Raises BeatRoleViolation on the FIRST breach."""
    n = len(ordered_char_beat_ids)
    if n == 0:
        return
    for bid in ordered_char_beat_ids:
        role = roles_by_beat.get(bid, "")
        if role not in BEAT_ROLES:
            raise BeatRoleViolation(
                f"beat {bid} has beat_role={role!r}; not in {BEAT_ROLES!r}"
            )
    climax_indices = [
        i for i, bid in enumerate(ordered_char_beat_ids)
        if roles_by_beat.get(bid) in CLIMAX_CLASS_ROLES
    ]
    if len(climax_indices) != 1:
        raise BeatRoleViolation(
            f"expected exactly one climax-class beat, found {len(climax_indices)}"
        )
    if climax_indices[0] != n - 1:
        raise BeatRoleViolation(
            "the climax-class beat must be the LAST voiced beat "
            f"(found at index {climax_indices[0]} of {n})"
        )
    if n >= 3:
        ps_indices = [
            i for i, bid in enumerate(ordered_char_beat_ids)
            if roles_by_beat.get(bid) == BEAT_ROLE_PERSONAL_STAKE
        ]
        if len(ps_indices) != 1:
            raise BeatRoleViolation(
                f"expected exactly one personal_stake beat, found {len(ps_indices)}"
            )
        if ps_indices[0] >= climax_indices[0]:
            raise BeatRoleViolation(
                "the personal_stake beat must precede the climax-class beat"
            )


# ---------------------------------------------------------------------------
# L2 -- deterministic fallback content (personal_cost / sensory / state_change)
# ---------------------------------------------------------------------------
# No structured cost/fear field exists upstream, so a deterministic
# (role, domain) table is the PRIMARY source. Concrete, SFW, ASCII.
_PERSONAL_COST: Dict[str, Tuple[str, ...]] = {
    "general": (
        "what it costs them to be the one who decides",
        "the trust they will lose either way",
        "the part of themselves they have to set down to do this",
    ),
}

_SENSORY_CONSEQUENCE: Tuple[str, ...] = (
    "the room goes quiet enough to hear the decision land",
    "a hand stops halfway, then finishes the motion",
    "the thing changes shape in front of everyone present",
)

_STATE_CHANGE: Tuple[str, ...] = (
    "and nothing between them can go back to how it was",
    "and the choice is now on the record, in the open",
    "and the person they were before this is gone",
)


def fallback_content(role: str, domain: str, seed: Any, beat_index: int) -> Dict[str, str]:
    """Seed-keyed fallback personal_cost / sensory_consequence / state_change."""
    cost_pool = _PERSONAL_COST.get(domain) or _PERSONAL_COST["general"]
    return {
        "personal_cost": _seeded_pick(cost_pool, f"{seed}:{beat_index}:{domain}:cost"),
        "sensory_consequence": _seeded_pick(
            _SENSORY_CONSEQUENCE, f"{seed}:{beat_index}:sense"
        ),
        "state_change": _seeded_pick(
            _STATE_CHANGE, f"{seed}:{beat_index}:state"
        ),
    }


# ---------------------------------------------------------------------------
# Public entrypoint -- build the writer-side sq dict (called ONLY when flag on)
# ---------------------------------------------------------------------------
_INTENT_MAX = 200   # mirrors Beat.intent max_length


def _is_voiced(role: str) -> bool:
    return role in ("character", "announcer")


def build_sq_data(
    beats: List[Any],
    meta: Any,
    premise: str,
    seed: Any,
    roster: Any = (),
    *,
    climax_role: str = BEAT_ROLE_IRREVERSIBLE_CHOICE,
) -> Dict[str, Dict[str, Any]]:
    """Build the writer-side ``dict[beat_id -> sq]`` and ground beat intents.

    For each voiced beat: assign a seed-keyed premise-anchored
    ``conflict_object`` + ``conflict_type``; ground the GENERIC crisis nouns in
    ``beat.intent`` (mutates intent ONLY). For each voiced CHARACTER beat:
    assign a ``beat_role`` (the dramatic-function sequence) + deterministic
    fallback ``personal_cost`` / ``sensory_consequence`` / ``state_change``, and
    minimally enrich a thin intent so the climax lands ON-stage. Runs the
    ``beat_role`` validator LAST.

    Returns the sq dict. The caller threads ``beat_role`` / ``conflict_object``
    / ``conflict_type`` into ``LineRequest``; the rest reach the composer
    through the (possibly enriched) ``beat.intent``. Pure aside from the
    in-place ``beat.intent`` mutation; fully deterministic for a fixed seed."""
    domain = select_domain(meta, premise)
    grounded = premise_noun_palette(
        roster,
        premise,
        *premise_texts(meta),
    )

    ordered_char_ids: List[str] = [
        str(getattr(b, "beat_id", ""))
        for b in beats
        if _is_voiced(str(getattr(b, "speaker_role", "") or ""))
        and str(getattr(b, "speaker_role", "")) == "character"
        and getattr(b, "beat_id", "")
    ]
    # The climax's TYPE is style-selected (climax_role) when the style-grammar
    # lever is on, and defaults to irreversible_choice (so an unkeyed call is
    # byte-identical to the pre-2026-06-24 behavior). assign_beat_roles fails
    # soft on an unknown role -> irreversible_choice, so this never raises.
    roles_by_beat = assign_beat_roles(ordered_char_ids, climax_role=climax_role)
    validate_beat_roles(roles_by_beat, ordered_char_ids)

    sq: Dict[str, Dict[str, Any]] = {}
    for idx, b in enumerate(beats):
        role = str(getattr(b, "speaker_role", "") or "")
        if not _is_voiced(role):
            continue
        bid = str(getattr(b, "beat_id", ""))
        if not bid:
            continue
        slot = assign_conflict_slot(domain, idx, seed)
        entry: Dict[str, Any] = {
            "conflict_object": slot["conflict_object"],
            "conflict_type": slot["conflict_type"],
            "beat_role": roles_by_beat.get(bid, ""),
            "personal_cost": "",
            "sensory_consequence": "",
            "state_change": "",
        }

        # L1: ground the crisis nouns in this beat's intent (intent ONLY).
        intent = str(getattr(b, "intent", "") or "")
        new_intent, n_sub = ground_crisis_nouns(
            intent, slot["conflict_object"], grounded,
        )

        # L2: fallback content for the dramatic-function beats + a minimal,
        # APPEND-only intent enrichment when the model under-delivered (keeps a
        # good intent intact; guarantees the climax is concrete + on-stage).
        beat_role = entry["beat_role"]
        grounded_intent = new_intent
        enrichment = ""
        if beat_role in _ENRICH_ROLES:
            fc = fallback_content(beat_role, domain, seed, idx)
            entry.update(fc)
            enrichment = _enrich_tail(grounded_intent, beat_role, slot, fc)

        # KILL 4 (2026-06-24): reserve room for the enrichment so a long intent
        # cannot truncate the concrete climax clause off the end (the old
        # [:_INTENT_MAX] cut the tail). max(0, ...) guards the negative slice.
        if enrichment:
            base = grounded_intent.strip()
            tail = enrichment.strip()
            if not base:
                new_intent = tail[:_INTENT_MAX]
            else:
                sep = "; "
                reserve = _INTENT_MAX - len(sep) - len(tail)
                if reserve <= 0:
                    new_intent = tail[:_INTENT_MAX]
                else:
                    new_intent = (
                        base[:max(0, reserve)].strip() + sep + tail
                    )[:_INTENT_MAX]
        else:
            new_intent = grounded_intent.strip()[:_INTENT_MAX].strip()
        if new_intent and new_intent != intent:
            try:
                setattr(b, "intent", new_intent)
            except Exception:  # noqa: BLE001 -- never break the writer
                pass
        sq[bid] = entry
    return sq


def premise_texts(meta: Any) -> Tuple[str, ...]:
    out: List[str] = []
    if isinstance(meta, Mapping):
        news = meta.get("news") if isinstance(meta.get("news"), Mapping) else {}
        for fld in ("title", "logline", "script_brief", "theme", "headline"):
            for src in (meta, news):
                if isinstance(src, Mapping) and src.get(fld):
                    out.append(str(src.get(fld)))
    return tuple(out)


# KILL 4 (2026-06-24): role-keyed enrichment. The dramatic-function beats that
# earn an APPEND-only concrete clause -- setup / pressure / personal_stake + every
# climax class. CONSEQUENCE is omitted (unreachable under climax-last today;
# revisit with KILL 3 -- a deliberate omission, not a stub).
_ENRICH_ROLES = frozenset(
    {BEAT_ROLE_SETUP, BEAT_ROLE_PRESSURE, BEAT_ROLE_PERSONAL_STAKE}
) | CLIMAX_CLASS_ROLES

# Per-role tail templates ({obj} = conflict_object). 3.6 (story-quality R2): the
# {cost} clause (personal_cost) was DROPPED from all five tails that carried it
# -- it homogenized nearly every beat ("the trust they will lose either way" rode
# almost everything and leaked into a spoken line). W-E (judge): DROP, do not
# replace with a generic cost phrase (that re-homogenizes). Each tail now anchors
# ONLY to the concrete {obj}; personal_cost stays in the SQ dict for telemetry but
# is never formatted into a tail.
_ENRICH_TAILS: Dict[str, str] = {
    BEAT_ROLE_SETUP:
        "the scene is set: {obj} is already in play before the pressure builds",
    BEAT_ROLE_PRESSURE:
        "the pressure tightens around {obj}",
    BEAT_ROLE_PERSONAL_STAKE:
        "what is personally at stake turns on {obj}",
    BEAT_ROLE_IRREVERSIBLE_CHOICE:
        "on-stage, the decision about {obj} is made now",
    BEAT_ROLE_REVELATION:
        "on-stage, the truth about {obj} comes to light now",
    BEAT_ROLE_REVERSAL:
        "on-stage, the situation around {obj} turns against what was expected",
    BEAT_ROLE_UNRESOLVED_FINAL_SOUND:
        "the question of {obj} is left hanging as the final sound fades",
    BEAT_ROLE_RECONCILIATION:
        "on-stage, they make their peace over {obj} now",
    BEAT_ROLE_BITTERSWEET_PARTING:
        "on-stage, they part over {obj} -- gaining and losing in one breath",
    BEAT_ROLE_IRONIC_TWIST:
        "the outcome of {obj} lands with an ironic turn",
    BEAT_ROLE_QUIET_ACCEPTANCE:
        "on-stage, they quietly accept what {obj} has come to",
    BEAT_ROLE_CONFESSION:
        "on-stage, the truth about {obj} is confessed aloud now",
}


def _enrich_tail(
    intent: str, beat_role: str, slot: Mapping[str, str], fc: Mapping[str, str],
) -> str:
    """The concrete deterministic clause for a dramatic-function beat, or "" when
    the intent is already substantive AND lands the grounded object (never
    overwrites a good intent) or the role earns no enrichment. Returns ONLY the
    tail; the caller joins it + reserves room through truncation (KILL 4)."""
    obj = slot.get("conflict_object", "")
    has_obj = bool(obj) and obj.casefold().split()[-1] in intent.casefold()
    thin = len(intent.split()) < 6
    if not thin and has_obj:
        return ""
    template = _ENRICH_TAILS.get(beat_role)
    if not template:
        return ""
    return template.format(obj=obj, cost=fc.get("personal_cost", ""))


def _enrich_intent(
    intent: str, beat_role: str, slot: Mapping[str, str], fc: Mapping[str, str],
) -> str:
    """Back-compat combined-string form: APPEND the role's tail to the intent (or
    return it unchanged when no enrichment is earned). build_sq_data now calls
    _enrich_tail directly so it can reserve room for the tail through truncation;
    this wrapper keeps the old contract for any direct caller. The two
    pre-KILL-4 roles (irreversible_choice / personal_stake) are byte-identical."""
    tail = _enrich_tail(intent, beat_role, slot, fc)
    if not tail:
        return intent
    sep = "" if not intent.strip() else "; "
    return f"{intent.strip()}{sep}{tail}"
