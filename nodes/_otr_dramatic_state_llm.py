"""nodes/_otr_dramatic_state_llm.py -- story-quality Phase 1, B1 (THE SPINE).

The #1 must-fix: the news -> story disconnect. The opposed wants in
DramaticState were hardcoded `_DEFAULT_A/B_WANTS` constants that IGNORE the
news brief, so the central conflict was generic and the news reached only
the dramatic question -- the leg_0013 "ancient-DNA brief -> a story about
aliens" drift. This module derives the opposed wants + dramatic question +
ending DIRECTLY from `meta["news"]`, at the WRITER CALL SITE (which has
`meta` + the resident technical `generate_fn`), via a structured LLM call on
the resident technical slot -- NO per-model branch, NO new model load.

Design (roundtable-converged):
  * The pure `_otr_dramatic_state` module stays PURE (no LLM); this module
    owns the LLM call + the deterministic fallback. We do NOT route the B1
    path through `derive_dramatic_state_from_meta` -- its optional-override
    path silently falls back to the `_DEFAULT_*` constants on any empty
    field, which is exactly the boilerplate we are replacing.
  * `structured_call` with a NEW Pydantic schema emitting ONLY the four
    DramaticState-compatible strings (the costly_choice_beat is derived
    deterministically, not by the LLM).
  * A POST-VALIDATOR (the about-the-news guard): Pydantic only checks
    lengths + non-identical wants, so a weak model can emit generic-but-
    valid strings. After the call we REQUIRE >= 1 news key term to appear
    across the wants / question / ending, and reject same-direction wants;
    on rejection `structured_call` advances its ladder, then -> fallback.
  * DETERMINISTIC FALLBACK (LLM reject / news failure): opposed wants built
    from `meta["news"]["key_terms"]` + PAIRED conflict templates opposed BY
    CONSTRUCTION. Never the `_DEFAULT_*` constant path.
  * TURNING-SLOT DETAIL FLOOR: `validate_contract` requires the must_turn
    slot's concrete_detail to be a subset of `active_props U key_terms`. So
    we GUARANTEE >= 1 entry in `meta["news"]["key_terms"]` (seeding it from
    the script brief's first noun phrase, else "the event") BEFORE contract
    derivation runs.

NEVER raises (Prime Directive 1, audio is king): any LLM / validation
failure degrades to the deterministic fallback. Content-only -- the ledger
wire format is untouched; only `meta` sub-keys change. UTF-8 no BOM.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from ._otr_dramatic_state import DramaticState, pick_costly_choice_slot

log = logging.getLogger("OTR_DramaticStateB1")

__all__ = [
    "DramaticStateLLM",
    "derive_news_dramatic_state",
    "ARC_SHAPES",
    "pick_arc_shape",
    "is_nonownable_story_object",
    "derive_safe_fallback_term",
]

# F8 (story-engine v1): arc-shape variety. A seeded pre-step picks one of these
# shapes per episode so every story is not the same setup->complication->
# resolution mold. CONFRONTATION shapes turn on two opposed wants; the others
# (investigation / slow_dread) are NOT a clash of wills, so the post-validator
# relaxes the opposed-wants requirement for them (prevents a generation stall).
ARC_SHAPES: Tuple[str, ...] = (
    "setup_complication_resolution",
    "investigation_without_answer",
    "slow_dread",
    "heist",
    "betrayal",
)
_CONFRONTATION_SHAPES = frozenset({
    "setup_complication_resolution", "heist", "betrayal",
})


def pick_arc_shape(seed: Any) -> str:
    """Deterministically pick an arc_shape from a reproducibility seed.

    Stable across processes (md5, not Python's salted hash) so a fixed
    (seed, news) pair always yields the same shape. Empty seed -> the first
    (default) shape. Never raises."""
    try:
        s = str(seed or "")
        if not s:
            return ARC_SHAPES[0]
        h = hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()
        return ARC_SHAPES[int(h, 16) % len(ARC_SHAPES)]
    except Exception:  # noqa: BLE001
        return ARC_SHAPES[0]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# A small stopword set for the script-brief noun-phrase seed (B1 fallback).
_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "of", "in", "on",
    "at", "to", "for", "and", "or", "but", "with", "as", "is", "are", "was",
    "were", "be", "been", "by", "from", "into", "about", "after", "before",
    "when", "while", "how", "what", "who", "why", "their", "his", "her",
    "its", "our", "your", "they", "it", "he", "she", "we", "you",
})


class DramaticStateLLM(BaseModel):
    """The LLM-authored subset of DramaticState (no costly_choice_beat).

    Constraints MIRROR DramaticState so a schema-valid instance always
    builds a valid DramaticState. The opposition + news-grounding checks
    live in the post-validator (the model only forbids zero-length fields).
    """

    dramatic_question: str = Field(..., min_length=10, max_length=240)
    character_a_wants: str = Field(..., min_length=4, max_length=120)
    character_b_wants: str = Field(..., min_length=4, max_length=120)
    ending_change: str = Field(..., min_length=4, max_length=200)


def _norm(s: Any) -> str:
    try:
        return _WS.sub(" ", _PUNCT.sub(" ", str(s or ""))).strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _trunc(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0].strip()
    return cut or s[:n].strip()


def _first_noun_phrase(script_brief: str) -> Optional[str]:
    """A best-effort short content phrase from the script brief, for seeding
    a key term when the news brief carried none. Deterministic."""
    words = _norm(script_brief).split()
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not content:
        return None
    return " ".join(content[:3])


def _key_terms(meta: Any) -> List[str]:
    try:
        news = meta.get("news") if isinstance(meta, dict) else None
        news = news if isinstance(news, dict) else {}
        return [str(t).strip() for t in (news.get("key_terms") or [])
                if str(t).strip()]
    except Exception:  # noqa: BLE001
        return []


def _script_brief(meta: Any) -> str:
    try:
        news = meta.get("news") if isinstance(meta, dict) else None
        news = news if isinstance(news, dict) else {}
        return str(news.get("script_brief") or "")
    except Exception:  # noqa: BLE001
        return ""


def _inject_key_term(meta: Any, term: str) -> None:
    """Append `term` to meta['news']['key_terms'] if missing, initializing the
    dict/list as needed. This is the turning-slot detail floor -- it lets
    validate_contract's must_turn branch find a concrete detail in the
    active_props U key_terms pool. Never raises."""
    if not term or not isinstance(meta, dict):
        return
    try:
        news = meta.get("news")
        if not isinstance(news, dict):
            news = {}
            meta["news"] = news
        kt = news.get("key_terms")
        if not isinstance(kt, list):
            kt = []
        if term not in kt:
            kt.append(term)
        news["key_terms"] = kt
    except Exception as exc:  # noqa: BLE001
        log.warning("[B1] key-term injection failed: %r", exc)


# Paired conflict templates -- opposed BY CONSTRUCTION. {t} = a news key term.
# (a_wants, b_wants, dramatic_question, ending_change)
_TEMPLATES: Tuple[Tuple[str, str, str, str], ...] = (
    ("see {t} fully revealed to everyone",
     "keep {t} hidden at any cost",
     "Will the truth about {t} come out before it is too late?",
     "{t} is finally brought into the open, and no one involved can go back."),
    ("act on {t} now, before the chance is lost",
     "hold back and verify {t} before anyone moves",
     "Should they act on {t} now, or wait until it is certain?",
     "They commit to a hard course on {t}, accepting the cost of being wrong."),
    ("take sole credit for {t}",
     "make sure {t} is shared openly and freely",
     "Who will control what becomes of {t}?",
     "Control of {t} passes to whoever is willing to pay the higher price."),
    ("use {t} to protect the people they love",
     "use {t} to expose the ones in power",
     "What matters more once {t} is in hand, safety or the truth?",
     "The choice over {t} redraws who can be trusted by the end."),
)


# F8 (story-engine v1): shape-appropriate template sets for the NON-
# confrontation arc shapes. The two wants are still DISTINCT (the DramaticState
# floor forbids identical strings) but they are not a head-to-head clash --
# investigation is seeker-vs-withheld-truth, slow_dread is hold-back-vs-the-
# threat. Confrontation shapes fall through to the opposed _TEMPLATES default.
_INVESTIGATION_TEMPLATES: Tuple[Tuple[str, str, str, str], ...] = (
    ("piece together what really happened with {t}",
     "keep the full story of {t} from ever surfacing",
     "What truly happened with {t}, and will anyone ever know?",
     "The inquiry into {t} closes with as many questions as it answered."),
    ("follow {t} wherever the evidence leads",
     "bury the loose ends around {t} before they are found",
     "Can the truth about {t} be reconstructed before the trail goes cold?",
     "They learn just enough about {t} to know how much stays unknown."),
)
_DREAD_TEMPLATES: Tuple[Tuple[str, str, str, str], ...] = (
    ("hold the threat of {t} at bay a little longer",
     "let {t} run its course, whatever it costs them",
     "How long can they keep {t} from overtaking everything?",
     "{t} closes in, and the quiet they knew is gone for good."),
    ("warn the others about {t} before it is too late",
     "stay silent about {t} and hope it passes",
     "Will anyone act on {t} before it reaches them?",
     "{t} arrives on its own schedule, indifferent to who was ready."),
)
_SHAPE_TEMPLATES = {
    "investigation_without_answer": _INVESTIGATION_TEMPLATES,
    "slow_dread": _DREAD_TEMPLATES,
}


# 3.1 dignity guard (story-quality R2): the fallback DramaticState must NEVER
# template ownership / credit / control over PEOPLE or a protected-identity /
# harm-population group (frostbite_facility shipped "take sole credit for
# transgender people" / "Control of transgender people passes to whoever is
# willing to pay the higher price."). The head-noun rule keeps object heads with
# a people MODIFIER ownable ("patient records", "children's health study",
# "survey data"); the protected-identity substring set also catches an ownable
# head riding a harm phrase ("suicide thoughts").
_PEOPLE_NOUNS = frozenset({
    "people", "person", "persons", "man", "men", "woman", "women",
    "child", "children", "kid", "kids", "resident", "residents",
    "patient", "patients", "worker", "workers", "citizen", "citizens",
    "population", "populations", "community", "communities", "group", "groups",
    "humanity", "victim", "victims", "family", "families",
    "individual", "individuals",
})
#: Protected-identity / harm-population substrings -- non-ownable even when the
#: grammatical head is an object noun. Casefolded substring match.
_PROTECTED_IDENTITY_SUBSTRINGS = ("transgender", "suicide")
#: Index of the ownership tuple in _TEMPLATES ("take sole credit" / "Control of
#: {t} passes ...") -- dropped from the candidate set when ownership is barred.
_OWNERSHIP_TEMPLATE_INDEX = 2
#: Neutral, always-ownable default object term for the dignity fallback.
_SAFE_DEFAULT_TERM = "the findings"


def is_nonownable_story_object(term: object) -> bool:
    """True when ``term`` names PEOPLE or a protected-identity / harm-population
    group -- a thing a dramatic-state template must never put up for ownership,
    credit, or control (3.1 dignity guard). An object head with a people MODIFIER
    is ownable and returns False (``patient records``, ``children's health
    study``, ``survey data``). Deterministic; never raises."""
    t = _norm(term)
    if not t:
        return False
    for sub in _PROTECTED_IDENTITY_SUBSTRINGS:
        if sub in t:
            return True
    words = t.split()
    if not words:
        return False
    head = words[-1]
    if head in _PEOPLE_NOUNS:
        return True
    if head.endswith("s") and head[:-1] in _PEOPLE_NOUNS:   # naive plural fold
        return True
    return False


def derive_safe_fallback_term(key_terms: object, cast: object = (),
                              default: str = _SAFE_DEFAULT_TERM) -> str:
    """The first key term that is a safe, OWNABLE story object (not people-class
    / protected-identity, not a cast member's name); else ``default``.
    Deterministic; never raises."""
    cast_norm = {_norm(c) for c in (cast or ()) if _norm(c)}
    for kt in (key_terms or ()):
        s = str(kt).strip()
        if not s or _norm(s) in cast_norm:
            continue
        if not is_nonownable_story_object(s):
            return s
    return default


def _pick_templates(term: str, arc_shape: str = "", *,
                    allow_ownership: bool = True) -> Tuple[str, str, str, str]:
    tmpls = _SHAPE_TEMPLATES.get(arc_shape, _TEMPLATES)
    if not allow_ownership and tmpls is _TEMPLATES:
        # Drop the ownership/credit tuple so a people-class term can never be
        # templated as something to "take sole credit for" / "control".
        tmpls = tuple(t for i, t in enumerate(_TEMPLATES)
                      if i != _OWNERSHIP_TEMPLATE_INDEX)
    candidates = tmpls or _TEMPLATES        # defensive belt (never empty)
    idx = (sum(ord(c) for c in term) % len(candidates)) if term else 0
    a, b, q, e = candidates[idx]
    return (a.replace("{t}", term), b.replace("{t}", term),
            q.replace("{t}", term), e.replace("{t}", term))


def _fallback_state(meta: Any, voice_slot_ids, costly_choice_beat: str,
                    arc_shape: str = "") -> DramaticState:
    """Deterministic news-templated wants (shape-appropriate). NEVER the
    _DEFAULT_* constant path. Guarantees the chosen term is in meta key_terms
    (the turning-slot detail floor)."""
    key_terms = _key_terms(meta)
    raw_first = key_terms[0] if key_terms else (
        _first_noun_phrase(_script_brief(meta)) or "the event")
    # 3.1 dignity guard: pick the first OWNABLE key term (else the neutral
    # default) BEFORE templating, and drop the ownership tuple if the chosen term
    # is somehow still people-class (belt). Preserves the legacy term byte-for-
    # byte whenever the first key term is already an ownable object.
    candidates = list(key_terms) or [raw_first]
    term = derive_safe_fallback_term(candidates, default=_SAFE_DEFAULT_TERM)
    replaced = _norm(term) != _norm(raw_first)
    _inject_key_term(meta, term)
    allow_ownership = not is_nonownable_story_object(term)
    a, b, q, e = _pick_templates(term, arc_shape, allow_ownership=allow_ownership)
    if isinstance(meta, dict):
        meta["dramatic_state_fallback_term"] = term
        meta["dramatic_state_fallback_term_replaced"] = bool(replaced)
    return DramaticState(
        dramatic_question=_trunc(q, 240),
        character_a_wants=_trunc(a, 120),
        character_b_wants=_trunc(b, 120),
        costly_choice_beat=costly_choice_beat,
        ending_change=_trunc(e, 200),
    )


def _make_post_validator(key_terms: List[str], confrontation: bool = True):
    """The about-the-news guard. Returns an error string to REJECT (advancing
    the structured_call ladder), None to ACCEPT. ``confrontation`` (F8) gates
    whether the two wants must be OPPOSED or merely distinct + news-grounded."""
    norm_terms = [t for t in (_norm(x) for x in key_terms) if t]

    def _pv(ds: DramaticStateLLM) -> Optional[str]:
        a = _norm(ds.character_a_wants)
        b = _norm(ds.character_b_wants)
        if a == b:
            # F8: identical wants are bad for ANY shape, but only confrontation
            # shapes are asked to be OPPOSED; non-confrontation shapes
            # (investigation / slow_dread) just need distinct, news-grounded
            # wants -- relaxing this prevents a generation stall on a shape
            # that is not a clash of wills.
            if confrontation:
                return ("character_a_wants and character_b_wants are not "
                        "opposed; make them genuinely conflict over the news")
            return ("character_a_wants and character_b_wants are identical; "
                    "make them distinct and grounded in the news event")
        if norm_terms:
            blob_words = set(" ".join([
                a, b, _norm(ds.dramatic_question), _norm(ds.ending_change),
            ]).split())

            def _hit(term: str) -> bool:
                tw = term.split()
                return bool(tw) and all(w in blob_words for w in tw)

            if not any(_hit(t) for t in norm_terms):
                return ("the conflict is not grounded in the news; at least "
                        "one news key term must appear in the wants, the "
                        "dramatic question, or the ending")
        return None

    return _pv


_ARC_SHAPE_GUIDANCE = {
    "setup_complication_resolution":
        "a clear problem that complicates, then resolves",
    "investigation_without_answer":
        "a search for the truth that never fully arrives -- the wants pull "
        "toward and away from an answer, not head-to-head",
    "slow_dread":
        "a creeping threat that closes in -- the wants are hold-it-back vs "
        "let-it-come, not a clash of two equals",
    "heist": "a risky plan to take or pull off something, under a clock",
    "betrayal": "trust that turns -- an alliance that breaks by the end",
}


def _build_prompt(meta: Any, cast_rows, key_terms: List[str],
                  arc_shape: str = "") -> str:
    script_brief = _script_brief(meta)
    names = []
    for row in (cast_rows or []):
        if isinstance(row, dict):
            nm = str(row.get("name") or "").strip()
            if nm:
                names.append(nm)
        if len(names) >= 2:
            break
    a_name = names[0] if names else "the protagonist"
    b_name = names[1] if len(names) > 1 else "the antagonist"
    terms_line = ", ".join(key_terms) if key_terms else "(none provided)"
    confrontation = (not arc_shape) or (arc_shape in _CONFRONTATION_SHAPES)
    shape_line = ""
    if arc_shape:
        shape_line = (
            f"ARC SHAPE: {arc_shape} -- "
            f"{_ARC_SHAPE_GUIDANCE.get(arc_shape, '')}\n"
        )
    if confrontation:
        wants_line = (
            "Produce two OPPOSED wants -- A and B must want things that "
            "cannot both be satisfied, and both wants must be rooted in the "
            "news event. "
        )
    else:
        wants_line = (
            "Give A and B two DISTINCT wants that fit the arc shape above "
            "(they need not be head-to-head opposed), both rooted in the "
            "news event. "
        )
    return (
        "You are the story architect for a short audio drama whose premise "
        "comes from a real news item. Define the CENTRAL CONFLICT so it is "
        "authentically ABOUT that news -- not a generic story that merely "
        "name-drops it.\n\n"
        f"{shape_line}"
        f"NEWS KEY TERMS: {terms_line}\n"
        f"NEWS PREMISE: {script_brief or '(none)'}\n"
        f"LEAD CHARACTERS: {a_name} (A) and {b_name} (B)\n\n"
        f"{wants_line}"
        "Then a single dramatic question the audience holds the whole way, "
        "and the ending change (how the situation ends up different), both "
        "referencing the news. Use at least one of the news key terms in the "
        "wants, the question, or the ending.\n\n"
        "Return ONLY a JSON object with exactly these string keys:\n"
        '{"character_a_wants": "...", "character_b_wants": "...", '
        '"dramatic_question": "...", "ending_change": "..."}\n'
        "character_a_wants and character_b_wants: 4-120 characters each. "
        "dramatic_question: 10-240 characters. ending_change: 4-200 "
        "characters. No commentary outside the JSON."
    )


def derive_news_dramatic_state(
    *,
    meta: Any,
    cast_rows,
    voice_slot_ids,
    slot_fn,
    base_temperature: float = 0.5,
    structural_retry_temperature: float = 0.3,
    structured_call_fn=None,
    arc_shape: str = "",
) -> DramaticState:
    """Construct a DramaticState whose conflict is authentically about the
    news, via the resident technical slot. NEVER raises -- any LLM /
    validation failure degrades to the deterministic news-templated fallback.

    ``arc_shape`` (F8) steers the prompt + the post-validator + the fallback
    templates toward the episode's chosen shape; CONFRONTATION shapes require
    opposed wants, the others only distinct + news-grounded.

    Side effect (intended): guarantees >= 1 entry in
    ``meta['news']['key_terms']`` (the turning-slot detail floor) and stamps
    ``meta['dramatic_state_source']`` for observability. The caller stamps
    ``meta['dramatic_state']`` from the returned object.
    """
    costly_choice_beat = pick_costly_choice_slot(voice_slot_ids)
    orig_key_terms = _key_terms(meta)
    script_brief = _script_brief(meta)
    confrontation = (not arc_shape) or (arc_shape in _CONFRONTATION_SHAPES)
    if arc_shape and isinstance(meta, dict):
        meta["arc_shape"] = arc_shape

    # No usable news context at all -> deterministic fallback (no wasted LLM
    # call). _fallback_state still seeds meta key_terms ("the event") so the
    # turning-slot detail floor holds even with news=None.
    if not orig_key_terms and not script_brief:
        state = _fallback_state(meta, voice_slot_ids, costly_choice_beat, arc_shape)
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "fallback_no_news"
        return state

    # Turning-slot detail floor: ensure key_terms is non-empty before the LLM
    # grounding + contract derivation (seed from the brief's first noun phrase).
    key_terms = orig_key_terms
    if not key_terms:
        seed = _first_noun_phrase(script_brief) or "the event"
        _inject_key_term(meta, seed)
        key_terms = _key_terms(meta)

    sc = structured_call_fn
    if sc is None:
        try:
            from ._otr_structured_call import structured_call as sc  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log.warning("[B1] structured_call import failed (%r); fallback", exc)
            state = _fallback_state(meta, voice_slot_ids, costly_choice_beat, arc_shape)
            if isinstance(meta, dict):
                meta["dramatic_state_source"] = "fallback_import_error"
            return state

    try:
        llm = sc(
            prompt=_build_prompt(meta, cast_rows, key_terms, arc_shape),
            schema=DramaticStateLLM,
            slot_fn=slot_fn,
            base_temperature=base_temperature,
            structural_retry_temperature=structural_retry_temperature,
            post_validator=_make_post_validator(key_terms, confrontation),
            helper_name="derive_dramatic_state",
        )
        state = DramaticState(
            dramatic_question=_trunc(llm.dramatic_question, 240),
            character_a_wants=_trunc(llm.character_a_wants, 120),
            character_b_wants=_trunc(llm.character_b_wants, 120),
            costly_choice_beat=costly_choice_beat,
            ending_change=_trunc(llm.ending_change, 200),
        )
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "llm"
        log.info("[B1] news-derived DramaticState built from the LLM.")
        return state
    except Exception as exc:  # noqa: BLE001 -- never break audio
        log.warning(
            "[B1] news-derived DramaticState LLM path failed (%s: %s); "
            "using the deterministic news-templated fallback.",
            type(exc).__name__, str(exc)[:200],
        )
        state = _fallback_state(meta, voice_slot_ids, costly_choice_beat, arc_shape)
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "fallback"
        return state
