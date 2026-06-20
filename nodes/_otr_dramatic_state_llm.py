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

import logging
import re
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from ._otr_dramatic_state import DramaticState, pick_costly_choice_slot

log = logging.getLogger("OTR_DramaticStateB1")

__all__ = [
    "DramaticStateLLM",
    "derive_news_dramatic_state",
]

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


def _pick_templates(term: str) -> Tuple[str, str, str, str]:
    idx = (sum(ord(c) for c in term) % len(_TEMPLATES)) if term else 0
    a, b, q, e = _TEMPLATES[idx]
    return (a.replace("{t}", term), b.replace("{t}", term),
            q.replace("{t}", term), e.replace("{t}", term))


def _fallback_state(meta: Any, voice_slot_ids, costly_choice_beat: str) -> DramaticState:
    """Deterministic news-templated opposed wants. NEVER the _DEFAULT_*
    constant path. Guarantees the chosen term is in meta key_terms (the
    turning-slot detail floor)."""
    key_terms = _key_terms(meta)
    term = key_terms[0] if key_terms else (
        _first_noun_phrase(_script_brief(meta)) or "the event")
    _inject_key_term(meta, term)
    a, b, q, e = _pick_templates(term)
    return DramaticState(
        dramatic_question=_trunc(q, 240),
        character_a_wants=_trunc(a, 120),
        character_b_wants=_trunc(b, 120),
        costly_choice_beat=costly_choice_beat,
        ending_change=_trunc(e, 200),
    )


def _make_post_validator(key_terms: List[str]):
    """The about-the-news guard. Returns an error string to REJECT (advancing
    the structured_call ladder), None to ACCEPT."""
    norm_terms = [t for t in (_norm(x) for x in key_terms) if t]

    def _pv(ds: DramaticStateLLM) -> Optional[str]:
        a = _norm(ds.character_a_wants)
        b = _norm(ds.character_b_wants)
        if a == b:
            return ("character_a_wants and character_b_wants are not opposed; "
                    "make them genuinely conflict over the news event")
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


def _build_prompt(meta: Any, cast_rows, key_terms: List[str]) -> str:
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
    return (
        "You are the story architect for a short audio drama whose premise "
        "comes from a real news item. Define the CENTRAL CONFLICT so it is "
        "authentically ABOUT that news -- not a generic story that merely "
        "name-drops it.\n\n"
        f"NEWS KEY TERMS: {terms_line}\n"
        f"NEWS PREMISE: {script_brief or '(none)'}\n"
        f"LEAD CHARACTERS: {a_name} (A) and {b_name} (B)\n\n"
        "Produce two OPPOSED wants -- A and B must want things that cannot "
        "both be satisfied, and both wants must be rooted in the news event. "
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
) -> DramaticState:
    """Construct a DramaticState whose conflict is authentically about the
    news, via the resident technical slot. NEVER raises -- any LLM /
    validation failure degrades to the deterministic news-templated fallback.

    Side effect (intended): guarantees >= 1 entry in
    ``meta['news']['key_terms']`` (the turning-slot detail floor) and stamps
    ``meta['dramatic_state_source']`` for observability. The caller stamps
    ``meta['dramatic_state']`` from the returned object.
    """
    costly_choice_beat = pick_costly_choice_slot(voice_slot_ids)
    orig_key_terms = _key_terms(meta)
    script_brief = _script_brief(meta)

    # No usable news context at all -> deterministic fallback (no wasted LLM
    # call). _fallback_state still seeds meta key_terms ("the event") so the
    # turning-slot detail floor holds even with news=None.
    if not orig_key_terms and not script_brief:
        state = _fallback_state(meta, voice_slot_ids, costly_choice_beat)
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
            state = _fallback_state(meta, voice_slot_ids, costly_choice_beat)
            if isinstance(meta, dict):
                meta["dramatic_state_source"] = "fallback_import_error"
            return state

    try:
        llm = sc(
            prompt=_build_prompt(meta, cast_rows, key_terms),
            schema=DramaticStateLLM,
            slot_fn=slot_fn,
            base_temperature=base_temperature,
            structural_retry_temperature=structural_retry_temperature,
            post_validator=_make_post_validator(key_terms),
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
        state = _fallback_state(meta, voice_slot_ids, costly_choice_beat)
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "fallback"
        return state
