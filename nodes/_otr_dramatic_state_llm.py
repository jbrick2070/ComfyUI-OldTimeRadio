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
  * The prompt asks for news-grounded, opposed wants as creative direction.
    Those quality preferences are deliberately non-gating: once the four
    fields satisfy the typed schema, they are accepted without a model retry.
  * DETERMINISTIC FALLBACK (LLM reject / news failure): opposed wants built
    from `meta["news"]["key_terms"]` + PAIRED conflict templates opposed BY
    CONSTRUCTION. Never the `_DEFAULT_*` constant path.
  * Optional news key terms remain optional. This module never fabricates a
    term from the brief or mutates an empty key-term list to satisfy a later
    quality preference.

NEVER raises (Prime Directive 1, audio is king): any LLM / validation
failure degrades to the deterministic fallback. Content-only -- the ledger
wire format is untouched; only `meta` sub-keys change. UTF-8 no BOM.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Tuple

from pydantic import BaseModel, field_validator

from ._otr_dramatic_state import DramaticState, pick_costly_choice_slot

log = logging.getLogger("OTR_DramaticStateB1")

__all__ = [
    "DramaticStateLLM",
    "derive_news_dramatic_state",
    "ARC_SHAPES",
    "pick_arc_shape",
]

# F8 (story-engine v1): arc-shape variety. A seeded pre-step picks one of these
# shapes per episode so every story is not the same setup->complication->
# resolution mold. CONFRONTATION shapes ask for two opposed wants; the others
# (investigation / slow_dread) are not framed as a clash of wills. This is
# prompt guidance only and never an acceptance gate.
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


class DramaticStateLLM(BaseModel):
    """LLM-authored dramatic context with structural validation only."""

    dramatic_question: str
    character_a_wants: str
    character_b_wants: str
    ending_change: str

    @field_validator(
        "dramatic_question",
        "character_a_wants",
        "character_b_wants",
        "ending_change",
    )
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dramatic-state text must not be blank")
        return value




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
# confrontation arc shapes. They are not framed as a head-to-head clash:
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



def _pick_templates(term: str, arc_shape: str = "") -> Tuple[str, str, str, str]:
    tmpls = _SHAPE_TEMPLATES.get(arc_shape, _TEMPLATES)
    candidates = tmpls or _TEMPLATES        # defensive belt (never empty)
    idx = (sum(ord(c) for c in term) % len(candidates)) if term else 0
    a, b, q, e = candidates[idx]
    return (a.replace("{t}", term), b.replace("{t}", term),
            q.replace("{t}", term), e.replace("{t}", term))


def _fallback_state(meta: Any, voice_slot_ids, costly_choice_beat: str,
                    arc_shape: str = "") -> DramaticState:
    """Deterministic news-templated wants (shape-appropriate). NEVER the
    _DEFAULT_* constant path."""
    key_terms = _key_terms(meta)
    raw_first = key_terms[0] if key_terms else "the event"
    # Preserve the accepted source vocabulary. Python does not classify or
    # replace story nouns; the prompt and shared safety boundary own prose.
    term = str(raw_first).strip() or "the event"
    a, b, q, e = _pick_templates(term, arc_shape)
    if isinstance(meta, dict):
        meta["dramatic_state_fallback_term"] = term
    return DramaticState(
        dramatic_question=q,
        character_a_wants=a,
        character_b_wants=b,
        costly_choice_beat=costly_choice_beat,
        ending_change=e,
    )



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
        "Every value must be a non-empty string. Wording and length are "
        "creative choices, not rejection criteria. No commentary outside "
        "the JSON."
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

    ``arc_shape`` (F8) steers the prompt and no-news fallback templates toward
    the episode's chosen shape. Shape, opposition, and grounding are creative
    guidance and never reject a structurally valid model response.

    The optional key-term list is preserved as supplied. The function stamps
    ``meta['dramatic_state_source']`` for observability; the caller stamps
    ``meta['dramatic_state']`` from the returned object.
    """
    costly_choice_beat = pick_costly_choice_slot(voice_slot_ids)
    orig_key_terms = _key_terms(meta)
    script_brief = _script_brief(meta)
    if arc_shape and isinstance(meta, dict):
        meta["arc_shape"] = arc_shape

    # No usable news context at all -> deterministic fallback (no wasted LLM
    # call). Optional source key terms remain empty.
    if not orig_key_terms and not script_brief:
        state = _fallback_state(meta, voice_slot_ids, costly_choice_beat, arc_shape)
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "fallback_no_news"
        return state

    key_terms = orig_key_terms

    sc = structured_call_fn
    if sc is None:
        try:
            from ._otr_structured_call import structured_call as sc  # type: ignore
        except Exception as exc:  # noqa: BLE001
            # NO-FALLBACK (operator 2026-07-03): an IMPORT failure here means the
            # dramatic-state feature's own code cannot load -- a broken install /
            # missing dependency, not a "no news" case. FAIL LOUD; never silently
            # ship a stock templated dramatic skeleton that steers the episode.
            raise RuntimeError(
                "[B1] dramatic-state: structured_call could not be imported "
                "(%r). NO templated-state fallback (no-fallback rip) -- fix the "
                "install/dependency; a broken feature must not silently downgrade "
                "the storytelling." % (exc,)
            ) from exc

    try:
        llm = sc(
            prompt=_build_prompt(meta, cast_rows, key_terms, arc_shape),
            schema=DramaticStateLLM,
            slot_fn=slot_fn,
            base_temperature=base_temperature,
            structural_retry_temperature=structural_retry_temperature,
            helper_name="derive_dramatic_state",
        )
        state = DramaticState(
            dramatic_question=llm.dramatic_question,
            character_a_wants=llm.character_a_wants,
            character_b_wants=llm.character_b_wants,
            costly_choice_beat=costly_choice_beat,
            ending_change=llm.ending_change,
        )
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "llm"
        log.info("[B1] news-derived DramaticState built from the LLM.")
        return state
    except Exception as exc:  # noqa: BLE001 - optional authoring pass
        log.warning(
            "[B1] dramatic-state LLM path failed (%s: %s); preserving episode "
            "liveness with the source-grounded fallback.",
            type(exc).__name__,
            exc,
        )
        state = _fallback_state(
            meta, voice_slot_ids, costly_choice_beat, arc_shape,
        )
        if isinstance(meta, dict):
            meta["dramatic_state_source"] = "fallback_llm_error"
        return state
