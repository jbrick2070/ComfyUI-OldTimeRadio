"""nodes/_otr_specificity.py -- Story-Quality R2 C1 + C2 (roundtable-converged).

Pure, deterministic derivations + zero-ripple prompt injectors for:
  * C1 specificity anchors  -> meta["specificity_anchors"] (list[str]), injected
    into the per-line canon_header as concrete context.
  * C2 central story-object  -> meta["central_object"] (str, "" = none), injected
    into the announcer-close news_close_brief.

Both derive SOLELY from the CURATED `key_terms` (the news brief's LLM-distilled
salient entities) -- no new LLM call, no raw-news regex (panel: brittle/garbage).
central_object is CONSERVATIVE: it returns "" rather than invent one.

PURE module: stdlib only (re). Deterministic + idempotent. Never raises (every
function returns a safe default on bad input). UTF-8 no BOM, ASCII source.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Set

MAX_ANCHORS = 5

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z]+")
#: org / place suffix words that mark a key_term as an entity, not an object.
_ENTITY_SUFFIX = frozenset({
    "agency", "administration", "university", "institute", "corp",
    "corporation", "inc", "company", "department", "island", "city",
    "county", "state", "observatory", "laboratory", "lab", "center", "centre",
})
#: hidden sentinels for the prompt blocks (the writer also tracks a meta flag).
_ANCHOR_BLOCK_HEAD = (
    "Specificity anchors (when natural, ground a line in one of these concrete "
    "details; do not force them into every line):"
)
def _as_str_list(items: Any) -> List[str]:
    """Coerce key_terms-like input to a clean list[str]. Never raises."""
    if items is None:
        return []
    if isinstance(items, str):
        items = (items,)
    try:
        out: List[str] = []
        for it in items:  # type: ignore[assignment]
            s = _WS.sub(" ", str(it or "")).strip()
            if s:
                out.append(s)
        return out
    except TypeError:
        return []


def _cast_tokens(cast: Any) -> Set[str]:
    """Casefolded exclusion set from the locked cast: each full name + each
    sub-token of length > 2 (so 'a'/'j'/'will' do not over-exclude). Never raises."""
    toks: Set[str] = set()
    try:
        rows: Iterable = cast if cast is not None else ()
        for row in rows:
            if isinstance(row, str):
                name = row                       # a plain cast-name string
            elif isinstance(row, dict):
                name = str(row.get("name") or "")
            else:
                name = str(getattr(row, "name", "") or "")
            name = _WS.sub(" ", name).strip()
            if not name:
                continue
            toks.add(name.casefold())
            for t in _TOKEN.findall(name):
                if len(t) > 2:
                    toks.add(t.casefold())
    except TypeError:
        return toks
    return toks


def _candidate_tokens(s: str) -> Set[str]:
    return {t.casefold() for t in _TOKEN.findall(s)}


def derive_specificity_anchors(key_terms: Any, cast: Any) -> List[str]:
    """3-5 concrete anchors from the curated key_terms, excluding cast names.
    Pure, deterministic, never raises."""
    cast_tok = _cast_tokens(cast)
    seen: Set[str] = set()
    out: List[str] = []
    for term in _as_str_list(key_terms):
        if len(term) < 2:
            continue
        fold = term.casefold()
        if fold in cast_tok:
            continue
        if _candidate_tokens(term) & cast_tok:
            continue
        if fold in seen:
            continue
        seen.add(fold)
        out.append(term)
        if len(out) >= MAX_ANCHORS:
            break
    return out


def derive_central_object(key_terms: Any, cast: Any) -> str:
    """The first key_term that reads as a concrete physical OBJECT phrase, else ""
    (conservative -- never invents). Pure, deterministic, never raises."""
    cast_tok = _cast_tokens(cast)
    for term in _as_str_list(key_terms):
        if len(term) < 2:
            continue
        if term.casefold() in cast_tok or (_candidate_tokens(term) & cast_tok):
            continue
        if term.isupper():                       # ALL-CAPS org acronym (NASA)
            continue
        words = term.split()
        if len(words) == 1 and words[0][:1].isupper():
            continue                             # bare proper-noun entity (Swift)
        alpha = _TOKEN.findall(term)
        if not alpha:
            continue
        if all(w[:1].isupper() for w in alpha):  # Title-Case proper noun
            continue
        if any(w.casefold() in _ENTITY_SUFFIX for w in alpha):
            continue
        if not any(w.islower() for w in alpha):  # need a lowercase common noun
            continue
        return term
    return ""


def sanitize_for_prompt(s: Any) -> str:
    """Collapse whitespace + strip control chars/newlines. No length cap (curated
    terms are short). Pure; never raises."""
    try:
        text = str(s or "")
        # map control chars / newlines / tabs to a SPACE (not drop -- dropping
        # would glue adjacent words), then collapse runs of whitespace.
        text = "".join(ch if ord(ch) >= 32 else " " for ch in text)
        return _WS.sub(" ", text).strip()
    except Exception:  # noqa: BLE001
        return ""


def inject_anchors_into_header(canon_header: Any, anchors: Any) -> str:
    """Append the anchors block to canon_header (unchanged when none survive
    sanitize). Pure; idempotent at the writer via a meta flag."""
    header = str(canon_header or "")
    safe = [a for a in (sanitize_for_prompt(x) for x in _as_str_list(anchors)) if a]
    if not safe:
        return header
    block = _ANCHOR_BLOCK_HEAD + "\n- " + "\n- ".join(safe)
    return f"{header}\n\n{block}" if header else block


def inject_central_object_into_brief(brief: Any, central_object: Any) -> str:
    """Land the news close on the central object as a concrete FINAL IMAGE
    (unchanged when "").

    The close brief is SPOKEN VERBATIM by the news coda (compose_news_coda
    appends it deterministically), so this emits a CLEAN, capitalized,
    end-stopped sentence -- NEVER the prompt-scaffold label. The literal
    "Central object, if useful:" used to be concatenated here and the announcer
    SPOKE it ("...Central object, if useful: a cracked phone."); BUG flagged +
    fixed 2026-06-25. Pure; deterministic; never raises."""
    text = str(brief or "")
    obj = sanitize_for_prompt(central_object)
    if not obj:
        return text
    if obj[:1].islower():            # read as a sentence, not a sentence fragment
        obj = obj[0].upper() + obj[1:]
    tail = obj if obj[-1:] in ".!?" else obj + "."
    return f"{text} {tail}".strip() if text else tail
