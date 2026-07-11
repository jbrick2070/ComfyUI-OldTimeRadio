"""nodes/_otr_json.py -- tolerant JSON extraction for LLM responses.

Single home for the "pull the first JSON object out of a model
response" logic. Consolidates four naive ``_extract_json_block``
duplicates (``_otr_casting``, ``_otr_outline``, ``_otr_ledger_reviewer``,
``_otr_story_brief``) that sliced from the first ``{`` to the LAST
``}``. When a model emits two top-level objects, that slice returns
``{...}{...}`` and the strict ``json.loads`` that follows rejects the
second as ``Extra data`` -- the BUG-LOCAL-261 casting crash
("HAYES VANCE", 2026-05-24; gemma-4-E4B-it at temperature 0.95 emitted
a valid cast object followed by a second object).

The correct logic already lived in ``news_interpreter.extract_json_block``:
a fenced-block match plus a brace-walk + ``json.JSONDecoder.raw_decode``
that takes the FIRST complete object and ignores any trailing content.
This module is that logic's single home; ``news_interpreter`` now
re-exports ``extract_json_block`` from here.

Pure stdlib (``json`` + ``re``); no sibling imports, safe to import
from any node module.
"""
from __future__ import annotations

import json
import re

# ```json ... ``` / ``` ... ``` fenced block.  The body is decoded with
# ``JSONDecoder.raw_decode`` below; a regex cannot balance nested braces.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def extract_first_json_block(raw: str) -> str:
    """Return the first complete top-level JSON object in ``raw`` as a
    substring, or ``""`` when none is found.

    Primary form: a ```json ... ``` fenced block. Fallback: decode from the
    first ``{``. ``raw_decode`` stops at the end of the first complete object,
    so trailing content (a second hallucinated object or prose note) is
    ignored rather than concatenated into the slice. A malformed outer object
    never falls through to one of its decodable child objects. Never raises.
    """
    if not raw:
        return ""
    text = raw.strip()

    decoder = json.JSONDecoder()

    # Primary: a fenced JSON block. Decode the whole fence body rather than
    # trying to balance nested braces with a regex. A malformed fenced outer
    # object must fail closed; otherwise the fallback could wrongly salvage a
    # valid nested scene/beat object as the model's top-level artifact.
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            obj, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            return ""
        if isinstance(obj, dict):
            return candidate[:end]
        return ""

    # Fallback: raw_decode the first outer object. Scanning onward after that
    # object fails would make a malformed envelope look valid by returning a
    # nested child (the Codex P5 live smoke exposed this exact drift).
    first_brace = text.find("{")
    if first_brace < 0:
        return ""
    try:
        obj, end = decoder.raw_decode(text[first_brace:])
    except json.JSONDecodeError:
        return ""
    if isinstance(obj, dict):
        return text[first_brace:first_brace + end]
    return ""


def parse_first_json_object(raw: str) -> dict:
    """Parse and return the first complete top-level JSON object in
    ``raw``.

    Tolerates markdown fences, leading prose, and -- the BUG-LOCAL-261
    failure mode -- a second object or trailing prose AFTER the first
    object. Raises ``json.JSONDecodeError`` when ``raw`` carries no
    decodable top-level object, so existing ``except json.JSONDecodeError``
    handlers at the call sites still fire unchanged.
    """
    block = extract_first_json_block(raw)
    if not block:
        raise json.JSONDecodeError(
            "no decodable top-level JSON object found", raw or "", 0,
        )
    return json.loads(block)
