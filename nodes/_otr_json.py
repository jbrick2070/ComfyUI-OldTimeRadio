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

# ```json ... ``` / ``` ... ``` fenced block. Non-greedy ``{...}`` so the
# first balanced object between fences wins.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def extract_first_json_block(raw: str) -> str:
    """Return the first complete top-level JSON object in ``raw`` as a
    substring, or ``""`` when none is found.

    Primary form: a ```json ... ``` fenced block. Fallback: walk to the
    first ``{`` and ``json.JSONDecoder().raw_decode`` from there --
    raw_decode stops at the end of the first complete object, so any
    trailing content (a second hallucinated object, a prose note) is
    ignored rather than concatenated into the slice. Never raises.
    """
    if not raw:
        return ""
    text = raw.strip()

    # Primary: a fenced JSON block. Non-greedy capture picks the first
    # balanced {...} between fences.
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            if isinstance(json.loads(candidate), dict):
                return candidate
        except json.JSONDecodeError:
            pass

    # Fallback: walk to the first '{' and raw_decode from there.
    # raw_decode returns the parsed object plus the index where it
    # ended, so trailing content after the first object is ignored.
    decoder = json.JSONDecoder()
    first_brace = text.find("{")
    # When a response starts with an object, a malformed outer object must
    # fail closed. Scanning onward into a nested object would silently turn a
    # broken envelope into a plausible-looking child artifact and bypass the
    # typed repair ladder (the sci-fi P0 live smoke exposed this exact drift).
    outer_object = first_brace >= 0 and not text[:first_brace].strip()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            if outer_object and i == first_brace:
                return ""
            continue
        if isinstance(obj, dict):
            return text[i:i + end]
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
