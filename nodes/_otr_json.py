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
a fenced-block match plus ``json.JSONDecoder.raw_decode`` that takes the
FIRST complete object and ignores any trailing content. This module is
that logic's single home; ``news_interpreter`` now re-exports
``extract_json_block`` from here.

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


def _escape_raw_controls_in_strings(blob: str) -> str:
    """Turn raw control characters inside JSON strings into escapes.

    Gemma pretty-prints dialogue across real line breaks. Strict JSON
    forbids an unescaped U+000A in a string, so ``raw_decode`` rejects
    the whole object. The heartbeat logger collapses whitespace, which
    is why the live log can look like valid JSON while the extractor
    returns empty. Unescaped quotes are left alone -- inventing where
    a string ends is not this helper's job.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in blob:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = False
            continue
        if ch == "\n":
            out.append("\\n")
            continue
        if ch == "\r":
            out.append("\\r")
            continue
        if ch == "\t":
            out.append("\\t")
            continue
        code = ord(ch)
        if code < 32:
            out.append("\\u%04x" % code)
            continue
        out.append(ch)
    return "".join(out)


def _strip_trailing_commas(blob: str) -> str:
    """Drop commas that only precede a closing } or ] outside strings."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    n = len(blob)
    while i < n:
        ch = blob[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and blob[j] in " \t\r\n":
                j += 1
            if j < n and blob[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_llm_json(blob: str) -> str:
    """Apply the two Gemma JSON defects that strict ``raw_decode`` rejects."""
    return _strip_trailing_commas(_escape_raw_controls_in_strings(blob))


def _try_object(decoder: json.JSONDecoder, blob: str):
    try:
        obj, end = decoder.raw_decode(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj, end


def _decode_first_object(blob: str) -> str:
    """First complete nonempty top-level object starting at the first ``{``.

    Tries the slice as written, then one LLM-JSON repair of that same
    slice. Never scans onward after the first brace fails: that would
    salvage a nested child of a malformed envelope (Codex P5).

    Empty ``{}`` is skipped only when another ``{`` follows it (a
    preamble like ``Here is {}`` before the real artifact). A repair
    that turns ``{,}`` into ``{}`` with nothing after is fail-closed so
    it stays a JSON syntax miss, not a schema miss that skips the
    structural retry.
    """
    decoder = json.JSONDecoder()
    first_brace = blob.find("{")
    if first_brace < 0:
        return ""
    original = blob[first_brace:]
    repaired = _repair_llm_json(original)
    seen: set[str] = set()
    for candidate, is_repair in ((original, False), (repaired, True)):
        if candidate in seen:
            continue
        seen.add(candidate)
        remaining = candidate
        hops = 0
        while remaining and hops < 4:
            hops += 1
            got = _try_object(decoder, remaining)
            if got is None:
                break
            obj, end = got
            if obj:
                return remaining[:end]
            rest = remaining[end:].lstrip()
            nxt = rest.find("{")
            if nxt >= 0:
                remaining = rest[nxt:]
                continue
            if is_repair:
                return ""
            return remaining[:end]
    return ""


def extract_first_json_block(raw: str) -> str:
    """Return JSON text for the first complete top-level object, or ``""``.

    The returned string is always ``json.loads``-able when nonempty. It is
    a substring of ``raw`` when the model already emitted strict JSON; it
    is a repaired *copy* when Gemma left raw newlines in strings or
    trailing commas -- callers must parse it, never require ``block in raw``.

    Primary form: a ```json ... ``` fenced block. Fallback: decode from the
    first ``{``. ``raw_decode`` stops at the end of the first complete object,
    so trailing content (a second hallucinated object or prose note) is
    ignored rather than concatenated into the slice. A malformed outer object
    never falls through to one of its decodable child objects. Never raises.

    A fenced body may start with prose (``Here is the act:``) before the
    object; decode still starts at the first ``{`` *inside the fence*.
    If that body still does not decode, fail closed rather than searching
    past the fence. Unescaped quotes inside a string still fail closed --
    inventing the string boundary is not this helper's job.
    """
    if not raw:
        return ""
    text = raw.strip()

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return _decode_first_object(fence_match.group(1).strip())
    return _decode_first_object(text)


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
