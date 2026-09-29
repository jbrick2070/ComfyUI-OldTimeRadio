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

Padded-key identity (live Gemma 2026-09-14): leftover JSON keys such as
``"speaker "`` are the same keys after whitespace is stripped. That is
identity of leftover JSON, not new content -- the same rule SpokenLine
documents. ``parse_first_json_object`` applies it after ``json.loads``;
string VALUES are never rewritten. Key-strip is structural identity, so
``validate_tolerant_data``'s "never rewrites strings" rule is untouched.
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
_JSON_FENCE_LABELED_RE = re.compile(
    r"```json\s*(.*?)\s*```",
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
                prev = None
                for prev_ch in reversed(out):
                    if prev_ch in " \t\r\n":
                        continue
                    prev = prev_ch
                    break
                if prev not in ("{", "["):
                    i += 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_llm_json(blob: str) -> str:
    """Apply the two Gemma JSON defects that strict ``raw_decode`` rejects."""
    return _strip_trailing_commas(_escape_raw_controls_in_strings(blob))


#: How the repair writes a raw control character it finds inside a string.
_ESCAPED_CONTROLS = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _origin_of(repaired: str, original: str) -> "list | None":
    """For each index of ``repaired``, and its end, the index in ``original``
    it came from. The repair only writes a raw control character in a string
    as its escape and drops a trailing comma, so the two align greedily; None
    if they ever do not (a repair this does not know about)."""
    origin: list = []
    i = j = 0
    while j < len(repaired):
        if i < len(original) and original[i] == repaired[j]:
            origin.append(i)
            i += 1
            j += 1
            continue
        if i < len(original):
            ch = original[i]
            escaped = _ESCAPED_CONTROLS.get(ch) or ("\\u%04x" % ord(ch) if ord(ch) < 32 else "")
            if escaped and repaired.startswith(escaped, j):
                origin.extend([i] * len(escaped))
                i += 1
                j += len(escaped)
                continue
            if ch == ",":
                i += 1
                continue
        return None
    origin.append(i)
    return origin


def _try_object(decoder: json.JSONDecoder, blob: str, errors=None, is_repair=False):
    try:
        obj, end = decoder.raw_decode(blob)
    except json.JSONDecodeError as exc:
        if errors is not None:
            errors.setdefault(is_repair, (exc.msg, blob, exc.pos))
        return None
    except RecursionError:
        # Nested deeper than the decoder follows: a JSON miss like any other,
        # never an exception out of an extractor documented never to raise.
        if errors is not None:
            errors.setdefault(is_repair, ("nesting too deep to decode", blob, 0))
        return None
    if not isinstance(obj, dict):
        return None
    return obj, end


def _as_written(errors: dict, original: str, repaired: str) -> tuple:
    """A recorded decode error as ``(msg, doc, pos)`` in the object as the model
    wrote it: the error left after the repair when there is one -- a raw newline
    or a trailing comma the repair tolerates is not the defect -- moved back onto
    ``original``; else the error in ``original`` itself."""
    if True in errors:
        msg, doc, pos = errors[True]
        at = len(repaired) - len(doc) + pos
        origin = _origin_of(repaired, original)
        if origin is not None and 0 <= at < len(origin):
            return msg, original, origin[at]
        if False not in errors:
            return msg, repaired, at
    msg, doc, pos = errors[False]
    return msg, original, len(original) - len(doc) + pos


def _decode_first_object(blob: str, failures: "list | None" = None) -> str:
    """First complete nonempty top-level object starting at the first ``{``.

    Tries the slice as written, then one LLM-JSON repair of that same
    slice. Never scans onward after the first brace fails: that would
    salvage a nested child of a malformed envelope (Codex P5).

    Empty ``{}`` is skipped only when the next token is another object
    (``Here is {} {artifact}``). A later ``{`` buried in leftover keys
    is not a hop. A repair that turns ``{,}`` into ``{}`` with nothing
    after is fail-closed so it stays a JSON syntax miss, not a schema
    miss that skips the structural retry.

    ``failures``, when given, gets ``(msg, doc, pos)`` for an object this walk
    gave up on (see ``_as_written``). Nothing is recorded when it is not given.
    """
    decoder = json.JSONDecoder()
    first_brace = blob.find("{")
    if first_brace < 0:
        return ""
    original = blob[first_brace:]
    repaired = _repair_llm_json(original)
    errors = {} if failures is not None else None

    def give_up() -> str:
        if errors:
            failures.append(_as_written(errors, original, repaired))
        return ""

    seen: set[str] = set()
    for candidate, is_repair in ((original, False), (repaired, True)):
        if candidate in seen:
            continue
        seen.add(candidate)
        remaining = candidate
        hops = 0
        while remaining and hops < 4:
            hops += 1
            got = _try_object(decoder, remaining, errors, is_repair)
            if got is None:
                break
            obj, end = got
            if obj:
                return remaining[:end]
            rest = remaining[end:].lstrip()
            # Hop only when the next token is another object. Searching
            # for a later `{` inside leftover keys would salvage a nested
            # child of `{} "lines": [ {...} ]`.
            if rest.startswith("{"):
                remaining = rest
                continue
            if is_repair:
                return give_up()
            return remaining[:end]
    return give_up()


def extract_first_json_block(raw: str) -> str:
    """Return JSON text for the first complete top-level object, or ``""``.

    The returned string is always ``json.loads``-able when nonempty. It is
    a substring of ``raw`` when the model already emitted strict JSON; it
    is a repaired *copy* when Gemma left raw newlines in strings or
    trailing commas -- callers must parse it, never require ``block in raw``.

    Primary form: a ```json ... ``` fenced block, preferred over an earlier
    unlabelled thinking fence. Fallback: decode from the first ``{``. ``raw_decode`` stops at the end of the first complete object,
    so trailing content (a second hallucinated object or prose note) is
    ignored rather than concatenated into the slice. A malformed outer object
    never falls through to one of its decodable child objects. Never raises.

    A fenced body may start with prose (``Here is the act:``) before the
    object; decode still starts at the first ``{`` *inside the fence*.
    If that body still does not decode, fail closed rather than searching
    past the fence. Unescaped quotes inside a string still fail closed --
    inventing the string boundary is not this helper's job.
    """
    return _extract(raw)


def _extract(raw: str, failures: "list | None" = None) -> str:
    """``extract_first_json_block``, recording into ``failures`` the decoder's
    error for each object it gave up on, in the order it tried them, as
    ``(preference, msg, doc, pos)``: 0 for a ```json fence, which the
    extractor prefers, 1 for any other fence or the bare reply."""
    if not raw:
        return ""
    text = raw.strip()

    labeled = list(_JSON_FENCE_LABELED_RE.finditer(text))
    labeled_spans = {(m.start(), m.end()) for m in labeled}
    other = [
        m for m in _JSON_FENCE_RE.finditer(text)
        if (m.start(), m.end()) not in labeled_spans
    ]
    def attempt(blob: str, preference: int) -> str:
        found = [] if failures is not None else None
        block = _decode_first_object(blob, found)
        if found:
            failures.extend((preference,) + tuple(f) for f in found)
        return block

    saw_fence = False
    for preference, match in [(0, m) for m in labeled] + [(1, m) for m in other]:
        saw_fence = True
        block = attempt(match.group(1).strip(), preference)
        if block:
            return block
    if saw_fence:
        return ""
    return attempt(text, 1)


def normalize_json_keys(value):
    """Strip whitespace from leftover JSON object keys.

    Live Gemma 2026-09-14 emitted ``"speaker "`` (trailing space). The
    words were already in the object; the schema missed the key.
    Stripping is identity of leftover JSON keys, not new content --
    string VALUES are not mutated.

    Recurses into nested dicts and lists; scalars are returned unchanged.
    Empty keys after strip are dropped. Duplicate keys after strip
    last-wins (the same overwrite SpokenLine uses when it rebuilds the
    dict).
    """
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            stripped = str(key).strip()
            if not stripped:
                continue
            out[stripped] = normalize_json_keys(val)
        return out
    if isinstance(value, list):
        return [normalize_json_keys(item) for item in value]
    return value


def parse_first_json_object(raw: str) -> dict:
    """Parse and return the first complete top-level JSON object in
    ``raw``.

    Tolerates markdown fences, leading prose, and -- the BUG-LOCAL-261
    failure mode -- a second object or trailing prose AFTER the first
    object. After ``json.loads``, leftover padded keys (``"speaker "``,
    ``"lines "``) are stripped to their identity -- the same rule
    SpokenLine documents -- so structured consumers see the keys they
    declared. String VALUES are not rewritten.

    Raises ``json.JSONDecodeError`` when ``raw`` carries no decodable
    top-level object, so existing ``except json.JSONDecodeError``
    handlers at the call sites still fire unchanged.
    """
    failures: list = []
    block = _extract(raw, failures)
    if not block:
        if not failures:
            raise json.JSONDecodeError(
                "no decodable top-level JSON object found", raw or "", 0,
            )
        # Name the defect and show where it is. "line 1 column 1 (char 0)" sent
        # a log reader nowhere, and it is the error a typed repair is given when
        # a pass's first two replies both fail to decode. The object named is
        # the one the extractor prefers -- a ```json fence over any other --
        # and within that, the first the decoder got past its opening brace
        # on, so a stray "{a, b}" in a prose fence does not outrank the answer
        # and a longer draft fence does not either (Sonnet QA of 725af773:
        # comparing positions across fences did exactly that). The position
        # is in the text as the model wrote it.
        order = range(len(failures))
        best = min(order, key=lambda i: (failures[i][0], failures[i][3] <= 1, i))
        _preference, msg, doc, pos = failures[best]
        raise json.JSONDecodeError(
            "no decodable top-level JSON object found; in the object, %s near: %s"
            % (msg, _near(doc, pos)),
            doc, pos,
        )
    try:
        return normalize_json_keys(json.loads(block))
    except RecursionError:
        # An object nested deeper than Python will walk: the same JSON miss,
        # so the ladder retries it, instead of an exception that ends the pass.
        raise json.JSONDecodeError(
            "no decodable top-level JSON object found; in the object, nesting "
            "too deep to decode", block, 0,
        ) from None


def _near(doc: str, pos: int, span: int = 60) -> str:
    """The text either side of ``pos`` on one line, the spot marked <<HERE>>."""
    before = " ".join(doc[max(0, pos - span):pos].split())
    after = " ".join(doc[pos:pos + span].split())
    return "%s<<HERE>>%s" % (before, after)
