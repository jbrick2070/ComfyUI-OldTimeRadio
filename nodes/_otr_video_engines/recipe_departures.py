"""What a MEASUREMENT run departed from, in the receipt it stamps (LANE 2).

THE DEFECT. Since B6 a prequalification clip stamps a single generic
``+prequalification`` suffix, so a winning sweep artifact cannot prove WHICH
knob values produced it. The durable ledger says a sweep ran; it does not say
which cell. Four cells of the 2026-07-27 ltx sweep are indistinguishable in
``meta.render_engines`` -- the one record that outlives the run -- and the
winner was selected from a table kept outside it.

THE FIX. Name the DEPARTURES from the frozen recipe:
``ltx098_distilled_2b_i2v_single_pass_v2+prequalification[tiled_vae=off]``.
A cell that changed nothing still reads ``+prequalification``, so the marking
contract B6 established is unchanged.

WHY THIS MODULE EXISTS AT ALL. Three adapters now carry the freeze mechanism
(``ltx_8gb``, ``wan_ti2v``, ``wan_i2v``) across two lanes that must NOT import
each other -- ``eng_ltx_8gb`` reaching into ``wan_recipe`` would be exactly the
cross-lane coupling this build keeps paying for. But the suffix is read by ONE
consumer chain, so its FORMAT must be one implementation or the ledger grows
two dialects. This module is that implementation and nothing else: pure
functions over plain data, no engine imports, no environment reads.

WHAT IT DELIBERATELY DOES NOT DO. It does not truncate the departure list. A
cap would silently destroy the distinguishability the whole chunk exists for,
and a silent cap reads as "covered everything" when it did not. Each VALUE is
bounded instead (see ``render_value``), so the worst case is bounded without
any entry being dropped. It also never decides WHICH keys are in play -- an
adapter passes the keys that actually bound its render, because a knob that
could not reach the render is not a departure that describes the artifact.

UTF-8, no BOM, ASCII-only. Stdlib only.
"""
from __future__ import annotations

import hashlib
import re

#: The bare mark a measurement run stamps. THE one literal -- both lanes import
#: it rather than spelling it, so the two can never drift apart in the ledger.
PREQUALIFICATION_SUFFIX = "+prequalification"

#: Values longer than this are rendered as a short digest instead of inline.
#: A negative prompt is a render input and a real departure, but it is prose --
#: inlining one would put a paragraph in a ledger field.
MAX_INLINE_VALUE = 24

#: What may appear inline. Anything else (a comma, a bracket, an equals sign,
#: whitespace) would make the suffix unreadable at a glance and unparseable if
#: anyone ever does parse it, so it is digested instead of escaped -- escaping
#: invents a second dialect nobody documented.
_INLINE_SAFE = re.compile(r"^[A-Za-z0-9._:+-]+$")


def render_value(value):
    """One departed value, rendered SHORT, SAFE and DETERMINISTIC.

    Booleans read ``on``/``off`` -- the vocabulary the consent-act knobs already
    use, and the shape the bug report itself asked for. Long or unsafe values
    become ``#<8 hex>`` of their text: two different negatives still differ, and
    the same negative always renders the same way, which is what a sweep needs
    to tell its cells apart. A digest is honest about being a digest; a
    truncation would look like a value."""
    if isinstance(value, bool):
        return "on" if value else "off"
    text = repr(value) if isinstance(value, float) else str(value)
    if len(text) <= MAX_INLINE_VALUE and _INLINE_SAFE.match(text):
        return text
    return "#" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def departures(frozen, resolved):
    """Which of ``resolved`` differ from ``frozen``, as an ORDERED mapping.

    Sorted by key so the same cell always stamps the same string -- a receipt
    whose text depends on dict iteration order cannot be compared between runs,
    which is the one thing a sweep receipt is for.

    Only keys present in ``resolved`` are considered. That is the caller's
    contract and it is load-bearing: an adapter passes the knobs that ACTUALLY
    BOUND its render, so tile geometry that tiled decode never reached is not
    reported as a departure describing a clip it had no effect on.

    A key in ``resolved`` that is absent from ``frozen`` raises: it means the
    recipe and its resolver have drifted apart, and silently reporting it as a
    departure would describe a knob no version of the recipe ever had."""
    unknown = sorted(set(resolved) - set(frozen))
    if unknown:
        raise KeyError(
            "resolved recipe keys %s are not in the frozen recipe -- the "
            "recipe and its resolver have drifted" % (unknown,))
    return {k: resolved[k] for k in sorted(resolved) if resolved[k] != frozen[k]}


def format_suffix(departed):
    """The suffix a measurement clip is stamped with.

    Empty departures still stamp the bare mark: a cell that changed nothing IS
    a prequalification run, and B6's contract is that such a run marks its own
    artifacts. Losing the mark when the list happens to be empty would put a
    sweep clip back into production's namespace on exactly the cell that looks
    most like production."""
    if not departed:
        return PREQUALIFICATION_SUFFIX
    body = ",".join("%s=%s" % (k, render_value(v)) for k, v in departed.items())
    return "%s[%s]" % (PREQUALIFICATION_SUFFIX, body)


def receipt_suffix(frozen, resolved):
    """``departures`` then ``format_suffix`` -- the whole rule in one call, so
    an adapter cannot do half of it."""
    return format_suffix(departures(frozen, resolved))


__all__ = [
    "PREQUALIFICATION_SUFFIX", "MAX_INLINE_VALUE",
    "render_value", "departures", "format_suffix", "receipt_suffix",
]
