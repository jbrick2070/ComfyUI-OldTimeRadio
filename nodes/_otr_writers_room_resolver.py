"""nodes/_otr_writers_room_resolver.py -- shared fallback resolver.

When the operator wants to run the writers' room end-to-end WITHOUT
manually typing cast names or news seeds into the Director / Story
Room / Extract widgets, this module reads them from the in-flight
ledger the writer just stamped.

The writer node always stamps:
  * meta.news_seed (dict or string, BUG-LOCAL-277 normalization)
  * cast[*] rows with name / char_id

These are the same facts the writer just decided. Pulling them
from the ledger is more reliable than asking the operator to type
them again -- and it works automatically with `commit=True` end-
to-end runs.

PURE: no torch, no Comfy. Just stdlib + in-flight ledger reader.
PD1: never raises; any failure returns a safe empty default.
"""
from __future__ import annotations

import logging
from typing import List, Optional


log = logging.getLogger("OTR")


__all__ = [
    "resolve_cast_names",
    "resolve_news_seed",
]


# Reserved speaker names that should NEVER appear in the writers'-
# room cast list -- announcer rides the bookend bus separately, music
# is a slot kind not a character.
_RESERVED_SPEAKERS: frozenset[str] = frozenset({
    "ANNOUNCER", "MUSIC", "announcer", "music",
})


def _load_in_flight_ledger() -> Optional[dict]:
    """Return the parsed in-flight ledger dict, or None on any miss.

    Never raises.
    """
    try:
        from . import _otr_ledger as _OTRL
    except ImportError:
        return None
    try:
        led_path = _OTRL.in_flight_ledger_path()
    except Exception:  # noqa: BLE001
        return None
    if led_path is None:
        return None
    try:
        return _OTRL.load_ledger_safe(led_path)
    except Exception:  # noqa: BLE001
        return None


def resolve_cast_names(
    widget_value: str,
    *,
    fallback_ledger: Optional[dict] = None,
) -> List[str]:
    """Parse cast names from the operator widget; fall back to the
    in-flight ledger when the widget is empty.

    Args:
        widget_value: the comma-separated string the operator typed.
            When non-empty, this is the source of truth.
        fallback_ledger: optional pre-loaded ledger dict (tests use
            this; production passes None and the resolver loads via
            the in-flight singleton).

    Returns:
        List of cast names with the announcer + music reserved
        speakers filtered out. Empty list when nothing resolves --
        the consuming node decides whether that's a hard failure
        or a soft degrade.
    """
    rows = [
        s.strip() for s in (widget_value or "").split(",")
        if s.strip()
    ]
    if rows:
        return [n for n in rows if n not in _RESERVED_SPEAKERS]

    ledger = fallback_ledger
    if ledger is None:
        ledger = _load_in_flight_ledger()
    if not ledger or not isinstance(ledger, dict):
        return []

    cast = ledger.get("cast") or []
    if not isinstance(cast, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for row in cast:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or name in _RESERVED_SPEAKERS:
            continue
        char_id = str(row.get("char_id") or "").strip()
        # Belt-and-braces -- some pipelines stamp char_id='announcer'
        # but the name is something else; filter both.
        if char_id and char_id.lower() in _RESERVED_SPEAKERS:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if out:
        log.info(
            "[WritersRoomResolver] resolved %d cast name(s) from "
            "in-flight ledger (widget was empty): %s",
            len(out), ", ".join(out),
        )
    return out


def resolve_news_seed(
    widget_value: str,
    *,
    fallback_ledger: Optional[dict] = None,
) -> str:
    """Resolve the news seed prose.

    Returns the widget value when non-empty. Otherwise reads from
    `meta.news_seed` on the in-flight ledger -- the writer stamps
    this as either a dict (`headline` + `body_chars` + `source`...)
    or a plain string per BUG-LOCAL-277. Both shapes are handled.

    Returns "" when neither resolves -- the consuming node falls
    back to its own placeholder text.
    """
    if (widget_value or "").strip():
        return widget_value.strip()

    ledger = fallback_ledger
    if ledger is None:
        ledger = _load_in_flight_ledger()
    if not ledger or not isinstance(ledger, dict):
        return ""

    meta = ledger.get("meta") or {}
    seed = meta.get("news_seed")
    if isinstance(seed, dict):
        # BUG-LOCAL-277 normalization: prefer headline + body shape.
        headline = str(seed.get("headline") or "").strip()
        body = str(
            seed.get("body") or seed.get("source") or "",
        ).strip()
        if headline and body:
            resolved = f"{headline}\n\n{body}"
        elif headline:
            resolved = headline
        elif body:
            resolved = body
        else:
            resolved = ""
    elif isinstance(seed, str):
        resolved = seed.strip()
    else:
        resolved = ""

    if resolved:
        log.info(
            "[WritersRoomResolver] resolved news_seed from in-flight "
            "ledger (widget was empty; %d chars).",
            len(resolved),
        )
    return resolved
