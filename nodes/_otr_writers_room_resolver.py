"""nodes/_otr_writers_room_resolver.py -- shared fallback resolver.

2026-05-27 evening hotfix: even when `in_flight_ledger_path()` returns
None (the writer's singleton may not survive across node executions
inside one ComfyUI queue), the resolver now does its OWN mtime walk
of `output/otr/episodes/*/audio/*_ledger.json` as a hard guarantee.
Live evidence (writers'-room run pending_20260527_203427): Director got
cast=0 / news_seed=0 chars despite the ledger on disk having
cast=['ANNOUNCER','JOHN BEESLY','MINA TANAKA'] -- the singleton
lookup silently bailed and there was no second-tier fallback.

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

    Never raises. Logs WARN at every miss so a live operator soak
    can tell which fallback step bailed out (the production runs
    on 2026-05-27 found 0/0 cast/seed because the singleton lookup
    silently returned None and the silent failure looked like the
    resolver was a no-op).
    """
    try:
        from . import _otr_ledger as _OTRL
    except ImportError as exc:
        log.warning(
            "[WritersRoomResolver] _otr_ledger import failed: %s -- "
            "cannot auto-resolve from in-flight ledger.", exc,
        )
        return None
    try:
        led_path = _OTRL.in_flight_ledger_path()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] in_flight_ledger_path() raised "
            "%s: %s -- cannot auto-resolve.",
            type(exc).__name__, str(exc)[:200],
        )
        return None
    if led_path is None:
        log.warning(
            "[WritersRoomResolver] in_flight_ledger_path() returned "
            "None -- no in-flight ledger to read; auto-resolve "
            "yielding empty defaults.",
        )
        return None
    try:
        ledger = _OTRL.load_ledger_safe(led_path)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] load_ledger_safe(%s) raised "
            "%s: %s -- auto-resolve yielding empty defaults.",
            led_path, type(exc).__name__, str(exc)[:200],
        )
        return None
    if ledger is None:
        log.warning(
            "[WritersRoomResolver] load_ledger_safe(%s) returned "
            "None -- auto-resolve yielding empty defaults.",
            led_path,
        )
        return None
    log.info(
        "[WritersRoomResolver] loaded in-flight ledger from %s "
        "(cast=%d lines=%d).",
        led_path,
        len(ledger.get("cast") or []),
        len(ledger.get("lines") or []),
    )
    return ledger


def _load_latest_ledger_by_mtime() -> Optional[dict]:
    """Hard-guarantee fallback: walk `output/otr/episodes/*/audio/
    *_ledger.json` directly and load the newest by mtime.

    Used when `in_flight_ledger_path()` returns None inside the
    ComfyUI cascade (the writer's singleton may not survive across
    node executions). PURE stdlib: glob + max + json.load.
    """
    try:
        try:
            from . import _otr_paths as _P
        except ImportError:
            import _otr_paths as _P  # type: ignore
        episodes_root = _P.otr_episodes_root()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] hard-fallback: episodes_root "
            "lookup raised %s: %s -- giving up.",
            type(exc).__name__, str(exc)[:200],
        )
        return None
    try:
        candidates = list(
            episodes_root.glob("*/audio/*_ledger.json")
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] hard-fallback: glob raised %s "
            "in %s -- giving up.",
            type(exc).__name__, str(exc)[:200],
        )
        return None
    if not candidates:
        log.warning(
            "[WritersRoomResolver] hard-fallback: no ledgers found "
            "under %s -- giving up.", episodes_root,
        )
        return None
    try:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] hard-fallback: mtime sort raised "
            "%s -- giving up.", type(exc).__name__,
        )
        return None
    try:
        import json as _j
        with open(newest, "r", encoding="utf-8") as fh:
            ledger = _j.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] hard-fallback: parse failed on "
            "%s: %s -- giving up.", newest, str(exc)[:200],
        )
        return None
    log.info(
        "[WritersRoomResolver] hard-fallback loaded ledger from "
        "%s (cast=%d lines=%d).",
        newest,
        len(ledger.get("cast") or []),
        len(ledger.get("lines") or []),
    )
    return ledger


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
    if ledger is None or not isinstance(ledger, dict) or not ledger:
        # Singleton was unavailable; try the hard mtime fallback
        # before giving up.
        ledger = _load_latest_ledger_by_mtime()
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
    if ledger is None or not isinstance(ledger, dict) or not ledger:
        ledger = _load_latest_ledger_by_mtime()
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
