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


def _find_latest_ledger_with_news_seed(
    max_lookback: int = 8,
) -> Optional[dict]:
    """BUG-LOCAL-294 (2026-05-27): walk the N most-recent ledgers
    backward looking for one with a populated `meta.news_seed`.

    Background (writers'-room context, Wave 2 Agent F): NewsFetcher
    and Writer are separate ComfyUI nodes, each instantiating their
    own Ledger() with a time.time()-based episode_id. When they
    land on different seconds (common -- the LLM-load latency
    between fetch and writer-start is often >=1 sec), NewsFetcher
    stamps news_seed into `pending_<T>_ledger` and Writer saves a
    fresh skeleton into `pending_<T+1>_ledger` WITHOUT carrying
    the news_seed forward. The writers'-room downstream nodes
    (Wave 1 Agent D / Wave 2 Agent F / Wave 2 Agent G) then run on
    the latest ledger (T+1) which lacks news_seed.

    This walker scans the N most-recent ledgers ordered by mtime
    descending and returns the first one whose `meta.news_seed` is
    non-empty. The lookback is bounded so we don't drift to a
    months-old episode -- N=8 covers normal NewsFetcher + Writer
    pair-saves plus any retry / restart noise within the same run.

    Returns the ledger dict (with `meta.news_seed` confirmed
    populated), or None if no recent ledger carries one.

    Never raises.
    """
    try:
        try:
            from . import _otr_paths as _P
        except ImportError:
            import _otr_paths as _P  # type: ignore
        episodes_root = _P.otr_episodes_root()
    except Exception:  # noqa: BLE001
        return None
    try:
        candidates = sorted(
            episodes_root.glob("*/audio/*_ledger.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:max_lookback]
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[WritersRoomResolver] news_seed walker: glob/sort "
            "raised %s -- giving up.", type(exc).__name__,
        )
        return None
    import json as _j
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                ledger = _j.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(ledger, dict):
            continue
        meta = ledger.get("meta") or {}
        seed = meta.get("news_seed")
        # A real news_seed is either a non-empty string or a dict
        # with at least one non-empty value.
        has_seed = False
        if isinstance(seed, str) and seed.strip():
            has_seed = True
        elif isinstance(seed, dict) and any(
            (str(v or "").strip() for v in seed.values())
        ):
            has_seed = True
        if has_seed:
            log.info(
                "[WritersRoomResolver] news_seed walker found "
                "populated seed on %s (after scanning %d ledger(s)).",
                path, candidates.index(path) + 1,
            )
            return ledger
    log.warning(
        "[WritersRoomResolver] news_seed walker scanned %d ledger(s) "
        "under %s with no populated meta.news_seed -- giving up.",
        len(candidates), episodes_root,
    )
    return None


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


def _extract_seed_from_ledger(ledger: dict) -> str:
    """Pull a usable news-seed string from the writer's ledger.

    PRIMARY source: `meta.news` (the LLM-summarized briefs the
    writer's `build_news_briefs` call already produced --
    `script_brief` + `scene_atmosphere` + `casting_brief` + etc.).
    This is the DELIBERATE writer-curated summary; the
    writers'-room director (Wave 1 Agent D) should always prefer
    it over the raw RSS dump because it has already been condensed
    into prose ready for downstream consumption.

    FALLBACK source: `meta.news_seed` (NewsFetcher's raw RSS
    headline + body). Only used when `meta.news` is missing or
    empty -- typically when `build_news_briefs` exhausted its
    retry ladder and stamped `meta["news"] = None`.

    Returns the best string available; never raises.
    """
    if not isinstance(ledger, dict):
        return ""
    meta = ledger.get("meta") or {}

    # PRIMARY: LLM-summarized news briefs (build_news_briefs
    # output stamped on meta["news"] -- see OTR_LedgerScriptWriter
    # line 2225). script_brief is the concise prose summary the
    # writers'-room director can ground on; scene_atmosphere adds
    # setting flavor.
    briefs = meta.get("news") or {}
    if isinstance(briefs, dict):
        script_brief = str(briefs.get("script_brief") or "").strip()
        scene_atmosphere = str(
            briefs.get("scene_atmosphere") or "",
        ).strip()
        if script_brief and scene_atmosphere:
            return f"{script_brief}\n\n{scene_atmosphere}"
        if script_brief:
            return script_brief
        if scene_atmosphere:
            return scene_atmosphere

    # FALLBACK: raw NewsFetcher news_seed dict / string. Used only
    # when the LLM briefing pass failed (briefs missing / empty).
    seed = meta.get("news_seed")
    if isinstance(seed, dict):
        headline = str(seed.get("headline") or "").strip()
        body = str(
            seed.get("body") or seed.get("source") or "",
        ).strip()
        if headline and body:
            return f"{headline}\n\n{body}"
        if headline:
            return headline
        if body:
            return body
    elif isinstance(seed, str) and seed.strip():
        return seed.strip()

    return ""


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

    # Try the in-flight / latest ledger first.
    resolved = _extract_seed_from_ledger(ledger or {})
    if resolved:
        log.info(
            "[WritersRoomResolver] resolved news_seed from in-flight "
            "ledger (widget was empty; %d chars).",
            len(resolved),
        )
        return resolved

    # BUG-LOCAL-294 cross-ledger walker: when the writer's own
    # ledger lacks both meta.news and meta.news_seed (e.g.
    # NewsFetcher and Writer raced on different episode_ids -- one
    # second drift puts them in different pending_* dirs and the
    # Writer's skeleton save doesn't carry the seed forward), scan
    # the N most-recent ledgers for one that DOES carry it. The
    # NewsFetcher-only ledger from the same queue's prior second
    # typically does.
    older_ledger = _find_latest_ledger_with_news_seed()
    if older_ledger is not None:
        resolved = _extract_seed_from_ledger(older_ledger)
        if resolved:
            log.info(
                "[WritersRoomResolver] BUG-LOCAL-294 cross-ledger "
                "walker resolved news_seed (widget was empty; "
                "%d chars).",
                len(resolved),
            )
            return resolved

    return ""
