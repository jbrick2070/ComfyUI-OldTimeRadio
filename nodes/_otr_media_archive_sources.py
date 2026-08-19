"""Media-archive RSS/Atom source fetcher for the media_archive lane.

This module is the source-side sibling of the science RSS wrapper in
``_otr_source_payload``. It owns media-history feed normalization only: RSS/Atom
entry -> the existing legacy_many_pass source payload shape. No writer imports,
no news_interpreter imports, no ComfyUI imports.
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
from typing import Any

log = logging.getLogger("OTR")

DEFAULT_MEDIA_ARCHIVE_FEEDS: tuple[str, ...] = (
    "https://blogs.loc.gov/now-see-hear/feed/",
    "https://www.filmpreservation.org/blog.atom",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class MediaArchiveSourceError(RuntimeError):
    """A media archive feed could not produce a valid source payload."""


def _clean_text(value: Any, *, max_chars: int = 6000) -> str:
    text = html.unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() or text[:max_chars]
    return text


def _entry_content(entry: Any) -> str:
    for key in ("content",):
        content = entry.get(key) if hasattr(entry, "get") else None
        if isinstance(content, list) and content:
            first = content[0]
            if hasattr(first, "get"):
                val = first.get("value") or first.get("content")
            elif isinstance(first, dict):
                val = first.get("value") or first.get("content")
            else:
                val = first
            cleaned = _clean_text(val)
            if cleaned:
                return cleaned
    for key in ("summary", "description", "subtitle"):
        val = entry.get(key) if hasattr(entry, "get") else None
        cleaned = _clean_text(val)
        if cleaned:
            return cleaned
    return ""


def _feed_title(feed: Any, source_url: str) -> str:
    meta = getattr(feed, "feed", {}) or {}
    title = meta.get("title") if hasattr(meta, "get") else ""
    return _clean_text(title, max_chars=120) or source_url


def _entry_date(entry: Any) -> str:
    for key in ("published", "updated", "created"):
        val = entry.get(key) if hasattr(entry, "get") else None
        cleaned = _clean_text(val, max_chars=80)
        if cleaned:
            return cleaned
    return ""


def _entry_link(entry: Any) -> str:
    for key in ("link", "id"):
        val = entry.get(key) if hasattr(entry, "get") else None
        cleaned = _clean_text(val, max_chars=500)
        if cleaned:
            return cleaned
    return ""


def _source_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _payload_from_entry(entry: Any, *, source_url: str, feed_label: str) -> dict:
    title = _clean_text(entry.get("title") if hasattr(entry, "get") else "",
                        max_chars=240)
    body = _entry_content(entry)
    link = _entry_link(entry)
    if not title or not body or not link:
        raise MediaArchiveSourceError(
            "media archive feed entry missing required title/body/link"
        )
    date = _entry_date(entry)
    summary = body[:800].rsplit(" ", 1)[0].rstrip() or body[:800]
    digest = _source_hash(title, link, date, body)
    full_text = (
        f"{body}\n\n"
        f"Media archive source: {feed_label}\n"
        f"Source URL: {link}\n"
        f"Source hash: {digest}"
    )
    return {
        "headline": title,
        "summary": summary,
        "full_text": full_text,
        "source": feed_label,
        "date": date,
        "link": link,
        "seed_text": f"{title}\n\n{summary}",
    }


def parse_media_archive_feed(
    raw_or_url: str, *, source_url: str = "", strict: bool = False
) -> list[dict]:
    """Parse an RSS/Atom feed document or URL into source-payload dicts.

    Uses ``feedparser`` deliberately: it is already a project dependency and is
    more robust than a small bespoke XML parser for mixed RSS/Atom blog feeds.

    Wave 5: when ``raw_or_url`` is a URL the bytes come from the bounded seam
    (``_otr_feed_fetch``) and feedparser is handed a DOCUMENT. It previously
    received the URL and did its own fetch, which had no timeout, no size cap,
    no redirect cap, no scheme check and no address check. A ``FeedFetchRefused``
    is deliberately NOT wrapped in ``MediaArchiveSourceError``: the caller
    (``fetch_media_archive_rss``) collects that error per feed and carries on to
    the next one, so wrapping a tripped bound would let a misconfigured feed URL
    pass unnoticed whenever another feed happened to succeed.

    The imports are function-local by design -- this module stays import-light
    and free of network imports at module scope.
    """
    try:
        import feedparser
    except Exception as exc:  # noqa: BLE001 -- import environment varies
        raise MediaArchiveSourceError(
            "feedparser is required for media archive RSS parsing"
        ) from exc

    from ._otr_feed_fetch import FeedFetchUnavailable, fetch_feed, looks_like_url

    document = raw_or_url
    if looks_like_url(raw_or_url):
        try:
            document = fetch_feed(raw_or_url).text
        except FeedFetchUnavailable as exc:
            raise MediaArchiveSourceError(
                f"media archive feed could not be fetched: {exc}"
            ) from exc
        source_url = source_url or raw_or_url

    feed = feedparser.parse(document)
    feed_label = _feed_title(feed, source_url or raw_or_url)
    out: list[dict] = []
    for entry in getattr(feed, "entries", []) or []:
        try:
            out.append(_payload_from_entry(
                entry, source_url=source_url or raw_or_url,
                feed_label=feed_label,
            ))
        except MediaArchiveSourceError:
            if strict:
                raise
            continue
    if not out:
        raise MediaArchiveSourceError(
            f"media archive feed produced no usable entries: {source_url or raw_or_url}"
        )
    return out


def _configured_feeds() -> tuple[str, ...]:
    raw = os.environ.get("OTR_MEDIA_ARCHIVE_FEEDS", "").strip()
    if not raw:
        return DEFAULT_MEDIA_ARCHIVE_FEEDS
    vals = [
        p.strip() for p in re.split(r"[;\n]", raw)
        if p.strip()
    ]
    return tuple(vals) or DEFAULT_MEDIA_ARCHIVE_FEEDS


def _configured_index() -> int:
    raw = os.environ.get("OTR_MEDIA_ARCHIVE_ITEM_INDEX", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _explicit_index():
    """The operator's index if they set a USABLE one, else None.

    Distinct from `_configured_index()` on purpose. That helper answers "which
    index should I use", defaulting to 0 and swallowing a bad value -- correct
    for its own callers, and exactly wrong as an "did the operator choose?"
    test. A non-empty but unparseable value must NOT be read as a choice, or the
    override branch fires, the index collapses to 0, and the lane is back to
    adapting the newest post forever with no sign anything went wrong.
    """
    raw = os.environ.get("OTR_MEDIA_ARCHIVE_ITEM_INDEX", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("[media_archive] OTR_MEDIA_ARCHIVE_ITEM_INDEX=%r is not an "
                    "integer; ignoring it and selecting by recent history", raw)
        return None


def _recently_used_urls() -> set:
    """URLs this box has adapted recently, from the SHARED news history.

    THIS IS THE SCIENCE LANE'S MECHANISM, REUSED ON PURPOSE -- not a second
    implementation. `story_orchestrator` already keeps
    `<output>/otr/state/news_history.json`: article URLs with a rolling cap and
    a 5-day TTL, so a headline recycles once it has aged out. It keys on URL and
    nothing about it is science-specific, which is exactly why a second history
    for this lane would be a second thing to drift.

    Best-effort by contract, like the science lane's own use of it: any failure
    returns an empty set and the caller simply selects as it did before. A feed
    lane must never fail an episode because a dedup file was unreadable.
    """
    try:
        try:
            from . import story_orchestrator as _so
        except ImportError:  # pragma: no cover -- flat-import test harnesses
            import story_orchestrator as _so  # type: ignore
        return set(_so._load_news_history() or ())
    except Exception:  # noqa: BLE001 -- dedup is advisory, never a gate
        log.debug("[media_archive] news history unavailable; no dedup",
                  exc_info=True)
        return set()


def _record_used(payload: dict) -> None:
    """Record the selected post in the shared history so the NEXT run skips it.

    Without this the filter above can never do anything -- reading a history
    nothing writes to is a no-op that looks like a feature.
    """
    url = str((payload or {}).get("url") or (payload or {}).get("link") or "")
    if not url:
        return
    try:
        try:
            from . import story_orchestrator as _so
        except ImportError:  # pragma: no cover
            import story_orchestrator as _so  # type: ignore
        _so._record_news_usage(url, str((payload or {}).get("headline") or ""))
    except Exception:  # noqa: BLE001 -- advisory
        log.debug("[media_archive] could not record news usage", exc_info=True)


def fetch_media_archive_rss(*, bank: Any, technical_model: str = "",
                            source_ref: str = "") -> dict:
    """Registered fetcher body for ``media_archive_rss``.

    ``bank``, ``technical_model``, and ``source_ref`` are accepted for the
    shared fetcher contract; RSS feeds ignore ``source_ref``.

    SELECTION, AND WHY IT IS NOT JUST INDEX 0 ANY MORE (PBUG-20260815-06).
    Feed entries arrive newest-first and this returned
    `payloads[_configured_index() % len(payloads)]` with the index defaulting to
    `"0"`, so absent an operator-set env var the lane adapted THE NEWEST POST
    EVERY TIME -- forever, with no dedup, ranking or history anywhere in the
    module. The science lane never had that problem because it filters against a
    shared news history; this lane now uses the SAME one.

    Order of precedence, deliberately:
      1. An explicitly set ``OTR_MEDIA_ARCHIVE_ITEM_INDEX`` still wins outright.
         It is an operator override and dedup must not fight it.
      2. Otherwise prefer entries not in the recent history.
      3. If every entry has been used, fall back to the full list rather than
         failing -- a repeat is worse than nothing, but nothing is worse still.
    """
    del bank, technical_model, source_ref
    payloads: list[dict] = []
    errors: list[str] = []
    for feed_url in _configured_feeds():
        try:
            payloads.extend(parse_media_archive_feed(feed_url, source_url=feed_url))
        except MediaArchiveSourceError as exc:
            errors.append(str(exc))
    if not payloads:
        raise MediaArchiveSourceError(
            "media_archive_rss found no usable feed entries; "
            + "; ".join(errors)
        )

    # AN OVERRIDE ONLY COUNTS IF IT PARSES. A non-empty but unparseable value
    # ("abc", "2.5", "-") is NOT an operator choice, and treating it as one
    # would take this branch, let `_configured_index()` swallow the ValueError,
    # return 0, and silently restore the always-newest behaviour this function
    # exists to end. Caught by review; the first cut tested only for non-empty.
    override = _explicit_index()
    if override is not None:
        chosen = payloads[override % len(payloads)]
        # NOT recorded. An explicit index is a deliberate repeat -- a re-run of
        # one post for testing or for the operator's own reasons -- so writing it
        # to the shared history would let a debugging run consume a headline for
        # the automatic path (and, since the history is shared, for the science
        # lane's TTL window too).
        return chosen

    used = _recently_used_urls()
    fresh = [p for p in payloads
             if str(p.get("url") or p.get("link") or "") not in used]
    pool = fresh or payloads
    if not fresh:
        log.info("[media_archive] every feed entry (%d) is in the recent "
                 "history; re-using the newest rather than failing",
                 len(payloads))
    chosen = pool[_configured_index() % len(pool)]
    _record_used(chosen)
    return chosen


__all__ = [
    "DEFAULT_MEDIA_ARCHIVE_FEEDS",
    "MediaArchiveSourceError",
    "fetch_media_archive_rss",
    "parse_media_archive_feed",
]
