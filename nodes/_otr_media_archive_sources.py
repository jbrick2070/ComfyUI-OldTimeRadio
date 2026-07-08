"""Media-archive RSS/Atom source fetcher for the media_archive lane.

This module is the source-side sibling of the science RSS wrapper in
``_otr_source_payload``. It owns media-history feed normalization only: RSS/Atom
entry -> the existing legacy_many_pass source payload shape. No writer imports,
no news_interpreter imports, no ComfyUI imports.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
from typing import Any

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
    """
    try:
        import feedparser
    except Exception as exc:  # noqa: BLE001 -- import environment varies
        raise MediaArchiveSourceError(
            "feedparser is required for media archive RSS parsing"
        ) from exc

    feed = feedparser.parse(raw_or_url)
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


def fetch_media_archive_rss(*, bank: Any, technical_model: str = "",
                            source_ref: str = "") -> dict:
    """Registered fetcher body for ``media_archive_rss``.

    ``bank``, ``technical_model``, and ``source_ref`` are accepted for the
    shared fetcher contract. RSS feeds ignore ``source_ref``; the feed choice
    itself is deterministic: gather usable entries from configured feeds and
    select ``OTR_MEDIA_ARCHIVE_ITEM_INDEX`` modulo the entry count.
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
    return payloads[_configured_index() % len(payloads)]


__all__ = [
    "DEFAULT_MEDIA_ARCHIVE_FEEDS",
    "MediaArchiveSourceError",
    "fetch_media_archive_rss",
    "parse_media_archive_feed",
]
