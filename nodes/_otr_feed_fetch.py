"""OTR bounded fetch seam -- the ONE way OTR reaches the network for source text.

Wave 5 of the independent client-authored source banks build.

WHY THIS MODULE EXISTS
----------------------
Before this seam there were THREE separate network hops, each with its own
(mostly absent) idea of a bound:

  * ``story_orchestrator._fetch_full_article`` -- ``requests.get(timeout=20)``
    with no size cap, no scheme check and no address reject list. A redirect to
    ``http://169.254.169.254/`` was a perfectly legal fetch.
  * ``story_orchestrator._fetch_single_feed`` -- ``feedparser.parse(url)``,
    which performs its OWN unbounded urllib fetch, wrapped in a PROCESS-GLOBAL
    ``socket.setdefaulttimeout(7)`` that a thread pool of ~30 feeds raced.
  * ``_otr_media_archive_sources.parse_media_archive_feed`` -- the same
    ``feedparser.parse(url)`` hop with no bound at all.

The r3 kibitz finding that network hardening is NOT INHERITED is the whole
point of the wave: hardening one caller leaves the others exactly as exposed,
and a client-authored bank writing its own ``fetch_source`` would start again
from zero. So the bound lives HERE, once, and every hop -- shipped or
client-authored -- goes through it.

THE TWO FAILURE CLASSES (this distinction IS the contract)
----------------------------------------------------------
``FeedFetchRefused``     -- OUR bound tripped: non-https scheme, a loopback /
                            private / link-local address, too many redirects,
                            an oversized body, an unusable content encoding, or
                            the wrong media type. The URL or the configuration
                            is wrong, or something is trying to walk us
                            somewhere we do not go. LOUD, never retried, and no
                            caller may swallow it.
``FeedFetchUnavailable`` -- the REMOTE did not deliver: DNS failure, refused
                            connection, timeout, 4xx/5xx after the retries.
                            An ordinary per-URL miss. A caller holding other
                            candidates MAY catch this one and move on -- that
                            is the behaviour the article scraper already had,
                            and it keeps it.

Collapsing the two would either let one paywalled article kill a run, or make a
redirect into the private network look like a paywall. Neither is acceptable.

The module is stdlib-only ON PURPOSE: a client bundle may import it without
adding a single dependency, and activation-time import of a client module never
drags in ``requests`` or ``feedparser``.
"""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import logging
import re
import socket
import ssl
import time
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

log = logging.getLogger(__name__)

# ---- the bounds (operator-specified, wave 5) -------------------------------

#: Only https. http:// is refused rather than upgraded -- a silent upgrade
#: would hide a misconfigured feed row from the operator who wrote it.
ALLOWED_SCHEME = "https"

CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 10.0
MAX_REDIRECTS = 3
MAX_DECODED_BYTES = 2 * 1024 * 1024      # 2 MiB, AFTER content-encoding decode
MAX_RETRIES = 2                          # 2 retries == 3 attempts total
TOTAL_DEADLINE_S = 25.0                  # one monotonic deadline for everything

#: Kept byte-identical to the User-Agent the article scraper shipped with.
#: Several science publishers 403 an unfamiliar agent, so changing this string
#: is a live-behaviour change, not a cosmetic one.
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; OTR-ScriptBot/1.0)"

KIND_FEED = "feed"
KIND_ARTICLE = "article"

#: A feed is RSS/Atom/XML; an article is HTML. Anything else means the URL does
#: not point at what the caller believes it points at -- refuse rather than
#: hand a writer a login page or a PDF and let it "interpret" the bytes.
_FEED_MEDIA_TYPES = frozenset({
    "application/rss+xml", "application/atom+xml", "application/rdf+xml",
    "application/xml", "text/xml",
})
_ARTICLE_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

_XML_DECL_RE = re.compile(
    rb"""<\?xml[^>]*?encoding\s*=\s*["']([A-Za-z0-9_.:+-]+)["']""")
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]*?charset\s*=\s*["']?([A-Za-z0-9_.:+-]+)""", re.IGNORECASE)


# ---- failure classes -------------------------------------------------------

class FeedFetchError(RuntimeError):
    """Base class. Carries the url and a stable machine-readable reason code."""

    def __init__(self, url: str, reason: str, detail: str) -> None:
        super().__init__(f"{detail} [reason={reason}] [url={url}]")
        self.url = url
        self.reason = reason
        self.detail = detail


class FeedFetchRefused(FeedFetchError):
    """A bound of OURS tripped. Never retried, never swallowed, always loud."""


class FeedFetchUnavailable(FeedFetchError):
    """The remote did not deliver. A caller with another candidate may move on.

    ``retryable`` says whether trying the SAME url again could plausibly help.
    A 503 or a dropped connection: yes. A 404, or an exhausted deadline: no --
    re-sending those burns the remaining budget to learn what we already know.
    """

    def __init__(self, url: str, reason: str, detail: str,
                 *, retryable: bool = True) -> None:
        super().__init__(url, reason, detail)
        self.retryable = retryable


@dataclass(frozen=True)
class FetchedDocument:
    """One bounded fetch result. ``text`` is already decoded."""

    requested_url: str
    final_url: str
    text: str
    encoding: str
    media_type: str
    byte_count: int
    redirects: int
    elapsed_s: float


# ---- the monotonic deadline ------------------------------------------------

class _Deadline:
    """One monotonic budget for the whole call: retries and redirects included.

    ``time.monotonic`` and not ``time.time`` deliberately -- a clock step must
    not extend or collapse a network budget.
    """

    def __init__(self, url: str, total_s: float) -> None:
        self._url = url
        self._start = time.monotonic()
        self._total = float(total_s)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def remaining(self) -> float:
        return self._total - self.elapsed

    def slice_for(self, per_op_s: float) -> float:
        """The timeout for one socket operation, clipped by what is left."""
        left = self.remaining()
        if left <= 0.0:
            raise FeedFetchUnavailable(
                self._url, "deadline",
                f"exceeded the {self._total:.0f}s fetch deadline",
                retryable=False,
            )
        return min(per_op_s, left)


# ---- address policy --------------------------------------------------------

def _reject_reason(ip) -> str:
    """The name of the rule an address breaks, or "" when it is routable.

    Checked in a fixed order so the message names the most specific rule. A
    v4-mapped v6 address is unwrapped first: ``::ffff:127.0.0.1`` is loopback
    no matter which family carries it.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_private:
        return "private"
    if ip.is_reserved:
        return "reserved"
    return ""


def _resolve_public(url: str, host: str, port: int, deadline: _Deadline) -> list:
    """Resolve ``host`` and refuse unless EVERY answer is routable.

    Every answer, not just the one we pick: a name that resolves to one public
    and one private address is a DNS-rebinding shape, and there is no legitimate
    source feed that needs it.
    """
    deadline.slice_for(CONNECT_TIMEOUT_S)  # raises when the budget is gone
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FeedFetchUnavailable(
            url, "dns", f"could not resolve {host!r}: {exc}") from exc
    if not infos:
        raise FeedFetchUnavailable(url, "dns", f"{host!r} resolved to nothing")
    for info in infos:
        raw = info[4][0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError as exc:  # pragma: no cover -- getaddrinfo is literal
            raise FeedFetchUnavailable(
                url, "dns", f"{host!r} resolved to unparsable {raw!r}") from exc
        reason = _reject_reason(ip)
        if reason:
            raise FeedFetchRefused(
                url, "blocked_address",
                f"{host!r} resolves to {raw} which is {reason}; OTR fetches "
                f"public addresses only",
            )
    return infos


def _connect(url: str, infos: list, host: str, deadline: _Deadline):
    """A TLS socket to one of ``infos``, pinned to the address we validated.

    The socket is connected to the VALIDATED address and only then wrapped with
    ``server_hostname=host``, so certificate verification still checks the name
    while the connection cannot be re-pointed by a second DNS answer between
    the check and the connect.
    """
    context = ssl.create_default_context()
    last: OSError | None = None
    for family, socktype, proto, _canon, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(deadline.slice_for(CONNECT_TIMEOUT_S))
            sock.connect(sockaddr)
            sock.settimeout(deadline.slice_for(READ_TIMEOUT_S))
            return context.wrap_socket(sock, server_hostname=host)
        except (OSError, ssl.SSLError) as exc:
            sock.close()
            last = exc
            continue
    raise FeedFetchUnavailable(
        url, "connect", f"could not connect to {host!r}: {last}")


# ---- body: read bounded, decode bounded, decode to text --------------------

def _read_bounded(url: str, resp, deadline: _Deadline) -> bytes:
    """Read the body, refusing the moment it crosses the cap.

    The cap is enforced DURING the read, not after: a caller that only checks
    the length afterwards has already put the bytes in memory.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        deadline.slice_for(READ_TIMEOUT_S)
        try:
            chunk = resp.read(65536)
        except (OSError, http.client.HTTPException) as exc:
            raise FeedFetchUnavailable(
                url, "read", f"body read failed: {exc}") from exc
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_DECODED_BYTES:
            raise FeedFetchRefused(
                url, "body_too_large",
                f"body exceeded the {MAX_DECODED_BYTES} byte cap",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_content_encoding(url: str, raw: bytes, encoding: str) -> bytes:
    """Undo Content-Encoding under the SAME cap the raw read used.

    The request asks for ``identity``; this exists for servers that send an
    encoding anyway. Decompression is bounded because the cap the operator
    specified is on the DECODED bytes -- 2 MiB of gzip is not 2 MiB of feed.
    """
    enc = (encoding or "identity").strip().lower()
    if enc in ("", "identity"):
        return raw
    if enc == "gzip":
        wbits_options = (31,)
    elif enc == "deflate":
        wbits_options = (15, -15)   # zlib-wrapped, then raw: both are seen live
    else:
        raise FeedFetchRefused(
            url, "content_encoding",
            f"unsupported Content-Encoding {enc!r}",
        )
    last: Exception | None = None
    for wbits in wbits_options:
        obj = zlib.decompressobj(wbits)
        try:
            out = obj.decompress(raw, MAX_DECODED_BYTES + 1)
        except zlib.error as exc:
            last = exc
            continue
        if obj.unconsumed_tail or len(out) > MAX_DECODED_BYTES:
            raise FeedFetchRefused(
                url, "body_too_large",
                f"{enc} body decoded past the {MAX_DECODED_BYTES} byte cap",
            )
        try:
            out += obj.flush()
        except zlib.error as exc:  # pragma: no cover -- truncated stream
            last = exc
            continue
        if len(out) > MAX_DECODED_BYTES:
            raise FeedFetchRefused(
                url, "body_too_large",
                f"{enc} body decoded past the {MAX_DECODED_BYTES} byte cap",
            )
        return out
    raise FeedFetchUnavailable(
        url, "content_encoding", f"{enc} body would not decode: {last}")


def _detect_charset(raw: bytes, header_charset: str, media_type: str) -> str:
    """Header charset, else BOM, else the document's own declaration, else utf-8."""
    if header_charset:
        return header_charset
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    head = raw[:1024]
    is_xml = media_type.endswith("+xml") or media_type in _FEED_MEDIA_TYPES
    match = (_XML_DECL_RE.search(head) if is_xml
             else _META_CHARSET_RE.search(head))
    if match:
        return match.group(1).decode("ascii", "ignore") or "utf-8"
    return "utf-8"


def _decode_text(url: str, raw: bytes, charset: str) -> tuple[str, str]:
    """Decode strictly; fall back to utf-8/replace only with a LOUD warning.

    A replacement-character fallback is a real loss of fidelity, so it is never
    silent -- but it is not fatal either: a feed with three bad bytes in one
    entry is still a usable feed, and refusing it would fail a story for an
    encoding defect the publisher owns.
    """
    try:
        return raw.decode(charset), charset
    except (UnicodeDecodeError, LookupError) as exc:
        log.warning(
            "[OTR fetch] %s did not decode as %r (%s); falling back to "
            "utf-8 with replacement characters", url, charset, exc,
        )
        return raw.decode("utf-8", errors="replace"), "utf-8/replace"


# ---- one request -----------------------------------------------------------

def _require_https(url: str) -> tuple[str, int, str]:
    """Split a url, refusing anything that is not https with a host."""
    parts = urlsplit(url)
    if parts.scheme.lower() != ALLOWED_SCHEME:
        raise FeedFetchRefused(
            url, "scheme",
            f"scheme {parts.scheme or '(none)'!r} is not https; OTR does not "
            f"fetch over plaintext and does not silently upgrade",
        )
    if not parts.hostname:
        raise FeedFetchRefused(url, "no_host", "url carries no host")
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return parts.hostname, parts.port or 443, path


def _request_once(url: str, kind: str, user_agent: str, deadline: _Deadline):
    """One https GET. Returns ``("redirect", location)`` or ``("body", doc_parts)``."""
    host, port, path = _require_https(url)
    infos = _resolve_public(url, host, port, deadline)
    sock = _connect(url, infos, host, deadline)
    conn = http.client.HTTPSConnection(host, port)
    conn.sock = sock                      # already connected AND already TLS
    try:
        conn.request("GET", path, headers={
            "User-Agent": user_agent,
            "Accept": ("application/rss+xml, application/atom+xml, "
                       "application/xml;q=0.9, text/xml;q=0.9, */*;q=0.1")
                      if kind == KIND_FEED else
                      ("text/html, application/xhtml+xml;q=0.9, */*;q=0.1"),
            "Accept-Encoding": "identity",
            "Connection": "close",
        })
        resp = conn.getresponse()
        status = resp.status

        if status in _REDIRECT_STATUS:
            location = resp.getheader("Location") or ""
            if not location:
                raise FeedFetchUnavailable(
                    url, "redirect", f"HTTP {status} carried no Location",
                    retryable=False)
            return "redirect", urljoin(url, location)

        if status != 200:
            # A 404 or a 403 is settled; only the transient statuses are worth
            # a second request against the same url.
            raise FeedFetchUnavailable(
                url, "http_status", f"HTTP {status} {resp.reason}",
                retryable=status in _RETRYABLE_STATUS)

        media_type = (resp.headers.get_content_type() or "").lower()
        allowed = _FEED_MEDIA_TYPES if kind == KIND_FEED else _ARTICLE_MEDIA_TYPES
        ok = media_type in allowed or (
            kind == KIND_FEED and media_type.endswith("+xml"))
        if not ok:
            raise FeedFetchRefused(
                url, "media_type",
                f"served {media_type or '(none)'!r} where a {kind} must be "
                f"one of {sorted(allowed)}",
            )

        raw = _read_bounded(url, resp, deadline)
        raw = _decode_content_encoding(
            url, raw, resp.getheader("Content-Encoding") or "")
        charset = _detect_charset(
            raw, (resp.headers.get_content_charset() or ""), media_type)
        text, used = _decode_text(url, raw, charset)
        return "body", (text, used, media_type, len(raw))
    except (OSError, http.client.HTTPException) as exc:
        # Our own failures are RuntimeError subclasses and pass straight
        # through; this only catches genuine transport faults.
        raise FeedFetchUnavailable(
            url, "transport", f"request failed: {exc}") from exc
    finally:
        conn.close()


# ---- the public seam -------------------------------------------------------

def fetch_bounded(url: str, *, kind: str, deadline_s: float | None = None,
                  user_agent: str | None = None) -> FetchedDocument:
    """Fetch ``url`` under every bound in this module. See the module docstring.

    Raises ``FeedFetchRefused`` when a bound of ours trips (loud, terminal) and
    ``FeedFetchUnavailable`` when the remote simply did not deliver.
    """
    if kind not in (KIND_FEED, KIND_ARTICLE):
        raise ValueError(f"kind must be {KIND_FEED!r} or {KIND_ARTICLE!r}")
    requested = str(url or "").strip()
    if not requested:
        raise FeedFetchRefused("", "empty_url", "no url was given")

    budget = TOTAL_DEADLINE_S if deadline_s is None else float(deadline_s)
    deadline = _Deadline(requested, min(budget, TOTAL_DEADLINE_S))
    agent = user_agent or DEFAULT_USER_AGENT

    last: FeedFetchUnavailable | None = None
    for attempt in range(MAX_RETRIES + 1):
        current = requested
        try:
            for hop in range(MAX_REDIRECTS + 1):
                what, payload = _request_once(current, kind, agent, deadline)
                if what == "body":
                    text, used, media_type, count = payload
                    return FetchedDocument(
                        requested_url=requested, final_url=current, text=text,
                        encoding=used, media_type=media_type,
                        byte_count=count, redirects=hop,
                        elapsed_s=deadline.elapsed,
                    )
                if hop == MAX_REDIRECTS:
                    raise FeedFetchRefused(
                        requested, "too_many_redirects",
                        f"more than {MAX_REDIRECTS} redirects, last to "
                        f"{payload}",
                    )
                current = payload
        except FeedFetchRefused:
            raise                       # a tripped bound is never retried
        except FeedFetchUnavailable as exc:
            last = exc
            if (not exc.retryable or attempt == MAX_RETRIES
                    or deadline.remaining() <= 0.0):
                break
            pause = min(0.5 * (attempt + 1), max(deadline.remaining(), 0.0))
            log.debug("[OTR fetch] %s attempt %d/%d failed (%s); retrying",
                      requested, attempt + 1, MAX_RETRIES + 1, exc.reason)
            time.sleep(pause)
    if last is None:  # pragma: no cover -- the loop only exits via break/return
        raise FeedFetchUnavailable(
            requested, "no_attempt", "the fetch loop made no attempt")
    raise last


def fetch_feed(url: str, *, deadline_s: float | None = None,
               user_agent: str | None = None) -> FetchedDocument:
    """An RSS/Atom/XML feed document, bounded. Never hands bytes to a parser
    that would fetch them itself."""
    return fetch_bounded(url, kind=KIND_FEED, deadline_s=deadline_s,
                         user_agent=user_agent)


def fetch_article(url: str, *, deadline_s: float | None = None,
                  user_agent: str | None = None) -> FetchedDocument:
    """One HTML article body, bounded."""
    return fetch_bounded(url, kind=KIND_ARTICLE, deadline_s=deadline_s,
                         user_agent=user_agent)


def looks_like_url(value: str) -> bool:
    """True when ``value`` is a url to fetch rather than a document to parse.

    ``http://`` counts: the seam must SEE it and refuse it loudly, instead of
    the caller quietly treating a plaintext url as an inline document.
    """
    return str(value or "").strip().lower().startswith(("http://", "https://"))


__all__ = [
    "ALLOWED_SCHEME",
    "CONNECT_TIMEOUT_S",
    "DEFAULT_USER_AGENT",
    "FeedFetchError",
    "FeedFetchRefused",
    "FeedFetchUnavailable",
    "FetchedDocument",
    "KIND_ARTICLE",
    "KIND_FEED",
    "MAX_DECODED_BYTES",
    "MAX_REDIRECTS",
    "MAX_RETRIES",
    "READ_TIMEOUT_S",
    "TOTAL_DEADLINE_S",
    "fetch_article",
    "fetch_bounded",
    "fetch_feed",
    "looks_like_url",
]
