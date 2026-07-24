"""Wave 5: the bounded `_otr_feed_fetch` seam, both hops.

Every test here is OFFLINE. The seam refuses loopback by design, so a local
test server is not available to us -- that is a real consequence of the address
policy, not an oversight. The transport-level tests therefore drive
`_request_once` with a fake connection, which is the correct level anyway: it
exercises the status / redirect / media-type / size / charset decisions without
depending on anyone's network.

The last class is the important one. The r3 finding is that network hardening
is NOT INHERITED -- hardening one caller leaves every other hop exactly as
exposed. `TestEveryHopRoutesThroughTheSeam` is the machine-checkable version of
that finding: it fails if any hop grows its own unbounded fetch again.
"""
from __future__ import annotations

import ast
import email.message
import ipaddress
import pathlib
import zlib

import pytest

from nodes import _otr_feed_fetch as ff

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Captured before any fixture patches it, so a test can put the REAL address
#: policy back while keeping the fake transport.
_real_resolve_public = ff._resolve_public


# ---- AST helpers -----------------------------------------------------------
#
# The call-site guards below read the AST rather than grepping the text. The
# first version of this file grepped, and it failed against the comments that
# explain WHICH unbounded call was removed -- the guard cannot be allowed to
# fight the documentation of the thing it guards.

def _tree(module_name: str) -> ast.Module:
    return ast.parse(
        (REPO / "nodes" / module_name).read_text(encoding="utf-8"))


def _method_calls(tree: ast.Module, owner: str, attr: str) -> list:
    """Every ``owner.attr(...)`` call in the module."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr and isinstance(n.func.value, ast.Name)
            and n.func.value.id == owner]


def _plain_calls(tree: ast.Module, name: str) -> list:
    """Every bare ``name(...)`` call in the module."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


# ---- fake transport --------------------------------------------------------

class _FakeResponse:
    def __init__(self, status, body=b"", content_type="application/rss+xml",
                 headers=None, reason="OK"):
        self.status = status
        self.reason = reason
        self._body = body
        self._offset = 0
        msg = email.message.Message()
        if content_type is not None:
            msg["Content-Type"] = content_type
        for key, value in (headers or {}).items():
            msg[key] = value
        self.headers = msg

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, amount=-1):
        if amount is None or amount < 0:
            amount = len(self._body) - self._offset
        chunk = self._body[self._offset:self._offset + amount]
        self._offset += len(chunk)
        return chunk


class _FakeConnection:
    """Records every request so a test can assert how many attempts were made."""

    log: list = []
    script: list = []

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None

    def request(self, method, path, headers=None):
        type(self).log.append((self.host, path, dict(headers or {})))

    def getresponse(self):
        if not type(self).script:
            raise AssertionError("fake transport ran out of scripted responses")
        nxt = type(self).script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def close(self):
        pass


@pytest.fixture
def transport(monkeypatch):
    """Route the seam at a scripted fake connection, address checks intact."""
    _FakeConnection.log = []
    _FakeConnection.script = []
    monkeypatch.setattr(ff.http.client, "HTTPSConnection", _FakeConnection)
    monkeypatch.setattr(ff, "_resolve_public",
                        lambda url, host, port, deadline: [
                            (2, 1, 6, "", ("93.184.216.34", port))])
    monkeypatch.setattr(ff, "_connect",
                        lambda url, infos, host, deadline: object())
    monkeypatch.setattr(ff.time, "sleep", lambda _s: None)
    return _FakeConnection


# ---- scheme policy ---------------------------------------------------------

class TestSchemePolicy:
    @pytest.mark.parametrize("url", [
        "http://example.com/feed",
        "ftp://example.com/feed",
        "file:///etc/passwd",
        "//example.com/feed",
    ])
    def test_non_https_is_refused(self, url):
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed(url)
        assert caught.value.reason == "scheme"

    def test_plaintext_is_never_silently_upgraded(self):
        """An http:// row is an operator-visible defect, not something to fix
        behind their back."""
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("http://example.com/feed")
        assert "does not silently upgrade" in caught.value.detail

    def test_empty_url_is_refused(self):
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("   ")
        assert caught.value.reason == "empty_url"

    def test_https_without_a_host_is_refused(self):
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https:///nowhere")
        assert caught.value.reason == "no_host"


# ---- address policy --------------------------------------------------------

class TestAddressPolicy:
    @pytest.mark.parametrize("text,expected", [
        ("127.0.0.1", "loopback"),
        ("127.9.9.9", "loopback"),
        ("::1", "loopback"),
        ("::ffff:127.0.0.1", "loopback"),      # v4-mapped v6 is unwrapped first
        ("::ffff:10.0.0.1", "private"),
        ("10.1.2.3", "private"),
        ("172.16.0.1", "private"),
        ("192.168.1.1", "private"),
        ("169.254.169.254", "link-local"),     # the cloud metadata address
        ("fe80::1", "link-local"),
        ("224.0.0.1", "multicast"),
        ("0.0.0.0", "unspecified"),
        ("93.184.216.34", ""),
        ("2606:2800:220:1:248:1893:25c8:1946", ""),
    ])
    def test_reject_reason_table(self, text, expected):
        assert ff._reject_reason(ipaddress.ip_address(text)) == expected

    @pytest.mark.parametrize("url", [
        "https://127.0.0.1/feed",
        "https://10.0.0.5/feed",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/feed",
    ])
    def test_private_targets_are_REFUSED_not_unavailable(self, url):
        """The class matters: Refused is loud and terminal, Unavailable is a
        miss a caller may swallow. A private address must never be swallowed."""
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed(url)
        assert caught.value.reason == "blocked_address"

    def test_a_mixed_answer_is_refused(self, monkeypatch):
        """One public and one private answer for the same name is a rebinding
        shape; no legitimate feed needs it."""
        monkeypatch.setattr(ff.socket, "getaddrinfo", lambda *a, **k: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ])
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/feed")
        assert caught.value.reason == "blocked_address"

    def test_dns_failure_is_unavailable_not_refused(self, monkeypatch):
        def _boom(*_a, **_k):
            raise ff.socket.gaierror("name or service not known")
        monkeypatch.setattr(ff.socket, "getaddrinfo", _boom)
        monkeypatch.setattr(ff.time, "sleep", lambda _s: None)
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            ff.fetch_feed("https://example.invalid/feed")
        assert caught.value.reason == "dns"


# ---- content encoding ------------------------------------------------------

class TestContentEncoding:
    def test_identity_passes_through(self):
        assert ff._decode_content_encoding("u", b"body", "") == b"body"
        assert ff._decode_content_encoding("u", b"body", "identity") == b"body"

    def test_deflate_roundtrip(self):
        blob = zlib.compress(b"hello feed")
        assert ff._decode_content_encoding("u", blob, "deflate") == b"hello feed"

    def test_gzip_roundtrip(self):
        obj = zlib.compressobj(wbits=31)
        blob = obj.compress(b"hello feed") + obj.flush()
        assert ff._decode_content_encoding("u", blob, "gzip") == b"hello feed"

    def test_the_cap_is_on_DECODED_bytes(self):
        """2 MiB of gzip is not 2 MiB of feed. A small blob that expands past
        the cap is refused, which is the whole reason the cap is applied here
        and not only on the raw read."""
        blob = zlib.compress(b"x" * (ff.MAX_DECODED_BYTES + 4096))
        assert len(blob) < ff.MAX_DECODED_BYTES
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff._decode_content_encoding("u", blob, "deflate")
        assert caught.value.reason == "body_too_large"

    def test_unknown_encoding_is_refused(self):
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff._decode_content_encoding("u", b"...", "br")
        assert caught.value.reason == "content_encoding"


# ---- charset ---------------------------------------------------------------

class TestCharset:
    def test_header_charset_wins(self):
        assert ff._detect_charset(b"<rss/>", "utf-16", "application/xml") == "utf-16"

    def test_bom_beats_the_declaration(self):
        assert ff._detect_charset(
            b"\xef\xbb\xbf<rss/>", "", "application/xml") == "utf-8-sig"

    def test_xml_declaration_is_read(self):
        raw = b"<?xml version='1.0' encoding='iso-8859-1'?><rss/>"
        assert ff._detect_charset(raw, "", "application/xml") == "iso-8859-1"

    def test_meta_charset_is_read_for_html(self):
        raw = b'<html><head><meta charset="windows-1252">'
        assert ff._detect_charset(raw, "", "text/html") == "windows-1252"

    def test_default_is_utf8(self):
        assert ff._detect_charset(b"<rss/>", "", "application/xml") == "utf-8"

    def test_bad_bytes_fall_back_loudly_not_silently(self, caplog):
        text, used = ff._decode_text("https://x/y", b"\xff\xfe ok", "utf-8")
        assert used == "utf-8/replace"
        assert "�" in text
        assert any("falling back" in r.message or "falling back" in r.getMessage()
                   for r in caplog.records)


# ---- the deadline ----------------------------------------------------------

class TestDeadline:
    def test_a_spent_deadline_refuses_to_hand_out_a_slice(self):
        deadline = ff._Deadline("https://x/y", 0.0)
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            deadline.slice_for(5.0)
        assert caught.value.reason == "deadline"

    def test_deadline_exhaustion_is_not_retryable(self):
        """Re-sending after the budget is gone burns what is left to learn what
        we already know."""
        deadline = ff._Deadline("https://x/y", 0.0)
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            deadline.slice_for(5.0)
        assert caught.value.retryable is False

    def test_a_slice_never_exceeds_what_is_left(self):
        deadline = ff._Deadline("https://x/y", 0.25)
        assert deadline.slice_for(ff.READ_TIMEOUT_S) <= 0.25

    def test_caller_budget_cannot_exceed_the_module_ceiling(self, transport):
        transport.script = [_FakeResponse(200, b"<rss/>")]
        doc = ff.fetch_feed("https://example.com/f", deadline_s=9999)
        assert doc.elapsed_s < ff.TOTAL_DEADLINE_S


# ---- status, redirects, media type, size -----------------------------------

class TestBoundedRequest:
    def test_a_feed_comes_back_decoded(self, transport):
        transport.script = [_FakeResponse(
            200, b"<?xml version='1.0'?><rss><channel/></rss>")]
        doc = ff.fetch_feed("https://example.com/feed")
        assert doc.text.startswith("<?xml")
        assert doc.media_type == "application/rss+xml"
        assert doc.redirects == 0
        assert doc.final_url == "https://example.com/feed"

    def test_default_user_agent_is_the_one_that_shipped(self, transport):
        """Several publishers 403 an unfamiliar agent, so this string is live
        behaviour, not decoration."""
        transport.script = [_FakeResponse(200, b"<rss/>")]
        ff.fetch_feed("https://example.com/feed")
        assert transport.log[0][2]["User-Agent"] == ff.DEFAULT_USER_AGENT
        assert transport.log[0][2]["Accept-Encoding"] == "identity"

    def test_html_served_as_a_feed_is_refused(self, transport):
        transport.script = [_FakeResponse(200, b"<html/>",
                                          content_type="text/html")]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/feed")
        assert caught.value.reason == "media_type"

    def test_a_feed_served_as_an_article_is_refused(self, transport):
        transport.script = [_FakeResponse(200, b"<rss/>",
                                          content_type="application/rss+xml")]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_article("https://example.com/story")
        assert caught.value.reason == "media_type"

    def test_any_plus_xml_subtype_counts_as_a_feed(self, transport):
        transport.script = [_FakeResponse(
            200, b"<feed/>", content_type="application/vendor.thing+xml")]
        assert ff.fetch_feed("https://example.com/feed").text == "<feed/>"

    def test_xhtml_counts_as_an_article(self, transport):
        transport.script = [_FakeResponse(
            200, b"<html/>", content_type="application/xhtml+xml")]
        assert ff.fetch_article("https://example.com/story").text == "<html/>"

    def test_a_missing_content_type_is_refused(self, transport):
        transport.script = [_FakeResponse(200, b"<rss/>", content_type=None)]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/feed")
        assert caught.value.reason == "media_type"

    def test_redirects_are_followed_up_to_the_cap(self, transport):
        transport.script = [
            _FakeResponse(301, headers={"Location": "https://example.com/b"}),
            _FakeResponse(302, headers={"Location": "/c"}),
            _FakeResponse(200, b"<rss/>"),
        ]
        doc = ff.fetch_feed("https://example.com/a")
        assert doc.redirects == 2
        assert doc.final_url == "https://example.com/c"

    def test_too_many_redirects_is_refused(self, transport):
        transport.script = [
            _FakeResponse(301, headers={"Location": f"https://example.com/{n}"})
            for n in range(ff.MAX_REDIRECTS + 1)
        ]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/a")
        assert caught.value.reason == "too_many_redirects"

    def test_a_redirect_into_plaintext_is_refused(self, transport):
        transport.script = [
            _FakeResponse(302, headers={"Location": "http://example.com/b"})]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/a")
        assert caught.value.reason == "scheme"

    def test_a_redirect_into_the_private_network_is_refused(self, monkeypatch,
                                                            transport):
        """The check runs on EVERY hop, not just the first. This is the exact
        shape the old requests.get had no defence against."""
        monkeypatch.setattr(ff, "_resolve_public", _real_resolve_public)
        transport.script = [
            _FakeResponse(302, headers={"Location": "https://127.0.0.1/b"})]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/a")
        assert caught.value.reason == "blocked_address"
        assert len(transport.log) == 1   # refused before the second request

    def test_a_redirect_without_a_location_is_unavailable(self, transport):
        transport.script = [_FakeResponse(302)]
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            ff.fetch_feed("https://example.com/a")
        assert caught.value.reason == "redirect"

    def test_an_oversize_body_is_refused_during_the_read(self, transport):
        transport.script = [_FakeResponse(
            200, b"x" * (ff.MAX_DECODED_BYTES + 1))]
        with pytest.raises(ff.FeedFetchRefused) as caught:
            ff.fetch_feed("https://example.com/feed")
        assert caught.value.reason == "body_too_large"

    def test_a_body_exactly_at_the_cap_is_allowed(self, transport):
        transport.script = [_FakeResponse(200, b"x" * ff.MAX_DECODED_BYTES)]
        doc = ff.fetch_feed("https://example.com/feed")
        assert doc.byte_count == ff.MAX_DECODED_BYTES

    def test_a_404_is_settled_and_tried_once(self, transport):
        transport.script = [_FakeResponse(404, reason="Not Found")]
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            ff.fetch_feed("https://example.com/gone")
        assert caught.value.retryable is False
        assert len(transport.log) == 1

    def test_a_503_is_retried_exactly_MAX_RETRIES_times(self, transport):
        transport.script = [_FakeResponse(503, reason="Unavailable")
                            for _ in range(ff.MAX_RETRIES + 1)]
        with pytest.raises(ff.FeedFetchUnavailable) as caught:
            ff.fetch_feed("https://example.com/flaky")
        assert caught.value.retryable is True
        assert len(transport.log) == ff.MAX_RETRIES + 1

    def test_a_retry_can_succeed(self, transport):
        transport.script = [_FakeResponse(503), _FakeResponse(200, b"<rss/>")]
        assert ff.fetch_feed("https://example.com/flaky").text == "<rss/>"

    def test_a_tripped_bound_is_never_retried(self, transport):
        transport.script = [_FakeResponse(200, b"<html/>",
                                          content_type="text/html")]
        with pytest.raises(ff.FeedFetchRefused):
            ff.fetch_feed("https://example.com/feed")
        assert len(transport.log) == 1

    def test_header_charset_is_honoured_end_to_end(self, transport):
        transport.script = [_FakeResponse(
            200, "<rss>café</rss>".encode("iso-8859-1"),
            content_type="application/rss+xml; charset=iso-8859-1")]
        doc = ff.fetch_feed("https://example.com/feed")
        assert "café" in doc.text
        assert doc.encoding == "iso-8859-1"


_real_resolve_public = ff._resolve_public


# ---- the bounds themselves -------------------------------------------------

class TestTheBoundsAreWhatTheOperatorSpecified:
    """A guard against a later edit quietly loosening a number. Each of these
    was specified by the operator for wave 5; changing one is a decision, not
    a tidy-up."""

    def test_the_numbers(self):
        assert ff.ALLOWED_SCHEME == "https"
        assert ff.CONNECT_TIMEOUT_S == 5.0
        assert ff.READ_TIMEOUT_S == 10.0
        assert ff.MAX_REDIRECTS == 3
        assert ff.MAX_DECODED_BYTES == 2 * 1024 * 1024
        assert ff.MAX_RETRIES == 2
        assert ff.TOTAL_DEADLINE_S == 25.0

    def test_the_seam_is_stdlib_only(self):
        """A client bundle must be able to import the seam without adding a
        dependency, and activation must not drag in requests or feedparser."""
        tree = ast.parse(
            (REPO / "nodes" / "_otr_feed_fetch.py").read_text(encoding="utf-8"))
        third_party = {"requests", "feedparser", "bs4", "httpx", "torch",
                       "transformers", "numpy"}
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                seen.add((node.module or "").split(".")[0])
        assert not (seen & third_party), f"non-stdlib import: {seen & third_party}"


# ---- the r3 finding, made machine-checkable --------------------------------

class TestEveryHopRoutesThroughTheSeam:
    """Network hardening is NOT inherited. Each hop has to be wired on purpose,
    so each hop gets a test that fails if it grows its own fetch again."""

    def test_the_article_hop_no_longer_calls_requests(self):
        tree = _tree("story_orchestrator.py")
        assert not _method_calls(tree, "requests", "get")
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)
                    and any(a.name == "requests" for a in n.names)]
        assert _plain_calls(tree, "fetch_article"), "the seam is not wired in"

    def test_the_feed_hop_hands_feedparser_a_document_never_a_url(self):
        """feedparser.parse(url) does its own unbounded urllib fetch. Every
        call must receive text that already came through the seam."""
        tree = _tree("story_orchestrator.py")
        parses = _method_calls(tree, "feedparser", "parse")
        assert parses, "the feed hop still has to parse a feed"
        for call in parses:
            arg = call.args[0]
            assert isinstance(arg, ast.Attribute) and arg.attr == "text", (
                "feedparser was handed something that is not seam-fetched text")

    def test_the_feed_hop_no_longer_sets_a_process_global_timeout(self):
        """setdefaulttimeout is process-global; a ~30-wide pool raced it, so it
        was never the per-feed timeout it looked like."""
        tree = _tree("story_orchestrator.py")
        assert not [n for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute)
                    and n.attr == "setdefaulttimeout"]

    def test_the_media_archive_hop_routes_urls_through_the_seam(self):
        tree = _tree("_otr_media_archive_sources.py")
        assert _plain_calls(tree, "looks_like_url")
        assert _plain_calls(tree, "fetch_feed")
        for call in _method_calls(tree, "feedparser", "parse"):
            arg = call.args[0]
            assert isinstance(arg, ast.Name) and arg.id == "document", (
                "feedparser must parse the resolved document, not the input")

    def test_media_archive_still_parses_an_inline_document_without_fetching(
            self, monkeypatch):
        """A document is not a URL: passing raw XML must never touch the seam."""
        from nodes import _otr_media_archive_sources as mas

        def _forbidden(*_a, **_k):
            raise AssertionError("an inline document must not be fetched")

        monkeypatch.setattr(ff, "fetch_feed", _forbidden)
        rows = mas.parse_media_archive_feed(
            "<?xml version='1.0'?><rss version='2.0'><channel>"
            "<title>Archive</title><item><title>A 1948 transcription disc</title>"
            "<description>Body text for the entry.</description>"
            "<link>https://example.com/a</link></item></channel></rss>")
        assert rows and rows[0]["headline"]

    def test_media_archive_refuses_a_plaintext_feed_url_loudly(self):
        """Not wrapped in MediaArchiveSourceError on purpose: the caller
        collects that error per feed and moves on, which would hide the defect
        whenever another feed happened to work."""
        from nodes import _otr_media_archive_sources as mas
        with pytest.raises(ff.FeedFetchRefused) as caught:
            mas.parse_media_archive_feed("http://example.com/feed")
        assert caught.value.reason == "scheme"

    def test_shipped_feed_lists_are_all_https(self):
        """Every shipped feed must satisfy the policy the seam now enforces --
        otherwise wave 5 turns a working lane into a loud failure."""
        from nodes import _otr_media_archive_sources as mas
        from nodes import story_orchestrator as so
        for url in mas.DEFAULT_MEDIA_ARCHIVE_FEEDS:
            assert url.startswith("https://"), url
        for url in so.SCIENCE_NEWS_FEEDS:
            assert url.startswith("https://"), url
