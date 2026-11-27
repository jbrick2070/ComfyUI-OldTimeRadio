"""A provider-returned URL is fetched only on its own origin, without the key
leaving it, and only over https with a ceiling; the adapter that uploads an
image proves the path at its own seat.

From the 2026-10-30 registry review (Gemini 4 Argon, grounded by a Sonnet
reader): the Google client accepted any absolute URL a response named and put
the API key on it, urllib carried the key across a cross-origin redirect, the
partner-result download followed any scheme to any size, and the omni video
adapter read whatever reference path it was handed. None is reachable from a
workflow field; each is a hardening of a seat that trusted its caller.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pytest


def test_an_absolute_url_off_the_api_origin_is_refused_before_any_request(monkeypatch):
    from nodes._otr_google_api import client as C
    monkeypatch.delenv("OTR_GOOGLE_API_BASE", raising=False)
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *a, **k: pytest.fail("a request was built for a foreign origin"))
    with pytest.raises(C.GoogleAPIRequestShapeError, match="not on the API origin"):
        C._get_bytes("https://evil.example/x.mp4", api_key="K", timeout_s=5)
    with pytest.raises(C.GoogleAPIRequestShapeError, match="not on the API origin"):
        C._absolute_url("http://generativelanguage.googleapis.com/v1beta/files/x")
    assert C._absolute_url("https://generativelanguage.googleapis.com/v1beta/files/x") \
        == "https://generativelanguage.googleapis.com/v1beta/files/x"
    assert C._absolute_url("v1beta/files/x") == "https://generativelanguage.googleapis.com/v1beta/files/x"


def test_an_operator_base_keeps_its_own_scheme_and_host(monkeypatch):
    from nodes._otr_google_api import client as C
    monkeypatch.setenv("OTR_GOOGLE_API_BASE", "http://localhost:8099/proxy")
    assert C._absolute_url("/v1beta/x") == "http://localhost:8099/proxy/v1beta/x"
    assert C._absolute_url("http://localhost:8099/proxy/v1beta/files/y") \
        == "http://localhost:8099/proxy/v1beta/files/y"
    with pytest.raises(C.GoogleAPIRequestShapeError):
        C._absolute_url("https://generativelanguage.googleapis.com/v1beta/files/y")


def test_the_key_does_not_follow_a_redirect_off_the_origin():
    from nodes._otr_google_api import client as C
    handler = C._KeyStaysOnOrigin()
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/files/x",
                                 headers={"x-goog-api-key": "K"})
    away = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "https://other.example/x")
    assert away is not None and not away.has_header("X-goog-api-key")
    same = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "https://generativelanguage.googleapis.com/v1beta/files/y")
    assert same is not None and same.has_header("X-goog-api-key")


def test_the_omni_adapter_refuses_an_image_outside_the_read_roots(tmp_path):
    from nodes._otr_video_engines import eng_google_omni_video as OV
    from nodes._otr_google_api.client import GoogleAPIRequestShapeError
    with pytest.raises(GoogleAPIRequestShapeError, match="outside the approved"):
        OV._image_part({"path": sys.executable}, field="init_image")
    png = tmp_path / "ref.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    part = OV._image_part({"path": str(png)}, field="init_image")
    assert isinstance(part, dict) and part


def test_a_partner_result_is_https_only_and_bounded(tmp_path, monkeypatch):
    from nodes._otr_shared import cloud_media_invoke as I
    dest = tmp_path / "out.mp4"
    with pytest.raises(I.CloudMediaError, match="not https"):
        I._stream_to_temp("http://provider.example/out.mp4", dest)
    assert not dest.exists()

    class _Endless:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b"\x00" * n
    class _Opener:
        def open(self, req, timeout=None):
            return _Endless()
    monkeypatch.setattr(I, "_DOWNLOAD_MAX_BYTES", 3 << 20)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _Opener())
    with pytest.raises(I.CloudMediaError, match="exceeds"):
        I._stream_to_temp("https://provider.example/out.mp4", dest)
    assert not dest.exists()


def test_a_result_redirected_off_https_is_refused_before_it_is_followed():
    """Sonnet review of 636b18227: the https check read the first URL only,
    and urllib's stock handler follows a 30x to http or ftp. The redirect
    handler now asks the same question of every target."""
    import inspect
    from nodes._otr_shared import cloud_media_invoke as I
    for bad in ("http://provider.example/out.mp4", "ftp://provider.example/out.mp4",
                "file:///etc/hosts", ""):
        with pytest.raises(I.CloudMediaError, match="not https"):
            I._refuse_off_https(bad, redirected=True)
    I._refuse_off_https("https://provider.example/out.mp4")
    src = inspect.getsource(I._stream_to_temp)
    assert "_open_https_only(req" in src and "urlopen(" not in src
    opener_src = inspect.getsource(I._open_https_only)
    assert "_refuse_off_https(newurl, redirected=True)" in opener_src
    assert "build_opener(_HttpsOnly())" in opener_src


def test_the_post_path_keeps_the_key_on_the_origin_and_a_scheme_change_strips_it(monkeypatch):
    """Sonnet review of 636b18227: _post_json used the stock opener, so a
    30x on a POST became a GET to the new URL with the key copied; and the
    handler compared hosts only, so https://H -> http://H kept the key."""
    from nodes._otr_google_api import client as C
    monkeypatch.delenv("OTR_GOOGLE_API_BASE", raising=False)
    seen, handlers = [], []

    class _Opener:
        def open(self, req, timeout=None):
            seen.append(req)
            return io.BytesIO(b'{"ok": true}')
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *h: handlers.extend(h) or _Opener())
    assert C._post_json("/v1beta/models/x:generateContent", {"a": 1},
                        api_key="K", timeout_s=5) == {"ok": True}
    assert any(isinstance(h, C._KeyStaysOnOrigin) for h in handlers)
    assert seen and seen[0].get_header("X-goog-api-key") == "K"
    handler = C._KeyStaysOnOrigin()
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/files/x",
                                 headers={"x-goog-api-key": "K"})
    down = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "http://generativelanguage.googleapis.com/v1beta/files/x")
    assert down is not None and not down.has_header("X-goog-api-key")


def test_the_wallet_key_does_not_follow_a_redirect_off_its_origin():
    """THE DEFECT THIS PINS (shipped through 2.3.64): the balance preflight
    sent ``Authorization: Bearer <key>`` through urllib's stock opener, whose
    redirect handler copies every header to the new request -- so a 30x to
    another host, or to the same host over plain http, carried the Comfy or
    OpenRouter key there."""
    from nodes._otr_shared import cloud_balance_preflight as B
    handler = B._BearerStaysOnOrigin()
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": "Bearer K"})
    away = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "https://other.example/credits")
    assert away is not None and not away.has_header("Authorization")
    down = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "http://openrouter.ai/api/v1/credits")
    assert down is not None and not down.has_header("Authorization")
    same = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", {},
                                    "https://openrouter.ai/api/v1/key")
    assert same is not None and same.get_header("Authorization") == "Bearer K"


def test_the_wallet_read_goes_through_the_guarded_opener(monkeypatch):
    """The guard is wired, not just defined: `_http_get_json` builds its
    opener with the handler and never reaches the stock `urlopen`."""
    from nodes._otr_shared import cloud_balance_preflight as B
    handlers, seen = [], []

    class _Resp:
        status = 200

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            seen.append(req)
            return _Resp()

    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *h: handlers.extend(h) or _Opener())
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stock urlopen")))
    assert B._http_get_json("https://openrouter.ai/api/v1/credits", "K") == (200, {"ok": True})
    assert any(isinstance(h, B._BearerStaysOnOrigin) for h in handlers)
    assert seen and seen[0].get_header("Authorization") == "Bearer K"


def _answers(monkeypatch, module, table):
    """Answer `_host_addresses` from a table: no DNS in the suite."""
    def fake(host):
        if host not in table:
            raise OSError("no such host in the test table: %s" % host)
        return list(table[host])
    monkeypatch.setattr(module, "_host_addresses", fake)


@pytest.mark.parametrize("address, word", [
    ("127.0.0.1", "loopback"),
    ("10.0.0.7", "private"),
    ("192.168.1.20", "private"),
    ("169.254.169.254", "link-local"),
    ("::1", "loopback"),
    ("100.64.0.9", "not publicly routable"),
])
def test_a_partner_result_on_a_non_public_host_is_refused_before_any_request(
        tmp_path, monkeypatch, address, word):
    """THE DEFECT THIS PINS (shipped through 2.3.64): the result download
    checked the scheme only, so a provider fault or a tampered job result
    naming an https URL on this machine's own network was fetched and handed
    on as media."""
    from nodes._otr_shared import cloud_media_invoke as I
    _answers(monkeypatch, I, {"inside.example": [address]})
    monkeypatch.setattr(I, "_open_https_only",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    dest = tmp_path / "out.mp4"
    with pytest.raises(I.CloudMediaError, match="public hosts only") as caught:
        I._stream_to_temp("https://inside.example/out.mp4?sig=SECRET", dest)
    assert word in str(caught.value)
    assert "SECRET" not in str(caught.value)
    assert not dest.exists()


def test_a_name_with_one_private_answer_is_refused(monkeypatch):
    """Every answer must be public: one public and one private address is the
    DNS-rebinding shape."""
    from nodes._otr_shared import cloud_media_invoke as I
    _answers(monkeypatch, I, {"mixed.example": ["93.184.216.34", "10.0.0.7"],
                              "cdn.example": ["93.184.216.34"]})
    with pytest.raises(I.CloudMediaError, match="public hosts only"):
        I._refuse_non_public_host("https://mixed.example/out.mp4")
    I._refuse_non_public_host("https://cdn.example/out.mp4")
    # A name that does not resolve is left to fail as a transport error.
    I._refuse_non_public_host("https://unknown.example/out.mp4")


def test_a_redirect_to_a_non_public_host_is_refused_before_it_is_followed(monkeypatch):
    import inspect
    from nodes._otr_shared import cloud_media_invoke as I
    _answers(monkeypatch, I, {"inside.example": ["192.168.1.20"]})
    with pytest.raises(I.CloudMediaError, match="redirected to") as caught:
        I._refuse_non_public_host("https://inside.example/x.mp4?sig=SECRET", redirected=True)
    assert "SECRET" not in str(caught.value)
    opener_src = inspect.getsource(I._open_https_only)
    assert "_refuse_non_public_host(newurl, redirected=True)" in opener_src


def test_a_failed_download_names_the_url_without_its_query(tmp_path, monkeypatch):
    """A provider's result URL is usually signed: the query string is the
    secret. Exception text reaches /history, so it carries host and path only."""
    from nodes._otr_shared import cloud_media_invoke as I
    _answers(monkeypatch, I, {"cdn.example": ["93.184.216.34"]})

    def boom(req, timeout=None):
        raise OSError("connection reset")
    monkeypatch.setattr(I, "_open_https_only", boom)
    with pytest.raises(I.CloudMediaError) as caught:
        I._stream_to_temp("https://cdn.example/jobs/42/out.mp4?X-Sig=SECRET&exp=1", tmp_path / "o.mp4")
    text = str(caught.value)
    assert "https://cdn.example/jobs/42/out.mp4" in text
    assert "SECRET" not in text and "X-Sig" not in text
    assert I._shown_url("https://u:p@cdn.example:8443/a/b.mp4?k=v#f") == "https://cdn.example:8443/a/b.mp4"
    assert I._shown_url("not a url") == "<not a url>"
