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
    monkeypatch.setattr(I, "_DOWNLOAD_MAX_BYTES", 3 << 20)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Endless())
    with pytest.raises(I.CloudMediaError, match="exceeds"):
        I._stream_to_temp("https://provider.example/out.mp4", dest)
    assert not dest.exists()
