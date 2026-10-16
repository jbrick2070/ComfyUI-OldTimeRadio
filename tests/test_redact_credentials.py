# -*- coding: utf-8 -*-
"""``_redact_credentials``: every leak form found on 2026-10-16, pinned.

The function was patched twice in one day and each review found new leaks, so
under the two-strikes rule a kibitz panel (Codex gpt-6.1-sol + Gemini 4 Argon)
redesigned it as one KEY + SEPARATOR + VALUE grammar. This file is the
contract: the secret below contains no key-like substring, so a test can only
pass because the grammar redacted it -- not because a word filter happened to
catch "SECRET" (which hid the opaque-Bearer leak in the older test).
"""
import importlib
import pathlib
import sys
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent.parent))

VA = importlib.import_module("nodes._otr_visual_assets")
SECRET = "Q7m9Z2x4"
PH = "<credential redacted>"


def redact(text):
    return VA._redact_credentials(text)


LEAKS = [
    # the eight forms the reviews found, in order
    "Authorization: Bearer Q7m9Z2x4",
    'Authorization: "Bearer Q7m9Z2x4"',
    'Authorization: Digest username="a", response="Q7m9Z2x4"',
    "Authorization: Bearer abc, Q7m9Z2x4",
    "Authorization:\nQ7m9Z2x4\nrest",
    "X-Authorization-Token: Q7m9Z2x4",
    '{"Authorization": "Q7m9Z2x4"}',
    "'Authorization': 'Q7m9Z2x4'",
    "Authorization -> Q7m9Z2x4",
    # the same rule for every credential key, not only Authorization
    '{"token": "Q7m9Z2x4"}',
    "'hf_token': 'Q7m9Z2x4'",
    "password: Q7m9Z2x4",
    "api_key = Q7m9Z2x4",
    "X-Api-Key: Q7m9Z2x4",
    "Cookie: session=Q7m9Z2x4",
    "Set-Cookie: session=Q7m9Z2x4; Path=/; HttpOnly",
    "Proxy-Authorization: Basic Q7m9Z2x4",
    "Authorization = Bearer Q7m9Z2x4",
    "authorization%3A%20Bearer%20Q7m9Z2x4",
    # the shapes the original design was written for (stay handled)
    "cdn-lfs.hf.co/repos/ab/m.safetensors?X-Amz-Signature=Q7m9Z2x4 failed",
    "GET https%3A%2F%2Fcdn-lfs.hf.co%2Ff%3FX-Amz-Signature%3DQ7m9Z2x4",
    "https://cdn-lfs.hf.co/file?\nX-Amz-Signature=Q7m9Z2x4",
    "hf_token=Q7m9Z2x4 was rejected",
    "GET https://cdn-lfs.hf.co/f?X-Amz-Signature=Q7m9Z2x4",
    "pip exit 1: GET https://idx.example/simple?access_key=Q7m9Z2x4 403",
]


@pytest.mark.parametrize("text", LEAKS)
def test_the_secret_never_survives(text):
    out = redact(text)
    assert SECRET not in out, out
    assert PH in out, out


SURVIVE = [
    "ERROR: No matching distribution found for nvidia-vfx>=0.2,<0.3",
    "HTTP 401",
    r"FileNotFoundError: [WinError 3] The system cannot find the path "
    r"specified: 'C:\Users\a\models\diffusion_models\x.safetensors'",
    r"C:\Users\a\.cache\huggingface\tokenizer.json is 282 chars -- OVER "
    r"the 259-char Windows path budget",
    "unauthorized; please log in",
]


@pytest.mark.parametrize("text", SURVIVE)
def test_diagnostics_without_a_credential_are_untouched(text):
    assert redact(text) == text


def test_text_around_a_redacted_value_survives_on_the_same_line():
    """pip output and HF errors arrive as one line: only the key and its
    value go, the rest of the diagnostic stays."""
    out = redact(r"unauthorized; authorization = failed for C:\Users\a\file.txt")
    assert out.startswith("unauthorized; " + PH)
    assert r"C:\Users\a\file.txt" in out
    out = redact("Authorization: Bearer Q7m9Z2x4\nERROR: No matching distribution")
    assert SECRET not in out and out.endswith("\nERROR: No matching distribution")


def test_the_placeholder_is_never_rescanned():
    once = redact("password: Q7m9Z2x4")
    assert once == PH
    assert redact(once) == once                    # idempotent, no double wrap
    assert "redacted> redacted>" not in redact("token=a token=b")


def test_it_fails_closed(monkeypatch):
    class Boom:
        def sub(self, *a, **k):
            raise RuntimeError("regex engine failure")

        def search(self, *a, **k):
            raise RuntimeError("regex engine failure")
    monkeypatch.setattr(VA, "_REDACT_RE", Boom())
    assert redact("password: Q7m9Z2x4") == PH


def test_it_is_cheap_on_a_large_log():
    blob = ("Downloading nvidia_vfx-0.2.0.0-cp312-abi3-win_amd64.whl (536 MB) "
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa token\n") * 1000
    start = time.perf_counter()
    redact(blob + "Authorization: Bearer Q7m9Z2x4")
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("text", [
    "401 Client Error: Unauthorized for url: https://huggingface.co/x/resolve/main/f.safetensors",
    "Looking in indexes: https://pypi.org/simple",
    "Using cached https://files.pythonhosted.org/x.whl (1 MB)",
])
def test_a_plain_url_survives_as_the_diagnostic(text):
    """Sonnet QA 2026-10-16: wiping every URL erased which repo/file failed."""
    for url in [w for w in text.split() if "://" in w]:
        assert url in redact(text), redact(text)


@pytest.mark.parametrize("text", [
    "PIP_INDEX_URL=https://u:Q7m9Z2x4@h/simple",
    "--index-url https://u:Q7m9Z2x4@h/simple",
    "git+https://Q7m9Z2x4@github.com/x",
    "UV_INDEX_URL=//u:Q7m9Z2x4@h/simple",
    "fetch https%3A%2F%2Fu%3AQ7m9Z2x4%40h%2Fsimple",
])
def test_a_url_with_a_login_is_redacted(text):
    out = redact(text)
    assert "Q7m9Z2x4" not in out and PH in out, out


@pytest.mark.parametrize("url", [
    "https://h/f?code=Q7m9Z2x4",
    "https://h/f?st=Q7m9Z2x4",
    "https://h/f?a=1&sas=Q7m9Z2x4",
    "https://h/f?k=Q7m9Z2x4",
    "https://h/f?jwt=Q7m9Z2x4",
    "https://h/f?hdnts=exp=1~hmac=Q7m9Z2x4",
    "https://h/cb#access_token=Q7m9Z2x4",
    "https://h/f?download=true&code=Q7m9Z2x4",
])
def test_a_signed_url_loses_its_query_whatever_the_names(url):
    """Sonnet QA 2026-10-16: signed-URL parameters are not enumerable."""
    out = redact("failed for url: " + url + " (403)")
    assert "Q7m9Z2x4" not in out and PH in out, out
    assert "https://h/" in out and out.endswith(" (403)"), out   # host/path kept


def test_a_harmless_query_survives():
    url = "https://huggingface.co/x/resolve/main/f.safetensors?download=true"
    assert redact("GET " + url) == "GET " + url
