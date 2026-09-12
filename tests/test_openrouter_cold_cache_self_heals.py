"""A cold OpenRouter catalog cache heals itself before it clamps anything.

`resolve_context_window` used to fall back to 8,192 tokens on a slug miss --
which is the exact clamp that cut a 9,520-token script off mid-JSON
(the function's own docstring records it). A FRESH install of the cloud lane
is cold until someone runs `scripts/otr_openrouter_refresh.py` by hand, so
every first run was that silent wrong render. Found by the codex contrarian
on the 2026-09-12 arc batch, which refuted "close as accepted".

The rule: on a miss, with the API key present, refresh ONCE per process and
re-read; without a key, never touch the network; a failed refresh falls back
exactly as before; never raise.
"""
from __future__ import annotations

import pytest

from nodes import _otr_openrouter_backend as orb

SLUG = "aion-labs/aion-3.0-mini"
RAW = [{"id": SLUG, "context_length": 131072,
        "supported_parameters": ["response_format"],
        "architecture": {"output_modalities": ["text"]}}]


@pytest.fixture
def cold(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))   # no cache file
    monkeypatch.setattr(orb, "_COLD_CACHE_REFRESH_TRIED", False)
    return tmp_path


def test_cold_cache_with_a_key_refreshes_once_and_reads_the_real_window(
        cold, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    fetched = []

    def _models(**kw):
        fetched.append(kw)
        return list(RAW)

    monkeypatch.setattr(orb, "_fetch_models_json", _models)
    assert orb.resolve_context_window(SLUG, row_default=8192) == 131072
    assert len(fetched) == 1, "one inline refresh"
    # A later miss (an unknown slug) does NOT refresh again this process.
    assert orb.resolve_context_window("nobody/nothing", row_default=8192) == 8192
    assert len(fetched) == 1
    # And the catalog it wrote is the one on disk now.
    assert orb.load_catalog_cache().get("source") == "live"


def test_cold_cache_without_a_key_never_touches_the_network(cold, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def _boom(**kw):
        raise AssertionError("network fetch fired without an API key")

    monkeypatch.setattr(orb, "_fetch_models_json", _boom)
    assert orb.resolve_context_window(SLUG, row_default=8192) == 8192
    assert orb._COLD_CACHE_REFRESH_TRIED is False


def test_a_failed_inline_refresh_falls_back_and_never_raises(cold, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    calls = []

    def _fail(**kw):
        calls.append(kw)
        raise RuntimeError("offline")

    monkeypatch.setattr(orb, "_fetch_models_json", _fail)
    assert orb.resolve_context_window(SLUG, row_default=8192) == 8192
    assert len(calls) == 1
    assert orb._COLD_CACHE_REFRESH_TRIED is True
    # The empty cache is still the designed safe state afterwards.
    assert orb.load_catalog_cache().get("count") == 0


def test_a_warm_cache_never_refreshes(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(orb, "_COLD_CACHE_REFRESH_TRIED", False)
    monkeypatch.setattr(orb, "_fetch_models_json", lambda **kw: list(RAW))
    orb.refresh_catalog_cache()                      # warm it explicitly

    def _boom(**kw):
        raise AssertionError("a warm cache must not refetch")

    monkeypatch.setattr(orb, "_fetch_models_json", _boom)
    assert orb.resolve_context_window(SLUG, row_default=8192) == 131072
    assert orb._COLD_CACHE_REFRESH_TRIED is False
