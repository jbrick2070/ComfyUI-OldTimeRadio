"""Two-file API keys -- Google, OpenRouter, Comfy Cloud.

The README heading is the user copy. These tests prove the reader matches
that copy: environment wins, then ``<lane>.secret``, then a path named in
``<lane>_api_key.location``. Test mode skips the pack files unless a test
opts in, so a leftover secret on the box cannot satisfy a missing-key case.
"""
from __future__ import annotations

import pytest

from nodes._otr_google_api.client import GoogleAPIKeyMissingError, resolve_api_key
from nodes._otr_google_api.models import google_api_enabled
from nodes._otr_openrouter_backend import openrouter_enabled, refresh_catalog_cache
from nodes._otr_shared import api_key_files as keys
from nodes._otr_shared import cloud_media_backend as cmb


def _clear_google(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _enable_files(monkeypatch, root):
    monkeypatch.setenv("OTR_ALLOW_KEY_FILES", "1")
    monkeypatch.setenv("OTR_API_KEY_PACK_ROOT", str(root))


def test_env_wins_over_secret_file(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "google.secret").write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "env-key")
    assert resolve_api_key() == "env-key"


def test_secret_file_is_way_one(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "google.secret").write_text(
        "# comment\n  file-key-one  \n", encoding="utf-8"
    )
    assert resolve_api_key() == "file-key-one"


def test_location_file_is_way_two(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    held = tmp_path / "elsewhere" / "my-google.txt"
    held.parent.mkdir()
    held.write_text("pointed-key\n", encoding="utf-8")
    (tmp_path / "google_api_key.location").write_text(
        "# pointer\n%s\n" % held, encoding="utf-8"
    )
    assert resolve_api_key() == "pointed-key"


def test_broken_pointer_fails_closed(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "google_api_key.location").write_text(
        str(tmp_path / "missing.txt") + "\n", encoding="utf-8"
    )
    with pytest.raises(GoogleAPIKeyMissingError, match="not a file"):
        resolve_api_key()


def test_missing_key_names_both_files(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    with pytest.raises(GoogleAPIKeyMissingError, match="google.secret"):
        resolve_api_key()
    with pytest.raises(GoogleAPIKeyMissingError, match="google_api_key.location"):
        resolve_api_key()


def test_test_mode_ignores_pack_files_unless_opted_in(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    monkeypatch.delenv("OTR_ALLOW_KEY_FILES", raising=False)
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    monkeypatch.setenv("OTR_API_KEY_PACK_ROOT", str(tmp_path))
    (tmp_path / "google.secret").write_text("should-not-count\n", encoding="utf-8")
    with pytest.raises(GoogleAPIKeyMissingError):
        resolve_api_key()


def test_google_broken_pointer_disables_the_lane(monkeypatch, tmp_path):
    _clear_google(monkeypatch)
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "google_api_key.location").write_text(
        str(tmp_path / "missing.txt") + "\n", encoding="utf-8"
    )
    assert google_api_enabled() is False


def test_openrouter_broken_pointer_is_disabled(monkeypatch, tmp_path):
    from nodes import _otr_openrouter_backend as orb

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path / "or-cache"))
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "openrouter_api_key.location").write_text(
        str(tmp_path / "missing.txt") + "\n", encoding="utf-8"
    )
    assert openrouter_enabled() is False

    def _boom(**_kw):
        raise AssertionError("broken pointer must not hit the network")

    monkeypatch.setattr(orb, "_fetch_models_json", _boom)
    catalog = refresh_catalog_cache()
    assert isinstance(catalog, dict)


def test_openrouter_file_recipe_still_works(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "openrouter.secret").write_text("or-key\n", encoding="utf-8")
    assert keys.resolve_lane_key("openrouter") == "or-key"


def test_comfy_has_no_key_file_lane(monkeypatch, tmp_path):
    """Rip 2026-09-19: a leftover comfy.secret is NOT a credential. The
    Comfy lane reads only the api_key_comfy_org hidden input."""
    assert "comfy" not in keys.LANES
    _enable_files(monkeypatch, tmp_path)
    (tmp_path / "comfy.secret").write_text("comfy-key\n", encoding="utf-8")
    with pytest.raises(cmb.CloudMediaError):
        cmb.resolve_auth(None)
    logged_in = cmb.resolve_auth("hidden-from-app")
    assert logged_in.kind == "api_key_hidden" and logged_in.value == "hidden-from-app"


def test_injected_key_lands_on_api_key_hidden_input_only():
    from nodes._otr_shared.cloud_media_backend import CloudAuth
    from nodes._otr_shared.cloud_media_invoke import _inject_hidden_inputs

    class _Sess:
        auth = CloudAuth("api_key_hidden", "queue-key")

    row = {"inputs": {"hidden": {
        "api_key_comfy_org": "APIKEY",
        "auth_token_comfy_org": "TOKEN",
    }}}
    out = _inject_hidden_inputs(row, {}, _Sess())
    assert out["api_key_comfy_org"] == "queue-key"
    assert "auth_token_comfy_org" not in out
