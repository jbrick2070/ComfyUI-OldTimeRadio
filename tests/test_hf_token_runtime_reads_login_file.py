"""`hf auth login` must actually work with OTR.

PBUG-20260829-10. `resolve_hf_token()` read the process environment and HKCU
and nothing else, so the documented, recommended way to authenticate -- a
cached `hf auth login`, which writes a plain file containing the token -- was
invisible. The README could not honestly recommend it.

The subtle half: OTR relocates HF_HOME to the canonical models root, and the
Hub derives its token path FROM HF_HOME. A login done in an ordinary shell
lands in the user's default cache while OTR-inside-ComfyUI looks elsewhere --
same machine, same user, same token, two paths. Both are checked.
"""
from __future__ import annotations

import os

from nodes import _otr_hf_auth as auth


def _blind_the_env(monkeypatch):
    """Remove every non-file source so only the file paths can answer."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(auth, "resolve_hf_token", lambda: None)


def test_runtime_resolver_finds_a_cached_login(monkeypatch):
    _blind_the_env(monkeypatch)
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub",
        type("M", (), {"get_token": staticmethod(lambda: "hf_from_cached_login")}))
    assert auth.resolve_hf_token_runtime() == "hf_from_cached_login"


def test_runtime_resolver_falls_back_to_the_default_token_file(monkeypatch, tmp_path):
    """The path the Hub STOPS seeing once OTR relocates HF_HOME."""
    _blind_the_env(monkeypatch)
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub",
        type("M", (), {"get_token": staticmethod(lambda: None)}))
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    # A real token file is the raw token and nothing else -- no quotes, no key=.
    (home / ".cache" / "huggingface" / "token").write_text(
        "hf_from_default_file\n", encoding="utf-8")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    assert auth.resolve_hf_token_runtime() == "hf_from_default_file"


def test_env_still_wins_over_any_file(monkeypatch):
    monkeypatch.setattr(auth, "resolve_hf_token", lambda: "hf_from_env")
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub",
        type("M", (), {"get_token": staticmethod(lambda: "hf_from_file")}))
    assert auth.resolve_hf_token_runtime() == "hf_from_env"


def test_no_token_anywhere_is_a_valid_answer(monkeypatch, tmp_path):
    """Every ungated model must keep working with no credential at all."""
    _blind_the_env(monkeypatch)
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub",
        type("M", (), {"get_token": staticmethod(lambda: None)}))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path / "empty") if p == "~" else p)
    assert auth.resolve_hf_token_runtime() is None


def test_a_broken_hub_does_not_break_a_public_load(monkeypatch, tmp_path):
    _blind_the_env(monkeypatch)
    def boom():
        raise RuntimeError("hub exploded")
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub",
        type("M", (), {"get_token": staticmethod(boom)}))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path / "empty") if p == "~" else p)
    assert auth.resolve_hf_token_runtime() is None


def test_import_time_resolver_stays_pure_stdlib():
    """Node registration must not import the Hub client."""
    import inspect
    src = inspect.getsource(auth.resolve_hf_token)
    assert "huggingface_hub" not in src, (
        "the import-time resolver reaches for the Hub client; registration "
        "must stay pure stdlib")
