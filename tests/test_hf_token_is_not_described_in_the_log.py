"""The server log says nothing about the Hugging Face token.

ComfyUI serves its log at /internal/logs with no login. Until 2.3.64 the pack
logged, at INFO on every start, where the token was found (the process
environment or HKCU) and how many characters long it is. Neither is the token,
and both are facts about a credential that a caller of an unauthenticated
endpoint has no business reading.

The resolver still works exactly as before; what it found is written at DEBUG
only. The token used here is a made-up string. CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import logging

from nodes._otr_shared import hf_token

_FAKE = "hf_" + "notARealToken" * 2          # 29 characters, invented


def _fresh(monkeypatch):
    monkeypatch.setitem(hf_token._HF_TOKEN_CACHE, "resolved", False)
    monkeypatch.setitem(hf_token._HF_TOKEN_CACHE, "value", None)


def test_a_found_token_leaves_nothing_at_info(monkeypatch, caplog):
    _fresh(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", _FAKE)
    with caplog.at_level(logging.INFO):
        assert hf_token.ensure_hf_token() == _FAKE
    shown = "\n".join(record.getMessage() for record in caplog.records
                      if record.levelno >= logging.INFO)
    assert "hf_token" not in shown
    assert _FAKE not in shown
    assert "len=" not in shown and str(len(_FAKE)) not in shown
    assert "os.environ" not in shown and "HKCU" not in shown


def test_no_token_leaves_nothing_at_info_either(monkeypatch, caplog):
    """Absence is a fact about the credential too."""
    _fresh(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_token, "_read_from_winreg", lambda: None)
    with caplog.at_level(logging.INFO):
        assert hf_token.ensure_hf_token() is None
    assert not [record for record in caplog.records
                if record.levelno >= logging.INFO and "hf_token" in record.getMessage()]


def test_the_token_itself_is_never_logged_at_any_level(monkeypatch, caplog):
    _fresh(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", _FAKE)
    with caplog.at_level(logging.DEBUG):
        hf_token.ensure_hf_token()
    assert _FAKE not in "\n".join(record.getMessage() for record in caplog.records)
