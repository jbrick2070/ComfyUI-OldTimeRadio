"""S6 -- disabled-state smoke proof.

With remote OFF (no API key), generate() / backend.load() still fail
closed and local run-meta stays empty. Virtual rows and curated aliases
remain in the dropdowns so a saved deluxe graph loads on a keyless
canvas; the pick is the enable. The ENABLED live smoke (a real
OpenRouter call that spends credits on a GPU episode) is the one
operator gate left for Jeffrey.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb
from nodes import _otr_model_loader as loader
A = "openrouter:slot-a"
B = "openrouter:slot-b"
LOCAL = "mistralai/Mistral-Nemo-Instruct-2407"


@pytest.fixture(autouse=True)
def _disabled(monkeypatch):
    """Force the disabled (default) state regardless of the host env."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)


def test_remote_reports_disabled():
    assert orb.openrouter_enabled() is False


def test_openrouter_rows_in_dropdowns_when_disabled(tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert A in choices
    assert B in choices


def test_validate_admits_remote_handles_when_disabled():
    assert cat.validate_model_id(A) == A
    assert cat.validate_model_id(B) == B


def test_request_slot_remote_fails_closed_when_disabled():
    with pytest.raises(orb.OpenRouterConfigError):
        loader.request_slot("creative", A)


def test_local_run_gets_no_remote_meta():
    # Even the helper yields nothing for two local ids -> a local run's
    # meta is byte-identical to the pre-OpenRouter baseline.
    assert orb.openrouter_meta_for(LOCAL, LOCAL) == {}
