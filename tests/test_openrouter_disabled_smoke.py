"""S6 -- disabled-state smoke proof.

With remote OFF (the default), the OpenRouter feature must be completely
inert: no virtual rows in the dropdowns, no remote path reachable, no
extra run-meta on local runs, and the audio byte-identical baseline
fixtures untouched. This is the autonomous half of S6; the ENABLED live
smoke (a real OpenRouter call that spends credits on a GPU episode) is
the one operator gate left for Jeffrey.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb
from nodes import _otr_model_loader as loader
from nodes._otr_model_inputs import UnknownModelError


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


def test_no_openrouter_rows_in_dropdowns(tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert all("openrouter:" not in c for c in choices)


def test_validate_rejects_remote_handles_when_disabled():
    for handle in (A, B):
        with pytest.raises(UnknownModelError):
            cat.validate_model_id(handle)


def test_request_slot_remote_unreachable_when_disabled():
    with pytest.raises(UnknownModelError):
        loader.request_slot("creative", A)


def test_local_run_gets_no_remote_meta():
    # Even the helper yields nothing for two local ids -> a local run's
    # meta is byte-identical to the pre-OpenRouter baseline.
    assert orb.openrouter_meta_for(LOCAL, LOCAL) == {}


def test_audio_baseline_fixtures_present_and_consistent():
    """The C7 byte-identical baseline fixtures are the audio-is-king
    anchor; confirm they are still on disk + internally consistent so
    the operator's enabled/disabled runtime compare has a baseline."""
    here = pathlib.Path(__file__).resolve().parent
    wav = here / "fixtures" / "baseline_v1.5.wav"
    sha = here / "fixtures" / "baseline_v1.5.sha256"
    if not (wav.is_file() and sha.is_file()):
        pytest.skip("audio baseline fixtures not captured on this host")
    import hashlib
    h = hashlib.sha256()
    with open(wav, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert h.hexdigest() == sha.read_text(encoding="utf-8").strip()
