"""Direct Google API LLM lane guardrails."""

from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_model_loader as ml
from nodes import _otr_model_runtime as rt
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
from nodes.OTR_LedgerScriptWriter import _build_truncating_generate_fn, _resolve_inputs
from nodes._otr_google_api import client as gclient
from nodes._otr_google_api import llm as gllm
from nodes._otr_google_api import models as gmodels


@pytest.fixture(autouse=True)
def _clean_google(monkeypatch, tmp_path):
    for key in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OTR_GOOGLE_API_MODEL_CACHE", str(tmp_path / "missing.json"))
    gmodels.clear_slot_bindings()
    yield
    gmodels.clear_slot_bindings()


def test_key_precedence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini")
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "otr")
    assert gclient.resolve_api_key() == "otr"


def test_no_key_slot_choices_sentinel_only():
    assert cat.google_api_catalog_dropdown_choices("a") == [
        gmodels.GOOGLE_API_MODEL_UNSELECTED
    ]


def test_key_slot_choices_include_static_text_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    choices = cat.google_api_catalog_dropdown_choices("b")
    assert choices[0] == gmodels.GOOGLE_API_MODEL_UNSELECTED
    assert gmodels.GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT in choices
    assert gmodels.GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT in choices


def test_virtual_rows_present_only_with_key(monkeypatch):
    assert gmodels.GOOGLE_API_SLOT_A_ID not in cat.dropdown_choices()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    labels = cat.dropdown_choices()
    assert gmodels.GOOGLE_API_SLOT_A_ID in labels
    assert gmodels.GOOGLE_API_SLOT_B_ID in labels


def test_runtime_and_loader_register_google(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert gmodels.GOOGLE_API_BACKEND_KEY in rt.BACKENDS_BY_KEY
    assert "google_api" in ml._REMOTE_CACHE_PROVIDERS
    assert ml.request_slot


def test_request_slot_unbound_google_fails_before_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(gmodels.GoogleAPISlotBindingError):
        ml.request_slot("technical", gmodels.GOOGLE_API_SLOT_A_ID)


def test_request_slot_routes_bound_google_without_local_handles(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    gmodels.set_slot_bindings(slot_a=gmodels.GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT)
    entry = ml.request_slot("creative", gmodels.GOOGLE_API_SLOT_A_ID)
    assert entry["provider"] == "google_api"
    assert entry["model_id"] == gmodels.GOOGLE_API_SLOT_A_ID
    assert entry["google_model"] == gmodels.GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT
    assert "model" not in entry and "tokenizer" not in entry


def test_generate_payload_shape(monkeypatch):
    captured = {}

    def fake_create(payload):
        captured["payload"] = payload
        return {"output_text": "signal acquired"}

    monkeypatch.setattr(gllm, "create_interaction", fake_create)
    entry = {
        "provider": "google_api",
        "model_id": gmodels.GOOGLE_API_SLOT_B_ID,
        "google_model": gmodels.GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT,
    }
    fn = gllm.make_google_api_generate_fn(
        entry,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}},
        },
    )
    out = fn(
        [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": "Write a tiny object."},
        ],
        temperature=0.2,
        max_new_tokens=64,
    )
    assert out == "signal acquired"
    assert captured["payload"]["model"] == gmodels.GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT
    assert captured["payload"]["system_instruction"] == "Return only JSON."
    assert captured["payload"]["input"] == "Write a tiny object."
    assert captured["payload"]["response_format"]["mime_type"] == "application/json"
    assert captured["payload"]["generation_config"]["max_output_tokens"] == 64


def test_generate_rejects_wrong_message_shape_before_network(monkeypatch):
    calls = []
    monkeypatch.setattr(gllm, "create_interaction", lambda payload: calls.append(payload))
    entry = {
        "provider": "google_api",
        "model_id": gmodels.GOOGLE_API_SLOT_A_ID,
        "google_model": gmodels.GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT,
    }
    fn = gllm.make_google_api_generate_fn(entry)
    with pytest.raises(gclient.GoogleAPIRequestShapeError):
        fn([{"role": "tool", "content": "bad"}], temperature=0.1, max_new_tokens=8)
    assert calls == []


def test_writer_appends_google_slots_after_visual_style():
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order[23] == "source_bank"
    assert order[24] == "visual_style"
    assert order[25] == "google_api_slot_a_model"
    assert order[26] == "google_api_slot_b_model"
    assert order[27] == "source_ref"


def test_resolve_inputs_threads_google_slots():
    out = _resolve_inputs(
        custom_premise="seed",
        google_api_slot_a_model="gemini-3.5-flash",
        google_api_slot_b_model="gemini-3.1-flash-lite",
    )
    assert out["google_api_slot_a_model"] == "gemini-3.5-flash"
    assert out["google_api_slot_b_model"] == "gemini-3.1-flash-lite"
    assert out["source_ref"] == ""


def test_writer_truncating_wrapper_routes_google(monkeypatch):
    monkeypatch.setattr(
        gllm,
        "create_interaction",
        lambda payload: {"output_text": "from google"},
    )
    entry = {
        "provider": "google_api",
        "model_id": gmodels.GOOGLE_API_SLOT_A_ID,
        "google_model": gmodels.GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT,
    }
    fn = _build_truncating_generate_fn(entry)
    assert getattr(fn, "_otr_google_api", False) is True
    assert fn([{"role": "user", "content": "hi"}], temperature=0.2, max_new_tokens=8) == "from google"


def test_google_run_meta_never_contains_key(monkeypatch):
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "secret-key")
    gmodels.set_slot_bindings(slot_a="gemini-3.5-flash", slot_b="gemini-3.1-flash-lite")
    meta = gmodels.google_api_run_meta(
        gmodels.GOOGLE_API_SLOT_A_ID,
        gmodels.GOOGLE_API_SLOT_B_ID,
    )
    dumped = repr(meta)
    assert "secret-key" not in dumped
    assert meta["google_api_model_bindings"]["slot_a"] == "gemini-3.5-flash"
