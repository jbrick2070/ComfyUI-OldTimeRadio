"""Writer slots show TEXT models only (2026-06-18).

The OpenRouter A/B pickers drive the LLM writer, so an image generator like
google/gemini-3-pro-image (output_modalities ["image","text"]) must never appear
there -- it returns a planning monologue / a picture, never a usable line. The
cache now captures architecture.output_modalities and the slot filter hides any
non-text-output model (tolerant of an old cache; OTR_OPENROUTER_ALLOW_NONTEXT=1
disables the filter).
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb


# --- cache captures output modality -----------------------------------------

def test_slim_model_captures_output_modalities():
    raw = {"id": "x/y", "architecture": {"output_modalities": ["Text"]}}
    slim = orb._slim_model(raw)
    assert slim["output_modalities"] == ["text"]  # lower-cased


def test_slim_model_missing_architecture_is_empty_list():
    slim = orb._slim_model({"id": "x/y"})
    assert slim["output_modalities"] == []


def test_slim_model_image_generator_records_image():
    raw = {"id": "google/gemini-3-pro-image",
           "architecture": {"output_modalities": ["image", "text"]}}
    assert orb._slim_model(raw)["output_modalities"] == ["image", "text"]


# --- the text-writer predicate ----------------------------------------------

@pytest.mark.parametrize("mods,expected", [
    (["text"], True),
    (["text", "file"], True),
    (["image", "text"], False),     # an image generator that also emits text
    (["image"], False),
    (["audio"], False),
    ([], True),                     # unknown (old cache) -> not hidden
    (None, True),
])
def test_is_text_writer_model(mods, expected):
    assert cat._is_text_writer_model({"id": "x/y", "output_modalities": mods}) is expected


# --- the slot filter --------------------------------------------------------

_MODELS = [
    {"id": "openai/gpt-4o", "output_modalities": ["text"]},
    {"id": "anthropic/claude-opus-4.8", "output_modalities": ["text"]},
    {"id": "google/gemini-3-pro-image", "output_modalities": ["image", "text"]},
    {"id": "openai/gpt-5-image", "output_modalities": ["image"]},
    {"id": "legacy/old-cache-row"},  # no output_modalities key at all
]


def _ids(rows):
    return [m["id"] for m in rows]


def test_filter_hides_image_models_by_default(monkeypatch):
    monkeypatch.delenv("OTR_OPENROUTER_ALLOW_NONTEXT", raising=False)
    for k in ("OTR_OPENROUTER_PROVIDER_FILTER", "OTR_OPENROUTER_MODEL_ALLOWLIST",
              "OTR_OPENROUTER_MODEL_DENYLIST", "OTR_OPENROUTER_SLOT_A_REQUIRE_JSON"):
        monkeypatch.delenv(k, raising=False)
    out = _ids(cat._filter_catalog_models(_MODELS, slot="a"))
    assert "google/gemini-3-pro-image" not in out
    assert "openai/gpt-5-image" not in out
    # text models + the unknown-modality legacy row survive
    assert "openai/gpt-4o" in out
    assert "anthropic/claude-opus-4.8" in out
    assert "legacy/old-cache-row" in out


def test_escape_hatch_shows_nontext(monkeypatch):
    for k in ("OTR_OPENROUTER_PROVIDER_FILTER", "OTR_OPENROUTER_MODEL_ALLOWLIST",
              "OTR_OPENROUTER_MODEL_DENYLIST", "OTR_OPENROUTER_SLOT_A_REQUIRE_JSON"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OTR_OPENROUTER_ALLOW_NONTEXT", "1")
    out = _ids(cat._filter_catalog_models(_MODELS, slot="a"))
    assert "google/gemini-3-pro-image" in out
    assert "openai/gpt-5-image" in out
