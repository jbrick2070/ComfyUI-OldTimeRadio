"""S3 -- slot resolution + preservation + the validator admit-path.

Pins the 2026-06-01 go-forward plan S5 preservation core:
  * resolve_slug runs on the STORED slot binding; env is DEMOTED to a fallback.
  * Case 2: a bound slug is used VERBATIM; if absent from a stale/cold cache it
    is warned-about but still attempted -- never silently swapped.
  * Case 1: an empty / placeholder-sentinel binding falls through the chain
    SLOT_x_DEFAULT -> OPENROUTER_MODEL_x -> recommended -> config error.
  * The otr_api widget-value validator ADMITS an out-of-list OpenRouter slug /
    handle (BUG-LOCAL-280) instead of rejecting it at patch time.
  * openrouter_run_meta stamps the resolved slugs + catalog staleness.
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_PACK_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from otr_api import _validate_widget_value  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Clear the process-global bindings + the slot env knobs around each
    test so resolution is deterministic and no binding leaks out of the file."""
    orb.clear_slot_bindings()
    for k in ("OTR_OPENROUTER_SLOT_A_DEFAULT", "OTR_OPENROUTER_SLOT_B_DEFAULT"):
        monkeypatch.delenv(k, raising=False)
    yield
    orb.clear_slot_bindings()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")


def _write_cache(cache_dir, ids):
    payload = {
        "schema_version": orb.CATALOG_SCHEMA_VERSION,
        "fetched_at": "2026-06-01T00:00:00+00:00",
        "source": "live", "count": len(ids),
        "models": [{"id": i, "name": i, "provider": i.split("/")[0],
                    "created": 1, "supports_json": True} for i in ids],
    }
    (cache_dir / "openrouter_models.json").write_text(
        json.dumps(payload), encoding="utf-8")


# --- case 2: bound slug is used verbatim ------------------------------------


def test_bound_slug_used_verbatim(enabled):
    orb.set_slot_bindings(slot_a="vendor/specific-model", slot_b="vendor/tech")
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/specific-model"
    assert orb.resolve_slug(orb.SLOT_B_ID) == "vendor/tech"


def test_bound_slug_present_in_cache_no_warning(enabled, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    _write_cache(tmp_path, ["anthropic/claude-opus-4.8"])
    orb.set_slot_bindings(slot_a="anthropic/claude-opus-4.8")
    with caplog.at_level(logging.WARNING, logger="nodes._otr_openrouter_backend"):
        assert orb.resolve_slug(orb.SLOT_A_ID) == "anthropic/claude-opus-4.8"
    assert "not in the local catalog cache" not in caplog.text


def test_bound_slug_absent_from_cache_warns_but_attempts(enabled, monkeypatch, tmp_path, caplog):
    """Preservation: a saved slug absent from a stale/cold cache is WARNED
    about but used as-is (never swapped to a cached model)."""
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    _write_cache(tmp_path, ["anthropic/claude-opus-4.8"])  # cache lacks the bound slug
    orb.set_slot_bindings(slot_a="exotic/rare-model-not-cached")
    with caplog.at_level(logging.WARNING, logger="nodes._otr_openrouter_backend"):
        resolved = orb.resolve_slug(orb.SLOT_A_ID)
    assert resolved == "exotic/rare-model-not-cached"  # verbatim, no swap
    assert "not in the local catalog cache" in caplog.text


# --- case 1: placeholder / empty binding -> fallback chain ------------------


def test_placeholder_sentinel_treated_as_unset(enabled, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/env-model")
    orb.set_slot_bindings(slot_a=cat.OPENROUTER_ENABLE_SENTINEL)
    # The sentinel is not a slug -> falls through to the env fallback.
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/env-model"


def test_empty_binding_treated_as_unset(enabled, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/env-model")
    orb.set_slot_bindings(slot_a="   ")
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/env-model"


def test_clear_slot_bindings_returns_to_fallback(enabled, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/env-model")
    orb.set_slot_bindings(slot_a="vendor/bound")
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/bound"
    orb.clear_slot_bindings()
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/env-model"


# --- the otr_api validator admit-path (BUG-LOCAL-280) -----------------------


def test_validator_admits_out_of_list_slot_slug():
    """A saved slot slug absent from the current (cache-derived) choice list
    must be ADMITTED, not rejected -- the cache may be stale/cold."""
    spec = ([cat.OPENROUTER_ENABLE_SENTINEL], {})  # remote-disabled choice list
    # Must not raise.
    _validate_widget_value(
        "OTR_LedgerScriptWriter", "openrouter_slot_a_model", spec,
        "anthropic/claude-opus-4.8",
    )


def test_validator_admits_openrouter_handle():
    spec = (["mistralai/Mistral-Nemo-Instruct-2407"], {})
    _validate_widget_value(
        "OTR_LedgerScriptWriter", "creative_writing_model", spec,
        "openrouter:slot-a",
    )


def test_validator_still_rejects_unrelated_out_of_list():
    spec = (["balanced", "high", "low"], {})
    with pytest.raises(ValueError):
        _validate_widget_value(
            "OTR_LedgerScriptWriter", "creativity", spec, "nonsense-value",
        )


# --- run-meta stamping ------------------------------------------------------


def test_run_meta_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    assert orb.openrouter_run_meta() == {}


def test_run_meta_stamps_resolved_slugs_and_cache(enabled, monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    _write_cache(tmp_path, ["anthropic/claude-opus-4.8", "deepseek/deepseek-v4-pro"])
    orb.set_slot_bindings(
        slot_a="anthropic/claude-opus-4.8", slot_b="deepseek/deepseek-v4-pro")
    meta = orb.openrouter_run_meta()
    assert meta["slot_a_resolved_slug"] == "anthropic/claude-opus-4.8"
    assert meta["slot_b_resolved_slug"] == "deepseek/deepseek-v4-pro"
    assert meta["openrouter_catalog_staleness"] in ("live", "cache", "stale", "empty")
    assert meta["openrouter_catalog_source"] == "live"


# --- writer wiring source-pin ----------------------------------------------


def test_writer_sets_slot_bindings_and_stamps_run_meta():
    """A refactor must not silently drop the binding wire or the run-meta
    stamp from the writer's run() path."""
    from tests.fixtures.writer_family import family_source

    text = family_source()
    assert "set_slot_bindings(" in text
    assert "openrouter_run_meta(" in text
