"""S2 -- OpenRouter virtual catalog rows are enabled-gated.

Pins the contract: the two virtual rows (openrouter:slot-a|b) appear in
both writer dropdowns and pass validate_model_id Path 1 ONLY when remote
is enabled (OPENROUTER_API_KEY + OTR_ENABLE_OPENROUTER=1); when disabled
they are absent everywhere and validate raises cleanly -- with NO change
to the validator's admit-paths (FC4). The dropdown shows the named
handle, never the real model slug.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes._otr_model_inputs import UnknownModelError


SLOT_A = "openrouter:slot-a"
SLOT_B = "openrouter:slot-b"


@pytest.fixture
def disabled(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OTR_ENABLE_OPENROUTER"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")


# ---------------------------------------------------------------------------
# Disabled (default) -- rows absent everywhere
# ---------------------------------------------------------------------------


def test_rows_absent_from_by_repo_id_when_disabled(disabled):
    ids = cat._by_repo_id()
    assert SLOT_A not in ids
    assert SLOT_B not in ids


def test_rows_absent_from_dropdown_when_disabled(disabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert all("openrouter:" not in c for c in choices)


def test_validate_raises_when_disabled(disabled):
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_A)
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_B)


def test_flag_without_key_stays_disabled(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    assert SLOT_A not in cat._by_repo_id()
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_A)


# ---------------------------------------------------------------------------
# Enabled -- rows present, validated, FC4 schema
# ---------------------------------------------------------------------------


def test_rows_present_in_by_repo_id_when_enabled(enabled):
    ids = cat._by_repo_id()
    assert SLOT_A in ids
    assert SLOT_B in ids


def test_rows_present_in_both_dropdowns_when_enabled(enabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert SLOT_A in choices
    assert SLOT_B in choices
    # The same builder feeds BOTH writer dropdowns, so presence here ==
    # presence in creative + technical.


def test_validate_admits_unchanged_when_enabled(enabled):
    assert cat.validate_model_id(SLOT_A) == SLOT_A
    assert cat.validate_model_id(SLOT_B) == SLOT_B


def test_virtual_rows_match_fc4_schema(enabled):
    row = cat._by_repo_id()[SLOT_A]
    assert row.loader_backend == "openrouter_http"
    assert row.provider == "openrouter"
    assert row.vram_fit_tier == "PASS"
    assert row.approx_safetensors_gb == 0.0
    assert row.context_window == 8192
    assert row.requires_auth is False


def test_dropdown_label_has_no_not_downloaded_suffix(enabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    # The remote handle is shown clean -- it is not a local download.
    assert SLOT_A in choices
    assert (SLOT_A + cat.NOT_DOWNLOADED_SUFFIX) not in choices


def test_dropdown_never_shows_real_slug(enabled, tmp_path):
    """The named handle is shown; the operator's real model slug (bound
    in env) must never leak into the dropdown."""
    choices = cat.dropdown_choices(hub_root=tmp_path)
    joined = " ".join(choices)
    assert "anthropic/claude-3.5-sonnet" not in joined
    assert "openai/gpt-4o" not in joined


# ---------------------------------------------------------------------------
# FC4: no validator surgery needed -- the id is structurally safe
# ---------------------------------------------------------------------------


def test_structural_reject_passes_openrouter_ids():
    assert cat._structural_reject(SLOT_A) is None
    assert cat._structural_reject(SLOT_B) is None


def test_gated_set_unaffected_by_virtual_rows(enabled):
    """GATED_CURATED_MODELS is keyed off the real gated set; the
    requires_auth=False virtual rows must never appear there."""
    assert SLOT_A not in cat.GATED_CURATED_MODELS
    assert SLOT_B not in cat.GATED_CURATED_MODELS
