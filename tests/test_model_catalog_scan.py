"""S30 B1a tests -- offline catalog dataclass + scan + dropdown + validator.

Pure-Python tests against the new nodes/_otr_model_catalog.py + nodes/_otr_model_inputs.py
module surfaces. No HF API calls; no real GPU; no real downloads.

Coverage:
    * CURATED_LLM_MODELS shape (CuratedModel dataclass, derived sets).
    * scan_local_llm_cache walks a fixture hub root.
    * build_dropdown_choices applies the [NOT DOWNLOADED] suffix.
    * validate_model_id strips suffix, structurally rejects unsafe ids,
      admits on each of curated / locally-scanned / arbitrary-org-name
      paths, raises UnknownModelError with actionable recovery hint on miss.

B1a2 will add tests/test_model_catalog_download.py for the network surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_model_catalog as catalog
from nodes._otr_model_inputs import UnknownModelError, MissingModelInputError, require_model


# ---------------------------------------------------------------------------
# Fixture HF cache
# ---------------------------------------------------------------------------


def _make_snapshot(
    hub_root: Path,
    org: str,
    name: str,
    *,
    advertised_context: int | None = 8192,
) -> Path:
    """Create a fake HF cache snapshot directory structure under `hub_root`.

    Layout matches the real HuggingFace hub cache:
        hub_root/models--<org>--<name>/snapshots/<commit_sha>/config.json
    Returns the snapshot path.
    """
    repo_dir = hub_root / f"models--{org}--{name}"
    snapshot = repo_dir / "snapshots" / "fakecommit0001"
    snapshot.mkdir(parents=True, exist_ok=True)
    if advertised_context is not None:
        cfg = snapshot / "config.json"
        cfg.write_text(
            json.dumps({"max_position_embeddings": advertised_context}),
            encoding="utf-8",
        )
    return snapshot


@pytest.fixture
def empty_hub_root(tmp_path: Path) -> Path:
    """A clean hub root with no models installed."""
    root = tmp_path / "hub"
    root.mkdir()
    return root


@pytest.fixture
def hub_root_with_mistral_nemo(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "mistralai", "Mistral-Nemo-Instruct-2407", advertised_context=131072)
    return root


@pytest.fixture
def hub_root_with_uncurated(tmp_path: Path) -> Path:
    """User has Llama-3-8B locally (not in OTR's curated set)."""
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "meta-llama", "Llama-3-8B-Instruct", advertised_context=8192)
    return root


# ---------------------------------------------------------------------------
# Constants + dataclass shape
# ---------------------------------------------------------------------------


def test_curated_set_is_a_tuple_of_curated_model():
    assert isinstance(catalog.CURATED_LLM_MODELS, tuple)
    assert len(catalog.CURATED_LLM_MODELS) >= 4
    for entry in catalog.CURATED_LLM_MODELS:
        assert isinstance(entry, catalog.CuratedModel)
        assert "/" in entry.repo_id
        assert entry.vram_fit_tier in ("PASS", "WARN", "UNKNOWN", "FAIL")
        assert entry.loader_backend in (
            "transformers_safetensors",
            "transformers_multimodal_text_only",
        )
        assert entry.approx_safetensors_gb > 0


def test_default_llm_constant_is_in_curated_set():
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.DEFAULT_LLM in repo_ids


def test_test_technical_llm_constant_is_in_curated_set():
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.TEST_TECHNICAL_LLM in repo_ids


def test_test_oversized_llm_constant_NOT_in_curated_set():
    """TEST_OVERSIZED_LLM is the 70B target used only by VRAM-fit tests."""
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.TEST_OVERSIZED_LLM not in repo_ids


def test_gated_curated_models_derived_from_requires_auth():
    expected = {m.repo_id for m in catalog.CURATED_LLM_MODELS if m.requires_auth}
    assert set(catalog.GATED_CURATED_MODELS) == expected
    assert catalog.DEFAULT_LLM in catalog.GATED_CURATED_MODELS  # Mistral is gated


def test_default_llm_is_pass_tier_for_c7_baseline():
    """The C7 audio-baseline model MUST be PASS-tier."""
    for m in catalog.CURATED_LLM_MODELS:
        if m.repo_id == catalog.DEFAULT_LLM:
            assert m.vram_fit_tier == "PASS"
            return
    pytest.fail("DEFAULT_LLM not found in curated set")


# ---------------------------------------------------------------------------
# Local cache scan
# ---------------------------------------------------------------------------


def test_scan_empty_hub_returns_empty_list(empty_hub_root):
    assert catalog.scan_local_llm_cache(hub_root=empty_hub_root) == []


def test_scan_with_mistral_nemo_returns_one_result(hub_root_with_mistral_nemo):
    results = catalog.scan_local_llm_cache(hub_root=hub_root_with_mistral_nemo)
    assert len(results) == 1
    r = results[0]
    assert r.repo_id == catalog.DEFAULT_LLM
    assert r.on_disk is True
    assert r.snapshot_path is not None
    assert r.advertised_context == 131072


def test_scan_skips_non_models_dirs(tmp_path):
    root = tmp_path / "hub"
    root.mkdir()
    (root / "datasets--foo--bar").mkdir()
    (root / "models--mistralai--Mistral-Nemo-Instruct-2407" / "snapshots" / "x").mkdir(parents=True)
    results = catalog.scan_local_llm_cache(hub_root=root)
    assert len(results) == 1
    assert results[0].repo_id == catalog.DEFAULT_LLM


# ---------------------------------------------------------------------------
# Dropdown builder
# ---------------------------------------------------------------------------


def test_dropdown_empty_cache_marks_all_curated_not_downloaded(empty_hub_root):
    entries = catalog.build_dropdown_choices(hub_root=empty_hub_root)
    curated_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert {e.repo_id for e in entries} == curated_ids
    for e in entries:
        assert e.on_disk is False
        assert e.label.endswith(catalog.NOT_DOWNLOADED_SUFFIX)


def test_dropdown_with_mistral_nemo_marks_only_that_on_disk(hub_root_with_mistral_nemo):
    entries = catalog.build_dropdown_choices(hub_root=hub_root_with_mistral_nemo)
    for e in entries:
        if e.repo_id == catalog.DEFAULT_LLM:
            assert e.on_disk is True
            assert e.label == catalog.DEFAULT_LLM
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        else:
            assert e.on_disk is False
            assert e.label.endswith(catalog.NOT_DOWNLOADED_SUFFIX)


def test_dropdown_appends_uncurated_locally_scanned_at_end(hub_root_with_uncurated):
    entries = catalog.build_dropdown_choices(hub_root=hub_root_with_uncurated)
    uncurated_entry = next(
        (e for e in entries if e.repo_id == "meta-llama/Llama-3-8B-Instruct"), None
    )
    assert uncurated_entry is not None
    assert uncurated_entry.on_disk is True
    assert uncurated_entry.curated is False
    assert uncurated_entry.label == "meta-llama/Llama-3-8B-Instruct"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_strips_not_downloaded_suffix(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    labelled = catalog.DEFAULT_LLM + catalog.NOT_DOWNLOADED_SUFFIX
    out = catalog.validate_model_id(labelled, hub_root=empty_hub_root)
    assert out == catalog.DEFAULT_LLM


def test_validator_admits_curated(empty_hub_root):
    out = catalog.validate_model_id(catalog.DEFAULT_LLM, hub_root=empty_hub_root)
    assert out == catalog.DEFAULT_LLM


def test_validator_admits_locally_scanned_uncurated(hub_root_with_uncurated, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    out = catalog.validate_model_id(
        "meta-llama/Llama-3-8B-Instruct", hub_root=hub_root_with_uncurated
    )
    assert out == "meta-llama/Llama-3-8B-Instruct"


def test_validator_admits_arbitrary_org_name_when_auto_download_enabled(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    out = catalog.validate_model_id("some-user/some-model", hub_root=empty_hub_root)
    assert out == "some-user/some-model"


def test_validator_rejects_arbitrary_when_auto_download_disabled(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    monkeypatch.delenv("OTR_MODEL_CATALOG_ALLOW_REMOTE", raising=False)
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("some-user/some-model", hub_root=empty_hub_root)
    assert "some-user/some-model" in str(exc.value)
    assert "huggingface-cli" in str(exc.value)


def test_validator_rejects_path_traversal(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("../etc/passwd", hub_root=empty_hub_root)
    assert "traversal" in str(exc.value).lower() or ".." in str(exc.value)


def test_validator_rejects_absolute_path(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("/etc/passwd", hub_root=empty_hub_root)
    assert "absolute" in str(exc.value).lower() or "/" in str(exc.value)


def test_validator_rejects_windows_drive_letter(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError):
        catalog.validate_model_id("C:/Windows/foo", hub_root=empty_hub_root)


def test_validator_rejects_backslash(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError):
        catalog.validate_model_id(r"some\path\here", hub_root=empty_hub_root)


def test_validator_rejects_gguf_extension(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("foo/bar.gguf", hub_root=empty_hub_root)
    assert "gguf" in str(exc.value).lower() or "transformers" in str(exc.value)


def test_validator_recovery_hint_lists_top_installed(hub_root_with_mistral_nemo, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    monkeypatch.delenv("OTR_MODEL_CATALOG_ALLOW_REMOTE", raising=False)
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("totally/unknown-id-xyz", hub_root=hub_root_with_mistral_nemo)
    # Top installed alternative (Mistral-Nemo) should be in the message.
    assert catalog.DEFAULT_LLM in str(exc.value)


# ---------------------------------------------------------------------------
# require_model helper
# ---------------------------------------------------------------------------


def test_require_model_passes_non_empty_string():
    assert require_model(catalog.DEFAULT_LLM, slot="creative") == catalog.DEFAULT_LLM


def test_require_model_raises_on_empty():
    with pytest.raises(MissingModelInputError) as exc:
        require_model("", slot="technical")
    assert "technical" in str(exc.value)


def test_require_model_raises_on_none():
    with pytest.raises(MissingModelInputError):
        require_model(None, slot="creative")


def test_require_model_raises_on_whitespace_only():
    with pytest.raises(MissingModelInputError):
        require_model("   ", slot="creative")
