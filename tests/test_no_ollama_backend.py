"""Negative coverage for the removed local sidecar writer path."""
from __future__ import annotations

from pathlib import Path

from nodes import _otr_model_catalog as catalog
from nodes import _otr_model_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_runtime_has_no_removed_sidecar_dispatch_key():
    assert "ollama_local_http" not in runtime.BACKENDS_BY_KEY
    for row in catalog.CURATED_LLM_MODELS:
        assert row.loader_backend != "ollama_local_http"
        assert row.provider != "ollama"


def test_gemma_12b_hf_pin_is_accepted_without_a_sidecar(tmp_path):
    hub = tmp_path / "hub"
    snap = hub / "models--google--gemma-4-12b-it" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        '{"architectures":["Gemma4ForConditionalGeneration"]}',
        encoding="utf-8",
    )

    assert catalog.validate_model_id(
        "google/gemma-4-12b-it",
        auto_download_enabled=True,
        allow_remote=True,
        hub_root=hub,
    ) == "google/gemma-4-12b-it"


def test_the_gemma_12b_row_is_the_transformers_one():
    ids = catalog._by_repo_id()
    assert "google/gemma-4-12b-it" in ids
    hf_row = ids["google/gemma-4-12b-it"]
    assert hf_row.loader_backend == "transformers_multimodal_text_only"
    assert hf_row.provider == "local"
    assert hf_row.requires_auth is False
    assert hf_row.vram_fit_tier == "PASS"
    assert hf_row.context_window == 8192
