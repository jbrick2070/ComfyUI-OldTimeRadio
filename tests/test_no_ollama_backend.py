"""Negative coverage for the removed local sidecar writer path."""
from __future__ import annotations

from pathlib import Path

import pytest

from nodes import _otr_model_catalog as catalog
from nodes import _otr_model_runtime as runtime
from nodes._otr_model_inputs import UnknownModelError


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_removed_backend_module_is_gone():
    assert not (REPO_ROOT / "nodes" / "_otr_ollama_backend.py").exists()


def test_runtime_has_no_removed_sidecar_dispatch_key():
    assert "ollama_local_http" not in runtime.BACKENDS_BY_KEY
    for row in catalog.CURATED_LLM_MODELS:
        assert row.loader_backend != "ollama_local_http"
        assert row.provider != "ollama"


def test_generate_factories_have_no_removed_sidecar_branch():
    for rel in (
        "nodes/_otr_model_loader.py",
        "nodes/OTR_LedgerScriptWriter.py",
        "nodes/_otr_constrained_generate.py",
        "nodes/_otr_model_runtime.py",
    ):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "_otr_ollama_backend" not in src
        assert 'provider") == "ollama' not in src
        assert "ollama_local_http" not in src


def test_stale_gemma_12b_pin_is_blocked_even_if_hf_shape_is_valid(tmp_path):
    hub = tmp_path / "hub"
    snap = hub / "models--google--gemma-4-12b-it" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        '{"architectures":["Gemma4ForConditionalGeneration"]}',
        encoding="utf-8",
    )

    with pytest.raises(UnknownModelError, match="local_gemma4_12b"):
        catalog.validate_model_id(
            "google/gemma-4-12b-it",
            auto_download_enabled=True,
            allow_remote=True,
            hub_root=hub,
        )


def test_local_openai_handle_is_the_only_gemma_12b_catalog_route(monkeypatch):
    monkeypatch.setenv("GEMMA4_12B_ENABLED", "1")
    ids = catalog._by_repo_id()
    assert "local_gemma4_12b" in ids
    assert "google/gemma-4-12b-it" not in ids
    row = ids["local_gemma4_12b"]
    assert row.loader_backend == "local_openai_http"
    assert row.provider == "local_openai"
