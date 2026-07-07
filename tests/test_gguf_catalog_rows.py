"""Catalog contract for the visible native GGUF Gemma 4 12B row."""
from __future__ import annotations

from nodes import _otr_gguf_backend as gguf
from nodes import _otr_model_catalog as cat


def test_row_present_by_default(monkeypatch, tmp_path):
    for key in (
        "GEMMA4_12B_GGUF_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    ids = cat._by_repo_id()
    assert gguf.ROW_ID in ids
    assert gguf.ROW_ID in cat.dropdown_choices(hub_root=tmp_path)
    assert cat.validate_model_id(gguf.ROW_ID) == gguf.ROW_ID


def test_virtual_row_schema():
    row = cat._by_repo_id()[gguf.ROW_ID]
    assert row.repo_id == gguf.ROW_ID
    assert row.loader_backend == gguf.GGUF_BACKEND_KEY
    assert row.provider == "gguf_native"
    assert row.vram_fit_tier == "PASS"
    assert row.approx_safetensors_gb == 13.4
    assert row.requires_auth is False
    assert row.context_window == 8192


def test_dropdown_label_has_no_download_suffix(tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert gguf.ROW_ID in choices
    assert (gguf.ROW_ID + cat.NOT_DOWNLOADED_SUFFIX) not in choices


def test_dropdown_orders_row_as_gemma_peer(tmp_path):
    ids = [entry.repo_id for entry in cat.build_dropdown_choices(hub_root=tmp_path)]
    assert ids.index("google/gemma-4-E4B-it") + 1 == ids.index(gguf.ROW_ID)
    assert ids.index(gguf.ROW_ID) < ids.index("google/gemma-2-2b-it")


def test_structural_reject_passes_virtual_handle():
    assert cat._structural_reject(gguf.ROW_ID) is None


def test_row_is_not_static_curated_model():
    assert gguf.ROW_ID not in {m.repo_id for m in cat.CURATED_LLM_MODELS}
    assert gguf.ROW_ID not in cat.GATED_CURATED_MODELS
