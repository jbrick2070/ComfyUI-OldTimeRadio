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
    assert (gguf.ROW_ID + cat.vram_badge_for(gguf.ROW_ID)
            ) in cat.dropdown_choices(hub_root=tmp_path)
    assert cat.validate_model_id(gguf.ROW_ID) == gguf.ROW_ID


def test_virtual_row_schema():
    row = cat._by_repo_id()[gguf.ROW_ID]
    assert row.repo_id == gguf.ROW_ID
    assert row.loader_backend == gguf.GGUF_BACKEND_KEY
    assert row.provider == "gguf_native"
    assert row.vram_fit_tier == "PASS"
    # DERIVED from the pinned Q8_0 bytes (no guessed constant): the first
    # pinned artifact size in GiB, rounded to 1 dp.
    assert row.approx_safetensors_gb == round(
        gguf.EXPECTED_Q8_0_SIZE_BYTES / (1024 ** 3), 1
    )
    assert row.requires_auth is False
    assert row.context_window == 8192   # promoted 2026-08-01; see test_gemma_row_shape


def test_dropdown_gguf_row_is_bare_and_validates(tmp_path):
    """The GGUF row carries NO STATE badge -- no [LOCAL GGUF], no download
    marker. It DOES carry the VRAM figure (operator, 2026-08-01: the picker must
    say what a model needs so only runnable ones get chosen), which is a property
    of the model and identical on every machine. Values saved by an older
    workflow -- bare, or bearing the retired state badge -- still normalize."""
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert (gguf.ROW_ID + cat.vram_badge_for(gguf.ROW_ID)) in choices
    assert (gguf.ROW_ID + cat.LOCAL_GGUF_SUFFIX) not in choices
    assert (gguf.ROW_ID + cat.NOT_DOWNLOADED_SUFFIX) not in choices
    # every stored spelling resolves back to the one canonical id
    assert cat.validate_model_id(gguf.ROW_ID) == gguf.ROW_ID
    assert cat.validate_model_id(gguf.ROW_ID + cat.LOCAL_GGUF_SUFFIX) == gguf.ROW_ID
    assert cat.validate_model_id(
        gguf.ROW_ID + cat.vram_badge_for(gguf.ROW_ID)) == gguf.ROW_ID


def test_dropdown_orders_row_as_gemma_peer(tmp_path):
    ids = [entry.repo_id for entry in cat.build_dropdown_choices(hub_root=tmp_path)]
    assert ids.index("google/gemma-4-E4B-it") + 1 == ids.index(
        "google/gemma-4-12b-it"
    )
    assert ids.index("google/gemma-4-12b-it") + 1 == ids.index(gguf.ROW_ID)
    assert ids.index(gguf.ROW_ID) < ids.index("google/gemma-2-2b-it")


def test_structural_reject_passes_virtual_handle():
    assert cat._structural_reject(gguf.ROW_ID) is None


def test_row_is_not_static_curated_model():
    assert gguf.ROW_ID not in {m.repo_id for m in cat.CURATED_LLM_MODELS}
    assert gguf.ROW_ID not in cat.GATED_CURATED_MODELS
