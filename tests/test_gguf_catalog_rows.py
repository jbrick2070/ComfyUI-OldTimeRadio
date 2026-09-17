"""Writer GGUF is gone from the catalog and the COMBO."""
from __future__ import annotations

import pytest

from nodes import _otr_gguf_backend as gguf
from nodes import _otr_model_catalog as cat
from nodes._otr_model_inputs import UnknownModelError
from nodes._otr_model_loader import ModelLoaderError, request_slot
from nodes._otr_shared import llm_policy as lp


def test_gguf_writer_is_not_a_catalog_row():
    ids = cat._by_repo_id()
    assert gguf.ROW_ID not in ids
    assert gguf.ROW_ID not in {m.repo_id for m in cat.CURATED_LLM_MODELS}
    assert gguf.ROW_ID not in cat.GATED_CURATED_MODELS
    assert not gguf.GGUF_ROWS


def test_gguf_writer_is_not_in_the_combo(tmp_path):
    ids = [entry.repo_id for entry in cat.build_dropdown_choices(hub_root=tmp_path)]
    labels = cat.dropdown_choices(hub_root=tmp_path)
    assert gguf.ROW_ID not in ids
    assert all("gguf" not in label.lower() for label in labels)
    assert "google/gemma-4-12b-it" in ids
    gemma = ids.index("google/gemma-4-12b-it")
    assert ids[gemma - 1] == "google/gemma-4-E4B-it"
    assert ids[gemma + 1] == "google/gemma-2-2b-it"


@pytest.mark.parametrize(
    "label",
    [
        gguf.ROW_ID,
        gguf.ROW_ID + cat.LOCAL_GGUF_SUFFIX,
        gguf.ROW_ID + cat.vram_badge_for(gguf.ROW_ID),
        "unsloth/Qwen3-8B-GGUF",
        "someone/local-model.gguf",
    ],
)
def test_validator_rejects_retired_gguf_writers(tmp_path, label):
    with pytest.raises(UnknownModelError, match="retired GGUF writer"):
        cat.validate_model_id(label, hub_root=tmp_path)


def test_auto_download_refuses_gguf_writer(tmp_path):
    with pytest.raises(UnknownModelError, match="retired GGUF writer"):
        cat.auto_download_if_missing(gguf.ROW_ID, hub_root=tmp_path)


def test_request_slot_refuses_gguf_writer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GGUF writer must fail before load")

    monkeypatch.setattr("nodes._otr_model_loader.load_llm", forbidden)
    with pytest.raises((UnknownModelError, ModelLoaderError), match="retired GGUF writer"):
        request_slot(
            "creative",
            gguf.ROW_ID,
            policy=lp.LLMRuntimePolicy(vram_ceiling_gb=16.0, quant_policy="none"),
        )


def test_gemma_12b_owns_nf4():
    row = cat._by_repo_id()["google/gemma-4-12b-it"]
    assert row.implied_quant_policy == "bnb_nf4"
    assert row.loader_backend == "transformers_multimodal_text_only"
    assert row.provider == "local"
    assert cat.quant_pick_mismatch("google/gemma-4-12b-it", "none") is None
    assert cat.quant_pick_mismatch("google/gemma-4-12b-it", "bnb_nf4") is None
    assert cat.effective_quant_policy("google/gemma-4-12b-it", "none") == "bnb_nf4"


def test_video_and_image_gguf_weight_files_stay(tmp_path):
    """Writer GGUF is gone. Foley / mime / Klein / LTX / Wan still name
    their .gguf artifacts. Those filenames must not become writer rows;
    a writer dropdown pick of one is refused. Video loaders resolve the
    same files through folder_paths, not validate_model_id."""
    from nodes._otr_image_engines.flux2_klein import _DEFAULT_CKPT
    from nodes._otr_video_engines.eng_wan_ti2v import (
        _TI2V_DEFAULT_CLIP,
        _TI2V_DEFAULT_UNET,
    )
    from nodes._otr_video_engines.ltx25_recipe import (
        LTX25_DIT_GGUF,
        LTX25_TEXT_ENCODER_GGUF,
    )

    artifacts = (
        LTX25_DIT_GGUF,
        LTX25_TEXT_ENCODER_GGUF,
        _DEFAULT_CKPT,
        _TI2V_DEFAULT_UNET,
        _TI2V_DEFAULT_CLIP,
        "ltx-2.3-22b-dev-Q3_K_M.gguf",
    )
    ids = cat._by_repo_id()
    curated = {m.repo_id for m in cat.CURATED_LLM_MODELS}
    for name in artifacts:
        assert name.endswith(".gguf"), name
        assert name not in ids
        assert name not in curated
        with pytest.raises(UnknownModelError, match="retired GGUF writer"):
            cat.validate_model_id(name, hub_root=tmp_path)


def test_foley_and_mime_engines_still_registered():
    from nodes._otr_video_engines.eng_ltx25 import _JOINT_AV_ENGINES

    assert "ltx25_foley_plus" in _JOINT_AV_ENGINES
    assert "ltx25_mime" in _JOINT_AV_ENGINES
