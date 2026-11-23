"""Qwen-Image 2.1: an opt-in local still engine that fetches its own weights.

The pack ships no weights. The three files come from Comfy-Org's ungated
repackage ``Comfy-Org/Qwen-Image-2.1`` at an exact pinned revision, under the Qwen
Research License (non-commercial only). These tests pin the registry shape, the
licence flags, Comfy's template recipe, the nothing-installed download set, the
fail-closed gate, and the alpha flattening a 4-channel decode needs. No GPU, no
network.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from nodes import _otr_visual_assets as VA
from nodes._otr_image_engines import qwen_image_21 as QI
from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines.registry import EngineUnusable
from nodes._otr_shared import shortcodes
from nodes._otr_shared.role_compat import ROLES

_REPO = "Comfy-Org/Qwen-Image-2.1"
_REVISION = "aacf31a6d48e707f352ecba714cee0680da3479b"

#: (category, basename) -> (size, sha256), read off the Hub API and recomputed from
#: the downloaded bytes on the 5080, 2026-11-22.
_PINS = {
    ("diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"): (
        7_256_783_064,
        "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d"),
    ("text_encoders", "qwen3vl_8b_int8_convrot.safetensors"): (
        9_350_798_360,
        "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"),
    ("vae", "qwen_image_2.1_vae_bf16.safetensors"): (
        675_509_688,
        "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
}
_SET = set(_PINS)


@pytest.fixture
def no_qwen_env(monkeypatch):
    for var in (QI.MODEL_ENV, QI.CLIP_ENV, QI.VAE_ENV, "OTR_QWEN21_STEPS",
                "OTR_QWEN21_CFG", "OTR_QWEN21_SAMPLER", "OTR_QWEN21_SCHEDULER",
                "OTR_QWEN21_WIDTH", "OTR_QWEN21_HEIGHT", "OTR_QWEN21_UNET_DTYPE"):
        monkeypatch.delenv(var, raising=False)


def test_registered_opt_in_non_commercial_and_serves_every_slot():
    eng = ireg.get_engine("qwen_image_21")
    assert "qwen_image_21" in ireg.all_engine_names()
    assert tuple(eng.default_roles) == ()          # never automatic
    assert tuple(eng.roles) == tuple(ROLES)        # announcer, character and music slots
    assert len(set(eng.roles)) == 3
    assert eng.commercial_clean is False           # Qwen Research License
    assert eng.requires_flag is None
    assert eng.required_inputs == ("text_prompt",)
    assert getattr(eng, "accepts_reference_image", False) is False


def test_it_is_declared_cuda_only_because_nothing_else_is_measured():
    row = ireg.CAPABILITIES["qwen_image_21"]
    assert row["device_backends"] == ["cuda"]
    assert row["requires_sidecar"] is False


def test_the_shortcode_is_registered_and_unique():
    code = shortcodes.IMAGE_GEN["qwen_image_21"]
    assert code == "qwi2"
    assert list(shortcodes.IMAGE_GEN.values()).count(code) == 1


def test_a_fresh_install_fetches_the_int8_set(no_qwen_env):
    assert VA.planned_downloads({"qwen_image_21"}) == _SET


def test_every_fetched_file_is_pinned_to_comfys_repackage():
    for key, (size, sha256) in _PINS.items():
        spec = VA.MANIFEST[key]
        assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}, key
        assert spec["repo_id"] == _REPO, key
        assert spec["revision"] == _REVISION, key
        assert spec["size"] == size, key
        assert spec["sha256"] == sha256, key
        assert spec["filename"] == "%s/%s" % (key[0], key[1]), key


def test_the_recipe_is_comfys_template(no_qwen_env):
    eng = ireg.get_engine("qwen_image_21")
    p = eng._params({"prompt": "a fox", "seed": 7, "width": 1010, "height": 700})
    assert (p["steps"], p["cfg"], p["sampler_name"], p["scheduler"]) == (
        25, 1.0, "euler", "simple")
    assert p["clip_type"] == "qwen_image"
    assert (p["clip_name"], p["vae_name"]) == (
        "qwen3vl_8b_int8_convrot.safetensors", "qwen_image_2.1_vae_bf16.safetensors")
    assert (p["width"], p["height"]) == (1024, 704)           # nearest 32
    g = eng._build_graph(p, lambda node, slot: ("wire", node, slot))
    assert g["clip"]["inputs"]["type"] == "qwen_image"
    assert g["ksampler"]["inputs"]["cfg"] == 1.0
    assert g["ksampler"]["inputs"]["denoise"] == 1.0
    # The model reaches the sampler THROUGH the cache node, as in Comfy's template.
    assert g["cache"]["inputs"]["model"] == ("wire", "unet", 0)
    assert g["ksampler"]["inputs"]["model"] == ("wire", "cache", 0)
    # One encode node feeds both conditionings: slot 0 positive, slot 1 negative.
    assert g["ksampler"]["inputs"]["positive"] == ("wire", "encode", 0)
    assert g["ksampler"]["inputs"]["negative"] == ("wire", "encode", 1)
    assert g["encode"]["inputs"]["prompt"] == "a fox"
    assert g["decode"]["inputs"]["samples"] == ("wire", "ksampler", 0)
    assert set(g) == {"unet", "clip", "vae", "cache", "encode", "latent",
                      "ksampler", "decode"}
    assert set(eng._node_candidates()) == set(g)
    assert eng._node_candidates()["encode"] == ("TextEncodeQwenImage21",)
    # text-to-image: no reference pictures reach the encoder.
    assert not any(k.startswith("images") for k in g["encode"]["inputs"])


def test_every_wire_in_the_graph_points_at_a_node_in_the_graph(no_qwen_env):
    eng = ireg.get_engine("qwen_image_21")
    p = eng._params({"prompt": "a fox"})
    refs = []

    def wire(node, slot):
        refs.append(node)
        return ("wire", node, slot)

    g = eng._build_graph(p, wire)
    assert refs and set(refs) <= set(g)


def test_the_default_negative_carries_no_content_guardrail(no_qwen_env):
    p = ireg.get_engine("qwen_image_21")._params({"prompt": "a fox"})
    assert p["negative"] == ""          # inert at cfg 1.0; nothing to filter


def test_env_overrides_reach_the_params(monkeypatch, no_qwen_env):
    monkeypatch.setenv("OTR_QWEN21_STEPS", "50")
    monkeypatch.setenv("OTR_QWEN21_CFG", "3")
    monkeypatch.setenv("OTR_QWEN21_SAMPLER", "heun")
    p = ireg.get_engine("qwen_image_21")._params({"prompt": "a fox"})
    assert (p["steps"], p["cfg"], p["sampler_name"]) == (50, 3.0, "heun")
    assert (p["width"], p["height"]) == (1024, 1024)


def test_snap_floors_and_rounds_to_32():
    assert QI._snap(1) == 256
    assert QI._snap(1010) == 1024
    assert QI._snap(1050) == 1056
    assert QI._snap(2048) == 2048


def test_unusable_until_the_model_is_installed(monkeypatch, no_qwen_env):
    monkeypatch.setattr(QI, "_installed_unets", lambda: [])
    monkeypatch.setattr(QI, "_core_problem", lambda mapping=None: None)
    with pytest.raises(EngineUnusable):
        ireg.get_engine("qwen_image_21").assert_usable({}, {"role": "character_image"})


def test_usable_when_installed(monkeypatch, no_qwen_env):
    monkeypatch.setattr(QI, "_installed_unets",
                        lambda: ["qwen_image_2.1_int8_convrot.safetensors"])
    monkeypatch.setattr(QI, "_core_problem", lambda mapping=None: None)
    assert ireg.get_engine("qwen_image_21").assert_usable(
        {}, {"role": "character_image"}) == "qwen_image_21"


def test_refuses_an_old_comfyui_core(monkeypatch, no_qwen_env):
    monkeypatch.setattr(QI, "_installed_unets",
                        lambda: ["qwen_image_2.1_int8_convrot.safetensors"])
    monkeypatch.setattr(QI, "_core_problem", lambda mapping=None: QI._CORE_UPDATE_MESSAGE)
    with pytest.raises(EngineUnusable, match="v0.37.0"):
        ireg.get_engine("qwen_image_21").assert_usable({}, {"role": "character_image"})


def _core(missing=()):
    names = ("TextEncodeQwenImage21", "QwenImage21Cache", "EmptyLatentImage",
             "CLIPLoader", "UNETLoader", "VAELoader", "KSampler", "VAEDecode")
    return {n: object for n in names if n not in missing}


def test_the_core_probe_is_the_qwen_image_21_encode_node():
    # An older core still has CLIPLoader type `qwen_image` (Qwen-Image 1); only the
    # 2.1 encode node marks v0.37.0.
    assert QI._core_problem(_core(missing=("TextEncodeQwenImage21",))) == QI._CORE_UPDATE_MESSAGE
    assert QI._core_problem(_core(missing=("QwenImage21Cache",))) == QI._CORE_UPDATE_MESSAGE
    assert QI._core_problem(_core()) is None
    assert QI._core_problem({}) is None                     # no registry: never fail
    assert "v0.37.0" in QI._CORE_UPDATE_MESSAGE
    assert "Text Encode Qwen Image 2.1" in QI._CORE_UPDATE_MESSAGE


def test_the_queue_time_gate_refuses_an_old_core_before_any_download(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _core(missing=("TextEncodeQwenImage21",))
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    with pytest.raises(VA.VisualAssetError, match="v0.37.0") as err:
        VA._refuse_missing_node_packs({"qwen_image_21"})
    assert "image engine 'qwen_image_21'" in str(err.value)
    assert "Nothing was downloaded" in str(err.value)


def test_the_queue_time_gate_passes_a_current_core(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _core()
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    VA._refuse_missing_node_packs({"qwen_image_21"})


def test_turbo_and_other_qwen_image_builds_are_never_auto_picked(monkeypatch):
    import folder_paths  # noqa: F401  (stubbed by conftest if absent)
    names = ["qwen_image_2.1_turbo_int8_convrot.safetensors",
             "qwen_image_2.1_turbo_bf16.safetensors",
             "qwen_image_2.1_bf16.safetensors",
             "qwen_image_2.1_int8_convrot.safetensors",
             "qwen_image_2512_bf16.safetensors",          # a different model
             "qwen_image_edit_2511_fp8.safetensors"]      # a different model
    monkeypatch.setattr("folder_paths.get_filename_list", lambda c: names,
                        raising=False)
    assert QI._installed_unets() == ["qwen_image_2.1_int8_convrot.safetensors",
                                     "qwen_image_2.1_bf16.safetensors"]


def test_fetch_lane_and_provision_route_name_the_same_pins():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "otr_fetch_lane_weights_for_qwen21",
        os.path.join(root, "scripts", "otr_fetch_lane_weights.py"))
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    names = {os.path.basename(row[1]) for row in fetch.LANES["qwen_image_21"]}
    assert names == {basename for _category, basename in _SET}
    assert "qwen_image_21" in fetch.LANE_INFO


# --- the 4-channel decode ----------------------------------------------------------
def _frames(*pixels):
    """A (1,1,N,4) uint8 batch from RGBA tuples."""
    return np.array([[list(pixels)]], dtype=np.uint8)


def test_an_rgb_batch_comes_back_untouched():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)
    assert np.array_equal(QI._flatten_alpha(rgb), rgb)


def test_a_plain_still_keeps_its_exact_rgb_bytes():
    # Measured: an ordinary prompt decodes with alpha 0.984-1.0. That is the opaque
    # noise floor, not transparency, so the colours must not shift toward white.
    out = QI._flatten_alpha(_frames((10, 20, 30, 255), (10, 20, 30, 251), (200, 100, 50, 252)))
    assert out.shape == (1, 1, 3, 3)
    assert out[0, 0].tolist() == [[10, 20, 30], [10, 20, 30], [200, 100, 50]]


def test_real_transparency_is_composited_over_white():
    out = QI._flatten_alpha(_frames((0, 0, 0, 0), (0, 0, 0, 128), (40, 40, 40, 0)))
    assert out[0, 0, 0].tolist() == [255, 255, 255]           # fully transparent -> white
    assert out[0, 0, 2].tolist() == [255, 255, 255]           # whatever RGB sat underneath
    assert out[0, 0, 1].tolist() == [127, 127, 127]           # half alpha -> mid blend
    assert out.dtype == np.uint8


def test_an_unexpected_channel_count_is_refused_not_guessed():
    with pytest.raises(ValueError, match="3 or 4 channels"):
        QI._flatten_alpha(np.zeros((1, 2, 2, 2), dtype=np.uint8))


def test_render_image_really_flattens_the_decode():
    # A test that calls the helper proves the helper, never the wiring: assert the
    # call exists at the one site that returns pixels to the dispatcher.
    src = inspect.getsource(QI.QwenImage21Engine.render_image)
    assert "_flatten_alpha(_wb.images_to_uint8(" in src
    assert "return frames[0]" in src
