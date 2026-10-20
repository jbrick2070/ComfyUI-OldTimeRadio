"""MOSS-Image SFT: an opt-in local still engine that fetches its own weights.

The pack ships no weights. The three files come from Comfy-Org's ungated
Apache-2.0 repackage ``Comfy-Org/MOSS-Image`` at an exact pinned revision.
These tests pin the registry shape, the licence flags, Comfy's template recipe,
the nothing-installed download set and the fail-closed gate. No GPU, no network.
"""
from __future__ import annotations

import pytest

from nodes import _otr_visual_assets as VA
from nodes._otr_image_engines import moss_image as MI
from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines.registry import EngineUnusable

_REPO = "Comfy-Org/MOSS-Image"
_REVISION = "3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb"

#: (category, basename) -> (size, sha256), read off the Hub API 2026-10-19.
_PINS = {
    ("diffusion_models", "moss_image_sft_int8_convrot.safetensors"): (
        5_413_698_958,
        "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
    ("text_encoders", "qwen3.5_9b_w4a8_gptq.safetensors"): (
        7_343_591_088,
        "4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c"),
    ("vae", "flux2-vae_small.safetensors"): (
        249_521_340,
        "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b"),
}
_SET = set(_PINS)


@pytest.fixture
def no_moss_env(monkeypatch):
    for var in (MI.MODEL_ENV, MI.CLIP_ENV, MI.VAE_ENV, "OTR_MOSS_STEPS",
                "OTR_MOSS_CFG", "OTR_MOSS_SAMPLER", "OTR_MOSS_SCHEDULER",
                "OTR_MOSS_WIDTH", "OTR_MOSS_HEIGHT", "OTR_MOSS_UNET_DTYPE"):
        monkeypatch.delenv(var, raising=False)


def test_registered_opt_in_and_commercial_clean():
    eng = ireg.get_engine("moss_image")
    assert "moss_image" in ireg.all_engine_names()
    assert tuple(eng.default_roles) == ()          # never automatic
    assert eng.commercial_clean is True            # Apache-2.0
    assert eng.requires_flag is None
    assert getattr(eng, "accepts_reference_image", False) is False


def test_a_fresh_install_fetches_the_sft_int8_set(no_moss_env):
    assert VA.planned_downloads({"moss_image"}) == _SET


def test_every_fetched_file_is_pinned_to_comfys_repackage():
    for key, (size, sha256) in _PINS.items():
        spec = VA.MANIFEST[key]
        assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}, key
        assert spec["repo_id"] == _REPO, key
        assert spec["revision"] == _REVISION, key
        assert spec["size"] == size, key
        assert spec["sha256"] == sha256, key
        assert spec["filename"] == "%s/%s" % (key[0], key[1]), key
    # The small Flux 2 VAE is its own file, not the Ideogram 4 flux2-vae row.
    assert VA.MANIFEST[("vae", "flux2-vae.safetensors")]["sha256"] != _PINS[
        ("vae", "flux2-vae_small.safetensors")][1]


def test_the_recipe_is_comfys_template(no_moss_env):
    eng = ireg.get_engine("moss_image")
    p = eng._params({"prompt": "a fox", "seed": 7, "width": 1010, "height": 700})
    assert (p["steps"], p["cfg"], p["sampler_name"], p["scheduler"]) == (
        25, 2.5, "er_sde", "simple")
    assert p["clip_type"] == "moss_image"
    assert (p["clip_name"], p["vae_name"]) == (
        "qwen3.5_9b_w4a8_gptq.safetensors", "flux2-vae_small.safetensors")
    assert (p["width"], p["height"]) == (1008, 704)           # nearest 16
    g = eng._build_graph(p, lambda node, slot: ("wire", node, slot))
    assert g["clip"]["inputs"]["type"] == "moss_image"
    assert g["ksampler"]["inputs"]["cfg"] == 2.5
    assert g["ksampler"]["inputs"]["denoise"] == 1.0
    # No ModelSampling node: shift 5 is the model's own default.
    assert g["ksampler"]["inputs"]["model"] == ("wire", "unet", 0)
    assert set(g) == {"unet", "clip", "vae", "pos", "neg", "latent",
                      "ksampler", "decode"}
    assert set(eng._node_candidates()) == set(g)
    assert eng._node_candidates()["latent"] == ("EmptyFlux2LatentImage",)


def test_the_default_negative_carries_no_content_guardrail(no_moss_env):
    p = ireg.get_engine("moss_image")._params({"prompt": "a fox"})
    assert p["negative"] == "worst quality, bad anatomy, watermark, jpeg artifact"
    assert "nsfw" not in p["negative"].lower()


def test_env_overrides_reach_the_params(monkeypatch, no_moss_env):
    monkeypatch.setenv("OTR_MOSS_STEPS", "50")
    monkeypatch.setenv("OTR_MOSS_CFG", "3")
    monkeypatch.setenv("OTR_MOSS_SAMPLER", "euler")
    p = ireg.get_engine("moss_image")._params({"prompt": "a fox"})
    assert (p["steps"], p["cfg"], p["sampler_name"]) == (50, 3.0, "euler")
    assert (p["width"], p["height"]) == (1024, 1024)


def test_snap_floors_and_rounds_to_16():
    assert MI._snap(1) == 256
    assert MI._snap(1023) == 1024
    assert MI._snap(1040) == 1040


def test_unusable_until_the_model_is_installed(monkeypatch, no_moss_env):
    monkeypatch.setattr(MI, "_installed_unets", lambda: [])
    monkeypatch.setattr(MI, "_core_problem", lambda mapping=None: None)
    with pytest.raises(EngineUnusable):
        ireg.get_engine("moss_image").assert_usable({}, {"role": "character_image"})


def test_usable_when_installed(monkeypatch, no_moss_env):
    monkeypatch.setattr(MI, "_installed_unets",
                        lambda: ["moss_image_sft_int8_convrot.safetensors"])
    monkeypatch.setattr(MI, "_core_problem", lambda mapping=None: None)
    assert ireg.get_engine("moss_image").assert_usable(
        {}, {"role": "character_image"}) == "moss_image"


def test_refuses_an_old_comfyui_core(monkeypatch, no_moss_env):
    monkeypatch.setattr(MI, "_installed_unets",
                        lambda: ["moss_image_sft_int8_convrot.safetensors"])
    monkeypatch.setattr(MI, "_core_problem", lambda mapping=None: MI._CORE_UPDATE_MESSAGE)
    with pytest.raises(EngineUnusable, match="v0.42.0"):
        ireg.get_engine("moss_image").assert_usable({}, {"role": "character_image"})


def _clip_loader(types):
    class _CLIPLoader:
        @staticmethod
        def INPUT_TYPES():
            return {"required": {"clip_name": (["x"],), "type": (list(types),)}}
    return _CLIPLoader


def _core(types):
    return {"EmptyFlux2LatentImage": object, "CLIPLoader": _clip_loader(types)}


def test_the_core_probe_is_the_moss_image_clip_type():
    # v0.41.0 already had EmptyFlux2LatentImage and the flux3 type; only the
    # CLIPLoader `moss_image` type marks v0.42.0.
    assert MI._core_problem(_core(["flux2", "flux3"])) == MI._CORE_UPDATE_MESSAGE
    assert MI._core_problem(_core(["flux3", "moss_image"])) is None
    assert MI._core_problem({}) is None                     # no registry: never fail
    assert "v0.42.0" in MI._CORE_UPDATE_MESSAGE
    assert "'moss_image'" in MI._CORE_UPDATE_MESSAGE


def _full_core(types):
    mapping = _core(types)
    mapping.update({n: object for n in ("UNETLoader", "VAELoader", "CLIPTextEncode",
                                        "KSampler", "VAEDecode")})
    return mapping


def test_the_queue_time_gate_refuses_an_old_core_before_any_download(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _full_core(["flux2", "flux3"])
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    with pytest.raises(VA.VisualAssetError, match="v0.42.0") as err:
        VA._refuse_missing_node_packs({"moss_image"})
    assert "image engine 'moss_image'" in str(err.value)
    assert "Nothing was downloaded" in str(err.value)


def test_the_queue_time_gate_passes_a_current_core(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _full_core(["flux3", "moss_image"])
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    VA._refuse_missing_node_packs({"moss_image"})


def test_base_checkpoints_are_never_auto_picked(monkeypatch):
    import folder_paths  # noqa: F401  (stubbed by conftest if absent)
    names = ["moss_image_base_int8_convrot.safetensors",
             "moss_image_base_bf16.safetensors",
             "moss_image_sft_bf16.safetensors",
             "moss_image_sft_int8_convrot.safetensors"]
    monkeypatch.setattr("folder_paths.get_filename_list", lambda c: names,
                        raising=False)
    assert MI._installed_unets() == ["moss_image_sft_int8_convrot.safetensors",
                                     "moss_image_sft_bf16.safetensors"]


def test_fetch_lane_and_provision_route_name_the_same_pins():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "otr_fetch_lane_weights_for_moss",
        os.path.join(root, "scripts", "otr_fetch_lane_weights.py"))
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    names = {os.path.basename(row[1]) for row in fetch.LANES["moss_image"]}
    assert names == {basename for _category, basename in _SET}
    assert "moss_image" in fetch.LANE_INFO
