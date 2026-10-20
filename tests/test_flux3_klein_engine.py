"""FLUX 3 Image Klein: an opt-in local still engine that fetches its own weights.

The pack ships no weights. Black Forest Labs' repos are gated, so the three
files come from the ungated mirror ``ml-resources/flux-3-image-klein-comfy`` at an
exact pinned revision (copied byte for byte: the mirror's blob ids equal BFL's).
These tests pin the registry shape, the licence flags, the BFL recipe, the
nothing-installed download set and the fail-closed gate. No GPU, no network.
"""
from __future__ import annotations

import math

import pytest

from nodes import _otr_visual_assets as VA
from nodes._otr_image_engines import flux3_klein as FK
from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines.registry import EngineUnusable

_SET = {
    ("diffusion_models", "flux3-image-klein-int8-convrot.safetensors"),
    ("text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
    ("vae", "flux2-vae.safetensors"),
}


@pytest.fixture
def no_flux3_env(monkeypatch):
    for var in (FK.MODEL_ENV, FK.CLIP_ENV, FK.VAE_ENV):
        monkeypatch.delenv(var, raising=False)


def test_registered_opt_in_and_flagged_non_commercial_clean():
    eng = ireg.get_engine("flux3_klein")
    assert "flux3_klein" in ireg.all_engine_names()
    assert tuple(eng.default_roles) == ()          # never automatic
    assert eng.commercial_clean is False           # FLUX Kommunity: Qualifying Users
    assert eng.requires_flag is None
    # The portrait-reference path is opt-in until a measured A/B flips it.
    assert getattr(eng, "accepts_reference_image", False) is False


def test_reference_switch_advertises_and_wires_the_portrait(monkeypatch, no_flux3_env):
    eng = ireg.get_engine("flux3_klein")
    request = {"prompt": "a fox", "seed": 7, "reference_image": "C:/p/c02.png"}
    monkeypatch.delenv("OTR_FLUX3_REFERENCE", raising=False)
    # Off: the reference is dropped at the adapter, so the base graph runs.
    assert eng._params(request)["reference_image"] == ""
    monkeypatch.setenv("OTR_FLUX3_REFERENCE", "1")
    assert eng.accepts_reference_image is True
    p = eng._params(request)
    assert p["reference_image"] == "C:/p/c02.png"
    g = eng._build_graph(p, lambda node, slot: ("wire", node, slot))
    assert set(g) - set(eng._node_candidates()) == set(eng._REF_CANDIDATES)
    assert g["load_ref"]["inputs"]["image"] == "C:/p/c02.png"
    assert g["scale_ref"]["inputs"]["height"] == 768
    assert g["ksampler"]["inputs"]["positive"] == ("wire", "ref_pos", 0)
    assert g["ksampler"]["inputs"]["negative"] == ("wire", "ref_neg", 0)
    assert g["ref_pos"]["inputs"]["conditioning"] == ("wire", "pos", 0)
    assert g["ref_pos"]["inputs"]["latent"] == ("wire", "encode_ref", 0)
    # A request with no reference renders the plain t2i graph even when on.
    plain = eng._build_graph(eng._params({"prompt": "a fox"}),
                             lambda node, slot: ("wire", node, slot))
    assert set(plain) == set(eng._node_candidates())


def test_a_fresh_install_fetches_the_int8_set(no_flux3_env):
    assert VA.planned_downloads({"flux3_klein"}) == _SET


def test_every_fetched_file_is_pinned_and_not_gated_repo():
    for key in _SET:
        spec = VA.MANIFEST[key]
        assert set(spec) == {"repo_id", "filename", "revision", "size", "sha256"}, key
        assert len(spec["revision"]) == 40 and len(spec["sha256"]) == 64
    # The two Klein rows come from the ungated mirror, never a gated BFL repo.
    for key in sorted(_SET - {("vae", "flux2-vae.safetensors")}):
        assert VA.MANIFEST[key]["repo_id"] == "ml-resources/flux-3-image-klein-comfy"
        assert not VA.MANIFEST[key]["repo_id"].startswith("black-forest-labs/")
    assert VA.MANIFEST[("diffusion_models", "flux3-image-klein-int8-convrot.safetensors")][
        "size"] == 5_827_874_779
    assert VA.MANIFEST[("text_encoders", "qwen3vl_4b_int8_convrot.safetensors")][
        "size"] == 4_858_559_240


def test_the_recipe_is_bfls_distilled_t2i(no_flux3_env):
    eng = ireg.get_engine("flux3_klein")
    p = eng._params({"prompt": "a fox", "seed": 7, "width": 1010, "height": 700})
    assert (p["steps"], p["cfg"], p["sampler_name"], p["scheduler"]) == (
        16, 1.0, "euler", "simple")
    assert p["clip_type"] == "flux3"
    assert math.isclose(p["shift"], 2.77259, rel_tol=1e-5)    # ln(16)
    assert (p["width"], p["height"]) == (1008, 704)           # nearest 16
    g = eng._build_graph(p, lambda node, slot: ("wire", node, slot))
    assert g["sampling"]["inputs"]["max_shift"] == g["sampling"]["inputs"]["base_shift"]
    assert g["clip"]["inputs"]["type"] == "flux3"
    assert g["ksampler"]["inputs"]["cfg"] == 1.0
    assert set(g) == {"unet", "clip", "vae", "sampling", "pos", "neg", "latent",
                      "ksampler", "decode"}
    assert set(eng._node_candidates()) == set(g)


def test_snap_floors_and_rounds_to_16():
    assert FK._snap(1) == 256
    assert FK._snap(1023) == 1024
    assert FK._snap(1040) == 1040


def test_unusable_until_the_model_is_installed(monkeypatch, no_flux3_env):
    monkeypatch.setattr(FK, "_installed_unets", lambda: [])
    with pytest.raises(EngineUnusable):
        ireg.get_engine("flux3_klein").assert_usable({}, {"role": "character_image"})


def test_usable_when_installed(monkeypatch, no_flux3_env):
    monkeypatch.setattr(FK, "_installed_unets",
                        lambda: ["flux3-image-klein-int8-convrot.safetensors"])
    assert ireg.get_engine("flux3_klein").assert_usable(
        {}, {"role": "character_image"}) == "flux3_klein"


def test_refuses_an_old_comfyui_core(monkeypatch, no_flux3_env):
    monkeypatch.setattr(FK, "_installed_unets",
                        lambda: ["flux3-image-klein-int8-convrot.safetensors"])
    monkeypatch.setattr(FK, "_core_problem", lambda mapping=None: FK._CORE_UPDATE_MESSAGE)
    with pytest.raises(EngineUnusable, match="v0.41.0"):
        ireg.get_engine("flux3_klein").assert_usable({}, {"role": "character_image"})


def _clip_loader(types):
    class _CLIPLoader:
        @staticmethod
        def INPUT_TYPES():
            return {"required": {"clip_name": (["x"],), "type": (list(types),)}}
    return _CLIPLoader


def _core(types):
    return {"EmptyFlux2LatentImage": object, "ModelSamplingFlux": object,
            "CLIPLoader": _clip_loader(types)}


def test_the_core_probe_is_the_flux3_clip_type_not_the_node_names():
    # v0.40.0 already had EmptyFlux2LatentImage and ModelSamplingFlux (Ideogram 4
    # uses them); only the CLIPLoader `flux3` type marks v0.41.0.
    assert FK._core_problem(_core(["flux2", "qwen_image"])) == FK._CORE_UPDATE_MESSAGE
    assert FK._core_problem(_core(["flux2", "flux3"])) is None
    assert FK._core_problem({}) is None                     # no registry: never fail


def test_the_queue_time_gate_refuses_an_old_core_before_any_download(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _core(["flux2"])
    mapping.update({n: object for n in ("UNETLoader", "VAELoader", "CLIPTextEncode",
                                        "KSampler", "VAEDecode")})
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    with pytest.raises(VA.VisualAssetError, match="v0.41.0") as err:
        VA._refuse_missing_node_packs({"flux3_klein"})
    assert "image engine 'flux3_klein'" in str(err.value)
    assert "Nothing was downloaded" in str(err.value)


def test_the_queue_time_gate_passes_a_current_core(monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge as WB
    mapping = _core(["flux2", "flux3"])
    mapping.update({n: object for n in ("UNETLoader", "VAELoader", "CLIPTextEncode",
                                        "KSampler", "VAEDecode")})
    monkeypatch.setattr(WB, "node_class_mappings",
                        lambda m=None: m if m is not None else mapping)
    VA._refuse_missing_node_packs({"flux3_klein"})


def test_base_checkpoints_are_never_auto_picked(monkeypatch):
    import folder_paths  # noqa: F401  (stubbed by conftest if absent)
    names = ["flux3-image-klein-base-bf16.safetensors",
             "flux3-image-klein-bf16.safetensors"]
    monkeypatch.setattr("folder_paths.get_filename_list", lambda c: names,
                        raising=False)
    assert FK._installed_unets() == ["flux3-image-klein-bf16.safetensors"]
