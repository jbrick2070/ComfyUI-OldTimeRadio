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
    # Edit / multi-reference mode is unmeasured for face identity: OFF.
    assert getattr(eng, "accepts_reference_image", False) is False


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
    monkeypatch.setattr(FK, "_core_has_nodes", lambda: False)
    with pytest.raises(EngineUnusable, match="v0.41.0"):
        ireg.get_engine("flux3_klein").assert_usable({}, {"role": "character_image"})


def test_base_checkpoints_are_never_auto_picked(monkeypatch):
    import folder_paths  # noqa: F401  (stubbed by conftest if absent)
    names = ["flux3-image-klein-base-bf16.safetensors",
             "flux3-image-klein-bf16.safetensors"]
    monkeypatch.setattr("folder_paths.get_filename_list", lambda c: names,
                        raising=False)
    assert FK._installed_unets() == ["flux3-image-klein-bf16.safetensors"]
