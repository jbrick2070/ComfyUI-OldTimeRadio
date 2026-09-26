"""The H3 fetch lanes are complete, pinned, one per DiT, and automatic.

Operator 2026-09-25: "keep them as long as they are auto download". The two
MiniMax H3 lanes download their own weights at queue time, so the dev-tree
fetcher and the provisioner treat them like every other automatic lane: one
fetcher lane per DiT (as LTX 2.5 has), routed automatically for a profile that
selects the engine, and the `minimax_h3` bundle for the whole five-file stack.
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib
import re

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"

#: The five-file receipt, written out literally. The fetcher reads these from
#: nodes/_otr_visual_assets.py, so this literal is what catches an accidental
#: edit of the one shared table.
FL2VA = ("Comfy-Org/MiniMax-H3",
         "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         REVISION, 20_970_379_616,
         "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a")
REF2VA = ("Comfy-Org/MiniMax-H3",
          "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
          "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
          REVISION, 20_970_379_616,
          "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779")
ENCODER = ("Comfy-Org/MiniMax-H3",
           "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
           "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
           REVISION, 15_687_142_551,
           "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6")
VIDEO_VAE = ("Comfy-Org/MiniMax-H3",
             "vae/minimax_h3_video_vae_fp16.safetensors",
             "vae/minimax_h3_video_vae_fp16.safetensors",
             REVISION, 5_207_808_496,
             "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522")
AUDIO_VAE = ("Comfy-Org/MiniMax-H3",
             "vae/minimax_h3_audio_vae_fp32.safetensors",
             "vae/minimax_h3_audio_vae_fp32.safetensors",
             REVISION, 605_254_808,
             "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetcher():
    return _load(ROOT / "scripts" / "otr_fetch_lane_weights.py",
                 "_otr_h3_fetch_tests")


def _provisioner():
    return _load(ROOT / "scripts" / "otr_provision.py",
                 "_otr_h3_provision_tests")


def _h3_profile(provision, row_id, video):
    """A matrix row with every video selection moved onto an H3 lane.

    No shipped row selects H3, so the profile is built from a real row rather
    than read from one.
    """
    profile = copy.deepcopy(provision.load_profile(row_id))
    profile["id"] = "%s_%s" % (row_id, video)
    for key in ("announcer_visual", "music_visual", "character_visual"):
        profile["role_overrides"][key] = video
    return profile


def test_each_h3_lane_carries_its_exact_pinned_receipt():
    fetcher = _fetcher()
    spec = fetcher.WeightSpec
    video = [fetcher.weight_spec(row) for row in fetcher.LANES["minimax_h3_video"]]
    audio_in = [fetcher.weight_spec(row)
                for row in fetcher.LANES["minimax_h3_audio_in"]]

    assert video == [spec(*FL2VA), spec(*ENCODER), spec(*VIDEO_VAE)]
    assert audio_in == [spec(*REF2VA), spec(*ENCODER), spec(*VIDEO_VAE),
                        spec(*AUDIO_VAE)]
    for row in video + audio_in:
        assert row.revision == REVISION
        assert re.fullmatch(r"[0-9a-f]{64}", row.expected_sha256)


def test_the_minimax_h3_bundle_is_the_complete_five_file_stack():
    """The old one-command fetch still works and still lands all five files."""
    fetcher = _fetcher()
    assert fetcher.BUNDLES["minimax_h3"] == ["minimax_h3_video",
                                             "minimax_h3_audio_in"]
    union = {}
    for lane in fetcher.BUNDLES["minimax_h3"]:
        for row in fetcher.LANES[lane]:
            union[fetcher.destination_name(row)] = fetcher.weight_spec(row)
    assert len(union) == 5
    assert sum(row.expected_bytes for row in union.values()) == 63_440_965_087
    assert round(sum(row.expected_bytes for row in union.values())
                 / 1024 ** 3, 3) == 59.084


def test_a_bundle_fetches_a_shared_file_once(tmp_path, monkeypatch):
    """Both H3 lanes load the encoder and the video VAE; the bundle must not
    fetch or SHA-verify either of them twice."""
    fetcher = _fetcher()
    fetched = []
    monkeypatch.setattr(fetcher, "models_root", lambda: str(tmp_path))
    monkeypatch.setattr(fetcher, "fetch",
                        lambda entry, root, dry_run: fetched.append(
                            fetcher.destination_name(entry)) or True)
    monkeypatch.setattr("sys.argv", ["otr_fetch_lane_weights.py", "minimax_h3",
                                     "--dry-run"])
    assert fetcher.main() == 0
    assert fetched == [
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ]


def test_each_h3_lane_is_exactly_its_engine_recipe(monkeypatch):
    from nodes._otr_video_engines.eng_minimax_h3 import (
        MiniMaxH3AudioInEngine,
        MiniMaxH3VideoEngine,
    )

    for name in (
        "OTR_MINIMAX_H3_UNET_NAME", "OTR_MINIMAX_H3_CLIP_NAME",
        "OTR_MINIMAX_H3_VAE_NAME", "OTR_MINIMAX_H3_AUDIO_VAE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    fetcher = _fetcher()
    for lane, engine in (("minimax_h3_video", MiniMaxH3VideoEngine()),
                         ("minimax_h3_audio_in", MiniMaxH3AudioInEngine())):
        fetched = {fetcher.destination_name(row) for row in fetcher.LANES[lane]}
        loaded = {default for _label, _cats, default, _floor
                  in engine._weight_rows()}
        assert fetched == loaded, lane


@pytest.mark.parametrize("row_id, video, h3_lane, image_lane", [
    ("otr_8gb_video", "h3_low_video", "minimax_h3_video", "z_image_int8"),
    ("otr_16gb_video", "h3_low_audio_in", "minimax_h3_audio_in", "z_image"),
])
def test_an_h3_profile_routes_its_own_lane_automatically(
        row_id, video, h3_lane, image_lane):
    provision = _provisioner()
    profile = _h3_profile(provision, row_id, video)

    routes = provision.profile_lanes(profile)
    assert routes == {
        "automatic": [h3_lane, image_lane, "stable_audio_3"],
        "manual": [],
    }


def test_every_h3_id_routes_to_an_automatic_lane():
    """Public and internal ids alike -- the dropdown matrix reads `manual`
    off this route, so a manual route would print "manual" again."""
    provision = _provisioner()
    for engine in ("h3_low_video", "h3_low_audio_in",
                   "minimax_h3_video", "minimax_h3_audio_in"):
        lane = provision.lane_for_engine(engine, "video")
        assert lane.manual is False, engine


def test_main_fetches_the_h3_lane_with_the_profile_dependencies(monkeypatch):
    provision = _provisioner()
    fetched = []
    verified = []
    models_root = str(ROOT / "models")

    monkeypatch.setattr(provision, "comfy_root", lambda: str(ROOT.parent.parent))
    monkeypatch.setattr(provision, "models_root", lambda _comfy: models_root)
    # main() is a CLI boundary and intentionally exports this value for the
    # helpers it invokes. Own the value through monkeypatch so the in-process
    # unit test cannot leave its fixture root behind for later resolver tests.
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", models_root)
    monkeypatch.setattr(provision, "ensure_hf_home", lambda _root: None)
    monkeypatch.setattr(provision, "install_node_packs", lambda _comfy: None)
    monkeypatch.setattr(provision, "install_requirements", lambda: None)
    monkeypatch.setattr(
        provision, "fetch_lane_weights", lambda lanes: fetched.append(list(lanes)))
    monkeypatch.setattr(provision, "warm_profile_writer_models", lambda _profile: None)

    def verify(_root, download_id):
        verified.append(download_id)
        return True

    monkeypatch.setattr(provision, "verify_manual_download", verify)
    h3 = _h3_profile(provision, "otr_8gb_video", "h3_low_video")
    monkeypatch.setattr(provision, "load_profile", lambda _profile_id: h3)

    rc = provision.main(["--profile", h3["id"]])

    assert rc == 0
    assert fetched == [["minimax_h3_video", "z_image_int8", "stable_audio_3"]]
    assert verified == []
