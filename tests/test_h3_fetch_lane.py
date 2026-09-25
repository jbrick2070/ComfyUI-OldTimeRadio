"""The explicit H3 lane is complete, pinned, and never auto-selected."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import pathlib
import re
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"


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

    No shipped row selects H3 -- it is operator-local by design -- so the
    profile is built from a real row rather than read from one.
    """
    profile = copy.deepcopy(provision.load_profile(row_id))
    profile["id"] = "%s_%s" % (row_id, video)
    profile["slot_overrides"]["video_render_engine"] = video
    for key in ("announcer_visual", "music_visual", "character_visual"):
        profile["role_overrides"][key] = video
    return profile


def test_h3_lane_has_exact_five_file_receipt():
    fetcher = _fetcher()
    entries = [fetcher.weight_spec(row) for row in fetcher.LANES["minimax_h3"]]

    assert entries == [
        fetcher.WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            REVISION, 20_970_379_616,
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"),
        fetcher.WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            REVISION, 20_970_379_616,
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
        fetcher.WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            REVISION, 15_687_142_551,
            "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
        fetcher.WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            REVISION, 5_207_808_496,
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
        fetcher.WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            REVISION, 605_254_808,
            "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
    ]
    assert sum(row.expected_bytes for row in entries) == 63_440_965_087
    assert round(sum(row.expected_bytes for row in entries) / 1024 ** 3, 3) == 59.084
    assert all(row.revision == REVISION for row in entries)
    assert all(re.fullmatch(r"[0-9a-f]{64}", row.expected_sha256)
               for row in entries)


def test_h3_fetcher_is_the_union_of_both_engine_recipes(monkeypatch):
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
    fetched = {
        fetcher.destination_name(row) for row in fetcher.LANES["minimax_h3"]
    }
    loaded = {
        default
        for engine in (MiniMaxH3VideoEngine(), MiniMaxH3AudioInEngine())
        for _label, _categories, default, _floor in engine._weight_rows()
    }

    assert fetched == loaded
    assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in fetched
    assert "minimax_h3_ref2va_pruned_int8_convrot.safetensors" in fetched


@pytest.mark.parametrize("row_id, video, image_lane", [
    ("otr_8gb_video", "h3_low_video", "z_image_int8"),
    ("otr_16gb_video", "h3_low_audio_in", "z_image"),
])
def test_every_h3_profile_stays_operator_only(row_id, video, image_lane):
    provision = _provisioner()
    profile = _h3_profile(provision, row_id, video)

    routes = provision.profile_lanes(profile)
    assert routes == {
        "automatic": [image_lane, "stable_audio_3"],
        "manual": ["h3_operator_only"],
    }
    assert "minimax_h3" not in routes["automatic"]


def test_operator_download_prints_the_exact_explicit_command():
    provision = _provisioner()
    detail = provision.OPERATOR_ONLY_DOWNLOADS["h3_operator_only"]
    assert "python scripts/otr_fetch_lane_weights.py minimax_h3" in detail
    assert "auto-selected" in detail


def test_operator_download_becomes_present_only_after_exact_explicit_fetch(
        tmp_path, monkeypatch):
    provision = _provisioner()
    payload = b"receipt-bearing operator-only H3 fixture"
    artifact = SimpleNamespace(
        destination="diffusion_models/h3-fixture.safetensors",
        path_in_repo="diffusion_models/h3-fixture.safetensors",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    fetcher = SimpleNamespace(
        LANES={"minimax_h3": [artifact]},
        weight_spec=lambda entry: entry,
        destination_path=lambda root, entry: str(
            pathlib.Path(root) / pathlib.PurePosixPath(entry.destination)),
    )
    monkeypatch.setattr(provision, "_load_fetcher_manifest", lambda: fetcher)

    final = tmp_path / "diffusion_models" / "h3-fixture.safetensors"
    final.parent.mkdir(parents=True)
    part = pathlib.Path(str(final) + ".part")
    part.write_bytes(payload)

    assert provision.verify_manual_download(
        str(tmp_path), "h3_operator_only") is False
    assert not final.exists()

    part.replace(final)
    assert provision.verify_manual_download(
        str(tmp_path), "h3_operator_only") is True


def test_main_fetches_h3_profile_dependencies_then_verifies_operator_lane(
        monkeypatch):
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
    assert fetched == [["z_image_int8", "stable_audio_3"]]
    assert verified == ["h3_operator_only"]
    assert all("minimax_h3" not in lanes for lanes in fetched)
