"""The explicit H3 lane is complete, pinned, and never auto-selected."""
from __future__ import annotations

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


def test_every_h3_profile_stays_operator_only():
    provision = _provisioner()
    profiles = []
    for path in sorted((ROOT / "config" / "profiles").glob("*.json")):
        profile = provision.load_profile(path.stem)
        roles = profile.get("role_overrides") or {}
        slots = profile.get("slot_overrides") or {}
        selected = str(slots.get("video_render_engine") or
                       roles.get("character_visual") or "")
        if provision._PUBLIC_VIDEO_IDS.get(selected, selected) in provision._H3_ENGINES:
            profiles.append(profile)

    assert profiles, "no H3 profiles found"
    for profile in profiles:
        images = {
            profile["role_overrides"].get(name)
            for name in ("announcer_image", "music_image", "character_image")
        }
        if images == {"lumina_image"}:
            with pytest.raises(provision.ProvisionFailure,
                               match="unrecognized image engine"):
                provision.profile_lanes(profile)
            continue

        expected_automatic = [
            "z_image_int8"
            if float(profile["llm"]["vram_ceiling_gb"]) <= 8.0
            else "z_image"
        ]
        if profile["slot_overrides"].get("music_engine") == "stable_audio_3":
            expected_automatic.append("stable_audio_3")
        routes = provision.profile_lanes(profile)
        assert routes == {
            "automatic": expected_automatic,
            "manual": ["h3_operator_only"],
        }
        assert "minimax_h3" not in routes["automatic"]


def test_operator_tier_prints_the_exact_explicit_command():
    provision = _provisioner()
    detail = provision.OPERATOR_ONLY_TIERS["h3_operator_only"]
    assert "python scripts/otr_fetch_lane_weights.py minimax_h3" in detail
    assert "auto-selected" in detail


def test_operator_tier_becomes_present_only_after_exact_explicit_fetch(
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

    assert provision.verify_manual_tier(
        str(tmp_path), "h3_operator_only") is False
    assert not final.exists()

    part.replace(final)
    assert provision.verify_manual_tier(
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

    def verify(_root, tier_id):
        verified.append(tier_id)
        return True

    monkeypatch.setattr(provision, "verify_manual_tier", verify)

    rc = provision.main(["--profile", "otr_4060_h3_nano"])

    assert rc == 0
    assert fetched == [["z_image_int8", "stable_audio_3"]]
    assert verified == ["h3_operator_only"]
    assert all("minimax_h3" not in lanes for lanes in fetched)
