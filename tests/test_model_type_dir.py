r"""``model_type_dir(category)``: the folder for one model TYPE.

WHY THIS FILE EXISTS. ``_models_root()`` is one folder, and callers joined a type
name onto it. extra_model_paths.yaml sets every type separately, so on the
reference machine ``checkpoints`` resolves first to ``C:\ComfyUI-Models`` while
``upscale_models`` resolves first to the Documents tree (live
``/internal/folder_paths``, 2026-09-25). Inside ComfyUI the owner now asks
ComfyUI per type; outside it the root plus the type is the only real answer.

Every test here CALLS the real function; none copies its logic.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

import pytest

from nodes import _otr_models_root as mr

_ENV = ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def fake_comfy(folders):
    """A folder_paths shaped like ComfyUI's: map_legacy, and a KeyError for an
    unregistered type."""
    legacy = {"unet": "diffusion_models", "clip": "text_encoders"}
    fake = types.ModuleType("folder_paths")
    fake.map_legacy = lambda name: legacy.get(name, name)
    fake.get_folder_paths = lambda name: [str(p) for p in folders[legacy.get(name, name)]]
    return fake


def test_it_is_exported():
    assert "model_type_dir" in mr.__all__


class TestInsideComfyEachTypeGetsItsOwnFolder:
    def test_types_in_different_trees_resolve_to_their_own_first_folder(
            self, clean_env, tmp_path):
        fake = fake_comfy({
            "checkpoints": [tmp_path / "A" / "checkpoints"],
            "upscale_models": [tmp_path / "B" / "upscale_models",
                               tmp_path / "A" / "upscale_models"],
        })
        clean_env.setitem(sys.modules, "folder_paths", fake)
        assert mr.model_type_dir("checkpoints") == tmp_path / "A" / "checkpoints"
        assert mr.model_type_dir("upscale_models") == tmp_path / "B" / "upscale_models"

    def test_an_injected_folder_paths_is_used_over_the_runtime_module(
            self, clean_env, tmp_path):
        clean_env.setitem(sys.modules, "folder_paths", None)
        fake = fake_comfy({"vae": [tmp_path / "injected" / "vae"]})
        assert mr.model_type_dir("vae", folder_paths=fake) == tmp_path / "injected" / "vae"

    def test_a_pin_selects_the_registered_folder_under_it(self, clean_env, tmp_path):
        pin = tmp_path / "pod"
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(pin))
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            "loras": [tmp_path / "desktop" / "loras", pin / "loras"],
        }))
        assert mr.model_type_dir("loras") == pin / "loras"

    def test_a_pin_comfy_does_not_scan_falls_back_to_the_first_folder(
            self, clean_env, tmp_path):
        """A file fetched where the loader never looks would be a fetch the
        render cannot use."""
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "unscanned"))
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            "loras": [tmp_path / "desktop" / "loras"],
        }))
        assert mr.model_type_dir("loras") == tmp_path / "desktop" / "loras"

    @pytest.mark.parametrize("legacy, canonical",
                             [("unet", "diffusion_models"), ("clip", "text_encoders")])
    def test_legacy_names_resolve_to_the_canonical_type(
            self, clean_env, tmp_path, legacy, canonical):
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            canonical: [tmp_path / canonical],
        }))
        assert mr.model_type_dir(legacy) == tmp_path / canonical

    def test_an_unknown_type_falls_back_to_the_root_join(self, clean_env, tmp_path):
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            "checkpoints": [tmp_path / "tree" / "checkpoints"],
        }))
        assert mr.model_type_dir("not_a_type") == tmp_path / "tree" / "not_a_type"

    def test_an_empty_registration_falls_back_to_the_root_join(self, clean_env, tmp_path):
        """VHS_video_formats is registered with an empty list on the live box."""
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            "checkpoints": [tmp_path / "tree" / "checkpoints"],
            "VHS_video_formats": [],
        }))
        assert mr.model_type_dir("VHS_video_formats") == \
            tmp_path / "tree" / "VHS_video_formats"

    def test_off_windows_inside_comfy_it_never_raises(self, clean_env, tmp_path):
        """checkpoints is always registered, so the root resolves through the
        configured tree before any refusal."""
        # Only the owner's view of os goes POSIX: pathlib cannot build a
        # PosixPath on Windows, and the refusal branch reads os.name alone.
        clean_env.setattr(mr, "os", types.SimpleNamespace(
            name="posix", path=os.path, fspath=os.fspath))
        clean_env.setattr(pathlib.Path, "is_dir", lambda self: False)
        clean_env.setattr(os.path, "isdir", lambda p: False)
        clean_env.setitem(sys.modules, "folder_paths", fake_comfy({
            "checkpoints": [tmp_path / "tree" / "checkpoints"],
        }))
        assert mr.model_type_dir("upscale_models") == tmp_path / "tree" / "upscale_models"


class TestOutsideComfyItIsTheRootPlusTheType:
    @pytest.mark.parametrize("stub", [None, "conftest_shape"])
    def test_without_a_usable_folder_paths_it_is_the_root_join(self, clean_env, stub):
        clean_env.setitem(sys.modules, "folder_paths",
                          None if stub is None else types.ModuleType("folder_paths"))
        assert mr.model_type_dir("vae") == mr._models_root() / "vae"

    def test_an_env_pin_is_exclusive(self, clean_env, tmp_path):
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "pin"))
        assert mr.model_type_dir("checkpoints") == tmp_path / "pin" / "checkpoints"

    def test_the_first_pin_beats_the_second(self, clean_env, tmp_path):
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "first"))
        clean_env.setenv("COMFYUI_MODELS_ROOT", str(tmp_path / "second"))
        assert mr.model_type_dir("vae") == tmp_path / "first" / "vae"

    def test_a_pin_is_user_expanded(self, clean_env):
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", "~/otr-weights")
        assert "~" not in str(mr.model_type_dir("vae"))

    @pytest.mark.parametrize("legacy, canonical",
                             [("unet", "diffusion_models"), ("clip", "text_encoders")])
    def test_legacy_names_are_canonical_off_the_runtime_too(
            self, clean_env, tmp_path, legacy, canonical):
        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
        assert mr.model_type_dir(legacy) == tmp_path / canonical


class TestTheRealSitesUseIt:
    def test_the_wan_resolver_finds_a_weight_only_in_the_types_first_folder(
            self, clean_env, tmp_path):
        """Behavioural: get_full_path misses, the comfy-root join misses, and
        the weight exists only in the folder ComfyUI lists first for the type."""
        from nodes._otr_video_engines.wan_shared import WanInitImageMixin

        first = tmp_path / "desktop" / "diffusion_models"
        first.mkdir(parents=True)
        (first / "wan.safetensors").write_bytes(b"\x00")
        fake = fake_comfy({"diffusion_models": [first],
                           "checkpoints": [tmp_path / "desktop" / "checkpoints"]})
        fake.get_full_path = lambda category, name: None
        clean_env.setitem(sys.modules, "folder_paths", fake)
        clean_env.setattr(WanInitImageMixin, "_comfy_root",
                          staticmethod(lambda: str(tmp_path / "nowhere")))
        got = WanInitImageMixin._resolve_model_file_by_token(
            WanInitImageMixin(), ["diffusion_models"], "wan.safetensors")
        assert pathlib.Path(got) == first / "wan.safetensors"


class TestTheHfCacheDefault:
    def test_on_windows_it_is_the_models_root(self, clean_env, monkeypatch):
        from nodes import _otr_hf_env as hf

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(hf, "_DEFAULT_HF_HOME_WINDOWS",
                            str(pathlib.Path("Z:/no/such/huggingface")))
        assert hf._default_hf_home() == str(mr._models_root() / "huggingface")

    def test_an_existing_populated_cache_keeps_its_place(self, monkeypatch, tmp_path):
        from nodes import _otr_hf_env as hf

        old = tmp_path / "huggingface"
        (old / "hub").mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(hf, "_DEFAULT_HF_HOME_WINDOWS", str(old))
        assert hf._default_hf_home() == str(old)


class TestIndexTts2Refs:
    def test_the_default_refs_dir_is_under_the_models_root(self):
        import importlib.util

        repo = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "otr_dl_indextts2_refs_under_test", repo / "scripts" / "otr_dl_indextts2_refs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert pathlib.Path(module.DEFAULT_REFS_DIR) == \
            mr._models_root() / "TTS" / "refs" / "indextts2"

    def test_the_voice_ref_resolver_tries_the_owner_before_the_literal(
            self, clean_env, tmp_path):
        from nodes._otr_audio_engines import base

        clean_env.setitem(sys.modules, "folder_paths", None)
        clean_env.setattr(mr, "_models_root", lambda: tmp_path)
        ref = tmp_path / "TTS" / "refs" / "voice.wav"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"RIFF")
        got = base.resolve_voice_ref_path("models/TTS/refs/voice.wav")
        assert os.path.normpath(got) == str(ref)
