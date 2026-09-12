"""flux2_klein resolves its GGUF in the same order as the shared models root.

`_otr_gguf_backend._models_root()` lets `OTR_COMFYUI_MODELS_ROOT` /
`COMFYUI_MODELS_ROOT` win outright -- that is how every pod run is pinned.
`flux2_klein._resolve_unet_path` used to ask the native loader FIRST and the
env roots second, so a box with a root pinned and a second copy visible to
`folder_paths` rendered stills from one file while the GGUF writers read
another, silently. Found by the codex contrarian on the 2026-09-12 arc batch,
which refuted "nothing" for the model-root row.

Order now: explicit override; the env root's file when it exists; the native
loader; the env candidate registered as before (absent -> loud MISSING_MODEL).
"""
from __future__ import annotations

import sys
import types

import pytest


def _fake_folder_paths(monkeypatch, found_path, registered):
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda category, name: (
        str(found_path) if category == "unet" and found_path else None)
    fake.add_model_folder_path = lambda category, path, is_default=False: (
        registered.append((category, path, is_default)))
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    return fake


@pytest.mark.parametrize("root_env", ["OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"])
def test_an_env_root_that_holds_the_file_beats_the_native_loader(
        monkeypatch, tmp_path, root_env):
    from nodes._otr_image_engines import flux2_klein as module

    env_copy = tmp_path / "pinned" / "diffusion_models" / module._DEFAULT_CKPT
    env_copy.parent.mkdir(parents=True)
    env_copy.write_bytes(b"pinned copy")
    loader_copy = tmp_path / "native" / module._DEFAULT_CKPT
    loader_copy.parent.mkdir(parents=True)
    loader_copy.write_bytes(b"native copy")
    registered = []
    _fake_folder_paths(monkeypatch, loader_copy, registered)
    monkeypatch.delenv(module.MODEL_ENV, raising=False)
    for name in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(root_env, str(tmp_path / "pinned"))

    assert module._resolve_unet_path() == str(env_copy)
    assert registered == [("unet", str(env_copy.parent), True)]


def test_the_native_loader_wins_when_the_env_root_lacks_the_file(
        monkeypatch, tmp_path):
    from nodes._otr_image_engines import flux2_klein as module

    loader_copy = tmp_path / "native" / module._DEFAULT_CKPT
    loader_copy.parent.mkdir(parents=True)
    loader_copy.write_bytes(b"native copy")
    (tmp_path / "pinned" / "diffusion_models").mkdir(parents=True)   # empty
    registered = []
    _fake_folder_paths(monkeypatch, loader_copy, registered)
    monkeypatch.delenv(module.MODEL_ENV, raising=False)
    monkeypatch.delenv("COMFYUI_MODELS_ROOT", raising=False)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "pinned"))

    assert module._resolve_unet_path() == str(loader_copy)
    assert registered == []


def test_the_first_configured_root_is_the_one_consulted(monkeypatch, tmp_path):
    """Both variables set: the FIRST configured root is the candidate, exactly
    as `_otr_gguf_backend._models_root()` reads them -- it does not scan the
    second root when the first lacks the file (codex asked for a scan; that
    would diverge from the shared resolver this fix exists to match). With
    the first root empty the native loader wins, as before."""
    from nodes._otr_image_engines import flux2_klein as module

    second_copy = tmp_path / "second" / "diffusion_models" / module._DEFAULT_CKPT
    second_copy.parent.mkdir(parents=True)
    second_copy.write_bytes(b"second root copy")
    (tmp_path / "first" / "diffusion_models").mkdir(parents=True)   # empty
    loader_copy = tmp_path / "native" / module._DEFAULT_CKPT
    loader_copy.parent.mkdir(parents=True)
    loader_copy.write_bytes(b"native copy")
    registered = []
    _fake_folder_paths(monkeypatch, loader_copy, registered)
    monkeypatch.delenv(module.MODEL_ENV, raising=False)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "first"))
    monkeypatch.setenv("COMFYUI_MODELS_ROOT", str(tmp_path / "second"))

    assert module._resolve_unet_path() == str(loader_copy)
    assert registered == []


def test_with_no_env_root_the_order_is_unchanged(monkeypatch, tmp_path):
    """The reference box sets neither variable: native loader, then the bare
    default name -- exactly what it did before."""
    from nodes._otr_image_engines import flux2_klein as module

    registered = []
    _fake_folder_paths(monkeypatch, None, registered)
    for name in (module.MODEL_ENV, "OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    assert module._resolve_unet_path() == module._DEFAULT_CKPT
    assert registered == []
