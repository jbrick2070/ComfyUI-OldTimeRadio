"""FLUX.2 [klein] 4B image engine -- BUILT-engine contract suite (C8).

flux2_klein graduated from a NotImplementedError stub peer (test_image_engine_
matrix_peers.py) to a real, in-stack engine with an implemented ``render_image``
(the GGUF + custom-sampler ComfyUI flux2 recipe: UnetLoaderGGUF + CLIPLoader
[type=flux2] + VAELoader + CLIPTextEncode + FluxGuidance -> BasicGuider;
EmptyFlux2LatentImage + Flux2Scheduler + KSamplerSelect + RandomNoise ->
SamplerCustomAdvanced -> VAEDecode). This file holds its dedicated CPU contract
coverage (registry / explicit selection / weight gate / env constants / cold-import)
PLUS the graph-build + render-drive coverage, all without a GPU (the live render
is the operator verify-on-5080).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest

from nodes._otr_image_engines import registry as ireg


def test_klein_registered_optin_clean():
    eng = ireg.get_engine("flux2_klein")
    assert eng.name == "flux2_klein"
    assert eng.default_roles == ()                 # opt-in peer, never "primary"
    assert eng.commercial_clean is True            # FLUX.2 klein 4B = Apache-2.0
    assert eng.required_inputs == ("text_prompt",)
    assert eng.requires_flag is None               # registry IS the menu (no flag gate)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")        # reduced prompt->image set


def test_klein_weight_gate(monkeypatch, tmp_path):
    from nodes._otr_image_engines import flux2_klein as module

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_full_path = lambda *_args: None
    fake_folder_paths.add_model_folder_path = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.delenv("OTR_COMFYUI_MODELS_ROOT", raising=False)
    monkeypatch.delenv("COMFYUI_MODELS_ROOT", raising=False)
    eng = ireg.get_engine("flux2_klein")
    # (1) Registry IS the menu: a registered engine is selectable, no flag gate.
    monkeypatch.delenv("OTR_ENABLE_FLUX2_KLEIN", raising=False)
    assert ireg.assert_usable("flux2_klein", "music_visual") == "flux2_klein"
    # (2) engine-level: no weight -> MISSING_MODEL (fail-closed, BUG-046).
    monkeypatch.delenv("OTR_FLUX2_KLEIN_CKPT", raising=False)
    with pytest.raises(ireg.EngineUnusable) as ei2:
        eng.assert_usable({}, {"role": "music_visual"})
    assert ei2.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL
    # (3) weight present -> the engine is usable (the gate is genuinely the file).
    ck = tmp_path / "flux-2-klein-4b-Q4_K_M.gguf"
    ck.write_bytes(b"\x00\x01\x02\x03")
    monkeypatch.setenv("OTR_FLUX2_KLEIN_CKPT", str(ck))
    assert eng.assert_usable({}, {"role": "music_visual"}) == "flux2_klein"


def test_klein_default_checkpoint_resolves_through_folder_paths(
        monkeypatch, tmp_path):
    from nodes._otr_image_engines import flux2_klein as module

    checkpoint = tmp_path / module._DEFAULT_CKPT
    checkpoint.write_bytes(b"installed GGUF")
    fake_folder_paths = types.ModuleType("folder_paths")
    calls = []

    def get_full_path(category, name):
        calls.append((category, name))
        return str(checkpoint) if category == "unet" else None

    fake_folder_paths.get_full_path = get_full_path
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    for name in (
        module.MODEL_ENV, "OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    eng = module.Flux2KleinEngine()
    assert eng.assert_usable({}, {"role": "music_visual"}) == eng.name
    assert eng._klein_params({"prompt": "x"})["unet_name"] == checkpoint.name
    assert calls == [
        ("unet", module._DEFAULT_CKPT),
        ("unet", module._DEFAULT_CKPT),
    ]


def test_klein_default_checkpoint_falls_back_to_configured_models_root(
        monkeypatch, tmp_path):
    from nodes._otr_image_engines import flux2_klein as module

    fake_folder_paths = types.ModuleType("folder_paths")
    registered = []

    def add_model_folder_path(category, path, is_default=False):
        registered.append((category, path, is_default))

    fake_folder_paths.get_full_path = lambda category, name: (
        str(checkpoint) if registered and category == "unet" and
        name == module._DEFAULT_CKPT else None)
    fake_folder_paths.add_model_folder_path = add_model_folder_path
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.delenv(module.MODEL_ENV, raising=False)
    monkeypatch.delenv("COMFYUI_MODELS_ROOT", raising=False)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
    checkpoint = tmp_path / "diffusion_models" / module._DEFAULT_CKPT
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"installed GGUF")

    eng = module.Flux2KleinEngine()
    assert module._resolve_unet_path() == str(checkpoint)
    assert eng.assert_usable({}, {"role": "music_visual"}) == eng.name
    assert registered == [("unet", str(checkpoint.parent), True)]


def test_klein_explicit_override_is_registered_and_wins(monkeypatch, tmp_path):
    from nodes._otr_image_engines import flux2_klein as module

    default = tmp_path / "default" / module._DEFAULT_CKPT
    explicit = tmp_path / "outside" / module._DEFAULT_CKPT
    default.parent.mkdir()
    explicit.parent.mkdir()
    default.write_bytes(b"default")
    explicit.write_bytes(b"explicit")
    search = [str(default.parent)]

    fake_folder_paths = types.ModuleType("folder_paths")

    def add_model_folder_path(category, path, is_default=False):
        assert category == "unet"
        if path in search:
            search.remove(path)
        search.insert(0 if is_default else len(search), path)

    def get_full_path(category, name):
        assert category == "unet"
        for root in search:
            candidate = pathlib.Path(root) / name
            if candidate.is_file():
                return str(candidate)
        return None

    fake_folder_paths.add_model_folder_path = add_model_folder_path
    fake_folder_paths.get_full_path = get_full_path
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.setenv(module.MODEL_ENV, str(explicit))

    eng = module.Flux2KleinEngine()
    assert eng.assert_usable({}, {"role": "music_visual"}) == eng.name
    assert module._resolve_unet_path() == str(explicit)
    assert eng._klein_params({"prompt": "x"})["unet_name"] == explicit.name
    assert search[0] == str(explicit.parent)


@pytest.mark.parametrize("root_env", [
    "OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT",
])
def test_klein_models_root_fallbacks(monkeypatch, tmp_path, root_env):
    from nodes._otr_image_engines import flux2_klein as module

    checkpoint = tmp_path / "diffusion_models" / module._DEFAULT_CKPT
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"installed")
    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.add_model_folder_path = lambda *_args, **_kwargs: None
    fake_folder_paths.get_full_path = lambda category, name: (
        str(checkpoint) if category == "unet" and checkpoint.is_file() else None)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.delenv(module.MODEL_ENV, raising=False)
    for name in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(root_env, str(tmp_path))

    assert module._resolve_unet_path() == str(checkpoint)


def test_klein_env_constants_exported():
    from nodes._otr_image_engines import flux2_klein as m
    assert m.ENABLE_FLAG == "OTR_ENABLE_FLUX2_KLEIN"
    assert m.MODEL_ENV == "OTR_FLUX2_KLEIN_CKPT"
    assert m.CLIP_ENV == "OTR_FLUX2_KLEIN_TE"
    assert m.VAE_ENV == "OTR_FLUX2_KLEIN_VAE"


def test_klein_params_honor_request_and_env(monkeypatch):
    eng = ireg.get_engine("flux2_klein")
    monkeypatch.setenv("OTR_FLUX2_KLEIN_STEPS", "16")
    monkeypatch.setenv("OTR_FLUX2_KLEIN_GUIDANCE", "3")
    monkeypatch.setenv("OTR_FLUX2_KLEIN_CKPT",
                       r"X:\m\flux-2-klein-4b-Q4_K_M.gguf")
    p = eng._klein_params({"prompt": "a 1940s radio studio",
                           "seed": 11, "w": 1248, "h": 832})
    assert p["steps"] == 16 and p["guidance"] == 3.0        # env overrides
    assert p["seed"] == 11 and p["width"] == 1248 and p["height"] == 832
    assert p["prompt"] == "a 1940s radio studio"
    # the loader takes a basename even when the env is an absolute path
    assert p["unet_name"] == "flux-2-klein-4b-Q4_K_M.gguf"
    assert p["clip_name"].endswith(".safetensors")
    assert p["vae_name"].endswith(".safetensors")


def test_klein_dims_snap_to_16(monkeypatch):
    eng = ireg.get_engine("flux2_klein")
    p = eng._klein_params({"prompt": "x", "seed": 1, "w": 1000, "h": 833})
    # flux2 latent is width//16 -> dims must be a multiple of 16
    assert p["width"] % 16 == 0 and p["height"] % 16 == 0
    assert p["width"] == 992 and p["height"] == 832


def test_klein_render_builds_flux2_graph(monkeypatch):
    """render_image drives wrapper_bridge with the native flux2 custom-sampler
    graph -- the flux2 CLIP type, FluxGuidance->BasicGuider, the Flux2Scheduler +
    SamplerCustomAdvanced custom path, the 128-ch EmptyFlux2LatentImage, terminal
    decode -- without a GPU (wrapper_bridge is stubbed)."""
    import numpy as np
    from nodes._otr_video_engines import wrapper_bridge as wb

    eng = ireg.get_engine("flux2_klein")
    monkeypatch.setattr(eng, "_classes", None, raising=False)
    captured = {}

    def fake_run_graph(graph, classes, terminal=None, **kwargs):
        captured["graph"] = graph
        captured["terminal"] = terminal
        # The text encoder must be droppable before the sampler (4060 clean room,
        # 2026-09-02: 0 MB usable for the DiT without it); the MODEL node is kept.
        assert kwargs == {"free_after_use": True, "keep": {"unet"},
                          "evict_after_use": {"clip"}}, kwargs
        return [object()]

    monkeypatch.setattr(wb, "resolve_graph_classes", lambda cands: {k: k for k in cands})
    monkeypatch.setattr(wb, "run_graph", fake_run_graph)
    monkeypatch.setattr(wb, "images_to_uint8",
                        lambda imgs: np.zeros((1, 8, 8, 3), dtype="uint8"))
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda **k: None)
    monkeypatch.setattr(wb, "Wire", lambda node, slot: {"_w": [node, slot]})

    out = eng.render_image({"prompt": "radio console", "seed": 7, "w": 1024, "h": 768})
    assert out.shape == (8, 8, 3)
    g = captured["graph"]
    assert g["unet"]["class"] == "unet"                    # UnetLoaderGGUF
    assert g["unet"]["inputs"]["unet_name"].endswith(".gguf")
    assert g["clip"]["inputs"]["type"] == "flux2"          # the flux2 text path
    assert g["guidance"]["class"] == "guidance"            # FluxGuidance
    assert g["guider"]["inputs"]["conditioning"] == {"_w": ["guidance", 0]}
    assert g["guider"]["inputs"]["model"] == {"_w": ["unet", 0]}
    # custom-sampler wiring: SamplerCustomAdvanced pulls noise/guider/sampler/sigmas
    assert g["sample"]["inputs"]["sigmas"] == {"_w": ["scheduler", 0]}
    assert g["sample"]["inputs"]["sampler"] == {"_w": ["sampler", 0]}
    assert g["sample"]["inputs"]["noise"] == {"_w": ["noise", 0]}
    assert g["sample"]["inputs"]["guider"] == {"_w": ["guider", 0]}
    assert g["decode"]["inputs"]["samples"] == {"_w": ["sample", 0]}
    assert g["latent"]["inputs"]["width"] == 1024 and g["latent"]["inputs"]["height"] == 768
    assert captured["terminal"] == "decode"


def test_klein_cold_import_clean():
    """Importing the adapter pulls NO heavy framework (V-12)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    code = (
        "import sys; import nodes._otr_image_engines.flux2_klein;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
