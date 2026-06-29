"""C3 -- third image engine (Qwen-Image GGUF), generic peer, default-OFF. CPU tests.

Carries the model-agnostic image registry from >=2 engines (C2) to an OPEN set of
>=3, and proves a DIFFERENT integration mode behind the SAME protocol: Z-Image
(C2) runs in a cu128 sidecar; Qwen-Image GGUF rides the in-stack ComfyUI-GGUF
loader, so its fail-closed gate is the GGUF WEIGHTS file (OTR_QWEN_IMAGE_GGUF),
not a sidecar venv. Flux stays gen 1; Qwen-Image is an opt-in peer, greyed until
OTR_ENABLE_QWEN_IMAGE=1 AND its checkpoint exists.

Qwen-Image GRADUATED from a NotImplementedError stub to a real, in-stack engine
(2026-06-19) with an implemented ``render_image`` driving the native GGUF recipe
(UnetLoaderGGUF + CLIPLoader[type=qwen_image] + VAELoader -> ModelSamplingAuraFlow
-> KSampler -> VAEDecode). The graph-build is covered here on CPU (wrapper_bridge
stubbed); the live render + the per-quant VRAM ceiling check is the operator GPU
smoke, gated before promotion to VALIDATED_ENGINES.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines import qwen_image as qwi
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_registry_holds_three_engines():
    names = set(ireg.all_engine_names())
    assert {"flux_gen1", "z_image_turbo", "qwen_image"} <= names
    assert len(names) >= 3  # model-agnostic: an OPEN, growable set of image engines


def test_qwen_image_registry_round_trip():
    # registry round-trip: registered by import, fetchable, name matches.
    assert ireg.is_registered("qwen_image")
    assert "qwen_image" in ireg.all_engine_names()
    assert ireg.get_engine("qwen_image").name == "qwen_image"


def test_qwen_image_selectable_no_flag(monkeypatch):
    # Registry IS the menu: qwen_image is selectable for a role it serves with NO
    # flag gate (the deeper disk check is the adapter's).
    monkeypatch.delenv(qwi.ENABLE_FLAG, raising=False)
    assert ireg.assert_usable("qwen_image", "announcer_visual") == "qwen_image"


def test_flux_remains_gen1_default():
    # Flux is still the in-stack default everywhere; qwen_image is an OPT-IN peer.
    assert ireg.get_engine("qwen_image").default_roles == ()
    assert ireg.get_engine("flux_gen1").default_roles  # non-empty (gen-1 default)


def test_qwen_image_protocol_parity():
    eng = ireg.get_engine("qwen_image")
    for attr in ("name", "roles", "default_roles", "commercial_clean",
                 "requires_flag", "required_inputs"):
        assert hasattr(eng, attr)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")     # reduced prompt->image set (AS-4)
    assert eng.required_inputs == ("text_prompt",)
    assert eng.commercial_clean is True         # Apache-2.0 (per the C2 matrix)
    assert eng.requires_flag is None            # registry IS the menu (no flag gate)


def test_qwen_image_role_filter_shared():
    descs = [
        {"engine_id": n, "roles": tuple(ireg.get_engine(n).roles),
         "required_inputs": tuple(getattr(ireg.get_engine(n), "required_inputs", ()))}
        for n in ireg.all_engine_names()
    ]
    # needs only text_prompt -> fits every role (incl. text-only background_abstract)
    assert "qwen_image" in rc.filter_engines_for_role("background_abstract", descs)
    assert "qwen_image" in rc.filter_engines_for_role("character_video", descs)


def test_qwen_image_adapter_assert_usable_fail_closed(monkeypatch):
    """The adapter's OWN assert_usable fails closed (MISSING_MODEL) until the GGUF
    checkpoint exists -- ABSENT/greyed, never a stub (BUG-046). The gate is the
    weights file (in-stack GGUF), not a sidecar venv."""
    monkeypatch.delenv(qwi.MODEL_ENV, raising=False)
    eng = ireg.get_engine("qwen_image")
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


def test_qwen_image_assert_usable_passes_when_ckpt_present(monkeypatch, tmp_path):
    """With a real checkpoint file on disk the adapter's disk check passes -- the
    fail-closed gate is genuinely the weights file, not a hardcoded raise."""
    ckpt = tmp_path / "qwen-image-Q4_K_M.gguf"
    ckpt.write_bytes(b"GGUF\x00\x00\x00\x00")  # presence only; the real load is the GPU smoke
    monkeypatch.setenv(qwi.MODEL_ENV, str(ckpt))
    eng = ireg.get_engine("qwen_image")
    assert eng.assert_usable({}, {"role": "character_video"}) == "qwen_image"


def test_qwen_image_env_constants_exported():
    assert qwi.ENABLE_FLAG == "OTR_ENABLE_QWEN_IMAGE"
    assert qwi.MODEL_ENV == "OTR_QWEN_IMAGE_GGUF"
    assert qwi.CLIP_ENV == "OTR_QWEN_IMAGE_CLIP"
    assert qwi.VAE_ENV == "OTR_QWEN_IMAGE_VAE"


def test_qwen_image_params_honor_request_and_env(monkeypatch):
    eng = ireg.get_engine("qwen_image")
    monkeypatch.setenv("OTR_QWEN_IMAGE_STEPS", "24")
    monkeypatch.setenv("OTR_QWEN_IMAGE_SHIFT", "3.2")
    monkeypatch.setenv(qwi.MODEL_ENV, r"X:\m\qwen-image-Q4_K_S.gguf")
    p = eng._qwen_params({"prompt": "a 1940s radio studio",
                          "seed": 13, "w": 1216, "h": 832})
    assert p["steps"] == 24 and p["shift"] == 3.2          # env overrides
    assert p["seed"] == 13 and p["width"] == 1216 and p["height"] == 832
    assert p["prompt"] == "a 1940s radio studio"
    # the loader takes a basename even when the env is an absolute path
    assert p["unet_name"] == "qwen-image-Q4_K_S.gguf"
    assert p["clip_name"].endswith(".safetensors")
    assert p["vae_name"].endswith(".safetensors")


def test_qwen_image_render_builds_qwen_graph(monkeypatch):
    """render_image drives wrapper_bridge with the native Qwen-Image GGUF graph --
    UnetLoaderGGUF, the qwen_image CLIP type, the AuraFlow sampling node, an SD3
    latent, terminal decode -- without a GPU (wrapper_bridge is stubbed)."""
    import numpy as np
    from nodes._otr_video_engines import wrapper_bridge as wb

    eng = ireg.get_engine("qwen_image")
    monkeypatch.setattr(eng, "_classes", None, raising=False)
    captured = {}

    def fake_run_graph(graph, classes, terminal=None):
        captured["graph"] = graph
        captured["terminal"] = terminal
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
    assert g["unet"]["class"] == "unet"                    # UnetLoaderGGUF diffusion
    assert g["clip"]["inputs"]["type"] == "qwen_image"     # the qwen text path
    assert g["sampling"]["class"] == "sampling"            # ModelSamplingAuraFlow
    assert g["ksampler"]["inputs"]["model"] == {"_w": ["sampling", 0]}  # sampled model
    assert g["latent"]["inputs"]["width"] == 1024
    assert g["latent"]["inputs"]["height"] == 768
    assert captured["terminal"] == "decode"


def test_qwen_image_node_candidates_gguf_loader():
    """The diffusion loader is the GGUF loader (in-stack, no sidecar)."""
    eng = ireg.get_engine("qwen_image")
    cands = eng._node_candidates()
    assert cands["unet"] == ("UnetLoaderGGUF",)
    assert cands["sampling"] == ("ModelSamplingAuraFlow",)
    assert cands["latent"] == ("EmptySD3LatentImage",)


def test_qwen_image_cold_import_clean():
    code = (
        "import sys; import nodes._otr_image_engines.qwen_image;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
