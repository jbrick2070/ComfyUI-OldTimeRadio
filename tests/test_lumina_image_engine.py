"""Lumina-Image 2.0 image engine -- BUILT-engine contract suite (C6).

Lumina graduated from a NotImplementedError stub peer (test_image_engine_matrix_
peers.py) to a real, in-stack engine with an implemented ``render_image`` (the
native split-file ComfyUI recipe: UNETLoader + CLIPLoader[type=lumina2] +
VAELoader + ModelSamplingAuraFlow -> KSampler -> VAEDecode). This file holds its
dedicated CPU contract coverage (registry / opt-in / flag+weight gates / env
constants / cold-import) PLUS the new graph-build + render-drive coverage, all
without a GPU (the live render is the operator smoke).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from nodes._otr_image_engines import registry as ireg


def test_lumina_registered_optin_clean():
    eng = ireg.get_engine("lumina_image")
    assert eng.name == "lumina_image"
    assert eng.default_roles == ()                 # opt-in peer, never "primary"
    assert eng.commercial_clean is True            # Apache-2.0
    assert eng.required_inputs == ("text_prompt",)
    assert eng.requires_flag is None               # registry IS the menu (no flag gate)
    # protocol parity (reduced prompt->image set; no canonicalize)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")


def test_lumina_weight_gate(monkeypatch, tmp_path):
    eng = ireg.get_engine("lumina_image")
    # (1) Registry IS the menu: a registered engine is selectable, no flag gate.
    monkeypatch.delenv("OTR_ENABLE_LUMINA", raising=False)
    assert ireg.assert_usable("lumina_image", "music_visual") == "lumina_image"
    # (2) engine-level: no weight -> MISSING_MODEL (fail-closed, BUG-046).
    monkeypatch.delenv("OTR_LUMINA_CKPT", raising=False)
    with pytest.raises(ireg.EngineUnusable) as ei2:
        eng.assert_usable({}, {"role": "music_visual"})
    assert ei2.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL
    # (3) weight present -> the engine is usable (the gate is genuinely the file).
    ck = tmp_path / "lumina_2_model_bf16.safetensors"
    ck.write_bytes(b"\x00\x01\x02\x03")
    monkeypatch.setenv("OTR_LUMINA_CKPT", str(ck))
    assert eng.assert_usable({}, {"role": "music_visual"}) == "lumina_image"


def test_lumina_env_constants_exported():
    from nodes._otr_image_engines import lumina_image as m
    assert m.ENABLE_FLAG == "OTR_ENABLE_LUMINA"
    assert m.MODEL_ENV == "OTR_LUMINA_CKPT"
    assert m.CLIP_ENV == "OTR_LUMINA_CLIP"
    assert m.VAE_ENV == "OTR_LUMINA_VAE"


def test_lumina_params_honor_request_and_env(monkeypatch):
    eng = ireg.get_engine("lumina_image")
    monkeypatch.setenv("OTR_LUMINA_STEPS", "18")
    monkeypatch.setenv("OTR_LUMINA_SHIFT", "3")
    monkeypatch.setenv("OTR_LUMINA_CKPT",
                       r"X:\m\lumina_2_model_bf16.safetensors")
    p = eng._lumina_params({"prompt": "a 1940s radio studio",
                            "seed": 11, "w": 1216, "h": 832})
    assert p["steps"] == 18 and p["shift"] == 3.0          # env overrides
    assert p["seed"] == 11 and p["width"] == 1216 and p["height"] == 832
    assert p["prompt"] == "a 1940s radio studio"
    # the loader takes a basename even when the env is an absolute path
    assert p["unet_name"] == "lumina_2_model_bf16.safetensors"
    assert p["clip_name"].endswith(".safetensors")
    assert p["vae_name"].endswith(".safetensors")


def test_lumina_render_builds_lumina2_graph(monkeypatch):
    """render_image drives wrapper_bridge with the native Lumina-2 graph -- the
    lumina2 CLIP type, the AuraFlow sampling node, an SD3 latent, terminal decode
    -- without a GPU (wrapper_bridge is stubbed)."""
    import numpy as np
    from nodes._otr_video_engines import wrapper_bridge as wb

    eng = ireg.get_engine("lumina_image")
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
    assert g["clip"]["inputs"]["type"] == "lumina2"        # the lumina text path
    assert g["sampling"]["class"] == "sampling"            # ModelSamplingAuraFlow
    assert g["ksampler"]["inputs"]["model"] == {"_w": ["sampling", 0]}  # sampled model
    assert g["latent"]["inputs"]["width"] == 1024
    assert g["latent"]["inputs"]["height"] == 768
    assert captured["terminal"] == "decode"


# --------------------------------------------------------------------------- #
# The trained input convention (2026-08-17 engine input-convention audit).
# Lumina-2 wants `{system} <Prompt Start> {user}`; comfy/text_encoders/lumina2.py
# applies NO template of its own, so the caller must -- unlike Z-Image, whose
# qwen_image tokenizer wraps the text internally.
# --------------------------------------------------------------------------- #

def test_lumina_composed_text_matches_comfyui_verbatim():
    """The composed string is byte-identical to what CLIPTextEncodeLumina2
    hands the tokenizer. The system lines are copied from ComfyUI, so this
    test is where a silent upstream drift becomes visible."""
    from nodes._otr_image_engines import lumina_image as m

    superior = ("You are an assistant designed to generate superior images "
                "with the superior degree of image-text alignment based on "
                "textual prompts or user prompts.")
    alignment = ("You are an assistant designed to generate high-quality "
                 "images with the highest degree of image-text alignment "
                 "based on textual prompts.")
    assert m.SYSTEM_PROMPTS["superior"] == superior
    assert m.SYSTEM_PROMPTS["alignment"] == alignment
    assert m.PROMPT_START_TAG == "<Prompt Start>"

    # CLIPTextEncodeLumina2.execute: f'{system_prompt} <Prompt Start> {user_prompt}'
    assert (m.compose_encoder_text("a brass microphone", "superior")
            == superior + " <Prompt Start> a brass microphone")
    assert (m.compose_encoder_text("a brass microphone", "alignment")
            == alignment + " <Prompt Start> a brass microphone")


def test_lumina_compose_is_idempotent_and_handles_empty():
    """Composing twice must not double-prefix, and an EMPTY prompt still gets
    the system line -- which is exactly what the ComfyUI node emits for an
    empty user_prompt."""
    from nodes._otr_image_engines import lumina_image as m

    once = m.compose_encoder_text("radio console", "superior")
    assert m.compose_encoder_text(once, "superior") == once
    assert once.count("<Prompt Start>") == 1

    empty = m.compose_encoder_text("", "superior")
    assert empty.startswith(m.SYSTEM_PROMPTS["superior"])
    assert empty.endswith("<Prompt Start> ")
    assert m.compose_encoder_text(None, "superior") == empty


def test_lumina_system_prompt_env_knob(monkeypatch):
    """The knob selects a known key; an unknown value degrades LOUDLY to the
    default rather than killing the render (BUG-046 degrades, never dies)."""
    from nodes._otr_image_engines import lumina_image as m
    eng = ireg.get_engine("lumina_image")

    monkeypatch.delenv(m.SYSTEM_ENV, raising=False)
    assert eng._lumina_params({"prompt": "x"})["system_prompt"] == "superior"

    monkeypatch.setenv(m.SYSTEM_ENV, "alignment")
    assert eng._lumina_params({"prompt": "x"})["system_prompt"] == "alignment"

    monkeypatch.setenv(m.SYSTEM_ENV, "not-a-real-key")
    assert eng._lumina_params({"prompt": "x"})["system_prompt"] == "superior"


def test_lumina_graph_carries_convention_on_both_branches(monkeypatch):
    """BOTH pos and neg carry the convention -- wiring a CLIPTextEncodeLumina2
    into each side of the KSampler is what conformance means, and that node has
    no negative-specific mode. cfg defaults to 4.0, so the uncond branch is
    genuinely sampled."""
    from nodes._otr_image_engines import lumina_image as m
    eng = ireg.get_engine("lumina_image")

    monkeypatch.delenv(m.SYSTEM_ENV, raising=False)
    params = eng._lumina_params({"prompt": "a brass microphone",
                                 "negative_prompt": "blurry, watermark"})
    g = eng._build_lumina_graph(params, lambda node, slot: {"_w": [node, slot]})

    assert g["pos"]["inputs"]["text"] == m.compose_encoder_text(
        "a brass microphone", "superior")
    assert g["neg"]["inputs"]["text"] == m.compose_encoder_text(
        "blurry, watermark", "superior")
    for branch in ("pos", "neg"):
        assert g[branch]["inputs"]["text"].count("<Prompt Start>") == 1


def test_lumina_params_keep_the_raw_request_text(monkeypatch):
    """The composition happens in the GRAPH, never in the params. The
    dispatcher derives prompt_hash / cache key / seed from the REQUEST prompt
    upstream, so the composed string must never leak back into it."""
    from nodes._otr_image_engines import lumina_image as m
    eng = ireg.get_engine("lumina_image")

    monkeypatch.delenv("OTR_LUMINA_NEGATIVE", raising=False)
    p = eng._lumina_params({"prompt": "a brass microphone",
                            "negative_prompt": "blurry"})
    assert p["prompt"] == "a brass microphone"
    assert p["negative"] == "blurry"
    assert m.PROMPT_START_TAG not in p["prompt"]
    assert m.PROMPT_START_TAG not in p["negative"]


def test_lumina_cold_import_clean():
    """Importing the adapter pulls NO heavy framework (V-12)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    code = (
        "import sys; import nodes._otr_image_engines.lumina_image;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
