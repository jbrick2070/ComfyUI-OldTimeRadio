"""CPU tests for the in-process LTX + Wan forwards (A-ship render slices).

These graphs run on CORE ComfyUI nodes (LTX 2.x / core Wan 2.2 -- NOT the KJ Wan
wrapper, which conflicts with the numpy2/transformers5 box and drags in Sage);
the topologies are verified against the installed /object_info on the GPU box
(2026-06-12). The SHARED forward mechanics (node resolution,
the declarative executor, the silent bt709 encode, MODEL-patcher retention, the
VRAM guard) are proven here by injecting fake ComfyUI node classes -- the proven
path runs through to a real ffmpeg-encoded silent mp4. ffmpeg-running tests skip
cleanly without ffmpeg. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import shutil

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_video import LtxVideoEngine
from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _mk(fn):
    return type("FakeNode", (), {"FUNCTION": "f", "f": fn})


class _FakeModel:
    def detach(self, unpatch_all=False):
        return None


def _ltx_fakes(np, n=4):
    img = np.zeros((n, 24, 32, 3), dtype="float32")
    return {
        "checkpoint": _mk(lambda self, **k: (_FakeModel(), object(), object())),
        "encoder": _mk(lambda self, **k: (object(),)),   # CLIPLoader -> clip
        "pos": _mk(lambda self, **k: (("c",),)),
        "neg": _mk(lambda self, **k: (("c",),)),
        "latent": _mk(lambda self, **k: (("latent",),)),
        "cond": _mk(lambda self, **k: (("p",), ("n",))),
        "ksampler": _mk(lambda self, **k: (("latent",),)),
        # LK-1b distilled chain fakes (the default sampling mode)
        "samplersel": _mk(lambda self, **k: (("sampler",),)),
        "sigmas": _mk(lambda self, **k: (("sigmas",),)),
        "noise": _mk(lambda self, **k: (("noise",),)),
        "guider": _mk(lambda self, **k: (("guider",),)),
        "sampleradv": _mk(lambda self, **k: (("latent",), ("denoised",))),
        "vaedecode": _mk(lambda self, **k: (img,)),
    }


def _wan_fakes(np, n=4):
    img = np.zeros((n, 24, 32, 3), dtype="float32")
    return {
        "unet": _mk(lambda self, **k: (_FakeModel(),)),
        "modelsampling": _mk(lambda self, **k: (_FakeModel(),)),
        "clip": _mk(lambda self, **k: (object(),)),
        "pos": _mk(lambda self, **k: (("c",),)),
        "neg": _mk(lambda self, **k: (("c",),)),
        "vae": _mk(lambda self, **k: (object(),)),
        "loadimage": _mk(lambda self, **k: (object(), object())),
        "wan": _mk(lambda self, **k: (("p",), ("n",), ("latent",))),
        "ksampler": _mk(lambda self, **k: (("latent",),)),
        "vaedecode": _mk(lambda self, **k: (img,)),
    }


# --- topology -------------------------------------------------------------- #
def test_ltx_graph_topology_ksampler_rollback(monkeypatch):
    """OTR_LTX_SAMPLER=ksampler restores the round-5 30-step graph."""
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    monkeypatch.delenv("OTR_LTX_CFG", raising=False)
    eng = LtxVideoEngine()
    cand = eng._node_candidates()
    assert cand["latent"] == ("EmptyLTXVLatentVideo",)
    assert cand["cond"] == ("LTXVConditioning",)
    plan = eng._build_render_request(
        {"text_prompt": "x", "negative_prompt": "y",
         "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 2}})
    g = eng._build_graph(plan, 49, 768, 512)
    assert cand["encoder"] == ("CLIPLoader",)
    assert g["latent"]["inputs"]["length"] == 49
    # T5-XXL CLIPLoader feeds pos + neg (GPU-verified probe_f3 2026-06-09)
    assert g["pos"]["inputs"]["clip"] == wb.Wire("encoder", 0)
    assert g["neg"]["inputs"]["clip"] == wb.Wire("encoder", 0)
    assert g["encoder"]["inputs"]["type"] == "ltxv"
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("checkpoint", 0)
    assert ks["positive"] == wb.Wire("cond", 0)
    assert ks["negative"] == wb.Wire("cond", 1)
    assert ks["latent_image"] == wb.Wire("latent", 0)
    assert ks["cfg"] == 3.0                     # the ksampler-mode default
    assert "sampleradv" not in g and "guider" not in g
    assert g[eng._TERMINAL]["inputs"]["vae"] == wb.Wire("checkpoint", 2)


def test_ltx_graph_topology_distilled_default(monkeypatch):
    """Distilled mode: 8-step LTX_DISTILLED_SIGMAS + euler_cfg_pp + CFGGuider
    cfg=1.0 -- the DEFAULT path (BUG-LOCAL-412 restored 8-step distilled as the
    default; ksampler is the opt-in 30-step). Env left UNSET to prove the
    default; CFG/ckpt envs cleared so the distilled defaults apply."""
    from nodes._otr_video_engines.eng_ltx_video import LTX_DISTILLED_SIGMAS
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)   # prove the DEFAULT
    monkeypatch.delenv("OTR_LTX_SAMPLER_NAME", raising=False)
    monkeypatch.delenv("OTR_LTX_CFG", raising=False)
    monkeypatch.delenv("OTR_LTX_VIDEO_CKPT_NAME", raising=False)
    eng = LtxVideoEngine()
    assert eng._sampler_mode() == "distilled"   # the new default
    cand = eng._node_candidates_sampling()
    assert "ksampler" not in cand
    assert cand["samplersel"] == ("KSamplerSelect",)
    assert cand["noise"] == ("RandomNoise",)
    assert cand["guider"] == ("CFGGuider",)
    assert cand["sampleradv"] == ("SamplerCustomAdvanced",)
    assert "sigmas" not in cand                # injected post-resolve
    plan = eng._build_render_request(
        {"text_prompt": "x", "negative_prompt": "y",
         "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 2}})
    g = eng._build_graph(plan, 49, 768, 512)
    assert "ksampler" not in g
    # BUG-LOCAL-412: the restored dynamic-motion sampler is the default
    assert g["samplersel"]["inputs"]["sampler_name"] == "euler_cfg_pp"
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    assert len(g["sigmas"]["inputs"]["values"]) == 9      # 8 sampling steps
    assert g["noise"]["inputs"]["noise_seed"] == plan["seed"]
    gd = g["guider"]["inputs"]
    assert gd["cfg"] == 1.0                    # legacy distilled LTX_CFG
    assert gd["model"] == wb.Wire("checkpoint", 0)   # 2B default: NO LoRA
    assert "lora" not in g
    sa = g["sampleradv"]["inputs"]
    assert sa["sampler"] == wb.Wire("samplersel", 0)
    assert sa["sigmas"] == wb.Wire("sigmas", 0)
    assert sa["latent_image"] == wb.Wire("latent", 0)
    assert g[eng._TERMINAL]["inputs"]["samples"] == wb.Wire("sampleradv", 0)


def test_ltx_distilled_lora_gating(monkeypatch, tmp_path):
    """The 22B distilled LoRA wires ONLY on a 22B ckpt with the file on
    disk; the 2B default and a missing file render WITHOUT it (LOUD).
    Requires distilled mode explicitly -- LoRA gates on the guider node."""
    monkeypatch.setenv("OTR_LTX_SAMPLER", "distilled")   # LoRA lives in distilled path
    eng = LtxVideoEngine()
    plan = {"text_prompt": "x", "negative_prompt": "y", "fps": 25,
            "target_frame_count": 49, "seed": 2}
    # 2B default name -> never wired, regardless of the file.
    monkeypatch.setenv("OTR_LTX_VIDEO_CKPT_NAME", "ltx-video-2b-v0.9.safetensors")
    assert eng._use_distilled_lora() is False
    # 22B ckpt + file on disk -> wired between checkpoint and the guider.
    monkeypatch.setenv("OTR_LTX_VIDEO_CKPT_NAME", "ltx-2.3-22b-dev.safetensors")
    lora_rel = "test_distilled_lora.safetensors"
    (tmp_path / lora_rel).write_bytes(b"x")
    monkeypatch.setenv("OTR_LTX_DISTILLED_LORA", lora_rel)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    monkeypatch.setattr(
        "nodes._otr_video_engines.eng_ltx_video._COMFY_ROOT", "Z:/nope")
    # HF_HOME-derived loras dir = tmp_path/loras; place the file there.
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / lora_rel).write_bytes(b"x")
    assert eng._use_distilled_lora() is True
    g = eng._build_graph(plan, 49, 768, 512)
    assert g["lora"]["inputs"]["lora_name"] == lora_rel
    assert g["lora"]["inputs"]["strength_model"] == 0.7
    assert g["lora"]["inputs"]["model"] == wb.Wire("checkpoint", 0)
    assert g["guider"]["inputs"]["model"] == wb.Wire("lora", 0)
    # 22B ckpt + file MISSING -> LOUD render without it.
    monkeypatch.setenv("OTR_LTX_DISTILLED_LORA", "gone.safetensors")
    assert eng._use_distilled_lora() is False
    g2 = eng._build_graph(plan, 49, 768, 512)
    assert "lora" not in g2


def test_ltx_sampler_mode_invalid_falls_back_loud(monkeypatch):
    monkeypatch.setenv("OTR_LTX_SAMPLER", "warp_drive")
    # BUG-LOCAL-412: invalid now falls back to the distilled default (was ksampler)
    assert LtxVideoEngine._sampler_mode() == "distilled"


def test_wan_graph_topology():
    eng = WanI2VEngine()
    cand = eng._node_candidates()
    assert cand["wan"] == ("WanImageToVideo",)
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"}, "init_w": 480, "init_h": 832,
         "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
         "timing": {"target_frame_count": 33}, "seed_bundle": {"request_seed": 5}})
    g = eng._build_graph({"text_prompt": "move"}, "p.png", plan, 33, 832, 480)
    wan = g["wan"]["inputs"]
    assert wan["start_image"] == wb.Wire("loadimage", 0)
    assert wan["length"] == 33 and wan["positive"] == wb.Wire("pos", 0)
    # ModelSamplingSD3 sits between the UNET loader and the sampler (sigma shift).
    assert g["modelsampling"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert cand["modelsampling"] == ("ModelSamplingSD3",)
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("modelsampling", 0)
    assert ks["latent_image"] == wb.Wire("wan", 2)


# --- fail-closed (NAMED) --------------------------------------------------- #
def test_ltx_load_fails_closed_named(monkeypatch, tmp_path):
    ck = tmp_path / "ltx.safetensors"
    ck.write_bytes(b"x")
    monkeypatch.setenv("OTR_ENABLE_LTX_VIDEO", "1")
    monkeypatch.setenv("OTR_LTX_VIDEO_CKPT", str(ck))
    monkeypatch.setattr(wb, "node_class_mappings",
                        lambda mapping=None: {} if mapping is None else mapping)
    with pytest.raises(wb.WrapperNodeMissing):
        LtxVideoEngine().load()


def test_wan_render_requires_init_image():
    eng = WanI2VEngine()
    eng._classes = {"vaedecode": object}            # non-empty so resolution skipped
    req = {"asset_refs": {}, "timing": {"target_frame_count": 33},
           "seed_bundle": {"request_seed": 1}}
    with pytest.raises(wb.GraphExecutionError):
        eng.render_clip(req, prepared={"patchers": []})


# --- end-to-end with fakes + real ffmpeg ----------------------------------- #
@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_ltx_render_clip_to_silent_mp4():
    np = pytest.importorskip("numpy")
    eng = LtxVideoEngine()
    eng._classes = _ltx_fakes(np, n=4)
    req = {"shot_id": "s1", "text_prompt": "a neon diner",
           "canvas": {"w": 768, "h": 512, "fps": 25},
           "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 7}}
    prepared = {"patchers": []}
    clip = eng.canonicalize(eng.render_clip(req, prepared), req, {})
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 4
        assert clip["has_audio"] is False and clip["engine_id"] == "ltx_video"
        assert len(prepared["patchers"]) == 1            # checkpoint MODEL retained
    finally:
        p.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_wan_render_clip_to_silent_mp4(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    # A REAL png: S10 requires Pillow to materialize the init into the canvas;
    # an unreadable source now fails LOUD instead of silently staging raw.
    Image.new("RGB", (480, 832), (120, 60, 30)).save(src / "p.png")
    monkeypatch.setattr(wb, "comfy_input_dir", lambda: str(in_dir))
    eng = WanI2VEngine()
    eng._classes = _wan_fakes(np, n=4)
    req = {"shot_id": "s1", "asset_refs": {"init_image": str(src / "p.png")},
           "text_prompt": "subtle motion", "init_w": 480, "init_h": 832,
           "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
           "timing": {"target_frame_count": 33}, "seed_bundle": {"request_seed": 3}}
    prepared = {"patchers": []}
    clip = eng.canonicalize(eng.render_clip(req, prepared), req, {})
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 4
        assert clip["engine_id"] == "wan_i2v" and clip["has_audio"] is False
        # S7: staged under a per-shot/seed name, not the fixed otr_wan_init_WxH.
        assert (in_dir / "otr_wan_init_s1_s3_832x480.png").exists()
        assert len(prepared["patchers"]) == 1            # unet MODEL retained
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
