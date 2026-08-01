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
        # GGUF recipe: UnetLoaderGGUF -> LoraLoaderModelOnly (the kept patcher) ;
        # LTXAVTextEncoderLoader -> clip ; VAELoader -> video VAE.
        "unet": _mk(lambda self, **k: (_FakeModel(),)),
        "lora": _mk(lambda self, **k: (_FakeModel(),)),
        "videovae": _mk(lambda self, **k: (object(),)),
        "te": _mk(lambda self, **k: (object(),)),        # LTXAVTextEncoderLoader -> clip
        "pos": _mk(lambda self, **k: (("c",),)),
        "neg": _mk(lambda self, **k: (("c",),)),
        "latent": _mk(lambda self, **k: (("latent",),)),
        "cond": _mk(lambda self, **k: (("p",), ("n",))),
        "ksampler": _mk(lambda self, **k: (("latent",),)),
        # distilled chain fakes (the default sampling mode)
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
    """OTR_LTX_SAMPLER=ksampler keeps the GGUF loaders but swaps the distilled
    chain for the 30-step KSampler (manual A/B path)."""
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    monkeypatch.delenv("OTR_LTX_CFG", raising=False)
    eng = LtxVideoEngine()
    cand = eng._node_candidates()
    assert cand["latent"] == ("EmptyLTXVLatentVideo",)
    assert cand["cond"] == ("LTXVConditioning",)
    plan = eng._build_render_request(
        {"text_prompt": "x", "negative_prompt": "y",
         "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 2}})
    g = eng._build_graph(plan, 49, 832, 480)
    # GGUF recipe loaders: Gemma-3 text encoder feeds pos + neg (the mini).
    assert cand["te"] == ("LTXAVTextEncoderLoader",)
    assert cand["unet"] == ("UnetLoaderGGUF",)
    assert g["latent"]["inputs"]["length"] == 49
    assert g["pos"]["inputs"]["clip"] == wb.Wire("te", 0)
    assert g["neg"]["inputs"]["clip"] == wb.Wire("te", 0)
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("lora", 0)    # the LoRA-wrapped GGUF unet
    assert ks["positive"] == wb.Wire("cond", 0)
    assert ks["negative"] == wb.Wire("cond", 1)
    assert ks["latent_image"] == wb.Wire("latent", 0)
    assert ks["cfg"] == 3.0                     # the ksampler-mode env default
    assert "sampleradv" not in g and "guider" not in g
    # VAEDecodeTiled (the mini's 512/64/4096/8) on the video VAE.
    term = g[eng._TERMINAL]["inputs"]
    assert term["vae"] == wb.Wire("videovae", 0)
    assert (term["tile_size"], term["overlap"],
            term["temporal_size"], term["temporal_overlap"]) == (512, 64, 4096, 8)


def test_ltx_graph_topology_distilled_default(monkeypatch):
    """Distilled mode (the DEFAULT GGUF mini recipe): 8-step LTX_DISTILLED_SIGMAS
    + euler_cfg_pp + CFGGuider cfg=1.0 + distilled LoRA @0.70 on the GGUF unet.
    Env knobs cleared to prove the mini values are HARDCODED (the #1 invariant)."""
    from nodes._otr_video_engines.eng_ltx_video import LTX_DISTILLED_SIGMAS
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)   # distilled is the DEFAULT
    monkeypatch.delenv("OTR_LTX_SAMPLER_NAME", raising=False)
    monkeypatch.delenv("OTR_LTX_CFG", raising=False)
    eng = LtxVideoEngine()
    assert eng._sampler_mode() == "distilled"   # the default
    cand = eng._node_candidates_sampling()
    assert "ksampler" not in cand
    assert cand["samplersel"] == ("KSamplerSelect",)
    assert cand["noise"] == ("RandomNoise",)
    assert cand["guider"] == ("CFGGuider",)
    assert cand["sampleradv"] == ("SamplerCustomAdvanced",)
    assert cand["lora"] == ("LoraLoaderModelOnly",)   # ALWAYS present (GGUF recipe)
    assert "sigmas" not in cand                # injected post-resolve
    plan = eng._build_render_request(
        {"text_prompt": "x", "negative_prompt": "y",
         "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 2}})
    g = eng._build_graph(plan, 49, 832, 480)
    assert "ksampler" not in g
    # the distilled chain uses euler_cfg_pp (the mini's KSamplerSelect)
    assert g["samplersel"]["inputs"]["sampler_name"] == "euler_cfg_pp"
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    assert len(g["sigmas"]["inputs"]["values"]) == 9      # 8 sampling steps
    assert g["noise"]["inputs"]["noise_seed"] == plan["seed"]
    gd = g["guider"]["inputs"]
    assert gd["cfg"] == 1.0                    # mini CFGGuider cfg (hardcoded)
    assert gd["model"] == wb.Wire("lora", 0)   # the LoRA-wrapped GGUF unet
    # LoRA ALWAYS wired (unet -> lora @0.70 -> guider.model)
    assert g["lora"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g["lora"]["inputs"]["strength_model"] == 0.7
    sa = g["sampleradv"]["inputs"]
    assert sa["sampler"] == wb.Wire("samplersel", 0)
    assert sa["sigmas"] == wb.Wire("sigmas", 0)
    assert sa["latent_image"] == wb.Wire("latent", 0)
    assert g[eng._TERMINAL]["inputs"]["samples"] == wb.Wire("sampleradv", 0)


def test_ltx_distilled_lora_always_wired(monkeypatch):
    """The GGUF recipe ALWAYS wires the distilled LoRA between the GGUF unet and
    the sampler/guider MODEL input, in BOTH sampler modes (no 2B/22B gating --
    the unet IS the 22B GGUF)."""
    plan = {"text_prompt": "x", "negative_prompt": "y", "fps": 25,
            "target_frame_count": 49, "seed": 2}
    eng = LtxVideoEngine()
    # distilled (the default): unet -> lora @0.70 -> guider.model
    monkeypatch.setenv("OTR_LTX_SAMPLER", "distilled")
    g = eng._build_graph(plan, 49, 832, 480)
    assert g["lora"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g["lora"]["inputs"]["strength_model"] == 0.7
    assert g["lora"]["inputs"]["lora_name"].endswith(
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    assert g["guider"]["inputs"]["model"] == wb.Wire("lora", 0)
    # ksampler (manual A/B): unet -> lora -> ksampler.model
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    g2 = eng._build_graph(plan, 49, 832, 480)
    assert g2["lora"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g2["ksampler"]["inputs"]["model"] == wb.Wire("lora", 0)


def test_ltx_sampler_mode_invalid_falls_back_loud(monkeypatch):
    monkeypatch.setenv("OTR_LTX_SAMPLER", "warp_drive")
    # GGUF splice: invalid falls back LOUD to the distilled default (the mini)
    assert LtxVideoEngine._sampler_mode() == "distilled"


def test_ltx_sampler_default_is_distilled(monkeypatch):
    # GGUF splice (2026-06-15): the frozen mini recipe (distilled -- euler_cfg_pp
    # + 8-step + cfg 1.0 + LoRA 0.70 on the 22B GGUF unet) is the DEFAULT; the
    # 30-step ksampler is the manual A/B opt-in (OTR_LTX_SAMPLER=ksampler).
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)
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
def test_ltx_load_fails_closed_named(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_LTX_VIDEO", "1")
    # No ComfyUI node classes registered -> load() resolves the GGUF graph
    # classes and fails closed NAMED (WrapperNodeMissing), never a silent success.
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
def test_ltx_render_clip_to_silent_mp4(monkeypatch):
    # Base (non-loop) render mechanics: pin the boomerang OFF so frame_count is
    # the raw decode (the loop path has its own test below -- BUG-LOCAL-117d).
    # S5: pin single_pass -- this test exercises the FROZEN GGUF-mini
    # mechanics; the dev-unet auto default now routes to hq_two_stage (which
    # requires an init still and has its own tests).
    # A4 (2026-07-27): this request carries no still, and it used to reach the
    # text path through the silent i2v degrade that has now been removed. The
    # test exercises sampler/encode mechanics, not the i2v decision, so it
    # declares the text path with the shipped opt-out instead of arriving
    # there by accident.
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", "off")
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
        assert len(prepared["patchers"]) == 1            # LoRA-wrapped unet retained
    finally:
        p.unlink(missing_ok=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_ltx_render_clip_does_not_boomerang_by_default(monkeypatch):
    # DEFAULT FLIPPED 2026-08-01 (operator): "no boomerangs, remove all
    # boomerangs in place of native clips". BUG-LOCAL-117d's loop_via_reverse was
    # the ltx_video default and render_clip mirrored the decoded frames
    # end-to-end (4 -> 2*4-1 = 7). Now 4 decoded frames stay 4: no reused frames,
    # 1/1 and done. This is the END-TO-END proof through the encoder, which is
    # why it is worth more than the unit pin on the flag.
    # S5: pin single_pass (frozen mechanics; hq_two_stage has its own tests).
    # A4 (2026-07-27): same as above -- stillless request, text path declared
    # explicitly now that the silent i2v degrade is gone.
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    monkeypatch.delenv("OTR_LTX_LOOP_VIA_REVERSE", raising=False)
    np = pytest.importorskip("numpy")
    eng = LtxVideoEngine()
    eng._classes = _ltx_fakes(np, n=4)
    req = {"shot_id": "s1", "text_prompt": "a neon diner",
           "canvas": {"w": 768, "h": 512, "fps": 25},
           "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 7}}
    prepared = {"patchers": []}
    raw = eng.render_clip(req, prepared)
    assert raw["ltx_loop_via_reverse"] is False
    clip = eng.canonicalize(raw, req, {})
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 4    # NOT 7 -- no mirror
        assert clip["engine_id"] == "ltx_video"
    finally:
        p.unlink(missing_ok=True)


def test_ltx_render_clip_still_boomerangs_when_explicitly_opted_in(monkeypatch):
    """The device is retired as a DEFAULT, not deleted. Opting back in must
    still work end-to-end -- the flag flip broke exactly this once already."""
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", "on")
    np = pytest.importorskip("numpy")
    eng = LtxVideoEngine()
    eng._classes = _ltx_fakes(np, n=4)
    req = {"shot_id": "s1", "text_prompt": "a neon diner",
           "canvas": {"w": 768, "h": 512, "fps": 25},
           "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 7}}
    raw = eng.render_clip(req, {"patchers": []})
    assert raw["ltx_loop_via_reverse"] is True
    clip = eng.canonicalize(raw, req, {})
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 7    # 2*4 - 1, mirrored
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
