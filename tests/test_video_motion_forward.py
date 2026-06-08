"""CPU tests for the in-process LTX + Wan forwards (A-ship render slices).

These graphs are ASSUMED native topologies (VERIFY-ON-GPU): the box runs the LTX
2.3 / KJ Wan 2.2 wrappers, so the operator confirms the node candidates + widgets
against the installed INPUT_TYPES. The SHARED forward mechanics (node resolution,
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
        "pos": _mk(lambda self, **k: (("c",),)),
        "neg": _mk(lambda self, **k: (("c",),)),
        "latent": _mk(lambda self, **k: (("latent",),)),
        "cond": _mk(lambda self, **k: (("p",), ("n",))),
        "ksampler": _mk(lambda self, **k: (("latent",),)),
        "vaedecode": _mk(lambda self, **k: (img,)),
    }


def _wan_fakes(np, n=4):
    img = np.zeros((n, 24, 32, 3), dtype="float32")
    return {
        "unet": _mk(lambda self, **k: (_FakeModel(),)),
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
def test_ltx_graph_topology():
    eng = LtxVideoEngine()
    cand = eng._node_candidates()
    assert cand["latent"] == ("EmptyLTXVLatentVideo",)
    assert cand["cond"] == ("LTXVConditioning",)
    plan = eng._build_render_request(
        {"text_prompt": "x", "negative_prompt": "y",
         "timing": {"target_frame_count": 49}, "seed_bundle": {"request_seed": 2}})
    g = eng._build_graph(plan, 49, 768, 512)
    assert g["latent"]["inputs"]["length"] == 49
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("checkpoint", 0)
    assert ks["positive"] == wb.Wire("cond", 0)
    assert ks["negative"] == wb.Wire("cond", 1)
    assert ks["latent_image"] == wb.Wire("latent", 0)
    assert g[eng._TERMINAL]["inputs"]["vae"] == wb.Wire("checkpoint", 2)


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
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("unet", 0)
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
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "p.png").write_bytes(b"\x89PNGfake")
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
        assert (in_dir / "p.png").exists()               # staged init image
        assert len(prepared["patchers"]) == 1            # unet MODEL retained
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
