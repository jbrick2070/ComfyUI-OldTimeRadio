"""CPU tests for the in-process HuMo forward (A-ship render slice).

The HuMo render graph topology + widget values are PROVEN in
scripts/render_humo_batch.build_humo_prompt (verified on the 5080); the exact
model FILENAMES are env-overridable (VERIFY-ON-GPU). These tests drive the forward
end-to-end on the CPU box by INJECTING fake ComfyUI node classes (so only the real
diffusion math -- the operator's GPU concern -- is faked): the proven graph is
executed in dependency order, the decoded IMAGE batch is encoded to a real silent
bt709 / yuv420p mp4 via ffmpeg, the MODEL patchers are retained for V-4 teardown,
and the forward fails closed NAMED when the wrapper nodes or inputs are absent.
ffmpeg-running tests skip cleanly without ffmpeg. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import shutil

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_humo import HuMoEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _humo_fakes(np, n_frames=4):
    """Fake ComfyUI node classes keyed by the graph's node ids. The MODEL nodes
    return an object with a callable detach (a ModelPatcher stand-in); the terminal
    VAEDecode returns a numpy IMAGE batch (B,H,W,3) float."""
    class FakeModel:
        def detach(self, unpatch_all=False):
            return None

    def mk(fn):
        return type("FakeNode", (), {"FUNCTION": "f", "f": fn})

    img = np.zeros((n_frames, 32, 24, 3), dtype="float32")
    return {
        "unet": mk(lambda self, **k: (FakeModel(),)),
        "lora": mk(lambda self, **k: (FakeModel(),)),
        "modelsampling": mk(lambda self, **k: (FakeModel(),)),
        "clip": mk(lambda self, **k: (object(),)),
        "pos": mk(lambda self, **k: (("cond_pos",),)),
        "neg": mk(lambda self, **k: (("cond_neg",),)),
        "vae": mk(lambda self, **k: (object(),)),
        "loadaudio": mk(lambda self, **k: (object(),)),
        "audioenc_loader": mk(lambda self, **k: (object(),)),
        "audioenc": mk(lambda self, **k: (object(),)),
        "loadimage": mk(lambda self, **k: (object(), object())),     # IMAGE, MASK
        "humo": mk(lambda self, **k: (("p",), ("n",), ("latent",))),  # 3 outputs
        "ksampler": mk(lambda self, **k: (("latent",),)),
        "vaedecode": mk(lambda self, **k: (img,)),
    }


# --- proven graph topology ------------------------------------------------- #
def test_humo_graph_topology_matches_proven_wiring():
    eng = HuMoEngine()
    cand = eng._node_candidates()
    assert cand["humo"] == ("WanHuMoImageToVideo",)
    assert cand["ksampler"] == ("KSampler",) and cand["vaedecode"] == ("VAEDecode",)
    plan = eng._build_render_request(
        {"text_prompt": "x", "timing": {"target_frame_count": 97},
         "seed_bundle": {"request_seed": 3},
         "asset_refs": {"init_image": "p"}, "audio_ref": {"path": "a"}})
    g = eng._build_graph("p.png", "a.wav", plan, 97, 480, 832)
    ks = g["ksampler"]["inputs"]
    assert ks["model"] == wb.Wire("modelsampling", 0)
    assert ks["positive"] == wb.Wire("humo", 0)
    assert ks["negative"] == wb.Wire("humo", 1)
    assert ks["latent_image"] == wb.Wire("humo", 2)
    hm = g["humo"]["inputs"]
    assert hm["length"] == 97 and hm["ref_image"] == wb.Wire("loadimage", 0)
    assert hm["audio_encoder_output"] == wb.Wire("audioenc", 0)
    assert hm["positive"] == wb.Wire("pos", 0) and hm["negative"] == wb.Wire("neg", 0)
    assert g["modelsampling"]["inputs"]["shift"] == 8.0
    assert g["lora"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g[eng._TERMINAL]["inputs"]["samples"] == wb.Wire("ksampler", 0)
    assert g[eng._TERMINAL]["inputs"]["vae"] == wb.Wire("vae", 0)


def test_humo_loader_names_and_dims_env(monkeypatch):
    eng = HuMoEngine()
    names = eng._loader_names()
    assert names["clip"].startswith("umt5") and names["vae"] == "wan_2.1_vae.safetensors"
    assert names["unet"].endswith(".safetensors")
    monkeypatch.setenv("OTR_HUMO_UNET_NAME", "custom_humo.safetensors")
    assert eng._loader_names()["unet"] == "custom_humo.safetensors"
    # DIMS ARE NO LONGER AN OPEN OVERRIDE ON A DECLARING TIER (lane 4,
    # 2026-08-11). This tier now declares render_canvas = (480, 832), and the
    # declaration is applied LAST in the request chain -- so an override that
    # DISAGREES would render one size while every receipt reported the other.
    # It is a named refusal; an override that AGREES still passes through.
    monkeypatch.setenv("OTR_HUMO_WIDTH", "512")
    monkeypatch.setenv("OTR_HUMO_HEIGHT", "768")
    from nodes._otr_video_engines.registry import EngineUnusable
    with pytest.raises(EngineUnusable):
        eng._native_dims()
    monkeypatch.setenv("OTR_HUMO_WIDTH", "480")
    monkeypatch.setenv("OTR_HUMO_HEIGHT", "832")
    assert eng._native_dims() == (480, 832)


# --- fail-closed (NAMED) without the wrapper / without inputs --------------- #
def test_humo_load_fails_closed_named_without_wrapper(monkeypatch, tmp_path):
    ck = tmp_path / "humo.safetensors"
    ck.write_bytes(b"x")
    monkeypatch.setenv("OTR_ENABLE_HUMO", "1")
    monkeypatch.setenv("OTR_HUMO_CKPT", str(ck))
    monkeypatch.setattr(wb, "node_class_mappings",
                        lambda mapping=None: {} if mapping is None else mapping)
    eng = HuMoEngine()
    with pytest.raises(wb.WrapperNodeMissing):
        eng.load()


def test_humo_render_clip_requires_audio_and_image():
    eng = HuMoEngine()
    eng._classes = {"vaedecode": object}            # non-empty so resolution skipped
    req = {"shot_id": "s1", "asset_refs": {}, "audio_ref": None,
           "timing": {"target_frame_count": 33}, "seed_bundle": {"request_seed": 1}}
    with pytest.raises(wb.GraphExecutionError):
        eng.render_clip(req, prepared={"patchers": []})


# --- end-to-end through the proven graph with fake nodes + real ffmpeg ------ #
@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_humo_render_clip_drives_graph_to_silent_mp4(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    in_dir = tmp_path / "comfy_in"
    in_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.wav").write_bytes(b"RIFFfakeaudio")
    (src / "p.png").write_bytes(b"\x89PNGfakeportrait")
    monkeypatch.setattr(wb, "comfy_input_dir", lambda: str(in_dir))
    eng = HuMoEngine()
    # The fake must return as many frames as the beat asks for. It used to
    # return 4 against a 33-frame target, which only passed because the Route-A
    # exact-fit branch is gated on `cap is not None` and `HuMoEngine` was the
    # UNCAPPED tier. Once both 14B routes gained the shared cap (2026-08-02),
    # that branch started running here and correctly refused a 4-frame render of
    # a 33-frame beat -- the "short render is TERMINAL and LOUD" rule doing
    # exactly its job. A fake that under-delivers frames was never realistic;
    # a real engine renders what it was asked for.
    eng._classes = _humo_fakes(np, n_frames=33)
    req = {"shot_id": "s1",
           "asset_refs": {"init_image": str(src / "p.png")},
           "audio_ref": {"path": str(src / "a.wav")},
           "text_prompt": "a calm anchor",
           "init_w": 480, "init_h": 832,
           "canvas": {"w": 480, "h": 832, "aspect_policy": "pad"},
           "timing": {"target_frame_count": 33},
           "seed_bundle": {"request_seed": 7}}
    prepared = {"patchers": []}
    raw = eng.render_clip(req, prepared=prepared)
    clip = eng.canonicalize(raw, req, {})
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and p.stat().st_size > 0
        assert clip["frame_count"] == 33 and clip["has_audio"] is False
        assert clip["engine_id"] == "humo" and clip["pixel_format"] == "yuv420p"
        assert (in_dir / "a.wav").exists() and (in_dir / "p.png").exists()  # staged
        # V-4: the three MODEL patchers were retained for teardown.detach()
        assert len(prepared["patchers"]) == 3
        assert all(callable(getattr(m, "detach", None)) for m in prepared["patchers"])
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
