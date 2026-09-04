"""viz_camera -- OTR-native Golden Flicker camera visualizer.

CPU-only contract tests (no GPU). The render-path tests DO shell out to ffmpeg
now: since 2026-07-28 the engine proves the clip it wrote, so the encoder is
wrapped to capture frames rather than replaced by a stub -- a proof cannot be
run against a file the test invented. They skip where ffmpeg is absent.
Covers registration + CAPABILITIES; role eligibility;
accepts_still=False; ambient master audio; content-oracle motion exemption; render
contract for idle and reactive paths; frame-count exactness; painter determinism.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

import nodes._otr_video_engines  # noqa: F401  (self-registers every engine)
from nodes._otr_shared import role_compat as rc
from nodes._otr_shared import scope_draw as sd
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes.otr_image_gen_dispatcher import engine_consumes_still

NAME = "viz_camera"


def test_registered_with_capability_row():
    assert NAME in vreg.all_engine_names()
    eng = vreg.get_engine(NAME)
    assert eng.name == NAME and eng.family == "abstract"
    assert eng.accepts_still is False
    assert tuple(eng.required_inputs) == ()
    assert eng.fallback_engine is None and eng.engine_version == "1"
    assert NAME in vreg.CAPABILITIES
    assert "cpu" in vreg.CAPABILITIES[NAME]["device_backends"]
    assert vreg.CAPABILITIES[NAME]["required_toolchain"] is None


def test_required_inputs_empty_fits_all_5_roles():
    desc = vreg.descriptor_for_engine(NAME)
    for role in rc.ROLES:
        assert rc.engine_fits_role(desc, role), role
    for role in rc.ROLES:
        assert vreg.assert_usable(NAME, role) == NAME


def test_accepts_still_false_mints_no_still():
    assert engine_consumes_still(vreg.get_engine(NAME)) is False


def test_ambient_master_audio_gate_includes_viz_camera():
    assert rd._uses_ambient_master_audio(NAME, "abstract") is True
    assert rd._uses_ambient_master_audio(NAME, "abstract", is_char_face=True) is False


def test_engine_family_map():
    assert rd.engine_family(NAME) == "abstract"


def test_the_frame_contract_DECLARES_continuity_none_and_says_why():
    """G3.3 / lesson L3 (lane 12, 2026-08-11).

    The class comment had claimed "CONTINUITY none" since this engine was
    written while `continuity=` was never passed, so the value was a dataclass
    DEFAULT -- the right answer nobody had decided, which is the same shape a
    wrong one would have had.

    NONE is true here for a reason this lane can state: `render_clip` paints
    every frame from the beat's own audio analysis and a per-beat rng key, and
    reads no predecessor frame, so no terminal state exists for a successor to
    inherit.

    Declared per lane because each visualizer owns its contract -- lane 10's
    shared-base fix reached the still shelf, not this family.
    """
    import inspect

    from nodes._otr_video_engines import frame_contract as fcm
    from nodes._otr_video_engines.eng_viz_camera import VizCameraEngine

    eng = vreg.get_engine(NAME)
    assert fcm.frame_contract_for(eng).continuity == fcm.CONTINUITY_NONE
    # DECLARED, not defaulted -- and read from the AST, not the source TEXT.
    # A substring check for "continuity=" is satisfied by the comment above the
    # declaration explaining it (the lane 12 QA finding), so it would pass with
    # the real keyword deleted. The resolved VALUE cannot catch it either: the
    # dataclass default is the same constant.
    assert fcm.declares_continuity_kwarg(eng)
    render_src = inspect.getsource(VizCameraEngine.render_clip)
    for consumes_predecessor in ("prev_frame", "last_frame", "init_image",
                                 "continuity_frame"):
        assert consumes_predecessor not in render_src


def test_the_lane_DECLARES_NO_canvas_and_honours_ANY_request_size():
    """G2 / lesson L19 (lane 12, 2026-08-11).

    This lane must NOT declare a `render_canvas`. It has no native canvas:
    `render_clip` builds the painter, the scanline table, the vignette and the
    encoder from the request's own w/h. The 1472x832 an episode hands it is the
    default of `OTR_VIDEO_LANDSCAPE_CANVAS` -- an operator lever -- and since
    `declared_render_canvas` is applied LAST, declaring would silently make
    this the one visualizer that ignores that lever.

    The premise was re-checked on THIS engine rather than inherited from
    viz_green, which is L19's own runnable check.
    """
    from nodes._otr_video_engines.eng_viz_camera import VizCameraEngine

    eng = vreg.get_engine(NAME)
    assert getattr(eng, "render_canvas", None) is None
    assert rd.declared_render_canvas(NAME) is None
    for size in ((1472, 832), (832, 480), (640, 384), (1024, 576)):
        got = VizCameraEngine()._canvas_dims(
            {"canvas": {"w": size[0], "h": size[1], "fps": 25}})
        assert got == size


def test_the_landscape_lever_still_reaches_this_lane(monkeypatch):
    """The half that would otherwise rot silently: if a future change declares
    a canvas here, this is the test that says what it cost."""
    ledger = {
        "episode_id": "ep_lane12",
        "images": {"images": [{"beat_id": "b001", "kind": "scene_beat",
                               "path": "C:/tmp/scene_b001.png"}]},
        "video": {"fps": 25, "canonical_canvas": None,
                  "shots": [{"shot_id": "shot_b001", "role": "music_visual",
                             "group_id": "grp_music_visual",
                             "engine_id": NAME, "family": "",
                             "target_frame_count": 25}]},
    }
    shot = ledger["video"]["shots"][0]

    req = rd.build_request_from_shot(shot, ledger)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (1472, 832)

    monkeypatch.setenv("OTR_VIDEO_LANDSCAPE_CANVAS", "1024x576")
    req = rd.build_request_from_shot(shot, ledger)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (1024, 576), (
        "the operator's landscape lever no longer reaches viz_camera -- if a "
        "render_canvas declaration was added, that is what it cost")


def _req(frames=3, w=96, h=64, seed=7, audio=None):
    r = {"shot_id": "s1", "canvas": {"w": w, "h": h, "fps": 25},
         "timing": {"target_frame_count": frames}, "seed_bundle": {"request_seed": seed}}
    if audio:
        r["audio_ref"] = {"path": audio}
    return r


def _render_capturing_frames(monkeypatch, request):
    captured = {}

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("the engine now PROVES its clip; that needs a real clip")
    real_encode = sd.encode_silent_mp4

    def _fake_encode(frames_iter, total, out_path, w, h, fps, ffmpeg):
        frames = [np.asarray(f) for f in frames_iter]
        captured["frames"] = frames
        captured["total"] = total
        # PASSES THROUGH to the real encoder instead of writing one zero byte.
        # M7 (2026-07-28): the engine now proves the clip it just wrote -- the
        # silent-clip colour/stream contract, plus a frame count read back off
        # the FILE -- and a one-byte stub is not something ffprobe can be
        # asked about. Stubbing it would also have made the frame_count
        # assertions below tautological: the test would be verifying a number
        # against a file the test itself invented. Capturing on the way
        # through keeps every frame-level assertion intact.
        return real_encode(iter(frames), total, out_path, w, h, fps, ffmpeg)

    monkeypatch.setattr(sd, "encode_silent_mp4", _fake_encode)
    raw = vreg.get_engine(NAME).render_clip(request)
    return raw, captured


def test_render_contract_audio_absent_idle(monkeypatch):
    raw, cap = _render_capturing_frames(monkeypatch, _req(frames=3))
    assert raw["frame_count"] == 3 and raw["mode"] == "idle"
    assert raw["audio_used"] is False
    assert cap["total"] == 3 and len(cap["frames"]) == 3
    assert cap["frames"][0].shape == (64, 96, 3)
    clip = vreg.get_engine(NAME).canonicalize(raw, _req(frames=3), {})
    assert clip["has_audio"] is False and clip["engine_id"] == NAME
    assert clip["qc"]["mode"] == "idle" and clip["qc"]["audio_used"] is False


def test_render_contract_audio_present_reactive(monkeypatch, tmp_path):
    import soundfile as sf
    wav = tmp_path / "beat.wav"
    sr = 24000
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    sf.write(str(wav), tone, sr)
    raw, cap = _render_capturing_frames(monkeypatch, _req(frames=4, audio=str(wav)))
    assert raw["frame_count"] == 4 and raw["mode"] == "reactive"
    assert raw["audio_used"] is True
    assert len(cap["frames"]) == 4


def test_paint_golden_camera_frame_is_deterministic_and_visible():
    w, h = 128, 72
    scan = sd.build_scanlines(w, h)
    vig = sd.build_vignette(w, h)
    freq = np.linspace(0.0, 1.0, 32).astype(np.float32)
    wave = np.zeros(200, dtype=np.float32)
    kw = dict(vol=0.5, freq=freq, wave=wave, signal=0.4, loss=0.6,
              scanlines=scan, vignette=vig, rng_key="viz_camera|7")
    a = np.asarray(sd.paint_golden_camera_frame(w, h, 1, 3, 25, **kw))
    b = np.asarray(sd.paint_golden_camera_frame(w, h, 1, 3, 25, **kw))
    assert np.array_equal(a, b)
    assert a.shape == (h, w, 3)
    assert float(np.mean(a)) > 8.0
    assert np.count_nonzero(a[:, :, 0] > a[:, :, 2]) > w * h * 0.20


def test_cold_import_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_viz_camera;"
        "heavy=[m for m in ('torch','transformers','diffusers','cairo') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
