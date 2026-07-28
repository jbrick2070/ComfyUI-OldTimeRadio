r"""viz_mxc_mandala -- the Cosmic Radio Mandala pycairo engine (2026-06-30,
kibitz r1-r4 hardened).

Registration/capability/wiring/cold-import tests stay cairo-free (importorskip
gates only the paint/determinism/visual-smoke tests, per the plan). Covers:
registration + CAPABILITIES; required_inputs=() fits all 5 roles by capability;
accepts_still=False (mints no still); the ambient-audio gate; content-oracle
motion-exemption; cold-import (cairo NEVER at module scope); the missing-cairo
assert_usable LOUD-fail path (monkeypatched, unskipped even with cairo present);
the silent-clip render contract for BOTH audio-present (reactive) and
audio-absent (idle) paths; frame-count exactness; paint + CRT-post determinism;
and a visual-acceptance smoke (nonblack ratio + frame-to-frame delta +
reproducibility). UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

import nodes._otr_video_engines  # noqa: F401  (self-registers every engine)
from nodes._otr_shared import role_compat as rc
from nodes._otr_shared import content_oracle as co
from nodes._otr_shared import scope_draw as sd
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes.otr_image_gen_dispatcher import engine_consumes_still

NAME = "viz_mxc_mandala"


def test_registered_with_capability_row():
    assert NAME in vreg.all_engine_names()
    eng = vreg.get_engine(NAME)
    assert eng.name == NAME and eng.family == "abstract"
    assert eng.accepts_still is False
    assert tuple(eng.required_inputs) == ()
    assert eng.fallback_engine is None and eng.engine_version == "1"
    assert eng.render_aspect == "wide"
    assert NAME in vreg.CAPABILITIES
    assert "cpu" in vreg.CAPABILITIES[NAME]["device_backends"]
    assert vreg.CAPABILITIES[NAME]["required_toolchain"] is None


def test_required_inputs_empty_fits_all_5_roles():
    desc = vreg.descriptor_for_engine(NAME)
    for role in rc.ROLES:
        assert rc.engine_fits_role(desc, role), role
    # registry eligibility agrees (capability, C2)
    for role in rc.ROLES:
        assert vreg.assert_usable(NAME, role) == NAME


def test_accepts_still_false_mints_no_still():
    assert engine_consumes_still(vreg.get_engine(NAME)) is False


def test_ambient_master_audio_gate_includes_mandala():
    # so a music-open / no-line beat gets its bounded master slice (else audio-starved)
    assert rd._uses_ambient_master_audio(NAME, "abstract") is True
    # a character-face beat is still excluded (contract)
    assert rd._uses_ambient_master_audio(NAME, "abstract", is_char_face=True) is False


def test_content_oracle_motion_exempt():
    assert co.motion_required_for_engine(NAME) is False
    assert co.family_for_engine(NAME) == "abstract"


def test_engine_family_map():
    assert rd.engine_family(NAME) == "abstract"


def test_cold_import_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_viz_mandala;"
        "heavy=[m for m in ('torch','transformers','diffusers','cairo') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs (incl. cairo) at import:\n{r.stdout}\n{r.stderr}"


def test_assert_usable_missing_cairo_fails_loud(monkeypatch):
    """Forces ``import cairo`` to raise ImportError (regardless of whether cairo
    is actually installed on the test box) via the sys.modules-None sentinel
    trick, proving the LOUD fail-closed path unconditionally -- this must NOT
    be skipped even when pycairo IS present."""
    monkeypatch.setitem(sys.modules, "cairo", None)
    eng = vreg.get_engine(NAME)
    with pytest.raises(vreg.EngineUnusable, match="pycairo"):
        eng.assert_usable({}, {})


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


def test_render_contract_audio_absent_idle(monkeypatch, tmp_path):
    pytest.importorskip("cairo")
    raw, cap = _render_capturing_frames(monkeypatch, _req(frames=3))
    assert raw["frame_count"] == 3 and raw["mode"] == "idle"
    assert raw["audio_used"] is False
    assert cap["total"] == 3 and len(cap["frames"]) == 3          # frame-count exact
    assert cap["frames"][0].shape == (64, 96, 3)                  # RGB canvas
    clip = vreg.get_engine(NAME).canonicalize(raw, _req(frames=3), {})
    assert clip["has_audio"] is False and clip["engine_id"] == NAME
    assert clip["qc"]["mode"] == "idle" and clip["qc"]["audio_used"] is False


def test_render_contract_audio_present_reactive(monkeypatch, tmp_path):
    pytest.importorskip("cairo")
    import soundfile as sf
    wav = tmp_path / "beat.wav"
    sr = 24000
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    sf.write(str(wav), tone, sr)
    raw, cap = _render_capturing_frames(monkeypatch, _req(frames=4, audio=str(wav)))
    assert raw["frame_count"] == 4 and raw["mode"] == "reactive"
    assert raw["audio_used"] is True
    assert len(cap["frames"]) == 4


def test_paint_mandala_is_deterministic():
    pytest.importorskip("cairo")
    import cairo
    w, h = 96, 64
    freq = np.linspace(0.0, 1.0, 32).astype(np.float32)

    def _paint_once():
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        sd.paint_mandala(ctx, w, h, 1, 3, 25, 0.5, freq, 0.4, 0.0)
        return sd.mandala_surface_to_rgb(surface, w, h)

    a = _paint_once()
    b = _paint_once()
    assert a.shape == (h, w, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)                                   # V-7 determinism


def test_apply_crt_post_rgb_deterministic_and_shape():
    w, h = 96, 64
    scan = sd.build_scanlines(w, h)
    vig = sd.build_vignette(w, h)
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    a = sd.apply_crt_post_rgb(rgb, scan, vig, 1, "mandala-test", vol=0.5)
    b = sd.apply_crt_post_rgb(rgb, scan, vig, 1, "mandala-test", vol=0.5)
    assert a.shape == (h, w, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)                                   # V-7 determinism
    # no in-place mutate of the input
    assert np.array_equal(rgb, np.full((h, w, 3), 128, dtype=np.uint8))


def test_visual_acceptance_smoke_nonblack_and_changes(monkeypatch, tmp_path):
    """Build-time visual QA (not a look/regression freeze): the rendered clip is
    not a black/empty floor, frames change beat-to-beat (catches an accidental
    frozen-frame bug -- the content-oracle itself treats 'abstract' as
    motion-EXEMPT; this is a separate build-time sanity check), and the same
    request/seed reproduces byte-identical frames."""
    pytest.importorskip("cairo")
    import soundfile as sf
    wav = tmp_path / "beat.wav"
    sr = 24000
    secs = 1.0
    tone = 0.4 * np.sin(2 * np.pi * 330 * np.arange(int(sr * secs)) / sr).astype(np.float32)
    sf.write(str(wav), tone, sr)
    raw, cap = _render_capturing_frames(
        monkeypatch, _req(frames=10, w=192, h=108, audio=str(wav)))
    frames = cap["frames"]
    assert len(frames) == 10
    for f in frames:
        nonblack = float(np.mean(f > 8))
        assert nonblack > 0.02, "frame reads as a black/empty floor"
    deltas = [float(np.mean(np.abs(frames[i].astype(np.int16) -
                                   frames[i - 1].astype(np.int16))))
              for i in range(1, len(frames))]
    assert any(d > 0.05 for d in deltas), "frames never change beat-to-beat"
    # reproducibility: same request + same seed -> byte-identical frame hashes
    raw2, cap2 = _render_capturing_frames(
        monkeypatch, _req(frames=10, w=192, h=108, audio=str(wav)))
    h1 = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in frames]
    h2 = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in cap2["frames"]]
    assert h1 == h2                                               # V-7 determinism
