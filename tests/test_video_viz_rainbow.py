"""viz_mxc_cpu -- the OTR rainbow visualizer (2026-06-30, kibitz-hardened).

CPU-only contract tests (no GPU, no real ffmpeg -- encode_silent_mp4 is
monkeypatched). Covers: registration + CAPABILITIES; required_inputs=() fits all 5
roles by capability; accepts_still=False (mints no still); the ambient-audio gate;
content-oracle motion-exemption; the silent-clip render contract for BOTH the
audio-present (reactive) and audio-absent (idle) paths; frame-count exactness; and
paint determinism. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

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

NAME = "viz_mxc_cpu"


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
    # registry eligibility agrees (capability, C2)
    for role in rc.ROLES:
        assert vreg.assert_usable(NAME, role) == NAME


def test_accepts_still_false_mints_no_still():
    assert engine_consumes_still(vreg.get_engine(NAME)) is False


def test_ambient_master_audio_gate_includes_viz_mxc():
    # so a music-open / no-line beat gets its bounded master slice (else audio-starved)
    assert rd._uses_ambient_master_audio(NAME, "abstract") is True
    # a character-face beat is still excluded (contract)
    assert rd._uses_ambient_master_audio(NAME, "abstract", is_char_face=True) is False


def test_content_oracle_motion_exempt():
    assert co.motion_required_for_engine(NAME) is False
    assert co.family_for_engine(NAME) == "abstract"


def test_engine_family_map():
    assert rd.engine_family(NAME) == "abstract"


def _req(frames=3, w=96, h=64, seed=7, audio=None):
    r = {"shot_id": "s1", "canvas": {"w": w, "h": h, "fps": 25},
         "timing": {"target_frame_count": frames}, "seed_bundle": {"request_seed": seed}}
    if audio:
        r["audio_ref"] = {"path": audio}
    return r


def _render_capturing_frames(monkeypatch, request):
    captured = {}

    def _fake_encode(frames_iter, total, out_path, w, h, fps, ffmpeg):
        captured["frames"] = [np.asarray(f) for f in frames_iter]
        captured["total"] = total
        # write a stub file so any exists() check passes
        with open(out_path, "wb") as fh:
            fh.write(b"\x00")
        return out_path

    monkeypatch.setattr(sd, "encode_silent_mp4", _fake_encode)
    raw = vreg.get_engine(NAME).render_clip(request)
    return raw, captured


def test_render_contract_audio_absent_idle(monkeypatch, tmp_path):
    raw, cap = _render_capturing_frames(monkeypatch, _req(frames=3))
    assert raw["frame_count"] == 3 and raw["mode"] == "idle"
    assert raw["audio_used"] is False
    assert cap["total"] == 3 and len(cap["frames"]) == 3          # frame-count exact
    assert cap["frames"][0].shape == (64, 96, 3)                  # RGB canvas
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


def test_paint_rainbow_frame_is_deterministic():
    w, h = 96, 64
    scan = sd.build_scanlines(w, h)
    vig = sd.build_vignette(w, h)
    font = sd._small_font(h)
    freq = np.linspace(0.0, 1.0, 32).astype(np.float32)
    wave = np.zeros(200, dtype=np.float32)
    kw = dict(vol=0.5, freq=freq, wave=wave, signal=0.4, loss=0.6,
              scanlines=scan, vignette=vig, rng_key="viz_mxc_cpu|7", font_small=font)
    a = np.asarray(sd.paint_rainbow_frame(w, h, 1, 3, 25, **kw))
    b = np.asarray(sd.paint_rainbow_frame(w, h, 1, 3, 25, **kw))
    assert np.array_equal(a, b)                                   # V-7 determinism


def test_cold_import_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_viz_rainbow;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
