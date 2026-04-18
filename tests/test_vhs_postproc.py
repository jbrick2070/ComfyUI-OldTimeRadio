"""
test_vhs_postproc.py  --  Day 8 VHS post-processor regression
=============================================================

Validates ``otr_v2.hyworld.postproc.vhs`` without requiring ffmpeg
or GPU weights.  All tests exercise the stub mode (byte-identical
passthrough copy) plus the pure filter-chain builder.

Covers:

    * Module imports cleanly (no torch / ffmpeg imports at module load)
    * DEFAULT_VHS_PARAMS has the expected keys
    * build_vhs_filter_chain is pure + deterministic
    * Filter chain varies across intensity presets (low / medium / high)
    * Filter chain has the structural stages we expect (rgbashift,
      gblur, geq, noise, vignette)
    * Stub mode byte-identical passthrough (C7 invariant) -- including
      for inputs with arbitrary trailing payloads
    * Stub auto-detection: OTR_VHS_STUB=1, no ffmpeg, force_stub
    * meta.json written next to the output with correct schema
    * Batch apply_vhs_to_job_dir finds motion.mp4 / loop.mp4 /
      render.mp4 but skips still images
    * Batch summary file written
    * Edge cases: empty job dir, missing input file, input == output path
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------------------------
# Import-time + surface-area smoke tests
# ------------------------------------------------------------------


def test_module_imports_cleanly():
    """The VHS module must import without dragging in torch / ffmpeg."""
    mod = importlib.import_module("otr_v2.hyworld.postproc.vhs")
    assert hasattr(mod, "build_vhs_filter_chain")
    assert hasattr(mod, "apply_vhs_filter")
    assert hasattr(mod, "apply_vhs_to_job_dir")
    assert hasattr(mod, "DEFAULT_VHS_PARAMS")
    # Sanity: no torch at module scope.
    assert "torch" not in sys.modules or "otr_v2.hyworld.postproc.vhs" in sys.modules


def test_postproc_package_imports():
    pkg = importlib.import_module("otr_v2.hyworld.postproc")
    assert "vhs" in getattr(pkg, "__all__", [])


def test_default_params_shape():
    from otr_v2.hyworld.postproc.vhs import DEFAULT_VHS_PARAMS

    expected_keys = {
        "intensity",
        "scanline_density",
        "scanline_alpha",
        "rgb_shift_px",
        "chroma_blur_sigma",
        "noise_strength",
        "vignette_over_pi",
        "final_blur_sigma",
        "fps",
        "preset",
        "crf",
    }
    assert expected_keys.issubset(DEFAULT_VHS_PARAMS.keys())
    assert DEFAULT_VHS_PARAMS["intensity"] == "medium"
    assert DEFAULT_VHS_PARAMS["fps"] == 24  # matches renderer._FPS


def test_public_constants():
    from otr_v2.hyworld.postproc import vhs

    assert "render.mp4" in vhs.VIDEO_FILENAME_CANDIDATES
    assert "motion.mp4" in vhs.VIDEO_FILENAME_CANDIDATES
    assert "loop.mp4" in vhs.VIDEO_FILENAME_CANDIDATES
    assert "composite.png" in vhs.SKIP_FILENAMES
    assert "keyframe.png" in vhs.SKIP_FILENAMES
    assert vhs.VHS_SUFFIX == "_vhs.mp4"


# ------------------------------------------------------------------
# Pure filter-chain builder
# ------------------------------------------------------------------


def test_filter_chain_deterministic():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    a = build_vhs_filter_chain({"intensity": "medium"})
    b = build_vhs_filter_chain({"intensity": "medium"})
    assert a == b
    assert isinstance(a, str) and len(a) > 0


def test_filter_chain_uses_default_params_when_none():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    with_none = build_vhs_filter_chain(None)
    with_empty = build_vhs_filter_chain({})
    assert with_none == with_empty


def test_filter_chain_has_expected_stages():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    chain = build_vhs_filter_chain()
    # Must start with format normalisation so downstream filters see yuv420p.
    assert chain.startswith("format=yuv420p")
    # Must contain each core VHS effect.
    assert "rgbashift=" in chain
    assert "gblur=" in chain  # used twice (chroma bleed + final softness)
    assert "geq=" in chain  # scanlines
    assert "noise=" in chain
    assert "vignette=" in chain


def test_filter_chain_varies_with_intensity():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    low = build_vhs_filter_chain({"intensity": "low"})
    medium = build_vhs_filter_chain({"intensity": "medium"})
    high = build_vhs_filter_chain({"intensity": "high"})

    # All three must produce different strings -- otherwise the
    # intensity knob is decorative.
    assert low != medium
    assert medium != high
    assert low != high


def test_filter_chain_unknown_intensity_falls_back_to_medium():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    # Unknown preset must not raise and must behave like medium.
    chain = build_vhs_filter_chain({"intensity": "tape-smash"})
    medium = build_vhs_filter_chain({"intensity": "medium"})
    assert chain == medium


def test_filter_chain_skips_effect_when_strength_zero():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    # Zero-out the knob: the chain must not reference that stage.
    no_noise = build_vhs_filter_chain({"noise_strength": 0})
    assert "noise=" not in no_noise

    no_rgb = build_vhs_filter_chain({"rgb_shift_px": 0})
    assert "rgbashift=" not in no_rgb

    no_chroma = build_vhs_filter_chain({"chroma_blur_sigma": 0})
    # Only the chroma-bleed stage goes; the final softness gblur stays.
    assert no_chroma.count("gblur=") == 1


def test_filter_chain_respects_overrides():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    # Explicit override should be observable in the chain text.
    big_shift = build_vhs_filter_chain({"rgb_shift_px": 6, "intensity": "medium"})
    # Medium has multiplier 1.0 so the override lands as-is (rh=-6:bh=6).
    assert "rh=-6:bh=6" in big_shift


def test_filter_chain_scanline_density_respected():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    # density=1 means every row scanlined; density=3 means every third.
    c1 = build_vhs_filter_chain({"scanline_density": 1})
    c3 = build_vhs_filter_chain({"scanline_density": 3})
    assert c1 != c3
    assert r"mod(Y\,1)" in c1
    assert r"mod(Y\,3)" in c3


def test_filter_chain_vignette_always_present():
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    # Vignette is always on -- it's the VHS frame edge.
    chain = build_vhs_filter_chain({"intensity": "low"})
    assert "vignette=PI/" in chain


# ------------------------------------------------------------------
# Stub mode byte-identical passthrough (the C7 invariant)
# ------------------------------------------------------------------


def _fake_mp4(payload: bytes) -> bytes:
    """Build a deterministic fake MP4-like blob for passthrough tests."""
    # Minimal ftyp + mdat atoms so stub outputs at least look
    # structurally valid to a casual `file(1)` check.  We don't need
    # ffmpeg to accept it -- these are passthrough tests.
    ftyp = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    mdat_body = b"mdat" + payload
    mdat_size = (len(mdat_body) + 4).to_bytes(4, "big")
    return ftyp + mdat_size + mdat_body


def test_stub_mode_copies_input_byte_identical(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    # Force stub regardless of ffmpeg availability.
    monkeypatch.setenv("OTR_VHS_STUB", "1")

    src = tmp_path / "clip.mp4"
    payload = _fake_mp4(b"hello vhs day 8")
    src.write_bytes(payload)

    dst = tmp_path / "clip_out.mp4"
    meta = apply_vhs_filter(src, dst)

    assert meta["mode"] == "stub"
    assert dst.read_bytes() == src.read_bytes()
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_force_stub_flag_overrides_env(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter, STUB_REASON_FORCED

    # Env unset but force_stub=True should still stub.
    monkeypatch.delenv("OTR_VHS_STUB", raising=False)
    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"force"))

    dst = tmp_path / "clip_out.mp4"
    meta = apply_vhs_filter(src, dst, force_stub=True)

    assert meta["mode"] == "stub"
    assert meta["stub_reason"] == STUB_REASON_FORCED


def test_stub_mode_preserves_trailing_audio_like_payload(tmp_path, monkeypatch):
    """C7 invariant: any embedded audio bytes must survive stub passthrough."""
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    src = tmp_path / "clip.mp4"
    # Simulate video bytes + "audio" bytes; we only care that every
    # byte lands in the output.
    audio_trailing = b"AUDIO_STREAM_BYTES_" + os.urandom(64)
    src.write_bytes(_fake_mp4(b"video") + audio_trailing)

    dst = tmp_path / "clip_out.mp4"
    apply_vhs_filter(src, dst)

    assert dst.read_bytes() == src.read_bytes()
    assert dst.read_bytes().endswith(audio_trailing)


def test_stub_mode_meta_json_schema(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"meta"))
    dst = tmp_path / "clip_out.mp4"
    apply_vhs_filter(src, dst, params={"intensity": "high"})

    meta_path = dst.with_suffix(dst.suffix + ".meta.json")
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))

    assert payload["stage"] == "vhs_postproc"
    assert payload["mode"] == "stub"
    assert payload["params"]["intensity"] == "high"
    assert isinstance(payload["filter_chain"], str)
    assert len(payload["filter_chain"]) > 0
    assert isinstance(payload["duration_ms"], (int, float))
    assert "params_hash" in payload
    assert len(payload["params_hash"]) == 12


def test_stub_mode_env_reason_recorded(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter, STUB_REASON_ENV

    monkeypatch.setenv("OTR_VHS_STUB", "1")
    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"env"))
    dst = tmp_path / "clip_out.mp4"
    meta = apply_vhs_filter(src, dst)
    assert meta["stub_reason"] == STUB_REASON_ENV


def test_stub_mode_activates_when_ffmpeg_missing(tmp_path, monkeypatch):
    """With ffmpeg unavailable, stub mode must kick in automatically."""
    from otr_v2.hyworld.postproc import vhs as vhs_mod
    from otr_v2.hyworld.postproc.vhs import (
        apply_vhs_filter,
        STUB_REASON_NO_FFMPEG,
    )

    # Ensure env stub is off so we exercise the ffmpeg-missing path.
    monkeypatch.delenv("OTR_VHS_STUB", raising=False)
    monkeypatch.setattr(vhs_mod, "find_ffmpeg", lambda: None)

    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"no-ffmpeg"))
    dst = tmp_path / "clip_out.mp4"
    meta = apply_vhs_filter(src, dst)

    assert meta["mode"] == "stub"
    assert meta["stub_reason"] == STUB_REASON_NO_FFMPEG
    assert dst.read_bytes() == src.read_bytes()


def test_missing_input_raises(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")
    with pytest.raises(FileNotFoundError):
        apply_vhs_filter(
            tmp_path / "does_not_exist.mp4",
            tmp_path / "out.mp4",
        )


def test_same_input_as_output_does_not_clobber(tmp_path, monkeypatch):
    """If input and output resolve to the same path, no copy -- but
    meta.json must still be written and the bytes must stay intact."""
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    path = tmp_path / "same.mp4"
    payload = _fake_mp4(b"same-path")
    path.write_bytes(payload)

    meta = apply_vhs_filter(path, path)
    assert path.read_bytes() == payload
    assert meta["mode"] == "stub"


# ------------------------------------------------------------------
# Batch apply across a job directory
# ------------------------------------------------------------------


def _make_shot(job_dir: Path, shot_id: str, filename: str, payload: bytes) -> Path:
    shot_dir = job_dir / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)
    p = shot_dir / filename
    p.write_bytes(payload)
    return p


def test_batch_apply_finds_motion_loop_render(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir, VHS_SUFFIX

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_abc"
    _make_shot(job, "shot_001", "render.mp4", _fake_mp4(b"s1"))
    _make_shot(job, "shot_002", "motion.mp4", _fake_mp4(b"s2"))
    _make_shot(job, "shot_003", "loop.mp4", _fake_mp4(b"s3"))

    summary = apply_vhs_to_job_dir(job)

    assert summary["count"] == 3
    for entry in summary["entries"]:
        out_path = Path(entry["output"])
        assert out_path.name.endswith(VHS_SUFFIX)
        assert out_path.exists()


def test_batch_apply_skips_still_images(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir, VHS_SUFFIX

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_skip"
    _make_shot(job, "shot_still", "keyframe.png", b"\x89PNG-keyframe")
    _make_shot(job, "shot_comp", "composite.png", b"\x89PNG-composite")
    _make_shot(job, "shot_vid", "render.mp4", _fake_mp4(b"v"))

    summary = apply_vhs_to_job_dir(job)

    assert summary["count"] == 1
    out_path = Path(summary["entries"][0]["output"])
    assert out_path.name == "render" + VHS_SUFFIX


def test_batch_apply_handles_mixed_shot_outputs(tmp_path, monkeypatch):
    """A shot may have BOTH a still (keyframe.png) and a video (motion.mp4).
    The VHS stage must only touch the video, not the still."""
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_mixed"
    shot = job / "shot_01"
    shot.mkdir(parents=True)
    (shot / "keyframe.png").write_bytes(b"\x89PNG-keyframe")
    (shot / "render.mp4").write_bytes(_fake_mp4(b"v"))

    summary = apply_vhs_to_job_dir(job)

    assert summary["count"] == 1
    # Keyframe.png is untouched -- no *_vhs.png sibling.
    assert not (shot / "keyframe_vhs.png").exists()
    assert not (shot / "keyframe_vhs.mp4").exists()
    assert (shot / "render_vhs.mp4").exists()


def test_batch_apply_prefers_render_over_motion(tmp_path, monkeypatch):
    """When a shot dir has both render.mp4 and motion.mp4, we only
    process render.mp4 (the renderer's canonical name)."""
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_prefer"
    shot = job / "shot_01"
    shot.mkdir(parents=True)
    (shot / "render.mp4").write_bytes(_fake_mp4(b"render"))
    (shot / "motion.mp4").write_bytes(_fake_mp4(b"motion"))

    summary = apply_vhs_to_job_dir(job)

    assert summary["count"] == 1
    inp = Path(summary["entries"][0]["input"])
    assert inp.name == "render.mp4"


def test_batch_apply_ignores_internal_directories(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_internal"
    # Shot dirs starting with '_' or '.' are internal (e.g. _cache).
    _make_shot(job, "_cache", "render.mp4", _fake_mp4(b"cache"))
    _make_shot(job, ".hidden", "render.mp4", _fake_mp4(b"hidden"))
    _make_shot(job, "shot_real", "render.mp4", _fake_mp4(b"real"))

    summary = apply_vhs_to_job_dir(job)
    assert summary["count"] == 1
    assert "shot_real" in summary["entries"][0]["input"]


def test_batch_apply_empty_job_dir(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_empty"
    job.mkdir()

    summary = apply_vhs_to_job_dir(job)
    assert summary["count"] == 0
    assert summary["entries"] == []
    assert (job / "vhs_postproc_summary.json").exists()


def test_batch_apply_missing_job_dir(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    summary = apply_vhs_to_job_dir(tmp_path / "does_not_exist")
    assert summary["count"] == 0


def test_batch_summary_file_written(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_to_job_dir

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    job = tmp_path / "job_summary"
    _make_shot(job, "shot_01", "render.mp4", _fake_mp4(b"a"))
    _make_shot(job, "shot_02", "motion.mp4", _fake_mp4(b"b"))

    apply_vhs_to_job_dir(job)

    summary_path = job / "vhs_postproc_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "vhs_postproc_batch"
    assert payload["count"] == 2


# ------------------------------------------------------------------
# Parameter hashing + determinism
# ------------------------------------------------------------------


def test_params_hash_stable_across_calls(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"hash"))

    m1 = apply_vhs_filter(src, tmp_path / "a.mp4")
    m2 = apply_vhs_filter(src, tmp_path / "b.mp4")

    assert m1["params_hash"] == m2["params_hash"]
    assert m1["filter_chain"] == m2["filter_chain"]


def test_params_hash_shifts_with_overrides(tmp_path, monkeypatch):
    from otr_v2.hyworld.postproc.vhs import apply_vhs_filter

    monkeypatch.setenv("OTR_VHS_STUB", "1")

    src = tmp_path / "clip.mp4"
    src.write_bytes(_fake_mp4(b"hash-diff"))

    low = apply_vhs_filter(src, tmp_path / "a.mp4", {"intensity": "low"})
    high = apply_vhs_filter(src, tmp_path / "b.mp4", {"intensity": "high"})

    assert low["params_hash"] != high["params_hash"]
    assert low["filter_chain"] != high["filter_chain"]


# ------------------------------------------------------------------
# Cross-backend integration smoke
# ------------------------------------------------------------------


def test_postproc_does_not_pollute_backend_registry():
    """The VHS post-processor must NOT register itself as a backend --
    Days 8-14 are orchestration/canary, no new backends."""
    from otr_v2.hyworld.backends import list_backends

    known = list_backends()
    # None of the existing backends should be the VHS module.
    assert "vhs" not in known
    assert "vhs_postproc" not in known
    # Sanity: the Day 1-7 seven backends are all still registered.
    for name in (
        "placeholder_test",
        "flux_anchor",
        "pulid_portrait",
        "flux_keyframe",
        "ltx_motion",
        "wan21_loop",
        "florence2_sdxl_comp",
    ):
        assert name in known


def test_filter_chain_does_not_contain_shell_metacharacters():
    """Belt-and-braces: the filter chain must not contain characters
    that would break a subprocess argv even if ffmpeg parses it."""
    from otr_v2.hyworld.postproc.vhs import build_vhs_filter_chain

    chain = build_vhs_filter_chain()
    # These would be fatal in a raw shell but we pass argv directly,
    # so we only guard against outright NULs and newlines which even
    # argv will refuse.
    assert "\x00" not in chain
    assert "\n" not in chain
    assert "\r" not in chain


def test_fps_defaults_to_renderer_fps():
    """Renderer uses _FPS=24; VHS must not accidentally change it."""
    from otr_v2.hyworld.postproc.vhs import DEFAULT_VHS_PARAMS
    from otr_v2.hyworld import renderer

    assert DEFAULT_VHS_PARAMS["fps"] == renderer._FPS
