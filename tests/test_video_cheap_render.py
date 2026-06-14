"""CPU tests for the radio-floor ffmpeg render (cheap_families.render_clip).

The cheap families are the always-succeeds radio floor: render_clip produces the
platform's silent bt709 / yuv420p clip with ffmpeg (no heavy model). This is the
fallback-chain terminus the A-S6 chain humo -> latentsync -> still_kenburns
converges on and the M1 episode's CPU path, so it is proven end-to-end here on the
build box (ffmpeg present); the ffmpeg-running tests skip cleanly without it.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import cheap_families  # noqa: F401  (registers floor)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None
_FAMILIES = ("abstract", "still_kenburns", "station_card", "visualizer", "flux_still")


def _req(frames=6, w=96, h=64, fps=25, **extra):
    r = {"shot_id": "s1", "canvas": {"w": w, "h": h, "fps": fps},
         "timing": {"target_frame_count": frames}, "text_prompt": "test card"}
    r.update(extra)
    return r


def _probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=" + ",".join(entries),
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return out.stdout.split()


def _has_audio(path):
    a = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return a.stdout.strip() != ""


# --- clip-dict contract + pure helpers (no ffmpeg) ------------------------- #
def test_floor_clip_contract_shape():
    eng = vreg.get_engine("still_kenburns")
    clip = eng._floor_clip(_req(), "C:/o.mp4", 25, 30)
    assert clip["has_audio"] is False and clip["pixel_format"] == "yuv420p"
    assert clip["color_primaries"] == clip["transfer"] == clip["matrix"] == "bt709"
    assert clip["engine_id"] == "still_kenburns"
    assert clip["family"] == "static_motion"
    assert clip["frame_count"] == 30 and clip["fps"] == 25 and clip["clip_id"] == "s1"


def test_canvas_and_frame_defaults():
    eng = vreg.get_engine("abstract")
    assert eng._canvas_dims({}) == (832, 480, 25)        # platform defaults
    assert eng._canvas_dims(
        {"canvas": {"w": 1280, "h": 720, "fps": 24}}) == (1280, 720, 24)
    assert eng._frame_count({}, 25) == 25                # 1s default when absent
    assert eng._frame_count({"timing": {"target_frame_count": 50}}, 25) == 50


def test_still_path_extraction():
    eng = vreg.get_engine("still_kenburns")
    assert eng._still_path({"asset_refs": {"init_image": "C:/p.png"}}) == "C:/p.png"
    assert eng._still_path({"asset_refs": {"still": {"path": "C:/s.png"}}}) == "C:/s.png"
    assert eng._still_path({"asset_refs": {}}) == ""


def test_uses_still_flags():
    assert vreg.get_engine("still_kenburns").uses_still is True
    assert vreg.get_engine("station_card").uses_still is True
    assert vreg.get_engine("flux_still").uses_still is True
    assert vreg.get_engine("abstract").uses_still is False
    assert vreg.get_engine("visualizer").uses_still is False


def test_flux_still_fits_all_roles_bug401():
    """BUG-LOCAL-401: flux_still is the fast 'just a still' pick and must be valid
    in EVERY video role -- it needs only text_prompt, which every role supplies.
    A missing role tag (music_visual / background_abstract) made
    music_video_model='flux_still' fail OTR_VideoDirector validation at execute
    even though a still is perfectly valid for a music beat."""
    from nodes._otr_shared import role_compat as rc
    eng = vreg.get_engine("flux_still")
    desc = {"engine_id": "flux_still", "roles": tuple(eng.roles),
            "required_inputs": tuple(eng.required_inputs)}
    for role in ("announcer_visual", "music_visual", "character_video",
                 "scene_broll", "background_abstract"):
        assert rc.engine_fits_role(desc, role), f"flux_still must fit role {role!r}"


# --- real ffmpeg renders (the floor leaf) ---------------------------------- #
@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg not on PATH")
@pytest.mark.parametrize("name", _FAMILIES)
def test_each_family_renders_silent_clip(name):
    eng = vreg.get_engine(name)
    clip = eng.render_clip(_req(frames=6), None)
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and p.stat().st_size > 0
        assert clip["frame_count"] == 6 and clip["has_audio"] is False
        fields = _probe(p, "nb_read_frames", "pix_fmt")
        assert "yuv420p" in fields and "6" in fields
        assert not _has_audio(p)
        assert eng.canonicalize(clip, _req(), {}) is clip   # identity
    finally:
        p.unlink(missing_ok=True)


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg not on PATH")
def test_still_kenburns_over_a_real_still(tmp_path):
    still = tmp_path / "still.png"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x223344:s=80x80",
                    "-frames:v", "1", str(still)], check=True, capture_output=True)
    eng = vreg.get_engine("still_kenburns")
    clip = eng.render_clip(_req(frames=8, asset_refs={"init_image": str(still)}), None)
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 8
        fields = _probe(p, "nb_read_frames", "pix_fmt")
        assert "yuv420p" in fields and "8" in fields
        assert not _has_audio(p)
    finally:
        p.unlink(missing_ok=True)


# --- cold-import + ASCII / no BOM ------------------------------------------ #
def test_cheap_families_cold_import_no_heavy_libs():
    code = ("import sys;"
            "import nodes._otr_video_engines.cheap_families;"
            "heavy=[m for m in ('torch','transformers','diffusers','numpy') "
            "if m in sys.modules];"
            "print('HEAVY', heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_cheap_families_source_ascii_no_bom():
    p = REPO_ROOT / "nodes" / "_otr_video_engines" / "cheap_families.py"
    raw = p.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf"
    src = raw.decode("utf-8")
    assert chr(0x2014) not in src
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
