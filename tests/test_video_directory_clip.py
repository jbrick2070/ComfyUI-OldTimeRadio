"""Directory-clip contract + OTR_SilentComposite read path (3D plan 7.2).

CPU-only coverage for the character_3d alpha handoff: the canonicalize-time
validator (type / pixel_format=rgba / alpha=straight / has_audio=False /
frame_count == frames on disk), the shared "dir exists + exactly N sorted
nonzero frames" rule consumed by _clip_summary / build_clip_manifest, and the
GOLDEN straight-alpha composite: frames sorted by name -> overlay -> flatten
yuv420p, with the flattened pixels actually probed (half-alpha red over black
== half red; alpha-0 regions show the background). The webm/vp9 +
mov/prores4444 alpha-video branches are CUT from v1 -- a frame directory is
the only alpha handoff. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from nodes._otr_video_engines import directory_clip as dc
from nodes._otr_video_engines import render_driver as rd
from nodes import otr_silent_composite as sc

pytest.importorskip("PIL", reason="Pillow renders the RGBA fixture frames")
from PIL import Image  # noqa: E402

W, H = 320, 180          # canonical canvas for this fixture (even dims)
FW, FH = 200, 100        # frame size (smaller -> overlay centering visible)
N_FRAMES = 8
FPS = 5


def _write_frames(dirpath, n=N_FRAMES, alpha=128):
    """n straight-alpha red frames (RGBA PNG), names sortable."""
    os.makedirs(dirpath, exist_ok=True)
    for i in range(n):
        img = Image.new("RGBA", (FW, FH), (255, 0, 0, alpha))
        img.save(os.path.join(dirpath, "frame_%04d.png" % i))


def _dir_clip(path, n=N_FRAMES):
    return {"clip_id": "c3d_0001", "type": "directory", "path": str(path),
            "pixel_format": "rgba", "alpha": "straight", "has_audio": False,
            "frame_count": n, "engine_id": "triposg_talk",
            "family": "character_3d"}


# --------------------------------------------------------------------------- #
# the canonicalize-time validator
# --------------------------------------------------------------------------- #
def test_validator_happy_path_returns_sorted_frames(tmp_path):
    d = tmp_path / "frames"
    _write_frames(d)
    frames = dc.validate_directory_clip(_dir_clip(d), expect_frames=N_FRAMES)
    assert len(frames) == N_FRAMES
    assert frames == sorted(frames)               # sorted-by-name contract


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c.__setitem__("type", "video"), "type"),
    (lambda c: c.__setitem__("pixel_format", "yuv420p"), "pixel_format"),
    (lambda c: c.__setitem__("alpha", "premultiplied"), "alpha"),
    (lambda c: c.__setitem__("has_audio", True), "has_audio"),
    (lambda c: c.__setitem__("frame_count", N_FRAMES + 1), "frame_count"),
])
def test_validator_fails_closed_on_contract_violations(tmp_path, mutate, needle):
    d = tmp_path / "frames"
    _write_frames(d)
    clip = _dir_clip(d)
    mutate(clip)
    with pytest.raises(ValueError, match=needle):
        dc.validate_directory_clip(clip)


def test_validator_rejects_missing_dir_and_zero_byte_frame(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        dc.validate_directory_clip(_dir_clip(tmp_path / "nope"))
    d = tmp_path / "frames"
    _write_frames(d)
    (d / "frame_0003.png").write_bytes(b"")       # a partial render
    with pytest.raises(ValueError, match="zero-byte"):
        dc.validate_directory_clip(_dir_clip(d))


# --------------------------------------------------------------------------- #
# THE V-1 AUDIO LAW FOR A DIRECTORY CLIP (lane 10, 2026-08-11 -- lesson L4)
#
# Every mp4 lane proves silence by ffprobing the emitted file for audio
# streams. A directory has no container to probe, and the old contract's only
# audio evidence was `has_audio is not False` read off the dict the adapter
# itself wrote -- a declaration checking a declaration. The replacement is
# structural: a PNG/EXR still has no audio stream to carry, so PROVE from the
# bytes that every frame really is one.
# --------------------------------------------------------------------------- #
def test_a_frame_is_proved_by_its_MAGIC_BYTES_not_its_extension(tmp_path):
    """`list_directory_frames` selects frames by filename extension, so before
    this a file named 0001.png containing an mp4 -- or a WAV, which is the case
    that makes this an AUDIO law and not a tidiness rule -- counted as a frame
    and shipped as proof of silence."""
    d = tmp_path / "frames"
    _write_frames(d, n=3)
    (d / "frame_0001.png").write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64)
    with pytest.raises(ValueError, match="NOT a png"):
        dc.validate_directory_clip(_dir_clip(d, n=3))
    # The one that would have been quietest of all: real audio, named .png.
    (d / "frame_0001.png").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    with pytest.raises(ValueError) as exc:
        dc.list_directory_frames(str(d))
    assert "frame_0001.png" in str(exc.value)


def test_the_magic_byte_proof_accepts_a_real_exr(tmp_path):
    """EXR is half of the declared FRAME_EXTS contract, so it is proved rather
    than merely tolerated -- otherwise the first EXR lane discovers this the
    hard way."""
    d = tmp_path / "frames"
    os.makedirs(d)
    (d / "frame_0000.exr").write_bytes(b"\x76\x2f\x31\x01" + b"\x00" * 32)
    assert len(dc.list_directory_frames(str(d))) == 1
    assert dc.prove_frame_is_a_silent_image(
        str(d / "frame_0000.exr")) == ".exr"


def test_the_tolerant_summary_also_refuses_an_impostor_frame(tmp_path):
    """`frame_dir_summary` is the read path the manifests and _clip_summary
    use, and it never raises -- so if the proof lived only in the strict
    validator, a receipt could still call an impostor directory real."""
    d = tmp_path / "frames"
    _write_frames(d, n=2)
    assert dc.frame_dir_summary(str(d), expect_frames=2)[0] is True
    (d / "frame_0001.png").write_bytes(b"not a png at all")
    assert dc.frame_dir_summary(str(d), expect_frames=2)[0] is False


def test_validator_rejects_ledger_target_mismatch(tmp_path):
    d = tmp_path / "frames"
    _write_frames(d)
    with pytest.raises(ValueError, match="target"):
        dc.validate_directory_clip(_dir_clip(d), expect_frames=N_FRAMES + 5)


def test_frame_dir_summary_is_tolerant(tmp_path):
    ok, n, size = dc.frame_dir_summary(tmp_path / "nope")
    assert (ok, n, size) == (False, 0, 0)
    d = tmp_path / "frames"
    _write_frames(d)
    ok, n, size = dc.frame_dir_summary(d, expect_frames=N_FRAMES)
    assert ok and n == N_FRAMES and size > 0
    ok, n, _ = dc.frame_dir_summary(d, expect_frames=N_FRAMES + 1)
    assert not ok and n == N_FRAMES               # count mismatch -> not real


# --------------------------------------------------------------------------- #
# _clip_summary + build_clip_manifest directory semantics (7.2 p3)
# --------------------------------------------------------------------------- #
def test_clip_summary_directory_semantics(tmp_path):
    d = tmp_path / "frames"
    _write_frames(d)
    s = rd._clip_summary(_dir_clip(d))
    assert s["exists"] is True and s["size"] > 0
    # wrong declared count -> NOT real (exactly-N rule)
    s = rd._clip_summary(_dir_clip(d, n=N_FRAMES + 1))
    assert s["exists"] is False
    # missing dir -> not real
    s = rd._clip_summary(_dir_clip(tmp_path / "nope"))
    assert s["exists"] is False


def test_build_clip_manifest_directory_row(tmp_path):
    d = tmp_path / "frames"
    _write_frames(d)
    result = {
        "ledger": {"video": {
            "video_revision": 1, "fps": FPS,
            "canonical_canvas": {"w": W, "h": H},
            "shots": [
                {"shot_id": "shot_b001", "source_line_ids": ["b001"],
                 "engine_id": "triposg_talk", "family": "character_3d",
                 "target_frame_count": N_FRAMES},
            ]}},
        "clips": {"shot_b001": _dir_clip(d)},
    }
    m = rd.build_clip_manifest(result, episode_id="ep_dir")
    row = m["clips"][0]
    assert row["type"] == "directory" and row["exists"] is True
    assert m["engine_histogram"] == {"triposg_talk": 1}
    # the exactly-N rule: declare a different target -> not real
    result["ledger"]["video"]["shots"][0]["target_frame_count"] = N_FRAMES + 1
    m2 = rd.build_clip_manifest(result, episode_id="ep_dir")
    assert m2["clips"][0]["exists"] is False


# --------------------------------------------------------------------------- #
# the GOLDEN straight-alpha flatten (manifest -> composite), CPU + ffmpeg
# --------------------------------------------------------------------------- #
def _decode_rgb_frame(path):
    """First frame of ``path`` as raw RGB24 bytes (ffmpeg pipe)."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    assert p.returncode == 0, p.stderr[:300]
    return p.stdout


def _px(buf, x, y, w=W):
    off = (y * w + x) * 3
    return tuple(buf[off:off + 3])


def _manifest_with_dir_clip(d):
    return {
        "episode_id": "ep_dir", "video_revision": 1, "fps": FPS,
        "canvas": {"w": W, "h": H}, "n_beats": 1, "clip_count": 1,
        "total_target_frames": N_FRAMES,
        "engine_histogram": {"triposg_talk": 1},
        "clips": [{
            "order": 0, "shot_id": "shot_b001", "beat_id": "b001",
            "engine_id": "triposg_talk", "family": "character_3d",
            "path": str(d), "type": "directory",
            "frame_count": N_FRAMES, "target_frame_count": N_FRAMES,
            "start_s": None, "exists": True,
        }],
    }


def test_golden_straight_alpha_flatten(tmp_path):
    """Half-alpha red frames over a black background flatten to ~half red at
    the center; the alpha-0 region (outside the overlay) stays background
    black; output is silent yuv420p with EXACTLY the budget frame count."""
    if not sc._ffmpeg_bin("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "frames"
    _write_frames(d)
    out = str(tmp_path / "silent.mp4")
    silent, report = sc.assemble_silent_timeline(
        _manifest_with_dir_clip(d), "", out, w=W, h=H, fps=FPS)
    assert silent == out
    assert sc.count_video_frames(out) == N_FRAMES        # exact budget
    assert sc.count_audio_streams(out) == 0              # V-1 silent
    assert sc.probe_video(out).get("pix_fmt") == "yuv420p"
    buf = _decode_rgb_frame(out)
    r, g, b = _px(buf, W // 2, H // 2)
    assert abs(r - 127) <= 18 and g <= 14 and b <= 14, (r, g, b)
    r2, g2, b2 = _px(buf, 4, 4)                          # outside the overlay
    assert r2 <= 12 and g2 <= 12 and b2 <= 12, (r2, g2, b2)


def test_node_entry_composites_directory_manifest(tmp_path):
    """manifest -> composite through the OTR_SilentComposite NODE entry: the
    directory clip assembles to the canonical silent mp4 (the same gates the
    terminal mux consumes -- silence, yuv420p, exact frame budget)."""
    if not sc._ffmpeg_bin("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "frames"
    _write_frames(d)
    out = str(tmp_path / "node_silent.mp4")
    path, report = sc.OTRSilentComposite().composite(
        base_video_path="", canvas_w=W, canvas_h=H, fps=FPS,
        output_path=out,
        clip_manifest_json=json.dumps(_manifest_with_dir_clip(d)))
    assert path == out, report
    assert sc.count_video_frames(out) == N_FRAMES
    assert sc.count_audio_streams(out) == 0


def test_vanished_directory_clip_gap_fills_black(tmp_path):
    """A manifest row whose frame dir vanished gap-fills (black with no
    floor) instead of crashing -- the episode still assembles."""
    if not sc._ffmpeg_bin("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    m = _manifest_with_dir_clip(tmp_path / "gone")
    out = str(tmp_path / "gap.mp4")
    silent, _ = sc.assemble_silent_timeline(m, "", out, w=W, h=H, fps=FPS)
    assert sc.count_video_frames(out) == N_FRAMES
    buf = _decode_rgb_frame(out)
    assert _px(buf, W // 2, H // 2) <= (12, 12, 12)      # black gap-fill


# --------------------------------------------------------------------------- #
# C1 -- the textured-hero 3D PoC per-clip still PLATE background
# --------------------------------------------------------------------------- #
def _green_plate(path, w=W, h=H):
    """A solid GREEN still plate (the generated background behind the mesh)."""
    Image.new("RGB", (w, h), (0, 255, 0)).save(path)
    return str(path)


def _manifest_with_plate(d, plate):
    m = _manifest_with_dir_clip(d)
    m["clips"][0]["bg_still_path"] = str(plate)
    return m


def test_build_clip_manifest_stamps_mesh_stage_plate(tmp_path):
    """A mesh_stage directory clip gets its bg_still_path stamped from the
    beat's scene still (ledger images), and ONLY when the plate file exists.
    A non-mesh engine never gets the field (byte-identical)."""
    d = tmp_path / "frames"
    _write_frames(d)
    plate = _green_plate(tmp_path / "scene_b001.png")
    result = {
        "ledger": {
            "video": {
                "video_revision": 1, "fps": FPS,
                "canonical_canvas": {"w": W, "h": H},
                "shots": [
                    {"shot_id": "shot_b001", "source_line_ids": ["b001"],
                     "engine_id": "mesh_stage", "family": "image_to_video",
                     "target_frame_count": N_FRAMES},
                ]},
            "images": {"images": [
                {"beat_id": "b001", "kind": "scene_beat", "path": str(plate)},
            ]}},
        "clips": {"shot_b001": dict(_dir_clip(d), engine_id="mesh_stage",
                                    family="image_to_video")},
    }
    row = rd.build_clip_manifest(result, episode_id="ep_mesh")["clips"][0]
    assert row["engine_id"] == "mesh_stage"
    assert row["bg_still_path"] == str(plate)
    # Plate file missing -> no bogus path on the manifest channel (LOUD warn).
    result["ledger"]["images"]["images"][0]["path"] = str(tmp_path / "gone.png")
    row2 = rd.build_clip_manifest(result, episode_id="ep_mesh")["clips"][0]
    assert "bg_still_path" not in row2
    # A non-mesh engine never carries the field (eid comes from the clip).
    result["ledger"]["images"]["images"][0]["path"] = str(plate)   # restore
    result["clips"]["shot_b001"] = dict(_dir_clip(d), engine_id="triposg_talk",
                                        family="character_3d")
    row3 = rd.build_clip_manifest(result, episode_id="ep_mesh")["clips"][0]
    assert row3["engine_id"] == "triposg_talk"
    assert "bg_still_path" not in row3


def test_segment_carries_bg_still_path():
    """The planner propagates a clip row's bg_still_path into its segment dict
    (the field is dead-on-arrival without this); other roles carry ''."""
    plate = "C:/plate.png"
    m = {"fps": FPS, "total_target_frames": N_FRAMES, "clips": [{
        "order": 0, "shot_id": "b001", "engine_id": "mesh_stage",
        "path": "C:/frames", "target_frame_count": N_FRAMES,
        "start_s": None, "exists": True, "bg_still_path": plate}]}
    segs, _ = sc.plan_timeline_segments(m, target_total_frames=N_FRAMES, fps=FPS)
    clip_segs = [s for s in segs if s["source"] == "clip"]
    assert clip_segs and clip_segs[0]["bg_still_path"] == plate
    # A plain beat with no plate -> empty string (legacy byte-identical).
    m2 = {"fps": FPS, "total_target_frames": N_FRAMES, "clips": [{
        "order": 0, "shot_id": "b001", "engine_id": "ltx_video",
        "path": "C:/frames", "target_frame_count": N_FRAMES,
        "start_s": None, "exists": True}]}
    segs2, _ = sc.plan_timeline_segments(m2, target_total_frames=N_FRAMES, fps=FPS)
    assert [s for s in segs2 if s["source"] == "clip"][0]["bg_still_path"] == ""


def test_dir_clip_composites_over_still_plate(tmp_path):
    """A directory clip with a per-clip still PLATE composites the half-alpha
    red mesh frames OVER the generated background: the center is red+green
    blend, the corner (outside the fg overlay) shows the PLATE green -- NOT
    black. Proves the still-aware bg branch fills the beat from the plate."""
    if not sc._ffmpeg_bin("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "frames"
    _write_frames(d)
    plate = _green_plate(tmp_path / "plate.png")
    out = str(tmp_path / "plated.mp4")
    silent, _ = sc.assemble_silent_timeline(
        _manifest_with_plate(d, plate), "", out, w=W, h=H, fps=FPS)
    assert sc.count_video_frames(out) == N_FRAMES          # exact budget
    assert sc.count_audio_streams(out) == 0                # V-1 silent
    buf = _decode_rgb_frame(out)
    r, g, b = _px(buf, 4, 4)                                # outside the overlay
    assert g >= 200 and r <= 60 and b <= 60, (r, g, b)     # the PLATE, not black
    rc, gc, bc = _px(buf, W // 2, H // 2)                   # red over green plate
    assert rc >= 90 and gc >= 90, (rc, gc, bc)             # blended, not pure


def test_dir_clip_missing_plate_falls_back_to_black(tmp_path):
    """A bg_still_path pointing at a NONEXISTENT plate does not crash the
    composite -- with no floor it falls back to black (the still branch is
    file-existence guarded)."""
    if not sc._ffmpeg_bin("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "frames"
    _write_frames(d)
    m = _manifest_with_plate(d, tmp_path / "no_such_plate.png")
    out = str(tmp_path / "noplate.mp4")
    silent, _ = sc.assemble_silent_timeline(m, "", out, w=W, h=H, fps=FPS)
    assert sc.count_video_frames(out) == N_FRAMES
    buf = _decode_rgb_frame(out)
    assert _px(buf, 4, 4) <= (12, 12, 12)                  # black gap-fill


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
