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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
