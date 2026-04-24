"""
test_otr_video_concat.py  --  MIT-original video-concat node regression
========================================================================

Validates ``nodes.otr_video_concat`` without requiring ffmpeg or
external deps.  All tests exercise either the pure helpers (argv
builders, filelist writer, path parser) or the stub mode
(byte-identical passthrough copy of the first input).

Covers:
    * Module imports cleanly (no torch / ffmpeg imports at load)
    * parse_input_paths strips comments + blanks, preserves order
    * write_concat_filelist produces valid ffmpeg concat-demuxer format
    * build_concat_argv builds correct argv with and without reencode
    * build_concat_argv always emits audio-passthrough mapping
    * concat_videos stub mode byte-identical copies first input
    * Stub autodetect: force_stub, env var, ffmpeg-missing
    * Empty input list raises ValueError
    * Missing input file raises FileNotFoundError
    * Node class INPUT_TYPES has the required widgets
    * Node class RETURN_TYPES is ("STRING",)
    * Stub-mode output path equals requested output path
    * Filelist uses forward slashes (Windows-safe ffmpeg convention)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------------------------
# Import-time smoke tests
# ------------------------------------------------------------------


def test_module_imports_cleanly():
    """Module must import without torch, diffusers, or ffmpeg."""
    mod = importlib.import_module("nodes.otr_video_concat")
    assert hasattr(mod, "OTRVideoConcat")
    assert hasattr(mod, "concat_videos")
    assert hasattr(mod, "parse_input_paths")
    assert hasattr(mod, "write_concat_filelist")
    assert hasattr(mod, "build_concat_argv")
    assert hasattr(mod, "find_ffmpeg")


def test_node_class_mappings_present():
    mod = importlib.import_module("nodes.otr_video_concat")
    assert "OTR_VideoConcat" in mod.NODE_CLASS_MAPPINGS
    assert mod.NODE_CLASS_MAPPINGS["OTR_VideoConcat"] is mod.OTRVideoConcat


# ------------------------------------------------------------------
# parse_input_paths
# ------------------------------------------------------------------


def test_parse_input_paths_basic():
    from nodes.otr_video_concat import parse_input_paths

    raw = "/a/b/clip_00.mp4\n/a/b/clip_01.mp4\n/a/b/clip_02.mp4"
    result = parse_input_paths(raw)
    assert len(result) == 3
    assert result[0] == Path("/a/b/clip_00.mp4")
    assert result[2] == Path("/a/b/clip_02.mp4")


def test_parse_input_paths_strips_comments_and_blanks():
    from nodes.otr_video_concat import parse_input_paths

    raw = (
        "# heading comment\n"
        "\n"
        "/a/b/clip_00.mp4\n"
        "  \n"
        "# another comment\n"
        "/a/b/clip_01.mp4\n"
    )
    result = parse_input_paths(raw)
    assert result == [Path("/a/b/clip_00.mp4"), Path("/a/b/clip_01.mp4")]


def test_parse_input_paths_preserves_order():
    from nodes.otr_video_concat import parse_input_paths

    raw = "\n".join(f"/p/clip_{i:02d}.mp4" for i in range(10))
    result = parse_input_paths(raw)
    # Use Path.name for OS-agnostic filename extraction -- str(Path)
    # normalises to backslashes on Windows, which breaks split("/").
    assert [p.name for p in result] == [
        f"clip_{i:02d}.mp4" for i in range(10)
    ]


def test_parse_input_paths_empty_string():
    from nodes.otr_video_concat import parse_input_paths

    assert parse_input_paths("") == []
    assert parse_input_paths("\n\n\n") == []
    assert parse_input_paths("# all comments\n# only comments") == []


def test_parse_input_paths_trims_whitespace():
    from nodes.otr_video_concat import parse_input_paths

    raw = "  /a/b/clip.mp4  \n\t/c/d/other.mp4\t"
    result = parse_input_paths(raw)
    assert result[0] == Path("/a/b/clip.mp4")
    assert result[1] == Path("/c/d/other.mp4")


# ------------------------------------------------------------------
# write_concat_filelist
# ------------------------------------------------------------------


def test_write_concat_filelist_basic(tmp_path):
    from nodes.otr_video_concat import write_concat_filelist

    list_path = tmp_path / "files.txt"
    write_concat_filelist(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"],
        list_path,
    )
    content = list_path.read_text(encoding="utf-8")
    assert "file '" in content
    assert "a.mp4'" in content
    assert "b.mp4'" in content
    # One line per file plus trailing newline
    lines = [ln for ln in content.splitlines() if ln]
    assert len(lines) == 2


def test_write_concat_filelist_forward_slashes(tmp_path):
    """ffmpeg concat demuxer prefers forward slashes even on Windows."""
    from nodes.otr_video_concat import write_concat_filelist

    list_path = tmp_path / "files.txt"
    # Simulate a Windows path
    write_concat_filelist(
        [Path(r"C:\videos\a.mp4"), Path(r"D:\videos\b.mp4")],
        list_path,
    )
    content = list_path.read_text(encoding="utf-8")
    assert "\\" not in content
    assert "C:/videos/a.mp4" in content
    assert "D:/videos/b.mp4" in content


def test_write_concat_filelist_escapes_single_quote(tmp_path):
    """A path with a literal single quote must be escaped."""
    from nodes.otr_video_concat import write_concat_filelist

    list_path = tmp_path / "files.txt"
    # Create a path with a single quote in it
    write_concat_filelist(
        [Path("/tmp/Jeffrey's Clip.mp4")],
        list_path,
    )
    content = list_path.read_text(encoding="utf-8")
    # ffmpeg concat escaping: ' -> '\''
    assert "Jeffrey'\\''s Clip.mp4" in content


# ------------------------------------------------------------------
# build_concat_argv
# ------------------------------------------------------------------


def test_build_concat_argv_no_reencode(tmp_path):
    from nodes.otr_video_concat import build_concat_argv

    argv = build_concat_argv(
        "/usr/bin/ffmpeg",
        tmp_path / "files.txt",
        tmp_path / "out.mp4",
        reencode=False,
    )
    assert argv[0] == "/usr/bin/ffmpeg"
    assert "-f" in argv
    assert "concat" in argv
    assert "-safe" in argv
    assert "-c" in argv and "copy" in argv
    assert "libx264" not in argv
    # Audio map safety
    assert "-map" in argv
    assert "0:a?" in argv


def test_build_concat_argv_reencode(tmp_path):
    from nodes.otr_video_concat import build_concat_argv

    argv = build_concat_argv(
        "/usr/bin/ffmpeg",
        tmp_path / "files.txt",
        tmp_path / "out.mp4",
        reencode=True,
    )
    assert "libx264" in argv
    assert "yuv420p" in argv
    assert "aac" in argv
    # Should still include audio map safety
    assert "-map" in argv
    assert "0:a?" in argv


def test_build_concat_argv_output_path_last(tmp_path):
    from nodes.otr_video_concat import build_concat_argv

    out_path = tmp_path / "final.mp4"
    argv = build_concat_argv(
        "/usr/bin/ffmpeg",
        tmp_path / "files.txt",
        out_path,
        reencode=False,
    )
    assert argv[-1] == str(out_path)


def test_build_concat_argv_overwrite_flag(tmp_path):
    """-y must be present so ffmpeg overwrites existing output."""
    from nodes.otr_video_concat import build_concat_argv

    argv = build_concat_argv(
        "/usr/bin/ffmpeg",
        tmp_path / "files.txt",
        tmp_path / "out.mp4",
        reencode=False,
    )
    assert "-y" in argv


# ------------------------------------------------------------------
# concat_videos stub mode
# ------------------------------------------------------------------


def _make_payload_file(path: Path, body: bytes) -> None:
    path.write_bytes(body)


def test_concat_videos_stub_byte_identical(tmp_path):
    """Stub mode copies the FIRST input byte-for-byte to output."""
    from nodes.otr_video_concat import concat_videos

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    c = tmp_path / "c.mp4"
    _make_payload_file(a, b"AAAA" + os.urandom(128))
    _make_payload_file(b, b"BBBB" + os.urandom(128))
    _make_payload_file(c, b"CCCC" + os.urandom(128))
    out = tmp_path / "out.mp4"

    result = concat_videos(
        [a, b, c],
        out,
        force_stub=True,
    )
    assert result["mode"] == "stub"
    assert result["stub_reason"] == "force_stub=True"
    assert result["output"] == str(out)
    assert result["clip_count"] == 3
    assert result["returncode"] == 0
    # Byte-identical to FIRST input
    assert out.read_bytes() == a.read_bytes()
    assert out.read_bytes() != b.read_bytes()


def test_concat_videos_stub_via_env(tmp_path, monkeypatch):
    from nodes.otr_video_concat import concat_videos

    a = tmp_path / "a.mp4"
    _make_payload_file(a, b"env-stub-payload")
    out = tmp_path / "out.mp4"

    monkeypatch.setenv("OTR_VIDEO_CONCAT_STUB", "1")
    result = concat_videos([a], out)
    assert result["mode"] == "stub"
    assert "env" in result["stub_reason"]
    assert out.read_bytes() == a.read_bytes()


def test_concat_videos_stub_via_missing_ffmpeg(tmp_path, monkeypatch):
    """When find_ffmpeg returns None, we fall into stub mode."""
    from nodes import otr_video_concat

    a = tmp_path / "a.mp4"
    _make_payload_file(a, b"missing-ffmpeg-payload")
    out = tmp_path / "out.mp4"

    # Clear the env just in case
    monkeypatch.delenv("OTR_VIDEO_CONCAT_STUB", raising=False)
    # Force find_ffmpeg to report missing
    monkeypatch.setattr(otr_video_concat, "find_ffmpeg", lambda: None)

    result = otr_video_concat.concat_videos([a], out)
    assert result["mode"] == "stub"
    assert "not found" in result["stub_reason"]


def test_concat_videos_empty_input_raises(tmp_path):
    from nodes.otr_video_concat import concat_videos

    with pytest.raises(ValueError, match="empty"):
        concat_videos([], tmp_path / "out.mp4", force_stub=True)


def test_concat_videos_missing_input_raises(tmp_path):
    from nodes.otr_video_concat import concat_videos

    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        concat_videos([missing], tmp_path / "out.mp4", force_stub=True)


def test_concat_videos_creates_output_parent(tmp_path):
    """If output_path's parent dir doesn't exist, we create it."""
    from nodes.otr_video_concat import concat_videos

    a = tmp_path / "a.mp4"
    _make_payload_file(a, b"parent-dir-test")
    out = tmp_path / "nested" / "subdir" / "concat.mp4"
    assert not out.parent.exists()

    result = concat_videos([a], out, force_stub=True)
    assert result["mode"] == "stub"
    assert out.parent.exists()
    assert out.exists()


# ------------------------------------------------------------------
# Node class surface-area
# ------------------------------------------------------------------


def test_input_types_schema():
    from nodes.otr_video_concat import OTRVideoConcat

    schema = OTRVideoConcat.INPUT_TYPES()
    assert "required" in schema
    assert "input_paths" in schema["required"]
    assert "output_path" in schema["required"]
    assert "reencode" in schema["required"]
    # Type tuples
    assert schema["required"]["input_paths"][0] == "STRING"
    assert schema["required"]["output_path"][0] == "STRING"
    assert schema["required"]["reencode"][0] == "BOOLEAN"


def test_input_types_multiline_widget():
    """input_paths must be marked multiline for the path list UX."""
    from nodes.otr_video_concat import OTRVideoConcat

    schema = OTRVideoConcat.INPUT_TYPES()
    input_cfg = schema["required"]["input_paths"][1]
    assert input_cfg.get("multiline") is True


def test_return_types_single_string():
    from nodes.otr_video_concat import OTRVideoConcat

    assert OTRVideoConcat.RETURN_TYPES == ("STRING",)
    assert OTRVideoConcat.RETURN_NAMES == ("video_path",)


def test_function_and_category_set():
    from nodes.otr_video_concat import OTRVideoConcat

    assert OTRVideoConcat.FUNCTION == "concat"
    assert OTRVideoConcat.CATEGORY.startswith("OldTimeRadio")


def test_is_output_node():
    """ComfyUI needs OUTPUT_NODE=True to execute a terminal file-writing
    node when its STRING output is not consumed by anything downstream."""
    from nodes.otr_video_concat import OTRVideoConcat

    assert getattr(OTRVideoConcat, "OUTPUT_NODE", False) is True


def test_concat_method_stub_end_to_end(tmp_path, monkeypatch):
    """Smoke test of the node's concat() entry point in stub mode."""
    from nodes.otr_video_concat import OTRVideoConcat

    a = tmp_path / "shot_00.mp4"
    b = tmp_path / "shot_01.mp4"
    _make_payload_file(a, b"first")
    _make_payload_file(b, b"second")
    out = tmp_path / "episode.mp4"

    monkeypatch.setenv("OTR_VIDEO_CONCAT_STUB", "1")
    node = OTRVideoConcat()
    result = node.concat(
        input_paths=f"{a}\n{b}",
        output_path=str(out),
        reencode=False,
        crossfade_ms=0,
    )
    assert result == (str(out),)
    assert out.read_bytes() == a.read_bytes()


def test_concat_method_raises_on_empty_input(tmp_path, monkeypatch):
    from nodes.otr_video_concat import OTRVideoConcat

    monkeypatch.setenv("OTR_VIDEO_CONCAT_STUB", "1")
    node = OTRVideoConcat()
    with pytest.raises(ValueError, match="no input paths"):
        node.concat(
            input_paths="# only comments\n# nothing real",
            output_path=str(tmp_path / "out.mp4"),
            reencode=False,
        )


def test_concat_method_default_output_path(tmp_path, monkeypatch):
    """If output_path is empty, we default to first-input-dir / concat.mp4."""
    from nodes.otr_video_concat import OTRVideoConcat

    a = tmp_path / "clip.mp4"
    _make_payload_file(a, b"default-output-test")

    monkeypatch.setenv("OTR_VIDEO_CONCAT_STUB", "1")
    node = OTRVideoConcat()
    result = node.concat(
        input_paths=str(a),
        output_path="",
        reencode=False,
    )
    # Result should point at tmp_path / "concat.mp4"
    expected = tmp_path / "concat.mp4"
    assert result == (str(expected),)
    assert expected.exists()
