"""BUG-LOCAL-030 Phase B regression: post-RTXUpscale procgen blend node."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


@pytest.fixture
def node():
    from nodes.otr_post_upscale_procgen_blend import PostUpscaleProcgenBlend

    return PostUpscaleProcgenBlend()


def _capture_run():
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    return captured, _fake_run


def test_input_types_shape(node):
    spec = node.INPUT_TYPES()
    req = spec["required"]
    opt = spec["optional"]
    assert "source_mp4_path" in req
    assert "procgen_mp4_path" in req
    assert "blend_mode" in opt
    assert "blend_opacity" in opt
    assert "bypass" in opt
    assert "out_suffix" in opt


def test_returns_two_strings(node):
    assert node.RETURN_TYPES == ("STRING", "STRING")
    assert node.RETURN_NAMES == ("final_mp4_path", "report")
    assert node.OUTPUT_NODE is True


def test_default_blend_opacity_is_05(node):
    spec = node.INPUT_TYPES()
    assert spec["optional"]["blend_opacity"][1]["default"] == 0.5


def test_blend_cmd_uses_audio_passthrough(node, tmp_path):
    src = tmp_path / "rtx_upscale.mp4"
    src.write_bytes(b"src")
    pgn = tmp_path / "procgen_1080p.mp4"
    pgn.write_bytes(b"pgn")
    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run", side_effect=fake_run):
        out, report = node.blend(
            source_mp4_path=str(src),
            procgen_mp4_path=str(pgn),
            blend_mode="lighten",
            blend_opacity=0.5,
        )
    cmd = captured["cmd"]
    # C7 audio passthrough
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy", (
        "audio MUST be -c:a copy (zero re-encodes for C7 byte-identity)"
    )
    # Both inputs present
    assert str(src) in cmd
    assert str(pgn) in cmd
    # Filter chain references both
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:v]" in fc and "[1:v]" in fc
    assert "blend=all_mode=lighten" in fc
    assert "all_opacity=0.500" in fc


def test_blend_cmd_uses_shortest(node, tmp_path):
    """-shortest prevents the procgen looping forever past source EOF."""
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"x")
    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run", side_effect=fake_run):
        node.blend(source_mp4_path=str(src), procgen_mp4_path=str(pgn))
    assert "-shortest" in captured["cmd"]


def test_bypass_mode_copies_source_to_output(node, tmp_path):
    """bypass=True must skip ffmpeg entirely and copy source -> out."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"original-content")
    pgn = tmp_path / "pgn.mp4"
    pgn.write_bytes(b"pgn")
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run") as mock_run:
        out, report = node.blend(
            source_mp4_path=str(src), procgen_mp4_path=str(pgn),
            bypass=True,
        )
        mock_run.assert_not_called()
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"original-content"
    assert "bypass" in report.lower()


def test_missing_source_returns_empty_path(node, tmp_path):
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"x")
    out, report = node.blend(
        source_mp4_path=str(tmp_path / "does_not_exist.mp4"),
        procgen_mp4_path=str(pgn),
    )
    assert out == ""
    assert "missing" in report.lower()


def test_missing_procgen_falls_back_to_source_copy(node, tmp_path):
    """If procgen mp4 isn't on disk, gracefully copy source -> output
    so the pipeline still produces a deliverable. Logs a warning."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"src-content")
    out, report = node.blend(
        source_mp4_path=str(src),
        procgen_mp4_path=str(tmp_path / "missing_procgen.mp4"),
    )
    assert out != ""
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"src-content"
    assert "skipped blend" in report.lower() or "procgen" in report.lower()


def test_ffmpeg_failure_falls_back_to_source_copy(node, tmp_path):
    """If ffmpeg blend itself fails, copy source -> output as fallback
    so the pipeline doesn't drop the deliverable."""
    src = tmp_path / "src.mp4"; src.write_bytes(b"src-content")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"pgn")
    import subprocess as _sub
    from nodes import otr_post_upscale_procgen_blend as M

    def _fail_run(cmd, **kwargs):
        raise _sub.CalledProcessError(returncode=1, cmd=cmd, stderr=b"ffmpeg boom")

    with mock.patch.object(M.subprocess, "run", side_effect=_fail_run):
        out, report = node.blend(source_mp4_path=str(src), procgen_mp4_path=str(pgn))
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"src-content"
    assert "ffmpeg" in report.lower() or "failed" in report.lower()


def test_node_registered_in_init():
    """OTR_PostUpscaleProcgenBlend must be in NODE_CLASS_MAPPINGS."""
    import importlib
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
        assert "OTR_PostUpscaleProcgenBlend" in pkg.NODE_CLASS_MAPPINGS, (
            "node not registered in __init__.py"
        )
    finally:
        sys.path.pop(0)


def test_signal_lost_video_resolution_default_is_1080p():
    """BUG-030 Phase B: procgen renders at native 1920x1080 by default."""
    src = (_REPO_ROOT / "nodes" / "video_engine.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"resolution":\s*\(\[([^\]]+)\],\s*\{\s*"default":\s*"([^"]+)"', src)
    assert m is not None, "could not locate resolution widget in video_engine.py"
    choices, default = m.group(1), m.group(2)
    assert default == "1920x1080", (
        f"expected resolution default=1920x1080 (BUG-030 Phase B); got {default}"
    )
    assert "1920x1080" in choices and "832x480" in choices, (
        "both 1920x1080 (new default) and 832x480 (legacy) must remain in choices"
    )
