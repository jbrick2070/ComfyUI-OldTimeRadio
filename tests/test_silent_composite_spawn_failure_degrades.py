"""A failed ffmpeg/ffprobe SPAWN must not discard a finished episode.

`OTRSilentComposite.composite()` reaches the video toolchain through
`_run` -> `otr_proc.run` -> `subprocess.run`. That raises **OSError**
(FileNotFoundError when the binary has moved, PermissionError when a scanner
holds it, and the transient spawn failures Windows produces under load) -- not
ValueError. The node caught only ValueError, so an OSError left the node.

That is the most expensive place in the pipeline to die. By the time the silent
composite runs, the script, the cast, every voice, the audio master and every
rendered clip are already done. Losing the episode there throws all of it away and
publishes nothing -- the operator's bar is "as long as it doesn't crash when it's
not supposed to", and this is that crash.

The remedy is not a new policy: it is the SAME degrade the node already performs one
line up for ValueError -- return the error tuple the caller knows how to read.

These tests pin the degrade AND the two places it must not reach.
"""
from __future__ import annotations

import pytest

import nodes.otr_silent_composite as SC
from nodes.otr_silent_composite import OTRSilentComposite
from nodes._otr_shared.proc import ExecutableNotAllowed


class _StubEngine:
    name = "stub"

    def load(self, *a, **k):
        return None

    def unload(self, *a, **k):
        return None


def _composite(tmp_path, monkeypatch, **over):
    """Drive the real node on its single-base path.

    The upscale-engine gate is stubbed the same way
    tests/test_upscale_composite_single_base_no_engine_load.py does it -- this
    test is about the SPAWN failure underneath, not about engine selection.
    """
    monkeypatch.setattr(SC, "_get_upscale_engine", lambda name: _StubEngine())
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    kw = dict(
        base_video_path=str(src),
        canvas_w=1920, canvas_h=1080, fps=25,
        ffmpeg="ffmpeg",
        output_path=str(tmp_path / "out.mp4"),
        gate_in="",
        clip_manifest_json="{}",   # no "clips" key -> single-base path
        upscale_engine="spandrel_esrgan",
        upscale_device="cpu",
    )
    kw.update(over)
    return OTRSilentComposite().composite(**kw)


@pytest.mark.parametrize("boom", [
    FileNotFoundError(2, "The system cannot find the file specified"),
    PermissionError(13, "Permission denied"),
    OSError(8, "Exec format error"),
])
def test_a_spawn_failure_returns_an_error_instead_of_killing_the_episode(
        tmp_path, monkeypatch, boom):
    """THE DEFECT: these all escaped the node's ValueError-only catch."""
    def _raises(src, out, **kw):
        raise boom

    monkeypatch.setattr(SC, "normalize_to_silent_canonical", _raises)

    silent, report = _composite(tmp_path, monkeypatch)

    assert silent == "", "a failed composite must not claim an output path"
    assert report.startswith("error:"), report


def test_a_refused_executable_is_NOT_swallowed(tmp_path, monkeypatch):
    """`otr_proc._check` raises ExecutableNotAllowed for a binary this pack
    refuses to run. That is a SECURITY contract and must stay fatal.

    It is a RuntimeError, not an OSError, so the new clause cannot absorb it --
    this test is what keeps that true if anyone ever widens the catch.
    """
    def _refused(src, out, **kw):
        raise ExecutableNotAllowed("'evil.exe' is not an executable this pack runs")

    monkeypatch.setattr(SC, "normalize_to_silent_canonical", _refused)

    with pytest.raises(ExecutableNotAllowed):
        _composite(tmp_path, monkeypatch)


def test_a_value_error_still_degrades_exactly_as_before(tmp_path, monkeypatch):
    """The pre-existing clause is untouched."""
    def _bad(src, out, **kw):
        raise ValueError("canvas is not even")

    monkeypatch.setattr(SC, "normalize_to_silent_canonical", _bad)

    silent, report = _composite(tmp_path, monkeypatch)
    assert silent == ""
    assert report == "error: canvas is not even"


def test_the_happy_path_is_untouched(tmp_path, monkeypatch):
    """Exception-only: a run that spawns successfully is unchanged."""
    out_seen = {}

    def _ok(src, out, **kw):
        out_seen["out"] = out
        return (out, ["stubbed"])

    monkeypatch.setattr(SC, "normalize_to_silent_canonical", _ok)

    silent, report = _composite(tmp_path, monkeypatch)
    assert silent == out_seen["out"] and silent != ""
    assert not str(report).startswith("error:")
