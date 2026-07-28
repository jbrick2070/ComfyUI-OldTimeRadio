"""The TERMINAL FRAME: what a CHAIN successor begins on.

Multi-clip coverage chunk 6c. Segment N+1 needs segment N's last frame
SYNCHRONOUSLY -- inside the render loop, not from a post-episode pass -- or the
chain cannot be a chain.

The extractor decodes the whole clip and lets ``-update 1`` overwrite, rather
than seeking with ``-sseof``. The test that matters is the one that proves it
picks the LAST frame and not merely A frame: the fixture paints a different
grey per frame, so "did we get frame N-1" is checkable rather than assumed.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ramp_frames(n, value_step=20):
    """``n`` frames, each a FLAT grey one step brighter than the last."""
    import numpy as np
    return [np.full((64, 64, 3), (i + 1) * value_step, dtype=np.uint8)
            for i in range(n)]


# ---------------------------------------------------------------------------
# The pure command builder
# ---------------------------------------------------------------------------

def test_the_command_updates_ONE_output_rather_than_seeking():
    cmd = wb.ffmpeg_terminal_frame_cmd("in.mp4", "out.png")
    assert "-update" in cmd and cmd[cmd.index("-update") + 1] == "1"
    assert "-an" in cmd, "a terminal frame must never carry audio (V-1)"
    assert cmd[-1] == "out.png"
    # A tail SEEK is exactly what this must not do -- it has nothing to land on
    # in a 9-frame segment.
    assert "-sseof" not in cmd


def test_the_command_honours_an_explicit_ffmpeg_binary():
    cmd = wb.ffmpeg_terminal_frame_cmd("in.mp4", "out.png", ffmpeg="/opt/ffmpeg")
    assert cmd[0] == "/opt/ffmpeg"


# ---------------------------------------------------------------------------
# The real extraction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_it_extracts_the_LAST_frame_not_merely_A_frame():
    """The whole point. Each frame is a different grey; the extracted image
    must be the brightest one."""
    import numpy as np
    from PIL import Image

    frames = _ramp_frames(8)
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "ramp.mp4")
        path, n = wb.encode_frames_to_silent_mp4(frames, clip, 25)
        assert n == 8
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        got = np.asarray(Image.open(out).convert("RGB"))
        # h264 is lossy, so compare to the EXPECTED level with tolerance, and
        # prove it is nearer the last frame than the first.
        mean = float(got.mean())
        assert abs(mean - 160.0) < 12.0, mean          # frame 8 == 8*20
        assert abs(mean - 20.0) > 100.0, "that is the FIRST frame"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_it_works_on_a_SHORT_clip_where_a_tail_seek_would_miss():
    """A 3-frame segment is well under the one-second tail an ``-sseof -1``
    seek assumes exists."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "short.mp4")
        path, _n = wb.encode_frames_to_silent_mp4(_ramp_frames(3, 60), clip, 25)
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        mean = float(np.asarray(Image.open(out).convert("RGB")).mean())
        assert abs(mean - 180.0) < 12.0, mean          # frame 3 == 3*60


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_the_extracted_frame_matches_the_clip_GEOMETRY():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "geom.mp4")
        path, _n = wb.encode_frames_to_silent_mp4(_ramp_frames(4), clip, 25)
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        assert Image.open(out).size == (64, 64)


def test_a_clip_ffmpeg_cannot_read_is_TERMINAL(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "nope.mp4")
        out = os.path.join(tmp, "terminal.png")
        with pytest.raises(wb.GraphExecutionError):
            wb.extract_terminal_frame(missing, out)


def test_a_SILENT_SUCCESS_that_wrote_nothing_is_TERMINAL(monkeypatch):
    """ffmpeg exits 0 for an input it decoded zero frames from.

    A 0-byte file handed on as the next segment's init image is a black frame
    at the cut with a clean exit code in front of it -- which is the failure
    mode this build keeps having to remove, not a new one.
    """
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: cmd)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "never_written.png")
        with pytest.raises(wb.GraphExecutionError, match="no usable image"):
            wb.extract_terminal_frame("whatever.mp4", out)


def test_a_ZERO_BYTE_output_is_TERMINAL(monkeypatch):
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: cmd)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "empty.png")
        open(out, "wb").close()
        with pytest.raises(wb.GraphExecutionError, match="0 bytes"):
            wb.extract_terminal_frame("whatever.mp4", out)


# ---------------------------------------------------------------------------
# The in-tree tmp allocator now reserves any suffix, not just .mp4
# ---------------------------------------------------------------------------

def test_the_allocator_reserves_a_png_in_the_same_tier(monkeypatch, tmp_path):
    from nodes import _otr_paths as _paths
    from nodes._otr_video_engines import _tmp as engine_tmp

    shared = tmp_path / "otr" / "episodes" / "_shared" / "tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    path = engine_tmp.otr_engine_tmp_path("otr_terminal_", ".png")
    assert path.endswith(".png")
    assert str(shared) in path
    assert not os.path.exists(path), "the allocator hands back a FREE path"


def test_the_mp4_allocator_is_unchanged(monkeypatch, tmp_path):
    from nodes import _otr_paths as _paths
    from nodes._otr_video_engines import _tmp as engine_tmp

    shared = tmp_path / "otr" / "episodes" / "_shared" / "tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    path = engine_tmp.otr_engine_tmp_mp4("otr_clip_")
    assert path.endswith(".mp4")
    assert str(shared) in path


# ---------------------------------------------------------------------------
# A5-lite (2026-07-27): the encoder boundary asserts its dtype.
#
# Cut as a LIVE bug -- every producer feeds an exact-size uint8 buffer through
# images_to_uint8, and ffmpeg raises on a short write. The residual this closes
# is a future wider-dtype caller: the rawvideo pipe is 8-bit, so float32 sends
# four times the bytes for the same frames, and the frame_count the encoder
# returns is taken from the ARRAY rather than from what ffmpeg wrote -- so that
# clip would carry a clean receipt describing a length it does not have.
# ---------------------------------------------------------------------------

def test_the_encoder_refuses_a_non_uint8_batch():
    import numpy as np

    frames = np.zeros((4, 8, 8, 3), dtype=np.float32)
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        wb.encode_frames_to_silent_mp4(frames, "unused.mp4", 25)
    message = str(excinfo.value)
    assert "uint8" in message
    assert "float32" in message
    assert "4x the bytes" in message
    assert "images_to_uint8" in message


def test_the_encoder_still_accepts_what_the_real_converter_produces():
    """The refusal must not reject the shipped path. images_to_uint8 is the
    converter every producer goes through, so its OUTPUT is the contract --
    asserted here rather than a hand-made uint8 array, so a change to the
    converter's dtype shows up as a failure instead of a silent divergence."""
    import numpy as np

    torch = pytest.importorskip("torch")
    images = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    converted = wb.images_to_uint8(images)
    assert converted.dtype == np.uint8
    # Shape/dtype are accepted: the encode itself needs ffmpeg, so drive only
    # the guards by pointing at a path ffmpeg would refuse to write.
    if not _HAS_FFMPEG:
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ok.mp4")
        _path, n = wb.encode_frames_to_silent_mp4(converted, out, 25)
        assert n == 2


# ---------------------------------------------------------------------------
# The frame_count asymmetry (2026-07-27): an adapter that ENCODES a clip must
# also PROVE it.
#
# CanonicalClip.frame_count is documented as "the integer timing authority",
# but it is decode-counted truth only for assembled multi-segment beats -- for
# a single-render beat it is whatever the adapter declared. Four of the seven
# adapters that emit an mp4 never ffprobed what they wrote, so their clips
# reached the ledger entirely self-declared. The two derivations agree today
# only because every producer pipes exact bytes, which is a property of the
# current producers rather than a contract anything enforced.
# ---------------------------------------------------------------------------

def test_every_adapter_that_encodes_a_clip_also_proves_it():
    """Derived from the SOURCE, not from a hand-kept list -- which is the whole
    point. The bug report named two adapters (eng_humo, eng_ltx_av) and listed
    eng_ltx_video among those that already probed; sweeping every encoder found
    eng_ltx_video did NOT, on either of its two recipe paths, and that
    eng_still_parallax was missing from the report altogether. The next adapter
    that encodes without proving fails HERE, by name."""
    import pathlib

    engines = (pathlib.Path(__file__).resolve().parents[1]
               / "nodes" / "_otr_video_engines")
    encoders, provers = set(), set()
    # The sweep is over EVERY module in the package, and counts BOTH ways of
    # writing a clip. Its first version globbed eng_*.py and looked only for
    # encode_frames_to_silent_mp4, which made cheap_families.py structurally
    # invisible: it is not named eng_*, and it builds its mp4 from an ffmpeg
    # arg list instead of the encoder. That hid FOUR registered engines
    # (still_motion / still_pan / still_flat / still_word) behind a test that
    # kept reporting PASS -- including the terminus of the documented
    # humo -> humo_1.7B -> still_motion degrade chain. Found by the post-push
    # QA fan-out; a test that is confidently blind is worse than no test.
    for path in sorted(engines.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if path.name in ("wrapper_bridge.py", "wan_shared.py"):
            continue                      # the helpers themselves, not clients
        if ("encode_frames_to_silent_mp4(" in src
                or "run_ffmpeg(" in src):
            encoders.add(path.name)
        if "validate_silent_clip_contract(" in src:
            provers.add(path.name)

    assert encoders, "found no encoders at all -- the sweep is broken"
    assert "cheap_families.py" in encoders, (
        "the sweep no longer sees cheap_families.py, which is exactly the "
        "blind spot this widening exists to close")
    assert encoders - provers == set(), sorted(encoders - provers)


def test_the_proof_runs_AFTER_the_encode_in_every_adapter():
    """Order matters: ffprobing before the file is written would raise on a
    path that does not exist yet, which is a different failure wearing the
    same name. Pinned per encode site, not per file.

    The pairing is asserted in BOTH directions, and the second direction was
    added because the mutation round caught the first version of this test
    being decorative: checking only that some proof FOLLOWS each encode stays
    green when a proof is inserted BEFORE one, which is the exact ordering
    defect the test is named for."""
    import pathlib
    import re

    engines = (pathlib.Path(__file__).resolve().parents[1]
               / "nodes" / "_otr_video_engines")
    checked = 0
    for path in sorted(engines.glob("eng_*.py")):
        src = path.read_text(encoding="utf-8")
        encodes = [m.start() for m
                   in re.finditer(r"encode_frames_to_silent_mp4\(", src)]
        proofs = [m.start() for m
                  in re.finditer(r"validate_silent_clip_contract\(", src)]
        if not encodes:
            continue
        assert len(proofs) >= len(encodes), (
            "%s: %d encode site(s) but only %d proof(s)"
            % (path.name, len(encodes), len(proofs)))
        for encode_at in encodes:
            assert any(p > encode_at for p in proofs), (
                "%s: an encode at offset %d has no proof after it"
                % (path.name, encode_at))
            checked += 1
        for proof_at in proofs:
            assert any(e < proof_at for e in encodes), (
                "%s: a proof at offset %d runs before any encode -- it would "
                "ffprobe a path that does not exist yet"
                % (path.name, proof_at))
    assert checked >= 7, checked
