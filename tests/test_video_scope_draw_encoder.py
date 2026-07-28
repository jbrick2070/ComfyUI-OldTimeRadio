"""THE SECOND CLIP ENCODER -- ``_otr_shared/scope_draw.py::encode_silent_mp4``.

Filed 2026-07-28 by the pre-push QA fan-out and closed here. Four live engines
(``viz_green`` / ``viz_camera`` / ``viz_mxc_mandala`` / ``viz_mxc_cpu``) write
every clip they emit through this function, and it was a hand-rolled Popen
wrapper with no proof of any kind. The M7 roster gate could not see them
either: it grepped for two literal call spellings and this is a third.

What this file pins is what the encoder now REFUSES. Each refusal is asserted
on its MESSAGE, not merely its type -- an exception type asserted without its
message is one of this build's two most reliable blind spots, because the line
below usually raises the same type incidentally.

UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from nodes._otr_shared import scope_draw as sd

_HAS_FFMPEG = (shutil.which("ffmpeg") is not None
               and shutil.which("ffprobe") is not None)
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg")


def _frames(n, w=160, h=96, value=40):
    for i in range(n):
        yield np.full((h, w, 3), (i + 1) * value % 256, dtype=np.uint8)


def _out(tmp, name="clip.mp4"):
    return os.path.join(tmp, name)


# ---------------------------------------------------------------------------
# THE HEADLINE: `total` was a DEAD PARAMETER.
# ---------------------------------------------------------------------------

def test_a_generator_that_stops_SHORT_is_refused_by_name():
    """``total`` was accepted and never read, so a generator that ran out
    early wrote a short clip and the engine stamped the REQUESTED number as
    CanonicalClip.frame_count -- the integer timing authority the composite
    positions the beat by. The beat would then drift against its own audio
    behind a clean exit code."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(_frames(3), 8, out, 160, 96, 25, "ffmpeg")
    message = str(excinfo.value)
    assert "declared 8 frame(s)" in message
    assert "produced 3" in message
    assert "CanonicalClip.frame_count" in message
    assert "NO FALLBACK" in message


def test_a_generator_that_overruns_is_ALSO_refused():
    """Asserted in BOTH directions. A refusal that only catches SHORT clips
    stays green for a beat carrying MORE frames than it declared, which drifts
    exactly as badly -- the same one-sided-comparison defect the mutation
    round found in the first encoder's count proof on 2026-07-28."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(_frames(6), 4, out, 160, 96, 25, "ffmpeg")
    message = str(excinfo.value)
    assert "declared 4 frame(s)" in message
    assert "produced 6" in message


def test_the_exact_count_is_accepted_and_writes_a_real_clip():
    """The CONTROL. A refusal that also rejected the shipped path would be
    caught here rather than in a render."""
    from nodes._otr_video_engines.wan_shared import (
        ffprobe_clip_fields, validate_silent_clip_contract)

    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        assert sd.encode_silent_mp4(_frames(5), 5, out, 160, 96, 25,
                                    "ffmpeg") == out
        fields = ffprobe_clip_fields(out)
        # The clip this encoder writes satisfies the SAME silent-clip contract
        # the other encoder's clips do -- asserted through the shared
        # authority rather than restated here, so the two cannot drift.
        validate_silent_clip_contract(fields, 25)
        assert fields["nb_frames"] == 5
        assert (fields["width"], fields["height"]) == (160, 96)


# ---------------------------------------------------------------------------
# THE STRIDE: the declared size and the piped bytes are ONE derivation now.
# ---------------------------------------------------------------------------

def test_a_caller_whose_dims_disagree_with_the_frames_is_refused():
    """The rawvideo ``-s`` used to be built from the caller's w/h while the
    pipe carried whatever the generator painted. When those disagree ffmpeg
    slices the byte stream on the wrong boundaries and writes a SKEWED clip
    with a clean exit code -- the same defect class filed against
    ffmpeg_silent_mp4_cmd's even_dim() on 2026-07-28."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            # frames are 160x96; the caller claims 320x180.
            sd.encode_silent_mp4(_frames(3), 3, out, 320, 180, 25, "ffmpeg")
    message = str(excinfo.value)
    assert "declared 320x180" in message
    assert "first frame is 160x96" in message
    assert "skewed" in message


def test_ONE_dimension_disagreeing_is_enough_to_be_refused():
    """The case above differs in BOTH dimensions, so it stays green against a
    check narrowed to the width alone. Each half is exercised on its own."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError) as wide:
            sd.encode_silent_mp4(_frames(2), 2, _out(tmp, "w.mp4"),
                                 176, 96, 25, "ffmpeg")
        with pytest.raises(RuntimeError) as tall:
            sd.encode_silent_mp4(_frames(2), 2, _out(tmp, "h.mp4"),
                                 160, 90, 25, "ffmpeg")
    assert "declared 176x96" in str(wide.value)
    assert "declared 160x90" in str(tall.value)


def test_a_frame_that_CHANGES_SHAPE_midway_is_refused():
    """One odd frame desynchronises the pipe for every frame after it, and the
    clip still exits 0."""
    def _ragged():
        yield np.zeros((96, 160, 3), dtype=np.uint8)
        yield np.zeros((96, 160, 3), dtype=np.uint8)
        yield np.zeros((90, 160, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(_ragged(), 3, out, 160, 96, 25, "ffmpeg")
    message = str(excinfo.value)
    assert "frame 2" in message, message
    assert "desynchronises" in message


def test_a_non_uint8_batch_is_refused_naming_the_multiplier():
    def _floaty():
        yield np.zeros((96, 160, 3), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(_floaty(), 1, out, 160, 96, 25, "ffmpeg")
    message = str(excinfo.value)
    assert "uint8" in message and "float32" in message
    assert "4x the bytes" in message


def test_ZERO_frames_is_refused_before_ffmpeg_is_started():
    """ffmpeg accepts an empty pipe and exits 0, writing a container with no
    video stream in it -- which the count proof would then report in the
    language of a missing stream rather than of an empty ask."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _out(tmp)
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(iter(()), 0, out, 160, 96, 25, "ffmpeg")
        assert not os.path.exists(out), "ffmpeg should never have been started"
    message = str(excinfo.value)
    assert "ZERO frames" in message
    assert "planning bug upstream" in message


# ---------------------------------------------------------------------------
# CODEC SELECTION: nvenc has a floor, and it used to be selected below it.
# ---------------------------------------------------------------------------

def _captured_codec(monkeypatch, w, h, has_nvenc=True):
    """Which encoder the command names, without running it."""
    seen = {}
    real_popen = sd.subprocess.Popen

    class _Stop(RuntimeError):
        pass

    def _fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        raise _Stop("captured")

    monkeypatch.setattr(sd, "_has_nvenc", lambda _fb: has_nvenc)
    monkeypatch.setattr(sd.subprocess, "Popen", _fake_popen)
    try:
        sd.encode_silent_mp4(_frames(1, w=w, h=h), 1, "unused.mp4",
                             w, h, 25, "ffmpeg")
    except _Stop:
        pass
    finally:
        monkeypatch.setattr(sd.subprocess, "Popen", real_popen)
    cmd = seen["cmd"]
    return cmd[cmd.index("-c:v") + 1]


def test_a_canvas_BELOW_the_nvenc_floor_selects_libx264(monkeypatch):
    """h264_nvenc refuses a canvas under 145x49 with an error naming four
    parameters and not the one that is wrong, so selecting it purely on "the
    box has it" made a small-canvas beat die on a machine WITH a GPU and
    succeed on one without. MEASURED on this box 2026-07-28: 144x48 refused,
    146x50 accepted.

    The literals are asserted, not sd._NVENC_MIN_W/_H -- a test that asserts
    against the CONSTANT the code uses agrees with any mutation of it."""
    assert _captured_codec(monkeypatch, 144, 48) == "libx264"
    assert _captured_codec(monkeypatch, 96, 64) == "libx264"


def test_a_canvas_AT_OR_ABOVE_the_floor_still_selects_nvenc(monkeypatch):
    """The CONTROL. A floor set too high would silently retire GPU encoding
    for every shipped canvas, and nothing else in this file would notice."""
    # AT the floor, not merely above it: the comparison is `>=`, and nothing
    # else in this suite would notice it becoming `>`. 145x49 is the smallest
    # size measured to ENCODE on this box, so refusing it would retire GPU
    # encoding for a canvas that works.
    assert _captured_codec(monkeypatch, 145, 49) == "h264_nvenc"
    assert _captured_codec(monkeypatch, 146, 50) == "h264_nvenc"
    assert _captured_codec(monkeypatch, 832, 480) == "h264_nvenc"


def test_a_box_with_no_nvenc_selects_libx264_at_every_size(monkeypatch):
    assert _captured_codec(monkeypatch, 832, 480, has_nvenc=False) == "libx264"


def test_the_floor_is_asserted_against_the_documented_literal():
    """Pinned so a later edit of the constant is a visible decision rather
    than a silent one."""
    assert (sd._NVENC_MIN_W, sd._NVENC_MIN_H) == (145, 49)


def test_the_multiplier_is_COMPUTED_from_the_dtype_not_hardcoded():
    """float32 alone leaves `"4x the bytes"` satisfiable by a literal."""
    def _wide():
        yield np.zeros((96, 160, 3), dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError) as excinfo:
            sd.encode_silent_mp4(_wide(), 1, _out(tmp), 160, 96, 25, "ffmpeg")
    assert "8x the bytes" in str(excinfo.value)


# ---------------------------------------------------------------------------
# THE ENGINES READ THE PROOF, not their own loop bound.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_name", ["viz_green", "viz_camera",
                                         "viz_mxc_mandala", "viz_mxc_cpu"])
def test_the_engine_stamps_the_PROVEN_count_not_the_declared_one(
        monkeypatch, engine_name):
    """The four viz engines used to return their pre-computed loop bound as
    ``frame_count``. They now return what ``proven_frame_count`` gives back.

    On a green render those two numbers are always equal -- the proof RAISES
    on a disagreement rather than correcting one -- so no fixture can tell
    them apart, and the mutation round duly found ``proven`` -> ``total``
    surviving in all four engines. The proof is stubbed here to return a
    value the engine could not have computed, which is the only way to ask
    whether the engine reads the proof or merely runs it and ignores it.
    """
    import nodes._otr_video_engines  # noqa: F401  (registers every engine)
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines import wrapper_bridge as wb

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("the engine renders a real clip to prove")
    pytest.importorskip("soundfile")
    if engine_name == "viz_mxc_mandala":
        pytest.importorskip("cairo")

    sentinel = 4242
    monkeypatch.setattr(wb, "proven_frame_count",
                        lambda _path, _declared, **_kw: sentinel)
    request = {"shot_id": "s1", "canvas": {"w": 160, "h": 96, "fps": 25},
               "timing": {"target_frame_count": 3},
               "seed_bundle": {"request_seed": 7}}
    raw = vreg.get_engine(engine_name).render_clip(request)
    assert raw["frame_count"] == sentinel, (
        "%s returned %r -- it is still stamping its own loop bound rather "
        "than the proven count" % (engine_name, raw["frame_count"]))
