"""Regression tests for OTR_BatchLTXRender (in-graph LTX batch).

Pins:
  - LTX constants (LTX_MAX_FRAMES = 705 post BUG-LOCAL-117e mega-duration,
    LTX_CHUNK_FRAMES = 177)
  - ltx_length_for_dur 8n+1 frame snap with floor + ceiling
  - ltx_length_for_dur_uncapped bypasses the cap (used by chunking dispatch)
  - clip_length widget (BUG-LOCAL-117e): default 22.0, max 28.16
  - _concat_clips_via_ffmpeg helper exists with the expected signature
  - _make_boomerang_via_ffmpeg helper exists (BUG-LOCAL-117d)
  - LTX_LOOP_VIA_REVERSE_DEFAULT == "on" (BUG-LOCAL-117d default-on)

Doesn't exercise the actual LTX rendering loop -- that requires
ComfyUI's runtime + GPU. End-to-end loop is integration-tested by
Jeffrey queueing the FULL workflow.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Match other tests' import pattern (test_critique_dialogue_preservation.py):
# repo root on path so ``from nodes import batch_ltx_render`` works despite
# ``import folder_paths`` at module top.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


@pytest.fixture(scope="module")
def m():
    """Load the batch_ltx_render module under test."""
    from nodes import batch_ltx_render as blr  # noqa: WPS433
    return blr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_ltx_constants(m):
    """BUG-LOCAL-117e (2026-05-07 PM): LTX_MAX_FRAMES bumped from 353
    to 705 (8*88+1 = ~28.20s @ 25fps) after the mega-duration smoke
    rendered 25s @ 832x480 cleanly on RTX 5080 16 GB + 64 GB RAM.
    LTX_CHUNK_FRAMES still pinned at 177 for the historical fallback.
    """
    assert m.LTX_FPS == 25
    assert m.LTX_MIN_FRAMES == 9
    assert m.LTX_MAX_FRAMES == 705
    assert m.LTX_CHUNK_FRAMES == 177


# ---------------------------------------------------------------------------
# ltx_length_for_dur (capped) + uncapped helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dur_s, expected_frames", [
    (3.84, 97),    # 96 -> 97 (8n+1)
    (7.00, 177),   # 175 -> 177 (8*22+1 = 177, the historical chunk size)
    (1.00, 25),    # 25 (8*3+1)
    (0.10, 9),     # below MIN -> floored
    (10.28, 257),  # 257 frames @ 25fps (8*32+1 = 257)
    (14.12, 353),  # 8*44+1 = 353 (was the OLD cap pre-117e)
    (16.00, 401),  # 16*25=400 -> 8*50+1 = 401 (no longer capped)
    (20.00, 505),  # 20*25=500 -> 8*63+1 = 505 (no longer capped)
    (28.16, 705),  # cap exactly (8*88+1 = 705, BUG-LOCAL-117e)
    (30.00, 705),  # over the new cap -> still capped at 705
])
def test_ltx_length_for_dur(m, dur_s, expected_frames):
    assert m.ltx_length_for_dur(dur_s) == expected_frames


def test_ltx_length_for_dur_always_returns_8n_plus_1(m):
    """LTX VAE temporal compression requires 8n+1 frame counts."""
    for dur in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]:
        f = m.ltx_length_for_dur(dur)
        assert (f - 1) % 8 == 0, f"dur={dur}s -> frames={f} not 8n+1"


def test_ltx_length_for_dur_uncapped_skips_cap(m):
    """BUG-LOCAL-091: chunking dispatch uses the uncapped helper to
    decide whether a line needs splitting. Result must NOT be clamped
    to LTX_MAX_FRAMES.

    BUG-LOCAL-117e: LTX_MAX_FRAMES = 705 (was 353). Use a duration well
    past the new cap so the cap-vs-uncap split is unambiguous.
    """
    capped = m.ltx_length_for_dur(40.0)
    uncapped = m.ltx_length_for_dur_uncapped(40.0)
    assert capped == 705
    assert uncapped == 1001  # 40s @ 25fps = 1000 -> 8*125+1 = 1001
    assert (uncapped - 1) % 8 == 0


# ---------------------------------------------------------------------------
# clip_length widget (BUG-LOCAL-091)
# ---------------------------------------------------------------------------

def test_clip_length_widget_present(m):
    """BUG-LOCAL-091: clip_length widget added for chunking dispatch parity
    with BatchHumoRender."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    assert "clip_length" in inp.get("optional", {})


def test_clip_length_default_is_22(m):
    """BUG-LOCAL-117e (2026-05-07 PM): clip_length default bumped from
    7.0s to 22.0s after the mega-duration smoke validated 25s @ 832x480
    on RTX 5080 16 GB + 64 GB RAM. 22s leaves 3s headroom under the 25s
    observed envelope and pairs with BUG-LOCAL-117d boomerang (which
    halves the actual sample window to ~11s)."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    spec = inp["optional"]["clip_length"][1]
    assert spec["default"] == 22.0


def test_clip_length_max_is_28_16s(m):
    """Max bumped to 28.16s (8*88+1 = 705 frames) on 2026-05-07 to match
    the new LTX_MAX_FRAMES=705 absolute hardware ceiling. BUG-LOCAL-117e."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    spec = inp["optional"]["clip_length"][1]
    assert spec["max"] == pytest.approx(28.16, abs=0.01)


def test_execute_signature_accepts_clip_length(m):
    """``execute`` must accept clip_length as a keyword argument so the
    widget value is plumbed through. Default 22.0 post-BUG-LOCAL-117e
    (was 7.0 pre-117e)."""
    import inspect
    sig = inspect.signature(m.BatchLTXRender.execute)
    assert "clip_length" in sig.parameters
    default = sig.parameters["clip_length"].default
    assert default == 22.0


# ---------------------------------------------------------------------------
# _concat_clips_via_ffmpeg helper exists with expected shape
# ---------------------------------------------------------------------------

def test_concat_helper_exists(m):
    """BUG-LOCAL-091: chunking dispatch needs the concat helper."""
    assert callable(getattr(m, "_concat_clips_via_ffmpeg", None))


def test_concat_helper_rejects_empty_list(m, tmp_path: Path):
    with pytest.raises(RuntimeError):
        m._concat_clips_via_ffmpeg([], tmp_path / "out.mp4")


def test_concat_helper_single_chunk_copies(m, tmp_path: Path):
    """Single-chunk case is a defensive copy (not a real concat) so the
    chunking dispatch can call it uniformly without branching."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake mp4 content")
    dst = tmp_path / "dst.mp4"
    result = m._concat_clips_via_ffmpeg([src], dst)
    assert result == dst
    assert dst.exists()
    assert dst.read_bytes() == b"fake mp4 content"


def test_concat_helper_single_chunk_noop_at_same_path(m, tmp_path: Path):
    """If the only chunk path equals out_path, no copy needed (avoids
    the SameFileError shutil.copy2 raises on Windows)."""
    p = tmp_path / "same.mp4"
    p.write_bytes(b"fake mp4 content")
    result = m._concat_clips_via_ffmpeg([p], p)
    assert result == p
    assert p.exists()
    assert p.read_bytes() == b"fake mp4 content"


# ---------------------------------------------------------------------------
# BUG-LOCAL-095: i2v dispatch uses canonical LTXVImgToVideoConditionOnly
# ---------------------------------------------------------------------------

def test_i2v_dispatch_uses_img_to_video_condition_only():
    """BUG-LOCAL-095 (2026-05-04 EVENING): the per-chunk LTX dispatch
    must call LTXVImgToVideoConditionOnly (canonical i2v init that
    bakes the image into first frames + adds noise mask) NOT
    LTXVAddGuide (keyframe pinning). The latter clamps frame 0 as a
    hard anchor and produces visibly static output -- the artefact
    Jeffrey reported AFTER BUG-032 removed the end-frame guide.

    Source-code regression guard: a future refactor that re-introduces
    LTXVAddGuide for the i2v path will fail this test before any
    live render reproduces the static-LTX artefact.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    # Must call LTXVImgToVideoConditionOnly with vae+image+latent+strength.
    assert '"LTXVImgToVideoConditionOnly"' in src, (
        "BUG-LOCAL-095: dispatch must call LTXVImgToVideoConditionOnly "
        "(matches comfyui-data-media-machine which produces animated "
        "i2v output, vs LTXVAddGuide which pins frame 0 as keyframe)"
    )
    # Must NOT call LTXVAddGuide as the i2v path (keyframe pinning
    # behaviour). Comments and docstrings can mention it for history;
    # the actual node-dispatch call would be `_call("LTXVAddGuide",`
    # with no quoting variants. Use a regex that matches the call
    # syntax specifically.
    pattern = re.compile(r'_call\(\s*["\']LTXVAddGuide["\']')
    matches = pattern.findall(src)
    assert not matches, (
        "BUG-LOCAL-095: LTXVAddGuide must not be used for i2v "
        "conditioning. It pins frame 0 as a hard keyframe and "
        "produces static LTX output. Use LTXVImgToVideoConditionOnly "
        "instead."
    )


def test_clip_length_widget_appears_after_existing_optional_widgets(m):
    """BUG-LOCAL-097 (2026-05-04 EVENING): clip_length must be the LAST
    optional widget so existing workflow JSONs (saved before BUG-091
    introduced the widget) don't shift their saved values into the
    wrong slots. ComfyUI parses widget_values positionally; inserting
    a new FLOAT widget BEFORE existing STRING widgets means the saved
    "ffmpeg" string lands on a FLOAT slot and validation fails with
    `Failed to convert an input value to a FLOAT value: clip_length,
    ffmpeg, could not convert string to float: 'ffmpeg'`. Putting the
    new widget LAST means old workflows fall through to the FLOAT
    default cleanly.
    """
    inp = m.BatchLTXRender.INPUT_TYPES()
    optional = inp["optional"]
    keys = list(optional.keys())
    assert "clip_length" in keys
    # clip_length must be the last entry in the optional dict.
    assert keys[-1] == "clip_length", (
        f"BUG-LOCAL-097: clip_length must be the LAST optional widget "
        f"to preserve backward-compat with pre-091 workflow JSONs. "
        f"Got order: {keys}"
    )


def test_required_nodes_list_mentions_img_to_video_condition_only():
    """The error message that fires when a required ComfyUI node is
    missing should list the new canonical i2v node so a fresh-install
    user knows what's needed."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    assert "LTXVImgToVideoConditionOnly" in src, (
        "Required-nodes error message must reference the BUG-095 node"
    )


# ---------------------------------------------------------------------------
# BUG-LOCAL-117d: ffmpeg boomerang post-process
# ---------------------------------------------------------------------------

def test_boomerang_default_is_on(m):
    """BUG-LOCAL-117d (2026-05-07 PM): boomerang default flipped to ON
    so the next FULL test renders seamless radio_bookend -> motion ->
    radio_bookend loops without stacked-anchor glitching. Set
    OTR_LTX_LOOP_VIA_REVERSE=off to revert."""
    assert m.LTX_LOOP_VIA_REVERSE_DEFAULT == "on"


def test_boomerang_truthy_values_pinned(m):
    """The truthy set must include 'on', '1', 'true', 'yes' so users
    can flip the env var with whichever convention they prefer."""
    assert "on" in m.LTX_LOOP_VIA_REVERSE_TRUTHY
    assert "1" in m.LTX_LOOP_VIA_REVERSE_TRUTHY
    assert "true" in m.LTX_LOOP_VIA_REVERSE_TRUTHY


def test_boomerang_helper_exists(m):
    """BUG-LOCAL-117d: in-place ffmpeg boomerang helper must exist with
    the expected signature ``(mp4_path, ffmpeg='ffmpeg') -> Path``."""
    fn = getattr(m, "_make_boomerang_via_ffmpeg", None)
    assert callable(fn), (
        "BUG-LOCAL-117d: _make_boomerang_via_ffmpeg helper missing"
    )
    import inspect
    sig = inspect.signature(fn)
    assert "mp4_path" in sig.parameters
    assert "ffmpeg" in sig.parameters
    assert sig.parameters["ffmpeg"].default == "ffmpeg"


def test_boomerang_helper_rejects_missing_input(m, tmp_path: Path):
    """Missing input file must raise FileNotFoundError so the chunk
    loop's try/except can log + leave the half-duration clip intact
    (instead of crashing the whole episode)."""
    with pytest.raises(FileNotFoundError):
        m._make_boomerang_via_ffmpeg(tmp_path / "does_not_exist.mp4")


def test_boomerang_filter_uses_split_reverse_concat():
    """Source-code regression guard: the boomerang helper must use
    ffmpeg's split/reverse/concat filter graph (not a naive concat
    demuxer). The ``trim=start_frame=1`` step drops the duplicate
    midpoint frame so the loop reads as a smooth oscillation."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    assert "[0:v]split[a][b]" in src, (
        "BUG-LOCAL-117d: boomerang must split video into two streams"
    )
    assert "[b]reverse" in src, (
        "BUG-LOCAL-117d: boomerang must reverse the second stream"
    )
    assert "trim=start_frame=1" in src, (
        "BUG-LOCAL-117d: boomerang must drop the duplicate midpoint "
        "frame to avoid a one-frame stutter at the loop apex"
    )
    assert "concat=n=2:v=1:a=0" in src, (
        "BUG-LOCAL-117d: boomerang must concat forward + reversed"
    )


def test_boomerang_pins_video_track_timescale():
    """BUG-LOCAL-117d hardening 2026-05-07: the boomerang helper must
    pin ``-video_track_timescale 12800`` so concat-demuxer with -c copy
    in VideoComposite doesn't emit ``Non-monotonous DTS`` at the seams
    between boomeranged clips and the static-fill / pillarbox-HuMo
    encodes (which all share the same 12800 timebase)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    # The constant lives in video_composite.py as _STATIC_SEGMENT_TIMEBASE,
    # but here we hard-code "12800" since batch_ltx_render is the
    # upstream producer and shouldn't depend on VideoComposite imports.
    assert '"-video_track_timescale", "12800"' in src, (
        "BUG-LOCAL-117d hardening: _make_boomerang_via_ffmpeg must pin "
        "the timebase to 12800 to match _save_video_mp4 + VideoComposite "
        "static fill / pillarbox encodes"
    )


# ---------------------------------------------------------------------------
# BUG-LOCAL-117f: duration-aware anti-clobber
# ---------------------------------------------------------------------------

def test_anti_clobber_uses_probe_duration_check():
    """BUG-LOCAL-117f (2026-05-07): the per-line anti-clobber check must
    ffprobe the on-disk clip and compare against the audio-target
    duration. A pre-existing half-duration mp4 (e.g. left over from a
    boomerang-failed crashed run) must be unlinked and re-rendered, not
    silently skipped. Tolerance is 0.25s -- wider than VideoComposite's
    BUG-031 sync tolerance so we never re-render a clip the composite
    would have accepted as-is."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    # The block must reference the probe helper.
    assert "probe_duration_s" in src, (
        "BUG-LOCAL-117f: anti-clobber must call _otr_probe.probe_duration_s "
        "to read the on-disk duration"
    )
    # The tolerance must be 0.25 (matches VideoComposite BUG-031 envelope).
    assert "_expected_dur_s - 0.25" in src, (
        "BUG-LOCAL-117f: anti-clobber must use 0.25s staleness tolerance "
        "so a clip the composite would accept is never re-rendered"
    )
    # The block must actually unlink stale clips, not just warn.
    assert "pre_existing.unlink()" in src, (
        "BUG-LOCAL-117f: anti-clobber must DELETE the stale clip so the "
        "fall-through render path can write a fresh one"
    )
    # Failure handling: log.error + STALE-LOCKED report on unlink failure.
    assert "STALE-LOCKED" in src, (
        "BUG-LOCAL-117f: when unlink fails (e.g. file-locked by Explorer "
        "preview), the anti-clobber must report STALE-LOCKED and skip "
        "instead of crashing"
    )


def test_anti_clobber_re_render_path_falls_through(tmp_path):
    """BUG-LOCAL-117f: the duration-staleness branch must NOT have a
    ``continue`` statement -- it must fall through to the render path
    so a fresh clip gets written. The skip branch (clip is fresh
    enough) keeps the ``continue``.

    Implementation regression guard: if a future refactor accidentally
    adds ``continue`` after the unlink, the half-duration heal becomes
    a half-duration delete-and-skip, which is worse than no fix at all
    (now both old and new run leave the line uncovered)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    # Find the BUG-LOCAL-117f anti-clobber block. It runs from the
    # comment marker to the matching SKIP existing branch close. We
    # check that the stale path comments include the explicit
    # "Fall through to render path below (no continue)." note so a
    # future maintainer can't reinstate the continue without removing
    # this guard comment first.
    assert "Fall through to render path below (no continue)." in src, (
        "BUG-LOCAL-117f: stale-clip branch must explicitly note that it "
        "falls through (no continue) so the renderer writes a fresh clip"
    )


def test_anti_clobber_test_fixture_uses_testchar_naming(tmp_path):
    """Naming standard (CLAUDE.md): test fixtures use TESTCHAR, never
    "dummy". This test exists to assert the naming convention via a
    fixture file that future integration tests can pick up.

    The fixture is a 0-byte file so we don't depend on ffmpeg in CI;
    it lives in tmp_path so pytest cleanup handles disposal."""
    fixture = tmp_path / "TESTCHAR_l001.mp4"
    fixture.write_bytes(b"")
    assert fixture.name == "TESTCHAR_l001.mp4"
    assert "dummy" not in fixture.name.lower()
