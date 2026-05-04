"""Regression tests for OTR_BatchLTXRender (in-graph LTX batch).

Pins:
  - LTX constants (LTX_MAX_FRAMES = 353 post BUG-LOCAL-091, LTX_CHUNK_FRAMES = 177)
  - ltx_length_for_dur 8n+1 frame snap with floor + ceiling
  - ltx_length_for_dur_uncapped bypasses the cap (used by chunking dispatch)
  - clip_length widget (BUG-LOCAL-091): default 7.0, max 14.12
  - _concat_clips_via_ffmpeg helper exists with the expected signature

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
    """BUG-LOCAL-091: LTX_MAX_FRAMES bumped from 177 to 353 to cover
    typical announcer/music beats in a single pass. LTX_CHUNK_FRAMES
    pinned at the historically-stable 177 for the chunking fallback.
    """
    assert m.LTX_FPS == 25
    assert m.LTX_MIN_FRAMES == 9
    assert m.LTX_MAX_FRAMES == 353
    assert m.LTX_CHUNK_FRAMES == 177


# ---------------------------------------------------------------------------
# ltx_length_for_dur (capped) + uncapped helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dur_s, expected_frames", [
    (3.84, 97),    # 96 -> 97 (8n+1)
    (7.00, 177),   # 175 -> 177 (8*22+1 = 177, was the OLD cap)
    (1.00, 25),    # 25 (8*3+1)
    (0.10, 9),     # below MIN -> floored
    (10.28, 257),  # 257 frames @ 25fps (8*32+1 = 257)
    (14.12, 353),  # cap exactly (8*44+1 = 353)
    (16.00, 353),  # over cap -> capped
    (20.00, 353),  # well over cap -> still capped (chunking handles it)
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
    to LTX_MAX_FRAMES."""
    capped = m.ltx_length_for_dur(30.0)
    uncapped = m.ltx_length_for_dur_uncapped(30.0)
    assert capped == 353
    assert uncapped == 753  # 30s @ 25fps = 750 -> 8*94+1 = 753
    assert (uncapped - 1) % 8 == 0


# ---------------------------------------------------------------------------
# clip_length widget (BUG-LOCAL-091)
# ---------------------------------------------------------------------------

def test_clip_length_widget_present(m):
    """BUG-LOCAL-091: clip_length widget added for chunking dispatch parity
    with BatchHumoRender."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    assert "clip_length" in inp.get("optional", {})


def test_clip_length_default_is_seven(m):
    """7.0s default matches BatchHumoRender + the historically-stable
    LTX_CHUNK_FRAMES=177 single-chunk size."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    spec = inp["optional"]["clip_length"][1]
    assert spec["default"] == 7.0


def test_clip_length_max_respects_humo_ceiling(m):
    """Max bumped to 14.12 to match the new LTX_MAX_FRAMES=353 cap +
    BatchHumoRender's clip_length max for symmetry."""
    inp = m.BatchLTXRender.INPUT_TYPES()
    spec = inp["optional"]["clip_length"][1]
    assert spec["max"] == pytest.approx(14.12, abs=0.01)


def test_execute_signature_accepts_clip_length(m):
    """``execute`` must accept clip_length as a keyword argument so the
    widget value is plumbed through. Defaults to 7.0 (single-pass at the
    historical chunk size)."""
    import inspect
    sig = inspect.signature(m.BatchLTXRender.execute)
    assert "clip_length" in sig.parameters
    default = sig.parameters["clip_length"].default
    assert default == 7.0


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
