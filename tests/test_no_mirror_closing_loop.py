"""NO-MIRROR, step 4 -- the ONE loop that survives must prove what it is.

Operator ruling 2026-08-06: *"there is no mirror or ping pong unless for
credits."* The sanctioned reuse is the CLOSING-THEME BACKDROP -- not the credits
roll, which freezes a frame and never loops.

WHAT THIS REPLACED. ``plan_timeline_segments``' tail block looped the last drama
clip to fill ANY shortfall between the assembled body and the master length,
whatever caused it. The sanctioned case and an unexplained underrun are
indistinguishable at that point by construction: both are "the video ran out
before the audio did". A named constant was rejected twice for this reason --
vocabulary without a manifest-backed window is not enforcement.

WHAT IT WAS COVERED BY, corrected. A first draft of this docstring claimed the
tail loop had NO test anywhere. That was wrong, and the suite said so: two tests
in ``test_video_render_path_cw4.py`` --
``test_plan_timeline_segments_positions_by_start_s_and_fills_to_master`` and
``test_positioned_crossfades_partition_visible_timeline_without_duplication`` --
both depend on the tail holding the last drama clip, and both went red on this
change. The wrong conclusion came from grepping two files and generalising.
Their fixtures simply carried no closing cue; they are production-shaped now.

WHAT WAS GENUINELY MISSING is narrower and still worth this file: nothing
asserted WHY the tail may loop. The old tests pinned the BUG-410 look ("credits
over the scene") and would have passed just as happily if the loop fired for a
drama beat that ran short -- which is the case this build had to separate out.

AND IT DOES NOT COST REAL EPISODES THEIR BACKDROP. Measured against shipped
ledgers before the change landed: 11 of 12 recent episodes carry exactly one
``music_close`` row in ``master_mix`` space, every one mints a valid window, and
the windows end exactly at the master length. The twelfth is a ``pending_*``
ledger captured before the assembler runs. The loop the operator confirmed keeps
firing; only the unexplained ones stop.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes import otr_silent_composite as sc


def _manifest(window=None, *, total=300, fps=25):
    """One positioned beat covering frames 0..199, so the tail is 200..299."""
    manifest = {
        "fps": fps,
        "total_target_frames": total,
        "clips": [{"shot_id": "shot_b001", "engine_id": "wan_ti2v",
                   "path": "body.mp4", "exists": True,
                   "frame_count": 200, "target_frame_count": 200,
                   "start_s": 0.0}],
    }
    if window is not None:
        manifest["closing_theme_frame_window"] = window
    return manifest


def _plan(manifest, total=300):
    return sc.plan_timeline_segments(manifest, target_total_frames=total, fps=25)


def _tail(segments):
    return segments[-1] if segments else None


# ---------------------------------------------------------------------------
# The sanctioned case still works -- and this is the half nothing tested before
# ---------------------------------------------------------------------------
def test_a_tail_INSIDE_the_closing_window_still_loops_the_backdrop():
    """The operator's confirmed-OK behaviour. If this goes red the build has
    removed the one reuse it was told to keep."""
    segments, _ = _plan(_manifest({"start": 180, "end": 300}))
    tail = _tail(segments)
    assert tail["source"] == "clip"
    assert tail["loop"] is True
    assert tail["n_frames"] == 100


def test_the_window_may_extend_PAST_the_master_because_the_theme_runs_into_credits():
    """``E >= target_total_frames`` is the normal shape -- the closing theme
    lives partly in the credits post-roll, past the body video's end."""
    segments, _ = _plan(_manifest({"start": 150, "end": 100000}))
    assert _tail(segments)["loop"] is True


def test_the_boundary_is_INCLUSIVE_at_the_start_and_at_the_end():
    """``S <= cursor < total <= E`` exactly. The tail begins at 200 here."""
    assert _tail(_plan(_manifest({"start": 200, "end": 300}))[0])["loop"] is True


# ---------------------------------------------------------------------------
# Everything else gets floor/black. Never loop on doubt.
# ---------------------------------------------------------------------------
def test_NO_window_means_NO_loop_which_is_the_whole_change():
    """A manifest with no closing window is every episode whose tail is not
    provably the theme -- and every legacy manifest."""
    tail = _tail(_plan(_manifest())[0])
    assert tail["source"] in ("floor", "black")
    assert not tail.get("loop")


def test_a_tail_that_STARTS_BEFORE_the_theme_is_drama_time_and_is_refused():
    """cursor 200 < start 250. Looping here would repeat a drama beat over the
    run-up to the theme."""
    tail = _tail(_plan(_manifest({"start": 250, "end": 300}))[0])
    assert tail["source"] in ("floor", "black")


def test_a_tail_that_RUNS_PAST_the_theme_is_refused():
    """total 300 > end 260: the last 40 frames are past the theme, so a loop
    would cover whatever follows it."""
    tail = _tail(_plan(_manifest({"start": 180, "end": 260}))[0])
    assert tail["source"] in ("floor", "black")


@pytest.mark.parametrize("window", [
    {}, {"start": 180}, {"end": 300},
    {"start": "180", "end": "300"},
    {"start": 180.5, "end": 300.5},
    {"start": True, "end": 300},
    {"start": None, "end": None},
    [180, 300], "180-300", 7,
])
def test_a_MALFORMED_window_authorizes_nothing(window):
    """Every one of these returns floor/black. ``True`` is called out because it
    is an ``int`` and would otherwise read as frame 1 -- the same rejection
    ``acceptance.frame_count`` makes for the same reason."""
    tail = _tail(_plan(_manifest(window))[0])
    assert tail["source"] in ("floor", "black")


# ---------------------------------------------------------------------------
# The classifier itself, directly
# ---------------------------------------------------------------------------
def test_the_classifier_is_TOTAL_over_any_input():
    for manifest in (None, {}, [], "x", 7, {"closing_theme_frame_window": 1}):
        for cursor in (0, -1, "x", None):
            assert sc.closing_window_authorizes_loop(manifest, cursor, 300) \
                is False


def test_a_ZERO_LENGTH_tail_is_not_authorized():
    """``cursor < total`` is required: with nothing to fill there is nothing to
    authorize, and a True here would be a licence looking for a use."""
    assert sc.closing_window_authorizes_loop(
        _manifest({"start": 0, "end": 500}), 300, 300) is False


def test_the_classifier_agrees_with_the_window_the_render_driver_MINTS():
    """END-TO-END ON ONE RULER, and this is what a named constant could not do.

    The window is not hand-written here -- it is derived by the real producer
    from real ledger rows, then handed to the real consumer. If the two ever
    disagreed about frame conventions (``round`` vs ``ceil``, inclusive vs
    exclusive) this is where it would show, rather than in a published episode.
    """
    from nodes._otr_video_engines import render_driver as rd
    lines = [{"line_id": "music_closing_001", "speaker_role": "music_close",
              "start_s_space": "master_mix", "start_s": 8.0, "dur_s": 4.0,
              "music_cue_id": "closing", "music_chunk_index": 1,
              "music_chunk_count": 1}]
    window = rd.closing_theme_frame_window(lines, 25)
    assert window == {"start": 200, "end": 300}
    # The body ends at 200 and the master is 300 -> squarely inside.
    assert sc.closing_window_authorizes_loop(
        {"closing_theme_frame_window": window}, 200, 300) is True
    # One frame of theme short of the master and it is refused.
    assert sc.closing_window_authorizes_loop(
        {"closing_theme_frame_window": window}, 200, 301) is False
