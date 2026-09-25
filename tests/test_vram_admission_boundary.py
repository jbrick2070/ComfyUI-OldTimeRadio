"""The ONE admission boundary for a coverage-planned beat.

``_planned_length`` never consults the VRAM predictor by design, so once
``ltx_8gb`` and its siblings became planning-capped, the only ENFORCING guard
was bypassed on the path doing most of the rendering. These
tests pin the replacement, including the half that is easy to get wrong: an
engine with no MEASURED cost row must be reported as unenforced rather than
judged against a borrowed row, because a guard that refuses and admits with
equal confidence is worse than no guard at all.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import render_driver as rd


class _Segment:
    def __init__(self, index, render_frames):
        self.index = index
        self.render_frames = render_frames


def _prebuilt(frames, width=832, height=480):
    return [(_Segment(i, n), {"canvas": {"w": width, "h": height}})
            for i, n in enumerate(frames)]


def test_engine_without_a_cost_row_is_reported_unenforced(monkeypatch):
    """The important one: no row must not look like a passed check."""
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 14000.0)
    shot = {"engine_id": "humo"}
    assert not mc.cost_row_may_refuse("humo")
    record = rd._assert_beat_affordable(shot, _prebuilt([49, 49]))
    assert record["enforced"] is False
    assert record["checked"] == []
    assert "no QUALIFIED cost row" in record["reason"]
    assert "humo" in record["reason"]


def test_a_PRESENT_but_disqualified_row_may_not_refuse(monkeypatch):
    """A lane that HAS a row may still not refuse anything.

    This is the regression that shipped for about an hour: the guard gated on
    "is there a row?", and a 5B video lane had one -- `(7000.0, 185.0)`, the
    seed this repo disqualified in writing after it refused a real production
    leg ("affordable 24 frames (free=13481 MB)") on an engine that had already
    shipped an episode.

    That lane is gone and the table is empty, so the row is SEEDED here onto a
    live coverage-planned engine: the property is the gate's, not the lane's.
    At every realistic free-VRAM level that seed refuses EVERY segment length
    the coverage planner produces, including 93 frames at 14,500 MB free -- so
    gating on existence would re-arm, at a brand new call site, the exact
    refusal `_planned_length` had already stopped issuing. The question a
    refusal must answer is "may this row refuse?", not "does one exist?".
    """
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 14500.0)
    monkeypatch.setitem(mc.FRAME_COST_MODEL, "ltx_8gb", mc._DEFAULT_FRAME_COST)
    assert "ltx_8gb" in mc.FRAME_COST_MODEL, "fixture assumes the row is present"
    assert not mc.cost_row_may_refuse("ltx_8gb"), "present is not qualified"

    # A planner-shaped output for a long beat.
    record = rd._assert_beat_affordable({"engine_id": "ltx_8gb"},
                                        _prebuilt([177, 177, 93]))
    assert record["enforced"] is False, (
        "the disqualified row must not enforce -- every one of these lengths "
        "would be refused by it")
    assert "disqualified" in record["reason"]

    # And the STATIC path does not refuse them either (2026-08-13).
    #
    # This assertion used to run the other way: it proved the row "really would
    # have refused", because until the render gate caught it,
    # `compute_real_frame_budget` priced frames without asking
    # `cost_row_may_refuse` at all. So the disqualified row was inert at THIS
    # boundary and live one call away -- and that is the path that killed two
    # live render-gate legs. One row, one authority, both call sites.
    for frames in (93, 177):
        assert mc.assert_frame_affordable(
            14500.0, frames, 832, 480, "ltx_8gb") == frames


def test_no_row_is_qualified_today_and_that_is_deliberate():
    """The tripwire. If this set silently fills, a row started refusing renders
    without anyone re-measuring it through the real render lifecycle."""
    assert mc.QUALIFIED_COST_ROWS == frozenset(), (
        "a cost row became enforceable: %r. That is allowed ONLY after the row "
        "is measured through the real prepare() + render_clip() path -- no "
        "bench graph may qualify one. Update this test with the evidence."
        % (set(mc.QUALIFIED_COST_ROWS),))


@pytest.fixture
def qualified(monkeypatch):
    """Seed and qualify a row for `ltx_8gb` so the ENFORCEMENT machinery can be
    tested.

    No row is qualified in production today and that is deliberate (see
    ``test_no_row_is_qualified_today_and_that_is_deliberate``), and the table
    itself is empty. These tests are about whether the boundary refuses
    correctly WHEN a row is trusted, which is a separate question from whether
    any row is trusted yet -- and qualification needs a row to qualify.
    """
    monkeypatch.setitem(mc.FRAME_COST_MODEL, "ltx_8gb", mc._DEFAULT_FRAME_COST)
    monkeypatch.setattr(mc, "QUALIFIED_COST_ROWS", frozenset({"ltx_8gb"}))
    assert mc.cost_row_may_refuse("ltx_8gb")
    return "ltx_8gb"


def test_unaffordable_planned_segment_proceeds_and_is_reported(
        monkeypatch, qualified, caplog):
    """A qualified row predicted too small for the planned length PROCEEDS.

    NEVER REFUSE ON A NUMBER; ONLY AN OOM DECIDES (operator directive,
    2026-09-23). This test asserted the opposite until that day -- it
    pinned a PREDICTED shortfall raising before any work ran. The cost
    model is still computed and still says what it thinks; it simply no
    longer ends the run. A real OOM names the allocation that failed and
    the size it wanted, which is the number worth having.

    The old docstring here argued that reaching the GPU was the harm, because
    an in-process CUDA OOM "corrupts the allocator rather than failing
    cleanly". The operator weighed that against the cost of refusing on a
    guess and chose the OOM: it is the only reading that is not an estimate,
    and it names the allocation and size to shrink to.
    """
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 1200.0)
    with caplog.at_level("WARNING"):
        record = rd._assert_beat_affordable(
            {"engine_id": qualified}, _prebuilt([177]))
    assert record is not None


def test_free_vram_unreadable_is_reported_not_guessed(monkeypatch, qualified):
    """On a box with no NVML/torch the guard says so instead of inventing one."""
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 0.0)
    record = rd._assert_beat_affordable({"engine_id": qualified}, _prebuilt([177]))
    assert record["enforced"] is False
    assert "unreadable" in record["reason"]


def test_affordable_beat_records_every_segment(monkeypatch, qualified):
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 14000.0)
    record = rd._assert_beat_affordable({"engine_id": qualified},
                                        _prebuilt([17, 17, 17]))
    assert record["enforced"] is True
    assert [row["segment_index"] for row in record["checked"]] == [0, 1, 2]
    assert {row["canvas"] for row in record["checked"]} == {"832x480"}
    # Free VRAM is read ONCE for the whole beat, with no hoist correction --
    # nothing is loaded yet, so there is nothing resident to credit back.
    assert record["free_vram_mb"] == 14000.0


def test_no_segments_is_not_an_enforced_pass():
    record = rd._assert_beat_affordable({"engine_id": "ltx_8gb"}, [])
    assert record["enforced"] is False
