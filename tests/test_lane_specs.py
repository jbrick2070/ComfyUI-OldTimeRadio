"""The ONE story-lane authority: `nodes/_otr_lane_specs.py`.

Pins the names-only/lazy-import contract (a LaneSpec stores strings, never
imported callables or exception classes, so building the table never drags a
runner module into ComfyUI startup) plus lane DISPATCH: `is_dispatched`,
`runner_for`, and the raise on a pipeline with no registered lane.

`assert_supported` and `is_roll_compatible` -- the two request-compatibility
entry points this file used to pin -- were REMOVED 2026-08-14 along with
`RollRequest` and the whole word-count authority. See the tombstone below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lane_specs as LANES
from nodes import _otr_scifi_fable2 as FABLE2


# ---------------------------------------------------------------------------
# names-only / lazy-import contract
# ---------------------------------------------------------------------------

def test_lane_specs_hold_names_never_callables_or_classes():
    """Importing objects to BUILD the table would drag every runner into
    ComfyUI startup, which is exactly what the lazy wrappers existed to
    prevent."""
    for pipeline_id, spec in LANES.LANE_SPECS.items():
        assert isinstance(spec.module, str) and spec.module, pipeline_id
        assert isinstance(spec.runner_attr, str) and spec.runner_attr


def test_every_declared_name_actually_resolves():
    """A typo in a NAMES-ONLY table is invisible until dispatch -- so resolve
    every one of them here."""
    for pipeline_id, spec in LANES.LANE_SPECS.items():
        module = LANES._load(spec.module)
        assert callable(getattr(module, spec.runner_attr)), pipeline_id


# `test_a_lane_with_declared_compat_errors_also_declares_a_hook` was DELETED
# 2026-08-14. It asserted that any LaneSpec declaring `compat_error_attrs`
# also declared a `compat_attr` hook to catch them with. Both fields were
# removed from the LaneSpec dataclass (and from both LANE_SPECS entries) in
# the same change that removed `RollRequest` and the word-count authority --
# there is no compat hook left to pair with anything.


# ---------------------------------------------------------------------------
# runner_for
# ---------------------------------------------------------------------------

def test_runner_for_resolves_the_dispatched_lanes():
    assert LANES.runner_for("scifi_news_pro_multipass") is (
        FABLE2.run_scifi_fable2_episode)


def test_runner_for_returns_none_for_a_known_inline_lane():
    """None is a real answer -- 'the writer's own body runs this'."""
    for pipeline_id in LANES.INLINE_PIPELINES:
        assert LANES.runner_for(pipeline_id) is None


def test_runner_for_raises_on_an_unregistered_pipeline():
    """No fallback: an unwired runnable bank is a wiring bug, and quietly
    running the legacy inline branch under its name would hide that."""
    with pytest.raises(LANES.UnknownLanePipelineError) as caught:
        LANES.runner_for("not_a_real_pipeline")
    assert "not_a_real_pipeline" in str(caught.value)
    assert "no fallback" in str(caught.value)


def test_is_dispatched_matches_the_table():
    assert LANES.is_dispatched("scifi_news_pro_multipass") is True
    assert LANES.is_dispatched("legacy_many_pass") is False
    assert LANES.is_dispatched("not_a_real_pipeline") is False


# ---------------------------------------------------------------------------
# assert_supported / is_roll_compatible / RollRequest -- REMOVED 2026-08-14
# ---------------------------------------------------------------------------
# This file used to carry two more blocks of tests here:
#
# * `assert_supported(bank, req)` -- raised the lane's OWN error type (e.g.
#   `CodexTargetRangeError`) when a `RollRequest(target_words=...)` fell
#   outside the lane's declared band (only one retired lane declared one,
#   30..900).
# * `is_roll_compatible(bank, req)` -- the same check as a bool, swallowing
#   ONLY the lane's declared compat errors, which the bank roll used to
#   filter its pool before drawing.
#
# Both, plus `RollRequest` itself and `_compat_hook`, were deleted with the
# word-count authority (operator directive 2026-08-14): the writer no longer
# takes a `target_words` request, so there is no target left for a lane to
# accept or decline, and a gate whose only input is gone is worse than no
# gate at all -- it still reads as live. The hoisted lane preflight these
# tests also pinned was removed in the same change; the act count is the
# only length-shaped knob now.
