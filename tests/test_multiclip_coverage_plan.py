"""Chunk 3 -- the partitioner: contract purity, exact-sum, seam arithmetic.

The invariant everything here defends:

    sum(render_frames - drop_head - trim_tail) == target_visible_frames

exactly. A beat whose assembled length drifts from its audio is the defect
this build removes, so an inexact plan must be a terminal error, never a
rounding.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc


LTX8 = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                        supports_multi_clip=True,
                        continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)

JUMPY = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                         supports_multi_clip=True,
                         continuity=fc.CONTINUITY_SOFT_REFERENCE)

VEO = fc.FrameContract(min_frames=100, max_frames=200,
                       discrete_durations=(100, 150, 200),
                       allow_tail_trim=True, supports_multi_clip=True,
                       continuity=fc.CONTINUITY_SOFT_REFERENCE)


# ---------------------------------------------------------------------------
# Purity -- the non-negotiable property
# ---------------------------------------------------------------------------

def test_partitioner_reads_no_environment(monkeypatch):
    """Stills are minted BEFORE the render phase, so a partition that depended
    on runtime state would be one the image phase could not plan stills for."""
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=viz_camera")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    a = cp.partition_beat(169, LTX8)
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cp.partition_beat(169, LTX8) == a


def test_partitioner_is_deterministic():
    assert cp.partition_beat(497, LTX8) == cp.partition_beat(497, LTX8)


# ---------------------------------------------------------------------------
# THE ADOPTED ACCEPTANCE CASE: a 169-frame beat == 161 + (9 - 1)
# ---------------------------------------------------------------------------

def test_the_169_frame_chain_is_two_forward_segments_with_no_trim():
    """The r3-adopted first live target. Exactly reproducible, unlike a vague
    "something over 161": the cap plus one legal minimum segment, less the
    chained duplicate head frame."""
    plan = cp.partition_beat(169, LTX8)
    assert plan.join_mode == cp.JOIN_CHAIN
    assert plan.segment_count == 2
    assert [s.render_frames for s in plan.segments] == [161, 9]
    assert [s.drop_head for s in plan.segments] == [0, 1]
    assert [s.trim_tail for s in plan.segments] == [0, 0]
    assert plan.total_visible_frames == 169


def test_162_frames_needs_the_cpu_tail_trim_case():
    """The separate 162-frame case r3 asked for: one over the cap, and 162 is
    not on the 8n+1 ladder, so this exercises legal tail trimming."""
    trimmable = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                 allow_tail_trim=True, supports_multi_clip=True,
                                 continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    plan = cp.partition_beat(162, trimmable)
    assert plan.total_visible_frames == 162
    cp.validate_coverage_plan(plan, trimmable)


# ---------------------------------------------------------------------------
# Exact sum, over a wide sweep -- the property r2 asked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", list(range(9, 900, 7)))
def test_every_chain_plan_sums_exactly(target):
    try:
        plan = cp.partition_beat(target, LTX8)
    except cp.CoveragePlanError:
        # Refusing is allowed; emitting a drifting plan is not.
        return
    assert plan.total_visible_frames == target
    cp.validate_coverage_plan(plan, LTX8)


@pytest.mark.parametrize("target", list(range(9, 900, 11)))
def test_every_jump_plan_sums_exactly(target):
    try:
        plan = cp.partition_beat(target, JUMPY)
    except cp.CoveragePlanError:
        return
    assert plan.total_visible_frames == target
    assert all(s.drop_head == 0 for s in plan.segments)
    cp.validate_coverage_plan(plan, JUMPY)


def test_a_beat_that_strands_a_remainder_still_resolves():
    """Greedy-largest-first strands 1 frame on an 8n+1 ladder at 313 visible
    frames, and 1 is not a legal render. Solving for the segment COUNT first
    finds the exact partition that a greedy walk cannot."""
    plan = cp.partition_beat(313, LTX8)
    assert plan.total_visible_frames == 313
    assert plan.segment_count >= 2
    for seg in plan.segments:
        assert LTX8.is_legal_length(seg.render_frames)


@pytest.mark.parametrize("target", [170, 200, 500])
def test_an_8n_plus_1_chain_can_only_assemble_to_8m_plus_1(target):
    """A REAL ARITHMETIC LIMIT, pinned rather than papered over.

    Chaining k segments of ``9 + 8a`` and dropping one duplicated head frame
    each assembles to ``8m + 1`` visible frames, always. So a beat whose target
    is not congruent to 1 mod 8 has NO exact cover on this ladder, and the
    partitioner must REFUSE rather than emit a plan that drifts from the beat
    audio. Covering such a beat requires tail trimming -- see the companion
    test below -- which is why ``allow_tail_trim`` is part of the declaration
    and not an assembler-side convenience.
    """
    assert target % 8 != 1              # the premise of the case
    with pytest.raises(cp.CoveragePlanError, match="no exact multi-clip cover"):
        cp.partition_beat(target, LTX8)


@pytest.mark.parametrize("target", [170, 200, 500])
def test_tail_trim_covers_the_beats_the_bare_ladder_cannot(target):
    trimmable = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                 allow_tail_trim=True, supports_multi_clip=True,
                                 continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    plan = cp.partition_beat(target, trimmable)
    assert plan.total_visible_frames == target
    assert sum(s.trim_tail for s in plan.segments) > 0
    cp.validate_coverage_plan(plan, trimmable)


# ---------------------------------------------------------------------------
# Join-mode selection comes from the adapter's own declaration
# ---------------------------------------------------------------------------

def test_single_only_engine_never_splits():
    plan = cp.partition_beat(17, fc.FrameContract(min_frames=9, max_frames=161,
                                                  quantum=8))
    assert plan.join_mode == cp.JOIN_SINGLE
    assert plan.segment_count == 1
    assert plan.is_multi_clip is False


def test_single_only_engine_over_its_cap_is_terminal():
    """It must NOT silently ping-pong, loop-fill or hold a frame -- those are
    exactly the three silent coverage mechanisms this build removes."""
    single = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    with pytest.raises(cp.CoveragePlanError, match="not opted in to multi-clip"):
        cp.partition_beat(400, single)


def test_a_beat_inside_one_render_stays_one_clip():
    plan = cp.partition_beat(161, LTX8)
    assert plan.segment_count == 1


def test_soft_reference_engines_jump_cut_rather_than_pretend():
    """HuMo's reference is an identity hint and Veo's lastFrame is
    interpolation inside ONE clip -- neither locks frame 0."""
    plan = cp.partition_beat(400, JUMPY)
    assert plan.join_mode == cp.JOIN_JUMP
    assert all(s.drop_head == 0 for s in plan.segments)


def test_discrete_duration_lane_partitions_over_its_menu():
    plan = cp.partition_beat(400, VEO)
    assert plan.total_visible_frames == 400
    for seg in plan.segments:
        assert seg.render_frames in VEO.discrete_durations
    cp.validate_coverage_plan(plan, VEO)


# ---------------------------------------------------------------------------
# Boundary validation -- called on BOTH sides of the wire
# ---------------------------------------------------------------------------

def test_round_trip_through_the_durable_stamp():
    plan = cp.partition_beat(169, LTX8)
    restored = cp.CoveragePlan.from_dict(plan.to_dict())
    assert restored == plan
    cp.validate_coverage_plan(restored, LTX8)


def test_a_plan_that_drifts_is_rejected():
    bad = cp.CoveragePlan(169, cp.JOIN_CHAIN, (
        cp.CoverageSegment(0, 161), cp.CoverageSegment(1, 9, drop_head=1),
    ))
    object.__setattr__(bad, "target_visible_frames", 170)
    with pytest.raises(cp.CoveragePlanError, match="drift from the beat audio"):
        cp.validate_coverage_plan(bad, LTX8)


def test_a_chain_successor_must_drop_exactly_one_head_frame():
    bad = cp.CoveragePlan(170, cp.JOIN_CHAIN, (
        cp.CoverageSegment(0, 161), cp.CoverageSegment(1, 9, drop_head=0),
    ))
    with pytest.raises(cp.CoveragePlanError, match="head frame"):
        cp.validate_coverage_plan(bad, LTX8)


def test_an_illegal_render_length_is_rejected_against_the_contract():
    bad = cp.CoveragePlan(20, cp.JOIN_CHAIN, (
        cp.CoverageSegment(0, 12), cp.CoverageSegment(1, 9, drop_head=1),
    ))
    with pytest.raises(cp.CoveragePlanError, match="cannot accept"):
        cp.validate_coverage_plan(bad, LTX8)


def test_chain_is_rejected_for_an_engine_that_cannot_lock_frame_zero():
    plan = cp.partition_beat(169, LTX8)          # a valid chain plan...
    with pytest.raises(cp.CoveragePlanError, match="must jump cut"):
        cp.validate_coverage_plan(plan, JUMPY)   # ...against a soft-ref engine
    

def test_a_segment_that_contributes_nothing_is_rejected():
    bad = cp.CoveragePlan(9, cp.JOIN_JUMP, (
        cp.CoverageSegment(0, 9), cp.CoverageSegment(1, 9, trim_tail=9),
    ))
    with pytest.raises(cp.CoveragePlanError, match="contributes"):
        cp.validate_coverage_plan(bad, JUMPY)


def test_segment_indexes_must_be_dense_and_ascending():
    bad = cp.CoveragePlan(18, cp.JOIN_JUMP, (
        cp.CoverageSegment(0, 9), cp.CoverageSegment(5, 9),
    ))
    with pytest.raises(cp.CoveragePlanError, match="assembly"):
        cp.validate_coverage_plan(bad, JUMPY)


def test_empty_plan_is_rejected():
    with pytest.raises(cp.CoveragePlanError):
        cp.validate_coverage_plan(cp.CoveragePlan(10, cp.JOIN_JUMP, ()), JUMPY)


@pytest.mark.parametrize("target", [0, -1])
def test_a_nonpositive_target_is_rejected(target):
    with pytest.raises(cp.CoveragePlanError):
        cp.partition_beat(target, LTX8)


def test_multi_segment_plan_rejected_for_single_only_contract():
    plan = cp.partition_beat(169, LTX8)
    single = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    with pytest.raises(cp.CoveragePlanError, match="not opted in"):
        cp.validate_coverage_plan(plan, single)
