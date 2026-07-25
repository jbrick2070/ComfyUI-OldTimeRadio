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

def _sweep(contract, targets):
    """Run a sweep and return (plans, refusals).

    Refusing is a legal outcome; emitting a DRIFTING plan is not. But a sweep
    that mostly refuses proves almost nothing, so callers assert on the plan
    count too -- see the QA note on :func:`test_chain_sweep_actually_covers`.
    """
    plans, refusals = [], 0
    for target in targets:
        try:
            plan = cp.partition_beat(target, contract)
        except cp.CoveragePlanError:
            refusals += 1
            continue
        assert plan.total_visible_frames == target, target
        cp.validate_coverage_plan(plan, contract)
        plans.append(plan)
    return plans, refusals


def test_chain_sweep_actually_covers():
    """QA 2026-07-25: this sweep used to be THEATRE and is now instrumented.

    In its first form it swallowed `CoveragePlanError` and returned, so 112 of
    its 128 targets asserted nothing -- and a mutation that corrupted the chain
    arithmetic (successors dropping 0 head frames instead of 1) left it 100%
    GREEN, because the corrupted builder raised and the test read that as a
    legitimate refusal. A sweep that passes when the thing it sweeps is broken
    is worse than no sweep. It now asserts a floor on plans ACTUALLY produced,
    so a regression that turns coverage into refusals fails here.
    """
    targets = list(range(9, 900, 7))
    plans, _refusals = _sweep(LTX8, targets)
    assert len(plans) >= 16, (
        "the bare 8n+1 ladder should still cover at least the 8m+1 targets; "
        "got %d plans from %d targets" % (len(plans), len(targets)))
    assert all(p.total_visible_frames in targets for p in plans)


def test_chain_sweep_with_tail_trim_covers_nearly_everything():
    """With trimming allowed the SAME ladder must cover essentially every beat.

    This is the sweep that has teeth: it forces the partitioner to produce a
    plan for almost every target rather than refusing, so the exact-sum
    assertion inside `_sweep` actually runs on hundreds of inputs.
    """
    trimmable = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                 allow_tail_trim=True, supports_multi_clip=True,
                                 continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    targets = list(range(9, 900, 7))
    plans, refusals = _sweep(trimmable, targets)
    assert refusals == 0, "tail trimming should cover every one of these beats"
    assert len(plans) == len(targets)


def test_jump_sweep_actually_covers():
    targets = list(range(9, 900, 11))
    plans, _refusals = _sweep(JUMPY, targets)
    assert len(plans) >= int(len(targets) * 0.9)
    assert all(s.drop_head == 0 for p in plans for s in p.segments)


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


# ---------------------------------------------------------------------------
# QA 2026-07-25 -- regressions for two defects an adversarial sweep found
# ---------------------------------------------------------------------------

def test_tail_trim_search_is_not_capped_at_one_quantum():
    """FALSE REFUSAL, found by an adversarial brute-force sweep.

    The tail-trim fallback used to try only ``extra in range(1, quantum + 1)``,
    assuming any shortfall could be bridged within one quantum step. That is
    false whenever ``count * min_frames`` overshoots the required total by more
    than one step -- routine just above ``max_frames`` when ``min_frames`` is
    large relative to the gap. The sweep found 832 beats that WERE coverable by
    trimming and were refused anyway. This is the smallest repro.
    """
    contract = fc.FrameContract(min_frames=4, max_frames=5, quantum=1,
                                allow_tail_trim=True, supports_multi_clip=True,
                                continuity=fc.CONTINUITY_NONE)
    plan = cp.partition_beat(6, contract)
    assert plan.total_visible_frames == 6
    assert [s.render_frames for s in plan.segments] == [4, 4]
    assert plan.segments[-1].trim_tail == 2
    cp.validate_coverage_plan(plan, contract)


def test_tail_trim_bridges_a_shortfall_larger_than_the_quantum():
    """The second repro from the same sweep: a 7-frame trim with quantum 2."""
    contract = fc.FrameContract(min_frames=10, max_frames=12, quantum=2,
                                allow_tail_trim=True, supports_multi_clip=True,
                                continuity=fc.CONTINUITY_NONE)
    plan = cp.partition_beat(13, contract)
    assert plan.total_visible_frames == 13
    assert plan.segments[-1].trim_tail > contract.quantum
    cp.validate_coverage_plan(plan, contract)


def test_no_false_refusal_across_a_trimmable_sweep():
    """Differential check: with trimming allowed, an exact cover exists for
    every target at or above ``min_frames``, so a refusal is a defect."""
    for min_f, q, max_f in ((4, 1, 5), (9, 8, 161), (10, 2, 12), (1, 1, 3)):
        contract = fc.FrameContract(min_frames=min_f, max_frames=max_f,
                                    quantum=q, allow_tail_trim=True,
                                    supports_multi_clip=True,
                                    continuity=fc.CONTINUITY_NONE)
        for target in range(min_f, max_f * 4):
            plan = cp.partition_beat(target, contract)
            assert plan.total_visible_frames == target, (min_f, q, max_f, target)
            for seg in plan.segments:
                assert contract.is_legal_length(seg.render_frames)
                assert seg.visible_frames >= 1


def test_discrete_partition_does_not_blow_up_exponentially():
    """DENIAL OF SERVICE, found by the same sweep.

    ``_discrete_partition`` recursed without memoization, so it explored up to
    ``len(menu) ** count`` states: 18 seconds at count=14 with a four-value
    menu, and still running past 20s at count=16. Because ``partition_beat``
    walks counts up to ``max_segments`` (64), an unsatisfiable target hung the
    calling thread forever instead of refusing -- a render node that never
    returns is worse than one that fails. Memoizing bounds it to
    ``total * count`` states.

    This test must complete in well under a second.
    """
    import time

    contract = fc.FrameContract(min_frames=96, max_frames=168,
                                discrete_durations=(96, 120, 144, 168),
                                allow_tail_trim=True, supports_multi_clip=True,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    started = time.perf_counter()
    try:
        cp.partition_beat(7501, contract, max_segments=64)
    except cp.CoveragePlanError:
        pass                     # refusing is fine; hanging is not
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, "discrete partition took %.1fs -- it is blowing up" % elapsed


def test_discrete_menu_still_partitions_correctly_after_memoization():
    contract = fc.FrameContract(min_frames=96, max_frames=168,
                                discrete_durations=(96, 120, 144, 168),
                                allow_tail_trim=True, supports_multi_clip=True,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    plan = cp.partition_beat(480, contract)
    assert plan.total_visible_frames == 480
    for seg in plan.segments:
        assert seg.render_frames in contract.discrete_durations
    cp.validate_coverage_plan(plan, contract)


def test_partition_terminates_quickly_for_a_long_beat():
    """A five-minute beat at 25fps against the ltx_8gb ladder."""
    import time

    trimmable = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                 allow_tail_trim=True, supports_multi_clip=True,
                                 continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    started = time.perf_counter()
    plan = cp.partition_beat(7500, trimmable)
    assert plan.total_visible_frames == 7500
    assert time.perf_counter() - started < 1.0


def test_join_mode_does_not_claim_SINGLE_for_an_uncoverable_target():
    """A THIRD math defect, found by a differential sweep after the first two.

    ``join_mode_for`` used to answer SINGLE whenever the target merely fit
    under ``max_frames`` and trimming was allowed -- without checking that a
    single legal render at or above the target actually EXISTS. A
    min=1 / quantum=2 / max=12 engine has the odd ladder 1,3,..,11, so a
    12-frame beat has no single legal render at all; the old code declared it
    SINGLE and then refused it, even though 11 + 1 covers it exactly as two
    jump-cut clips. 202 refusals in an 18k-call sweep traced to this one line.
    """
    contract = fc.FrameContract(min_frames=1, max_frames=12, quantum=2,
                                allow_tail_trim=True, supports_multi_clip=True,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    assert contract.smallest_legal_at_least(12) is None      # the premise
    assert cp.join_mode_for(contract, 12) == cp.JOIN_JUMP
    plan = cp.partition_beat(12, contract)
    assert plan.total_visible_frames == 12
    assert plan.segment_count >= 2
    cp.validate_coverage_plan(plan, contract)


def test_no_false_refusals_across_a_differential_sweep():
    """The standing guard for all three math defects at once.

    An independent reference (reachable-sum DP over the contract's own legal
    lengths) decides whether SOME exact cover exists; the partitioner must not
    refuse one that does. This is the check that caught the join-mode defect
    after the two the panel found, so it stays in the suite rather than living
    in a throwaway script.
    """
    import itertools

    def reachable(lengths, max_count):
        out = {0: {0}}
        for k in range(1, max_count + 1):
            out[k] = {b + L for b in out[k - 1] for L in lengths}
        return out

    max_count = 6
    checked = 0
    for min_f, q, max_f, trim in itertools.product(
            (1, 2, 4, 9), (1, 2, 3, 8), (5, 12, 25), (True, False)):
        if max_f < min_f:
            continue
        for continuity, drop in ((fc.CONTINUITY_STRICT_FIRST_FRAME, 1),
                                 (fc.CONTINUITY_SOFT_REFERENCE, 0)):
            contract = fc.FrameContract(
                min_frames=min_f, max_frames=max_f, quantum=q,
                allow_tail_trim=trim, supports_multi_clip=True,
                continuity=continuity)
            lengths = contract.legal_lengths()
            reach = reachable(lengths, max_count)
            smallest = min(lengths)
            for target in range(1, max_f * 2 + 3):
                checked += 1
                try:
                    plan = cp.partition_beat(target, contract)
                except cp.CoveragePlanError:
                    exists = False
                    for count in range(1, max_count + 1):
                        required = target + drop * (count - 1)
                        if required in reach[count]:
                            exists = True
                            break
                        if trim and any(
                                t > required and smallest - drop - (t - required) >= 1
                                for t in reach[count]):
                            exists = True
                            break
                    assert not exists, (
                        "refused a coverable beat: min=%d q=%d max=%d trim=%s "
                        "%s target=%d" % (min_f, q, max_f, trim, continuity, target))
                    continue
                assert plan.total_visible_frames == target
                for seg in plan.segments:
                    assert contract.is_legal_length(seg.render_frames)
                    assert seg.visible_frames >= 1
    assert checked > 2000
