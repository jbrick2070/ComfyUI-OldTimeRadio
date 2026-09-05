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
                        continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)

JUMPY = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                         continuity=fc.CONTINUITY_SOFT_REFERENCE)

VEO = fc.FrameContract(min_frames=100, max_frames=200,
                       discrete_frames=(100, 150, 200),
                       allow_tail_trim=True,
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
                                 allow_tail_trim=True,
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
                                 allow_tail_trim=True,
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
                                 allow_tail_trim=True,
                                 continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    plan = cp.partition_beat(target, trimmable)
    assert plan.total_visible_frames == target
    assert sum(s.trim_tail for s in plan.segments) > 0
    cp.validate_coverage_plan(plan, trimmable)


# ---------------------------------------------------------------------------
# Join-mode selection comes from the adapter's own declaration
# ---------------------------------------------------------------------------

def test_a_beat_that_FITS_never_splits():
    """Splitting is for beats that overflow, not a thing that happens to
    everyone now that the opt-in is gone. 17 is on this ladder, so one clip."""
    plan = cp.partition_beat(17, fc.FrameContract(min_frames=9, max_frames=161,
                                                  quantum=8))
    assert plan.join_mode == cp.JOIN_SINGLE
    assert plan.segment_count == 1
    assert plan.is_multi_clip is False


def test_an_engine_over_its_cap_SPLITS_rather_than_padding():
    """It must NOT silently ping-pong, loop-fill or hold a frame -- those are
    exactly the three silent coverage mechanisms this build removes.

    REWRITTEN chunk 7a (2026-07-26). This was
    ``test_single_only_engine_over_its_cap_is_terminal`` and asserted that a
    beat past the cap RAISED, because an engine had to opt in to multi-clip and
    this one had not. The operator deleted the opt-in -- "everything gets an
    equal term" -- so the answer to a beat past the cap is no longer a refusal,
    it is the second clip that multi-clip coverage exists to provide.

    The invariant the old test was really defending is unchanged and is what is
    asserted here: the beat gets its EXACT frame count out of real rendered
    clips. No padding, no held frame. Only the mechanism changed.
    """
    contract = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                allow_tail_trim=True,
                                continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    plan = cp.partition_beat(400, contract)
    assert plan.segment_count > 1
    assert plan.total_visible_frames == 400
    assert all(contract.is_legal_length(s.render_frames) for s in plan.segments)


def test_a_chained_8n1_ladder_needs_the_TAIL_TRIM_to_reach_400():
    """Why the test above carries ``allow_tail_trim=True``, written down.

    The first draft of it omitted the flag and refused, which looked like a bug
    in the partitioner and was not. On an 8n+1 ladder every segment is
    ``9 + 8a``, and under CHAIN covering ``t`` visible frames with ``k``
    segments requires rendering ``t + k - 1``. Setting those equal:
    ``9k + 8S = 399 + k``, i.e. ``8(k + S) = 399`` -- and 399 is not divisible
    by 8, for any segment count. So 400 visible frames has NO exact chained
    cover on this ladder, and the refusal was correct.

    The tail trim is what makes it coverable: render one step past and drop the
    remainder. Pinning the refusal keeps the fix honest -- the flag is buying a
    real capability, not papering over arithmetic nobody checked.
    """
    no_trim = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                               allow_tail_trim=False,
                               continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    with pytest.raises(cp.CoveragePlanError, match="no exact multi-clip cover"):
        cp.partition_beat(400, no_trim)


def test_a_beat_no_amount_of_splitting_can_cover_is_STILL_terminal():
    """The refusal did not go away -- it moved to where it is still true.

    An engine that forbids the tail trim can only cover totals its ladder sums
    to exactly. Give it a ladder whose every entry is odd and ask for an even
    total that no combination reaches, and there is genuinely no honest plan --
    so it raises rather than shipping a clip that drifts from the beat audio.
    That is the property the pre-7a test was reaching for, stated in terms that
    survive the opt-in's removal.
    """
    picky = fc.FrameContract(min_frames=9, max_frames=9, quantum=1,
                             allow_tail_trim=False,
                             continuity=fc.CONTINUITY_SOFT_REFERENCE)
    # Only 9 is legal, so reachable totals are 9, 18, 27... 400 is not one.
    with pytest.raises(cp.CoveragePlanError, match="no exact multi-clip cover"):
        cp.partition_beat(400, picky)


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
        assert seg.render_frames in VEO.discrete_frames
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


def test_multi_segment_plan_rejected_when_a_SEGMENT_is_off_the_ladder():
    """REPLACES ``test_multi_segment_plan_rejected_for_single_only_contract``.

    That test built a 2-segment plan and validated it against a contract with
    no opt-in, expecting "not opted in". With the opt-in deleted there is no
    such refusal -- and the check it was standing in for is still here and is
    the one that matters: a plan is rejected when a SEGMENT asks the adapter
    for a length the adapter cannot render. Same boundary, real reason.
    """
    plan = cp.partition_beat(169, LTX8)
    assert plan.segment_count > 1
    narrower = fc.FrameContract(min_frames=9, max_frames=161, quantum=16,
                                continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    with pytest.raises(cp.CoveragePlanError, match="cannot accept"):
        cp.validate_coverage_plan(plan, narrower)


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
                                allow_tail_trim=True,
                                continuity=fc.CONTINUITY_NONE)
    plan = cp.partition_beat(6, contract)
    assert plan.total_visible_frames == 6
    assert [s.render_frames for s in plan.segments] == [4, 4]
    assert plan.segments[-1].trim_tail == 2
    cp.validate_coverage_plan(plan, contract)


def test_tail_trim_bridges_a_shortfall_larger_than_the_quantum():
    """The second repro from the same sweep: a 7-frame trim with quantum 2."""
    contract = fc.FrameContract(min_frames=10, max_frames=12, quantum=2,
                                allow_tail_trim=True,
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
                                discrete_frames=(96, 120, 144, 168),
                                allow_tail_trim=True,
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
                                discrete_frames=(96, 120, 144, 168),
                                allow_tail_trim=True,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    plan = cp.partition_beat(480, contract)
    assert plan.total_visible_frames == 480
    for seg in plan.segments:
        assert seg.render_frames in contract.discrete_frames
    cp.validate_coverage_plan(plan, contract)


def test_partition_terminates_quickly_for_a_long_beat():
    """A five-minute beat at 25fps against the ltx_8gb ladder."""
    import time

    trimmable = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                 allow_tail_trim=True,
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
                                allow_tail_trim=True,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    assert contract.smallest_legal_at_least(12) is None      # the premise
    assert cp.join_mode_for(contract, 12) == cp.JOIN_JUMP
    plan = cp.partition_beat(12, contract)
    assert plan.total_visible_frames == 12
    assert plan.segment_count >= 2
    cp.validate_coverage_plan(plan, contract)


# ---------------------------------------------------------------------------
# W1 (2026-07-29) -- THE PARTITIONER OVER-SEGMENTS A TRIMMABLE BEAT
# ---------------------------------------------------------------------------
#
# THE MECHANISM, written down so the fix cannot be mistaken for a 184-shaped
# patch. ``partition_beat`` ran TWO separate walks over the segment count: the
# first tried an EXACT cover at every count from 2 up to ``max_segments``, and
# only after that whole walk failed did a second walk start again at count 2
# looking for a TRIMMED cover. So an exact cover at a HIGH count always beat a
# trimmed cover at a LOW one.
#
# That contradicts this module's own stated contract -- "Solving for the
# segment COUNT first always finds an exact partition when one exists" -- and
# it is not what anyone wants from the result: four model loads, four renders
# and four seams instead of two, for a beat whose two-clip cover was legal all
# along and differs only by discarding two frames the audio never asked for.
#
# The correct order is per COUNT: exact first, then a permitted TRIMMED total,
# then advance to the next count. Trimming is not a last resort to be reached
# only once the ladder is exhausted -- it is part of what a given count can do,
# and the adapter said so itself by declaring ``allow_tail_trim``.


#: HuMo's real ladder, transcribed from the DOCUMENTED literals in
#: ``eng_humo.py`` (33 is the legacy floor, 177 the last empirically verified
#: ceiling at 480x832 fp8, 4n+1 stride) rather than imported from it. A test
#: that reads the value it is checking cannot notice the value changing, so
#: the literals live here and ``test_the_pinned_humo_ladder_is_still_what_the
#: _engine_declares`` is what fails if the engine ever moves.
HUMO = fc.FrameContract(min_frames=33, max_frames=177, quantum=4,
                        native_fps=25, allow_tail_trim=True,
                        continuity=fc.CONTINUITY_SOFT_REFERENCE)


def test_the_pinned_humo_ladder_is_still_what_the_engine_declares():
    """The premise of every vector below, checked against the real adapter.

    Imported inside the test on purpose: this module is otherwise registry-free
    and stays that way. The partitioner is pure, and its tests should keep
    running without a registry -- but a pinned vector whose contract has
    silently drifted is a vector that proves nothing, so the premise gets
    checked exactly once, here.
    """
    import nodes._otr_video_engines  # noqa: F401  -- populate the registry
    from nodes._otr_video_engines import frame_contract as live_fc
    from nodes._otr_video_engines import registry as vreg

    # Checked against `humo_1.7B`, not `humo`. The pinned vectors below describe
    # the 33-177/q4 HuMo ladder, and that ladder is still exactly what the 1.7B
    # tiers declare. The two 14B routes were capped to a shared ceiling on
    # 2026-08-02 (the orientation-specific 49-vs-177 split had no architectural
    # basis and cited a receipt absent from this repo), so `humo` no longer
    # carries 177 -- but the partitioner behaviour these vectors pin is a
    # property of the LADDER, not of which engine happens to declare it.
    declared = live_fc.frame_contract_for(vreg.get_engine("humo_1.7B"))
    assert declared.min_frames == HUMO.min_frames == 33
    assert declared.max_frames == HUMO.max_frames == 177
    assert declared.quantum == HUMO.quantum == 4
    assert declared.allow_tail_trim is True
    assert declared.continuity == HUMO.continuity

    # And the 14B pair shares ONE ceiling, keyed on the checkpoint rather than
    # on the orientation -- the invariant that replaced the split.
    from nodes._otr_video_engines import eng_humo as _eh
    for name in ("humo", "humo_14B_169"):
        capped = live_fc.frame_contract_for(vreg.get_engine(name))
        assert capped.max_frames == _eh._HUMO_14B_SAFE_RENDER_FRAMES
        assert capped.min_frames == 33 and capped.quantum == 4


def test_a_184_frame_humo_beat_is_TWO_segments_not_four():
    """THE PINNED CASE (kibitz r3/r4 2026-07-28), live on the 2026-07-28
    engine-coverage campaign, where a 184-frame HuMo beat planned FOUR clips.

    The arithmetic, so these numbers are checkable rather than merely asserted.
    HuMo jump-cuts (its reference is a soft identity hint, not a first-frame
    lock), so ``drop`` is 0 and ``c`` segments must render exactly ``target``
    frames between them. On a ``33 + 4k`` ladder, ``c`` segments reach totals
    congruent to ``33c`` modulo 4 -- that is, to ``c`` modulo 4. So an EXACT
    cover of 184 (which is 0 mod 4) requires a count that is 0 mod 4, and the
    first such count is FOUR. A two-segment cover needs a total of 186, which
    is on the ladder as ``153 + 33`` and overshoots by two: trim two.

    Four renders and four model loads to avoid discarding two frames. That is
    the defect, and 153 + 33 is the plan that was available the whole time.
    """
    plan = cp.partition_beat(184, HUMO)
    assert plan.join_mode == cp.JOIN_JUMP
    assert plan.segment_count == 2
    assert [s.render_frames for s in plan.segments] == [153, 33]
    assert [s.drop_head for s in plan.segments] == [0, 0]
    assert [s.trim_tail for s in plan.segments] == [0, 2]
    assert plan.total_visible_frames == 184
    cp.validate_coverage_plan(plan, HUMO)


@pytest.mark.parametrize("target", list(range(185, 241)))
def test_every_target_from_185_to_240_covers_in_exactly_two_segments(target):
    """The second pinned vector: the whole band just above HuMo's ceiling.

    177 < target <= 240 no longer fits one render, and two are plainly enough
    -- two segments make 354 render frames available against a 240-frame ask.
    Before the fix, three of every four targets in this band planned three,
    four or five segments, decided purely by which count the bare ladder
    happened to reach an exact cover at first. A single vector at 184 could be
    satisfied by a special case; a contiguous band cannot.
    """
    plan = cp.partition_beat(target, HUMO)
    assert plan.segment_count == 2, (
        "target %d planned %d segments: %r"
        % (target, plan.segment_count,
           [s.render_frames for s in plan.segments]))
    assert plan.total_visible_frames == target
    cp.validate_coverage_plan(plan, HUMO)


def _fewest_legal_multiclip_cover(target, contract, drop, max_count):
    """INDEPENDENT REFERENCE: ``(count, last_render, trim)``, or ``None``.

    Returns the COMPOSITION, not just the count. A QA lens proved why: an
    earlier version of this reference returned the count alone, and a mutant
    that reversed ``_ladder_partition``'s fill order -- putting the leftover
    in the FIRST segment instead of the last -- passed it with 0 mismatches
    over 27,954 plans, because the count never changes when you shuffle the
    same lengths. The count is the cheap half of the answer.

    Re-derived from the ladder's arithmetic as documented on
    :class:`FrameContract` and :func:`_candidate_totals`, NOT by calling the
    partitioner -- a reference that asks the code under test what it thinks is
    a test that verifies a thing it also constructs, which this build treats as
    decorative until proven otherwise.

    For a count ``c``: the segments render ``target + drop*(c-1)`` frames
    between them, the reachable totals are ``c*min + quantum*A`` bounded by
    ``c*max``, and the cheapest legal one is the smallest such total at or
    above what is required. The ladder filler pushes every segment to the
    ceiling in order, so the LAST segment carries whatever steps the earlier
    ones could not absorb -- and it is the last segment that pays the trim, so
    it must still contribute a visible frame afterwards.
    """
    min_f = int(contract.min_frames)
    q = int(contract.quantum)
    max_f = int(contract.max_frames)
    steps_each = (max_f - min_f) // q
    if steps_each < 0:
        return None
    for count in range(2, max_count + 1):
        required = target + drop * (count - 1)
        base, span = count * min_f, count * max_f
        if required > span:
            continue
        total = (base if required <= base
                 else base + -(-(required - base) // q) * q)
        if total > span:
            continue
        steps = (total - base) // q
        if steps > count * steps_each:
            continue
        trim = total - required
        if trim and not contract.allow_tail_trim:
            continue
        last = min_f + max(0, steps - (count - 1) * steps_each) * q
        if last - drop - trim < 1:
            continue
        return count, last, trim
    return None


def test_the_partitioner_uses_THE_FEWEST_LEGAL_SEGMENTS():
    """THE MECHANISM, pinned as a property over a grid rather than a vector.

    A segment is a model load, a render and a seam. Whenever a lower count
    could have covered the beat legally -- exactly OR with a trim the adapter
    already permits -- taking a higher one buys nothing and costs all three.

    This is the test that makes W1 a fix instead of a special case: the 184
    vector above would be satisfied by hard-coding 184, and this will not be.
    Every mismatch it reports names the contract, the target, the count taken
    and the count that would have done.
    """
    import itertools

    checked = refused = single = 0
    for min_f, q, max_f in itertools.product((1, 4, 9, 33), (1, 2, 4, 8),
                                             (12, 25, 41, 177)):
        if max_f < min_f:
            continue
        for continuity, drop in ((fc.CONTINUITY_STRICT_FIRST_FRAME, 1),
                                 (fc.CONTINUITY_SOFT_REFERENCE, 0)):
            for trim_ok in (True, False):
                contract = fc.FrameContract(
                    min_frames=min_f, max_frames=max_f, quantum=q,
                    allow_tail_trim=trim_ok, continuity=continuity)
                for target in range(1, max_f * 3):
                    try:
                        plan = cp.partition_beat(target, contract,
                                                 max_segments=8)
                    except cp.CoveragePlanError:
                        refused += 1
                        continue
                    if plan.segment_count < 2:
                        single += 1
                        continue
                    where = ("min=%d q=%d max=%d trim=%s %s target=%d"
                             % (min_f, q, max_f, trim_ok, continuity, target))
                    lengths = [s.render_frames for s in plan.segments]
                    fewest = _fewest_legal_multiclip_cover(
                        target, contract, drop, 8)
                    checked += 1
                    assert fewest is not None, (
                        "%s -- planned %r but the reference says nothing "
                        "covers it" % (where, lengths))
                    count, last, trim = fewest
                    assert count == plan.segment_count, (
                        "%s -- planned %d segments %r but %d would have "
                        "covered it" % (where, plan.segment_count, lengths,
                                        count))
                    # COMPOSITION, not just the count -- see the reference's
                    # docstring for the mutant that made this necessary.
                    assert lengths[-1] == last, (
                        "%s -- last segment renders %d, reference says %d"
                        % (where, lengths[-1], last))
                    assert plan.segments[-1].trim_tail == trim, (
                        "%s -- trims %d, reference says %d"
                        % (where, plan.segments[-1].trim_tail, trim))
                    # ...and the DOCUMENTED fill order: `_ladder_partition`
                    # "fills each segment toward the ceiling in order, which
                    # keeps the plan deterministic and puts the short segment
                    # last". Asserted against the docstring, not against the
                    # reference, so it does not share its arithmetic.
                    assert lengths == sorted(lengths, reverse=True), (
                        "%s -- segments are not longest-first: %r"
                        % (where, lengths))
                    assert all(s.trim_tail == 0 for s in plan.segments[:-1]), (
                        "%s -- only the LAST segment may be trimmed: %r"
                        % (where, [s.trim_tail for s in plan.segments]))

    # A FLOOR THAT CAN ACTUALLY NOTICE A COLLAPSE. The old floor was 500
    # against a measured 27,954 -- 56x slack, so a regression that turned 90%
    # of this grid into refusals would still have sailed through. A QA lens
    # demonstrated exactly that: a mutant that made `_candidate_totals` round
    # the wrong way cut `checked` by 29% (to 19,770) and the old assertion
    # passed by 39x. The floor now sits just under the real value, and the
    # refusal count is asserted too so coverage cannot drain into the `except`.
    assert checked >= 27954, (
        "grid produced only %d multi-clip plans (measured 27954 at the time "
        "this was written) -- coverage has collapsed, or the grid moved and "
        "this floor needs re-measuring" % checked)
    assert refused <= 12000, (
        "%d targets refused -- coverage is draining into the except branch"
        % refused)


def test_the_segment_count_ceiling_is_INCLUSIVE():
    """``max_segments`` is a count the partitioner may actually use.

    Added because a mutation round SURVIVED: turning the walk's
    ``range(2, max_segments + 1)`` into ``range(2, max_segments)`` broke
    nothing in the suite, so the boundary was pinned by nobody and a
    silently-narrowed search would have looked exactly like a genuine refusal.

    17 * 3 == 51 on this ladder with no trim allowed, so 51 needs EXACTLY
    three segments: at two it is honestly uncoverable, at three it is exact.
    That makes the pair a boundary rather than a single-sided assertion.
    """
    contract = fc.FrameContract(min_frames=9, max_frames=17, quantum=8,
                                allow_tail_trim=False,
                                continuity=fc.CONTINUITY_SOFT_REFERENCE)
    with pytest.raises(cp.CoveragePlanError, match="no exact multi-clip"):
        cp.partition_beat(51, contract, max_segments=2)

    plan = cp.partition_beat(51, contract, max_segments=3)
    assert plan.segment_count == 3
    assert [s.render_frames for s in plan.segments] == [17, 17, 17]
    assert plan.total_visible_frames == 51


def test_a_trimmed_plan_never_starves_its_LAST_segment():
    """The trim guard's contract, asserted as a property.

    RECORDED HONESTLY: two mutants of the guard itself
    (``lengths[-1]`` -> ``lengths[0]``, and deleting the guard outright)
    SURVIVED a mutation round, and the reason is not a missing test -- it is
    that the guard cannot fire. ``_candidate_totals`` yields the SMALLEST
    reachable total at or above what is required, so the overshoot it hands
    back is never big enough to eat a whole final segment. An exhaustive
    sweep over every ladder with ``min<=10, quantum in {1,2,3,4,8}, max<=80``
    and nine discrete menus (including the real Veo and Pixverse ones) found
    ZERO cases where the guard rejects a candidate.

    So the guard is a backstop against a future ``_candidate_totals`` that
    offers looser candidates, and the honest test is not a contrived call
    that fakes one -- that would test the fake. It is this: every trimmed
    plan the partitioner actually emits leaves its last segment contributing
    real frames. If the candidate policy ever loosens, this is what notices.
    """
    seen_trims = 0
    for min_f, q, max_f in ((4, 1, 5), (9, 8, 161), (10, 2, 12), (33, 4, 177),
                            (1, 2, 12)):
        for continuity, drop in ((fc.CONTINUITY_STRICT_FIRST_FRAME, 1),
                                 (fc.CONTINUITY_SOFT_REFERENCE, 0)):
            contract = fc.FrameContract(min_frames=min_f, max_frames=max_f,
                                        quantum=q, allow_tail_trim=True,
                                        continuity=continuity)
            for target in range(1, max_f * 3):
                try:
                    plan = cp.partition_beat(target, contract, max_segments=8)
                except cp.CoveragePlanError:
                    continue
                last = plan.segments[-1]
                assert last.visible_frames >= 1, (
                    "min=%d q=%d max=%d %s target=%d -- last segment renders "
                    "%d, drops %d, trims %d and contributes NOTHING"
                    % (min_f, q, max_f, continuity, target,
                       last.render_frames, last.drop_head, last.trim_tail))
                assert last.drop_head == (drop if plan.segment_count > 1 else 0)
                if last.trim_tail:
                    seen_trims += 1
    assert seen_trims > 200, (
        "only %d trimmed plans in the sweep -- this property is not being "
        "exercised" % seen_trims)


def test_one_segment_plan_owing_a_trim_needs_the_coverage_executor():
    """A ONE-segment plan is still planned work when its length was rounded up.

    The router asked ``is_multi_clip``, which is a different question. A beat
    whose audio-derived length misses the engine's ladder gets a single segment
    with ``render_frames > visible_frames`` and the surplus recorded in
    ``trim_tail`` -- and the trim belongs to the coverage assembler, which only
    runs on the coverage path. Routed by segment count, those beats rendered
    their extra frames and kept every one, so the clip outran its own audio.

    Not exotic: `ltx_audio_in` at 442 renders 449, `humo` at 100 renders 101.
    """
    from nodes._otr_video_engines import coverage_plan as cp
    from nodes._otr_video_engines import frame_contract as fc

    # 9 + 8k ladder: 442 has no legal rung, 449 is the next one up.
    ladder = fc.FrameContract(min_frames=9, max_frames=497, quantum=8,
                              native_fps=25, allow_tail_trim=True)
    plan = cp.partition_beat(442, ladder)
    assert plan.segment_count == 1
    assert not plan.is_multi_clip, "still one clip -- that was never the issue"
    seg = plan.segments[0]
    assert (seg.render_frames, seg.visible_frames, seg.trim_tail) == (449, 442, 7)
    assert plan.requires_coverage_execution, (
        "a single segment rendering 449 frames against 442 frames of audio owes "
        "a 7-frame trim; sending it down the historical path keeps the surplus")

    # An EXACT fit owes nothing, so it must still take the historical path --
    # this fix must not reroute beats whose behaviour was already correct.
    exact = cp.partition_beat(449, ladder)
    assert exact.segment_count == 1
    assert exact.segments[0].trim_tail == 0
    assert not exact.requires_coverage_execution

    # And multi-clip is unconditionally coverage work, trim or no trim.
    multi = cp.partition_beat(900, ladder)
    assert multi.segment_count > 1
    assert multi.requires_coverage_execution
