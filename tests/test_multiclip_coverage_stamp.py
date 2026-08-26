"""Chunk 3b -- the CoveragePlan is stamped durably and validated at BOTH ends.

r3's ruling: a wire-only plan is useless. It must ride the durable ledger or it
cannot support replay, and the render boundary would have nothing to validate.
So ShotLock validates the plan when it builds it, and the render driver
validates the SAME plan again after it has crossed the wire.
"""

from __future__ import annotations

import json

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


def _beats(role="character_video", n=2):
    return [{"beat_id": "b%03d" % i, "role": role, "char_id": "c1", "dur_s": 2.0}
            for i in range(1, n + 1)]


def _budget(beats, frames=50):
    per = {b["beat_id"]: frames for b in beats}
    return {"per_beat": per, "total_frames": frames * len(beats), "warnings": []}


def _policy(engine="wan_ti2v"):
    return {
        "policy_version": 2,
        "video_models": {
            "announcer_video_model": {"engine_id": engine},
            "music_video_model": {"engine_id": engine},
            "character_video_model": {"engine_id": engine},
        },
    }


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------

def test_every_shot_carries_a_durable_coverage_plan():
    beats = _beats()
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    for shot in shots:
        plan = shot.get("coverage_plan")
        assert isinstance(plan, dict) and plan, shot["shot_id"]
        assert plan["target_visible_frames"] == shot["target_frame_count"]
        assert plan["segments"], "a plan with no segments covers nothing"


def test_the_stamp_is_json_serializable():
    """It rides the ledger, so it must survive the wire verbatim."""
    beats = _beats()
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    raw = json.dumps(shots[0]["coverage_plan"], ensure_ascii=True)
    assert cp.CoveragePlan.from_dict(json.loads(raw)) == \
        cp.CoveragePlan.from_dict(shots[0]["coverage_plan"])


def test_the_plan_snaps_to_the_adapters_REAL_ladder(monkeypatch):
    """The stamped plan renders a length the adapter can actually accept.

    REPLACES ``test_chunk_3b_is_behaviour_inert_today`` (chunk 7a,
    2026-07-26). That test asserted ``seg.render_frames == target_frame_count``
    and passed for one reason only: no adapter declared a ladder, so every
    engine resolved to SINGLE_ONLY, where every length is legal. It was
    measuring the ABSENCE of declarations, not a property of the build -- and
    it would have gone on passing while the plan promised wan_ti2v a 50-frame
    render that ``_floor_length`` was quietly snapping to 53 behind its back.
    Two policies over one state, which is the defect this build exists to
    remove.

    What is true now, and worth pinning: 50 visible frames on a 4n+1 ladder
    with min 33 has no exact single render, so the planner renders the
    SMALLEST LEGAL length at or above it (53) and trims the excess (3). The
    beat still gets EXACTLY the frames its audio asked for -- that is the
    invariant -- but the render length is now the adapter's number rather than
    the planner's wish.

    THE EXPECTED NUMBER IS A LITERAL, ON PURPOSE. Writing it as
    ``contract.smallest_legal_at_least(target)`` reads better and proves
    nothing: it sources the expectation from the very declaration under test,
    so deleting wan_ti2v's contract moves both sides together and the test goes
    on passing against an open ladder. A mutation run caught exactly that, in
    this test, before it shipped. 50 visible frames must render 53 -- say 53.
    """
    beats = _beats(n=3)
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    contract = fc.frame_contract_for(vreg.get_engine("wan_ti2v"))
    for shot in shots:
        assert shot["target_frame_count"] == 50           # the fixture's beat
        plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])
        assert plan.join_mode == cp.JOIN_SINGLE
        assert plan.segment_count == 1
        assert plan.is_multi_clip is False
        seg = plan.segments[0]
        # 33 + 4*5 = 53, the smallest length on wan_ti2v's ladder at or above 50.
        assert seg.render_frames == 53
        assert contract.is_legal_length(seg.render_frames)
        # ...and the VISIBLE frames still equal the beat exactly.
        assert plan.total_visible_frames == 50
        assert seg.drop_head == 0
        assert seg.trim_tail == 3


def test_a_beat_that_IS_a_legal_length_is_rendered_untrimmed():
    """The snap is smallest-legal-at-or-above, not blanket round-up.

    53 is on wan_ti2v's ladder (33 + 4*5), so a 53-frame beat must render
    exactly 53 with nothing trimmed. Without this, the test above would still
    pass if the planner always added a segment's worth of padding.
    """
    beats = _beats(n=1)
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=53), {}, _policy())
    plan = cp.CoveragePlan.from_dict(shots[0]["coverage_plan"])
    assert plan.segments[0].render_frames == 53
    assert plan.segments[0].trim_tail == 0


def test_a_zero_length_beat_gets_no_plan():
    beats = _beats(n=1)
    _groups, shots = sl.build_execution_plan(beats, _budget(beats, frames=0),
                                             {}, _policy())
    assert "coverage_plan" not in shots[0]


def test_an_unregistered_engine_gets_no_plan_rather_than_a_guess():
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats), {}, _policy(engine="not_a_real_engine"))
    assert all("coverage_plan" not in s for s in shots)


def test_a_beat_past_the_cap_is_COVERED_by_more_clips_not_refused():
    """The 8-second beat that proved the opt-in had to go (chunk 7a QA panel).

    This was ``test_a_beat_the_adapter_cannot_cover_is_TERMINAL_at_plan_time``
    and it monkeypatched a capped contract onto wan_ti2v to assert a refusal.
    Then wan_ti2v got a REAL cap of 177, and the refusal stopped being
    hypothetical: an ordinary 200-frame beat -- eight seconds of dialogue --
    raised ``CoveragePlanError`` out of ``build_execution_plan`` and took the
    whole episode's plan down with it, because the ceiling was declared while
    the multi-clip escape was still opt-in and nothing had opted in.

    No monkeypatch here. wan_ti2v's own declared ceiling, an ordinary beat
    length, and the plan that must come back.
    """
    beats = _beats(n=1)
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=200), {}, _policy())
    plan = cp.CoveragePlan.from_dict(shots[0]["coverage_plan"])
    contract = fc.frame_contract_for(vreg.get_engine("wan_ti2v"))
    assert 200 > contract.max_frames, "the fixture must exceed the real cap"
    assert plan.segment_count > 1
    assert plan.total_visible_frames == 200
    assert all(contract.is_legal_length(s.render_frames) for s in plan.segments)
    # wan_ti2v locks frame 0 to its init image, so this one earns a real chain
    # rather than a jump cut -- and every successor drops the duplicated frame.
    assert plan.join_mode == cp.JOIN_CHAIN
    assert [s.drop_head for s in plan.segments] == [0] + [1] * (plan.segment_count - 1)


def test_a_beat_that_NOTHING_can_cover_still_surfaces_at_plan_time(monkeypatch):
    """Splitting made most beats coverable; it did not make refusal impossible.

    An adapter that forbids the tail trim can only cover totals its ladder sums
    to exactly. When there is no such sum, the plan must still fail LOUD at
    plan time -- before any GPU work -- rather than fall back to ping-pong,
    loop-fill or a held frame, which are the three silent mechanisms this build
    exists to remove.
    """
    engine = vreg.get_engine("wan_ti2v")
    picky = fc.FrameContract(min_frames=9, max_frames=9, quantum=1,
                             allow_tail_trim=False)
    monkeypatch.setattr(engine, "frame_contract", lambda: picky, raising=False)

    beats = _beats()
    with pytest.raises(cp.CoveragePlanError):
        sl.build_execution_plan(beats, _budget(beats, frames=400), {}, _policy())


# ---------------------------------------------------------------------------
# The second boundary -- validated again after the wire
# ---------------------------------------------------------------------------

def _ledger_with(plan_dict, target=50, engine="wan_ti2v"):
    return {
        "episode_id": "ep_cov",
        "video": {
            "shots": [{"shot_id": "shot_b001", "role": "character_video",
                       "group_id": "grp_character_video", "engine_id": engine,
                       "family": "", "target_frame_count": target,
                       "coverage_plan": plan_dict}],
        },
    }


def _wan_plan(frames=50):
    """A plan built from wan_ti2v's OWN declared contract.

    Chunk 7a, 2026-07-26: these boundary tests used to partition against
    ``fc.SINGLE_ONLY`` while stamping ``engine_id="wan_ti2v"`` on the ledger.
    That worked only while wan_ti2v declared nothing, so the two contracts
    happened to be the same object. Now that it declares a real 4n+1 ladder,
    a SINGLE_ONLY plan is simply an ILLEGAL plan for that engine -- so a test
    named "accepts a sound plan" was building an unsound one and a test whose
    setup asserts the plan is legal for the picked engine was asserting
    something false. Build the plan from the engine the ledger names.
    """
    return cp.partition_beat(
        frames, fc.frame_contract_for(vreg.get_engine("wan_ti2v")))


def test_render_boundary_accepts_a_sound_plan():
    assert rd.assert_coverage_plans(_ledger_with(_wan_plan().to_dict())) == 1


def test_render_boundary_rejects_a_plan_that_drifts_from_its_beat():
    """A plan whose target no longer matches the beat's frame count.

    This is the case a second boundary exists for: the plan was valid when it
    was made, and something -- a replay, a hand edit, a stale revision --
    changed the beat under it.
    """
    plan = cp.partition_beat(50, fc.SINGLE_ONLY)
    ledger = _ledger_with(plan.to_dict(), target=77)
    with pytest.raises(rd.RenderError, match="drift from the beat audio"):
        rd.assert_coverage_plans(ledger)


def test_render_boundary_rejects_an_internally_broken_plan():
    bad = {"target_visible_frames": 50, "join_mode": "chain",
           "segments": [{"index": 0, "render_frames": 30, "drop_head": 0,
                         "trim_tail": 0},
                        {"index": 1, "render_frames": 30, "drop_head": 0,
                         "trim_tail": 0}]}
    with pytest.raises(rd.RenderError, match="cannot execute"):
        rd.assert_coverage_plans(_ledger_with(bad))


def test_render_boundary_rejects_a_plan_the_LIVE_contract_now_refuses(monkeypatch):
    """The contract can move after the plan was stamped.

    A version bump or a re-registered adapter must not silently execute a plan
    its CURRENT contract would reject -- so the second boundary re-validates
    against the live contract, not just the plan's own arithmetic.
    """
    plan = _wan_plan()                       # LEGAL under wan_ti2v as declared
    assert rd.assert_coverage_plans(_ledger_with(plan.to_dict())) == 1
    engine = vreg.get_engine("wan_ti2v")
    narrow = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    monkeypatch.setattr(engine, "frame_contract", lambda: narrow, raising=False)
    with pytest.raises(rd.RenderError, match="cannot execute"):
        rd.assert_coverage_plans(_ledger_with(plan.to_dict()))


def test_shots_without_a_plan_are_skipped_not_failed():
    ledger = _ledger_with({}, target=50)
    ledger["video"]["shots"][0].pop("coverage_plan")
    assert rd.assert_coverage_plans(ledger) == 0


def test_the_plan_is_validated_on_the_legacy_path_too():
    """A hand-built ledger that carries a plan is held to it as well."""
    bad = {"target_visible_frames": 50, "join_mode": "single",
           "segments": [{"index": 0, "render_frames": 30, "drop_head": 0,
                         "trim_tail": 0}]}
    ledger = _ledger_with(bad)                    # no roles_effective == legacy
    assert rd.frozen_route_from_ledger(ledger) == {}
    with pytest.raises(rd.RenderError, match="cannot execute"):
        rd.resolve_final_shot_engines(ledger)


def test_end_to_end_shotlock_to_render_boundary():
    beats = _beats()
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    ledger = {"video": {"shots": shots}}
    assert rd.assert_coverage_plans(ledger) == len(shots)


def test_the_legacy_path_validates_the_plan_against_the_FINAL_engine(monkeypatch):
    """Resolve the route FIRST, then hold the plan to it (2026-07-26, QA4).

    The legacy branch used to call ``assert_coverage_plans`` BEFORE
    ``apply_engine_override`` and the radio-host redirect, so a plan stamped
    for the PICKED engine was validated against that engine and then executed
    by a DIFFERENT one -- the ordering defect chunk 1c closed for the still
    spine, reintroduced one contract further down inside the very function
    that closed it. Checking early is worse than not checking: it writes
    COVERAGE PLANS OK for routing that no longer holds.
    """
    plan = _wan_plan()
    ledger = _ledger_with(plan.to_dict())        # no roles_effective == legacy
    assert rd.frozen_route_from_ledger(ledger) == {}
    # The plan IS legal for the picked engine -- so an early check passes, and
    # that is exactly what made the old order look correct.
    assert rd.assert_coverage_plans(ledger) == 1

    forced = vreg.get_engine("ltx_video")
    narrow = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    monkeypatch.setattr(forced, "frame_contract", lambda: narrow, raising=False)
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=ltx_video")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)

    with pytest.raises(rd.RenderError, match="cannot execute"):
        rd.resolve_final_shot_engines(ledger)
    # The force ran first: the refusal names the engine that actually renders.
    assert ledger["video"]["shots"][0]["engine_id"] == "ltx_video"


# ---------------------------------------------------------------------------
# B4 (2026-07-27): ShotRow is a CLOSED model (extra="forbid"), so it has to
# actually describe what the producers stamp. It was missing eight fields, so
# ShotRow(**real_row) raised on EVERY real ledger -- which is why the "live
# safety net" other docs cite could not validate one shipped episode.
# ---------------------------------------------------------------------------

def _real_rows(frames=50):
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=frames), {}, _policy())
    assert shots, "the producer emitted no shots -- this test proves nothing"
    return shots


def test_every_real_shotlock_row_validates_through_ShotRow():
    """Driven through the REAL producer on purpose: a hand-built row would
    only ever prove the fields the test itself chose to write, which is how
    this stayed green while being wrong."""
    from nodes._otr_video_engines.schemas import ShotRow

    for row in _real_rows():
        ShotRow(**row)


def test_a_multi_clip_row_validates_too():
    """A beat pushed past its adapter's cap really does partition here, so
    this exercises a row whose coverage_plan carries several segments."""
    from nodes._otr_video_engines.schemas import ShotRow

    rows = _real_rows(frames=400)
    assert any(len(r["coverage_plan"]["segments"]) > 1 for r in rows), \
        "nothing split -- the multi-clip case is not being exercised"
    for row in rows:
        ShotRow(**row)


def test_a_row_carrying_real_jump_still_requests_validates():
    """jump_still_requests is the field the bug report's list did not mention
    at all. It is NOT reachable through the fixture above: every multi-clip
    adapter here declares strict_first_frame, so its plan is a CHAIN and
    jump_still_requests is empty by contract. Drive the real minter over a
    real JUMP plan instead of writing the rows by hand."""
    from nodes._otr_video_engines.schemas import ShotRow

    jump_contract = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    plan = cp.partition_beat(400, jump_contract)
    assert plan.join_mode == "jump", plan.join_mode
    requests = cp.jump_still_requests(plan, "b001", role="character_video",
                                      engine_id="ltx_video", char_id="c1")
    assert requests, "the minter produced nothing -- nothing is being proven"

    row = dict(_real_rows()[0])
    row["jump_still_requests"] = [dict(r) for r in requests]
    assert len(ShotRow(**row).jump_still_requests) == len(requests)


def test_the_whole_video_section_validates_end_to_end():
    """The boundary other docs promise exists: model_validate over
    ledger['video']. Before B4 it hard-failed on the first shot."""
    from nodes._otr_video_engines.schemas import VideoLedgerSection

    beats = _beats()
    groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    section = VideoLedgerSection(execution_groups=groups, shots=shots)
    assert len(section.shots) == len(shots)
    assert len(section.execution_groups) == len(groups)


def test_the_motion_clause_pass_output_is_accepted_too():
    """motion_clause is stamped onto the same row by a LATER pass. Ask that
    pass's own producer for the object rather than transcribing one."""
    from nodes import _otr_motion_clause as mc
    from nodes._otr_video_engines.schemas import ShotRow

    clause = mc._clause_obj("", model=mc.GENERATED_MODEL_UNSET, fallback=True,
                            char_id="c1", beat_id="b001", dialogue="hello")
    assert isinstance(clause, dict) and clause
    row = dict(_real_rows()[0])
    row["motion_clause"] = clause
    assert ShotRow(**row).motion_clause == clause


def test_shotrow_covers_every_key_the_producers_stamp():
    """Mechanical drift guard. B4's field list was derived from the producers
    rather than from the bug report -- which named a beat_id no producer
    writes, and missed two that are written. A newly stamped key must fail
    HERE, by name, instead of at whichever boundary first validates."""
    from nodes._otr_video_engines.schemas import ShotRow

    produced = set()
    for row in _real_rows() + _real_rows(frames=400):
        produced |= set(row)

    declared = set(ShotRow.model_fields)
    assert produced <= declared, sorted(produced - declared)
    # The report's own list was wrong in both directions; pin the correction.
    assert "beat_id" not in produced
    assert {"role", "char_id", "coverage_plan"} <= produced


def test_absence_is_preserved_for_the_fields_that_encode_it():
    """coverage_plan / coverage_contract / motion_clause default to None, not
    to an empty container. 'Never stamped' and 'stamped empty' mean different
    things: an unregistered engine gets no plan, and a coverage_contract is
    written only when the tier ceiling narrowed something, which
    assert_coverage_plans re-derives and compares."""
    from nodes._otr_video_engines.schemas import ShotRow

    bare = ShotRow(shot_id="s1")
    assert bare.coverage_plan is None
    assert bare.coverage_contract is None
    assert bare.motion_clause is None
    assert bare.jump_still_requests == []
    assert bare.dur_s is None


def test_shotrow_is_still_a_CLOSED_model():
    """Everything above would pass for the wrong reason if ShotRow stopped
    forbidding extras -- completing the field list only means something while
    an unknown key is still an error. The closedness IS the contract that made
    the missing fields a bug rather than a cosmetic gap."""
    from nodes._otr_video_engines.schemas import ShotRow

    with pytest.raises(Exception) as excinfo:
        ShotRow(shot_id="s1", not_a_field_any_producer_writes=1)
    assert "not_a_field_any_producer_writes" in str(excinfo.value)
