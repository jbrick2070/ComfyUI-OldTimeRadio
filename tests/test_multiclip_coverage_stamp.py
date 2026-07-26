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


def _policy(engine="wan_i2v"):
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


def test_chunk_3b_is_behaviour_inert_today():
    """Every adapter is still single_only, so every beat is ONE clip.

    If this ever fails, an adapter opted in to multi-clip without its own live
    proof -- which is the failure mode the per-adapter declaration exists to
    prevent, and it must be caught here rather than on a render.
    """
    beats = _beats(n=3)
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {}, _policy())
    for shot in shots:
        plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])
        assert plan.join_mode == cp.JOIN_SINGLE
        assert plan.segment_count == 1
        assert plan.is_multi_clip is False
        # ...and the single segment renders exactly the beat, no trims.
        seg = plan.segments[0]
        assert seg.render_frames == shot["target_frame_count"]
        assert seg.drop_head == 0 and seg.trim_tail == 0


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


def test_a_beat_the_adapter_cannot_cover_is_TERMINAL_at_plan_time(monkeypatch):
    """The whole point: a beat past a single_only cap must SURFACE.

    Today that case is answered by ping-pong, loop-fill or a held frame -- the
    three silent coverage mechanisms this build removes. Once an adapter
    declares a real ceiling, exceeding it has to fail loudly at plan time,
    before any GPU work, rather than quietly producing padded video.
    """
    engine = vreg.get_engine("wan_i2v")
    capped = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    monkeypatch.setattr(engine, "frame_contract", lambda: capped, raising=False)

    beats = _beats()
    with pytest.raises(cp.CoveragePlanError):
        sl.build_execution_plan(beats, _budget(beats, frames=400), {}, _policy())


# ---------------------------------------------------------------------------
# The second boundary -- validated again after the wire
# ---------------------------------------------------------------------------

def _ledger_with(plan_dict, target=50, engine="wan_i2v"):
    return {
        "episode_id": "ep_cov",
        "video": {
            "shots": [{"shot_id": "shot_b001", "role": "character_video",
                       "group_id": "grp_character_video", "engine_id": engine,
                       "family": "", "target_frame_count": target,
                       "coverage_plan": plan_dict}],
        },
    }


def test_render_boundary_accepts_a_sound_plan():
    plan = cp.partition_beat(50, fc.SINGLE_ONLY)
    assert rd.assert_coverage_plans(_ledger_with(plan.to_dict())) == 1


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
    plan = cp.partition_beat(50, fc.SINGLE_ONLY)      # legal under an open ladder
    engine = vreg.get_engine("wan_i2v")
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
    plan = cp.partition_beat(50, fc.SINGLE_ONLY)
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
