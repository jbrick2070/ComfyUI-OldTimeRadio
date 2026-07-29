"""THE PATH THAT WAS DEAD CODE UNTIL CHUNK 7A, EXERCISED WITH REAL ENGINES.

Chunks 3-6 built the multi-segment coverage path and tested every piece of it
with STUB engines, because no real adapter could reach it: every engine
resolved to ``SINGLE_ONLY`` and ``join_mode_for`` returned SINGLE before it
read anything else. Chunk 7a gave all 31 adapters real ladders and deleted the
opt-in, which made that path live for the first time -- and a QA panel found it
refused every real beat it was handed.

The defect was the same shape three times over: the MINT and the DEMAND asked
different questions about the same state.

* ``coverage_plan.jump_still_requests`` mints nothing for a CHAIN plan, because
  a chained successor begins on its predecessor's terminal frame -- as
  ``render_beat_coverage``'s own loop says in as many words.
* ``otr_shot_lock._stamp_coverage_plan`` mints nothing for a lane the still
  spine never asks a scene still of.
* ``render_driver.jump_segment_still_path`` demanded one for EVERY segment >= 1
  regardless, and raised "NO FALLBACK" when it was missing.

Result: an ordinary 8-second beat on any of the four chain-capable local
engines, and every HuMo beat past its cap, died building the request for
segment 1 -- after segment 0 had already rendered on the GPU.

These tests use REAL registered engines and REAL declared ladders. No stubs, no
monkeypatched contracts. That is the whole point: the stubs all passed.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


def _shot_for(engine, frames, role="character_video"):
    """Plan one beat of ``frames`` frames through the REAL shot lock."""
    beats = [{"beat_id": "b001", "role": role, "char_id": "c1",
              "dur_s": frames / 25.0}]
    budget = {"per_beat": {"b001": frames}, "total_frames": frames,
              "warnings": []}
    policy = {"policy_version": 2, "video_models": {
        "announcer_video_model": {"engine_id": engine},
        "music_video_model": {"engine_id": engine},
        "character_video_model": {"engine_id": engine}}}
    _groups, shots = sl.build_execution_plan(beats, budget, {}, policy)
    return shots[0]


# ---------------------------------------------------------------------------
# The chain lanes: a successor owns NO still and must not be asked for one
# ---------------------------------------------------------------------------

CHAIN_ENGINES = ["wan_i2v", "wan_ti2v", "ltx_8gb", "ltx_video"]


@pytest.mark.parametrize("engine", CHAIN_ENGINES)
def test_a_chained_successor_needs_no_still_of_its_own(engine):
    """The exact beat that died: past the cap, on a real chain-capable engine.

    ``jump_segment_still_path`` must answer None for a chain segment. It begins
    on the terminal frame ``render_beat_coverage`` extracts from its
    predecessor and writes into ``asset_refs["init_image"]`` -- asking the
    still spine for a minted image would be asking for something the plan
    deliberately never ordered.
    """
    contract = fc.frame_contract_for(vreg.get_engine(engine))
    frames = int(contract.max_frames) + 60          # comfortably past the cap
    shot = _shot_for(engine, frames)
    plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])

    assert plan.join_mode == cp.JOIN_CHAIN, engine
    assert plan.segment_count > 1, engine
    assert "jump_still_requests" not in shot, (
        "%s built a CHAIN plan and still minted jump stills for it" % engine)
    for index in range(1, plan.segment_count):
        assert rd.jump_segment_still_path({}, shot, index) is None, (
            "%s segment %d was asked for a still a chain never orders"
            % (engine, index))


@pytest.mark.parametrize("engine", CHAIN_ENGINES)
def test_segment_zero_still_resolves_the_beats_own_still(engine):
    """Segment 0 is unchanged by any of this -- it renders from the beat's own
    scene still, which every pre-existing branch already resolves. Pinned so a
    fix to the successors cannot quietly move segment 0 too."""
    contract = fc.frame_contract_for(vreg.get_engine(engine))
    shot = _shot_for(engine, int(contract.max_frames) + 60)
    assert rd.jump_segment_still_path({}, shot, 0) is None


# ---------------------------------------------------------------------------
# The jump lanes that DO mint: the guarantee must survive the fix
# ---------------------------------------------------------------------------

def test_a_jump_lane_that_mints_stills_STILL_demands_them():
    """The fix must not become a blanket amnesty.

    ``google_veo_video`` is soft_reference, so it JUMPs; it is provider-side
    and accepts stills, so ``_lane_consumes_a_still`` says mint. A real request
    IS stamped for its successor -- and if the still spine never materialized
    that object id, the render must still refuse rather than substitute the
    beat's still. Without this test the fix above could have been written as
    "segment >= 1 returns None" and every test would still pass.
    """
    contract = fc.frame_contract_for(vreg.get_engine("google_veo_video"))
    frames = int(max(contract.discrete_frames)) + 60
    shot = _shot_for("google_veo_video", frames, role="announcer_visual")
    plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])

    assert plan.join_mode == cp.JOIN_JUMP
    assert plan.segment_count > 1
    rows = shot.get("jump_still_requests") or ()
    assert rows, "a minting jump lane stamped no requests"
    assert {int(r["segment_index"]) for r in rows} == set(
        range(1, plan.segment_count))
    # An empty ledger carries no spine receipt, so the demand must REFUSE.
    with pytest.raises(rd.RenderError, match="still-spine receipt"):
        rd.jump_segment_still_path({}, shot, 1)


# ---------------------------------------------------------------------------
# The audio-driven lanes: they PLAN multi-clip now (WIRE-W4e)
# ---------------------------------------------------------------------------

AUDIO_ENGINES = ["humo", "humo_1.7B", "humo_1.7B_169", "humo_14B_169"]


@pytest.mark.parametrize("engine", AUDIO_ENGINES)
def test_an_audio_driven_beat_past_its_cap_PLANS_MULTI_CLIP(engine):
    """This test used to assert the OPPOSITE, and the flip is the point.

    Until 2026-07-29 a lane requiring an ``audio_ref`` was REFUSED at plan time
    for any beat past its cap, because it renders frames FROM that audio and
    nothing sliced the audio per segment: every segment of a split beat would
    have received the whole line from its start, and the assembled clip would
    speak the opening syllables once per segment. Garbled lip-sync is worse
    than a refusal, so it refused -- and it named the prerequisite rather than
    inventing a workaround: "per-segment audio is the prerequisite".

    That prerequisite is now built for BOTH audio sources a beat can have --
    the frozen-master slice (WIRE-W4b, tail silenced by W4c) and a per-line
    voice wav (WIRE-W4e) -- so the beat plans real coverage instead.

    THE REFUSAL WAS RIGHT WHILE IT STOOD. If per-segment slicing is ever
    removed, this test should go back to expecting the refusal rather than
    being deleted: a split lip-synced beat with whole-line audio is a sync
    defect that SHIPS, and nothing downstream would say so.
    """
    contract = fc.frame_contract_for(vreg.get_engine(engine))
    frames = int(contract.max_frames) + 60
    shot = _shot_for(engine, frames)
    plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])
    assert plan.segment_count > 1, (
        "%s should split a %d-frame beat over its %s-frame cap"
        % (engine, frames, contract.max_frames))
    assert plan.total_visible_frames == frames, (
        "the split must still assemble to exactly the beat the audio needs")
    for segment in plan.segments:
        assert contract.is_legal_length(segment.render_frames)


@pytest.mark.parametrize("engine", AUDIO_ENGINES)
def test_an_audio_driven_beat_INSIDE_its_cap_is_STILL_ONE_CLIP(engine):
    """The split is scoped to beats that need it. A beat that fits must not
    grow a second segment now that the refusal is gone -- that would put a jump
    cut in the middle of every ordinary line."""
    contract = fc.frame_contract_for(vreg.get_engine(engine))
    shot = _shot_for(engine, int(contract.max_frames))
    plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])
    assert plan.segment_count == 1
    assert plan.total_visible_frames == int(contract.max_frames)


def test_humo_14B_169_is_the_tier_this_actually_bites():
    """Its cap is 49 frames -- under two seconds. Worth stating outright, so
    the next reader knows the SPLIT above is routine for this tier and rare
    for its three siblings rather than uniformly rare."""
    capped = fc.frame_contract_for(vreg.get_engine("humo_14B_169"))
    assert capped.max_frames == 49
    assert capped.max_frames / 25.0 < 2.0
    for sibling in ("humo", "humo_1.7B", "humo_1.7B_169"):
        assert fc.frame_contract_for(vreg.get_engine(sibling)).max_frames == 177


# ---------------------------------------------------------------------------
# The unbounded lanes: they never reach any of this
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["still_pan", "still_flat", "viz_mxc_cpu",
                                    "mesh_stage"])
def test_an_unbounded_lane_stays_ONE_clip_at_any_length(engine):
    """Composed end to end through the real shot lock, not just asserted of the
    contract in isolation. A very long beat on a lane with no ceiling must
    still produce exactly one segment and no jump stills."""
    shot = _shot_for(engine, 4000)                  # 160 seconds
    plan = cp.CoveragePlan.from_dict(shot["coverage_plan"])
    assert plan.segment_count == 1
    assert plan.segments[0].render_frames == 4000
    assert plan.segments[0].trim_tail == 0
    assert "jump_still_requests" not in shot
