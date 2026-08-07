"""THE MOTION FLOOR -- operator direction 2026-07-29, pinned as a roster gate.

> "For video models, there needs to be video for every beat. ... if the
> minimum is, like, four seconds, then we should have video for four seconds."

THE CREDITS ARE AN EXPLICIT EXCEPTION (same direction, clarified): a still, a
ping-pong or even a black background is fine under the credits console. This
file is about BEATS, and only beats.

WHAT THIS PINS, AND WHY IT IS A GATE RATHER THAN A FEATURE. The behaviour the
direction asks for is ALREADY the shipped behaviour and has been since
2026-07-25: for any contract permitting a tail trim, ``partition_beat`` renders
the SMALLEST LEGAL LENGTH at or above the beat target and trims the surplus, so
a beat shorter than an engine's minimum still gets real rendered frames rather
than a held image. Nobody had written that down as a contract, though, and one
``allow_tail_trim=False`` on a video adapter would silently reopen the still
floor for that lane. So this is the roster gate that fails BY NAME.

Measured at the time of writing: all 31 registered engines cover a 1-second
beat with real video. `google_veo` renders 100 frames (4.0 s) and trims 75 --
which is the operator's sentence, executed.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg

#: The canvas rate every beat target is expressed at.
FPS = 25

#: A one-second beat. Shorter than the minimum render of every heavy engine in
#: the roster, which is exactly the case the direction is about.
SHORT_BEAT = FPS


def _engines():
    return sorted(vreg.all_engine_names())


def _contract(name):
    return fc.frame_contract_for(vreg.get_engine(name))


@pytest.mark.parametrize("name", _engines())
def test_every_engine_covers_a_SHORT_beat_with_REAL_RENDERED_FRAMES(name):
    """No engine may answer a one-second beat with a refusal.

    A refusal here is what the 2026-07-28 still-floor ruling would route to a
    still, and the 2026-07-29 direction says a still is not the answer where
    the engine can render at all. Every engine CAN render at all -- the
    question is only whether it is allowed to render LONGER than the beat and
    trim, which is what ``allow_tail_trim`` grants.
    """
    contract = _contract(name)
    plan = cp.partition_beat(SHORT_BEAT, contract)

    rendered = sum(s.render_frames for s in plan.segments)
    assert rendered >= SHORT_BEAT or rendered >= int(contract.min_frames), (
        "%s renders only %d frame(s) for a %d-frame beat"
        % (name, rendered, SHORT_BEAT))
    # ...and every one of those frames is a real render on the engine's ladder.
    for seg in plan.segments:
        assert contract.is_legal_length(seg.render_frames), (
            "%s planned an illegal render length %d"
            % (name, seg.render_frames))
    # The beat still assembles to EXACTLY its target -- the surplus is trimmed,
    # never padded and never held.
    assert plan.total_visible_frames == SHORT_BEAT


@pytest.mark.parametrize("name", _engines())
def test_no_engine_forbids_the_TAIL_TRIM_that_makes_the_floor_work(name):
    """The single declaration that would reopen the still floor.

    ``join_mode_for`` routes a below-minimum beat to a single trimmed render
    ONLY when the contract permits the trim. An adapter that declares
    ``allow_tail_trim=False`` and a ``min_frames`` above a short beat has no
    legal cover at all -- it refuses, and a refusal is where a still floor
    would get reached. Nothing in the roster does this today; this is what
    notices the day something does.
    """
    contract = _contract(name)
    if int(contract.min_frames) <= 1 and not contract.discrete_frames:
        pytest.skip("%s has no minimum a short beat could fall under" % name)
    assert contract.allow_tail_trim is True, (
        "%s declares min_frames=%s but forbids the tail trim, so a beat "
        "shorter than its minimum has NO legal cover and falls through to a "
        "still floor" % (name, contract.min_frames))


def test_the_operators_own_example_is_the_shipped_arithmetic():
    """"If the minimum is four seconds, we should have video for four seconds."

    Veo's smallest menu entry is 100 frames = 4.0 s at the 25 fps canvas. A
    one-second beat on that lane renders the whole 4 s and trims 3 s of it --
    real video for the beat, at the cost of frames nobody sees, which is the
    trade the direction asks for explicitly.
    """
    contract = _contract("google_veo_video")
    assert 100 in contract.discrete_frames
    plan = cp.partition_beat(SHORT_BEAT, contract)
    assert [s.render_frames for s in plan.segments] == [100]
    assert plan.segments[0].render_frames / FPS == 4.0
    assert plan.total_visible_frames == SHORT_BEAT       # trimmed to the beat


def test_a_heavy_local_lane_renders_its_own_minimum_rather_than_holding():
    """The same sentence on the lane it actually costs GPU time: HuMo's floor
    is 33 frames (1.3 s at 25 fps), and a 1 s beat pays for all 33."""
    for name in ("humo", "wan_i2v"):
        contract = _contract(name)
        assert int(contract.min_frames) == 33
        plan = cp.partition_beat(SHORT_BEAT, contract)
        assert [s.render_frames for s in plan.segments] == [33]
        assert plan.total_visible_frames == SHORT_BEAT


def test_the_roster_is_not_empty_and_this_gate_is_not_vacuous():
    """A sweep that finds nothing passes every gate built on it -- this build
    has paid for that twice. Assert the roster is really being billed --
    DERIVED from the registry's own CAPABILITIES table, never a hand-typed
    count that rots when an engine is added or retired (rip-sfx 2026-08-06
    removed five and the old `>= 30` floor went red at 27)."""
    names = _engines()
    assert names, "registry discovered no engines at all"
    assert set(names) == set(vreg.CAPABILITIES), (
        "live roster and CAPABILITIES disagree: only-live=%s only-declared=%s"
        % (sorted(set(names) - set(vreg.CAPABILITIES)),
           sorted(set(vreg.CAPABILITIES) - set(names))))
    from nodes._otr_shared.public_engines import RETIRED_ENGINE_IDS
    assert not (set(names) & RETIRED_ENGINE_IDS), (
        "retired engine id(s) re-registered: %s"
        % sorted(set(names) & RETIRED_ENGINE_IDS))
    with_minimum = [n for n in names
                    if int(_contract(n).min_frames) > 1
                    or _contract(n).discrete_frames]
    assert len(with_minimum) >= 12, (
        "only %d engines have a minimum a short beat could fall under -- the "
        "trim gate above is close to vacuous" % len(with_minimum))
    # Named anchors the trim gate exists for -- these must always carry a
    # minimum or a discrete menu:
    for anchor in ("humo", "ltx_video", "wan_ti2v", "word_razzle"):
        assert anchor in with_minimum, anchor
