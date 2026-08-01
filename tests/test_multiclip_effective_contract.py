"""B3 -- the tier ceiling PLANS, for exactly one lane, and WAN does not move.

``max_render_frames`` is a per-clip render-length ceiling that rides
profile -> OTR_VideoDirector widget -> policy -> ledger. Until B3 the coverage
PLANNER ignored it entirely and partitioned every beat against the adapter's
STATIC contract, so a tier that pinned 65 still got 161-frame segments it
cannot afford.

THE TRAP THIS FILE EXISTS TO GUARD (judgment section 3): the ceiling is NOT a
general planning cap. WAN reads 17, renders a short native clip, and
ping-pongs it up to the beat length -- its beat length is unchanged by the
ceiling. Narrowing WAN's contract before ``partition_beat`` would turn every
WAN beat into a pile of 17-frame renders and silently rewrite the low-VRAM
tier PBUG-20260723-02 just fixed. So the derivation is scoped to
``frame_contract.PLANNING_CAP_ENGINES``, an allowlist of one, and the WAN
regression below ships in the same commit as the narrowing.

WHAT MAKES THESE TESTS NON-DECORATIVE. A pre-code panel warned that the
obvious assertions here ("a receipt was stamped", "the ceiling rode the
ledger") all still pass if ``effective_frame_contract`` returns its input
unchanged -- i.e. if B3 does nothing at all. The assertion that cannot survive
that is on the SEGMENT LENGTHS the real planner emitted, with the unpinned
partition of the same beat as its differential control. Both are here, and
``test_the_SAME_beat_unpinned_emits_a_segment_the_ceiling_forbids`` is the
control that proves the pinned case is measuring the ceiling rather than
arithmetic that would have happened anyway.
"""

from __future__ import annotations

import pathlib

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

#: The canonical acceptance beat (judgment section 5): the shipped
#: OTR_EpisodeAssembler produces round((10.0 - 0.5) * 25) = 237 frames for the
#: opening beat, and 237 necessarily exercises tail trim on an 8n+1 ladder.
CANONICAL_TARGET = 237

#: A ceiling ON the 8n+1 grid, chosen because the judgment states the exact
#: partition it must produce.
CEILING_ON_GRID = 65


def _beats(role="character_video", n=1):
    return [{"beat_id": "b%03d" % i, "role": role, "char_id": "c1", "dur_s": 9.5}
            for i in range(1, n + 1)]


def _budget(beats, frames=CANONICAL_TARGET):
    per = {b["beat_id"]: frames for b in beats}
    return {"per_beat": per, "total_frames": frames * len(beats), "warnings": []}


def _policy(engine, ceiling=None):
    policy = {
        "policy_version": 2,
        "video_models": {
            "announcer_video_model": {"engine_id": engine},
            "music_video_model": {"engine_id": engine},
            "character_video_model": {"engine_id": engine},
        },
    }
    if ceiling is not None:
        policy["max_render_frames"] = ceiling
    return policy


def _plan_one(engine, ceiling=None, target=CANONICAL_TARGET):
    """Drive the REAL planner and return the single shot row it stamped.

    Deliberately NOT a direct ``_stamp_coverage_plan`` call: the ceiling has to
    travel the same policy -> build_execution_plan path production uses, or the
    test proves only that a helper works when handed the right number.
    """
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, target), {}, _policy(engine, ceiling))
    assert len(shots) == 1
    return shots[0]


def _renders(shot):
    return [int(s["render_frames"])
            for s in (shot.get("coverage_plan") or {}).get("segments") or ()]


# ---------------------------------------------------------------------------
# The narrowing -- ltx_8gb only
# ---------------------------------------------------------------------------

def test_the_ltx_8gb_ceiling_narrows_what_the_planner_may_emit():
    """THE ASSERTION THAT BREAKS IF B3 DOES NOTHING.

    Every emitted segment must fit under the tier ceiling, and the judgment
    names the exact partition: [65, 65, 65, 49], drop_head=1 on each successor,
    trim_tail=4 on the last, every length 8n+1.
    """
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    segments = shot["coverage_plan"]["segments"]
    assert _renders(shot) == [65, 65, 65, 49], (
        "the ceiling did not reach the partitioner: got %r" % (_renders(shot),))
    assert max(_renders(shot)) <= CEILING_ON_GRID
    assert all((n - 1) % 8 == 0 for n in _renders(shot))
    assert [int(s["drop_head"]) for s in segments] == [0, 1, 1, 1]
    assert [int(s["trim_tail"]) for s in segments] == [0, 0, 0, 4]
    assert int(shot["coverage_plan"]["target_visible_frames"]) == CANONICAL_TARGET


def test_the_SAME_beat_unpinned_emits_a_segment_the_ceiling_forbids():
    """THE DIFFERENTIAL CONTROL for the test above.

    Without it, "every segment is <= 65" could be true of a partition the
    planner would have produced anyway, and the whole file would pass against a
    no-op ``effective_frame_contract``. Unpinned, this beat is covered with a
    161-frame segment -- comfortably over the ceiling.
    """
    unpinned = _plan_one("ltx_8gb", None)
    assert max(_renders(unpinned)) > CEILING_ON_GRID, (
        "the unpinned partition must exceed the ceiling or the pinned test "
        "proves nothing; got %r" % (_renders(unpinned),))
    assert _renders(unpinned) != _renders(_plan_one("ltx_8gb", CEILING_ON_GRID))


def test_a_pinned_ltx_8gb_beat_carries_its_contract_receipt():
    """The receipt must be derived from the DECLARED contract.

    The first draft of the stamp site rebound one variable and fed the
    ALREADY-NARROWED contract back in. A narrowed contract narrows to itself,
    compares equal, and yields None -- so the receipt silently never existed
    and the render boundary had nothing to check. This is that regression.
    """
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    receipt = shot.get("coverage_contract")
    assert receipt is not None, "no receipt was stamped for a narrowed contract"
    # THE WHOLE DICT, not a subset (2026-07-27 post-code panel). Spot-checking
    # five of eight fields let a swap of min_frames and native_fps survive the
    # entire file: both sides of the render-boundary equality call the same
    # pure function, so a bug INSIDE the receipt builder reproduces identically
    # on both sides and can never be caught by comparing them to each other.
    # Only a literal here has an opinion about what the fields mean.
    assert receipt == {
        "engine_id": "ltx_8gb",
        "min_frames": 9,
        "max_frames": CEILING_ON_GRID,
        "quantum": 8,
        "native_fps": 25,
        "allow_tail_trim": True,
        "continuity": fc.CONTINUITY_STRICT_FIRST_FRAME,
        "max_render_frames": CEILING_ON_GRID,
    }


def test_the_ceiling_snaps_DOWN_to_a_legal_rung_never_to_itself():
    """A ceiling off the 8n+1 grid must become the largest legal rung BELOW it.

    Narrowing to the raw ceiling would emit segments the adapter refuses at
    render time, after the weights load -- the failure this derivation exists
    to prevent, one level up.
    """
    shot = _plan_one("ltx_8gb", 100)
    assert shot["coverage_contract"]["max_frames"] == 97
    assert shot["coverage_contract"]["max_render_frames"] == 100
    assert max(_renders(shot)) <= 97
    assert all((n - 1) % 8 == 0 for n in _renders(shot))


def test_a_ceiling_at_or_above_the_declared_maximum_changes_nothing():
    """161 is ltx_8gb's own maximum; the widget accepts up to 240.

    A non-binding ceiling must leave the contract identical AND stamp no
    receipt -- an inert receipt would make the render boundary compare
    something that carries no information.
    """
    pinned = _plan_one("ltx_8gb", 240)
    unpinned = _plan_one("ltx_8gb", None)
    assert _renders(pinned) == _renders(unpinned)
    assert pinned.get("coverage_contract") is None


def test_a_ceiling_below_the_engines_floor_is_TERMINAL_not_swallowed():
    """THE SWALLOW GUARD (chunk 1a's lesson, applied before the fact).

    ``_stamp_coverage_plan`` resolves the contract inside a broad
    ``except Exception: return`` that means "unbuildable engine, give it no
    plan". Folding the ceiling derivation into that block would turn a
    misconfigured tier into a beat with NO coverage plan, indistinguishable in
    the log from an unregistered engine. The derivation therefore sits OUTSIDE
    it, and this test fails the moment someone moves it back in: a swallowed
    error returns a shot with no plan instead of raising.
    """
    with pytest.raises(fc.PlanningCapError) as excinfo:
        _plan_one("ltx_8gb", 8)
    assert "ltx_8gb" in str(excinfo.value)
    assert "NO FALLBACK" in str(excinfo.value)


def test_a_ceiling_exactly_at_the_floor_is_legal():
    """9 is a legal rung, so it must plan rather than refuse -- the boundary
    on the other side of the refusal above."""
    shot = _plan_one("ltx_8gb", 9, target=25)
    assert shot["coverage_contract"]["max_frames"] == 9
    assert set(_renders(shot)) == {9}


# ---------------------------------------------------------------------------
# THE WAN REGRESSION -- ships in this commit by judgment section 3
# ---------------------------------------------------------------------------

def test_a_pinned_ceiling_does_not_move_wan_ti2v_coverage_topology():
    """17 is the SHIPPED 8GB WAN ceiling (config/profiles/otr_8gb_wan.json).

    If the ceiling ever became a general planning cap, this beat would
    partition into a pile of 17-frame renders. The plan must be byte-identical
    to the unpinned one, and no receipt may be stamped.
    """
    pinned = _plan_one("wan_ti2v", 17)
    unpinned = _plan_one("wan_ti2v", None)
    assert pinned["coverage_plan"] == unpinned["coverage_plan"], (
        "a pinned tier ceiling moved WAN's coverage topology: %r vs %r"
        % (_renders(pinned), _renders(unpinned)))
    assert max(_renders(pinned)) > 17, (
        "this beat must exceed 17 frames per segment or the regression is "
        "vacuous; got %r" % (_renders(pinned),))
    assert pinned.get("coverage_contract") is None


def test_wan_ti2v_keeps_its_adapter_side_ping_pong():
    """The other half of the same guarantee.

    WAN's ceiling stays an ADAPTER-side native cap plus ping-pong extension.
    B4 deletes ping-pong on the ltx_8gb lane only; if a later change rips WAN's
    with it, the 8GB WAN tier renders 17-frame beats and this fails by name.
    """
    source = pathlib.Path(
        vreg.get_engine("wan_ti2v").__class__.__module__.replace(".", "/"))
    path = pathlib.Path(__file__).resolve().parents[1] / (str(source) + ".py")
    text = path.read_text(encoding="utf-8")
    # The exact CALL, argument order included -- a bare substring check passes
    # a reversed-argument mutation that silently corrupts the extension
    # (2026-07-27 post-code panel). This is a structural pin, not a behavioural
    # one: WAN's extension behaviour is proven by tests/test_clip_fill.py and
    # tests/test_remaining_video_contracts.py, and this only guards against B4
    # ripping the call out along with the ltx_8gb one.
    assert "_wb.extend_frames_to_target(frames, target_frames)" in text, (
        "wan_ti2v lost its ping-pong extension -- the 8GB WAN tier renders a "
        "short native clip and fills the beat with it; removing that is a "
        "different lane's decision, not this one's")


def test_a_discrete_menu_contract_cannot_be_narrowed_by_a_ceiling():
    """The one branch no registry-driven test can reach.

    ``is_legal_length`` consults ``discrete_frames`` BEFORE ``max_frames``, so
    lowering ``max_frames`` on a menu contract changes nothing about what the
    adapter accepts -- a "narrowed" discrete contract would be one that lies.
    No engine in the allowlist declares a menu, so this is only reachable by
    calling the function directly, which is exactly why it needs a test: it is
    the guard the NEXT allowlist entry will meet.
    """
    veo_shaped = fc.FrameContract(
        discrete_frames=(96, 144, 192), native_fps=24, allow_tail_trim=True,
        continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    with pytest.raises(fc.PlanningCapError, match="discrete frame menu"):
        fc.effective_frame_contract("ltx_8gb", veo_shaped, 100)


def test_a_discrete_menu_is_UNTOUCHED_by_a_ceiling_it_never_reaches():
    """The other half, and the reason the branch is not a blanket refusal.

    A ceiling at or above the whole menu constrains nothing, and the
    function's documented guarantee -- non-binding ceilings return the
    contract unchanged -- has to hold for the discrete shape too. Refusing
    here would fail a tier that never limited the engine at all.
    """
    veo_shaped = fc.FrameContract(
        discrete_frames=(96, 144, 192), native_fps=24, allow_tail_trim=True,
        continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)
    assert fc.effective_frame_contract("ltx_8gb", veo_shaped, 192) is veo_shaped
    assert fc.effective_frame_contract("ltx_8gb", veo_shaped, 240) is veo_shaped
    assert fc.coverage_contract_receipt("ltx_8gb", veo_shaped, 240) is None


def test_the_derivation_is_a_NAMED_allowlist():
    """Pinned as an EXACT TUPLE, not a spirit: adding an engine here is a
    per-engine decision with a live proof attached, and this fails until someone
    updates it deliberately.

    Was an allowlist of ONE until 2026-08-01. ``fastwan_8gb`` joined with the
    live proof the constant's own comment demands: a canonical run refused by
    name -- "fastwan_8gb was handed a coverage-planned segment of 177 frame(s)
    but this tier pins its render ceiling at 17". Both WAN-family adapters are
    CHAINABLE (strict_first_frame), so partition_beat plans multi-clip coverage
    for them, and a multi-clip beat never reaches ``_floor_length`` -- the
    ping-pong that makes a low ceiling harmless on the single-clip path never
    runs. Listing the engine lets the planner see the cap.

    ``wan_ti2v`` is deliberately NOT here. Its ceiling stays a RENDER cap and
    its topology stays unmoved (pinned below); the same contradiction is latent
    for it and is logged as its own defect rather than fixed inside this build."""
    assert fc.PLANNING_CAP_ENGINES == ("ltx_8gb", "fastwan_8gb")
    assert "wan_ti2v" not in fc.PLANNING_CAP_ENGINES


# ---------------------------------------------------------------------------
# The render boundary re-derives, and refuses on any difference
# ---------------------------------------------------------------------------

def _ledger_from(shot, ceiling):
    return {"episode_id": "ep_b3",
            "video": {"max_render_frames": ceiling, "shots": [dict(shot)]}}


def test_the_render_boundary_accepts_the_plan_it_was_stamped_with():
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    assert rd.assert_coverage_plans(
        _ledger_from(shot, CEILING_ON_GRID)) == 1


def test_the_render_boundary_refuses_a_ceiling_that_MOVED():
    """The plan is still arithmetically legal under the new ceiling -- 65-frame
    segments fit fine below 97 -- so only the receipt can catch this."""
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    with pytest.raises(rd.RenderError, match="COVERAGE CONTRACT"):
        rd.assert_coverage_plans(_ledger_from(shot, 97))


def test_the_render_boundary_refuses_a_ceiling_that_VANISHED():
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    with pytest.raises(rd.RenderError, match="COVERAGE CONTRACT"):
        rd.assert_coverage_plans(_ledger_from(shot, 0))


def test_the_render_boundary_refuses_a_ceiling_that_APPEARED():
    """The reverse direction: planned unpinned, rendered under a ceiling. The
    plan carries 161-frame segments a 65-frame tier cannot render."""
    shot = _plan_one("ltx_8gb", None)
    assert shot.get("coverage_contract") is None
    with pytest.raises(rd.RenderError):
        rd.assert_coverage_plans(_ledger_from(shot, CEILING_ON_GRID))


def test_the_render_boundary_validates_against_the_NARROWED_contract():
    """A receipt-valid ledger whose PLAN was tampered with must still refuse.

    Found by mutation: replacing the narrowed contract with the declared one at
    the validation call left the whole file green, because the receipt equality
    fires first in every other scenario. It does not fire here -- the receipt
    is exactly the one this ceiling derives -- so only validating the plan
    against the NARROWED contract catches it. This is the hand-edited or
    replayed ledger the second boundary exists for: 161-frame segments are
    perfectly legal for ltx_8gb's DECLARATION and unrenderable on a tier that
    pinned 65.
    """
    pinned = _plan_one("ltx_8gb", CEILING_ON_GRID)
    tampered = dict(_plan_one("ltx_8gb", None))
    tampered["coverage_contract"] = pinned["coverage_contract"]
    assert max(_renders(tampered)) > CEILING_ON_GRID
    with pytest.raises(rd.RenderError, match="cannot execute"):
        rd.assert_coverage_plans(_ledger_from(tampered, CEILING_ON_GRID))


def test_the_render_boundary_refuses_a_receipt_whose_ENGINE_was_swapped():
    """The legacy force-map path mutates engine_id and THEN validates.

    A plan built under ltx_8gb's ceiling, executed by another engine, is a plan
    built under a contract that is not the one in force. It must refuse via the
    CONTRACT receipt -- not by arithmetic accident.
    """
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    shot["engine_id"] = "wan_ti2v"
    with pytest.raises(rd.RenderError, match="COVERAGE CONTRACT"):
        rd.assert_coverage_plans(_ledger_from(shot, CEILING_ON_GRID))


def test_the_boundary_refuses_a_receipt_whose_engine_no_longer_RESOLVES():
    """The weaker branch, and it was too weak (2026-07-27 post-code panel).

    When the adapter cannot be resolved at render time the full receipt cannot
    be re-derived, so this branch holds the ledger to the two facts the receipt
    states about its own provenance. It originally compared only the CEILING,
    and two seats each built the same live repro: a stale ltx_8gb receipt on a
    shot swapped to an unregistered engine, ceiling unchanged, sailed straight
    through to an arithmetic-only check -- a plan built for an 8n+1 ladder
    accepted for an engine never checked against any length.
    """
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    shot["engine_id"] = "an_engine_this_registry_has_never_heard_of"
    with pytest.raises(rd.RenderError, match="planned for engine"):
        rd.assert_coverage_plans(_ledger_from(shot, CEILING_ON_GRID))


def test_the_boundary_refuses_a_MALFORMED_receipt():
    """An empty or non-dict receipt is damage, not absence.

    ``shot.get("coverage_contract") or None`` collapses ``{}`` to "nothing
    narrowed", so a corrupted ledger would read as an ordinary unpinned one.
    """
    shot = _plan_one("ltx_8gb", CEILING_ON_GRID)
    shot["coverage_contract"] = {}
    with pytest.raises(rd.RenderError, match="not a receipt"):
        rd.assert_coverage_plans(_ledger_from(shot, CEILING_ON_GRID))


def test_an_unpinned_ledger_with_no_ceiling_key_still_validates():
    """Every hand-built fixture and both HTTP entry points build a video
    section with no ``max_render_frames`` key at all."""
    shot = _plan_one("wan_ti2v", None)
    ledger = {"episode_id": "ep_b3", "video": {"shots": [dict(shot)]}}
    assert rd.assert_coverage_plans(ledger) == 1


# ---------------------------------------------------------------------------
# ONE normalization, not three copies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, 0), ("", 0), (0, 0), (-5, 0), ("17", 17), (17, 17),
    ("not a number", 0), (65.0, 65),
])
def test_the_ceiling_has_exactly_one_normalization(raw, expected):
    """``otr_shot_lock`` stamps the ledger and plans against it; the render
    boundary re-derives from what was stamped. Three literal copies of
    ``max(0, int(x or 0))`` is how those start disagreeing for reasons that
    have nothing to do with the ceiling, so there is one function and the
    planner-side helper delegates to it.

    A malformed value reads as UNPINNED rather than raising -- a ceiling
    nobody can parse must not silently become a small number.

    THE ADAPTER SIDE IS THE THIRD SITE and it is checked here (2026-07-27
    post-code panel). ``motion_common.profile_max_render_frames`` is what
    ``eng_wan_ti2v._floor_length`` reads to resolve WAN's native cap at render
    time; it was a hand-copied fourth expression that agreed with the others
    only because someone copied carefully. A test named "exactly one
    normalization" that never touched the site its own docstring cited was
    exactly the decorative shape this build keeps finding.
    """
    assert fc.normalized_planning_ceiling(raw) == expected
    assert sl._planning_ceiling({"max_render_frames": raw}) == expected
    engine = vreg.get_engine("wan_ti2v")
    engine._active_profile = {"max_render_frames": raw}
    try:
        assert engine.profile_max_render_frames() == expected
    finally:
        engine._active_profile = None
