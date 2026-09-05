"""Chunk 2 -- the declaration surface and the roster audit.

``FrameContract`` is the pure, static declaration the partitioner will reason
about in chunk 3. It never reads live VRAM or mutable environment: stills are
minted BEFORE the render phase, so a partition computed from runtime state
would be one the image phase could not have planned for.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg


# ---------------------------------------------------------------------------
# The default: what an adapter that declares NOTHING resolves to
#
# It is no longer "single_only until it proves otherwise" -- chunk 7a deleted
# the ``supports_multi_clip`` opt-in on the operator's ruling that every engine
# gets equal terms. SINGLE_ONLY survives as the answer for a stub or an adapter
# that failed to import: unbounded, so every length is legal, so it never
# splits and never refuses. No REGISTERED engine may resolve to it -- that is
# tests/test_engine_contract_roster.py's job, and it fails by name.
# ---------------------------------------------------------------------------

def test_the_undeclared_default_is_unbounded_and_never_chains():
    assert fc.SINGLE_ONLY.max_frames == 0
    assert fc.SINGLE_ONLY.continuity == fc.CONTINUITY_NONE
    # Unbounded means no beat can ever overflow one render, so there is
    # nothing to split -- which is what the old opt-in flag was really saying.
    assert fc.SINGLE_ONLY.is_legal_length(1)
    assert fc.SINGLE_ONLY.is_legal_length(100000)


def test_none_engine_is_single_only():
    assert fc.frame_contract_for(None) == fc.SINGLE_ONLY


def test_a_broken_declaration_degrades_to_single_only():
    """The safe direction: the worst case is a beat renders as it does today."""
    class _Broken:
        def frame_contract(self):
            raise RuntimeError("adapter is confused")

    assert fc.frame_contract_for(_Broken()) == fc.SINGLE_ONLY


def test_declaration_may_be_a_method_or_an_attribute():
    contract = fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                                continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)

    class _Method:
        def frame_contract(self):
            return contract

    class _Attr:
        frame_contract = contract

    assert fc.frame_contract_for(_Method()) == contract
    assert fc.frame_contract_for(_Attr()) == contract


def test_EVERY_BOUNDED_ENGINE_CAN_SPLIT_AND_UNBOUNDED_ONES_NEED_NOT():
    """The inverse of what this asserted through chunk 6.

    It used to be ``test_EVERY_REGISTERED_ENGINE_IS_SINGLE_ONLY_TODAY``, which
    passed because no adapter had set ``supports_multi_clip``. Chunk 7a deleted
    that flag: multi-clip is universal now, and the only question left is
    arithmetic -- does this engine have a ceiling a beat could exceed.

    So the assertion flips. Every engine with a real ceiling can split; the
    unbounded lanes (visualizers, still families, mesh_stage) cannot need to,
    because no length is ever illegal for them. An unbounded engine reporting
    ``can_split`` would mean a ceiling appeared without a declaration.
    """
    splittable, unbounded = [], []
    for name in sorted(vreg.all_engine_names()):
        try:
            engine = vreg.get_engine(name)
        except Exception:  # noqa: BLE001 -- unbuildable engines are not our subject
            continue
        (splittable if fc.can_split(engine) else unbounded).append(name)
    # NOT ``assert can_split(e) == bool(e.max_frames or e.discrete_frames)``.
    # That line was here and a QA panel called it: it re-executes can_split's
    # own body and compares it to the call, so it cannot fail for any engine
    # ever. The real check is the literal set below.
    assert splittable, "no engine declares a ceiling -- the sweep is vacuous"
    assert unbounded, "no engine is unbounded -- the sweep is vacuous"
    # The unbounded set is exactly the lanes that synthesise frames on demand.
    assert set(unbounded) == {
        # Ghost Signal (2026-08-22). Unbounded for a DIFFERENT reason than its
        # neighbours here, and the distinction is worth keeping: the others
        # synthesise frames from a still on demand, while this one generates a
        # fresh AnimateDiff timeline spanning the beat's whole audio budget. The
        # 16 in its context options is a sliding WINDOW, not a clip ceiling, so
        # there is nothing for the partitioner to split on either way.
        # The five non-haunted siblings retired 2026-08-23. The adapter patches
        # the image model and touches no timing, so the unbounded contract that
        # covered them all is unchanged on the lane that survived.
        "animatediff15_v3_haunted_video",
        # the still-in LAB PEER (2026-09-02) inherits the haunted lane's
        # unbounded contract: one timeline per beat, max_frames 0, no split
        "animatediff15_v3_stillin_lab_video",
        "mesh_stage", "still_flat", "still_motion", "still_pan", "still_word",
        "viz_camera", "viz_green", "viz_mxc_cpu", "viz_mxc_mandala",
    }, sorted(unbounded)


# ---------------------------------------------------------------------------
# Contract validation -- a declaration that lies must not be constructible
# ---------------------------------------------------------------------------

def test_continuity_token_is_closed():
    with pytest.raises(fc.FrameContractError):
        fc.FrameContract(continuity="sort_of")


@pytest.mark.parametrize("kwargs", [
    {"min_frames": 0},
    {"quantum": 0},
    {"min_frames": 100, "max_frames": 50},
])
def test_impossible_bounds_are_rejected(kwargs):
    with pytest.raises(fc.FrameContractError):
        fc.FrameContract(**kwargs)


def test_discrete_frames_require_tail_trim():
    """A fixed duration menu cannot sum to an arbitrary beat.

    Veo's 4/6/8s durations are the real case: without trimming there are beat
    lengths this engine could never cover exactly, so the contract would lie.
    """
    with pytest.raises(fc.FrameContractError):
        fc.FrameContract(discrete_frames=(100, 150, 200),
                         allow_tail_trim=False)
    fc.FrameContract(discrete_frames=(100, 150, 200), allow_tail_trim=True)


def test_only_a_ceiling_makes_an_engine_splittable():
    """Replaces ``test_multi_clip_requires_a_ceiling`` (chunk 7a).

    That test asserted a CONSTRUCTION error: declaring ``supports_multi_clip``
    without a ceiling raised, because opting in to splitting with nothing to
    split at was incoherent. With the opt-in deleted the incoherent state is
    unconstructible rather than rejected -- splittability is now derived from
    the ceiling instead of stored alongside it, so the two cannot disagree.
    """
    unbounded = fc.FrameContract(min_frames=1, max_frames=0, quantum=1)
    bounded = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    menu = fc.FrameContract(discrete_frames=(100, 200), allow_tail_trim=True)

    class _Eng:
        def __init__(self, contract):
            self.frame_contract = contract

    assert fc.can_split(_Eng(unbounded)) is False
    assert fc.can_split(_Eng(bounded)) is True
    assert fc.can_split(_Eng(menu)) is True


def test_contract_is_frozen():
    contract = fc.FrameContract(min_frames=9, max_frames=161, quantum=8)
    with pytest.raises(Exception):
        contract.max_frames = 9999


# ---------------------------------------------------------------------------
# The arithmetic the partitioner will depend on (ltx_8gb: 9 + 8n, cap 161)
# ---------------------------------------------------------------------------

@pytest.fixture
def ltx8():
    return fc.FrameContract(min_frames=9, max_frames=161, quantum=8,
                            continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)


@pytest.mark.parametrize("n,ok", [(8, False), (9, True), (17, True),
                                  (18, False), (161, True), (169, False)])
def test_is_legal_length(ltx8, n, ok):
    assert ltx8.is_legal_length(n) is ok


def test_largest_legal_at_most_clamps_to_the_ceiling(ltx8):
    assert ltx8.largest_legal_at_most(1000) == 161
    assert ltx8.largest_legal_at_most(20) == 17
    assert ltx8.largest_legal_at_most(9) == 9
    assert ltx8.largest_legal_at_most(8) is None


def test_smallest_legal_at_least(ltx8):
    assert ltx8.smallest_legal_at_least(10) == 17
    assert ltx8.smallest_legal_at_least(1) == 9
    assert ltx8.smallest_legal_at_least(162) is None      # over the ceiling


def test_the_169_frame_acceptance_case_decomposes(ltx8):
    """The r3-adopted first live target: 169 == 161 + (9 - 1).

    The cap plus one legal minimum segment, less the chained duplicate head
    frame. It proves a two-segment chain with NO tail trim, and unlike a vague
    "over 161" it is exactly reproducible.
    """
    target_visible = 169
    first = ltx8.largest_legal_at_most(target_visible)
    assert first == 161
    remainder_visible = target_visible - first          # 8 visible frames
    # the successor duplicates the predecessor's terminal frame, so it must
    # RENDER one more than it contributes
    successor = ltx8.smallest_legal_at_least(remainder_visible + 1)
    assert successor == 9
    assert first + (successor - 1) == target_visible     # exact sum, no trim


# ---------------------------------------------------------------------------
# Continuity -- "accepts a still" is NOT "guarantees first-frame continuity"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The roster audit -- the swallowed-import blindspot
# ---------------------------------------------------------------------------

def test_no_declared_engine_failed_to_register():
    """THE CI GATE for the swallowed-import blindspot.

    Every adapter import in ``_otr_video_engines/__init__.py`` is wrapped in a
    bare ``except Exception: pass``, so a broken adapter never registers and
    silently vanishes from every per-role dropdown. ``CAPABILITIES`` is the
    independent expected roster that survives such a failure.
    """
    roster = vreg.audit_engine_roster()
    assert roster["missing"] == (), (
        "these engines are declared in CAPABILITIES but did not register -- "
        "their import raised and was swallowed: %r" % (roster["missing"],))


def test_no_engine_registered_without_a_capabilities_row():
    """'registry IS the menu' (C0): a registered engine needs its row."""
    roster = vreg.audit_engine_roster()
    assert roster["unexpected"] == ()


def test_audit_is_pure():
    assert vreg.audit_engine_roster() == vreg.audit_engine_roster()


def test_a_still_LANE_is_ONE_still_because_it_is_UNBOUNDED_not_because_it_is_special():
    """``still_*`` lanes are ONE still -- operator directive, 2026-07-25.

    THE GUARDRAIL IS GONE AND THE INVARIANT IS STRONGER FOR IT (chunk 7a,
    2026-07-26). This test used to assert that no ``still_*`` engine had set
    ``supports_multi_clip`` -- a name-prefix special case defending against a
    copy-pasted opt-in. The operator removed the opt-in outright: "there's no
    gate with opt in or opt out... everything gets an equal term."

    Which leaves the invariant resting on arithmetic instead of on a rule about
    what an engine is called. A still lane declares NO CEILING, so every length
    is a legal single render, so ``partition_beat`` returns one segment for
    every beat there has ever been -- and one segment mints one still. A
    prefix check could be defeated by renaming an engine. This cannot: to split
    a still lane you would have to give it a ceiling it does not have.
    """
    still_lanes = [n for n in vreg.all_engine_names() if n.startswith("still_")]
    # A sweep over an empty roster asserts nothing. Pin that it is not empty --
    # two "exhaustive" sweeps in this build turned out to be theatre.
    assert still_lanes, "no still_* engines in the roster: this sweep is theatre"
    bounded, skipped = [], []
    for name in still_lanes:
        try:
            engine = vreg.get_engine(name)
        except Exception:  # noqa: BLE001 -- record it, never swallow it
            skipped.append(name)
            continue
        if fc.can_split(engine):
            bounded.append(name)
    assert bounded == [], (
        "these still_* lanes declare a ceiling, so a long enough beat would "
        "split them into segments and mint a second still for a lane that owns "
        "exactly one: %r" % (bounded,))
    # A still lane that cannot be BUILT cannot be CHECKED, and a guardrail that
    # quietly skips its subject is the same theatre in a different costume
    # (2026-07-26 QA panel). If one of these stops constructing, say so.
    assert skipped == [], (
        "these still_* engines could not be built, so the one-still invariant "
        "never actually checked them: %r" % (skipped,))
