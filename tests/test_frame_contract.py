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
# The default: every adapter is single_only until it proves otherwise
# ---------------------------------------------------------------------------

def test_default_contract_is_single_only():
    assert fc.SINGLE_ONLY.supports_multi_clip is False
    assert fc.SINGLE_ONLY.continuity == fc.CONTINUITY_NONE


def test_an_adapter_that_declares_nothing_is_single_only():
    class _Bare:
        pass

    assert fc.frame_contract_for(_Bare()) == fc.SINGLE_ONLY
    assert fc.supports_multi_clip(_Bare()) is False
    assert fc.can_chain(_Bare()) is False


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
                                supports_multi_clip=True,
                                continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)

    class _Method:
        def frame_contract(self):
            return contract

    class _Attr:
        frame_contract = contract

    assert fc.frame_contract_for(_Method()) == contract
    assert fc.frame_contract_for(_Attr()) == contract


def test_EVERY_REGISTERED_ENGINE_IS_SINGLE_ONLY_TODAY():
    """Chunk 2 adds the surface and changes NO behaviour.

    Multi-clip is opt-in and provable per engine. If this ever fails, an
    adapter opted in without its own live proof -- which is the failure mode
    the whole per-adapter declaration exists to prevent.
    """
    optedin = []
    for name in vreg.all_engine_names():
        try:
            engine = vreg.get_engine(name)
        except Exception:  # noqa: BLE001 -- unbuildable engines are not our subject
            continue
        if fc.supports_multi_clip(engine):
            optedin.append(name)
    assert optedin == []


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


def test_discrete_durations_require_tail_trim():
    """A fixed duration menu cannot sum to an arbitrary beat.

    Veo's 4/6/8s durations are the real case: without trimming there are beat
    lengths this engine could never cover exactly, so the contract would lie.
    """
    with pytest.raises(fc.FrameContractError):
        fc.FrameContract(discrete_durations=(100, 150, 200),
                         allow_tail_trim=False)
    fc.FrameContract(discrete_durations=(100, 150, 200), allow_tail_trim=True)


def test_multi_clip_requires_a_ceiling():
    """Multi-clip exists to cover a beat LONGER than one render. An engine with
    no ceiling has nothing to split at, so the declaration is incoherent."""
    with pytest.raises(fc.FrameContractError):
        fc.FrameContract(supports_multi_clip=True, max_frames=0)


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
                            supports_multi_clip=True,
                            continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)


def test_legal_lengths_are_the_8n_plus_1_ladder(ltx8):
    lengths = ltx8.legal_lengths()
    assert lengths[0] == 9 and lengths[-1] == 161
    assert all((n - 1) % 8 == 0 for n in lengths)


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


def test_discrete_duration_lane_covers_by_trimming():
    veo = fc.FrameContract(min_frames=100, max_frames=200,
                           discrete_durations=(100, 150, 200),
                           allow_tail_trim=True)
    assert veo.legal_lengths() == (100, 150, 200)
    assert veo.largest_legal_at_most(175) == 150
    assert veo.smallest_legal_at_least(120) == 150       # render 150, trim 30
    assert veo.smallest_legal_at_least(500) is None


# ---------------------------------------------------------------------------
# Continuity -- "accepts a still" is NOT "guarantees first-frame continuity"
# ---------------------------------------------------------------------------

def test_only_strict_first_frame_earns_a_chain():
    def _with(mode):
        class _E:
            frame_contract = fc.FrameContract(
                min_frames=9, max_frames=161, quantum=8,
                supports_multi_clip=True, continuity=mode)
        return _E()

    assert fc.can_chain(_with(fc.CONTINUITY_STRICT_FIRST_FRAME)) is True
    # A soft identity hint (HuMo) or interpolation endpoints (Veo lastFrame)
    # do NOT lock frame 0 -- those lanes take a JUMP CUT.
    assert fc.can_chain(_with(fc.CONTINUITY_SOFT_REFERENCE)) is False
    assert fc.can_chain(_with(fc.CONTINUITY_NONE)) is False


def test_single_only_engine_never_chains():
    class _E:
        frame_contract = fc.FrameContract(
            min_frames=9, max_frames=161, quantum=8,
            continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)   # but no opt-in

    assert fc.can_chain(_E()) is False


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


def test_a_still_LANE_never_opts_in_to_multi_clip():
    """``still_*`` lanes are ONE still -- operator directive, 2026-07-25.

    Guardrail landed BEFORE the first opt-in (2026-07-26, QA4). Once adapters
    may declare ``supports_multi_clip=True``, neither ``FrameContract`` nor
    the partitioner can tell a still lane from a moving-video one -- a
    contract copy-pasted onto ``still_pan`` would let the partitioner split it
    and the minter mint a second still for a lane that owns exactly one. The
    invariant holds vacuously today; this is what keeps it holding once it
    stops being vacuous.
    """
    still_lanes = [n for n in vreg.all_engine_names() if n.startswith("still_")]
    # A sweep over an empty roster asserts nothing. Pin that it is not empty --
    # two "exhaustive" sweeps in this build turned out to be theatre.
    assert still_lanes, "no still_* engines in the roster: this sweep is theatre"
    offenders = []
    for name in still_lanes:
        try:
            engine = vreg.get_engine(name)
        except Exception:  # noqa: BLE001 -- unbuildable engines are not our subject
            continue
        if fc.supports_multi_clip(engine):
            offenders.append(name)
    assert offenders == []
