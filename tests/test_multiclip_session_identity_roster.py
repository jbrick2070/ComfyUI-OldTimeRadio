"""An engine the partitioner can split MUST be able to name its own handles.

THE LIVE FAILURE (45-word campaign, leg `ltx_video`, 2026-07-29, 730s in):

    SessionIdentityUnavailable: engine 'ltx_video' would render 2 segments from
    ONE set of handles but declares no session_identity(), so nothing can prove
    the model segment 2 renders with is the one segment 1 loaded. Declare the
    identity (engine, recipe, weights) or render this beat single-clip.
    NO FALLBACK.

The refusal is right. `MotionEngineBase.prepare()` calls `self.load()` once per
beat, so a multi-segment beat really does render every segment from one set of
handles, and an engine that cannot describe those handles cannot prove they did
not move underneath it.

What was wrong is WHERE it was discovered. `session_identity()` was added one
engine at a time -- ltx_8gb first, then wan_i2v, wan_ti2v and humo across the
wiring block -- and nothing ever checked the REST of the roster. So an engine
the partitioner is perfectly willing to split reached a GPU leg, wrote a
script, minted its stills, assembled its audio, and only THEN discovered it
could not render. That is the most expensive possible place to learn it.

This gate asks the partitioner itself which engines it will split, and requires
an identity from exactly those. It costs a second and it fails in the suite.

SCOPE, HONESTLY. The gate covers engines that hold LOCAL handles -- the ones
whose weights CAN move under a running beat. Remote-dispatch engines DECLARE
`session_residency = "remote"` (no local handles; every render is an
independent provider call, 2026-09-04) and are covered by
`test_the_cloud_gap_is_known_and_named` below instead of being silently
excluded.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import beat_session as bs
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg


FPS = 25

#: A beat long enough that any engine with a real per-clip ceiling has to split
#: it. Thirty seconds is not exotic -- the campaign's 45-word episodes produce
#: beats well past every heavy lane's cap, which is how `ltx_video` found this.
LONG_BEAT = FPS * 30


def _engines():
    return sorted(vreg.all_engine_names())


def _contract(name):
    return fc.frame_contract_for(vreg.get_engine(name))


def _splits(name):
    """Will the partitioner split a 30-second beat on this engine?"""
    try:
        plan = cp.partition_beat(LONG_BEAT, _contract(name))
    except Exception:  # noqa: BLE001 -- a refusal is not a split
        return False
    return bool(plan.is_multi_clip)


def _holds_local_handles(name):
    """Does this engine load handles in THIS process (or a sidecar)?

    Keyed on the engine's DECLARED ``session_residency`` (2026-09-04): an
    adapter that says ``"remote"`` holds no local handles -- ``prepare()``
    returns nothing reusable and each ``render_clip`` is an independent
    provider call -- so nothing can move under a running beat and nothing it
    can honestly be asked to name. Anything that does not say so is local,
    fail-closed, exactly as ``beat_session`` reads it.
    """
    return bs.holds_local_handles(vreg.get_engine(name))


IN_SCOPE = [n for n in _engines() if _splits(n) and _holds_local_handles(n)]
CLOUD_SPLITTERS = [n for n in _engines()
                   if _splits(n) and not _holds_local_handles(n)]


# The two PROPERTIES this file is about, each in exactly one place. The roster
# sweeps below apply them to the real registry; the CONTROLS at the bottom
# apply them to fakes built to fail them. One definition, both directions --
# a weakened predicate cannot pass the controls, and the sweep cannot drift
# away from what the controls proved.

def gate_would_accept(engine):
    """The gate's condition: can this adapter be asked what its handles are?"""
    return hasattr(engine, "session_identity")


def identity_is_stable(engine):
    """Two quiet reads, no load between them, must agree."""
    return bs.session_identity(engine) == bs.session_identity(engine)


#: The DECLARED remote set: engines the partitioner will split that say
#: `session_residency = "remote"` (no local handles, so BeatSession asks them
#: for no identity). Named once, asserted from two directions below --
#: the tripwire compares the live roster against it, and the control re-asserts
#: it independently, so weakening either one alone does not stop the other
#: noticing a substitution inside the set.
EXPECTED_CLOUD_GAP = {
    "cloud_seedance_2",
    "cloud_vidu_q2_pro_fast_720p",
    "cloud_wan_i2v",
    "cloud_wan_i2v_audio",
    "google_omni_video",
    "google_veo_video",
    "word_razzle",
}


@pytest.mark.parametrize("name", _engines())
def test_a_local_engine_the_partitioner_will_split_declares_an_identity(name):
    """THE GATE. Read both facts from the SAME roster entry so they cannot
    drift apart: whether the partitioner splits it, and whether it can say what
    it is about to reuse.

    Asserts the DECLARATION, not a successful read. `session_identity()` is
    documented as deliberately un-caught -- an adapter whose identity read
    raises has said it cannot describe its handles -- and off-box, with no
    weights installed, `ltx_8gb` raises `EngineUnusable` resolving its recipe.
    Requiring a clean read here would make this a weights-installed test, and
    it would then pass or fail for reasons that have nothing to do with the
    contract it exists to hold.
    """
    if name not in IN_SCOPE:
        pytest.skip("%s: splits=%s local_handles=%s"
                    % (name, _splits(name), _holds_local_handles(name)))
    engine = vreg.get_engine(name)
    assert gate_would_accept(engine), (
        "engine %r is splittable -- the partitioner hands it a multi-segment "
        "plan for a 30-second beat -- and holds local handles, but declares no "
        "session_identity(). BeatSession.open() will refuse it with "
        "SessionIdentityUnavailable at render time, AFTER the script, the "
        "stills and the audio have been paid for. Declare the identity "
        "(engine, recipe, weights), or give the engine a frame contract that "
        "says it cannot be split." % name
    )


@pytest.mark.parametrize("name", _engines())
def test_an_identity_that_reads_is_a_nonempty_tuple_of_strings(name):
    engine = vreg.get_engine(name)
    try:
        identity = bs.session_identity(engine)
    except Exception:  # noqa: BLE001 -- covered by the declaration gate above
        pytest.skip("%s cannot resolve its identity off-box" % name)
    if identity is None:
        pytest.skip("%s declares no session identity" % name)
    assert isinstance(identity, tuple) and identity, (
        "%s: session_identity() must be a non-empty tuple" % name)
    assert all(isinstance(part, str) for part in identity), (
        "%s: every part of a session identity must be a string" % name)


@pytest.mark.parametrize("name", _engines())
def test_an_identity_is_stable_when_nothing_moved(name):
    """Read twice, no load in between, and the answer must not change.

    `session_identity` is deliberately uncached so it NOTICES a weight that
    moved; that only works if a quiet read is repeatable. An identity carrying
    a timestamp, an object id, or a freshly-made temp path would make every
    multi-segment beat die as SessionIdentityDrift on a beat where nothing
    moved at all.
    """
    engine = vreg.get_engine(name)
    try:
        first = bs.session_identity(engine)
    except Exception:  # noqa: BLE001
        pytest.skip("%s cannot resolve its identity off-box" % name)
    if first is None:
        pytest.skip("%s declares no session identity" % name)
    assert identity_is_stable(engine), (
        "%s: two consecutive session_identity() reads disagree with no load "
        "between them -- BeatSession would report SessionIdentityDrift on a "
        "beat where nothing moved" % name)


def test_this_gate_is_not_vacuous():
    """A sweep that finds nothing passes every gate built on it. This build has
    paid for that before -- the W3b control that watched exactly one engine
    while a second quietly grew an identity. Assert the roster is really being
    billed -- DERIVED from the registry's own CAPABILITIES table, never a
    hand-typed count that rots when an engine is added or retired
    (PRODUCTION_SPRINT_LESSONS on count assertions; rip-sfx 2026-08-06)."""
    names = _engines()
    assert names, "registry discovered no engines at all"
    assert set(names) == set(vreg.CAPABILITIES), (
        "live roster and CAPABILITIES disagree: only-live=%s only-declared=%s"
        % (sorted(set(names) - set(vreg.CAPABILITIES)),
           sorted(set(vreg.CAPABILITIES) - set(names))))
    # `wan_i2v` (the Wan 2.2 14B) was an anchor here until it was retired
    # 2026-08-26 for not fitting the 14.5 GiB envelope. Requiring it in the live
    # roster now asserts a falsehood, so the row is REMOVED rather than pointed
    # at `wan_ti2v` -- the 5B TI2V lane is a different model and already anchors
    # itself on the line above. Nothing is orphaned: the retirement is asserted
    # positively three lines down, where RETIRED_ENGINE_IDS must not intersect
    # the live roster, which is the stronger of the two checks anyway.
    for anchor in ("humo", "ltx_video", "ltx_8gb", "wan_ti2v",
                   "fastwan_8gb", "viz_green", "still_flat"):
        assert anchor in names, anchor
    from nodes._otr_shared.public_engines import RETIRED_ENGINE_IDS
    assert not (set(names) & RETIRED_ENGINE_IDS), (
        "retired engine id(s) re-registered: %s"
        % sorted(set(names) & RETIRED_ENGINE_IDS))
    assert len(IN_SCOPE) >= 7, (
        "only %d local engine(s) split a 30-second beat (%s) -- the identity "
        "gate above is close to vacuous" % (len(IN_SCOPE), ", ".join(IN_SCOPE)))


def test_the_engines_that_forced_this_gate_are_still_covered_by_it():
    """`ltx_video` is the engine that failed live; `ltx_8gb`, `humo` and
    `wan_ti2v` are three that already had an identity. If a contract change
    ever drops one of them out of scope, this says so rather than letting the
    gate quietly stop watching it.

    `wan_i2v` was a fourth until the 14B was retired 2026-08-26. Its row is
    dropped rather than retargeted -- `wan_ti2v` is a different model and is
    already listed on its own account -- and it cost nothing to drop, because
    the `continue` guard below had silently stopped billing the row the moment
    the engine left the registry.
    """
    for name in ("ltx_video", "ltx_audio_in", "ltx_8gb", "humo", "wan_ti2v"):
        if name not in vreg.all_engine_names():
            continue
        assert name in IN_SCOPE, (
            "%s is no longer a splittable local engine, so the identity gate "
            "no longer covers it. If that is intended, say so here; if it is a "
            "frame-contract regression, this is the alarm." % name)


def test_the_cloud_gap_is_known_and_named():
    """The gate above deliberately does NOT cover remote engines, and that
    exclusion is DECLARED, not inferred: each of these adapters says
    ``session_residency = "remote"`` -- it holds no local handles, so
    ``BeatSession`` does not ask it to name one (2026-09-04). Before that
    the session refused every multi-segment beat on these lanes for a
    question they could not answer; ``word_razzle`` did exactly that on the
    2026-09-03 pod matrix.

    This test fails the day that set CHANGES, which is the point: a new
    remote engine must declare itself here, on purpose, and a local engine
    that starts claiming "remote" to dodge the gate trips the residency
    control below.
    """
    assert set(CLOUD_SPLITTERS) == EXPECTED_CLOUD_GAP, (
        "the set of splittable engines declaring session_residency='remote' "
        "changed: %s. If an engine ARRIVED here, confirm it holds no local "
        "handles (prepare() returns nothing reusable, teardown is a no-op). "
        "If one LEFT because it grew local handles, it now belongs in the "
        "gate above."
        % sorted(CLOUD_SPLITTERS))


# --------------------------------------------------------------------------
# CONTROLS -- test the gate, not just the roster.
#
# A first mutation round left five survivors, and every one of them was the
# same defect: the ASSERTIONS above were weakened and nothing noticed, because
# a gate that only ever sees passing input cannot tell whether it would reject
# anything. `assert hasattr(engine, "name")` passes for all 31 engines exactly
# as happily as `assert hasattr(engine, "session_identity")`.
#
# So the predicates are exercised here against fakes built to fail them. These
# controls are what make the roster sweep above mean something.
# --------------------------------------------------------------------------

class _FakeEngine:
    """The smallest thing the gate's predicates read."""

    def __init__(self, name, *, isolation=None, identity=None, residency=None):
        self.name = name
        if isolation is not None:
            self.declared_isolation = isolation
        if identity is not None:
            self.session_identity = identity
        if residency is not None:
            self.session_residency = residency


def test_CONTROL_the_gate_rejects_a_local_splittable_engine_with_no_identity():
    """If this passes while the gate is neutered, the gate is decoration.

    Drives `gate_would_accept` -- the SAME predicate the roster sweep uses --
    so a weakened condition (`hasattr(engine, "name")`, which every engine
    satisfies) fails here instead of sailing through 31 passing parametrized
    cases.
    """
    assert not gate_would_accept(
        _FakeEngine("fake_capped_lane", isolation="in_process"))
    assert gate_would_accept(
        _FakeEngine("fake_capped_lane", isolation="in_process",
                    identity=lambda: ("fake_capped_lane", "recipe", "w=1")))


def test_CONTROL_local_handles_are_told_apart_from_remote_dispatch():
    """`_holds_local_handles` keys on a DECLARED residency. A version that
    answers True for everything would drag the remote engines into a gate
    they cannot satisfy; one that answers False for everything would empty
    the gate silently; and an unknown spelling, or no declaration at all,
    must read as LOCAL -- the demand stands unless the adapter said
    otherwise."""
    local = _FakeEngine("fake_local", isolation="in_process")
    sidecar = _FakeEngine("fake_sidecar", isolation="sidecar_optional")
    remote = _FakeEngine("fake_remote", residency="remote")
    misspelt = _FakeEngine("fake_cloudish", residency="cloud")
    undeclared = _FakeEngine("fake_silent")
    assert bs.holds_local_handles(local)
    assert bs.holds_local_handles(sidecar)
    assert not bs.holds_local_handles(remote)
    assert bs.holds_local_handles(misspelt)
    assert bs.holds_local_handles(undeclared)


def test_a_remote_declaration_never_sits_on_an_engine_that_loads_weights():
    """The declaration is only honest on an adapter with nothing to name.
    A local engine (one that sets `declared_isolation`, i.e. loads weights
    in-process or in a sidecar) claiming "remote" would dodge the gate;
    this pins that no registered engine does both."""
    for name in _engines():
        engine = vreg.get_engine(name)
        if not bs.holds_local_handles(engine):
            assert not getattr(engine, "declared_isolation", None), (
                "%s declares session_residency='remote' AND a local isolation"
                % name)


def test_CONTROL_the_stability_check_catches_an_identity_that_drifts():
    """A drifting identity must FAIL a two-read comparison. Without this
    control, `assert x is not None` reads exactly like `assert x == first`."""
    counter = {"n": 0}

    def drifting():
        counter["n"] += 1
        return ("fake_drifter", "reading-%d" % counter["n"])

    drifter = _FakeEngine("fake_drifter", isolation="in_process",
                          identity=drifting)
    assert not identity_is_stable(drifter), (
        "a drifting identity passed the stability predicate -- the predicate "
        "is no longer comparing two reads, so every real lane's drift check "
        "is decoration")

    stable = _FakeEngine("fake_stable", isolation="in_process",
                         identity=lambda: ("fake_stable", "w=1"))
    assert identity_is_stable(stable), (
        "a genuinely stable identity failed the stability predicate")


def test_CONTROL_a_bare_string_identity_is_not_enough_to_name_handles():
    """`beat_session.session_identity` WRAPS a bare string into a 1-tuple, so a
    name-only identity satisfies "is a non-empty tuple of strings" while naming
    nothing that can move. The real lanes must carry their weights."""
    bare = _FakeEngine("fake_bare", isolation="in_process",
                       identity=lambda: "fake_bare")
    wrapped = bs.session_identity(bare)
    assert wrapped == ("fake_bare",), (
        "beat_session no longer wraps a bare string; this control's premise "
        "changed and the assertion below needs rereading")
    assert len(wrapped) == 1, "a name-only identity names no handles"


@pytest.mark.parametrize("name", ["ltx_video", "ltx_audio_in"])
def test_the_live_lanes_carry_their_WEIGHTS_not_just_their_name(name):
    """The two lanes added after the live failure must name what they load.

    An identity of just `(engine_name,)` passes every structural check above
    and detects NOTHING: it is identical before and after a weight is swapped,
    which is the entire failure this machinery exists to catch.
    """
    if name not in vreg.all_engine_names():
        pytest.skip("%s not registered" % name)
    engine = vreg.get_engine(name)
    identity = bs.session_identity(engine)
    assert identity is not None
    assert len(identity) > 3, (
        "%s's identity is %r -- too thin to name its handles. It must carry a "
        "receipt per required weight, or a swapped checkpoint is invisible to "
        "it." % (name, identity))
    assert any("transformer" in part for part in identity), (
        "%s's identity does not mention its transformer weight: %r"
        % (name, identity))


@pytest.mark.parametrize("name", ["ltx_video", "ltx_audio_in"])
def test_a_weight_receipt_is_stable_and_moves_when_the_file_moves(name, tmp_path):
    """The receipt is the part that has to NOTICE a swap.

    Off-box the real weights are absent, so the roster-wide stability test
    above never reaches the stat at all -- it returns the missing-file receipt
    and compares two identical absences. This drives the receipt against a real
    file so both halves of its contract are exercised: two quiet reads agree,
    and rewriting the file changes the answer.
    """
    if name not in vreg.all_engine_names():
        pytest.skip("%s not registered" % name)
    engine = vreg.get_engine(name)
    weight = tmp_path / "fake_weight.gguf"
    weight.write_bytes(b"x" * 2048)

    first = engine._weight_receipt(str(weight))
    assert engine._weight_receipt(str(weight)) == first, (
        "two quiet reads of the same untouched file disagree -- every "
        "multi-segment beat would die as SessionIdentityDrift")
    assert first[0] == "fake_weight.gguf"
    assert first[1] == 2048

    weight.write_bytes(b"y" * 4096)
    assert engine._weight_receipt(str(weight)) != first, (
        "the receipt did not change when the file did -- a swapped weight "
        "would be invisible")

    missing = engine._weight_receipt(str(tmp_path / "not_here.gguf"))
    assert missing == ("not_here.gguf", -1, -1), (
        "a missing weight must return a NAMED absence, not raise: "
        "assert_usable owns that refusal and says it far better")
    assert engine._weight_receipt("") == ("<unresolved>", -1, -1)


def test_CONTROL_the_cloud_gap_tripwire_is_an_exact_set_not_a_shrug():
    """The tripwire above is only a tripwire if it is an EQUALITY. A weakened
    version (`>=`, or an `or True`) would keep passing while the gap silently
    grew, so the same set is pinned here by size and by a named member."""
    assert sorted(CLOUD_SPLITTERS) == sorted(EXPECTED_CLOUD_GAP), (
        "the cloud gap moved: live=%s expected=%s. Asserted here as well as in "
        "the tripwire so a SUBSTITUTION inside the set -- one engine leaving "
        "as another arrives, size unchanged -- cannot pass unnoticed if either "
        "assertion is weakened."
        % (sorted(CLOUD_SPLITTERS), sorted(EXPECTED_CLOUD_GAP)))
    # No hand-typed count here: the exact-set equality above already pins
    # membership, and a count adds nothing but drift the next time an engine
    # arrives or retires (rip-sfx 2026-08-06 removed five). The retired ids
    # must never reappear in this set:
    from nodes._otr_shared.public_engines import RETIRED_ENGINE_IDS
    assert not (set(CLOUD_SPLITTERS) & RETIRED_ENGINE_IDS), (
        "retired engine id(s) back in the cloud gap: %s"
        % sorted(set(CLOUD_SPLITTERS) & RETIRED_ENGINE_IDS))
    assert "word_razzle" in CLOUD_SPLITTERS, (
        "word_razzle is a CLOUD i2v engine that the 45-word campaign wrongly "
        "carried in its local roster; it belongs in this set")
    # This row guards the SUBSTRING hazard: `cloud_wan_i2v` is a legitimate
    # member of the gap above, and a local Wan lane sharing most of that name
    # must never be swept in with it. The local lane named here was `wan_i2v`
    # until the 14B was retired 2026-08-26; `wan_ti2v` takes its place because
    # the hazard is unchanged and the property was checked rather than assumed
    # -- the 5B declares `sidecar_optional` isolation, so `_holds_local_handles`
    # answers True for it and it lands in IN_SCOPE, never in CLOUD_SPLITTERS.
    for name in ("ltx_video", "ltx_audio_in", "humo", "wan_ti2v"):
        assert name not in CLOUD_SPLITTERS, (
            "%s holds local handles and must never appear in the cloud gap"
            % name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
