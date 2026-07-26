"""THE BEAT SESSION: one load per beat, one teardown, and a named identity.

Multi-clip coverage chunk 5. The acceptance here counts LOADER calls, never
``prepare`` calls -- an adapter that loads lazily inside ``render_clip`` would
show one prepare and three loads, and that adapter is exactly the failure this
chunk exists to prevent. One of the tests below builds that adapter on purpose
so the distinction is proven rather than asserted.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import beat_session as bs


class _StubEngine:
    """A minimal adapter that records its own lifecycle.

    ``load`` is the LOADER -- the expensive thing. ``prepare`` calls it, the
    way ``motion_common.prepare`` does after taking the GPU lease.
    """

    name = "stub_session_engine"
    # ``abstract`` declares no required request inputs (schemas.py), so the
    # driver's family-input gate passes and these tests stay about the
    # LIFECYCLE rather than about request shape.
    family = "abstract"

    def __init__(self, identity=("stub_session_engine", "recipe_a", "w1")):
        self._identity = identity
        self.load_calls = 0
        self.prepare_calls = 0
        self.teardown_calls = 0
        self.render_calls = 0
        self.seen_ctx = []
        self.raise_on_segment = None
        self.raise_on_teardown = False

    def session_identity(self):
        return self._identity

    def set_identity(self, identity):
        self._identity = identity

    def load(self):
        self.load_calls += 1

    def prepare(self, host_caps, profile, session_ctx):
        self.prepare_calls += 1
        self.seen_ctx.append(dict(session_ctx or {}))
        self.load()
        return {"engine_id": self.name, "handles": object()}

    def render_clip(self, request, prepared):
        idx = self.render_calls
        self.render_calls += 1
        if self.raise_on_segment == idx:
            raise RuntimeError("segment %d exploded" % idx)
        return {"segment": idx, "prepared": prepared}

    def teardown(self, prepared):
        self.teardown_calls += 1
        if self.raise_on_teardown:
            raise RuntimeError("teardown exploded")


class _NoIdentityEngine(_StubEngine):
    """An adapter that cannot say what its handles are."""

    name = "stub_no_identity"
    session_identity = None


class _LazyLoaderEngine(_StubEngine):
    """The adapter the acceptance is aimed at: prepare takes the lease but
    the weights are loaded on first use, INSIDE render_clip."""

    name = "stub_lazy_loader"

    def prepare(self, host_caps, profile, session_ctx):
        self.prepare_calls += 1
        self.seen_ctx.append(dict(session_ctx or {}))
        return {"engine_id": self.name, "handles": None}

    def render_clip(self, request, prepared):
        self.load()                       # every clip pays for the weights
        return super().render_clip(request, prepared)


# ---------------------------------------------------------------------------
# THE ACCEPTANCE: one LOAD per beat, not one per segment
# ---------------------------------------------------------------------------

def test_ONE_LOAD_for_a_multi_segment_beat():
    """Four segments, one checkpoint load. The whole chunk in one assertion."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=4) as session:
        for index in range(4):
            prepared = session.begin_segment(index)
            eng.render_clip({"segment": index}, prepared)

    assert eng.load_calls == 1, "the beat reloaded its weights per segment"
    assert eng.prepare_calls == 1
    assert eng.teardown_calls == 1
    assert eng.render_calls == 4


def test_every_segment_renders_from_the_SAME_handles():
    """Not merely 'loaded once' -- the later segments must actually get the
    object the first one loaded, or the count is meaningless."""
    eng = _StubEngine()
    handed = []
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        for index in range(3):
            handed.append(session.begin_segment(index))
    assert len({id(obj) for obj in handed}) == 1


def test_a_lazy_loader_is_CAUGHT_by_counting_loads_not_prepares():
    """Why the acceptance counts LOADER calls.

    This adapter takes the session bracket perfectly -- one prepare, one
    teardown -- and still loads its weights three times, because it loads
    inside ``render_clip``. Counting ``prepare`` calls would call this a pass.
    """
    eng = _LazyLoaderEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        for index in range(3):
            eng.render_clip({}, session.begin_segment(index))

    assert eng.prepare_calls == 1         # the bracket looks perfect...
    assert eng.load_calls == 3            # ...and the weights loaded 3 times


# ---------------------------------------------------------------------------
# ONE outer finally
# ---------------------------------------------------------------------------

def test_teardown_runs_ONCE_when_a_middle_segment_raises():
    eng = _StubEngine()
    eng.raise_on_segment = 1
    with pytest.raises(RuntimeError, match="segment 1 exploded"):
        with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
            for index in range(3):
                eng.render_clip({}, session.begin_segment(index))

    assert eng.teardown_calls == 1        # not zero (leak), not two (double)
    assert eng.load_calls == 1


def test_close_is_idempotent():
    eng = _StubEngine()
    session = bs.BeatSession(eng, beat_id="b001", segment_count=2)
    session.open()
    session.close()
    session.close()
    assert eng.teardown_calls == 1


def test_a_teardown_that_raises_never_escapes_the_session():
    """A raise out of teardown would mask the real render error."""
    eng = _StubEngine()
    eng.raise_on_teardown = True
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        session.begin_segment(0)
    assert eng.teardown_calls == 1


def test_a_failed_prepare_leaves_nothing_to_tear_down():
    """Historical behaviour, preserved: prepare raising means no handles were
    taken, so teardown must not run on a half-open session."""
    eng = _StubEngine()

    def _boom(host_caps, profile, session_ctx):
        eng.prepare_calls += 1
        raise RuntimeError("prepare exploded")

    eng.prepare = _boom
    with pytest.raises(RuntimeError, match="prepare exploded"):
        with bs.BeatSession(eng, beat_id="b001", segment_count=2):
            pass
    assert eng.teardown_calls == 0


# ---------------------------------------------------------------------------
# The identity is the safety
# ---------------------------------------------------------------------------

def test_a_multi_segment_engine_without_an_identity_is_REFUSED_before_loading():
    eng = _NoIdentityEngine()
    with pytest.raises(bs.SessionIdentityUnavailable, match="session_identity"):
        bs.BeatSession(eng, beat_id="b001", segment_count=3).open()
    assert eng.load_calls == 0, "it refused AFTER loading, which is too late"


def test_a_single_segment_engine_never_needs_an_identity():
    """Nothing can drift between one segment and itself."""
    eng = _NoIdentityEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=1) as session:
        assert session.begin_segment(0) is not None
    assert eng.load_calls == 1
    assert eng.teardown_calls == 1


def test_a_RECIPE_CHANGE_mid_beat_is_TERMINAL():
    """The operator flips OTR_LTX_AV_RECIPE while segment 1 is on the GPU."""
    eng = _StubEngine(identity=("ltx_av", "daily", "unet_a"))
    with pytest.raises(bs.SessionIdentityDrift, match="changed identity mid-beat"):
        with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
            session.begin_segment(0)
            eng.set_identity(("ltx_av", "hero", "unet_a"))
            session.begin_segment(1)
    assert eng.teardown_calls == 1        # and it still let go of the handles


def test_a_WEIGHT_change_mid_beat_is_TERMINAL():
    eng = _StubEngine(identity=("ltx_av", "daily", "unet_a"))
    with pytest.raises(bs.SessionIdentityDrift):
        with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
            session.begin_segment(0)
            eng.set_identity(("ltx_av", "daily", "unet_b"))
            session.begin_segment(1)


def test_identity_is_RE_READ_per_segment_not_cached():
    """A cached identity would prove only that the beat started consistent."""
    reads = []

    class _Counting(_StubEngine):
        def session_identity(self):
            reads.append(len(reads))
            return self._identity

    eng = _Counting()
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        for index in range(3):
            session.begin_segment(index)
    assert len(reads) == 4                # once at open, once per segment


def test_session_identity_reads_a_method_or_an_attribute_or_nothing():
    class _Method:
        def session_identity(self):
            return ("a", "b")

    class _Attr:
        session_identity = ("a", "b")

    class _Bare:
        pass

    assert bs.session_identity(_Method()) == ("a", "b")
    assert bs.session_identity(_Attr()) == ("a", "b")
    assert bs.session_identity(_Bare()) is None


def test_an_identity_that_RAISES_is_not_swallowed():
    """An adapter that cannot describe its handles is a reason to stop."""
    class _Broken:
        name = "broken_identity"

        def session_identity(self):
            raise RuntimeError("cannot read the recipe")

    with pytest.raises(RuntimeError, match="cannot read the recipe"):
        bs.session_identity(_Broken())


# ---------------------------------------------------------------------------
# The prepare-time context
# ---------------------------------------------------------------------------

def test_prepare_learns_the_segment_count_BEFORE_it_loads():
    """An adapter must be able to size for four clips rather than one, and it
    has to learn that before the weights land."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b042", segment_count=4):
        pass
    assert eng.seen_ctx == [{"beat_id": "b042", "segment_count": 4,
                             "multi_clip": True}]


def test_a_single_clip_beat_says_so_in_its_context():
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b042"):
        pass
    assert eng.seen_ctx[0]["multi_clip"] is False
    assert eng.seen_ctx[0]["segment_count"] == 1


def test_reopening_an_open_session_would_strand_handles_and_is_refused():
    eng = _StubEngine()
    session = bs.BeatSession(eng, beat_id="b001", segment_count=2)
    session.open()
    try:
        with pytest.raises(bs.SessionError, match="already open"):
            session.open()
    finally:
        session.close()
    assert eng.load_calls == 1


def test_begin_segment_before_open_is_refused():
    eng = _StubEngine()
    session = bs.BeatSession(eng, beat_id="b001", segment_count=2)
    with pytest.raises(bs.SessionError, match="not open"):
        session.begin_segment(0)


# ---------------------------------------------------------------------------
# The wiring: _render_one is the session's only lifecycle path
# ---------------------------------------------------------------------------

from nodes._otr_video_engines import render_driver as rd  # noqa: E402


class _DriverStub(_StubEngine):
    """The stub as the render driver needs it: usable, and canonicalizing."""

    name = "stub_driver_session"

    def assert_usable(self, host_caps, profile, request_template=None):
        return True

    def canonicalize(self, raw, request, profile):
        return {"clip_id": "c%s" % raw["segment"], "type": "video"}


def _install(monkeypatch, engine):
    monkeypatch.setattr(rd._vreg, "is_registered",
                        lambda name: name == engine.name)
    monkeypatch.setattr(rd._vreg, "get_engine", lambda name: engine)


def test_render_one_with_a_session_reuses_the_handles_and_does_NOT_tear_down(
        monkeypatch):
    eng = _DriverStub()
    _install(monkeypatch, eng)
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        for index in range(3):
            rd._render_one(eng.name, {"shot_id": "s1"}, force_oom=False,
                           segment=bs.SegmentSlot(session, index))
        # INSIDE the session the handles are still held -- a per-clip teardown
        # here is precisely the bug: segment 2 would reload what segment 1 had.
        assert eng.teardown_calls == 0
        assert eng.load_calls == 1
    assert eng.teardown_calls == 1        # released once, by the outer finally
    assert eng.render_calls == 3


def test_render_one_without_a_session_keeps_the_historical_bracket(monkeypatch):
    """Every beat today. One prepare, one render, one teardown, per clip."""
    eng = _DriverStub()
    _install(monkeypatch, eng)
    rd._render_one(eng.name, {"shot_id": "s1"}, force_oom=False)
    rd._render_one(eng.name, {"shot_id": "s2"}, force_oom=False)
    assert (eng.prepare_calls, eng.load_calls, eng.teardown_calls) == (2, 2, 2)


def test_assert_usable_runs_PER_SEGMENT(monkeypatch):
    """A later segment's canvas is not the first one's, so the usability gate
    is per request even though the handles are per beat."""
    eng = _DriverStub()
    seen = []
    eng.assert_usable = lambda host_caps, profile, request_template=None: \
        seen.append(request_template)
    _install(monkeypatch, eng)
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        rd._render_one(eng.name, {"shot_id": "s1"}, force_oom=False,
                       segment=bs.SegmentSlot(session, 0))
        rd._render_one(eng.name, {"shot_id": "s2"}, force_oom=False,
                       segment=bs.SegmentSlot(session, 1))
    assert [r["shot_id"] for r in seen] == ["s1", "s2"]


def test_a_session_holding_a_DIFFERENT_engine_is_refused(monkeypatch):
    """One session, one engine -- otherwise a beat renders half in one model
    and half in another and the handles prove nothing."""
    rendering = _DriverStub()
    other = _DriverStub()
    _install(monkeypatch, rendering)
    with bs.BeatSession(other, beat_id="b001", segment_count=2) as session:
        with pytest.raises(rd.RenderError, match="one session, one engine"):
            rd._render_one(rendering.name, {"shot_id": "s1"}, force_oom=False,
                           segment=bs.SegmentSlot(session, 0))


def test_render_shot_threads_the_session_through(monkeypatch):
    eng = _DriverStub()
    _install(monkeypatch, eng)
    shot = {"shot_id": "s1", "engine_id": eng.name}
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        rd.render_shot(shot, {"shot_id": "s1"},
                       segment=bs.SegmentSlot(session, 0, "b001"))
        rd.render_shot(shot, {"shot_id": "s1"},
                       segment=bs.SegmentSlot(session, 1, "b001"))
        assert eng.teardown_calls == 0
    assert eng.load_calls == 1
    assert eng.teardown_calls == 1


def test_identity_drift_through_the_driver_fails_the_render_LOUD(monkeypatch):
    """The drift is terminal at the render boundary too, not just in the
    session object -- render_shot wraps it as a RenderError with no fallback."""
    eng = _DriverStub(identity=("stub", "daily", "w1"))
    _install(monkeypatch, eng)
    shot = {"shot_id": "s1", "engine_id": eng.name}
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        rd.render_shot(shot, {"shot_id": "s1"},
                       segment=bs.SegmentSlot(session, 0))
        eng.set_identity(("stub", "hero", "w1"))
        with pytest.raises(rd.RenderError, match="fallbacks are disabled"):
            rd.render_shot(shot, {"shot_id": "s1"},
                           segment=bs.SegmentSlot(session, 1))


# ---------------------------------------------------------------------------
# The segment index is the CALLER'S claim, and it is bounded (chunk-5 QA lens)
# ---------------------------------------------------------------------------

def test_a_segment_past_the_opened_COUNT_is_refused():
    """The count is what the adapter was told at prepare time."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        session.begin_segment(0)
        session.begin_segment(1)
        with pytest.raises(bs.SessionError, match="opened for 2 segment"):
            session.begin_segment(2)


def test_a_NEGATIVE_segment_is_refused():
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        with pytest.raises(bs.SessionError, match="opened for 2 segment"):
            session.begin_segment(-1)


def test_a_REPEATED_segment_is_refused():
    """A repeat is a retry, and this build does not retry -- it fails LOUD."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        session.begin_segment(0)
        session.begin_segment(1)
        with pytest.raises(bs.SessionError, match="already served segment 1"):
            session.begin_segment(1)


def test_a_BACKWARDS_segment_is_refused():
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=3) as session:
        session.begin_segment(0)
        session.begin_segment(2)
        with pytest.raises(bs.SessionError, match="already served segment 2"):
            session.begin_segment(1)


def test_a_session_held_into_ANOTHER_BEAT_is_refused():
    """Two beats on the same engine pass the engine check trivially, so the
    beat is what the owner check is for."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        session.begin_segment(0, owner="b001")
        with pytest.raises(bs.SessionError, match="belongs to beat 'b001'"):
            session.begin_segment(1, owner="b002")


def test_an_owner_nobody_claims_is_not_checked():
    """A caller that names no beat has not made the claim -- it is not
    silently blessed either, there is simply nothing to compare."""
    eng = _StubEngine()
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        assert session.begin_segment(0) is not None
        assert session.begin_segment(1, owner=None) is not None


def test_a_SLOT_cannot_be_built_without_saying_which_segment():
    """The reason this is one argument and not three: 'a session with no
    segment index' is not a state a caller can construct, so the driver has no
    error branch to forget."""
    eng = _StubEngine()
    session = bs.BeatSession(eng, beat_id="b001", segment_count=2)
    with pytest.raises(TypeError):
        bs.SegmentSlot(session)           # index is not optional


def test_the_driver_passes_the_owner_through_to_the_session(monkeypatch):
    eng = _DriverStub()
    _install(monkeypatch, eng)
    shot = {"shot_id": "s1", "engine_id": eng.name}
    with bs.BeatSession(eng, beat_id="b001", segment_count=2) as session:
        rd.render_shot(shot, {"shot_id": "s1"},
                       segment=bs.SegmentSlot(session, 0, "b001"))
        with pytest.raises(rd.RenderError, match="fallbacks are disabled"):
            rd.render_shot(shot, {"shot_id": "s1"},
                           segment=bs.SegmentSlot(session, 1, "b002"))
