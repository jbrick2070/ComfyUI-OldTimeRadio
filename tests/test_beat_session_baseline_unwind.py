"""QA finding C-3 -- a raising BASELINE identity read must not strand the lease.

`BeatSession.open()` reads the identity twice for a multi-segment beat: once
BEFORE `prepare()` (so a refusal costs nothing) and once AFTER it, to take the
baseline. By the time the second read runs, `prepare()` has already taken the
heavy handles and the cross-process GPU lease.

If that second read raises, the exception leaves through `__enter__` -- and when
`__enter__` raises, Python NEVER calls `__exit__`. So `close()`, `teardown()`,
and the lease release inside teardown's own `finally` were all skipped, and the
lease owner is the live ComfyUI process, so the PID-liveness stale-lock reclaim
cannot free it either. Every later heavy render then blocks its full timeout.

This became reachable when `session_identity()` started doing real file I/O.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import beat_session as bs


class _Engine:
    """Minimal adapter: identity succeeds N times, then raises."""

    name = "fake_engine"
    family = "image_to_video"

    def __init__(self, raise_on_call=None, identity=("fake_engine", "r1")):
        self._identity = identity
        self._calls = 0
        self._raise_on_call = raise_on_call
        self.prepared_count = 0
        self.teardown_count = 0
        self.torn_down_with = []

    def session_identity(self):
        self._calls += 1
        if self._raise_on_call is not None and self._calls == self._raise_on_call:
            raise OSError("checkpoint vanished between reads")
        return self._identity

    def prepare(self, host_caps=None, profile=None, session_ctx=None):
        self.prepared_count += 1
        return {"handles": "HEAVY", "lease": "HELD"}

    def teardown(self, prepared):
        self.teardown_count += 1
        self.torn_down_with.append(prepared)


def _session(engine, segments=3):
    return bs.BeatSession(engine, host_caps={}, profile={},
                          beat_id="b0001", segment_count=segments)


# --- the defect ------------------------------------------------------------ #
def test_a_raising_BASELINE_read_tears_down_what_prepare_took():
    """Call 1 (pre-load) succeeds; call 2 (the baseline) raises."""
    eng = _Engine(raise_on_call=2)
    with pytest.raises(OSError, match="vanished"):
        _session(eng).open()
    assert eng.prepared_count == 1, "prepare must have run for the bug to exist"
    assert eng.teardown_count == 1, (
        "teardown did NOT run -- the GPU lease is stranded for the life of the "
        "server, which is exactly QA finding C-3")
    assert eng.torn_down_with == [{"handles": "HEAVY", "lease": "HELD"}]


def test_the_original_exception_is_re_raised_unchanged():
    """The unwind must not mask what actually went wrong."""
    eng = _Engine(raise_on_call=2)
    with pytest.raises(OSError) as e:
        _session(eng).open()
    assert "checkpoint vanished between reads" in str(e.value)


def test_the_same_unwind_happens_through_the_context_manager():
    eng = _Engine(raise_on_call=2)
    with pytest.raises(OSError):
        with _session(eng):
            pass                                  # never reached
    assert eng.teardown_count == 1


def test_a_later_close_does_not_tear_down_twice():
    """close() is idempotent; the unwind must not double-release."""
    eng = _Engine(raise_on_call=2)
    sess = _session(eng)
    with pytest.raises(OSError):
        sess.open()
    sess.close()
    assert eng.teardown_count == 1


def test_a_BaseException_in_the_baseline_also_unwinds():
    """A KeyboardInterrupt in that window strands the lease the same way."""

    class _Interrupting(_Engine):
        def session_identity(self):
            self._calls += 1
            if self._calls == 2:
                raise KeyboardInterrupt()
            return self._identity

    eng = _Interrupting()
    with pytest.raises(KeyboardInterrupt):
        _session(eng).open()
    assert eng.teardown_count == 1


# --- CONTROLS: the honest paths must be untouched -------------------------- #
def test_CONTROL_a_clean_open_does_NOT_tear_down():
    """The fix must not turn a successful open into a teardown."""
    eng = _Engine()
    sess = _session(eng)
    prepared = sess.open()
    assert prepared == {"handles": "HEAVY", "lease": "HELD"}
    assert eng.teardown_count == 0, "a healthy session was torn down at open"
    assert sess.identity is not None
    sess.close()
    assert eng.teardown_count == 1


def test_CONTROL_the_PRE_LOAD_refusal_still_costs_no_load():
    """The whole point of the first read: refuse BEFORE the weights land."""

    class _NoIdentity(_Engine):
        session_identity = None                   # duck-typed absent

    eng = _NoIdentity()
    with pytest.raises(bs.SessionIdentityUnavailable):
        _session(eng).open()
    assert eng.prepared_count == 0, "refused AFTER loading -- that is not refusing"
    assert eng.teardown_count == 0


def test_CONTROL_a_raising_PRE_LOAD_read_does_not_tear_down_nothing():
    """If the FIRST read raises, prepare never ran, so there is nothing to
    release -- and we must not invent a teardown for handles that never came."""
    eng = _Engine(raise_on_call=1)
    with pytest.raises(OSError):
        _session(eng).open()
    assert eng.prepared_count == 0
    assert eng.teardown_count == 0


def test_CONTROL_a_single_clip_session_is_unaffected():
    """Single-segment beats never read the identity at all."""
    eng = _Engine(raise_on_call=1)
    sess = bs.BeatSession(eng, host_caps={}, profile={}, beat_id="b0001")
    prepared = sess.open()
    assert prepared == {"handles": "HEAVY", "lease": "HELD"}
    assert eng.teardown_count == 0
    sess.close()
    assert eng.teardown_count == 1
