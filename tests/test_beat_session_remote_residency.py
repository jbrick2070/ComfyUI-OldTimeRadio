"""A remote adapter renders a multi-segment beat without being asked for handles.

WHY (kibitz runpod-found-fixes r3, 2026-09-04). ``BeatSession`` demands a
``session_identity()`` before it renders two segments from one set of handles
-- the right gate for an engine that loads weights. A cloud adapter holds no
handles at all: ``prepare()`` returns nothing reusable and every ``render_clip``
is an independent provider call. Asking it to name a handle is a question it
cannot honestly answer, and until this change the session refused every
multi-segment beat on those lanes; ``word_razzle`` did exactly that on the
2026-09-03 pod matrix.

The fix is a DECLARATION, ``session_residency = "remote"``, read fail-closed:
an absent attribute or any other spelling is local and the demand stands.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import beat_session as bs


class _Adapter:
    """The smallest thing BeatSession drives."""

    name = "fake"

    def __init__(self):
        self.prepared_with = []
        self.rendered = []
        self.torn_down = 0

    def prepare(self, host_caps, profile, session_ctx):
        self.prepared_with.append(dict(session_ctx))
        return {}

    def render_clip(self, request, prepared):
        self.rendered.append(request)
        return {"path": "clip.mp4"}

    def teardown(self, prepared):
        self.torn_down += 1


class _Remote(_Adapter):
    name = "fake_remote"
    session_residency = "remote"


class _LocalWithoutIdentity(_Adapter):
    name = "fake_local"
    declared_isolation = "in_process"


class _Misspelt(_Adapter):
    name = "fake_cloudish"
    session_residency = "cloud"


def test_a_remote_engine_renders_two_segments_and_is_never_asked_for_handles():
    engine = _Remote()
    with bs.BeatSession(engine, beat_id="b1", segment_count=2) as session:
        assert session.identity is None
        session.begin_segment(0, owner="b1")
        session.begin_segment(1, owner="b1")
    assert engine.torn_down == 1
    assert engine.prepared_with and engine.prepared_with[0]["segment_count"] == 2


def test_CONTROL_a_local_engine_without_an_identity_is_still_refused():
    engine = _LocalWithoutIdentity()
    with pytest.raises(bs.SessionIdentityUnavailable):
        with bs.BeatSession(engine, beat_id="b1", segment_count=2):
            pass
    # Refused BEFORE loading: nothing was prepared, nothing to tear down.
    assert engine.prepared_with == []
    assert engine.torn_down == 0


def test_CONTROL_an_unknown_residency_spelling_reads_as_local():
    with pytest.raises(bs.SessionIdentityUnavailable):
        with bs.BeatSession(_Misspelt(), beat_id="b1", segment_count=2):
            pass


def test_the_refusal_names_the_declaration():
    with pytest.raises(bs.SessionIdentityUnavailable, match="session_residency"):
        with bs.BeatSession(_LocalWithoutIdentity(), beat_id="b1", segment_count=2):
            pass


def test_a_single_segment_beat_never_asked_and_still_does_not():
    engine = _LocalWithoutIdentity()
    with bs.BeatSession(engine, beat_id="b1", segment_count=1) as session:
        session.begin_segment(0, owner="b1")
    assert engine.torn_down == 1


def test_holds_local_handles_is_fail_closed():
    assert bs.holds_local_handles(_LocalWithoutIdentity())
    assert bs.holds_local_handles(_Misspelt())
    assert bs.holds_local_handles(_Adapter())
    assert not bs.holds_local_handles(_Remote())
    assert not bs.holds_local_handles(type("R", (), {"session_residency": " REMOTE "})())
