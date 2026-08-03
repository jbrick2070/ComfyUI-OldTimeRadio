"""THE BEAT SESSION: one model load per BEAT, not one per segment.

Multi-clip coverage chunk 5 (2026-07-26). Chunks 3-4 gave a beat a plan of
several render segments and a still for each one. This module is what makes
executing that plan affordable: the heavy handles -- the MODEL/CLIP/VAE the
adapter loads, and the shared single-heavy-engine lease it takes to load them
(``motion_common.prepare``, AS-3) -- are acquired ONCE for the whole beat and
released ONCE, in a single outer ``finally``, no matter how many segments run
or which of them raises.

WHY THIS IS NOT A CACHE. ``_render_one`` bracketed every clip with
``prepare`` -> ``render_clip`` -> ``teardown``. That is exactly right for one
clip per beat and ruinous for four: each ``prepare`` re-acquires the GPU lease
and reloads weights, each ``teardown`` detaches them and then waits for VRAM
to settle. A three-segment beat would load one checkpoint three times and pay
three stability waits to render one beat of video. Keying a cache on the
engine id would fix the cost and buy a worse defect -- handles outliving the
recipe or the weights they were loaded from. So this is a SESSION with an
explicit lifetime and a named identity, never an ambient cache.

THE IDENTITY IS THE SAFETY. :func:`session_identity` is what the handles ARE:
engine, recipe, weights. It is captured when the session opens and RE-DERIVED
before every segment. If it moved -- an operator flipped ``OTR_LTX_AV_RECIPE``
between segments, a wrapper re-resolved a UNET -- the session REFUSES rather
than rendering segment 3 of a beat with a model that is not the one segments 1
and 2 used. Half a beat in one recipe and half in another is not a beat.

AND AN ADAPTER MUST BE ABLE TO NAME ITS HANDLES. A multi-segment session whose
engine declares no ``session_identity`` is refused at OPEN, not silently
degraded to per-segment loading: handles nobody can name are handles nobody
can invalidate, and the whole reuse is only safe because the invalidation
works. Single-segment sessions never ask -- there is nothing to drift between.

Cold-import clean (V-12): stdlib only at module scope.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

_LOG = logging.getLogger("OTR.beat_session")

__all__ = [
    "SessionError",
    "SessionIdentityUnavailable",
    "SessionIdentityDrift",
    "session_identity",
    "BeatSession",
    "SegmentSlot",
]


class SessionError(RuntimeError):
    """This beat session cannot be run as asked."""


class SessionIdentityUnavailable(SessionError):
    """A multi-segment session was asked for handles nobody can name."""


class SessionIdentityDrift(SessionError):
    """The recipe or the weights moved while the session was open."""


def session_identity(engine):
    """What this adapter's handles ARE -- engine, recipe, weights -- or None.

    Adapters declare it with an optional ``session_identity()`` method or
    attribute, duck-typed exactly like ``frame_contract``. Return anything
    comparable and stable; a tuple of strings is the shape every caller
    expects. ``None`` (or no declaration) means the adapter does not offer
    one, which single-clip engines never need to.

    DELIBERATELY NOT CACHED and deliberately NOT wrapped in a try/except. The
    entire job of this function is to notice a value that MOVED, so it is
    re-read every single time it is asked for; and an adapter whose identity
    read RAISES has told us it cannot describe its own handles, which is a
    reason to stop rather than a reason to guess.
    """
    declared = getattr(engine, "session_identity", None)
    if declared is None:
        return None
    value = declared() if callable(declared) else declared
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value)
    return (str(value),)


class BeatSession:
    """ONE engine, ONE set of handles, ONE beat -- opened and closed once.

    Use it as a context manager so the close is structural rather than
    remembered::

        with BeatSession(eng, host_caps=caps, profile=prof,
                         beat_id=bid, segment_count=len(plan.segments)) as s:
            for i, seg in enumerate(plan.segments):
                prepared = s.begin_segment(i)
                raw = eng.render_clip(request_for(seg), prepared)

    There are deliberately NO call counters on this object (cut 2026-07-26 on
    the QA panel's recommendation, and it was right): the acceptance counts
    LOADER calls on the ADAPTER, because an adapter that loads lazily inside
    ``render_clip`` shows one perfect ``prepare`` and three loads. A counter
    here would have measured the bracket, which is the thing that is obviously
    correct, instead of the loading, which is the thing that is not.
    """

    def __init__(self, engine, *, host_caps=None, profile=None, beat_id="",
                 segment_count=1, coverage_planned=None):
        self.engine = engine
        self.host_caps = dict(host_caps or {})
        self.profile = dict(profile or {})
        self.beat_id = str(beat_id or "")
        self.segment_count = max(1, int(segment_count or 1))
        #: Is this beat COVERAGE-PLANNED? Distinct from "does it have more than
        #: one segment", and adapters need the former. ``None`` falls back to
        #: the segment count, which is what every caller meant before a
        #: one-segment plan could exist. See :meth:`session_ctx`.
        self.coverage_planned = (bool(coverage_planned)
                                 if coverage_planned is not None else None)
        self.prepared = None
        self.identity = None
        self.segment_index = -1
        self._closed = False

    @property
    def is_multi_segment(self):
        return self.segment_count > 1

    def session_ctx(self):
        """The dict handed to ``prepare``.

        It was ``{}`` at every call site before this chunk. An adapter that
        wants to size a KV cache, pre-allocate, or simply log honestly needs
        to know it is about to render four clips rather than one -- and it
        must learn that BEFORE it loads, which is the only reason this is a
        prepare-time argument rather than a render-time one.

        ``multi_clip`` MEANS "coverage-planned", not "more than one segment"
        (corrected 2026-08-02). Every consumer uses it for the former:
        ``WanTi2vEngine.render_clip`` branches on it to choose ``_planned_length``
        over ``_floor_length``, and the clip-fill path it selects when False
        PING-PONG-EXTENDS a short render -- which that method's own docstring
        calls forbidden for a coverage-planned segment, since a mirrored tail
        would hand the next segment a mirrored frame to chain from.

        The two questions were the same until a ONE-segment plan carrying a tail
        trim became reachable. Such a beat is fully coverage-planned, and
        deriving the flag from ``segment_count > 1`` would have routed it to the
        mirror the operator ruled out unconditionally. So the flag is now passed
        in, and falls back to the segment count only when a caller does not say
        -- which is what every pre-existing caller meant.
        """
        planned = (self.coverage_planned if self.coverage_planned is not None
                   else self.is_multi_segment)
        return {"beat_id": self.beat_id,
                "segment_count": self.segment_count,
                "multi_clip": planned,
                "coverage_planned": planned}

    def open(self):
        """Take the handles ONCE. Raises before loading anything if a
        multi-segment beat cannot name what it is about to reuse."""
        if self.prepared is not None:
            raise SessionError(
                "beat session for %r is already open -- a second open would "
                "strand the first set of handles" % (self.beat_id or "?",))
        if self.is_multi_segment:
            # ASK BEFORE LOADING, BASELINE AFTER (2026-07-26, chunk-5 QA panel).
            # The refusal has to come before the weights land -- refusing after
            # a 10 GB load is not refusing. But the BASELINE is captured after
            # prepare returns, because the identity must describe the handles
            # that were actually loaded: an adapter that resolves "auto" to a
            # concrete UNET during load would otherwise report drift against
            # its own pre-load intention on segment 0.
            if session_identity(self.engine) is None:
                raise SessionIdentityUnavailable(
                    "engine %r would render %d segments from ONE set of "
                    "handles but declares no session_identity(), so nothing "
                    "can prove the model segment %d renders with is the one "
                    "segment 1 loaded. Declare the identity (engine, recipe, "
                    "weights) or render this beat single-clip. NO FALLBACK."
                    % (getattr(self.engine, "name", "?"), self.segment_count,
                       self.segment_count))
        self.prepared = self.engine.prepare(
            host_caps=self.host_caps, profile=self.profile,
            session_ctx=self.session_ctx())
        if self.is_multi_segment:
            try:
                self.identity = session_identity(self.engine)
            except BaseException:
                # THE BASELINE READ HAPPENS AFTER prepare(), so by now the heavy
                # handles AND the cross-process GPU lease are already held. A
                # raise here leaves through __enter__ -- and when __enter__
                # raises, Python NEVER calls __exit__, so close(), teardown()
                # and the lease release in teardown's own finally would all be
                # skipped. The lease owner is the LIVE ComfyUI process, so the
                # stale-lock reclaim (PID-liveness) cannot free it either: every
                # later heavy render then blocks its full timeout until someone
                # kills the server or deletes the lock dir by hand.
                #
                # This became reachable when session_identity() started doing
                # real file I/O (it re-stats the weights on every ask, by
                # design). Tear down what we took, THEN re-raise unchanged --
                # close() is idempotent and never raises, so it cannot mask the
                # original failure. BaseException on purpose: a KeyboardInterrupt
                # in this window strands the lease exactly the same way.
                self.close()
                raise
        return self.prepared

    def begin_segment(self, index, owner=None):
        """RE-PROVE the handles, then hand them to segment ``index``.

        The identity check is here rather than at open because drift happens
        BETWEEN segments -- an env flip while segment 1 was on the GPU. Asking
        once at the top would prove only that the beat started consistent.

        THE INDEX IS THE CALLER'S, NOT A COUNTER (2026-07-26 QA, chunk-5 lens).
        An earlier draft let ``_render_one`` derive it by incrementing, which
        silently accepted an off-by-one loop, a retried segment relabelled as
        the next one, and two shots interleaved through one session. So the
        caller states which segment it is rendering and this refuses anything
        that is not the next one, in range: a session opened for 3 segments
        serves exactly segments 0, 1, 2, in that order, once each.

        ``owner`` is the beat the caller believes it is rendering. Two beats on
        the same engine pass the engine check trivially, so without this a
        session held one loop too long would render beat B's shot from beat A's
        handles. Checked only when both sides name one -- a caller that claims
        nothing is not silently blessed, it simply has not made the claim.
        """
        index = int(index)
        if self.prepared is None:
            raise SessionError(
                "beat session for %r is not open -- begin_segment before "
                "open() would render with no handles" % (self.beat_id or "?",))
        if not 0 <= index < self.segment_count:
            raise SessionError(
                "beat session for %r was opened for %d segment(s) but was "
                "asked for segment %d. The count is what the adapter was told "
                "at prepare time, so a segment outside it renders from handles "
                "sized for a different beat."
                % (self.beat_id or "?", self.segment_count, index))
        if index != self.segment_index + 1:
            raise SessionError(
                "beat session for %r last served segment %d and was asked for "
                "%d. Segments render CONTIGUOUSLY forward, once each: a repeat "
                "is a retry this build does not do, a step backwards is a loop "
                "that lost its place, and a forward JUMP silently drops a "
                "segment the plan says the beat needs (2026-07-26 QA panel -- "
                "the earlier check allowed 0 then 2)."
                % (self.beat_id or "?", self.segment_index, index))
        if owner is not None:
            owner = str(owner)
            if not self.beat_id:
                # LATCH THE FIRST CLAIM (2026-07-26 QA panel). A session opened
                # without a beat_id -- the single-clip path does exactly that --
                # had nothing to compare against, so two beats could share one
                # session unchecked. The first caller to name a beat defines the
                # session's beat; every later claim is held to it.
                self.beat_id = owner
            elif owner != self.beat_id:
                raise SessionError(
                    "beat session belongs to beat %r but segment %d claims "
                    "beat %r. One session, one beat: the same engine serves "
                    "many beats, so the engine check alone cannot catch a "
                    "session held one loop too long."
                    % (self.beat_id, index, owner))
        if self.is_multi_segment:
            current = session_identity(self.engine)
            if current != self.identity:
                raise SessionIdentityDrift(
                    "engine %r changed identity mid-beat %r: opened as %r, is "
                    "now %r. Segment %d would render from a model the earlier "
                    "segments did not use, so the assembled beat would change "
                    "recipe partway through. NO FALLBACK -- fix the flip or "
                    "render the beat single-clip."
                    % (getattr(self.engine, "name", "?"), self.beat_id or "?",
                       self.identity, current, int(index)))
        self.segment_index = int(index)
        return self.prepared

    def close(self):
        """Release the handles ONCE. Idempotent, and never raises.

        Teardown is best-effort by long-standing contract (a raise here would
        mask the real render error on the failure path), but it is LOGGED
        rather than silently swallowed: a teardown that fails is a reclaim
        that did not happen, and the next heavy engine pays for it.
        """
        if self._closed:
            return
        self._closed = True
        prepared, self.prepared = self.prepared, None
        if prepared is None:
            return
        try:
            self.engine.teardown(prepared)
        except Exception:                # noqa: BLE001 -- teardown best-effort
            _LOG.exception(
                "[OTR.beat_session] teardown FAILED for engine %r beat %r -- "
                "the render result (or the render error) is preserved. The "
                "GPU lease itself is released in motion_common.teardown's own "
                "finally, so it is not stranded by this",
                getattr(self.engine, "name", "?"), self.beat_id or "?")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        # THE ONE OUTER FINALLY. Every segment's failure path, and the happy
        # path, leave through here exactly once.
        self.close()
        return False


class SegmentSlot(NamedTuple):
    """WHICH segment of WHICH open session a single render belongs to.

    ONE argument instead of three (2026-07-26, chunk-5 QA lens). The driver
    first grew a ``session`` plus a ``segment_index`` plus a ``session_owner``,
    which meant "a session with no index" was a state a caller could construct
    and the driver had to catch at runtime. Bundling them makes that state
    unconstructible: you cannot hand over a session without saying which
    segment it is for, so there is no error branch to forget.

    ``owner`` is the caller's INDEPENDENT claim about which beat it is on. It
    is deliberately not read off the session -- a value that agrees by
    construction proves nothing, and the point is to catch a session held one
    loop too long. Empty means the caller makes no claim.
    """

    session: "BeatSession"
    index: int
    owner: str = ""

    def begin(self):
        """Re-prove the handles and take them for this segment."""
        return self.session.begin_segment(self.index, owner=self.owner or None)
