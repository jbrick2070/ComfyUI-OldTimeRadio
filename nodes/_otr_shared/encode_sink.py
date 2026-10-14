"""The pack's one nvenc decision: ``has_nvenc``.

This module is intentionally small and torch-free. ``scope_draw._has_nvenc`` and
``video_engine`` delegate here; nothing else decides whether h264_nvenc can
actually encode.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

from .ffmpeg import resolve_ffmpeg

try:
    from . import proc as otr_proc
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import proc as otr_proc  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import proc as otr_proc  # type: ignore  # _otr_shared/ on sys.path

log = logging.getLogger("OTR")


#: Cached for the process, PER BINARY: it is an answer about ONE build, so the
#: key is the normalized binary path (kibitz r2, 2026-09-04) and a caller
#: naming a different ffmpeg gets its own probe, not the first caller's.
#: A FINAL answer is a bool and never changes. An INCONCLUSIVE probe leaves a
#: :class:`_Pending` here instead (see ``_classify``), so this one dict is the
#: whole state and resetting it resets everything.
_NVENC_PROBE: dict = {}
#: Guards ``_NVENC_PROBE`` and ``_NVENC_KEY_LOCKS`` only, never a probe.
_NVENC_PROBE_LOCK = threading.Lock()
#: The probe takes up to twenty seconds. It runs under a PER-BINARY lock, so
#: two first callers racing on one binary run it once, and a probe of one
#: build never stalls a cached answer for another.
_NVENC_KEY_LOCKS: dict = {}

#: An inconclusive probe is retried no sooner than this after the last one,
#: doubling each time up to the ceiling.
_NVENC_RETRY_FIRST_S = 120.0
_NVENC_RETRY_MAX_S = 1800.0
#: A TIMEOUT costs the full twenty seconds, so after this many the binary is
#: FINAL False. A fast refusal costs well under a second and is never capped:
#: on an 8 GB card, VRAM can stay full of model weights for a whole video phase.
_NVENC_MAX_TIMEOUTS = 3

#: Session-open refusals that depend on the MOMENT, not the machine, matched on
#: the OpenEncodeSessionEx line only so an unrelated allocation failure
#: elsewhere in stderr is not mistaken for one:
#:   * "incompatible client key (21)" -- the GeForce CONCURRENT-SESSION CAP on
#:     current drivers. MEASURED on MRKT (RTX 4060 Laptop, driver 617.42,
#:     ffmpeg 9.0, 2026-10-14): twelve
#:     live h264_nvenc encodes ran, and the thirteenth session was refused with
#:     exactly this. NVIDIA's header calls it a licensing refusal, and the r1
#:     review dropped it on that reading; the hardware says otherwise.
#:   * "out of memory (10)" -- the same cap on older drivers, or VRAM truly
#:     exhausted. (A probe with 48 MB of VRAM free still ENCODED on MRKT, so
#:     model weights alone do not trigger it on that Windows driver.)
#:   * "encoder busy" -- NV_ENC_ERR_ENCODER_BUSY.
#: If one of these is permanent on some box, each retry is a sub-second refusal
#: on a doubling cooldown -- noise, not a slowdown.
_TRANSIENT_REFUSAL = re.compile(
    r"OpenEncodeSessionEx failed: "
    r"(incompatible client key|out of memory|encoder busy)", re.IGNORECASE)

_TIMEOUT = "timeout"
_REFUSED = "refused"


class _Pending:
    """An inconclusive binary: CPU for now, re-probe after ``not_before``."""

    __slots__ = ("attempts", "timeouts", "not_before")

    def __init__(self, attempts: int, timeouts: int, not_before: float):
        self.attempts = attempts
        self.timeouts = timeouts
        self.not_before = not_before


def _now() -> float:
    """Monotonic seconds; a seam so tests can drive the cooldown."""
    return time.monotonic()


def has_nvenc(ffmpeg: str) -> bool:
    """Can h264_nvenc actually ENCODE here -- not merely exist in the build.

    THE OBVIOUS TEST IS WRONG AND COST A WHOLE EPISODE (2026-08-30). This used
    to answer with ``"h264_nvenc" in (ffmpeg -codecs)``, which reports whether
    ffmpeg was COMPILED with nvenc. On any container that ships a full ffmpeg
    without the NVIDIA encode library -- the normal case on rented GPUs --
    that is true while encoding is impossible:

        [h264_nvenc] Cannot load libnvidia-encode.so.1
        [vost#0:0/h264_nvenc] Error while opening encoder
        Conversion failed!

    Callers stream raw frames into ffmpeg's stdin, so a dead encoder surfaces
    as BrokenPipeError or a RuntimeError on the first write -- eighteen minutes
    into a render on the leg that found this, long after the script and audio
    were finished, naming nothing useful.

    **This is the SINGLE SOURCE OF TRUTH for nvenc, and every other site
    delegates to it.** There were two identical string tests, here and in
    ``video_engine``; fixing one left the other to fail the same render at a
    later node, which is exactly what happened. The node now delegates here.

    THAT CLAIM WAS WRONG FOR FOUR DAYS, AND IT READ AS COVERAGE (2026-09-03).
    This docstring said "the ONLY nvenc decision in the pack" while a THIRD
    string test still lived in ``scope_draw._has_nvenc`` -- and the four viz_*
    engines encode through that module, so they never reached this probe. A
    rented 4090 that lists h264_nvenc and cannot open a session found it.
    ``scope_draw`` now delegates here too, and
    ``tests/test_nvenc_single_decision.py`` fails if a fourth copy appears.

    Probes at 256x256 deliberately: NVENC rejects tiny frames outright with
    "Frame Dimension less than the minimum supported value", so a 64x64 probe
    reports a HEALTHY card as unavailable and silently drops it to CPU.

    A PROBE THAT COULD NOT DECIDE IS NOT A "NO" (VF-01, 2026-10-14). Every
    failure used to be cached as False for the life of the server, so one
    probe that timed out, or that found VRAM full of model weights on an 8 GB
    card, kept every later encode on CPU until a restart. Now only a FINAL
    answer is cached: success, or a refusal that waiting cannot change (no
    encode library, no capable device, an ffmpeg without nvenc, a missing
    binary). A box without NVENC is still probed exactly once. A timeout or a
    moment-dependent session refusal answers False for THIS call -- the
    caller encodes on libx264 as always -- and is re-probed after a cooldown;
    see ``_classify``. NVENC is an accelerator here, never a requirement.
    """
    if not ffmpeg:
        return False
    # A BARE name would key the cache on <cwd>/ffmpeg and probe whatever
    # PATH says; resolve it through the owner first (the pin, then PATH). An
    # explicit path is the caller's choice and is probed as given.
    ffmpeg = str(ffmpeg)
    if not os.path.dirname(ffmpeg):
        ffmpeg = resolve_ffmpeg(ffmpeg)
        if not ffmpeg:
            # Nothing to probe and nothing to remember: a name that does
            # not resolve must not key the cache on <cwd>/name (cursor r4).
            return False
    key = os.path.normcase(os.path.abspath(ffmpeg))
    with _NVENC_PROBE_LOCK:
        settled = _settled(key)
        if settled is not None:
            return settled
        key_lock = _NVENC_KEY_LOCKS.setdefault(key, threading.Lock())
    with key_lock:
        # A racing caller may have probed while this one waited for the lock.
        with _NVENC_PROBE_LOCK:
            settled = _settled(key)
            if settled is not None:
                return settled
            pending = _NVENC_PROBE.get(key)
        outcome = _probe(ffmpeg)
        with _NVENC_PROBE_LOCK:
            return _record(key, ffmpeg, pending, outcome)


def _settled(key: str):
    """The cached answer for ``key`` without probing, or None to probe.

    Caller holds ``_NVENC_PROBE_LOCK``. A pending binary still inside its
    cooldown answers False -- CPU for this call, no spawn."""
    entry = _NVENC_PROBE.get(key)
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, _Pending) and _now() < entry.not_before:
        return False
    return None


def _probe(ffmpeg: str):
    """One real encode. True / False are FINAL; ``_TIMEOUT`` / ``_REFUSED``
    are inconclusive. Never raises."""
    try:
        out = otr_proc.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-frames:v", "1",
             "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except otr_proc.TimeoutExpired:
        return _TIMEOUT
    except Exception:  # noqa: BLE001 -- a probe must never be fatal.
        return False
    if out.returncode == 0:
        return True
    return _classify(str(getattr(out, "stderr", "") or ""))


def _classify(stderr: str):
    """A nonzero probe: ``_REFUSED`` when the session refusal can clear by
    waiting, else FINAL False."""
    return _REFUSED if _TRANSIENT_REFUSAL.search(stderr) else False


def _record(key: str, ffmpeg: str, pending, outcome) -> bool:
    """Store a probe's outcome; return this call's answer. Caller holds
    ``_NVENC_PROBE_LOCK``."""
    if isinstance(outcome, bool):
        _NVENC_PROBE[key] = outcome
        if isinstance(pending, _Pending):
            log.info("[OTR nvenc] %s: h264_nvenc %s after %d inconclusive "
                     "probe(s)", ffmpeg,
                     "available" if outcome else "refused for good",
                     pending.attempts)
        return outcome
    attempts = (pending.attempts if isinstance(pending, _Pending) else 0) + 1
    timeouts = (pending.timeouts if isinstance(pending, _Pending) else 0)
    timeouts += 1 if outcome == _TIMEOUT else 0
    reason = ("timed out" if outcome == _TIMEOUT
              else "session refused for now (VRAM, session cap or busy)")
    if timeouts >= _NVENC_MAX_TIMEOUTS:
        _NVENC_PROBE[key] = False
        log.warning("[OTR nvenc] %s: probe %s; %d timeouts, so CPU libx264 for "
                    "the rest of this process", ffmpeg, reason, timeouts)
        return False
    # The exponent is capped: refusals are uncapped, and 2 ** 1024 overflows a
    # float -- a probe must never be the thing that fails a workflow.
    wait = min(_NVENC_RETRY_MAX_S,
               _NVENC_RETRY_FIRST_S * 2 ** min(attempts - 1, 16))
    _NVENC_PROBE[key] = _Pending(attempts, timeouts, _now() + wait)
    log.warning("[OTR nvenc] %s: probe %s; CPU libx264 for now, re-probing in "
                "%d s (inconclusive probe %d)", ffmpeg, reason, int(wait),
                attempts)
    return False
