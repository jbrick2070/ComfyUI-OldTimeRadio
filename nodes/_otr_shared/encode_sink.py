"""The pack's one nvenc decision: ``has_nvenc``.

This module is intentionally small and torch-free. ``scope_draw._has_nvenc`` and
``video_engine`` delegate here; nothing else decides whether h264_nvenc can
actually encode.
"""
from __future__ import annotations

import os
import threading

from .ffmpeg import resolve_ffmpeg

try:
    from . import proc as otr_proc
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import proc as otr_proc  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import proc as otr_proc  # type: ignore  # _otr_shared/ on sys.path


#: Cached for the process, PER BINARY. The probe costs about a second and the
#: answer cannot change while we run -- but it is an answer about ONE build,
#: so the key is the normalized binary path (kibitz r2, 2026-09-04): a caller
#: naming a different ffmpeg gets its own probe, not the first caller's.
_NVENC_PROBE: dict = {}
#: The probe takes up to twenty seconds; two first callers racing on the same
#: binary must run it once, not twice.
_NVENC_PROBE_LOCK = threading.Lock()


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
        cached = _NVENC_PROBE.get(key)
        if cached is not None:
            return cached
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
            verdict = out.returncode == 0
        except Exception:  # noqa: BLE001 -- a probe must never be fatal.
            verdict = False
        _NVENC_PROBE[key] = verdict
        return verdict
