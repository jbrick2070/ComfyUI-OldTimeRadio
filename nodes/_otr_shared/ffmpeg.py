"""``ffmpeg`` -- the ONE answer to "which ffmpeg does this box run".

WHY THIS FILE EXISTS (2026-09-04, RunPod-found fixes, kibitz r1-r3). Twelve
sites in ``nodes/`` each decided this for themselves, in four different
orders. Six of them shared one defect: a caller's own signature default --
the bare string ``"ffmpeg"`` -- was treated as an operator CHOICE, so on any
box with ffmpeg on PATH it won at step one and ``OTR_FFMPEG`` was never
consulted. Two never read ``OTR_FFMPEG`` at all (the registered SignalLost
renderer and the raw-video encode sink), and the three cloud preflights
refused an install whose ffmpeg is reachable only through the variable. Every
copy was tested by stubbing ``shutil.which`` to ``None`` -- exactly the
condition under which the defect cannot show.

The bare-name rule is NOT re-implemented here. ``ffprobe.py`` already owns it
(``_explicit`` / ``_BARE_FFMPEG_NAMES``) for the probe, and one rule spelled
twice is how the last one drifted.

``tests/test_ffmpeg_single_resolution.py`` walks the AST of ``nodes/`` and
fails if any other module reads ``OTR_FFMPEG`` or asks ``which`` for the bare
name. That test, not this docstring, is what makes "one owner" true.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

try:
    from .ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable
except ImportError:  # loaded flat
    try:
        from _otr_shared.ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable  # type: ignore  # nodes/ on sys.path
    except ImportError:
        from ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable  # type: ignore  # _otr_shared/ on sys.path

#: The operator's pin. One spelling, read in one place.
FFMPEG_ENV = "OTR_FFMPEG"

#: Where a Windows box puts ffmpeg when it was installed but PATH was not
#: refreshed for the process that runs ComfyUI: winget's shim directory, and
#: the hand-unzip location the README has always suggested. Carried over
#: from the SignalLost renderer, which was the only site that knew them.
_WINDOWS_INSTALL_CANDIDATES = (
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)


def resolve_ffmpeg(preferred=None) -> Optional[str]:
    """Which ffmpeg THIS box should run, or ``None`` when it has none.

    The order, most explicit first; every step is skipped unless it resolves
    to something that exists:

    1. ``preferred`` -- a real path, or a non-default name the caller chose.
       A bare ``ffmpeg`` / ``ffmpeg.exe`` is the caller's own signature
       default and carries no information, so it is not a choice.
    2. ``$OTR_FFMPEG`` -- the operator's explicit pin.
    3. ``ffmpeg`` on ``PATH``.
    4. the well-known Windows install locations above.

    NEVER RAISES. "This box has no ffmpeg" is a fact, and each caller has
    already decided what that fact costs it -- an empty string, its own
    literal, or a named refusal.
    """
    chosen = _usable(_explicit(preferred, _BARE_FFMPEG_NAMES))
    if chosen:
        return chosen
    chosen = _usable(os.path.expanduser(os.environ.get(FFMPEG_ENV) or ""))
    if chosen:
        return chosen
    chosen = _usable("ffmpeg")  # PATH, through the one function that reads it
    if chosen:
        return chosen
    for raw in _WINDOWS_INSTALL_CANDIDATES:
        candidate = os.path.expandvars(raw)
        if os.path.isfile(candidate):
            return candidate
    return None
