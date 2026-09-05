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

import logging
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

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import env as otr_env  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import env as otr_env  # type: ignore  # _otr_shared/ on sys.path

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


_log = logging.getLogger("OTR")


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
    # strip THEN expand: a pin typed with leading whitespace keeps its tilde
    # otherwise ("  ~/bin/ffmpeg" never expanded -- agy, manual r4).
    chosen = _usable(os.path.expanduser((otr_env.get(FFMPEG_ENV) or "").strip()))
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


#: Nodes that have already said their widget value is ignored. One line per
#: node per process -- an operator who typed a path learns it is dead, and a
#: soak does not print it on every beat.
_WIDGET_IGNORED_WARNED = set()


def widget_ffmpeg_is_ignored(value, node):
    """The ffmpeg preference a NODE may express: none, ever. Returns ``""``.

    A ComfyUI widget value arrives in the body of an unauthenticated
    ``/prompt`` request, and is whatever a downloaded workflow JSON says. It is
    UNTRUSTED INPUT, not operator intent, so it must not name the binary this
    pack spawns: honouring it let a workflow point argv[0] at any file on disk
    named ffmpeg, ahead of the operator's own ``OTR_FFMPEG`` pin, and the
    ffprobe sibling rule turned one such value into a SECOND attacker binary.

    ``OTR_FFMPEG`` remains the way to pin a build, and a workflow cannot set an
    environment variable -- which is exactly why the pin is the trustworthy
    channel and the widget is not.

    NOT A BEHAVIOUR CHANGE FOR ANY SHIPPED GRAPH, measured 2026-09-04: all 465
    ffmpeg widget values across all 101 workflow JSONs are the bare literal
    ``"ffmpeg"``, which :func:`_explicit` already treats as "no preference".
    The widget stays in ``INPUT_TYPES`` and in every execute signature, so
    ``widgets_values``, the ``inputs`` descriptors and every link ``dst_slot``
    are untouched -- removing it is a separate, scheduled migration.
    """
    try:
        expressed_a_choice = _explicit(value, _BARE_FFMPEG_NAMES) is not None
    except Exception:  # noqa: BLE001 -- a junk widget value is still ignored
        expressed_a_choice = bool(value)
    if expressed_a_choice and node not in _WIDGET_IGNORED_WARNED:
        _WIDGET_IGNORED_WARNED.add(node)
        _log.warning(
            "[%s] the 'ffmpeg' widget is ignored (%r): a workflow value cannot "
            "name the binary this pack runs. Set the OTR_FFMPEG environment "
            "variable to pin a build.", node, str(value)[:120])
    return ""
