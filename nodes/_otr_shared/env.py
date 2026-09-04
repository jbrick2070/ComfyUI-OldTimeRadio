"""ONE owner answers what an environment knob says: this module.

The registry scan of alpha.17 carried 103 `python_environment_manipulation`
findings -- one per FILE that touches ``os.environ`` -- across 103 files, and
the human reviewing that report cannot tell a render path from a test seam in
a hundred and three lines. More importantly for the pack itself: a knob read
in ninety places is a decision made in ninety places. The ffmpeg precedent
(``_otr_shared/ffmpeg.py``, 2026-09-04) found nine resolvers where the pack
believed it had one, and two of them ignored the operator's pin.

So this module is the only place under ``nodes/`` that spells ``os.environ``.
Everything else asks it.

WHAT THIS IS NOT. It is a SPELLING, not a schema. There is no knob catalog,
no typed getter, no default table, no credential name inside it. A caller's
default, its cast and its precedence stay exactly where they were:
``os.environ.get("OTR_X", "0") == "1"`` becomes
``otr_env.get("OTR_X", "0") == "1"`` and nothing else moves. That restraint
is the whole reason the migration is safe to do in one pass across a hundred
files: no site's meaning changes, so no box's numbers change.

READS ARE LIVE, ALWAYS. Every call reads ``os.environ`` at call time and
caches nothing. ``tests/conftest.py`` pops names at import, hundreds of tests
``monkeypatch.setenv``, and both launchers pin at boot; one cached read would
break all three silently.

Import it under the alias ``otr_env`` -- ``env`` is a parameter name in
``_otr_shared/route_freeze.py`` and ``_otr_video_engines/motion_common.py``,
and a shadowed module raises ``UnboundLocalError`` at the first migrated
site:

    try:
        from . import env as otr_env          # inside _otr_shared
    except ImportError:                       # pragma: no cover -- flat load
        import env as otr_env                 # type: ignore

Stdlib only, and it imports nothing from the pack, so it is a leaf: any
module may import it at any point in boot, including the package
``__init__`` before its first write. ``prestartup_script.py`` is the one
deliberate exception -- it runs with no package context, before the pack
exists -- and keeps its own inline writes.
"""
from __future__ import annotations

import os
from typing import Any, Optional

__all__ = ["get", "pin", "setdefault", "unpin", "snapshot"]


def get(name: str, default: Any = None) -> Any:
    """``os.environ.get(name, default)``, read live, returned unchanged.

    The caller keeps its own cast and its own fallback. This function has no
    opinion about either -- that is what makes the migration spelling-only.

    ``default`` is deliberately ``Any``, not ``str | None``: a live call site
    passes an INT (``int(os.environ.get("OTR_RADIO_BOOKEND_SEED", 4242))`` in
    ``otr_image_gen_dispatcher.py``), and a narrower annotation here would
    describe a contract this function does not enforce and the pack does not
    keep. Typing the knobs is the follow-on arc's job, with its own tests.
    """
    return os.environ.get(name, default)


def pin(name: str, value: str) -> None:
    """Set a knob for this process (``os.environ[name] = value``).

    A non-string value is a caller bug, and ``None`` is the most likely one:
    a site meaning "unset this" must say :func:`unpin`, because a silent
    unpin here would be indistinguishable from a pin that did nothing.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be pinned to a str, got {type(value).__name__}; "
            "use unpin() to remove a knob")
    os.environ[name] = value


def setdefault(name: str, value: str) -> str:
    """``os.environ.setdefault`` -- pin only if the operator has not."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must default to a str, got {type(value).__name__}")
    return os.environ.setdefault(name, value)


def unpin(name: str) -> Optional[str]:
    """Remove a knob, returning what it held. Never raises for an unset name.

    ``os.environ.pop(name)`` without a default raises ``KeyError``, and every
    caller in the pack already passes ``None`` (the style-grammar restore in
    ``OTR_LedgerScriptWriter`` is the live example), so the safe form is the
    contract rather than an option.
    """
    return os.environ.pop(name, None)


def snapshot() -> dict:
    """A plain ``dict`` COPY of the whole environment.

    For the two sites that hand the environment to something else as a
    mapping -- the Blender spawn's sanitizer and the routing freeze -- rather
    than reading one knob. A copy, never the live ``os.environ`` object, so a
    consumer that mutates its argument cannot reach this process.
    """
    return dict(os.environ)
