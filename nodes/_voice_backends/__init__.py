"""nodes/_voice_backends/__init__.py

Driver registry for voice-render engines.

Phase 0+ Voice Backend Abstraction §2 -- one driver per TTS engine
implementing a small interface (``VoiceBackend``). The registry maps
engine name (``bark``, ``kokoro``, ``cosyvoice``, ...) to a lazy
factory that returns a backend instance on demand.

Lazy registration matters: each driver pulls its own model dependency
(Bark imports the Bark python package, Kokoro imports the Kokoro
runtime, etc.). Importing this package must NOT trigger any of those
imports -- only the requested engine's module is loaded.

Public surface
--------------
    VoiceBackend            -- abstract protocol (see _protocol.py)
    register(engine, factory) -- add an engine to the registry
    get_factory(engine)     -- look up a registered factory
    available_engines()     -- list of registered engine names
    KNOWN_ENGINES           -- mirrors _otr_voice_resolver.KNOWN_ENGINES
                                so callers don't need both imports

The bundled drivers (bark, kokoro) self-register via lazy
``_register_default_drivers()`` so a user calling
``available_engines()`` after a fresh import sees the stock list.
"""
from __future__ import annotations

from typing import Callable

from nodes._voice_backends._protocol import VoiceBackend


# Mirror the resolver's known-engine set so callers using just this
# package don't have to also import nodes._otr_voice_resolver.
KNOWN_ENGINES: set[str] = {"bark", "kokoro", "cosyvoice", "xtts", "piper"}


_REGISTRY: dict[str, Callable[[], VoiceBackend]] = {}
_DEFAULTS_REGISTERED = False


def register(engine: str, factory: Callable[[], VoiceBackend]) -> None:
    """Register a backend factory under ``engine``.

    The factory must be callable with NO arguments and return a fresh
    ``VoiceBackend``-conforming instance. The factory is invoked lazily
    by ``get_factory(engine)()`` -- never at registration time -- so a
    driver module that pulls heavy ML weights does so only when the
    backend is actually needed.
    """
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine must be a non-empty string")
    _REGISTRY[engine.strip().lower()] = factory


def get_factory(engine: str) -> Callable[[], VoiceBackend]:
    """Return the registered factory for ``engine``.

    Auto-fires the bundled-driver self-registration on the FIRST call
    so callers using just `from nodes._voice_backends import
    get_factory` (without separately importing each driver module)
    still see the stock bark/kokoro registrations. Round-robin
    Element 4 catch (2026-05-08).

    Raises ``KeyError`` with a helpful message listing the currently
    registered engines if the requested one still isn't there after
    auto-init.
    """
    _ensure_defaults_registered()
    e = (engine or "").strip().lower()
    if e not in _REGISTRY:
        registered = sorted(_REGISTRY.keys())
        raise KeyError(
            f"voice backend {engine!r} not registered "
            f"(currently registered: {registered})"
        )
    return _REGISTRY[e]


def available_engines() -> list[str]:
    """Return the sorted list of engine names registered right now.

    Like ``get_factory``, fires the lazy default-driver registration
    so a caller using only the package surface sees the stock list.
    """
    _ensure_defaults_registered()
    return sorted(_REGISTRY.keys())


def unregister(engine: str) -> None:
    """Remove a previously-registered engine. No-op if not present.

    Helpful in tests when you want to reset registry state between
    cases without cross-test pollution.
    """
    _REGISTRY.pop((engine or "").strip().lower(), None)


def _register_default_drivers() -> None:
    """Lazy self-registration for the bundled drivers.

    Imports happen here (not at package import time) so a caller that
    only wants ``available_engines()`` to confirm the registry shape
    doesn't pay the cost of importing the heavy backend modules.
    """
    # We import the modules; each calls ``register(...)`` at module
    # scope on first import.
    from nodes._voice_backends import bark as _bark  # noqa: F401
    from nodes._voice_backends import kokoro as _kokoro  # noqa: F401


def _ensure_defaults_registered() -> None:
    """Fire ``_register_default_drivers()`` exactly once per process.

    Used by ``get_factory`` and ``available_engines`` to guarantee
    that a caller using just the package surface (without separately
    importing each driver module) still sees the stock bark/kokoro
    registrations. Idempotent: runs the registration on first call
    and returns immediately on subsequent calls.

    Safe even if a future driver fails to import: the flag is set
    BEFORE the imports run, so a failed driver doesn't trap us in an
    infinite re-register loop. Caller is responsible for noticing
    the missing engine via the empty registry list.
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True
    try:
        _register_default_drivers()
    except Exception:  # noqa: BLE001
        # Surface the underlying ImportError to the caller via the
        # empty-registry KeyError; do NOT crash here, because some
        # callers want the registry shape even when a driver's
        # heavyweight deps aren't installed yet.
        pass


__all__ = [
    "VoiceBackend",
    "KNOWN_ENGINES",
    "register",
    "get_factory",
    "available_engines",
    "unregister",
    "_register_default_drivers",
    "_ensure_defaults_registered",
]
