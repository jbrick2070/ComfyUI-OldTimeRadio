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

    Raises ``KeyError`` with a helpful message listing the currently
    registered engines if the requested one isn't there.
    """
    e = (engine or "").strip().lower()
    if e not in _REGISTRY:
        registered = sorted(_REGISTRY.keys())
        raise KeyError(
            f"voice backend {engine!r} not registered "
            f"(currently registered: {registered})"
        )
    return _REGISTRY[e]


def available_engines() -> list[str]:
    """Return the sorted list of engine names registered right now."""
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


__all__ = [
    "VoiceBackend",
    "KNOWN_ENGINES",
    "register",
    "get_factory",
    "available_engines",
    "unregister",
    "_register_default_drivers",
]
