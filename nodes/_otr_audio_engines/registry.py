"""Pluggable audio-engine registry -- the audio superstructure.

Each audio *role* -- character voice, announcer voice, music, sfx -- picks its
engine from this shared registry instead of being hardcoded to one model.
Adapters self-register on import; nodes build their ComfyUI engine dropdown
from ``engines_for_role(role)``. The engine that is the byte-identical default
for a role sorts first, so the default workflow keeps choosing it and stays
byte-identical.

Adding a new model later is a one-file adapter plus one import line -- no node
surgery. This mirrors the writer's multi-slot LLM design: independent slots,
one shared extensible pool.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AudioEngine(Protocol):
    """Contract every audio engine adapter implements.

    ``roles`` lists the roles the engine can serve (e.g. ``("char_voice",)``
    or ``("char_voice", "announcer_voice")``). ``default_roles`` lists the
    roles where this engine is the in-stack byte-identical default. Voice
    engines implement ``generate_voice``; music/sfx engines implement
    ``generate_clip``. ``load`` / ``unload`` bracket model residency and must
    be cheap to call when already in the desired state.
    """

    name: str
    roles: tuple
    default_roles: tuple
    commercial_clean: bool
    requires_flag: Optional[str]

    def load(self) -> None: ...
    def unload(self) -> None: ...


_REGISTRY: dict = {}


def register(adapter):
    """Class decorator / function that instantiates and records an adapter.

    Adapter ``__init__`` must be cheap (no model load) -- residency happens in
    ``load()`` so that merely importing the package never pulls heavy weights.
    """
    inst = adapter() if isinstance(adapter, type) else adapter
    _REGISTRY[inst.name] = inst
    return adapter


def get_engine(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"audio engine '{name}' is not registered")
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def engines_for_role(role: str) -> list:
    """Names of engines that serve ``role``, default engine(s) first.

    The default-for-role engine sorts first so a ComfyUI combo built from this
    list defaults to the byte-identical choice.
    """
    names = [n for n, e in _REGISTRY.items() if role in getattr(e, "roles", ())]
    names.sort(
        key=lambda n: (role not in getattr(_REGISTRY[n], "default_roles", ()), n)
    )
    return names


def default_engine_for_role(role: str) -> Optional[str]:
    """Return the name of the byte-identical default engine for ``role``."""
    for n, e in _REGISTRY.items():
        if role in getattr(e, "default_roles", ()):
            return n
    return None


def assert_usable(name: str, role: str) -> str:
    """Resolve the engine that may actually run for ``role``.

    A default-for-role engine always runs. A non-default engine runs only when
    its ``requires_flag`` env var is set to ``"1"``; otherwise this returns the
    role's default engine, so an un-flagged opt-in lane never changes the
    rendered audio (the safety property behind the whole v2 lane).
    """
    eng = get_engine(name)
    if role in getattr(eng, "default_roles", ()):
        return name
    flag = getattr(eng, "requires_flag", None)
    if flag and os.getenv(flag, "0") != "1":
        resolved = default_engine_for_role(role)
        return resolved if resolved is not None else name
    return name
