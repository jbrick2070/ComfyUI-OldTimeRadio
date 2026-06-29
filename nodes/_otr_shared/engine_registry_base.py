"""Reusable, dependency-free engine-registry pattern (A-Seam AS-4).

The SHIPPED audio registry (``nodes/_otr_audio_engines/registry.py``) proved a
fail-closed, model-agnostic "pluggable adapter" pattern: adapters self-register,
nodes build their dropdown from ``engines_for_role(role)``, and ``assert_usable``
either returns the validated engine name or raises :class:`EngineUnusable` with
a classified reason -- it NEVER silently swaps one engine for another.

AS-4 anchors the video + image platforms on the SHAPE of that proven pattern
WITHOUT importing the audio module (invariant V-12: video cold-import must not
pull the audio package, which hard-imports ``torch`` at module scope). This
module re-expresses the pattern as dependency-free, reusable primitives:

* :class:`EngineCore` -- a ``Protocol`` capturing the registry-facing contract
  shared by every namespace (audio/video/image). It mirrors the member names +
  return annotations of the shipped ``AudioEngine(Protocol)`` so
  ``test_protocol_parity`` can prove the video (and later image) protocols are
  structural supersets of the audio core -- the single source of "same shape".
* :class:`EngineUsabilityReason` -- the six fail-closed reason codes (identical
  taxonomy to the audio registry, single-sourced here for video + image).
* :class:`EngineUnusable` -- the classified, never-crash/never-silent-swap error.
* :class:`EngineRegistry` -- one reusable registry instance per namespace. The
  video registry owns one; the image registry (C1) will own another. Audio keeps
  its own frozen copy and is untouched.

Import-time is side-effect-free: only ``enum`` and ``typing`` -- no IO, no
network, no CUDA, no model framework. Importing this module pulls in nothing
heavy (V-12).
"""
from __future__ import annotations

import enum
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class EngineCore(Protocol):
    """The registry-facing contract shared by every engine namespace.

    These are exactly the members the registry mechanics (``register`` /
    ``engines_for_role`` / ``assert_usable``) read. They mirror the SHIPPED
    ``AudioEngine(Protocol)`` member set + annotations so the audio core is a
    structural subset of the video / image protocols (proven by
    ``test_protocol_parity``). Adapters duck-type against this -- inheritance is
    never required (the shipped audio adapters do not inherit either).

    ``roles`` lists the roles the engine can serve; ``default_roles`` lists the
    roles where it is the in-stack default (sorts first in the dropdown).
    ``requires_flag`` is VESTIGIAL (always ``None`` on registered engines, C2-C6
    -- the registry IS the menu, no flag gate); the field is retained only so the
    audio ``AudioEngine(Protocol)`` parity holds. ``load`` / ``unload`` bracket
    model residency and must be cheap when already in the desired state (residency
    belongs there, never in ``__init__`` -- importing a package must never pull
    weights).
    """

    name: str
    roles: tuple
    default_roles: tuple
    commercial_clean: bool
    requires_flag: Optional[str]

    def load(self) -> None: ...
    def unload(self) -> None: ...


class EngineUsabilityReason(str, enum.Enum):
    """The reasons an engine may be refused for a role (fail-closed).

    Registry-level checks (no IO) raise ``MALFORMED_CONFIG`` and
    ``INCOMPATIBLE_PROFILE``. ``GATED_BY_FLAG`` is now DEAD -- the flag gate was
    removed (C2-C6, "registry IS the menu"); the member is retained only so the
    enum stays identical to the frozen audio taxonomy (protocol-parity). The
    disk/token/commercial reasons (``MISSING_MODEL``, ``MISSING_HF_TOKEN``,
    ``NONCOMMERCIAL_BLOCKED``) require IO and are raised downstream by the profile
    resolver + release gate, which reuse this same enum + :class:`EngineUnusable`
    so the taxonomy is single-sourced across audio, video and image.
    """

    GATED_BY_FLAG = "gated_by_flag"   # DEAD: no engine gates on a flag (C2-C6); kept for parity
    MISSING_MODEL = "missing_model"
    MISSING_HF_TOKEN = "missing_hf_token"
    INCOMPATIBLE_PROFILE = "incompatible_profile"
    NONCOMMERCIAL_BLOCKED = "noncommercial_blocked"
    MALFORMED_CONFIG = "malformed_config"


class EngineUnusable(RuntimeError):
    """Raised when an engine cannot serve a role; FAIL CLOSED.

    Carries the classified ``reason`` (an :class:`EngineUsabilityReason`) plus
    the engine + role so a node can surface a clear, named queue-time error
    instead of crashing or silently substituting another engine. ``kind``
    namespaces the message ("video" / "image" / "audio") so the same error type
    is reusable across registries while still reading naturally.
    """

    def __init__(
        self,
        engine: str,
        role: str,
        reason,
        detail: str = "",
        *,
        kind: str = "engine",
    ):
        self.engine = engine
        self.role = role
        self.reason = EngineUsabilityReason(reason)
        self.detail = detail
        self.kind = kind
        msg = (
            f"{kind} engine '{engine}' is not usable for role '{role}': "
            f"{self.reason.value}"
        )
        if detail:
            msg += f" -- {detail}"
        super().__init__(msg)


class EngineRegistry:
    """A fail-closed, model-agnostic engine registry (one per namespace).

    Mirrors the shipped audio registry's ``register`` / ``get_engine`` /
    ``engines_for_role`` / ``default_engine_for_role`` / ``assert_usable``
    semantics EXACTLY -- the only delta is ``kind`` (for messages) and that each
    namespace owns its own instance + dict (no cross-pollution). Dependency-free:
    instantiating + populating it pulls in no heavy lib (V-12).

    Usability is fail-closed: ``assert_usable`` returns the validated name or
    raises :class:`EngineUnusable`; it NEVER silently resolves to another engine.
    A reduced-capability adapter (e.g. an image engine with ``canonicalize=None``)
    registers fine -- the registry only reads the :class:`EngineCore` members and
    never calls render-lifecycle methods, so optional methods may be absent/None.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._registry: dict = {}

    def register(self, adapter):
        """Class decorator / function: instantiate + record an adapter.

        ``adapter`` may be a class (instantiated with no args -- ``__init__``
        must be cheap, no model load) or an already-built instance. Returns the
        original ``adapter`` so it works as a ``@register`` class decorator.
        """
        inst = adapter() if isinstance(adapter, type) else adapter
        self._registry[inst.name] = inst
        return adapter

    def get_engine(self, name: str):
        if name not in self._registry:
            raise KeyError(f"{self.kind} engine '{name}' is not registered")
        return self._registry[name]

    def is_registered(self, name: str) -> bool:
        return name in self._registry

    def all_engine_names(self) -> list:
        """Every registered engine name, sorted (the STATIC dropdown source).

        The director COMBO is built from the full registry (V-6: the enum stays
        the complete static list); role compatibility is filtered at
        execute/validate time, never by mutating the COMBO.
        """
        return sorted(self._registry)

    def engines_for_role(self, role: str) -> list:
        """Names of engines serving ``role``, default engine(s) first.

        The default-for-role engine sorts first so a combo built from this list
        defaults to the in-stack choice.
        """
        names = [
            n for n, e in self._registry.items()
            if role in getattr(e, "roles", ())
        ]
        names.sort(
            key=lambda n: (
                role not in getattr(self._registry[n], "default_roles", ()),
                n,
            )
        )
        return names

    def default_engine_for_role(self, role: str) -> Optional[str]:
        for n, e in self._registry.items():
            if role in getattr(e, "default_roles", ()):
                return n
        return None

    def assert_usable(self, name: str, role: str) -> str:
        """Validate ``name`` may actually run for ``role``; FAIL CLOSED.

        Returns the validated engine name on success; raises
        :class:`EngineUnusable` with a classified reason otherwise -- never
        silently resolves to another engine:

        * ``MALFORMED_CONFIG`` -- no engine named ``name`` is registered.
        * ``INCOMPATIBLE_PROFILE`` -- the engine does not list ``role``.

        There is NO ``GATED_BY_FLAG`` case (C2-C6 -- the registry IS the menu): a
        registered, role-compatible engine is always usable. Disk/token/commercial
        checks require IO and are enforced downstream (this method does no IO).
        """
        if not self.is_registered(name):
            raise EngineUnusable(
                name, role, EngineUsabilityReason.MALFORMED_CONFIG,
                f"no {self.kind} engine named '{name}' is registered",
                kind=self.kind,
            )
        eng = self._registry[name]
        if role not in getattr(eng, "roles", ()):
            raise EngineUnusable(
                name, role, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                f"engine '{name}' serves roles "
                f"{tuple(getattr(eng, 'roles', ()))}, not '{role}'",
                kind=self.kind,
            )
        # Registered + role-compatible == usable. The registry IS the menu: a
        # registered engine is selectable and renders (it may still hard-fail
        # LOUD downstream); there is NO flag gate -- validation is the operator's
        # manual process, never a code gate. Disk/token/commercial checks require
        # IO and are enforced downstream (this method does no IO).
        return name
