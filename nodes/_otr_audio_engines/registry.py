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

Usability is **fail-closed** (plan C-6, "names the missing piece -- never
crash, never silent swap"). ``assert_usable`` validates that the requested
engine can actually run for the role and either returns the validated engine
name or raises :class:`EngineUnusable` carrying one of the six
:class:`EngineUsabilityReason` codes. It NEVER silently swaps an opt-in engine
for the role default -- the byte-identical safety property is provided by the
shipped workflow defaulting its engine widget to the legacy engine until
promotion, not by a hidden substitution at dispatch time.
"""
from __future__ import annotations

import enum
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


# ---------------------------------------------------------------------------
# Fail-closed usability taxonomy (plan piece 3 + dispatch D)
# ---------------------------------------------------------------------------


class EngineUsabilityReason(str, enum.Enum):
    """The six reasons an engine may be refused for a role (fail-closed).

    ``assert_usable`` (registry level, no IO) raises ``GATED_BY_FLAG``,
    ``MALFORMED_CONFIG`` and ``INCOMPATIBLE_PROFILE``. The disk/token/commercial
    reasons (``MISSING_MODEL``, ``MISSING_HF_TOKEN``, ``NONCOMMERCIAL_BLOCKED``)
    are raised by the profile resolver and the release gate, which reuse this
    same enum + :class:`EngineUnusable` so the taxonomy is single-sourced.
    """

    GATED_BY_FLAG = "gated_by_flag"
    MISSING_MODEL = "missing_model"
    MISSING_HF_TOKEN = "missing_hf_token"
    INCOMPATIBLE_PROFILE = "incompatible_profile"
    NONCOMMERCIAL_BLOCKED = "noncommercial_blocked"
    MALFORMED_CONFIG = "malformed_config"


class EngineUnusable(RuntimeError):
    """Raised when an engine cannot serve a role. Carries the classified
    ``reason`` (an :class:`EngineUsabilityReason`) plus the engine + role so a
    node can surface a clear, named queue-time error (C-7) instead of crashing
    or silently substituting another engine.
    """

    def __init__(self, engine: str, role: str, reason, detail: str = ""):
        self.engine = engine
        self.role = role
        self.reason = EngineUsabilityReason(reason)
        self.detail = detail
        msg = (
            f"audio engine '{engine}' is not usable for role '{role}': "
            f"{self.reason.value}"
        )
        if detail:
            msg += f" -- {detail}"
        super().__init__(msg)


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
    """Validate that ``name`` may actually run for ``role``; FAIL CLOSED.

    Returns the validated engine name on success. Raises :class:`EngineUnusable`
    with a classified reason otherwise -- never silently resolves to another
    engine:

    * ``MALFORMED_CONFIG`` -- no engine named ``name`` is registered.
    * ``INCOMPATIBLE_PROFILE`` -- the engine does not list ``role`` in ``roles``.

    There is NO ``GATED_BY_FLAG`` case (C6 -- the registry IS the menu): a
    registered, role-compatible engine is always usable. Disk/token/commercial
    checks require IO and are enforced downstream by the profile resolver and
    release gate, not here (the registry does no IO).
    """
    if not is_registered(name):
        raise EngineUnusable(
            name, role, EngineUsabilityReason.MALFORMED_CONFIG,
            f"no audio engine named '{name}' is registered",
        )
    eng = _REGISTRY[name]
    if role not in getattr(eng, "roles", ()):
        raise EngineUnusable(
            name, role, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
            f"engine '{name}' serves roles {tuple(getattr(eng, 'roles', ()))}, "
            f"not '{role}'",
        )
    # Registered + role-compatible == usable. The registry IS the menu: a
    # registered audio engine is selectable (C6 -- no flag gate; validation is the
    # operator's MANUAL process). Disk/token/commercial checks require IO and are
    # enforced downstream (this method does no IO). The byte-identical default
    # voice/music engines + the master-mux path are UNCHANGED -- only the
    # selectability of the non-default engines changes.
    return name


# ---------------------------------------------------------------------------
# GATE B S1 -- per-engine capability DECLARATIONS (the registry TABLE, not the
# adapters). Consumed by nodes/_otr_shared/capability_profiles.py to DERIVE
# the per-profile enable-set. DATA ONLY: this block changes no audio runtime
# behavior (the frozen audio spine is untouched). cpu_ok = the engine can
# render (slowly) with no GPU -- the cpu_floor tier filter.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "bark": {"required_toolchain": None,
             "requires_sidecar": False, "cpu_ok": True,
             "model_requirements": ["suno-bark"]},
    "kokoro": {"required_toolchain": None,
               "requires_sidecar": False, "cpu_ok": True,
               "model_requirements": ["kokoro-82m"]},
    "indextts2": {"required_toolchain": None,
                  "requires_sidecar": True, "cpu_ok": False,
                  "model_requirements": ["indextts2"]},
    "chatterbox": {"required_toolchain": None,
                   "requires_sidecar": True, "cpu_ok": False,
                   "model_requirements": ["chatterbox"]},
    "dia": {"required_toolchain": None,
            "requires_sidecar": True, "cpu_ok": False,
            "model_requirements": ["dia-1.6b"]},
    "musicgen": {"required_toolchain": None,
                 "requires_sidecar": False, "cpu_ok": True,
                 "model_requirements": ["musicgen-small"]},
    "stable_audio_music": {"required_toolchain": None, "requires_sidecar": False,
                           "cpu_ok": False, "model_requirements": ["stable-audio-open"]},
    "stable_audio_3": {"required_toolchain": None, "requires_sidecar": False,
                       "cpu_ok": False, "model_requirements": ["stable-audio-open-3"]},
}
