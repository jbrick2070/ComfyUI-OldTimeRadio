"""ONE shared role <-> required-inputs engine filter (A-Seam AS-1).

The model-agnostic platform offers, per role, a STATIC dropdown of every
registered engine (V-6: the COMBO is the full registry). But not every engine
fits every role -- an engine only fits where the role supplies every one of its
``required_inputs``. This module is the SINGLE source of that compatibility
rule, imported identically by:

* ``OTR_VideoDirector`` -- to validate / annotate the user's per-role pick,
* ``OTR_ShotLock`` -- to fail closed on an incompatible locked pick,
* the image director (C1) and the 3D ``character_3d`` availability check (B),

so all three agree on one rule instead of drifting apart. Keeping it in ONE
dep-free module is the whole point of AS-1.

It runs at OTR_VideoDirector.execute / ShotLock.validate time -- NEVER at COMBO build
time (the enum stays the full static registry; filtering the COMBO itself would
be dynamic-widget mutation, which V-6 forbids).

Dependency-free + fail-closed: an unknown role raises :class:`RoleCompatError`
(a caller bug); a malformed / under-specified engine descriptor is EXCLUDED
(never offered, never crashes) -- when unsure, do not offer it.
"""
from __future__ import annotations

import enum
from typing import Iterable, TypedDict


class Role(str, enum.Enum):
    """The three video roles (policy menu-filter keys, not pipelines).

    Maps onto the user-facing per-role selectors: ``announcer_visual`` -> A,
    ``music_visual`` -> B, ``character_video`` -> C. The former
    ``scene_broll`` / ``background_abstract`` roles were RIPPED 2026-07-01
    (rip-sfx-broll: proven to receive ZERO beats -- the only producer was the
    dead ``sfx`` speaker-role plus a default fallback, both removed).
    Unknown role tokens FAIL LOUD via :class:`RoleCompatError`.
    """

    ANNOUNCER_VISUAL = "announcer_visual"
    MUSIC_VISUAL = "music_visual"
    CHARACTER_VIDEO = "character_video"


#: Request-level input tokens (shared verbatim with the schema vocabulary in
#: ``nodes/_otr_video_engines/schemas.py``).
INPUT_TOKENS: frozenset = frozenset(
    {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
)


#: What each role CAN supply to an engine. An engine is offered in a role only
#: if every one of its ``required_inputs`` is available here. All three
#: surviving roles carry a real beat with per-beat audio + a mintable still,
#: so each supplies the full input vocabulary.
ROLE_AVAILABLE_INPUTS: dict = {
    Role.ANNOUNCER_VISUAL.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
    # MUSIC_VISUAL supplies audio_ref (the per-beat slice of the frozen master)
    # so the LTX-AV audio-reactive engine fits this role (M1, unconditional).
    # The slice is sync-loose for music -- precision is the talk lane's job --
    # but the audio input is genuinely available here.
    Role.MUSIC_VISUAL.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
    Role.CHARACTER_VIDEO.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
}

#: The canonical role names (single-sourced).
ROLES: tuple = tuple(r.value for r in Role)


class EngineDescriptor(TypedDict, total=False):
    """The minimal engine shape ``filter_engines_for_role`` reads.

    Built from a registered adapter (``engine_id``, ``roles``,
    ``required_inputs``). ``total=False`` so callers may pass a superset dict;
    the filter only reads these three keys and treats any missing key as a
    fail-closed exclusion.
    """

    engine_id: str
    roles: tuple
    required_inputs: tuple


class RoleCompatError(ValueError):
    """Raised for an UNKNOWN role (a caller bug). Incompatible engines are
    silently excluded, not raised -- only a bad role argument raises."""


def role_available_inputs(role: str) -> frozenset:
    """Inputs a role can supply; raises :class:`RoleCompatError` if unknown."""
    if role not in ROLE_AVAILABLE_INPUTS:
        raise RoleCompatError(
            f"unknown video role '{role}'; known roles: {ROLES}"
        )
    return ROLE_AVAILABLE_INPUTS[role]


def engine_fits_role(descriptor, role: str) -> bool:
    """True iff ``descriptor`` can serve ``role`` (fail-closed, capability-only).

    Eligibility is PURELY capability: every token in the engine's
    ``required_inputs`` must be available in the role
    (``required_inputs <= role_available_inputs``). The legacy per-engine
    ``roles`` whitelist is NO LONGER a gate (operator 2026-06-22, model-agnostic
    routing): an engine fits any role whose inputs satisfy it -- audio-driven
    engines stay limited to audio-supplying roles BY CAPABILITY, while b-roll
    (text/still) engines fit every role. A descriptor missing ``required_inputs``
    or declaring an unknown input token is excluded fail-closed. ``default_roles``
    (the auto-default pick) is a separate concern and is unaffected.
    """
    available = role_available_inputs(role)  # raises on unknown role
    if not isinstance(descriptor, dict):
        return False
    required = descriptor.get("required_inputs")
    if required is None:
        return False
    required_set = set(required)
    # An engine that declares an input token outside the known vocabulary is
    # excluded fail-closed (we cannot prove the role supplies it).
    if not required_set <= INPUT_TOKENS:
        return False
    return required_set <= available


def filter_engines_for_role(role: str, engine_descriptors: Iterable) -> list:
    """Engine ids that fit ``role``, fail-closed, order-preserving.

    ``engine_descriptors`` is any iterable of :class:`EngineDescriptor`-shaped
    dicts (one per registered engine). Returns the subset of ``engine_id``s the
    role can actually drive -- the list the director annotates / ShotLock
    validates against. Raises :class:`RoleCompatError` only for an unknown role.
    """
    available = role_available_inputs(role)  # validate role up-front
    out = []
    for desc in engine_descriptors or []:
        if not isinstance(desc, dict):
            continue
        engine_id = desc.get("engine_id")
        if not engine_id:
            continue
        if engine_fits_role(desc, role):
            out.append(engine_id)
    return out
