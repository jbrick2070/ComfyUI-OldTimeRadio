"""ONE shared role <-> required-inputs engine filter (A-Seam AS-1).

The model-agnostic platform offers, per role, a STATIC dropdown of every
registered engine (V-6: the COMBO is the full registry). But not every engine
fits every role -- an ``audio_driven_face`` engine needs an ``audio_ref`` that a
``background_abstract`` beat does not supply. This module is the SINGLE source of
that compatibility rule, imported identically by:

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
    """The five video roles (policy menu-filter keys, not pipelines).

    Maps onto the user-facing per-role selectors: ``announcer_visual`` -> A,
    ``music_visual`` -> B, and the three other-beats roles -> C.
    """

    ANNOUNCER_VISUAL = "announcer_visual"
    MUSIC_VISUAL = "music_visual"
    CHARACTER_VIDEO = "character_video"
    SCENE_BROLL = "scene_broll"
    BACKGROUND_ABSTRACT = "background_abstract"


#: Request-level input tokens (shared verbatim with the schema vocabulary in
#: ``nodes/_otr_video_engines/schemas.py``).
INPUT_TOKENS: frozenset = frozenset(
    {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
)


#: What each role CAN supply to an engine. An engine is offered in a role only
#: if every one of its ``required_inputs`` is available here. ``base_clip_ref``
#: is available in the motion-bearing roles (a provider engine can supply a base
#: clip); the pure-background role supplies only text.
ROLE_AVAILABLE_INPUTS: dict = {
    Role.ANNOUNCER_VISUAL.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
    # MUSIC_VISUAL supplies audio_ref (the per-beat slice of the frozen master)
    # so the LTX-AV ``ltx_av_music`` audio-reactive engine fits this role (M1,
    # unconditional). The slice is sync-loose for music -- precision is the talk
    # lane's job -- but the audio input is genuinely available here.
    Role.MUSIC_VISUAL.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
    Role.CHARACTER_VIDEO.value: frozenset(
        {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
    ),
    Role.SCENE_BROLL.value: frozenset(
        {"text_prompt", "init_image", "base_clip_ref"}
    ),
    Role.BACKGROUND_ABSTRACT.value: frozenset({"text_prompt"}),
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
    """True iff ``descriptor`` can serve ``role`` (fail-closed).

    Requires (1) ``role`` is in the engine's ``roles`` and (2) every token in
    the engine's ``required_inputs`` is available in the role. A descriptor
    missing ``roles`` / ``required_inputs``, or declaring an unknown input
    token, is treated as NOT fitting (excluded) rather than raising -- when
    unsure, do not offer it.
    """
    available = role_available_inputs(role)  # raises on unknown role
    if not isinstance(descriptor, dict):
        return False
    roles = descriptor.get("roles")
    required = descriptor.get("required_inputs")
    if roles is None or required is None:
        return False
    if role not in tuple(roles):
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
