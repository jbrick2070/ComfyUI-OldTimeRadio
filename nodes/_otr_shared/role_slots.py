"""ONE shared role -> video-slot map (Route-A, 2026-06-28 HuMo-14B promotion;
rip-sfx-broll 2026-07-01: scene_broll / background_abstract roles REMOVED).

The legacy platform routed character / scene-broll / background-abstract beats
through a SINGLE ``other_beats_video_model`` slot. Route-A split them; the
2026-07-01 rip then removed the scene_broll / background_abstract roles
outright (proven to receive ZERO beats), leaving ``character_video`` as the
only former other-beats role. This module is the SINGLE source of the
role -> slot rule, imported identically by the four former duplicate maps so
they cannot drift:

* ``OTR_VideoDirector`` (``VIDEO_SLOT_ROLES`` + ``_role_aspects``),
* ``OTR_ShotLock`` (``build_execution_plan`` engine pick),
* ``OTR_ImageDirector`` (the image-prompt role -> video-slot join),
* ``OTR_ImageGenDispatcher`` (per-role still-consumed lookup).

Dependency-free: stdlib + the :class:`Role` enum from :mod:`role_compat`. No
torch / comfy / numpy at module scope (cold-import clean, V-12).

Migration note: a ``video_models`` policy that carries only the legacy
``other_beats_video_model`` slot still resolves for ``character_video`` --
that ONE fallback is the documented Route-A migration lane for old profiles /
saved graphs. It is an EMPTY-SLOT fallback only; an UNKNOWN role token always
RAISES (NO FALLBACKS, rip-sfx-broll).
"""
from __future__ import annotations

from .role_compat import Role

#: The slot the former other-beats roles shared before Route-A. Kept as the
#: migration fallback so old profiles / saved graphs keep resolving the
#: character lane.
LEGACY_OTHER_BEATS_SLOT = "other_beats_video_model"

#: The former other-beats role(s) still allowed to fall back to the legacy
#: slot when their dedicated slot is EMPTY (migration scope). Post-rip this is
#: only ``character_video``.
_OTHER_BEATS_ROLES = (
    Role.CHARACTER_VIDEO.value,
)

#: role -> its dedicated per-role video slot. The SINGLE map the four
#: consumers share. Unknown roles are NOT mapped -- :func:`slot_for_role`
#: raises on them.
ROLE_TO_VIDEO_SLOT = {
    Role.ANNOUNCER_VISUAL.value: "announcer_video_model",
    Role.MUSIC_VISUAL.value: "music_video_model",
    Role.CHARACTER_VIDEO.value: "character_video_model",
}

#: video slot -> the role(s) it must be capability-compatible with (the
#: fail-closed ``role_compat`` filter input). The inverse of
#: :data:`ROLE_TO_VIDEO_SLOT`, PLUS the legacy slot, which serves only the
#: character lane now.
VIDEO_SLOT_ROLES = {
    "announcer_video_model": ("announcer_visual",),
    "music_video_model": ("music_visual",),
    "character_video_model": ("character_video",),
    LEGACY_OTHER_BEATS_SLOT: _OTHER_BEATS_ROLES,
}

#: The per-role video slots OTR_VideoDirector emits / the profile applier knows, in
#: canonical (serialized-widget) order. EXCLUDES the legacy slot (a
#: migration-only INPUT, never emitted fresh).
PER_ROLE_VIDEO_SLOTS = (
    "announcer_video_model",
    "music_video_model",
    "character_video_model",
)

#: The per-role slot Route-A added that survives the 2026-07-01 rip (the
#: scene_broll / background_abstract slots were removed with their roles).
NEW_ROUTE_A_VIDEO_SLOTS = (
    "character_video_model",
)


def _engine_id_of(entry) -> str:
    """Bare engine id from a slot value: either a ``{"engine_id": ...}`` dict
    (the OTR_VideoDirector resolved shape) or a bare id string. Empty -> ""."""
    if isinstance(entry, dict):
        return str(entry.get("engine_id") or "")
    return str(entry or "")


def slot_for_role(role: str) -> str:
    """The dedicated per-role video slot for ``role``.

    NO FALLBACKS (rip-sfx-broll, 2026-07-01): an unknown role RAISES
    ``ValueError`` -- the historical silent map to the legacy other-beats
    slot was exactly the kind of fallback that let dead roles ride along
    unnoticed. A raise here names the bad token so the producer gets fixed.
    """
    slot = ROLE_TO_VIDEO_SLOT.get(role)
    if slot is None:
        raise ValueError(
            f"slot_for_role: unknown video role {role!r}; known roles: "
            f"{tuple(ROLE_TO_VIDEO_SLOT)}. scene_broll/background_abstract "
            f"were removed 2026-07-01 (rip-sfx-broll). NO FALLBACKS."
        )
    return slot


def engine_id_for_role(video_models, role: str) -> str:
    """Resolve the engine id ``role`` should use from a ``video_models`` policy.

    The role is validated FIRST via :func:`slot_for_role` (unknown role ->
    ``ValueError``, never a silent lane). Then reads the role's dedicated
    per-role slot; if that slot is absent or EMPTY and the role is
    ``character_video``, falls back to the legacy ``other_beats_video_model``
    slot (old profiles / saved graphs -- the documented migration lane; this
    is an empty-SLOT fallback, never an unknown-ROLE fallback). Each slot
    value may be a bare engine-id string or a ``{"engine_id": ...}`` dict.
    Returns "" when nothing resolves (fail-open to the caller's own handling).
    """
    slot = slot_for_role(role)  # raises on unknown role
    vm = video_models or {}
    eid = _engine_id_of(vm.get(slot))
    if eid:
        return eid
    if role in _OTHER_BEATS_ROLES:
        return _engine_id_of(vm.get(LEGACY_OTHER_BEATS_SLOT))
    return eid
