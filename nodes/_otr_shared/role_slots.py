"""ONE shared role -> video-slot map (Route-A, 2026-06-28 HuMo-14B promotion).

The legacy platform routed character / scene-broll / background-abstract beats
through a SINGLE ``other_beats_video_model`` slot. The 14B promotion needs them
SPLIT: ``character_video`` must ride ``humo_14B_169`` (face + audio, no
fallbacks) while ``scene_broll`` / ``background_abstract`` ride lighter engines.
This module is the SINGLE source of the role -> slot rule, imported identically
by the four former duplicate maps so they cannot drift:

* ``OTR_VideoDirector`` (``VIDEO_SLOT_ROLES`` + ``_role_aspects``),
* ``OTR_ShotLock`` (``build_execution_plan`` engine pick),
* ``OTR_ImageDirector`` (the image-prompt role -> video-slot join),
* ``OTR_ImageGenDispatcher`` (per-role still-consumed lookup).

Dependency-free: stdlib + the :class:`Role` enum from :mod:`role_compat`. No
torch / comfy / numpy at module scope (cold-import clean, V-12).

Back-compatible (migration): a ``video_models`` policy that carries only the
legacy ``other_beats_video_model`` slot still resolves -- the three other-beats
roles fall back to it. That keeps every saved graph and the shipped profiles
(16gb_full / 8gb_lite / cpu_floor ``role_overrides.other_beats_visual``) working
until they are re-applied with the new per-role keys.
"""
from __future__ import annotations

from .role_compat import Role

#: The slot the three other-beats roles shared before Route-A. Kept as the
#: migration fallback so old profiles / saved graphs keep resolving.
LEGACY_OTHER_BEATS_SLOT = "other_beats_video_model"

#: The three roles that historically shared the legacy slot (migration scope).
_OTHER_BEATS_ROLES = (
    Role.CHARACTER_VIDEO.value,
    Role.SCENE_BROLL.value,
    Role.BACKGROUND_ABSTRACT.value,
)

#: role -> its dedicated per-role video slot (Route-A). The SINGLE map the four
#: consumers share.
ROLE_TO_VIDEO_SLOT = {
    Role.ANNOUNCER_VISUAL.value: "announcer_video_model",
    Role.MUSIC_VISUAL.value: "music_video_model",
    Role.CHARACTER_VIDEO.value: "character_video_model",
    Role.SCENE_BROLL.value: "scene_broll_video_model",
    Role.BACKGROUND_ABSTRACT.value: "background_abstract_video_model",
}

#: video slot -> the role(s) it must be capability-compatible with (the
#: fail-closed ``role_compat`` filter input). The inverse of
#: :data:`ROLE_TO_VIDEO_SLOT`, PLUS the legacy slot, which must still fit all
#: three other-beats roles (a legacy pick has to satisfy every role it serves).
VIDEO_SLOT_ROLES = {
    "announcer_video_model": ("announcer_visual",),
    "music_video_model": ("music_visual",),
    "character_video_model": ("character_video",),
    "scene_broll_video_model": ("scene_broll",),
    "background_abstract_video_model": ("background_abstract",),
    LEGACY_OTHER_BEATS_SLOT: _OTHER_BEATS_ROLES,
}

#: The per-role video slots OTR_VideoDirector emits / the profile applier knows, in
#: canonical (serialized-widget) order. EXCLUDES the legacy slot (a
#: migration-only INPUT, never emitted fresh).
PER_ROLE_VIDEO_SLOTS = (
    "announcer_video_model",
    "music_video_model",
    "character_video_model",
    "scene_broll_video_model",
    "background_abstract_video_model",
)

#: The three NEW per-role slots Route-A adds (the slots that did not exist before
#: the split). OTR_VideoDirector appends widgets for these; the profile schema /
#: applier / widget_mapping learn exactly these three.
NEW_ROUTE_A_VIDEO_SLOTS = (
    "character_video_model",
    "scene_broll_video_model",
    "background_abstract_video_model",
)


def _engine_id_of(entry) -> str:
    """Bare engine id from a slot value: either a ``{"engine_id": ...}`` dict
    (the OTR_VideoDirector resolved shape) or a bare id string. Empty -> ""."""
    if isinstance(entry, dict):
        return str(entry.get("engine_id") or "")
    return str(entry or "")


def slot_for_role(role: str) -> str:
    """The dedicated per-role video slot for ``role`` (Route-A).

    An unknown role tolerantly maps to the legacy other-beats slot (the four
    consumers historically defaulted there), never raises -- routing must not
    crash a normal episode on an unexpected role label.
    """
    return ROLE_TO_VIDEO_SLOT.get(role, LEGACY_OTHER_BEATS_SLOT)


def engine_id_for_role(video_models, role: str) -> str:
    """Resolve the engine id ``role`` should use from a ``video_models`` policy.

    Reads the role's dedicated per-role slot first; if that slot is absent or
    empty AND the role is one of the three other-beats roles, falls back to the
    legacy ``other_beats_video_model`` slot (old profiles / saved graphs). Each
    slot value may be a bare engine-id string or a ``{"engine_id": ...}`` dict.
    Returns "" when nothing resolves (fail-open to the caller's own handling).
    """
    vm = video_models or {}
    eid = _engine_id_of(vm.get(slot_for_role(role)))
    if eid:
        return eid
    if role in _OTHER_BEATS_ROLES:
        return _engine_id_of(vm.get(LEGACY_OTHER_BEATS_SLOT))
    return eid
