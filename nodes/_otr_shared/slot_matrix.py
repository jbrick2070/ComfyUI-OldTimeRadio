"""All-5-slots profile builder for the canonical-JSON soak (C5, 2026-06-30).

The slot-audit soak must drive the REAL workflow (workflows/otr_scifi_16gb_full.json)
with ALL FIVE video roles set INDEPENDENTLY -- announcer / music / character /
scene_broll / background_abstract -- via the capability-profile role_overrides, so
the applier patches each of the five OTR_VideoDirector video widgets (by node TYPE
through ``config/profiles/widget_mapping.json``, never a node id). It must NEVER
lean on the legacy ``other_beats_visual`` fallback for character / scene_broll /
background_abstract (that fallback is exactly what masked the per-slot drift the
sprint fixes).

This module is the ONE pure builder both the live soak and the offline mock test
use, so "all 5 slots are set" has a single source. Cold-import clean: only
``copy`` at module scope; the video registry is imported lazily for the eligibility
helper. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import copy
from typing import Optional

#: role_compat role token -> profile role_overrides KEY (whose widget_mapping
#: target is the matching OTR_VideoDirector per-role video model widget). These are
#: the FIVE independent slots; ``other_beats_visual`` is deliberately ABSENT (it is
#: the legacy fallback the soak must not use).
ROLE_TO_PROFILE_KEY: dict = {
    "announcer_visual": "announcer_visual",
    "music_visual": "music_visual",
    "character_video": "character_visual",
    "scene_broll": "scene_broll_visual",
    "background_abstract": "background_abstract_visual",
}

#: The five role_compat role tokens, in canonical order.
FIVE_ROLES: tuple = tuple(ROLE_TO_PROFILE_KEY.keys())

#: The three per-role IMAGE override keys (announcer / music / other-beats image).
IMAGE_KEYS: tuple = ("announcer_image", "music_image", "other_beats_image")

#: Named baselines for the slots NOT under test: a still video carrier that fits
#: every role + always renders, and the gen-1 image engine.
DEFAULT_VIDEO_BASELINE = "still_flat"
DEFAULT_IMAGE_BASELINE = "flux_gen1"


def build_all_five_role_profile(base_profile: dict, role_engines: Optional[dict] = None,
                                *, video_baseline: str = DEFAULT_VIDEO_BASELINE,
                                image_baseline: str = DEFAULT_IMAGE_BASELINE,
                                image_engines: Optional[dict] = None) -> dict:
    """A DEEP COPY of ``base_profile`` with ALL FIVE video role keys set.

    ``role_engines`` maps a role_compat role token (see :data:`FIVE_ROLES`) to the
    engine for that slot; any role absent falls to ``video_baseline``. The legacy
    ``other_beats_visual`` key is REMOVED so nothing routes through the fallback.
    The three image slots default to ``image_baseline`` (override per key via
    ``image_engines``). Pure -- never touches the graph; the applier does that."""
    role_engines = role_engines or {}
    image_engines = image_engines or {}
    profile = copy.deepcopy(base_profile)
    ro = profile.setdefault("role_overrides", {})
    # Set the five INDEPENDENT video slots; drop the legacy other-beats fallback.
    ro.pop("other_beats_visual", None)
    for role in FIVE_ROLES:
        ro[ROLE_TO_PROFILE_KEY[role]] = role_engines.get(role, video_baseline)
    for key in IMAGE_KEYS:
        ro[key] = image_engines.get(key, image_baseline)
    return profile


def profile_keys_for_all_five() -> list:
    """The five dotted profile keys the applier will patch (one per video slot)."""
    return [f"role_overrides.{ROLE_TO_PROFILE_KEY[r]}" for r in FIVE_ROLES]


def eligible_engines_for_role(role: str) -> list:
    """Capability-eligible engine names for ``role`` (the C2 registry override).
    Lazy registry import so this module stays cold-import clean."""
    from nodes._otr_video_engines import registry as _vreg
    return _vreg.engines_for_role(role)


def build_eligibility_matrix() -> dict:
    """``{role: [eligible engine ids]}`` over the five slots -- the matrix the soak
    enumerates (capability-grounded, C4)."""
    return {role: eligible_engines_for_role(role) for role in FIVE_ROLES}
