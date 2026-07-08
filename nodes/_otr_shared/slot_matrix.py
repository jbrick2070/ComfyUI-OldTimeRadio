"""All-slots profile builder for the canonical-JSON soak (C5, 2026-06-30;
rip-sfx-broll 2026-07-01: shrunk from five roles to three).

The slot-audit soak must drive the REAL workflow (workflows/otr_canonical.json)
with ALL video roles set INDEPENDENTLY -- announcer / music / character -- via the
capability-profile role_overrides, so the applier patches each OTR_VideoDirector
video widget (by node TYPE through ``config/profiles/widget_mapping.json``, never a
node id). It must NEVER lean on a legacy catch-all fallback for
character (that fallback is exactly what masked the per-slot drift the sprint
fixed).

This module is the ONE pure builder both the live soak and the offline mock test
use, so "all slots are set" has a single source. Cold-import clean: only
``copy`` at module scope; the video registry is imported lazily for the eligibility
helper. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import copy
from typing import Optional

#: role_compat role token -> profile role_overrides KEY (whose widget_mapping
#: target is the matching OTR_VideoDirector per-role video model widget). These are
#: the THREE independent slots; no legacy catch-all fallback is present (the soak
#: must not use one). retired_role_a_visual /
#: retired_role_b_visual were removed 2026-07-01 with their roles.
ROLE_TO_PROFILE_KEY: dict = {
    "announcer_visual": "announcer_visual",
    "music_visual": "music_visual",
    "character_video": "character_visual",
}

#: The role_compat role tokens, in canonical order.
ALL_ROLES: tuple = tuple(ROLE_TO_PROFILE_KEY.keys())

#: The three per-role IMAGE override keys (announcer / music / character image --
#: the character image slot carries the character stills).
IMAGE_KEYS: tuple = ("announcer_image", "music_image", "character_image")

#: Named baselines for the slots NOT under test: a still video carrier that fits
#: every role + always renders, and the gen-1 image engine.
DEFAULT_VIDEO_BASELINE = "still_flat"
DEFAULT_IMAGE_BASELINE = "flux_gen1"

# (No dead-key scrubber: retired role_overrides keys now fail LOUD at
# apply_profile -- clean break, no silent scrub.)


def build_all_role_profile(base_profile: dict, role_engines: Optional[dict] = None,
                           *, video_baseline: str = DEFAULT_VIDEO_BASELINE,
                           image_baseline: str = DEFAULT_IMAGE_BASELINE,
                           image_engines: Optional[dict] = None) -> dict:
    """A DEEP COPY of ``base_profile`` with ALL THREE video role keys set.

    ``role_engines`` maps a role_compat role token (see :data:`ALL_ROLES`) to the
    engine for that slot; any role absent falls to ``video_baseline``. Retired
    role_overrides keys are no longer scrubbed -- they fail LOUD at apply
    (clean break). The three image slots default to
    ``image_baseline`` (override per key via ``image_engines``). Pure -- never
    touches the graph; the applier does that."""
    role_engines = role_engines or {}
    image_engines = image_engines or {}
    profile = copy.deepcopy(base_profile)
    ro = profile.setdefault("role_overrides", {})
    # Set the three INDEPENDENT video slots (retired keys fail loud at apply).
    for role in ALL_ROLES:
        ro[ROLE_TO_PROFILE_KEY[role]] = role_engines.get(role, video_baseline)
    for key in IMAGE_KEYS:
        ro[key] = image_engines.get(key, image_baseline)
    return profile


def profile_keys_for_all_roles() -> list:
    """The three dotted profile keys the applier will patch (one per video slot)."""
    return [f"role_overrides.{ROLE_TO_PROFILE_KEY[r]}" for r in ALL_ROLES]


def eligible_engines_for_role(role: str) -> list:
    """Capability-eligible engine names for ``role`` (the C2 registry override).
    Lazy registry import so this module stays cold-import clean."""
    from nodes._otr_video_engines import registry as _vreg
    return _vreg.engines_for_role(role)


def build_eligibility_matrix() -> dict:
    """``{role: [eligible engine ids]}`` over the three slots -- the matrix the
    soak enumerates (capability-grounded, C4)."""
    return {role: eligible_engines_for_role(role) for role in ALL_ROLES}
