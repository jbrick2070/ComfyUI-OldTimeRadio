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
#: every role, and the gen-1 image engine. ("+ always renders" was here and is no
#: longer true -- lane 17 gave still_flat `_require_still`, so a MISSING still is
#: a loud refusal. Harmless today because this module only builds profile dicts
#: and workflow JSON offline and never calls render_clip, but a future live soak
#: built on this baseline must supply a still.)
DEFAULT_VIDEO_BASELINE = "still_flat"
DEFAULT_IMAGE_BASELINE = "flux_gen1"


def eligible_engines_for_role(role: str) -> list:
    """Capability-eligible engine names for ``role`` (the C2 registry override).
    Lazy registry import so this module stays cold-import clean.

    RELATIVE, for the reason written out in ``boot_contracts``'s Sage probe
    (lane 19, 2026-08-12): an ABSOLUTE ``nodes._otr_video_engines`` import
    resolves ``nodes`` against sys.path, which is THIS package in the CPU suite
    and ComfyUI's own node-registry module on a running server. This one raised
    outright rather than degrading quietly, but it was the same defect and it
    was the only other instance in either shared package, so it is swept here
    rather than left to be found by whichever caller reaches it first on a
    server (lesson L13).
    """
    from .._otr_video_engines import registry as _vreg
    return _vreg.engines_for_role(role)
