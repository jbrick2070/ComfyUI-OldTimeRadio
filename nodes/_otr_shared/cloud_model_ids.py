"""Single source of truth for partner V3 (DYNAMICCOMBO_V3) model IDs -- S1 B4.

Several pinned partner rows declare their ``model`` input as
``COMFY_DYNAMICCOMBO_V3`` (``partner_nodes.yaml`` -- the pin EXCLUDES combo
options by design, ``pin_meta.combo_options_excluded: true``). An adapter must
NEVER forward the literal ``"COMFY_DYNAMICCOMBO_V3"`` placeholder to the partner
node (it would 400 / TypeError at the provider); it must send a RESOLVED provider
model id. Those resolved ids live HERE (one place, env-overridable) instead of
being scattered inline across adapters (B4: "single source of truth").

The DEFAULTS are best-effort provider model strings; because the pin excludes
the live combo options, they are marked VERIFY -- the operator confirms/overrides
via the env var at the live smoke (operator-gated). An empty resolution (no env,
no default) fails LOUD rather than shipping the placeholder.

Cold-import-clean: stdlib only.
"""
from __future__ import annotations

import os

from .cloud_media_backend import CloudErrorCode, CloudMediaError

__all__ = ["V3_MODEL_IDS", "DYNAMICCOMBO_PLACEHOLDER", "resolve_model_id"]

#: the placeholder token the pin leaves in place of the real combo value.
DYNAMICCOMBO_PLACEHOLDER = "COMFY_DYNAMICCOMBO_V3"

#: node_key -> {env, default}. ``default=""`` means "operator must set the env"
#: (the row stays uninvocable until then -- e.g. the video rows whose live combo
#: id is not yet confirmed). VERIFY every default against the live provider menu.
V3_MODEL_IDS = {
    # image rows (S1 stills lane -- invocable with a best-effort default):
    "cloud_nano_banana_2": {
        "env": "OTR_CLOUD_NANO_BANANA_MODEL",
        "default": "gemini-2.5-flash-image-preview",  # VERIFY vs Gemini menu
    },
    "cloud_seedream_2": {
        "env": "OTR_CLOUD_SEEDREAM_MODEL",
        "default": "seedream-4-0-250828",             # VERIFY vs ByteDance menu
    },
    # video rows (resolved in their own sprints; env-required until confirmed):
    "cloud_seedance_2": {"env": "OTR_CLOUD_SEEDANCE_MODEL", "default": ""},
    "cloud_wan_i2v": {"env": "OTR_CLOUD_WAN_MODEL", "default": ""},
}


def resolve_model_id(node_key: str) -> str:
    """Resolve a partner row's V3 ``model`` id -- env override else the checked-in
    default. Fails LOUD (never returns the placeholder or an empty string)."""
    spec = V3_MODEL_IDS.get(node_key)
    if spec is None:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"no V3 model-id mapping for {node_key!r} (add it to "
            f"cloud_model_ids.V3_MODEL_IDS)")
    value = (os.environ.get(spec["env"], "") or spec["default"] or "").strip()
    if not value:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"{node_key}: no model id -- set {spec['env']} to the provider's "
            f"model id (the pin excludes the live DYNAMICCOMBO_V3 options)")
    if value == DYNAMICCOMBO_PLACEHOLDER:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"{node_key}: refusing to forward the {DYNAMICCOMBO_PLACEHOLDER!r} "
            f"placeholder as a model id -- set {spec['env']}")
    return value
