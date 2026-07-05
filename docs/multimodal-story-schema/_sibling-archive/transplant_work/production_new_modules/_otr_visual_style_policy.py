"""Production-side VisualStylePolicy reader + meta.visual_style helpers.

DESTINATION: ComfyUI-OldTimeRadio/nodes/_otr_visual_style_policy.py
STATUS: staged in the lab (transplant_work/); NOT installed yet.

The visual stage is SEPARATE and staged (locked pre-SFX r4 decision +
TRANSPLANT_MANIFEST risk note). This module only defines the policy reader,
the meta stamp/read helpers, and the tail accessors the shared seams
(finish_visual_prompt / compose_still_prompt) will consult. sci_fi_radio
must reproduce the current production constants byte-identically, so wiring
it changes nothing until a non-default style is selected.
"""

from __future__ import annotations

from typing import Any

#: Production motion-prompt role vocabulary (render_driver). Dead roles
#: (sfx / scene_broll / background_abstract) must stay dead.
ALLOWED_MOTION_ROLE_KEYS = frozenset(
    {"announcer", "music_open", "music_close", "music_inter"}
)

META_KEY = "visual_style"


class VisualPolicyError(ValueError):
    """The policy dict is unusable or a dead role was requested."""


REQUIRED_POLICY_FIELDS = ("style_id",)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise VisualPolicyError("visual_policy must be an object")
    missing = [
        f for f in REQUIRED_POLICY_FIELDS if not str(policy.get(f, "")).strip()
    ]
    if missing:
        raise VisualPolicyError(f"visual_policy missing required fields: {missing}")
    unknown = sorted(
        set(policy.get("motion_prompts", {})) - ALLOWED_MOTION_ROLE_KEYS
    )
    if unknown:
        raise VisualPolicyError(
            f"visual_policy {policy['style_id']!r} motion_prompts has unknown "
            f"role keys {unknown} (dead roles must stay dead)"
        )
    return policy


def stamp_meta(meta: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Stamp meta.visual_style = style_id. Refuses to overwrite a DIFFERENT
    existing stamp (two stamps disagreeing is a wiring bug, not a preference)."""

    policy = validate_policy(policy)
    existing = str(meta.get(META_KEY, "") or "")
    style_id = policy["style_id"]
    if existing and existing != style_id:
        raise VisualPolicyError(
            f"meta.visual_style already stamped {existing!r}; refusing to "
            f"restamp {style_id!r}"
        )
    meta[META_KEY] = style_id
    return meta


def read_style_id(meta: dict[str, Any]) -> str:
    """Read the stamped style id; empty string means 'pre-transplant ledger'
    and callers keep current production behavior (sci_fi_radio constants)."""

    return str((meta or {}).get(META_KEY, "") or "")


def tail_overrides(policy: dict[str, Any]) -> dict[str, str]:
    """The four tail replacements the shared seams consult. Empty string =
    keep the production constant (sci_fi_radio supplies the SAME strings, so
    default behavior is byte-identical either way).

    NAME MAPPING (kibitz r3, Claude Code M1 - do not misconnect at the
    transplant hunk):
        "era_tail"             <- policy.era_tail        -> ERA_TAIL_DEFAULT
        "style_tail"           <- policy.positive_tail   -> STYLE_TAIL_DEFAULT
        "image_grade_tail"     <- policy.image_grade_tail-> IMAGE_GRADE_TAIL
        "radio_broadcast_tail" <- policy.broadcast_tail  -> RADIO_BROADCAST_TAIL
    Never add JSON fields named "style_tail"/"radio_broadcast_tail" to
    VisualStylePolicy - those names are the override-dict vocabulary only."""

    policy = validate_policy(policy)
    return {
        "era_tail": str(policy.get("era_tail", "") or ""),
        "style_tail": str(policy.get("positive_tail", "") or ""),
        "image_grade_tail": str(policy.get("image_grade_tail", "") or ""),
        "radio_broadcast_tail": str(policy.get("broadcast_tail", "") or ""),
    }


def allow_radio_tails(policy: dict[str, Any]) -> bool:
    return bool(validate_policy(policy).get("allow_radio_tails", True))


def motion_prompt_for_role(policy: dict[str, Any], role: str) -> str:
    """Motion prompt for a role. Missing role = empty string (caller keeps
    the production _LTX table); UNKNOWN/dead role = loud error."""

    policy = validate_policy(policy)
    if role not in ALLOWED_MOTION_ROLE_KEYS:
        raise VisualPolicyError(
            f"unknown motion role {role!r}; allowed: {sorted(ALLOWED_MOTION_ROLE_KEYS)}"
        )
    return str(policy.get("motion_prompts", {}).get(role, "") or "")
