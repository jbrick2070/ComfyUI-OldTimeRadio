"""Capability-only video engine routing (role_compat.engine_fits_role).

The per-engine `roles` whitelist is no longer a gate -- eligibility is purely
`required_inputs <= role_available_inputs` (operator 2026-06-22, model-agnostic).
REWRITTEN 2026-07-01 (rip-sfx-broll): the retired_role_a / retired_role_b
roles are GONE; every surviving role supplies the full input vocabulary and
dead role tokens raise RoleCompatError. Deterministic, CPU-only.
"""

import importlib

import pytest

rc = importlib.import_module("nodes._otr_shared.role_compat")


def _desc(required, roles=None):
    d = {"engine_id": "x", "required_inputs": tuple(required)}
    if roles is not None:
        d["roles"] = tuple(roles)
    return d


# Real declarations (grounded): wan's roles whitelist OMITS announcer_visual.
WAN = _desc(("init_image",), roles=("music_visual", "character_video"))
LTX = _desc(("text_prompt",), roles=("music_visual", "announcer_visual"))
VIS = _desc(("audio_ref",), roles=("announcer_visual", "music_visual", "character_video"))
C3D = _desc(("audio_ref", "init_image"), roles=("announcer_visual", "character_video"))


def test_wan_now_fits_announcer_despite_roles_omission():
    # THE live-wall fix: wan's roles omits announcer_visual, but it supplies init_image.
    assert rc.engine_fits_role(WAN, "announcer_visual") is True
    assert rc.engine_fits_role(WAN, "music_visual") is True
    assert rc.engine_fits_role(WAN, "character_video") is True


def test_dead_roles_raise_loud():
    # rip-sfx-broll: retired_role_a / retired_role_b are unknown tokens now.
    for dead in ("retired_role_a", "retired_role_b"):
        with pytest.raises(rc.RoleCompatError):
            rc.engine_fits_role(WAN, dead)


def test_text_prompt_broll_fits_every_role():
    for role in rc.ROLES:
        assert rc.engine_fits_role(LTX, role) is True, role


def test_audio_specials_only_fit_audio_supplying_roles():
    # All three surviving roles supply audio_ref (per-beat master slice).
    assert rc.engine_fits_role(VIS, "announcer_visual") is True
    assert rc.engine_fits_role(VIS, "music_visual") is True
    assert rc.engine_fits_role(VIS, "character_video") is True


def test_character_3d_needs_audio_and_still():
    assert rc.engine_fits_role(C3D, "announcer_visual") is True
    assert rc.engine_fits_role(C3D, "music_visual") is True
    assert rc.engine_fits_role(C3D, "character_video") is True


def test_roles_whitelist_is_ignored():
    # a descriptor whose roles EXCLUDES a role still fits it by capability.
    d = _desc(("text_prompt",), roles=("music_visual",))
    assert rc.engine_fits_role(d, "announcer_visual") is True


def test_no_roles_key_fits_by_capability():
    d = _desc(("init_image",))  # no `roles` key at all -> must NOT fail-closed
    assert rc.engine_fits_role(d, "announcer_visual") is True
    assert rc.engine_fits_role(d, "character_video") is True


def test_fail_closed():
    assert rc.engine_fits_role({"engine_id": "x"}, "announcer_visual") is False  # no required_inputs
    assert rc.engine_fits_role(_desc(("frobnicate",)), "announcer_visual") is False  # unknown token
    assert rc.engine_fits_role("not-a-dict", "announcer_visual") is False


def test_unknown_role_raises():
    with pytest.raises(rc.RoleCompatError):
        rc.engine_fits_role(LTX, "no_such_role")


def test_non_regression_superset():
    # For every (engine, role): old eligibility (roles INTERSECT capability) IMPLIES
    # new eligibility (capability-only). Proves zero working routes are removed.
    for d in (WAN, LTX, VIS, C3D):
        roles = d.get("roles") or ()
        required = set(d["required_inputs"])
        for role in rc.ROLES:
            available = rc.role_available_inputs(role)
            old_fit = (role in roles) and (required <= available) and (required <= rc.INPUT_TOKENS)
            if old_fit:
                assert rc.engine_fits_role(d, role) is True, f"REGRESSION: {d['engine_id']} lost {role}"
