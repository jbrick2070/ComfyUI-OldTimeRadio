"""Additive CPU coverage for the role<->required-inputs engine filter (AS-1).

New, self-contained tests for nodes/_otr_shared/role_compat.py exercising the
pure functions directly (the platform integration is covered in
tests/test_video_platform_aseam.py). REWRITTEN 2026-07-01 (rip-sfx-broll):
the Role enum is exactly {announcer_visual, music_visual, character_video};
the scene_broll / background_abstract roles are GONE and unknown/dead role
tokens raise RoleCompatError. Pure stdlib + the module under test. UTF-8, no
BOM, ASCII, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import role_compat as rc


AUDIO_FACE = {
    "engine_id": "humo_like",
    "roles": (
        rc.Role.ANNOUNCER_VISUAL.value,
        rc.Role.CHARACTER_VIDEO.value,
        rc.Role.MUSIC_VISUAL.value,
    ),
    "required_inputs": ("text_prompt", "init_image", "audio_ref"),
}
TEXT_ONLY = {
    "engine_id": "abstract_like",
    "roles": tuple(rc.ROLES),
    "required_inputs": ("text_prompt",),
}


def test_roles_tuple_matches_enum():
    assert rc.ROLES == tuple(r.value for r in rc.Role)


def test_role_enum_is_exactly_three():
    assert {r.value for r in rc.Role} == {
        "announcer_visual", "music_visual", "character_video",
    }


def test_role_available_inputs_subset_of_input_tokens():
    for role in rc.ROLES:
        assert rc.role_available_inputs(role) <= rc.INPUT_TOKENS


def test_role_available_inputs_unknown_raises():
    with pytest.raises(rc.RoleCompatError):
        rc.role_available_inputs("no_such_role")


@pytest.mark.parametrize("dead", ["scene_broll", "background_abstract", "sfx"])
def test_dead_roles_raise_role_compat_error(dead):
    # rip-sfx-broll: the removed roles are UNKNOWN tokens now -- the string
    # keys were deleted from ROLE_AVAILABLE_INPUTS with the enum members, so
    # the whole capability subsystem rejects them loudly.
    with pytest.raises(rc.RoleCompatError):
        rc.role_available_inputs(dead)
    with pytest.raises(rc.RoleCompatError):
        rc.engine_fits_role(TEXT_ONLY, dead)
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role(dead, [TEXT_ONLY])


def test_every_surviving_role_supplies_full_vocabulary():
    # All three roles carry a real beat (audio slice + mintable still), so
    # each supplies the full input vocabulary.
    for role in rc.ROLES:
        assert rc.role_available_inputs(role) == frozenset(
            {"text_prompt", "init_image", "audio_ref", "base_clip_ref"}
        )


def test_audio_engine_offered_in_all_three_roles():
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.ANNOUNCER_VISUAL.value) is True
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.CHARACTER_VIDEO.value) is True
    # music role SUPPLIES audio_ref (LTX-AV ltx_audio_in lane, M1: the driver
    # slices the per-beat master), so an audio_driven_face engine fits it too.
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.MUSIC_VISUAL.value) is True


def test_roles_whitelist_no_longer_gates():
    # Capability-only routing (operator 2026-06-22): the per-engine `roles` list
    # is NO LONGER a gate. An engine whose `roles` omit a role STILL fits it when
    # the role supplies its required inputs (model-agnostic).
    eng = {
        "engine_id": "x",
        "roles": (rc.Role.MUSIC_VISUAL.value,),   # omits character_video
        "required_inputs": ("text_prompt",),
    }
    assert rc.engine_fits_role(eng, rc.Role.CHARACTER_VIDEO.value) is True


def test_text_only_engine_fits_every_listed_role():
    for role in rc.ROLES:
        assert rc.engine_fits_role(TEXT_ONLY, role) is True


def test_unknown_input_token_is_failed_closed():
    eng = {
        "engine_id": "depthy",
        "roles": (rc.Role.CHARACTER_VIDEO.value,),
        "required_inputs": ("text_prompt", "depth_map"),
    }
    assert rc.engine_fits_role(eng, rc.Role.CHARACTER_VIDEO.value) is False


def test_missing_keys_and_non_dict_excluded_not_raised():
    # Missing required_inputs -> fail-closed (excluded, not raised).
    assert rc.engine_fits_role(
        {"engine_id": "a"}, rc.Role.CHARACTER_VIDEO.value
    ) is False
    assert rc.engine_fits_role(
        "not_a_dict", rc.Role.CHARACTER_VIDEO.value
    ) is False


def test_filter_engines_for_role_is_order_preserving_and_filtered():
    descriptors = [
        AUDIO_FACE,
        TEXT_ONLY,
        {"engine_id": "", "roles": tuple(rc.ROLES),
         "required_inputs": ("text_prompt",)},
        "garbage",
        {"no_engine_id": True},
    ]
    out = rc.filter_engines_for_role(
        rc.Role.ANNOUNCER_VISUAL.value, descriptors
    )
    assert out == ["humo_like", "abstract_like"]


def test_filter_engines_for_role_unknown_role_raises():
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role("bogus", [TEXT_ONLY])


def test_filter_engines_for_role_empty_iterable():
    assert rc.filter_engines_for_role(rc.Role.CHARACTER_VIDEO.value, []) == []
    assert rc.filter_engines_for_role(rc.Role.CHARACTER_VIDEO.value, None) == []
