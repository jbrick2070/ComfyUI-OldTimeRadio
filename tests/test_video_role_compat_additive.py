"""Additive CPU coverage for the role<->required-inputs engine filter (AS-1).

New, self-contained tests for nodes/_otr_shared/role_compat.py exercising the
pure functions directly (the platform integration is covered in
tests/test_video_platform_aseam.py). Pins: an audio-driven engine is offered
only in roles that supply audio_ref; a text-only engine fits every role it
lists; fail-closed exclusion of malformed / unknown-token descriptors; and an
unknown role raising. Pure stdlib + the module under test. UTF-8, no BOM, ASCII,
SFW.
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


def test_role_available_inputs_subset_of_input_tokens():
    for role in rc.ROLES:
        assert rc.role_available_inputs(role) <= rc.INPUT_TOKENS


def test_role_available_inputs_unknown_raises():
    with pytest.raises(rc.RoleCompatError):
        rc.role_available_inputs("no_such_role")


def test_background_abstract_supplies_only_text():
    assert rc.role_available_inputs(
        rc.Role.BACKGROUND_ABSTRACT.value
    ) == frozenset({"text_prompt"})


def test_audio_engine_offered_only_where_audio_is_available():
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.ANNOUNCER_VISUAL.value) is True
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.CHARACTER_VIDEO.value) is True
    # music role does not supply audio_ref -> excluded even though it is listed
    assert rc.engine_fits_role(AUDIO_FACE, rc.Role.MUSIC_VISUAL.value) is False


def test_engine_excluded_when_role_not_listed():
    eng = {
        "engine_id": "x",
        "roles": (rc.Role.MUSIC_VISUAL.value,),
        "required_inputs": ("text_prompt",),
    }
    assert rc.engine_fits_role(eng, rc.Role.BACKGROUND_ABSTRACT.value) is False


def test_text_only_engine_fits_every_listed_role():
    for role in rc.ROLES:
        assert rc.engine_fits_role(TEXT_ONLY, role) is True


def test_unknown_input_token_is_failed_closed():
    eng = {
        "engine_id": "depthy",
        "roles": (rc.Role.SCENE_BROLL.value,),
        "required_inputs": ("text_prompt", "depth_map"),
    }
    assert rc.engine_fits_role(eng, rc.Role.SCENE_BROLL.value) is False


def test_missing_keys_and_non_dict_excluded_not_raised():
    assert rc.engine_fits_role(
        {"engine_id": "a"}, rc.Role.SCENE_BROLL.value
    ) is False
    assert rc.engine_fits_role(
        {"roles": (), "required_inputs": ()}, rc.Role.SCENE_BROLL.value
    ) is False
    assert rc.engine_fits_role("not_a_dict", rc.Role.SCENE_BROLL.value) is False


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
    assert rc.filter_engines_for_role(rc.Role.SCENE_BROLL.value, []) == []
    assert rc.filter_engines_for_role(rc.Role.SCENE_BROLL.value, None) == []
