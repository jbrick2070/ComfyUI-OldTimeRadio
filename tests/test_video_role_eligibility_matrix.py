"""C4 (2026-06-30) -- the ALL-ENGINES x ALL-ROLES eligibility matrix.

The pure CONTRACT test for the slot-audit fix: every REGISTERED video engine, in
each of the 5 video roles, is eligible IFF it fits the role BY CAPABILITY
(``role_compat.engine_fits_role`` on the shared ``descriptor_for_engine``) -- NOT
a flat ``True``, and NOT the legacy per-engine ``roles`` whitelist (the D1 drift
C2 killed). Capability-grounded exclusions ARE expected: text-only floors fit all
5 roles, audio-driven engines are excluded from retired_role_b / retired_role_a
(which cannot supply ``audio_ref``), image-to-video engines are excluded from
retired_role_b (no ``init_image``).

This is an eligibility/contract test only -- it never renders (no GPU, no model
load) and does not duplicate the canonical-render coverage (that is C5's soak).
Importing the package self-registers the full engine set. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import types

import pytest

import nodes._otr_video_engines  # noqa: F401  (self-registers every engine)
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import registry as vreg

_ENGINES = vreg.all_engine_names()
_ROLES = rc.ROLES


@pytest.mark.parametrize("name", _ENGINES)
@pytest.mark.parametrize("role", _ROLES)
def test_eligibility_is_capability_grounded(name, role):
    """For a real registered engine, registry eligibility == capability fit."""
    desc = vreg.descriptor_for_engine(name)
    if desc["required_inputs"] is None:
        pytest.skip(f"{name} declares no required_inputs -> legacy fail-soft path")
    expected = rc.engine_fits_role(desc, role)
    # engines_for_role membership matches capability
    assert (name in vreg.engines_for_role(role)) is expected, (name, role)
    # assert_usable agrees: returns the name iff fit, else raises EngineUnusable
    if expected:
        assert vreg.assert_usable(name, role) == name
    else:
        with pytest.raises(vreg.EngineUnusable):
            vreg.assert_usable(name, role)


def test_matrix_exclusion_machinery_still_bites():
    """Sanity: the capability filter is not a rubber stamp. rip-sfx-broll
    (2026-07-01): the input-poor roles (retired_role_a/retired_role_b) died,
    so every SURVIVING role supplies the full vocabulary and every registered
    engine fits everywhere -- real exclusions now come from (a) descriptors
    requiring UNKNOWN tokens and (b) DEAD role tokens raising."""
    # (a) an unknown required token is excluded fail-closed
    bogus = {"engine_id": "needs_depth", "roles": tuple(_ROLES),
             "required_inputs": ("depth_map",)}
    for role in _ROLES:
        assert rc.engine_fits_role(bogus, role) is False
    # (b) a dead role raises through the same shared filter
    ok = {"engine_id": "txt", "roles": tuple(_ROLES),
          "required_inputs": ("text_prompt",)}
    with pytest.raises(rc.RoleCompatError):
        rc.engine_fits_role(ok, "retired_role_b")


def test_empty_required_inputs_fits_all_roles_via_capability_not_legacy():
    """C4 verify #2: an engine declaring required_inputs=() is a valid capability
    that fits EVERY role -- it must NOT fall back to the legacy ``roles`` whitelist.
    Built in an ISOLATED registry so roles=() (which the LEGACY path would exclude
    from every role) proves the override uses capability, not roles."""
    reg = vreg.VideoEngineRegistry("video_c4")
    reg.register(types.SimpleNamespace(
        name="anyfit", roles=(), default_roles=(), required_inputs=(),
        commercial_clean=True, requires_flag=None, family="abstract"))
    for role in _ROLES:
        assert reg.engines_for_role(role) == ["anyfit"]
        assert reg.assert_usable("anyfit", role) == "anyfit"


def test_missing_required_inputs_falls_back_to_legacy_roles():
    """C2/C4: an engine that declares NO required_inputs (None) is the ONLY
    fail-soft case -- it is gated by the legacy ``roles`` whitelist."""
    reg = vreg.VideoEngineRegistry("video_c4")
    reg.register(types.SimpleNamespace(
        name="legacyish", roles=("retired_role_a",), default_roles=(),
        commercial_clean=True, requires_flag=None, family="abstract"))
    # required_inputs attr is ABSENT -> None -> legacy gate
    assert reg.engines_for_role("retired_role_a") == ["legacyish"]
    assert reg.engines_for_role("character_video") == []
    assert reg.assert_usable("legacyish", "retired_role_a") == "legacyish"
    with pytest.raises(vreg.EngineUnusable):
        reg.assert_usable("legacyish", "character_video")
