"""Additive CPU coverage for the pluggable video-engine registry (AS-4).

New, self-contained READ-ONLY assertions on the real video registry after the
core adapters self-register on import (humo / humo_1.7B / the cheap radio-floor
families). It never clears or mutates the global registry. Pins: the registry is
the shared AS-4 base instance, names are sorted + unique, engines_for_role is
self-consistent (every returned engine lists the role), and assert_usable /
get_engine fail closed on an unknown engine. Complements
test_video_platform_aseam.py. No GPU, no model load. UTF-8, no BOM, ASCII, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import engine_registry_base as base
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import registry as vreg
# Importing the adapters self-registers the core engines (the same import the
# retry-taxonomy test relies on); read-only -- we never clear/modify the global.
from nodes._otr_video_engines import eng_humo            # noqa: F401
from nodes._otr_video_engines import cheap_families      # noqa: F401


CORE_ENGINES = ("humo", "humo_1.7B", "still_kenburns")


def test_registry_is_the_shared_video_base_instance():
    assert isinstance(vreg._VIDEO_REGISTRY, base.EngineRegistry)
    assert vreg._VIDEO_REGISTRY.kind == "video"
    # the module functions bind to that one instance (AS-4 1:1 surface)
    assert vreg.is_registered.__self__ is vreg._VIDEO_REGISTRY
    assert vreg.assert_usable.__self__ is vreg._VIDEO_REGISTRY


def test_core_engines_self_registered_on_import():
    for name in CORE_ENGINES:
        assert vreg.is_registered(name), name
        assert vreg.get_engine(name).name == name


def test_all_engine_names_sorted_and_unique():
    names = vreg.all_engine_names()
    assert names == sorted(names)
    assert len(names) == len(set(names))
    for name in CORE_ENGINES:
        assert name in names


def test_engines_for_role_is_self_consistent_and_a_subset():
    all_names = set(vreg.all_engine_names())
    for role in rc.ROLES:
        listed = vreg.engines_for_role(role)
        assert set(listed) <= all_names
        for name in listed:                            # each actually serves it
            assert role in tuple(vreg.get_engine(name).roles)


def test_default_engine_for_role_is_registered_and_lists_role():
    for role in rc.ROLES:
        d = vreg.default_engine_for_role(role)
        if d is not None:
            assert vreg.is_registered(d)
            assert role in tuple(vreg.get_engine(d).default_roles)


def test_get_engine_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        vreg.get_engine("definitely_not_a_real_engine")


def test_assert_usable_unknown_engine_fails_closed():
    with pytest.raises(vreg.EngineUnusable) as exc:
        vreg.assert_usable(
            "definitely_not_a_real_engine", rc.Role.BACKGROUND_ABSTRACT.value)
    assert exc.value.reason is base.EngineUsabilityReason.MALFORMED_CONFIG
