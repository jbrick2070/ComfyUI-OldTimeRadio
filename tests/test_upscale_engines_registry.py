"""Queue item 8 (2026-08-08): the upscale-engine registry contract.

Mirrors ``tests/test_video_engines_registry.py`` style: Protocol conformance,
CAPABILITIES vs registered-engines roster (audit_engine_roster returns
empty missing + empty unexpected), retirement policy, and the
EngineUnusable(role, kind) shape.
"""
from __future__ import annotations

import pytest


def test_registry_module_imports_clean():
    from nodes._otr_upscale_engines import registry as r
    assert r is not None


def test_two_engines_ship_and_only_two():
    from nodes._otr_upscale_engines.registry import all_engine_names
    names = list(all_engine_names())
    assert set(names) == {"off", "spandrel_esrgan"}, (
        f"expected exactly {{'off', 'spandrel_esrgan'}}; got {names!r}")


def test_capabilities_roster_matches_registered():
    from nodes._otr_upscale_engines.registry import audit_engine_roster
    audit = audit_engine_roster()
    assert audit == {"missing": (), "unexpected": ()}, (
        f"roster drift: {audit!r} -- every CAPABILITIES row must have a "
        f"registered engine and vice-versa")


def test_retired_ids_is_empty_at_ship():
    from nodes._otr_upscale_engines.registry import RETIRED_UPSCALE_ENGINE_IDS
    assert RETIRED_UPSCALE_ENGINE_IDS == frozenset()


def test_engine_unusable_accepts_kind_kwarg():
    from nodes._otr_shared.engine_registry_base import (
        EngineUnusable, EngineUsabilityReason)
    exc = EngineUnusable("some_engine", "upscale_stage",
                          EngineUsabilityReason.MALFORMED_CONFIG,
                          "test", kind="upscale")
    assert getattr(exc, "kind", None) == "upscale"


def test_assert_usable_returns_off():
    from nodes._otr_upscale_engines.registry import assert_usable
    assert assert_usable("off", "upscale_stage") == "off"


def test_assert_usable_returns_spandrel():
    from nodes._otr_upscale_engines.registry import assert_usable
    assert assert_usable("spandrel_esrgan", "upscale_stage") == "spandrel_esrgan"


def test_assert_usable_raises_on_unknown():
    from nodes._otr_upscale_engines.registry import assert_usable
    from nodes._otr_shared.engine_registry_base import EngineUnusable
    with pytest.raises(EngineUnusable):
        assert_usable("does_not_exist", "upscale_stage")


def test_off_engine_shape():
    from nodes._otr_upscale_engines.registry import get_engine
    off = get_engine("off")
    assert off.name == "off"
    assert off.roles == ("upscale_stage",)
    assert off.default_roles == ("upscale_stage",)
    assert off.commercial_clean is True
    assert off.requires_flag is None
    assert set(off.device_backends).issubset({"cuda", "cpu", "mps"})
    assert off.requires_vendor is None
    assert off.intrinsic_scale == 1


def test_spandrel_engine_shape():
    from nodes._otr_upscale_engines.registry import get_engine
    sp = get_engine("spandrel_esrgan")
    assert sp.name == "spandrel_esrgan"
    assert sp.roles == ("upscale_stage",)
    assert sp.default_roles == ()
    assert sp.commercial_clean is True    # Real-ESRGAN x2plus BSD-3-Clause
    assert sp.requires_flag is None
    # First ship: NO MPS -- documented deferral until a Mac receipt.
    assert set(sp.device_backends) == {"cuda", "cpu"}
    assert "mps" not in sp.device_backends
    assert sp.requires_vendor is None
    assert sp.intrinsic_scale == 2


def test_capabilities_row_shape_matches_decl_keys():
    """Every CAPABILITIES row must pass validate_declaration
    (nodes/_otr_shared/capability_profiles.py:validate_declaration)."""
    from nodes._otr_upscale_engines.registry import CAPABILITIES
    from nodes._otr_shared.capability_profiles import validate_declaration
    for name, decl in CAPABILITIES.items():
        validate_declaration(name, decl, source=f"upscale/{name}")


def test_off_default_sort_first_in_engines_for_role():
    """OffUpscale declares default_roles=('upscale_stage',) so it sorts first
    in the dropdown -- that IS what makes it the byte-identity default."""
    from nodes._otr_upscale_engines.registry import engines_for_role
    seq = engines_for_role("upscale_stage")
    assert seq[0] == "off", (
        f"off must sort first in the upscale_stage dropdown; got {seq!r}")
