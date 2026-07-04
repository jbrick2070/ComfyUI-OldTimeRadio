"""Additive CPU coverage for the reusable engine-registry base (A-Seam AS-4).

New, self-contained tests for nodes/_otr_shared/engine_registry_base.py built on
a LOCAL EngineRegistry instance with stub adapters (no global registry mutation,
no GPU, no model load). Pins the fail-closed contract: register-by-class or
-instance, default-first role ordering, and assert_usable returning the
validated name or raising EngineUnusable with the right classified reason
(MALFORMED_CONFIG / INCOMPATIBLE_PROFILE / GATED_BY_FLAG) -- never a silent swap.
Pure stdlib + the module under test. UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import engine_registry_base as base


class _StubEngine:
    """A minimal EngineCore-shaped adapter for registry tests (no residency)."""

    def __init__(self, name, roles, default_roles=(), requires_flag=None,
                 commercial_clean=True):
        self.name = name
        self.roles = tuple(roles)
        self.default_roles = tuple(default_roles)
        self.requires_flag = requires_flag
        self.commercial_clean = commercial_clean

    def load(self):  # cheap no-op; residency is never exercised here
        pass

    def unload(self):
        pass


def _reg():
    r = base.EngineRegistry(kind="video")
    r.register(_StubEngine(
        "floor", roles=("retired_role_a", "retired_role_b"),
        default_roles=("retired_role_b",)))
    r.register(_StubEngine(
        "fancy", roles=("retired_role_a",), requires_flag="OTR_ENABLE_FANCY"))
    return r


def test_register_returns_adapter_and_records_instance():
    r = base.EngineRegistry(kind="video")
    eng = _StubEngine("e1", roles=("retired_role_a",))
    assert r.register(eng) is eng
    assert r.is_registered("e1")
    assert r.get_engine("e1") is eng


def test_register_accepts_a_class_and_instantiates_it():
    r = base.EngineRegistry(kind="video")

    class _Cls:
        name = "cls_eng"
        roles = ("retired_role_a",)
        default_roles = ()
        requires_flag = None
        commercial_clean = True

        def load(self):
            pass

        def unload(self):
            pass

    assert r.register(_Cls) is _Cls               # returns the class (decorator)
    assert r.is_registered("cls_eng")
    assert isinstance(r.get_engine("cls_eng"), _Cls)  # an instance was stored


def test_get_engine_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        base.EngineRegistry(kind="video").get_engine("nope")


def test_all_engine_names_sorted():
    r = base.EngineRegistry(kind="video")
    r.register(_StubEngine("zeta", roles=("retired_role_a",)))
    r.register(_StubEngine("alpha", roles=("retired_role_a",)))
    assert r.all_engine_names() == ["alpha", "zeta"]


def test_engines_for_role_sorts_by_name_when_no_default():
    r = _reg()
    # both serve retired_role_a; neither is default there -> sorted by name
    assert r.engines_for_role("retired_role_a") == ["fancy", "floor"]
    # retired_role_b: only floor serves it
    assert r.engines_for_role("retired_role_b") == ["floor"]


def test_engines_for_role_puts_default_engine_ahead():
    r = base.EngineRegistry(kind="video")
    r.register(_StubEngine(
        "bbb", roles=("retired_role_a",), default_roles=("retired_role_a",)))
    r.register(_StubEngine("aaa", roles=("retired_role_a",)))
    # bbb is the default for the role -> sorts first despite the later name
    assert r.engines_for_role("retired_role_a") == ["bbb", "aaa"]


def test_default_engine_for_role():
    r = _reg()
    assert r.default_engine_for_role("retired_role_b") == "floor"
    assert r.default_engine_for_role("retired_role_a") is None


def test_assert_usable_default_engine_ok():
    assert _reg().assert_usable("floor", "retired_role_b") == "floor"


def test_assert_usable_unregistered_is_malformed_config():
    with pytest.raises(base.EngineUnusable) as exc:
        _reg().assert_usable("ghost", "retired_role_a")
    assert exc.value.reason is base.EngineUsabilityReason.MALFORMED_CONFIG


def test_assert_usable_wrong_role_is_incompatible_profile():
    with pytest.raises(base.EngineUnusable) as exc:
        _reg().assert_usable("floor", "announcer_visual")
    assert exc.value.reason is base.EngineUsabilityReason.INCOMPATIBLE_PROFILE


def test_assert_usable_optin_without_flag_is_selectable(monkeypatch):
    # Registry IS the menu: a registered, role-compatible engine is usable with
    # NO flag gate -- validation is the operator's manual process, never a code
    # gate. The (vestigial) requires_flag no longer gates selectability.
    monkeypatch.delenv("OTR_ENABLE_FANCY", raising=False)
    assert _reg().assert_usable("fancy", "retired_role_a") == "fancy"


def test_assert_usable_optin_with_flag_set_ok(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_FANCY", "1")
    assert _reg().assert_usable("fancy", "retired_role_a") == "fancy"


def test_engine_unusable_carries_classified_fields():
    err = base.EngineUnusable(
        "humo", "music_visual", base.EngineUsabilityReason.GATED_BY_FLAG,
        "set OTR_ENABLE_HUMO=1", kind="video")
    assert err.engine == "humo"
    assert err.role == "music_visual"
    assert err.reason is base.EngineUsabilityReason.GATED_BY_FLAG
    assert err.kind == "video"
    assert "video engine 'humo'" in str(err)
    assert "gated_by_flag" in str(err)


def test_usability_reason_has_the_six_codes():
    assert {r.value for r in base.EngineUsabilityReason} == {
        "gated_by_flag",
        "missing_model",
        "missing_hf_token",
        "incompatible_profile",
        "noncommercial_blocked",
        "malformed_config",
    }
