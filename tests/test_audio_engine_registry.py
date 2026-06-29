"""Tests for the pluggable audio-engine registry (audio superstructure)."""
import pytest

from nodes._otr_audio_engines import registry as R


class _StubDefaultEngine:
    name = "stub_default"
    roles = ("stub_role",)
    default_roles = ("stub_role",)
    commercial_clean = True
    requires_flag = None

    def load(self):
        pass

    def unload(self):
        pass


class _StubOptInEngine:
    name = "stub_optin"
    roles = ("stub_role",)
    default_roles = ()
    commercial_clean = True
    requires_flag = "OTR_TEST_STUB_FLAG"

    def load(self):
        pass

    def unload(self):
        pass


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(R._REGISTRY)
    try:
        yield
    finally:
        R._REGISTRY.clear()
        R._REGISTRY.update(saved)


def test_register_and_get():
    R.register(_StubDefaultEngine)
    assert R.is_registered("stub_default")
    assert R.get_engine("stub_default").name == "stub_default"


def test_get_unregistered_raises():
    with pytest.raises(KeyError):
        R.get_engine("does_not_exist")


def test_engines_for_role_default_first():
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    order = R.engines_for_role("stub_role")
    assert order[0] == "stub_default", "default-for-role engine must sort first"
    assert set(order) == {"stub_default", "stub_optin"}


def test_default_engine_for_role():
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    assert R.default_engine_for_role("stub_role") == "stub_default"


def test_assert_usable_default_always_runs():
    R.register(_StubDefaultEngine)
    assert R.assert_usable("stub_default", "stub_role") == "stub_default"


def test_assert_usable_optin_selectable_no_flag(monkeypatch):
    # C6 -- registry IS the menu: a registered, role-compatible engine is usable
    # with NO flag gate (the requires_flag field is now vestigial; validation is
    # the operator's MANUAL process). It is never silently swapped for the default.
    monkeypatch.delenv("OTR_TEST_STUB_FLAG", raising=False)
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    assert R.assert_usable("stub_optin", "stub_role") == "stub_optin"


def test_assert_usable_optin_runs_when_flag_on(monkeypatch):
    monkeypatch.setenv("OTR_TEST_STUB_FLAG", "1")
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    assert R.assert_usable("stub_optin", "stub_role") == "stub_optin"


def test_usability_reason_taxonomy_is_exactly_six():
    assert {r.value for r in R.EngineUsabilityReason} == {
        "gated_by_flag", "missing_model", "missing_hf_token",
        "incompatible_profile", "noncommercial_blocked", "malformed_config",
    }


def test_assert_usable_unknown_engine_is_malformed_config():
    with pytest.raises(R.EngineUnusable) as ei:
        R.assert_usable("nope_not_registered", "stub_role")
    assert ei.value.reason is R.EngineUsabilityReason.MALFORMED_CONFIG


def test_assert_usable_role_mismatch_is_incompatible_profile():
    R.register(_StubDefaultEngine)
    with pytest.raises(R.EngineUnusable) as ei:
        R.assert_usable("stub_default", "a_role_it_does_not_serve")
    assert ei.value.reason is R.EngineUsabilityReason.INCOMPATIBLE_PROFILE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
