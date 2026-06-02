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


def test_assert_usable_optin_resolves_to_default_when_flag_off(monkeypatch):
    monkeypatch.delenv("OTR_TEST_STUB_FLAG", raising=False)
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    assert R.assert_usable("stub_optin", "stub_role") == "stub_default"


def test_assert_usable_optin_runs_when_flag_on(monkeypatch):
    monkeypatch.setenv("OTR_TEST_STUB_FLAG", "1")
    R.register(_StubOptInEngine)
    R.register(_StubDefaultEngine)
    assert R.assert_usable("stub_optin", "stub_role") == "stub_optin"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
