"""tests/test_voice_backends.py

Phase 0+ Voice Backend Abstraction §1+§2: registry shape + protocol
conformance + OTR_VoiceRender dispatch.

These tests verify the abstraction contract WITHOUT touching torch or
any real TTS engine. The bundled bark/kokoro drivers are stubs; tests
that exercise dispatch end-to-end use a fake backend.
"""
from __future__ import annotations

import pytest

from nodes._voice_backends import (
    KNOWN_ENGINES,
    VoiceBackend,
    _register_default_drivers,
    available_engines,
    get_factory,
    register,
    unregister,
)
from nodes._voice_backends._protocol import VoiceBackend as ProtocolVoiceBackend


# ---------- registry shape ----------------------------------------------


# `test_known_engines_mirrors_resolver_set` was REMOVED 2026-08-28 with the
# module it compared against. It asserted that the registry's KNOWN_ENGINES
# mirrored a SECOND copy of the same set in `_otr_voice_resolver`, which was a
# parser-only module with no production consumer. Deleting the duplicate is
# what removed the need to check that two copies agree -- the registry's set is
# now the only one, so there is nothing left for it to drift from.


def test_protocol_is_runtime_checkable():
    """A duck-typed object that has the right methods + engine_name
    should pass isinstance(obj, VoiceBackend)."""

    class FakeBackend:
        engine_name = "fake"

        def load(self, preset: str) -> None:  # noqa: ARG002
            return None

        def generate(self, text: str, **kw):  # noqa: ARG002
            return b""

        def unload(self) -> None:
            return None

    assert isinstance(FakeBackend(), ProtocolVoiceBackend)


def test_register_and_get_factory_round_trip():
    class FakeBackend:
        engine_name = "regtest"

        def load(self, preset): return None  # noqa: ARG002, E704
        def generate(self, text, **kw): return b"fake"  # noqa: ARG002, E704
        def unload(self): return None  # noqa: E704

    register("regtest", FakeBackend)
    try:
        factory = get_factory("regtest")
        instance = factory()
        assert instance.engine_name == "regtest"
        assert "regtest" in available_engines()
    finally:
        unregister("regtest")


def test_register_lowercases_and_strips_engine_name():
    class FakeBackend:
        engine_name = "casetest"

        def load(self, preset): return None  # noqa: ARG002, E704
        def generate(self, text, **kw): return b""  # noqa: ARG002, E704
        def unload(self): return None  # noqa: E704

    register("  CaseTest  ", FakeBackend)
    try:
        # Lookup is also normalized.
        factory = get_factory("casetest")
        assert factory().engine_name == "casetest"
    finally:
        unregister("casetest")


def test_register_rejects_blank_engine_name():
    with pytest.raises(ValueError, match="non-empty"):
        register("   ", lambda: None)
    with pytest.raises(ValueError, match="non-empty"):
        register("", lambda: None)


def test_get_factory_missing_engine_lists_registered_in_error():
    with pytest.raises(KeyError, match="not registered"):
        get_factory("definitely_not_an_engine")


def test_unregister_is_idempotent():
    # Should not raise even if engine never registered.
    unregister("never_existed")


# ---------- bundled-driver self-registration ----------------------------


def test_default_drivers_self_register_when_loaded():
    """After calling _register_default_drivers, bark + kokoro must be
    in the registry."""
    _register_default_drivers()
    engines = available_engines()
    assert "bark" in engines
    assert "kokoro" in engines


def _simulate_fresh_process_voice_state():
    """Helper: reset the voice-backends package to the state it would
    have on a fresh process import. Clears the registry, the
    once-per-process flag, flushes the bundled-driver modules from
    `sys.modules`, AND removes the package's cached submodule
    attributes so `from nodes._voice_backends import bark` actually
    re-executes the bark module body and re-fires its
    module-scope `register(...)` call. Returns a `restore` callable.
    """
    import sys
    import nodes._voice_backends as voice_pkg

    saved_flag = voice_pkg._DEFAULTS_REGISTERED
    saved_registry = dict(voice_pkg._REGISTRY)
    saved_modules: dict[str, object] = {}
    saved_attrs: dict[str, object] = {}

    for short_name, mod_name in (
        ("bark", "nodes._voice_backends.bark"),
        ("kokoro", "nodes._voice_backends.kokoro"),
    ):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)
        # Python caches submodules as attributes on their package;
        # `from nodes._voice_backends import bark` looks at this
        # attribute first and short-circuits the module body. Pop it.
        if hasattr(voice_pkg, short_name):
            saved_attrs[short_name] = getattr(voice_pkg, short_name)
            delattr(voice_pkg, short_name)

    voice_pkg._DEFAULTS_REGISTERED = False
    voice_pkg._REGISTRY.clear()

    def _restore():
        voice_pkg._REGISTRY.clear()
        voice_pkg._REGISTRY.update(saved_registry)
        voice_pkg._DEFAULTS_REGISTERED = saved_flag
        for k, v in saved_modules.items():
            sys.modules[k] = v
        for short_name, value in saved_attrs.items():
            setattr(voice_pkg, short_name, value)

    return _restore


def test_get_factory_lazy_initializes_default_drivers():
    """Round-robin Element 4 catch (2026-05-08): a caller that uses
    just `from nodes._voice_backends import get_factory` on a fresh
    process MUST still get the bundled drivers because `get_factory`
    lazy-fires `_register_default_drivers()` on first call.
    """
    import nodes._voice_backends as voice_pkg

    restore = _simulate_fresh_process_voice_state()
    try:
        # Sanity: registry empty before the call.
        assert voice_pkg._REGISTRY == {}
        # Single get_factory call should re-register the defaults.
        factory = voice_pkg.get_factory("bark")
        assert factory is not None
        assert "bark" in voice_pkg._REGISTRY
        assert "kokoro" in voice_pkg._REGISTRY
    finally:
        restore()


def test_available_engines_lazy_initializes_default_drivers():
    """Same as get_factory: available_engines() must fire the lazy
    init so callers using only the package surface see the stock
    list."""
    import nodes._voice_backends as voice_pkg

    restore = _simulate_fresh_process_voice_state()
    try:
        assert voice_pkg._REGISTRY == {}
        engines = voice_pkg.available_engines()
        assert "bark" in engines
        assert "kokoro" in engines
    finally:
        restore()


def test_bark_stub_raises_not_migrated_on_load():
    from nodes._voice_backends.bark import BarkBackend, BarkBackendNotMigrated

    backend = BarkBackend()
    with pytest.raises(BarkBackendNotMigrated, match="stub"):
        backend.load("v2/en_speaker_3")


def test_bark_stub_raises_not_migrated_on_generate():
    from nodes._voice_backends.bark import BarkBackend, BarkBackendNotMigrated

    backend = BarkBackend()
    with pytest.raises(BarkBackendNotMigrated):
        backend.generate("hello world")


def test_bark_stub_unload_is_noop():
    """unload must NOT raise even on a stub -- caller's defensive
    try/finally cleanup must be safe."""
    from nodes._voice_backends.bark import BarkBackend

    backend = BarkBackend()
    backend.unload()  # no exception
    assert backend._loaded_preset == ""


def test_kokoro_stub_raises_not_migrated():
    from nodes._voice_backends.kokoro import (
        KokoroBackend,
        KokoroBackendNotMigrated,
    )

    backend = KokoroBackend()
    with pytest.raises(KokoroBackendNotMigrated):
        backend.load("bm_fable")
    with pytest.raises(KokoroBackendNotMigrated):
        backend.generate("hello")
    backend.unload()  # no exception


# Voice-path-cleanbreak 2026-05-12 (P3): nodes.voice_render
# (OTR_VoiceRender) was deleted along with the other legacy single-line
# nodes (OTR_BarkTTS, OTR_SFXGenerator). Five dispatch tests that
# exercised the node class are gone in lockstep. The voice backend
# protocol + registry + Bark/Kokoro driver tests above remain in
# place -- the registry is still load-bearing for the cast-locked
# voice path even without the legacy single-line node wrapper.
