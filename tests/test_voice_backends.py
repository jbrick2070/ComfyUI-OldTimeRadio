"""tests/test_voice_backends.py

Phase 0+ Voice Backend Abstraction §1+§2: registry shape + protocol
conformance + OTR_VoiceRender dispatch.

These tests verify the abstraction contract WITHOUT touching torch or
any real TTS engine. The bundled bark/kokoro drivers are stubs; tests
that exercise dispatch end-to-end use a fake backend.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_resolver import KNOWN_ENGINES as RESOLVER_KNOWN_ENGINES
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


def test_known_engines_mirrors_resolver_set():
    """The registry's KNOWN_ENGINES set should mirror the resolver's
    so callers don't have to import both modules."""
    assert KNOWN_ENGINES == RESOLVER_KNOWN_ENGINES


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


# ---------- OTR_VoiceRender dispatch ------------------------------------


def test_voice_render_node_class_attributes():
    """The node class itself has the right ComfyUI metadata even
    while not registered in __init__.py."""
    from nodes.voice_render import OTR_VoiceRender

    assert OTR_VoiceRender.CATEGORY == "OTR/Voice"
    assert OTR_VoiceRender.RETURN_TYPES == ("AUDIO",)
    assert OTR_VoiceRender.RETURN_NAMES == ("audio",)
    assert OTR_VoiceRender.FUNCTION == "render"

    inputs = OTR_VoiceRender.INPUT_TYPES()
    assert "voice_model" in inputs["required"]
    assert "voice_preset" in inputs["required"]
    assert "text" in inputs["required"]
    # Voice model dropdown lists the known engines alphabetically.
    voice_model_def = inputs["required"]["voice_model"]
    enum_list = voice_model_def[0]
    assert "bark" in enum_list
    assert "kokoro" in enum_list
    assert enum_list == sorted(enum_list)


def test_voice_render_dispatches_to_registered_backend():
    """End-to-end dispatch through a fake backend, no real TTS."""
    from nodes.voice_render import OTR_VoiceRender

    calls: dict[str, list] = {"load": [], "generate": [], "unload": []}

    class FakeBackend:
        engine_name = "dispatchtest"

        def load(self, preset):
            calls["load"].append(preset)

        def generate(self, text, **kw):
            calls["generate"].append((text, kw))
            return b"FAKE_AUDIO"

        def unload(self):
            calls["unload"].append(True)

    register("dispatchtest", FakeBackend)
    try:
        node = OTR_VoiceRender()
        out = node.render(
            text="hello there",
            voice_model="dispatchtest",
            voice_preset="v2/en_speaker_3",
            temperature=0.5,
        )
        assert out == (b"FAKE_AUDIO",)
        assert calls["load"] == ["v2/en_speaker_3"]
        assert calls["generate"][0][0] == "hello there"
        assert calls["generate"][0][1]["temperature"] == 0.5
        assert calls["unload"] == [True]
    finally:
        unregister("dispatchtest")


def test_voice_render_unknown_engine_raises_runtime_error():
    from nodes.voice_render import OTR_VoiceRender

    node = OTR_VoiceRender()
    with pytest.raises(RuntimeError, match="not registered"):
        node.render(
            text="hi",
            voice_model="not_a_real_engine",
            voice_preset="something",
        )


def test_voice_render_malformed_spec_raises_value_error():
    """Empty preset -> parse_voice_spec rejects."""
    from nodes.voice_render import OTR_VoiceRender

    node = OTR_VoiceRender()
    with pytest.raises(ValueError):
        node.render(
            text="hi",
            voice_model="bark",
            voice_preset="",
        )


def test_voice_render_unloads_even_if_generate_raises():
    """try/finally contract: unload must run on the backend even if
    generate explodes."""
    from nodes.voice_render import OTR_VoiceRender

    unload_called = {"n": 0}

    class ExplodingBackend:
        engine_name = "explody"

        def load(self, preset): return None  # noqa: ARG002, E704

        def generate(self, text, **kw):  # noqa: ARG002
            raise RuntimeError("boom")

        def unload(self):
            unload_called["n"] += 1

    register("explody", ExplodingBackend)
    try:
        node = OTR_VoiceRender()
        with pytest.raises(RuntimeError, match="boom"):
            node.render(
                text="trigger",
                voice_model="explody",
                voice_preset="anything",
            )
        assert unload_called["n"] == 1
    finally:
        unregister("explody")
