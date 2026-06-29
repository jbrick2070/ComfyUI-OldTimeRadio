"""Guard: the registry IS the menu (C7 -- locks the C2-C6 delete-code-gating work).

The operator directive (2026-06-29): NO opt-in / validation / promotion gate
anywhere (video, image, voice, LLM). A REGISTERED engine is SELECTABLE and renders
(it may hard-fail LOUD on missing deps); validation is the operator's MANUAL
process, never a code gate. This test fails LOUD if any of those gates ever creep
back:

* no REGISTERED video / image / audio engine declares a live ``requires_flag``
  (the field is kept ONLY as a vestigial ``None`` for audio protocol-parity);
* the dark NotImplementedError scaffolds stay UNREGISTERED (not selectable);
* no REGISTERED video engine's render path is a dark NotImplementedError stub;
* the OpenRouter LLM lane is gated ONLY on its credentials, never on an opt-in flag.
"""
from __future__ import annotations


def test_no_registered_video_engine_declares_a_flag():
    from nodes._otr_video_engines import registry as vreg
    for name in vreg.all_engine_names():
        eng = vreg.get_engine(name)
        assert getattr(eng, "requires_flag", None) is None, (
            f"video engine {name!r} declares requires_flag={eng.requires_flag!r}"
            " -- registry IS the menu, no flag gate (C2/C3)"
        )


def test_no_registered_image_engine_declares_a_flag():
    from nodes._otr_image_engines import registry as ireg
    for name in ireg.all_engine_names():
        eng = ireg.get_engine(name)
        assert getattr(eng, "requires_flag", None) is None, (
            f"image engine {name!r} declares requires_flag={eng.requires_flag!r}"
            " -- registry IS the menu, no flag gate (C2)"
        )


def test_no_registered_audio_engine_declares_a_flag():
    from nodes import _otr_audio_engines as AE
    names = set()
    for role in ("char_voice", "announcer_voice", "music"):
        names.update(AE.engines_for_role(role))
    assert names, "no audio engines enumerated"
    for name in names:
        eng = AE.get_engine(name)
        assert getattr(eng, "requires_flag", None) is None, (
            f"audio engine {name!r} declares requires_flag={eng.requires_flag!r}"
            " -- registry IS the menu, no flag gate (C6); the field stays a"
            " vestigial None for protocol-parity"
        )


def test_dark_notimplemented_scaffolds_are_unregistered():
    # The dark scaffolds whose render path raises NotImplementedError were
    # UNREGISTERED (C3) -- they are NOT selectable. (The source files stay on disk
    # and return WITH their @register + a CAPABILITIES row when a real forward ships.)
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_video_engines import registry as vreg
    for name in ("triposr", "triposg_talk", "hunyuan3d_talk", "trellis_talk"):
        assert not vreg.is_registered(name), (
            f"video scaffold {name!r} is registered but renders NotImplementedError"
        )
    for name in ("hidream_i1", "sd35_large"):
        assert not ireg.is_registered(name), (
            f"image scaffold {name!r} is registered but renders NotImplementedError"
        )


def test_no_registered_video_render_is_a_dark_scaffold_stub():
    # Defence in depth: a registered engine's render path must not be a pure
    # NotImplementedError "dark scaffold" stub (the C3 signature). The scaffolds'
    # render_clip raises with a "dark scaffold" message -- assert no REGISTERED
    # engine's render_clip carries that marker in its docstring.
    from nodes._otr_video_engines import registry as vreg
    for name in vreg.all_engine_names():
        eng = vreg.get_engine(name)
        doc = (getattr(getattr(eng, "render_clip", None), "__doc__", "") or "").lower()
        assert "dark scaffold" not in doc, (
            f"video engine {name!r} render_clip looks like a dark NotImplementedError"
            " scaffold but is REGISTERED -- unregister it (C3)"
        )


def test_openrouter_llm_gated_only_on_credentials(monkeypatch):
    # LLM (C6): OpenRouter is selectable whenever its creds (API key) are present;
    # the OTR_ENABLE_OPENROUTER opt-in flag no longer gates.
    from nodes import _otr_openrouter_backend as orb
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-guard")
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    assert orb.openrouter_enabled() is True            # key present, no flag needed
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert orb.openrouter_enabled() is False           # no creds -> off
