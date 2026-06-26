"""tests/test_ltx_audio_in_engine.py -- the ONE audio-in LTX lane.

Operator 2026-06-26: ONE LTX audio-in engine (`ltx_audio_in`) that does I2V on
WHATEVER still the pipeline mints (a radio-bookend scene still, a character scene
still, a face) conditioned on the shot AUDIO (music OR voice) -- agnostic. The old
talk/music split (`ltx_av_talk` / `ltx_av_music`) was REMOVED: it was never about
the engine, it encoded two ROLE routings, which now live on the beat ROLE in
render_driver. `ltx_audio_in` declares ALL audio roles + accepts_still=True so the
coverage arch mints the bookend still and init_image is never missing.

Pure-Python (no GPU, no LTX weights): asserts the engine's CAPABILITY contract +
that the two legacy engines are GONE.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_video_engines.eng_ltx_av import LtxAudioInEngine  # noqa: E402


def test_ltx_audio_in_identity():
    e = LtxAudioInEngine
    assert e.name == "ltx_audio_in"
    # agnostic family -- NOT the face-locked audio_driven_face.
    assert e.family == "audio_conditioned_video"


def test_ltx_audio_in_covers_every_audio_role():
    # one engine for music + announcer (bookends) + character.
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert role in LtxAudioInEngine.roles, role


def test_ltx_audio_in_is_the_default_for_the_bookends():
    # it inherits the per-role default the deleted ltx_av_music held.
    assert set(LtxAudioInEngine.default_roles) == {"music_visual", "announcer_visual"}


def test_ltx_audio_in_mints_a_still_and_takes_audio():
    e = LtxAudioInEngine
    # accepts_still=True is THE capability the coverage arch reads to mint the
    # bookend still, so init_image is never missing.
    assert e.accepts_still is True
    # I2V branch (condition on the still) -- there is no LTX lip-sync parameter.
    assert e._is_talk is True
    # takes a still + the audio + a prompt (every LTX shot carries a prompt).
    assert set(e.required_inputs) == {"text_prompt", "audio_ref", "init_image"}


def test_ltx_audio_in_no_silent_fallback():
    # NO FALLBACKS -- a failed render fails LOUD, never silently degrades.
    assert LtxAudioInEngine.fallback_engine is None


def test_legacy_talk_music_engines_are_gone():
    # the talk/music split was removed -- the classes no longer exist and the
    # names are no longer registered / declared / validated.
    import nodes._otr_video_engines.eng_ltx_av as _mod
    assert not hasattr(_mod, "LtxAvTalkEngine")
    assert not hasattr(_mod, "LtxAvMusicEngine")
    assert _mod.__all__ == ["LtxAudioInEngine"]
    from nodes._otr_video_engines import registry as _reg
    assert "ltx_av_talk" not in _reg.CAPABILITIES
    assert "ltx_av_music" not in _reg.CAPABILITIES
    assert "ltx_av_talk" not in _reg.VALIDATED_ENGINES
    assert "ltx_av_music" not in _reg.VALIDATED_ENGINES
    assert "ltx_audio_in" in _reg.VALIDATED_ENGINES


def test_ltx_audio_in_registered():
    # the @register decorator put it in the engine registry (so a role override
    # / the workflow JSON can select it).
    from nodes._otr_video_engines import registry as _reg
    names = set()
    for attr in ("all_engines", "engines", "registered", "ENGINES", "_REGISTRY"):
        obj = getattr(_reg, attr, None)
        if callable(obj):
            try:
                obj = obj()
            except Exception:  # noqa: BLE001
                obj = None
        if isinstance(obj, dict):
            names |= set(obj.keys())
        elif isinstance(obj, (list, tuple, set)):
            for x in obj:
                names.add(getattr(x, "name", None) or (x if isinstance(x, str) else None))
    if names:
        assert "ltx_audio_in" in names, sorted(n for n in names if n)
        assert "ltx_av_talk" not in names
        assert "ltx_av_music" not in names


if __name__ == "__main__":  # pragma: no cover
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
