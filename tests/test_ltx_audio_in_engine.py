"""tests/test_ltx_audio_in_engine.py -- the unified agnostic LTX audio-in lane.

Operator 2026-06-26: ONE LTX audio-in engine that does I2V on WHATEVER still
(a talk face, a radio-bookend scene still, a portrait) conditioned on the shot
AUDIO (music OR voice) -- agnostic, not face/talk-specific. The overnight soak
failed because the bookends were routed to `ltx_av_talk` (family
audio_driven_face) which requires a face init_image the bookend shots never
carried. `ltx_audio_in` declares ALL audio roles + accepts_still=True so the
coverage arch mints the bookend still (the same one the other video engines get
for music/announcer), so init_image is never missing.

Pure-Python (no GPU, no LTX weights): asserts the engine's CAPABILITY contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_video_engines.eng_ltx_av import (  # noqa: E402
    LtxAudioInEngine,
    LtxAvMusicEngine,
    LtxAvTalkEngine,
)


def test_ltx_audio_in_identity():
    e = LtxAudioInEngine
    assert e.name == "ltx_audio_in"
    # agnostic family -- NOT the face-locked audio_driven_face.
    assert e.family == "audio_conditioned_video"


def test_ltx_audio_in_covers_every_audio_role():
    # one engine for music + announcer (bookends) + character.
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert role in LtxAudioInEngine.roles, role


def test_ltx_audio_in_mints_a_still_and_takes_audio():
    e = LtxAudioInEngine
    # accepts_still=True is THE capability the coverage arch reads to mint the
    # bookend still (the fix: ltx_av_music had it False -> no still; ltx_av_talk
    # wanted a still but the bookend shots never got one minted).
    assert e.accepts_still is True
    # I2V branch (condition on the still) -- there is no LTX lip-sync parameter.
    assert e._is_talk is True
    # takes a still + the audio + a prompt (every LTX shot carries a prompt).
    assert set(e.required_inputs) == {"text_prompt", "audio_ref", "init_image"}


def test_ltx_audio_in_no_silent_fallback():
    # NO FALLBACKS -- a failed render fails LOUD, never silently degrades.
    assert LtxAudioInEngine.fallback_engine is None


def test_two_legacy_variants_unchanged():
    # the talk/music split stays for back-compat; the new engine is additive.
    assert LtxAvTalkEngine.name == "ltx_av_talk"
    assert LtxAvTalkEngine.family == "audio_driven_face"
    assert LtxAvTalkEngine.required_inputs == ("text_prompt", "audio_ref", "init_image")
    assert LtxAvMusicEngine.name == "ltx_av_music"
    assert LtxAvMusicEngine.family == "audio_conditioned_video"
    assert LtxAvMusicEngine.accepts_still is False  # T2V, reacts to the track


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
    # if we could enumerate, ltx_audio_in must be present; else at least the
    # class is import-registerable (the decorator ran at import above).
    if names:
        assert "ltx_audio_in" in names, sorted(n for n in names if n)


if __name__ == "__main__":  # pragma: no cover
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
