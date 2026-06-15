"""BUG-LOCAL-408: SA3 music prompt + conditioning helpers (pure, no GPU).

SA3 replaced MusicGen as the default music engine (2026-06-03) but sounded
non-musical because the MusicGen-shaped abstract-mood prompts and the
``seconds_total == dur`` conditioning gave it no genre/instrument anchor and no
structural context. These helpers add an SA3-shaped genre anchor + a per-cue
structural window. They are pure (no ComfyUI runtime) so they unit-test here.
"""
from nodes._otr_audio_engines.eng_stable_audio_3 import (
    _sa3_augment_prompt, _sa3_clip_window, _SA3_DEFAULT_GENRE,
)


def test_sa3_augment_prepends_genre_keeps_prompt_bug408():
    base = ("minor mode, unresolved tension, evokes derelict station, slow "
            "atmospheric build, instrumental only, no dialogue, no vocals")
    out = _sa3_augment_prompt(base)
    assert out.endswith(base)                 # original brief text preserved
    assert out != base                        # a genre anchor was prepended
    # era-aware: a 1950s cue gets the vintage sci-fi anchor (theremin etc.)
    assert "theremin" in _sa3_augment_prompt("1950s atomic-age, instrumental intro")
    # no era keyword -> the default cinematic anchor leads
    assert _sa3_augment_prompt("just vibes").startswith(_SA3_DEFAULT_GENRE)
    # empty / None safe
    assert _sa3_augment_prompt("") == _SA3_DEFAULT_GENRE
    assert _sa3_augment_prompt(None) == _SA3_DEFAULT_GENRE


def test_sa3_clip_window_places_cue_in_context_bug408():
    ctx = 30.0
    # opening (intro) -> head/build
    s, t = _sa3_clip_window("slow build, instrumental intro", 12.0, ctx)
    assert s == 0.0 and t == 30.0
    # closing (outro) -> tail/resolving
    s, t = _sa3_clip_window("gentle decay, instrumental outro", 8.0, ctx)
    assert abs(s - 22.0) < 1e-6 and t == 30.0
    # interstitial (neither) -> middle/unresolved bridge
    s, t = _sa3_clip_window("short instrumental transition", 4.0, ctx)
    assert abs(s - 13.0) < 1e-6 and t == 30.0
    # context is never shorter than the clip (a long clip widens the context)
    s, t = _sa3_clip_window("instrumental intro", 40.0, 30.0)
    assert t == 40.0 and s == 0.0
