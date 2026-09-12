"""BUG-LOCAL-408: SA3 conditioning window (pure, no GPU).

SA3 replaced MusicGen as the default music engine (2026-06-03) and sounded
non-musical because ``seconds_total == dur`` gave it no structural context.
``_sa3_clip_window`` places each cue inside a longer conditioning window: an
opening at the head, a closing at the tail, an interstitial in the middle.

The OTHER half of the 408 fix -- an SA3-only era/genre anchor prepended to
the prompt, with "analog tape warmth" in every branch -- was retired on
2026-09-11: the operator withdrew the radio-hiss texture ("make them more
musical") and the instruments now come from the story palette through the
shared composer (`_otr_music_prompt.compose_engine_prompt`), for EVERY engine.
`tests/test_music_prompts_are_musical.py` pins that the engine no longer
prepends anything.
"""
from nodes._otr_audio_engines.eng_stable_audio_3 import _sa3_clip_window


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
    # "opening"/"closing" cue words also map (belt-and-braces alongside intro/outro)
    # -- baked default context is 12s (= the longest cue, a tight phrase)
    s, t = _sa3_clip_window("opening theme, slow build", 12.0, 12.0)
    assert s == 0.0 and t == 12.0
    s, t = _sa3_clip_window("closing theme, gentle decay", 8.0, 12.0)
    assert abs(s - 4.0) < 1e-6 and t == 12.0


def test_sa3_clip_window_is_driven_by_the_placement_name_first():
    """The theme node hands the cue's placement over (2026-09-11); a bare
    placement word resolves the window without any prompt text at all."""
    assert _sa3_clip_window("opening", 12.0, 12.0) == (0.0, 12.0)
    s, t = _sa3_clip_window("closing", 8.0, 12.0)
    assert abs(s - 4.0) < 1e-6 and t == 12.0
    s, t = _sa3_clip_window("interstitial", 4.0, 12.0)
    assert abs(s - 4.0) < 1e-6 and t == 12.0
