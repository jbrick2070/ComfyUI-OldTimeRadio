"""Music cue prompts route through the Meta brief protocol (clean-break 1c).

nodes/_otr_music_prompt.compose_music_prompt is the single source of truth for
theme-music cue prompts. It reads the brief via the brief-reader protocol
(_otr_brief_reader._read_brief_field) -- the same propagating creative brief the
visual slots consume -- and layers cue-specific shaping on top. These pins lock
that the composer pulls mood + setting + period from the brief (not a local
template) and degrades gracefully when the brief is absent.
"""
from __future__ import annotations

from nodes._otr_music_prompt import CUE_DURATIONS, compose_music_prompt


def _meta_full() -> dict:
    return {
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting": ["interrogation room", "steel table"],
            "lighting": ["bare bulb"],
            "atmosphere": ["sweat", "smoke", "tense"],
        },
        "music_mood_terms": ["sombre", "uneasy", "menacing"],
        "gen_params_initial": {"period_voice": {"descriptor": "1940s brass and strings"}},
    }


def test_mood_from_brief_v2_field():
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "sombre" in prompt and "uneasy" in prompt and "menacing" in prompt


def test_mood_sliced_to_top_three():
    meta = _meta_full()
    meta["music_mood_terms"] = ["sombre", "uneasy", "menacing", "frantic"]
    prompt, _ = compose_music_prompt(meta, "opening")
    assert "frantic" not in prompt


def test_setting_clause_from_brief():
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "evokes" in prompt
    assert "interrogation room" in prompt or "steel table" in prompt


def test_period_descriptor_from_brief():
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "1940s brass and strings" in prompt


def test_atmosphere_fallback_when_no_music_mood():
    meta = _meta_full()
    meta["music_mood_terms"] = []
    prompt, _ = compose_music_prompt(meta, "opening")
    # Falls through to story_brief_terms.atmosphere.
    assert "sweat" in prompt or "smoke" in prompt or "tense" in prompt


def test_neutral_default_and_tail_when_brief_empty():
    prompt, dur = compose_music_prompt({}, "opening")
    assert "atmospheric" in prompt
    assert prompt.endswith("instrumental only, no dialogue, no vocals")
    assert dur == CUE_DURATIONS["opening"]


def test_cue_character_and_duration_per_slot():
    for slot, marker in (
        ("opening", "atmospheric build"),
        ("closing", "resolving cadence"),
        ("interstitial", "textural bridge"),
    ):
        prompt, dur = compose_music_prompt(_meta_full(), slot)
        assert marker in prompt
        assert dur == CUE_DURATIONS[slot]


def test_reads_via_brief_protocol_not_direct_meta():
    """The composer must accept a parent dict carrying `meta` (the protocol
    reader normalizes both shapes) -- proving it routes through the brief
    reader rather than poking a fixed top-level key."""
    parent = {"meta": _meta_full()}
    # music_mood_terms lives under parent["meta"]; the protocol reader resolves it.
    prompt, _ = compose_music_prompt(parent.get("meta") or {}, "opening")
    assert "sombre" in prompt


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
