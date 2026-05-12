"""tests/test_musicgen_news_brief_used.py

P1 guardrail — when ``meta.news.script_brief`` contains a mood
keyword from ``_MOOD_TAGS``, the resulting MusicGen prompt for that
episode includes the corresponding mood string.

This is the proof that MusicGen actually consumes the L3 ledger's
news brief (and not only the style slug). It exercises the public
``_resolve_cue_from_style`` + ``_mood_suffix`` contract through the
same call path the node uses at render time.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def musicgen_mod():
    from nodes import musicgen_theme as mod
    return mod


def test_brief_with_keyword_lands_mood_tag_in_prompt(musicgen_mod):
    """Brief containing a mood keyword surfaces the matching tag."""
    keyword, tag = next(iter(musicgen_mod._MOOD_TAGS.items()))
    brief = f"The crew faces {keyword} aboard the orbital station."
    suffix = musicgen_mod._mood_suffix(brief)
    prompt, _ = musicgen_mod._resolve_cue_from_style(
        "opening", "deep_space_distress_call", suffix
    )
    assert tag in prompt, (
        f"expected mood tag {tag!r} from keyword {keyword!r} "
        f"to surface in prompt; got {prompt!r}"
    )


def test_brief_without_keyword_produces_no_mood_tag(musicgen_mod):
    """Empty / keyword-less brief produces no mood overlay."""
    bland = "An episode about routine maintenance schedules and paperwork."
    # Sanity: none of the mood keywords appear in the bland brief.
    bland_low = bland.lower()
    for kw in musicgen_mod._MOOD_TAGS:
        assert kw not in bland_low, (
            f"test fixture broken: mood keyword {kw!r} present in "
            f"supposedly-bland brief"
        )
    suffix = musicgen_mod._mood_suffix(bland)
    assert suffix == "", f"expected empty suffix, got {suffix!r}"
    # And the prompt should not contain any mood-tag text.
    prompt, _ = musicgen_mod._resolve_cue_from_style(
        "opening", "closed_room_suspense", suffix
    )
    for tag in musicgen_mod._MOOD_TAGS.values():
        assert tag not in prompt, (
            f"mood tag {tag!r} unexpectedly surfaced in prompt "
            f"derived from keyword-free brief; got {prompt!r}"
        )


def test_brief_with_multiple_keywords_lands_all_matching_tags(musicgen_mod):
    """Multiple keywords in the brief surface all matching tags."""
    keys = list(musicgen_mod._MOOD_TAGS.keys())
    if len(keys) < 2:
        pytest.skip("_MOOD_TAGS has fewer than 2 entries")
    k1, k2 = keys[0], keys[1]
    t1 = musicgen_mod._MOOD_TAGS[k1]
    t2 = musicgen_mod._MOOD_TAGS[k2]
    brief = f"The story explores {k1} and {k2} in equal measure."
    suffix = musicgen_mod._mood_suffix(brief)
    assert t1 in suffix
    assert t2 in suffix


def test_brief_match_is_case_insensitive(musicgen_mod):
    """Keyword matching ignores brief case."""
    keyword, tag = next(iter(musicgen_mod._MOOD_TAGS.items()))
    brief_upper = f"A STORY ABOUT {keyword.upper()} ABOARD."
    brief_mixed = f"A Story About {keyword.title()} aboard."
    for brief in (brief_upper, brief_mixed):
        suffix = musicgen_mod._mood_suffix(brief)
        assert tag in suffix, (
            f"case-insensitive match failed for brief {brief!r}; "
            f"got suffix {suffix!r}"
        )
