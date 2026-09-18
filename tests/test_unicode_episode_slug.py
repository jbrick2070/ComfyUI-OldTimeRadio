"""Native episode titles keep their script in the durable directory id."""
from __future__ import annotations

import unicodedata

from nodes import video_engine


def test_devanagari_vowel_marks_survive_episode_slug():
    title = "अंधकार में गोपनीय"
    slug = video_engine._safe_episode_title_slug(title)
    assert slug == "अंधकार_में_गोपनीय"
    expected_marks = [
        char for char in title
        if unicodedata.category(char).startswith("M")
    ]
    assert all(char in slug for char in expected_marks)


def test_english_slug_contract_is_unchanged():
    assert video_engine._safe_episode_title_slug(
        "Signal Lost - The Crystal!"
    ) == "signal_lost_the_crystal"
    assert video_engine._safe_episode_title_slug("!!!") == "untitled"


def test_truncation_does_not_split_a_base_from_its_mark():
    title = ("a" * 39) + "कि"
    slug = video_engine._safe_episode_title_slug(title, max_chars=40)
    assert slug == "a" * 39
    assert not slug.endswith("क")


def test_cjk_title_survives_episode_slug():
    assert video_engine._safe_episode_title_slug(
        "项链之争"
    ) == "项链之争"
