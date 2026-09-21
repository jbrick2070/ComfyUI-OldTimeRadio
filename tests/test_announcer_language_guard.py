# -*- coding: utf-8 -*-
"""The announcer line must be in the EPISODE's language, not the writer's.

Measured 2026-09-20/21 on live episodes. The native-language instruction is
sent -- `_announcer_system` adds the row's own lead -- but a weak writer
ignores it and nothing downstream noticed, because the structural validator
only rejected brackets. CastLock had already cast the Spanish voices, so a
Spanish voice read an English sentence aloud.

Every string below is REAL: the two failures are the lines that shipped, and
the four accepted openings are lines from published episodes.
"""
import pytest

from nodes._otr_line_composer import announcer_line_is_in_language


# The English announcer that shipped on a Spanish episode
# (signal_lost_el_pulso_de_cristal_en_el_abismo_es_20260920_223822).
SPANISH_EPISODE_ENGLISH_LINE = (
    "The engine hums low as Truman Terwilliger and Meredith Hayes argue over "
    "a garment on the boat while Truman threatens to destroy the vessel."
)
# The English closing that shipped on a French episode
# (signal_lost_une_maitresse_de_verre_et_de_faux_pleurs_fr_20260920_170919).
FRENCH_EPISODE_ENGLISH_CLOSING = (
    "Out of tonight's shadowed Arden where love turns to madness."
)


@pytest.mark.parametrize("iso,text", [
    ("es", SPANISH_EPISODE_ENGLISH_LINE),
    ("fr", FRENCH_EPISODE_ENGLISH_CLOSING),
    ("zh", "Tonight, a scene from Hamlet by William Shakespeare."),
    ("ja", "Tonight, a scene from A Dreamer's Tales by Lord Dunsany."),
])
def test_an_english_line_is_refused_on_a_non_english_episode(iso, text):
    assert announcer_line_is_in_language(text, {"episode_language": iso}) is False


@pytest.mark.parametrize("iso,text", [
    # Each of these was spoken in a published episode.
    ("fr", "Ce soir, une scène de Twelfth Night, de William Shakespeare, "
           "acte 1, scène 5."),
    ("zh", "今晚, 一场戏：Hamlet，William Shakespeare著 第1幕第1场."),
    ("ja", "今夜、一場面：A Dreamer's Tales、Lord Dunsany作."),
    ("es", "Esta noche, una escena de Hamlet, de William Shakespeare."),
])
def test_a_native_line_passes(iso, text):
    assert announcer_line_is_in_language(text, {"episode_language": iso}) is True


def test_english_episodes_are_never_checked():
    assert announcer_line_is_in_language(
        "Good evening. This is SIGNAL LOST.", {"episode_language": "en"}) is True


def test_the_guard_degrades_open_rather_than_killing_a_render():
    """No row, no meta, empty text: never block. This guard catches a WRONG
    language; it is not a prose gate and must not fail a render on its own."""
    assert announcer_line_is_in_language("anything", {}) is True
    assert announcer_line_is_in_language("anything", None) is True
    assert announcer_line_is_in_language("", {"episode_language": "es"}) is True
    assert announcer_line_is_in_language(
        "anything", {"episode_language": "zz-not-a-row"}) is True
