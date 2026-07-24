"""Positive regressions for unbounded source terms and brief output capacity."""
from __future__ import annotations

from nodes import news_interpreter as news
from nodes import _otr_story_brief as story_brief


def test_news_briefs_preserve_all_authored_terms_and_long_prose():
    terms = [
        "NASA",
        "FireSense",
        "Wildfire",
        "Thermal Sensor",
        "Fire Mission",
        "Fire Bulldozer",
        "Firefighter",
        "Pine Barrens",
        "University Consortium for Atmospheric Research and Public Safety",
    ]
    long_brief = "source-grounded premise " * 100

    result = news.NewsBriefs(
        casting_brief=long_brief,
        script_brief=long_brief,
        news_close_brief=long_brief,
        key_terms=terms,
    )

    assert result.casting_brief == long_brief
    assert result.script_brief == long_brief
    assert result.news_close_brief == long_brief
    assert result.key_terms == terms


def test_story_brief_structured_output_budget_has_json_headroom():
    assert story_brief._REFLECTION_MAX_NEW_TOKENS >= 512
    assert story_brief._PROMPT_VERSION == "v2"
    assert (
        story_brief._REFLECTION_STRUCTURAL_RETRY_TEMPERATURE
        < story_brief._REFLECTION_TEMPERATURE
    )
