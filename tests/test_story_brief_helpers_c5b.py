"""Positive tests for the four live story-brief consumer helpers."""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_story_brief_helpers as helpers


def _meta_ok() -> dict:
    return {
        "story_brief": (
            "a dim interrogation room under a swinging bulb, rain on tin"
        ),
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting": ["interrogation room", "steel table"],
            "lighting": ["swinging bare bulb", "neon label"],
            "atmosphere": ["smoke", "claustrophobic", "unlisted mood"],
        },
    }


def test_four_live_helpers_are_available():
    for name in (
        "get_story_brief_full",
        "get_story_brief_ltx",
        "get_story_brief_lighting",
        "get_story_brief_status",
    ):
        assert hasattr(helpers, name)

    signature = inspect.signature(helpers.get_story_brief_ltx)
    assert signature.parameters["max_chars"].default == 90


@pytest.mark.parametrize(
    ("meta", "status"),
    [
        (_meta_ok(), "ok"),
        (
            {
                "story_brief": "",
                "story_brief_status": "failed",
                "story_brief_terms": {},
            },
            "failed",
        ),
        ({}, "absent"),
    ],
)
def test_status_resolution(meta, status):
    assert helpers.get_story_brief_status(meta) == status


def test_full_and_ltx_helpers_transport_the_brief():
    meta = _meta_ok()
    full = helpers.get_story_brief_full(meta)
    assert full == meta["story_brief"]
    assert helpers.get_story_brief_ltx(meta, max_chars=10_000) == full
    assert len(helpers.get_story_brief_ltx(meta, max_chars=40)) <= 40


def test_lighting_helper_preserves_arbitrary_authored_terms():
    rendered = helpers.get_story_brief_lighting(_meta_ok())

    assert "neon label" in rendered
    assert "smoke" in rendered
    assert "unlisted mood" in rendered
    assert "interrogation room" not in rendered


def test_failed_or_absent_brief_returns_empty_consumer_text():
    failed = {
        "story_brief": "",
        "story_brief_status": "failed",
        "story_brief_terms": {},
    }
    assert helpers.get_story_brief_full(failed) == ""
    assert helpers.get_story_brief_lighting(failed) == ""
    assert helpers.get_story_brief_full({}) == ""
