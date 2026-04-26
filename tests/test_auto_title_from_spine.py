"""Regression tests for the spine-derived auto-title feature.

Added 2026-04-26 alongside `LLMScriptWriter._generate_title_from_spine`.

Covers:
  - clean LLM output passes through (single 2-5 word title)
  - leading "Title:" / "**" / smart-quote wrappers are stripped
  - stuck defaults ("untitled", "signal lost", etc.) are rejected
  - overlong sentence-leak output is rejected
  - empty winning_outline returns "" without calling the LLM
  - LLM exception is swallowed and returns "" (caller falls back)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes.story_orchestrator import LLMScriptWriter  # noqa: E402


SAMPLE_SPINE = """1. **Inciting Incident:** A magnetic pulse crashes the city's vertical
farms, leaving Rod Beesley as the sole hope for restoring crops.
2. **Protagonist Goal:** Rod must infiltrate AgriNet to steal the
proprietary algorithms.
3. **First Obstacle:** Palmer Simpson dispatches security forces.
4. **Midpoint Twist:** Rod uncovers that Palmer orchestrated the pulse.
5. **Climax Preparation:** Cornered, Rod triggers a digital lockdown.
6. **Resolution:** Rod extracts the algorithms and exposes Palmer."""

SAMPLE_NEWS = "Fish oil may be hurting your brain, new study finds | ScienceDaily\nMore headlines below."


def _writer():
    return LLMScriptWriter()


def test_clean_title_passes_through():
    with patch("nodes.story_orchestrator._generate_with_llm",
               return_value="The Greenhouse Gambit"):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="hard-sci-fi procedural",
            news_block=SAMPLE_NEWS,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            temperature=0.7,
        )
    assert out == "The Greenhouse Gambit"


def test_strip_title_prefix_and_quotes():
    with patch("nodes.story_orchestrator._generate_with_llm",
               return_value='**Title:** "Pulse"'):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="noir mystery",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == "Pulse"


def test_smart_quotes_stripped():
    with patch("nodes.story_orchestrator._generate_with_llm",
               return_value="“Agri-Crash”"):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="hard-sci-fi procedural",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == "Agri-Crash"


def test_only_first_line_taken_explanation_dropped():
    raw = "Pulse Out\n\nThis title evokes the magnetic pulse and the blackout."
    with patch("nodes.story_orchestrator._generate_with_llm",
               return_value=raw):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="hard-sci-fi procedural",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == "Pulse Out"


def test_stuck_default_rejected():
    for stuck in ("Untitled", "Signal Lost", "The Last Frequency", "Episode"):
        with patch("nodes.story_orchestrator._generate_with_llm",
                   return_value=stuck):
            out = _writer()._generate_title_from_spine(
                winning_outline=SAMPLE_SPINE,
                genre_flavor="cyberpunk",
                style_variant="noir mystery",
                news_block="",
                model_id="m",
                temperature=0.7,
            )
        assert out == "", f"stuck default {stuck!r} should be rejected"


def test_overlong_sentence_rejected():
    full_sentence = (
        "Here is a title that the model leaked as a complete English "
        "sentence well over ten words long indeed"
    )
    with patch("nodes.story_orchestrator._generate_with_llm",
               return_value=full_sentence):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="hard-sci-fi procedural",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == ""


def test_empty_outline_returns_empty_no_llm_call():
    """Helper short-circuits without calling the LLM when outline is empty."""
    with patch("nodes.story_orchestrator._generate_with_llm") as mock_gen:
        out = _writer()._generate_title_from_spine(
            winning_outline="",
            genre_flavor="cyberpunk",
            style_variant="noir mystery",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == ""
    mock_gen.assert_not_called()


def test_llm_exception_swallowed_returns_empty():
    """Caller relies on graceful '' to trigger the legacy title chain."""
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated LLM blowup")

    with patch("nodes.story_orchestrator._generate_with_llm",
               side_effect=_boom):
        out = _writer()._generate_title_from_spine(
            winning_outline=SAMPLE_SPINE,
            genre_flavor="cyberpunk",
            style_variant="noir mystery",
            news_block="",
            model_id="m",
            temperature=0.7,
        )
    assert out == ""


def test_2_to_5_word_titles_pass():
    for cand in ("Pulse", "The Greenhouse", "Agri-Crash", "The Long Pulse Out"):
        with patch("nodes.story_orchestrator._generate_with_llm",
                   return_value=cand):
            out = _writer()._generate_title_from_spine(
                winning_outline=SAMPLE_SPINE,
                genre_flavor="cyberpunk",
                style_variant="hard-sci-fi procedural",
                news_block="",
                model_id="m",
                temperature=0.7,
            )
        assert out == cand, f"clean title {cand!r} should pass through"
