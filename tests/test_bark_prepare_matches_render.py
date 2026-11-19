"""Bark prepares text with the renderer's speech-only rule (audit secondary).

prepare_text used the full token whitelist while generation used speech-only,
so a pure "[music]" line passed the voice gate's empty-text check and then
rendered as empty input, re-rolling to exhaustion. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_audio_engines import get_engine


@pytest.fixture
def bark(monkeypatch):
    monkeypatch.delenv("OTR_BARK_SPEECH_ONLY", raising=False)
    return get_engine("bark")


def test_a_pure_squeal_line_prepares_to_empty(bark):
    assert bark.prepare_text("[music]").strip() == ""


def test_spoken_words_and_safe_tokens_survive(bark):
    out = bark.prepare_text("[laughs] Well, hello there.")
    assert "hello there" in out.lower()
    assert "[laughs]" in out


def test_the_env_switch_still_restores_the_full_whitelist(monkeypatch):
    monkeypatch.setenv("OTR_BARK_SPEECH_ONLY", "0")
    assert "[music]" in get_engine("bark").prepare_text("[music]")
