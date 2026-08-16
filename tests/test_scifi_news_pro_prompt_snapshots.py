"""Load-bearing prompt shapes for the six Fable model seams."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as ROUTING


@pytest.fixture(scope="module")
def stages():
    ROUTING._REGISTRY = None
    try:
        return dict(ROUTING.resolve_story_pack("scifi_news_pro").prompt_stages)
    finally:
        ROUTING._REGISTRY = None


def test_exact_six_model_seams(stages):
    assert set(stages) == {
        "scifi_news_pro_dossier_system",
        "scifi_news_pro_pitch_system",
        "scifi_news_pro_treatment_system",
        "scifi_news_pro_news_read_system",
        "scifi_news_pro_script_system",
        "scifi_news_pro_casting_system",
    }


@pytest.mark.parametrize(
    "seam",
    [
        "scifi_news_pro_dossier_system",
        "scifi_news_pro_pitch_system",
        "scifi_news_pro_treatment_system",
        "scifi_news_pro_news_read_system",
        "scifi_news_pro_casting_system",
    ],
)
def test_structured_seams_request_one_json_artifact(stages, seam):
    assert "return one json object only" in stages[seam].lower()


def test_script_seam_owns_complete_plain_text_grammar(stages):
    prompt = stages["scifi_news_pro_script_system"]
    for marker in (
        "TITLE: <episode title>",
        "MUSIC: <mood and instruments>",
        "SCENE <n>: <concrete setting>",
        "ANNOUNCER: <spoken words>",
        "<CAST NAME>: <spoken words>",
        "CODA: <spoken coda>",
        "END.",
    ):
        assert marker in prompt
    assert "Write the whole episode once" in prompt
    assert "requested duration is loose generation guidance" in prompt


def test_pitch_seam_describes_the_direct_pitch_artifact(stages):
    prompt = stages["scifi_news_pro_pitch_system"]
    for field in (
        "frame_card", "logline", "hook", "scifi_device", "cast_size",
        "ending_shape",
    ):
        assert field in prompt
    assert "dossier facts" in prompt.lower()


def test_casting_seam_assigns_voices_without_rewriting_story(stages):
    prompt = stages["scifi_news_pro_casting_system"]
    assert "available voice-stock id" in prompt.lower()
    assert "copy names and voice-stock ids exactly" in prompt.lower()
    assert "do not rewrite story text" in prompt.lower()
