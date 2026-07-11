from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodes import _otr_scifi_gemini as lane
from nodes import _otr_story_routing as routing


def _payload() -> dict[str, str]:
    text = ("Researchers measured a quiet signal from an ice moon through a calibrated "
            "antenna during an orbital survey. " * 20)
    return {
        "headline": "Researchers measure a quiet signal",
        "summary": "A cautious result was reported.",
        "full_text": text,
        "source": "Test Wire",
        "date": "2026-07-11",
        "link": "https://example.invalid/gemini",
        "seed_text": text,
    }


def test_gemini_registry_and_exact_three_pitch_shape():
    bank = routing.get_bank("scifi_gemini")
    pipe = routing.get_pipeline("scifi_gemini_multipass")
    assert bank.runnable is True and pipe.executable is True
    assert len(lane.PitchSlateV4(pitches=(
        {"premise": "a", "setting": "b", "tonal_palette": "c"},
        {"premise": "d", "setting": "e", "tonal_palette": "f"},
        {"premise": "g", "setting": "h", "tonal_palette": "i"},
    )).pitches) == 3


def test_payload_pin_and_advisory_centers():
    env = lane.validate_gemini_payload(_payload(), {"seed_source": "rss_fetch", "target_words": 721})
    assert env.source_mode == "rss"
    pinned = dict(_payload())
    pinned["seed_text"] = "Pinned premise includes enough distinct words here for testing."
    assert lane.validate_gemini_payload(pinned, {"seed_source": "custom_premise", "target_words": 30}).source_mode == "operator_pinned"
    bands = lane.make_advisory_word_blueprint(719, ["b001", "b002", "b003"])
    assert sum(x.advisory_word_center for x in bands) == 719


def test_clean_critique_and_spoken_rejections():
    clean = lane.SceneCritiqueV4(passed=True, feedback="", line_fact_ids={}, sfw_pass=True)
    assert clean.passed and clean.feedback == ""
    assert lane._spoken_error("(pause) the signal arrives", "ORUM")
    assert lane._spoken_error("NASA confirms it", "ORUM")
    assert lane._spoken_error("The signal arrives", "ORUM") is None
    with pytest.raises(lane.SciFiGeminiTargetRangeError):
        lane.validate_gemini_payload(_payload(), {"seed_source": "rss_fetch", "target_words": 901})


def test_gemini_tail_canon_uses_complete_episode_canon_protocol(tmp_path):
    from nodes import _otr_canon

    outline = SimpleNamespace(
        title="Ice Moon Signal",
        premise="Will the team risk an answer to the quiet signal?",
        setting="an orbital listening station",
        time_of_day="night shift",
    )
    canon = lane._build_gemini_episode_canon(outline)

    assert isinstance(canon, _otr_canon.EpisodeCanon)
    assert canon.title == outline.title
    assert canon.premise == outline.premise
    assert canon.setting == outline.setting
    assert canon.time_of_day == outline.time_of_day
    assert canon.sound_palette == []
    _otr_canon.write_episode_canon(tmp_path, canon)
    assert _otr_canon.load_episode_canon(tmp_path) == canon
