from __future__ import annotations

import pytest

from nodes import _otr_scifi_sonnet as lane
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
        "link": "https://example.invalid/sonnet",
        "seed_text": text,
    }


def _frame() -> lane.SessionFrameV4:
    return lane.SessionFrameV4(
        session_title="Recovery Session",
        session_premise="The fragment records a cautious measurement.",
        registrar_cold_open="The archive opens tonight.",
        orum_register="clipped, exact",
        thessaly_register="restless, associative",
        vesh_register="flat, unhurried",
        scene_description="An archive chamber holds one recovered fragment.",
        scene_env="archive chamber",
        shot_description="A console illuminates the sealed record.",
        visual_prompt="A ceremonial archive console in blue light.",
        music_description="Measured archival tones.",
        music_generation_prompt="Measured archival tones with a quiet pulse.",
    )


def test_clear_path_uses_satisfied_schema_first():
    pack = routing.resolve_story_pack("scifi_sonnet")
    seam = pack.prompt_stages["sonnet_warden_system"]
    clear = lane.select_warden_mode_block(seam, "clear")
    defect = lane.select_warden_mode_block(seam, "defect")
    assert "Warden" in clear and "CLEAR MODE" in clear
    assert "Warden" in defect and "DEFECT MODE" in defect
    satisfied = lane.WardenSatisfiedV4(vesh_satisfied="The record holds.")
    challenge = lane.WardenChallengeV4(vesh_objection="Recheck the record.", registrar_reopening="Return to the fragment.")
    assert satisfied.vesh_satisfied and challenge.registrar_reopening


def test_sonnet_payload_cast_lock_and_spoken_hygiene():
    env, steer = lane.validate_sonnet_payload(_payload(), {"seed_source": "rss_fetch", "target_words": 721})
    assert env.source_mode == "rss" and steer["requested_words"] == 721
    pinned = dict(_payload())
    pinned["seed_text"] = "Pinned premise includes enough distinct words here for testing."
    assert lane.validate_sonnet_payload(pinned, {"seed_source": "custom_premise", "target_words": 30})[0].source_mode == "operator_pinned"
    cast = lane.lock_archive_cast(_frame())
    assert cast["c02"].name == "ORUM" and cast["c04"].voice_preset == "v2/en_speaker_0"
    bad = lane.DraftLineV4(text="(pause) the record holds", cites=["fact_0"], speaker="ORUM", char_id="c02", source_pass="P2a")
    with pytest.raises(lane.SonnetSpokenTextError):
        lane.validate_spoken_text_and_lock([bad], cast)


def test_sonnet_target_and_mode_fail_loud():
    with pytest.raises(lane.SonnetTargetRangeError):
        lane.validate_sonnet_payload(_payload(), {"seed_source": "rss_fetch", "target_words": 901})
    with pytest.raises(lane.SonnetPackContractError):
        lane.select_warden_mode_block("[CLEAR MODE only]", "clear")
