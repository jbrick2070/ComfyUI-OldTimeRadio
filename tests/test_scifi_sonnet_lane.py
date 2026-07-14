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


def test_the_grounding_proof_goes_to_the_doctor_not_to_an_exception():
    """Live prompt 7199a7b1: the model invented the number 13 and the proof was
    RIGHT to catch it -- but it raised at the END, after every pass had already
    succeeded. The lane has a script doctor whose job is fixing flagged lines.
    """
    dossier = lane.FragmentDossierV4.model_validate({
        "verified_facts": [{
            "fact_id": "fact_1", "claim": "The probe carried 3 sensors.",
            "source_spans": [{"field": "summary", "start": 0, "end": 27,
                              "quote": "The probe carried 3 sensors"}],
        }],
        "key_numbers": [{
            "number_id": "num_1", "verbatim": "3", "fact_id": "fact_1",
            "source_span": {"field": "summary", "start": 18, "end": 19,
                            "quote": "3"},
        }],
        "named_entities": [], "tone": "measured",
        "headline_clean": "Probe returns", "provenance_note": "One fragment.",
        "payload_sha256": "a" * 64,
    })
    events = [
        lane.DraftLineV4(text="The record opens.", cites=[], non_fact=True,
                         speaker="ANNOUNCER", char_id="announcer",
                         source_pass="P1"),
        lane.DraftLineV4(text="The probe carried 3 sensors.", cites=["fact_1"],
                         speaker="ORUM", char_id="c02", source_pass="P2a"),
        lane.DraftLineV4(text="Flight 13 will follow.", cites=["fact_1"],
                         speaker="THESSALY", char_id="c03", source_pass="P2b"),
    ]
    clear = lane.AuditVerdictV4.model_validate({"status": "clear"})

    merged = lane._merge_grounding_into_audit(clear, events, dossier)
    # The Warden said "clear"; the proof says otherwise, and the proof wins.
    assert merged.status == "defect"
    assert any("13" in defect for defect in merged.defects)
    # It flags the line the DOCTOR numbers, not the raw index.
    assert merged.flagged_line_refs == [1]

    # A grounded record is left alone.
    events[2] = lane.DraftLineV4(
        text="Three sensors is a careful number.", cites=["fact_1"],
        speaker="THESSALY", char_id="c03", source_pass="P2b",
    )
    assert lane._merge_grounding_into_audit(
        clear, events, dossier,
    ).status == "clear"


def test_the_sources_own_acronyms_may_be_spoken():
    """Live prompt c7023ae8: the announcer's cold open said an acronym the
    fragment itself states, and the gate killed the episode AFTER every pass had
    already succeeded. The record exists to state the source's words.
    """
    dossier = lane.FragmentDossierV4.model_validate({
        "verified_facts": [{
            "fact_id": "fact_1", "claim": "NASA confirmed the signal.",
            "source_spans": [{"field": "summary", "start": 0, "end": 25,
                              "quote": "NASA confirmed the signal"}],
        }],
        "key_numbers": [], "named_entities": [],
        "tone": "measured", "headline_clean": "Signal confirmed",
        "provenance_note": "One fragment.", "payload_sha256": "a" * 64,
    })
    allowed = lane.allowed_spoken_all_caps(dossier)
    assert "NASA" in allowed

    assert lane._spoken_error("NASA confirmed it.", "ORUM", allowed) is None
    assert "all-caps" in lane._spoken_error("It ARRIVED.", "ORUM", allowed)
    # And with no dossier, the old strictness stands.
    assert lane._spoken_error("NASA confirmed it.", "ORUM")


def test_grounding_is_proven_against_the_dossier_not_judged_by_the_warden():
    """`invented_fact_flags` was a model's feeling. This is a proof.

    A factual line may only cite a fact the dossier holds, and may only speak a
    number the source itself states. Both are mechanical, so both may be fatal.
    """
    dossier = lane.FragmentDossierV4.model_validate({
        "verified_facts": [{
            "fact_id": "fact_1", "claim": "The probe carried 3 sensors.",
            "source_spans": [{"field": "summary", "start": 0, "end": 27,
                              "quote": "The probe carried 3 sensors"}],
        }],
        "key_numbers": [{
            "number_id": "num_1", "verbatim": "3", "fact_id": "fact_1",
            "source_span": {"field": "summary", "start": 18, "end": 19,
                            "quote": "3"},
        }],
        "named_entities": [],
        "tone": "measured",
        "headline_clean": "Probe returns",
        "provenance_note": "One recovered fragment.",
        "payload_sha256": "a" * 64,
    })

    def line(text, cites=(), non_fact=False, speaker="ORUM", char_id="c02"):
        return lane.DraftLineV4(
            text=text, cites=list(cites), non_fact=non_fact, speaker=speaker,
            char_id=char_id, source_pass="P2a",
        )

    grounded = [
        line("The record opens.", non_fact=True, speaker="ANNOUNCER",
             char_id="announcer"),
        line("The probe carried 3 sensors.", cites=["fact_1"]),
    ]
    assert lane.ungrounded_lines(grounded, dossier) == []

    phantom_cite = [line("The probe carried 3 sensors.", cites=["fact_4"])]
    assert "fact_4" in lane.ungrounded_lines(phantom_cite, dossier)[0]

    invented_number = [line("The probe carried 47 sensors.", cites=["fact_1"])]
    assert "47" in lane.ungrounded_lines(invented_number, dossier)[0]

    # A ceremonial line states nothing, so it can invent nothing.
    ceremonial = [line("The record holds. 47 seals.", non_fact=True,
                       speaker="VESH", char_id="c04")]
    assert lane.ungrounded_lines(ceremonial, dossier) == []


def test_sonnet_target_and_mode_fail_loud():
    with pytest.raises(lane.SonnetTargetRangeError):
        lane.validate_sonnet_payload(_payload(), {"seed_source": "rss_fetch", "target_words": 901})
    with pytest.raises(lane.SonnetPackContractError):
        lane.select_warden_mode_block("[CLEAR MODE only]", "clear")


def test_sonnet_tail_canon_uses_complete_episode_canon_protocol(tmp_path):
    from nodes import _otr_canon

    frame = _frame()
    canon = lane._build_sonnet_episode_canon(frame)

    assert isinstance(canon, _otr_canon.EpisodeCanon)
    assert canon.title == frame.session_title
    assert canon.premise == frame.session_premise
    assert canon.setting == frame.scene_env
    assert canon.time_of_day == ""
    assert canon.sound_palette == []
    _otr_canon.write_episode_canon(tmp_path, canon)
    assert _otr_canon.load_episode_canon(tmp_path) == canon
