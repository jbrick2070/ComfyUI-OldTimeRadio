"""Regression coverage for the five failed 45-word media-matrix runs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nodes import _otr_scifi_codex as codex
from nodes import _otr_scifi_fable2 as fable2
from nodes._otr_video_engines import eng_mesh_stage  # noqa: F401
from nodes._otr_video_engines import render_driver as rd


def _opening_ledger(tmp_path: Path) -> dict:
    still = tmp_path / "open_scene.png"
    fodder = tmp_path / "open_fodder.png"

    return {
        "video": {"video_revision": 1, "shots": []},
        "lines": [{
            "line_id": "music_opening_001",
            "speaker_role": "music_open",
            "mirrored_from": "music",
            "char_id": "",
            "start_s": 0.0,
            "dur_s": 10.0,
        }],
        "images": {"images": [
            {
                "object_id": "still_b000_music_open",
                "kind": "scene_open",
                "beat_id": "b000_music_open",
                "path": str(still),
            },
            {
                "object_id": "meshfodder_b000_music_open",
                "kind": "mesh_fodder",
                "beat_id": "b000_music_open",
                "mesh_subject_id": "radio_host",
                "path": str(fodder),
            },
        ]},
    }


def _opening_shot(engine_id: str, family: str) -> dict:
    return {
        "shot_id": "shot_music_opening_001",
        "source_line_ids": ["music_opening_001"],
        "char_id": "",
        "engine_id": engine_id,
        "family": family,
        "target_frame_count": 25,
        "creative": {},
    }


def test_mirrored_music_line_joins_canonical_scene_still():
    req = rd.build_request_from_shot(
        _opening_shot("still_word", "static_image_gen"),
        _opening_ledger(Path("C:/tmp")),
    )
    assert req["asset_refs"]["init_image"].endswith("open_scene.png")
    assert req["observability"]["init_source"] == "scene_still"


def test_mirrored_music_line_joins_canonical_mesh_fodder_and_subject():
    req = rd.build_request_from_shot(
        _opening_shot("mesh_stage", "image_to_video"),
        _opening_ledger(Path("C:/tmp")),
    )
    assert req["asset_refs"]["init_image"].endswith("open_fodder.png")
    assert req["observability"]["init_source"] == "mesh_fodder"
    assert req["mesh_subject_id"] == "radio_host"


def _valid_script() -> str:
    return "\n".join([
        "TITLE: The Quiet Relay",
        "MUSIC: low glass tones",
        "ANNOUNCER: A relay waits above the sleeping valley.",
        "SCENE 1: the mountain relay room",
        "SELA: The signal is steady now.",
        "ANNOUNCER: The relay returns to silence before dawn.",
        "CODA: The factual source beneath this fiction is:",
        "MUSIC: one low sustained note",
        "END.",
    ])


def test_markup_repair_explicitly_handles_standalone_stage_direction():
    malformed = _valid_script().splitlines()
    malformed.insert(5, "(A sharp, electronic beep sounds.)")
    responses = iter(("\n".join(malformed), _valid_script()))
    calls = []

    def creative_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return next(responses)

    _raw, _parsed, telemetry = fable2._run_markup_ladder(
        creative_fn,
        pass_id="script",
        system="Return one complete radio play in the required markup.",
        base_user="Write the complete episode now.",
        envelope=fable2._build_envelope(45),
        cast_names=["SELA"],
        initial_temperature=0.75,
    )
    assert telemetry["attempts"] == 2
    repair_user = next(
        row["content"] for row in calls[1]["messages"]
        if row["role"] == "user"
    )
    assert "standalone parenthetical" in repair_user
    assert "folding" in repair_user


def test_p3_transport_has_finite_schema_and_output_reservation():
    schema = codex.RadioScoreDraftV4.model_json_schema()
    assert schema["properties"]["premise"]["maxLength"] == 240
    scene = schema["$defs"]["RadioScoreDraftSceneV4"]
    assert scene["properties"]["description"]["maxLength"] == 144
    assert codex._RADIO_SCORE_CONTEXT_CAP_TOKENS == 8192
    assert codex._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS == 1829
    receipt = codex._radio_score_draft_surface_receipt()
    assert receipt["output_budget_mode"] == "fixed_reservation"
    assert receipt["input_token_reservation"] == 8192 - 1829

def test_p3_call_uses_bounded_prompt_transport(monkeypatch):
    observed = {}
    draft = codex.RadioScoreDraftV4.model_validate({
        "title": "Signal",
        "premise": "A quiet signal forces a choice.",
        "setting": "An observatory at night.",
        "scenes": [{
            "env": "Receiver room",
            "description": "Instruments glow.",
            "shots": [{
                "description": "A receiver hums.",
                "visual_prompt": "Blue instrument light.",
            }],
            "beats": [{
                "shot_index": 0,
                "char_id": "c01",
                "line_count": 1,
                "intent": "Advance the choice.",
                "arc_phase": "arrival",
                "fact_ids": [],
            }],
        }],
        "music_cues": [{
            "cue_id": "music_open",
            "description": "A low pulse.",
            "generation_prompt": "Low sustained radio pulse.",
            "anchor_beat_index": 0,
            "anchor_line_index": 0,
        }],
    })

    def fake_structured_call(**kwargs):
        observed.update(kwargs)
        return draft

    monkeypatch.setattr(codex, "structured_call", fake_structured_call)
    result = codex.invoke_codex_structured(
        pass_id="P3",
        slot="creative",
        slot_fn=lambda *_args, **_kwargs: "",
        pack=SimpleNamespace(prompt_stages={
            "codex_radio_score_system": "score seam",
            "codex_coda_contract_system": "coda seam",
        }),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={"advisory_word_plan": {"per_beat": [{"beat_id": "b000"}]}},
        result_type=codex.RadioScoreDraftV4,
        post_validator=lambda _value: None,
        base_temperature=0.72,
        structural_retry_temperature=0.32,
        max_new_tokens=codex._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={},
        prompt_must_fit=True,
        include_result_json_schema=False,
    )
    assert result == draft
    assert isinstance(observed["prompt"], codex._PromptMustFitMessages)
    assert observed["max_new_tokens"] == codex._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS