"""Regression coverage for the five failed 45-word media-matrix runs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import _otr_scifi_codex as codex
from nodes import _otr_scifi_fable2 as fable2
from nodes._otr_video_engines import eng_mesh_stage  # noqa: F401
from nodes._otr_video_engines import render_driver as rd


# THE OPENING-MUSIC BEAT EXISTS IN TWO SHAPES AND THEY ARE NOT ALIASES.
#
# These two fixtures used to be ONE, and it encoded the defect. The old
# `_opening_ledger` gave a shot carrying `source_line_ids: ["music_opening_001"]`
# image rows keyed ONLY `b000_music_open`, and asserted the join SUCCEEDED --
# which was satisfiable only through `render_driver._canonical_visual_beat_id`,
# a hardcoded rewrite of one id to the other.
#
# That was CORRECT WHEN WRITTEN: the image producer then read the PRE-AUDIO
# ledger and really did mint `b000_music_open`. Commit 3446af3f retargeted
# canonical link 255 to ShotLock's POST-AUDIO ledger, where EpisodeAssembler has
# already mirrored the cue into an ordinary beat named `music_opening_001`, and
# the producer now mints THAT. The test kept asserting the old world, so it
# would have gone on passing while production died -- and it did die, on the
# 2026-08-12 `mesh_stage` leg (PBUG-20260811-02).
#
# The two shapes are genuinely different episodes, not two names for one thing:
#   POSITIONED  -- a mirrored opening cue starting at 0.0. ShotLock inserts no
#                  synthetic beat (`derive_opening_music_beat` requires a >= 2 s
#                  head gap), the shot carries the mirror's own line id, and the
#                  producer keyed its still under that same id.
#   SYNTHETIC   -- a real head gap with no mirrored cue. ShotLock DOES insert
#                  `b000_music_open`, the shot has EMPTY source_line_ids, and
#                  `derive_opening_music_beat` returns non-None so the producer
#                  mints that id. `_beat_id_for_shot` recovers it by stripping
#                  the `shot_` prefix.
# Each resolves through its OWN id. Neither needs a translation layer.


def _positioned_ledger() -> dict:
    """Mirrored opening cue at 0.0 -- post-audio assembler ids throughout."""
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
                "object_id": "still_music_opening_001",
                "kind": "scene_open",
                "beat_id": "music_opening_001",
                "path": "C:/tmp/open_scene.png",
            },
            {
                "object_id": "meshfodder_music_opening_001",
                "kind": "mesh_fodder",
                "beat_id": "music_opening_001",
                "mesh_subject_id": "radio_host",
                "path": "C:/tmp/open_fodder.png",
            },
        ]},
    }


def _synthetic_open_ledger() -> dict:
    """Genuine head gap -- ShotLock's synthetic beat, keyed b000_music_open."""
    return {
        "video": {"video_revision": 1, "shots": []},
        "lines": [{
            "line_id": "b001",
            "speaker_role": "announcer",
            "char_id": "",
            "start_s": 9.5,
            "dur_s": 8.0,
        }],
        "images": {"images": [
            {
                "object_id": "still_b000_music_open",
                "kind": "scene_open",
                "beat_id": "b000_music_open",
                "path": "C:/tmp/synth_scene.png",
            },
            {
                "object_id": "meshfodder_b000_music_open",
                "kind": "mesh_fodder",
                "beat_id": "b000_music_open",
                "mesh_subject_id": "radio_host",
                "path": "C:/tmp/synth_fodder.png",
            },
        ]},
    }


def _positioned_shot(engine_id: str, family: str) -> dict:
    return {
        "shot_id": "shot_music_opening_001",
        "source_line_ids": ["music_opening_001"],
        "char_id": "",
        "engine_id": engine_id,
        "family": family,
        "target_frame_count": 25,
        "creative": {},
    }


def _synthetic_open_shot(engine_id: str, family: str) -> dict:
    """No source lines -- the beat id is recovered from the shot_id."""
    return {
        "shot_id": "shot_b000_music_open",
        "source_line_ids": [],
        "char_id": "",
        "engine_id": engine_id,
        "family": family,
        "target_frame_count": 25,
        "creative": {},
    }


def test_positioned_music_mirror_joins_the_still_under_its_OWN_id():
    req = rd.build_request_from_shot(
        _positioned_shot("still_word", "static_image_gen"),
        _positioned_ledger(),
    )
    assert req["asset_refs"]["init_image"].endswith("open_scene.png")
    assert req["observability"]["init_source"] == "scene_still"


def test_positioned_music_mirror_joins_the_mesh_fodder_under_its_OWN_id():
    req = rd.build_request_from_shot(
        _positioned_shot("mesh_stage", "image_to_video"),
        _positioned_ledger(),
    )
    assert req["asset_refs"]["init_image"].endswith("open_fodder.png")
    assert req["observability"]["init_source"] == "mesh_fodder"
    assert req["mesh_subject_id"] == "radio_host"


def test_the_GENUINE_synthetic_opener_still_joins_b000_music_open():
    """The synthetic head-gap beat is not going away and must keep resolving --
    this is the case that justifies keeping `_OPENING_MUSIC_SUFFIX` after the
    translation function is deleted. Green before AND after the deletion."""
    req = rd.build_request_from_shot(
        _synthetic_open_shot("still_word", "static_image_gen"),
        _synthetic_open_ledger(),
    )
    assert req["asset_refs"]["init_image"].endswith("synth_scene.png")
    assert req["observability"]["init_source"] == "scene_still"


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
        envelope=fable2._build_envelope(3),
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


def test_p3_transport_bounds_prose_structurally_and_never_clips_silently():
    """The P3 strings carry ceilings, and an exact hit rerolls rather than ships.

    THIS REVERSES A DELIBERATE EARLIER DECISION and the reason is a live
    defect, so the reasoning is recorded rather than just the assertions. The
    previous contract left every authored string bounded only by provider
    capacity, on the principle that creative text must never be clipped by the
    typed boundary. That principle is right and is PRESERVED here -- but the
    unbounded half of it was also the runaway's mechanism: under constrained
    decoding the closing quote is the only exit from a string, and a decode
    that falls into a verbatim loop never samples it. Measured 2026-08-13:
    13,912 output tokens inside ONE string, and with MAX_CANDIDATE_CYCLES=3 a
    long enough runaway now KILLS a leg rather than merely delaying it.

    So the strings are finite, the ceilings sit 3-6x above the longest string
    of their kind ever observed, and "never silently clipped" is enforced by
    the reroll below rather than by leaving the string unbounded.
    """
    ceiling = codex._RADIO_SCORE_MAX_AUTHORED_TEXT_CHARS
    schema = codex.RadioScoreDraftV4.model_json_schema()
    assert schema["properties"]["premise"]["maxLength"] == ceiling
    scene = schema["$defs"]["RadioScoreDraftSceneV4"]
    assert scene["properties"]["description"]["maxLength"] == ceiling
    assert schema["properties"]["scenes"]["maxItems"] == 3
    assert scene["properties"]["shots"]["maxItems"] == 2
    assert scene["properties"]["beats"]["maxItems"] == 4

    receipt = codex._radio_score_draft_surface_receipt()
    assert receipt["output_budget_mode"] == "provider_capacity"
    assert receipt["requested_max_new_tokens"] is None
    assert receipt["authored_text_bounds"] == "structural_ceilings"
    assert receipt["max_authored_text_chars"] == ceiling
    assert "context_cap_tokens" not in receipt
    assert "input_token_reservation" not in receipt

    # ONE ceiling, and it must never bind real authorship. The corpus
    # measurement behind the value (widest authored string ever shipped: a
    # 4,549-char premise) is recorded in
    # docs/2026-08-13-writer-runaway-root-cause.md -- this line documents the
    # intent and does NOT re-measure it, so do not cite it as validation of the
    # corpus claim.
    assert ceiling > 4549
    assert codex._SCRIPT_TEXT_DRAFT_MAX_LINE_CHARS == ceiling





def test_p3_exact_ceiling_hit_rerolls_instead_of_shipping_truncated_text():
    """A string forced shut by its ceiling is degeneracy, not authorship.

    lm-format-enforcer returns only the closing quote once maxLength is
    reached, so a runaway string comes back as valid JSON that passes pydantic
    and stops mid-word. Without this check the cure would be quieter than the
    disease: the episode would simply carry a clipped description and nothing
    downstream would know. The error code is in the rerollable set on purpose.
    """
    draft = codex.RadioScoreDraftV4.model_validate({
        "title": "Signal",
        "premise": "A quiet signal forces a choice.",
        "setting": "An observatory at night.",
        "scenes": [{
            "env": "Observatory",
            # Exactly the ceiling -- what a forced-shut runaway produces.
            "description": "x" * codex._RADIO_SCORE_MAX_AUTHORED_TEXT_CHARS,
            "shots": [{"description": "A dish turns.",
                       "visual_prompt": "A dish turns against the stars."}],
            "beats": [{"shot_index": 0, "char_id": "announcer",
                       "line_count": 1, "intent": "open",
                       "arc_phase": "Setup", "fact_ids": []}],
        }],
        "music_cues": [{"cue_id": "music_open", "description": "A low drone.",
                        "generation_prompt": "Low analog drone, sparse.",
                        "anchor_beat_index": 0, "anchor_line_index": 0}],
    })

    with pytest.raises(codex.RadioScoreDraftCompileError) as excinfo:
        codex._assert_authored_text_within_bounds(draft)

    assert excinfo.value.code == "text_cap"
    assert "scenes[0].description" in excinfo.value.path
    # Rerollable, not fatal: the ladder retries rather than killing the leg.
    assert "text_cap" in codex._DRAFT_ERROR_CODES


def test_p3_call_uses_provider_capacity_prompt_transport(monkeypatch):
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
        max_new_tokens=None,
        call_journal={},
        prompt_must_fit=True,
        include_result_json_schema=False,
    )
    assert result == draft
    assert isinstance(observed["prompt"], codex.ProviderCapacityMessages)
    assert observed["prompt"]._otr_reserve_remaining_output_capacity is True
    assert observed["prompt"]._otr_fail_on_output_limit is True
    assert observed["max_new_tokens"] is None
