from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nodes import _otr_scifi_codex as lane
from nodes import _otr_story_routing as routing


def _payload(words: int = 90) -> dict[str, str]:
    text = ("The observatory measured a quiet signal from a distant ice moon "
            "through a calibrated antenna during a cautious orbital survey. " * words)
    return {
        "headline": "Observatory measures a quiet signal",
        "summary": "Researchers report a cautious result.",
        "full_text": text,
        "source": "Test Wire",
        "date": "2026-07-11",
        "link": "https://example.invalid/story",
        "seed_text": text,
    }


def _metadata_repair_score() -> lane.RadioScoreV4:
    scene_id = "scene_001"
    return lane.RadioScoreV4(
        title="Signal at the observatory",
        premise="A signal forces a cautious choice.",
        setting="A quiet observatory control room.",
        advisory_word_plan=lane.AdvisoryWordPlanV4(
            advisory_total_center=30,
            per_beat=[
                {"beat_id": "b001", "advisory_word_center": 10},
                {"beat_id": "b002", "advisory_word_center": 10},
                {"beat_id": "b003", "advisory_word_center": 10},
            ],
        ),
        scenes=[
            lane.ScenePlanV4(
                scene_id=scene_id,
                env="Observatory",
                description="A receiver clicks under a cold sky.",
                shots=[
                    lane.ShotPlanV4(
                        shot_id="shot_001", scene_id=scene_id,
                        description="Receiver dial", visual_prompt="A receiver dial glows.",
                    ),
                    lane.ShotPlanV4(
                        shot_id="shot_002", scene_id=scene_id,
                        description="Observer listens", visual_prompt="An observer listens.",
                    ),
                ],
                beats=[
                    lane.BeatPlanV4(
                        beat_id="b001", scene_id=scene_id, shot_id="shot_001",
                        speaker="ANNOUNCER", char_id="announcer", speaker_role="announcer",
                        line_ids=["l001", "l002"], order=1,
                        intent="Establish the signal.", arc_phase="arrival",
                        fact_ids=["F01"], advisory_voiced_word_center=10,
                    ),
                    lane.BeatPlanV4(
                        beat_id="b002", scene_id=scene_id, shot_id="shot_001",
                        speaker="ANNOUNCER", char_id="announcer", speaker_role="announcer",
                        line_ids=["l003"], order=2,
                        intent="Raise the choice.", arc_phase="test",
                        fact_ids=["F01"], advisory_voiced_word_center=10,
                    ),
                    lane.BeatPlanV4(
                        beat_id="b003", scene_id=scene_id, shot_id="shot_002",
                        speaker="ANNOUNCER", char_id="announcer", speaker_role="announcer",
                        line_ids=["l004"], order=3,
                        intent="Commit to caution.", arc_phase="decision",
                        fact_ids=["F01"], advisory_voiced_word_center=10,
                    ),
                ],
            )
        ],
        music_cues=[
            lane.MusicCueV4(
                cue_id="music_open", placement="open", description="A low signal pulse.",
                generation_prompt="Low sustained radio pulse.", anchor_line_id="l001",
                anchor_beat_id="b001",
            )
        ],
    )


_GEMMA_E4B_TOKENIZER_SNAPSHOT = Path(
    r"C:\ComfyUI-Models\huggingface\hub\models--google--gemma-4-E4B-it"
    r"\snapshots\c53e9d33178b12afbad4a48334d21e19b8c29761"
)


def _bounded_prose(characters: int, seed: str) -> str:
    text = (f"{seed} radio signal " * (characters // 18 + 2))[:characters]
    return text.rstrip()


def _legacy_metadata_artifact() -> dict[str, object]:
    lines = [
        {
            "line_id": line_id,
            "beat_id": beat_id,
            "shot_id": "legacy_shot",
            "char_id": "announcer",
            "speaker_role": "announcer",
            "text": text,
            "skip": False,
            "tts_skip_reason": None,
            "traits": "measured",
            "boundary": "beat_end",
            "arc_phase": phase,
            "compose_flags": ["retain"],
            "beat_intent": intent,
            "dialogue_slot_id": f"slot_{line_id}",
            "fact_ids": ["F01"],
            "speaker": "ANNOUNCER",
            "legacy_line_metadata": "remove me",
        }
        for line_id, beat_id, phase, intent, text in (
            ("l001", "b001", "arrival", "Establish the signal.", "The receiver catches a quiet pulse."),
            ("l002", "b001", "arrival", "Establish the signal.", "Its timing unsettles the night staff."),
            ("l003", "b002", "test", "Raise the choice.", "They choose patience over a reckless reply."),
            ("l004", "b003", "decision", "Commit to caution.", "The report remains careful when dawn arrives."),
        )
    ]
    lines[0].pop("shot_id")
    lines[2].pop("shot_id")
    return {
        "schema_version": "scifi_codex.script_artifact.v1",
        "title": "Signal at the observatory",
        "scenes": [
            {
                "scene_id": "scene_001",
                "description": "A receiver clicks under a cold sky.",
                "premise": "A signal forces a cautious choice.",
            }
        ],
        # Deliberately shuffled: accepted score order, not response order,
        # controls the deterministic boundary derivation.
        "lines": [lines[2], lines[0], lines[3], lines[1]],
        "music_cues": [
            {
                "cue_id": "music_open",
                "placement": "open",
                "description": "A low signal pulse.",
                "generation_prompt": "Low sustained radio pulse.",
                "anchor_line_id": "l001",
                "anchor_beat_id": "b001",
                "speaker": "forbidden metadata",
            }
        ],
        "legacy_artifact_metadata": "remove me",
    }


def _fenced_json(data: object) -> str:
    return "Here is the repaired artifact.\n```json\n" + json.dumps(data) + "\n```\n"


def _metadata_repair_cast() -> lane.CastPlanV4:
    return lane.CastPlanV4(cast=[
        lane.CastPlanRowV4(
            char_id="announcer", name="ANNOUNCER", character_description="Calm witness.",
            gender="neutral", role_in_conflict="Reports the choice.", voice_slot="announcer",
        ),
        lane.CastPlanRowV4(
            char_id="c01", name="Iona", character_description="Careful observer.",
            gender="female", role_in_conflict="Tests the report.", voice_slot="c01",
        ),
    ])


def _metadata_repair_fact_index() -> lane.FactIndexV4:
    return lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id="F01", claim="A cautious signal reaches the observatory.",
            source_spans=[lane.SourceSpanV4(
                field="headline", start=0, end=6, quote="Signal",
            )],
        )],
        entities=[], numbers=[], tone="cautious", payload_sha256="0" * 64,
    )


def _draft_for_advisory(
    advisory: lane.AdvisoryWordPlanV4,
    *,
    max_width: bool = False,
    cast_ids: tuple[str, ...] = ("announcer", "c01"),
) -> lane.RadioScoreDraftV4:
    """Build a valid compact draft for the passed, possibly b000-based plan."""
    beat_total = len(advisory.per_beat)
    scenes: list[dict[str, object]] = []
    cursor = 0
    while cursor < beat_total:
        local_count = min(4, beat_total - cursor)
        scene_number = len(scenes) + 1
        shot_count = 1 if local_count == 1 else 2
        beats = []
        for local_index in range(local_count):
            flat_index = cursor + local_index
            beats.append({
                "shot_index": local_index % shot_count,
                "char_id": cast_ids[flat_index % len(cast_ids)],
                "line_count": 2 if max_width else 1,
                "intent": _bounded_prose(64 if max_width else 24, f"intent{flat_index}"),
                "arc_phase": _bounded_prose(28 if max_width else 12, f"arc{flat_index}"),
                "fact_ids": ["F01"],
            })
        scenes.append({
            "env": _bounded_prose(56 if max_width else 20, f"env{scene_number}"),
            "description": _bounded_prose(144 if max_width else 28, f"scene{scene_number}"),
            "shots": [
                {
                    "description": _bounded_prose(144 if max_width else 24, f"shot{scene_number}a"),
                    "visual_prompt": _bounded_prose(120 if max_width else 32, f"visual{scene_number}a"),
                },
            ] + ([
                {
                    "description": _bounded_prose(144 if max_width else 24, f"shot{scene_number}b"),
                    "visual_prompt": _bounded_prose(120 if max_width else 32, f"visual{scene_number}b"),
                },
            ] if shot_count == 2 else []),
            "beats": beats,
        })
        cursor += local_count
    cue_specs = [("music_open", 0, 0)]
    if beat_total >= 6:
        cue_specs.append(("music_inter", beat_total // 2, 0))
    if beat_total >= 9:
        cue_specs.append(("music_close", beat_total - 1, 1 if max_width else 0))
    return lane.RadioScoreDraftV4(
        title=_bounded_prose(64 if max_width else 32, "title"),
        premise=_bounded_prose(240 if max_width else 56, "premise"),
        setting=_bounded_prose(80 if max_width else 32, "setting"),
        scenes=scenes,
        music_cues=[
            {
                "cue_id": cue_id,
                "description": _bounded_prose(80 if max_width else 28, cue_id),
                "generation_prompt": _bounded_prose(120 if max_width else 36, f"prompt{cue_id}"),
                "anchor_beat_index": beat_index,
                "anchor_line_index": line_index,
            }
            for cue_id, beat_index, line_index in cue_specs
        ],
    )


def _draft_context(
    advisory: lane.AdvisoryWordPlanV4,
    cast: lane.CastPlanV4,
    fact_index: lane.FactIndexV4,
) -> dict[str, object]:
    question = lane.DramaticQuestionV4(
        question="What does the cautious signal demand?",
        consequence="A reckless reply could turn a warning into a disaster.",
        ending_direction="Choose restraint before the report reaches dawn.",
    )
    return {
        "question": question.model_dump(mode="json"),
        "cast": cast.model_dump(mode="json"),
        "fact_index": lane._compact_p0_fact_context(fact_index),
        "advisory_word_plan": advisory.model_dump(mode="json"),
    }


def test_cast_plan_locks_and_repairs_the_fixed_announcer_identity():
    cast = _metadata_repair_cast()
    assert lane._validate_cast_plan(cast) is None
    bad_announcer = cast.cast[0].model_copy(update={"name": "Narrator"})
    bad = cast.model_copy(update={"cast": [bad_announcer, *cast.cast[1:]]})
    assert lane._validate_cast_plan(bad) == (
        "announcer row must use the exact fixed name ANNOUNCER"
    )
    repaired = lane.repair_cast_plan_metadata(json.dumps(bad.model_dump(mode="json")))
    assert repaired is not None
    assert lane._validate_cast_plan(repaired) is None
    assert repaired.cast[0].name == "ANNOUNCER"

    full_name = repaired.cast[1].model_copy(update={"name": "Dr. Amelia Hart"})
    assert lane._validate_cast_plan(
        repaired.model_copy(update={"cast": [repaired.cast[0], full_name]}),
    ) is None
    acronym_name = repaired.cast[1].model_copy(update={"name": "AI Unit Seven"})
    assert lane._validate_cast_plan(
        repaired.model_copy(update={"cast": [repaired.cast[0], acronym_name]}),
    ) is None
    digit_name = repaired.cast[1].model_copy(update={"name": "AI Unit 7"})
    assert "canonical Title-Case name" in lane._validate_cast_plan(
        repaired.model_copy(update={"cast": [repaired.cast[0], digit_name]}),
    )
    label_name = repaired.cast[1].model_copy(update={"name": "AI UNIT"})
    assert "canonical Title-Case name" in lane._validate_cast_plan(
        repaired.model_copy(update={"cast": [repaired.cast[0], label_name]}),
    )
    invalid_name = repaired.cast[1].model_copy(update={"name": "dr. amelia hart"})
    assert "canonical Title-Case name" in lane._validate_cast_plan(
        repaired.model_copy(update={"cast": [repaired.cast[0], invalid_name]}),
    )

    calls: list[dict[str, object]] = []
    pack = SimpleNamespace(prompt_stages={"codex_pressure_cast_system": "Cast."})

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return json.dumps(bad.model_dump(mode="json"))

    result = lane.invoke_codex_structured(
        pass_id="P2", slot="creative", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_pressure_cast_system",), artifact_inputs={},
        result_type=lane.CastPlanV4, post_validator=lane._validate_cast_plan,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=1600, call_journal={},
    )
    assert isinstance(result, lane.CastPlanV4)
    assert result.cast[0].name == "ANNOUNCER"
    assert len(calls) == 1


def test_p2_repair_accepts_short_acronym_inside_title_case_character_name():
    cast = _metadata_repair_cast()
    invalid_row = cast.cast[1].model_copy(update={"name": "AI Unit 7"})
    repaired_row = invalid_row.model_copy(update={"name": "AI Unit Seven"})
    invalid = cast.model_copy(update={"cast": [cast.cast[0], invalid_row]})
    repaired = cast.model_copy(update={"cast": [cast.cast[0], repaired_row]})
    responses = [
        json.dumps(invalid.model_dump(mode="json")),
        json.dumps(repaired.model_dump(mode="json")),
    ]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane.invoke_codex_structured(
        pass_id="P2", slot="creative", slot_fn=slot_fn,
        pack=SimpleNamespace(prompt_stages={"codex_pressure_cast_system": "Cast."}),
        seam_refs=("codex_pressure_cast_system",), artifact_inputs={},
        result_type=lane.CastPlanV4, post_validator=lane._validate_cast_plan,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=1600, call_journal={},
    )

    assert result.cast[1].name == "AI Unit Seven"
    assert len(calls) == 2
    repair_system = calls[1]["messages"][0]["content"]
    assert "One short 2-3 letter acronym token is allowed" in repair_system
    assert "digits and all-uppercase full labels are forbidden" in repair_system


def test_codex_pack_and_registry_contract():
    bank = routing.get_bank("scifi_codex")
    assert bank.runnable is True
    assert bank.default_story_pipeline == "scifi_codex_circuit"
    pack = routing.resolve_story_pack("scifi_codex")
    assert pack.story_model_id == "scifi_codex_v1"
    assert set(pack.prompt_stages) == set(routing.get_pipeline("scifi_codex_circuit").declared_seams)
    pack_path = Path(__file__).parents[1] / "nodes" / "story_packs" / "scifi_codex" / "scifi_codex_v1.json"
    assert json.loads(pack_path.read_text(encoding="utf-8"))["source_bank_id"] == "scifi_codex"
    assert "ScriptArtifactV4" in pack.prompt_stages["codex_play_system"]
    assert "ScriptArtifactV4" in pack.prompt_stages["codex_retake_system"]
    assert not any("ScriptArtifactV1" in text for text in pack.prompt_stages.values())


def test_payload_route_and_one_use_word_steer():
    env, steer = lane.validate_payload_envelope(_payload(), {"seed_source": "rss_fetch", "target_words": 721})
    assert env.source_mode == "rss"
    assert steer.requested_words == 721
    pinned = dict(_payload())
    pinned["seed_text"] = "A pinned premise with enough distinct words for testing."
    env, _ = lane.validate_payload_envelope(pinned, {"seed_source": "custom_premise", "target_words": 30})
    assert env.source_mode == "operator_pinned"
    with pytest.raises(lane.CodexPayloadRouteError):
        lane.validate_payload_envelope(_payload(), {"seed_source": "other", "target_words": 720})


def test_p0_evidence_projection_dedupes_rss_aliases_without_rebasing_a0():
    body = (
        "The fallback observatory report preserves literal coordinate evidence "
        "for independent science review and careful archival verification. "
        * 90
    ).strip()
    payload = _payload()
    payload.update({
        "headline": "Fallback report",
        "summary": body,
        "full_text": body,
        "seed_text": f"Fallback report {body}",
    })
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 30},
    )
    canonical = envelope.model_dump(mode="json")

    inputs = lane._p0_artifact_inputs(envelope)
    projected = inputs["payload"]["payload"]

    assert projected == {"seed_text": payload["seed_text"]}
    assert inputs["allowed_source_fields"] == ["seed_text"]
    assert envelope.model_dump(mode="json") == canonical
    assert inputs["payload"]["source_digest"] == lane._digest(payload)
    assert projected["seed_text"][-48:] == payload["seed_text"][-48:]
    assert {"headline", "summary", "full_text", "source", "date", "link"}.isdisjoint(projected)

    omitted_alias = lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id="F01", claim="Fallback report evidence.",
            source_spans=[lane.SourceSpanV4(
                field="full_text", start=0, end=8, quote="Fallback",
            )],
        )],
        entities=[], numbers=[], tone="cautious", payload_sha256="0" * 64,
    )
    assert lane._validate_fact_index(
        omitted_alias, payload,
        allowed_source_fields=frozenset(inputs["allowed_source_fields"]),
    ) == "fact F01 cites source field 'full_text' outside the supplied P0 evidence"


def test_p0_evidence_projection_keeps_unique_legal_evidence_and_pinned_seed():
    payload = _payload()
    payload["seed_text"] = "A separately supplied RSS note contains unique literal evidence."
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 30},
    )
    projected = lane._p0_artifact_inputs(envelope)["payload"]["payload"]
    assert list(projected) == ["seed_text", "full_text", "headline", "summary"]
    assert projected["seed_text"] == payload["seed_text"]

    pinned = _payload()
    pinned_text = "A pinned premise retains its exact original coordinates for every source span."
    pinned.update({"headline": "", "summary": "", "full_text": pinned_text, "seed_text": pinned_text})
    pinned_envelope, _ = lane.validate_payload_envelope(
        pinned, {"seed_source": "custom_premise", "target_words": 30},
    )
    pinned_projected = lane._p0_artifact_inputs(pinned_envelope)["payload"]["payload"]
    assert pinned_projected == {"seed_text": pinned_text}


def test_p0_structured_prompt_receives_only_the_evidence_projection():
    payload = _payload()
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 30},
    )
    inputs = lane._p0_artifact_inputs(envelope)
    pack = SimpleNamespace(prompt_stages={"codex_fact_index_system": "Read A0."})
    calls: list[dict[str, object]] = []
    raw = json.dumps({
        "facts": [{
            "fact_id": "F01", "claim": "The observatory records a signal.",
            "source_spans": [{"field": "headline", "start": 0, "end": 11, "quote": "Observatory"}],
            "numeric_tokens": [],
        }],
        "entities": [], "numbers": [], "tone": "cautious", "payload_sha256": "0" * 64,
    })

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return raw

    lane.invoke_codex_structured(
        pass_id="P0", slot="technical", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_fact_index_system",), artifact_inputs=inputs,
        result_type=lane.FactIndexV4, post_validator=lambda _: None,
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=2000, call_journal={}, prompt_must_fit=True,
    )

    assert getattr(calls[0]["messages"], "_otr_prompt_must_fit", False) is True
    request = json.loads(calls[0]["messages"][1]["content"])
    assert request["artifact_inputs"] == inputs
    assert "result_json_schema" in request
    prompt_text = calls[0]["messages"][1]["content"]
    assert '"source"' not in prompt_text
    assert '"date"' not in prompt_text
    assert '"link"' not in prompt_text
    assert "P0 COMPACT EXTRACTION CONTRACT" in calls[0]["messages"][0]["content"]


def test_fact_index_contract_bounds_output_surface():
    """P0's accepted JSON has a finite envelope before a token cap is chosen."""
    span = lane.SourceSpanV4(
        field="headline", start=0, end=6, quote="Signal",
    )
    fact = lane.FactV4(
        fact_id="F01", claim="A signal is recorded.", source_spans=[span],
    )
    valid = {
        "facts": [fact.model_dump(mode="json")],
        "entities": [],
        "numbers": [],
        "tone": "cautious",
        "payload_sha256": "0" * 64,
    }
    assert lane.FactIndexV4.model_validate(valid).facts[0].fact_id == "F01"

    too_many = dict(valid)
    too_many["facts"] = [fact.model_dump(mode="json")] * 7
    with pytest.raises(Exception):
        lane.FactIndexV4.model_validate(too_many)
    with pytest.raises(Exception):
        lane.FactV4(
            fact_id="F01",
            claim="A signal is recorded.",
            source_spans=[span, span],
        )
    with pytest.raises(Exception):
        lane.SourceSpanV4(
            field="headline", start=0, end=241, quote="x" * 241,
        )
    with pytest.raises(Exception):
        lane.FactIndexV4.model_validate({**valid, "tone": []})


def test_fact_index_token_budget_keeps_the_live_120_word_window():
    """The live 2,455-token P0 prompt gets a named bounded-output reservation."""
    context_cap = 8192
    observed_base_prompt_tokens = 2455

    assert lane.p0_output_token_budget() == 2800
    assert lane.p0_output_token_budget(extra_root_fields=2) == 3000
    assert context_cap - lane.p0_output_token_budget() >= observed_base_prompt_tokens
    receipt = lane.p0_contract_receipt()
    assert receipt["max_new_tokens"] == 2800
    assert receipt["max_fact_rows"] == 6
    assert receipt["max_spans_per_evidence_row"] == 1


def test_p0_typed_repair_is_compact_and_requires_scalar_tone():
    """A model must repair tone; Python must not turn a list into authored prose."""
    payload = _payload()
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 120},
    )
    inputs = lane._p0_artifact_inputs(envelope)
    field, evidence = next(iter(inputs["payload"]["payload"].items()))
    quote = evidence[: min(32, len(evidence))]
    artifact = {
        "facts": [{
            "fact_id": "F01",
            "claim": "The observatory records a signal.",
            "source_spans": [{
                "field": field, "start": 0, "end": len(quote), "quote": quote,
            }],
            "numeric_tokens": [],
        }],
        "entities": [],
        "numbers": [],
        "tone": [],
        "payload_sha256": envelope.source_digest,
    }
    repaired = {**artifact, "tone": "cautious"}
    replies = iter((json.dumps(artifact), json.dumps(repaired)))
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return next(replies)

    result = lane.invoke_codex_structured(
        pass_id="P0", slot="technical", slot_fn=slot_fn,
        pack=SimpleNamespace(prompt_stages={"codex_fact_index_system": "Read A0."}),
        seam_refs=("codex_fact_index_system",), artifact_inputs=inputs,
        result_type=lane.FactIndexV4,
        post_validator=lambda value: lane._validate_fact_index(
            value, payload,
            allowed_source_fields=frozenset(inputs["allowed_source_fields"]),
            expected_payload_sha256=envelope.source_digest,
        ),
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=lane.p0_output_token_budget(), call_journal={},
        prompt_must_fit=True,
    )

    assert result.tone == "cautious"
    # A schema-shaped response skips the redundant structural retry and goes
    # straight to typed repair, so the model-owned scalar tone is call two.
    assert len(calls) == 2
    assert all(
        getattr(call["messages"], "_otr_prompt_must_fit", False)
        for call in calls
    )
    repair_system = calls[-1]["messages"][0]["content"]
    repair_user = calls[-1]["messages"][1]["content"]
    assert "tone is one nonempty scalar" in repair_system
    assert "<failed_fact_index>" in repair_user
    assert "<source_evidence>" in repair_user
    assert "original_request" not in repair_user
    assert "artifact_inputs" not in repair_user


def test_p0_deterministic_repair_cannot_reintroduce_an_omitted_alias():
    body = (
        "The fallback observatory report preserves literal coordinate evidence "
        "for independent science review and careful archival verification. "
        * 90
    ).strip()
    payload = _payload()
    payload.update({
        "headline": "Fallback report",
        "summary": body,
        "full_text": body,
        "seed_text": f"Fallback report {body}",
    })
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 30},
    )
    inputs = lane._p0_artifact_inputs(envelope)
    allowed = frozenset(inputs["allowed_source_fields"])
    pack = SimpleNamespace(prompt_stages={"codex_fact_index_system": "Read A0."})
    quote = "The fallback observatory report"
    raw = json.dumps({
        "facts": [{
            "fact_id": "F0", "claim": "The fallback report preserves evidence.",
            "source_spans": [{
                "field": "full_text", "start": 0,
                "end": len(quote), "quote": quote,
            }], "numeric_tokens": [],
        }],
        "entities": [], "numbers": [], "tone": "cautious", "payload_sha256": "0" * 64,
    })
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return raw

    repaired = lane.invoke_codex_structured(
        pass_id="P0", slot="technical", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_fact_index_system",), artifact_inputs=inputs,
        result_type=lane.FactIndexV4,
        post_validator=lambda value: lane._validate_fact_index(
            value, payload, allowed_source_fields=allowed,
            expected_payload_sha256=envelope.source_digest,
        ),
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=2000, call_journal={},
    )

    assert len(calls) == 1
    span = repaired.facts[0].source_spans[0]
    assert repaired.facts[0].fact_id == "F01"
    assert repaired.payload_sha256 == envelope.source_digest
    assert span.field == "seed_text"
    assert span.field in allowed
    assert payload[span.field][span.start:span.end] == quote


def test_p0_deterministic_repair_bounds_an_exact_overwide_literal_quote():
    payload = _payload()
    envelope, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 30},
    )
    inputs = lane._p0_artifact_inputs(envelope)
    allowed = frozenset(inputs["allowed_source_fields"])
    pack = SimpleNamespace(prompt_stages={"codex_fact_index_system": "Read A0."})
    quote = payload["seed_text"][: lane.MAX_QUOTE_CHARS + 60]
    raw = json.dumps({
        "facts": [{
            "fact_id": "F0",
            "claim": "The observatory measured a quiet signal.",
            "source_spans": [{
                "field": "seed_text", "start": 0, "end": 80, "quote": quote,
            }],
            "numeric_tokens": [],
        }],
        "entities": [],
        "numbers": [],
        "tone": "cautious",
        "payload_sha256": "0" * 64,
    })
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return raw

    repaired = lane.invoke_codex_structured(
        pass_id="P0", slot="technical", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_fact_index_system",), artifact_inputs=inputs,
        result_type=lane.FactIndexV4,
        post_validator=lambda value: lane._validate_fact_index(
            value, payload, allowed_source_fields=allowed,
            expected_payload_sha256=envelope.source_digest,
        ),
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=2000, call_journal={}, clamp_overlong_strings=False,
    )

    assert len(calls) == 1
    assert repaired.facts[0].claim == "The observatory measured a quiet signal."
    span = repaired.facts[0].source_spans[0]
    assert span.field == "seed_text"
    assert span.start == 0
    assert span.end == lane.MAX_QUOTE_CHARS
    assert span.quote == payload["seed_text"][: lane.MAX_QUOTE_CHARS]
    assert payload[span.field][span.start:span.end] == span.quote


def test_literal_fact_spans_and_reject_only_spoken_hygiene():
    payload = _payload()
    quote = payload["headline"][0:10]
    fact = lane.FactIndexV4(
        facts=[lane.FactV4(fact_id="F01", claim="a signal", source_spans=[lane.SourceSpanV4(field="headline", start=0, end=10, quote=quote)])],
        entities=[], numbers=[], tone="cautious", payload_sha256="0" * 64,
    )
    assert lane._validate_fact_index(fact, payload) is None
    broken = fact.model_copy(update={"facts": [fact.facts[0].model_copy(update={"source_spans": [lane.SourceSpanV4(field="headline", start=0, end=10, quote="wrong")]} )]})
    assert lane._validate_fact_index(broken, payload)
    assert lane._spoken_error("(pause) the signal arrives", "Iona")
    assert lane._spoken_error("NASA confirms it", "Iona")
    assert lane._spoken_error(
        "NASA confirms it", "Iona", frozenset({"NASA"}),
    ) is None
    assert lane._spoken_error(
        "NASA says STOP", "Iona", frozenset({"NASA"}),
    ) == "spoken text contains an all-caps lexical word"
    assert lane._spoken_error("Iona, listen", "Iona")
    assert lane._spoken_error("The signal arrives", "Iona") is None


def test_spoken_validator_allows_only_acronyms_grounded_in_accepted_fact_index():
    facts = lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id="F01",
            claim="RIO lets NASA researchers share robot software.",
            source_spans=[lane.SourceSpanV4(
                field="full_text", start=0, end=3, quote="RIO",
            )],
        )],
        entities=[], numbers=[], tone="technical", payload_sha256="0" * 64,
    )
    allowed = lane._source_grounded_all_caps(facts)

    assert allowed == frozenset({"RIO"})
    assert lane._spoken_error("RIO can move between robots.", allowed_all_caps=allowed) is None
    assert lane._spoken_error(
        "RIO must STOP now.", allowed_all_caps=allowed,
    ) == "spoken text contains an all-caps lexical word"


def test_allowed_spoken_acronyms_include_only_bounded_short_cast_role_tokens():
    cast = lane.CastPlanV4(cast=[
        lane.CastPlanRowV4(
            char_id="announcer", name="ANNOUNCER", character_description="Witness.",
            gender="neutral", role_in_conflict="Reports the CEO decision.", voice_slot="announcer",
        ),
        lane.CastPlanRowV4(
            char_id="c01", name="Iona", character_description="Observer.",
            gender="female", role_in_conflict="Challenges STOP signs.", voice_slot="c01",
        ),
    ])
    allowed = lane._allowed_spoken_all_caps(None, cast)
    assert allowed == frozenset({"CEO"})
    assert lane._spoken_error("The CEO decides.", allowed_all_caps=allowed) is None
    assert lane._spoken_error("The RUN ends.", allowed_all_caps=allowed) == "spoken text contains an all-caps lexical word"


def test_dramatic_question_repair_trims_overlong_fields_at_word_boundary():
    raw = json.dumps({
        "question": "Should park manager Lena approve drone flights over the river, knowing the science makes tracking possible but the drones may disturb nesting birds? " * 2,
        "consequence": "Approval yields precise data while denial avoids disturbance but misses a chance to understand the phenomenon. " * 2,
        "ending_direction": "The choice remains unresolved as the signal fades into the night. " * 3,
    })
    repaired = lane.repair_dramatic_question_metadata(raw)
    assert repaired is not None
    assert len(repaired.question) <= 160
    assert len(repaired.consequence) <= 160
    assert len(repaired.ending_direction) <= 120
    assert not repaired.question.endswith(" ")


def test_fact_index_provenance_rejects_out_of_range_spans_and_wrong_a0_digest():
    payload = {"headline": "literal evidence"}
    digest = lane._digest(payload)
    valid = lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id="F01", claim="literal evidence",
            source_spans=[lane.SourceSpanV4(
                field="headline", start=0, end=7, quote="literal",
            )],
        )],
        entities=[], numbers=[], tone="cautious", payload_sha256=digest,
    )
    assert lane._validate_fact_index(
        valid, payload, expected_payload_sha256=digest,
    ) is None
    assert lane._validate_fact_index(
        valid.model_copy(update={"payload_sha256": "0" * 64}), payload,
        expected_payload_sha256=digest,
    ) == "fact index payload_sha256 does not match the accepted A0 digest"
    out_of_range = valid.model_copy(update={"facts": [lane.FactV4(
        fact_id="F01", claim="literal evidence",
        source_spans=[lane.SourceSpanV4(
            field="headline", start=8, end=999, quote="evidence",
        )],
    )]})
    assert "non-literal source span" in lane._validate_fact_index(
        out_of_range, payload, expected_payload_sha256=digest,
    )


def test_provenance_prompt_context_guard_fails_before_generation(monkeypatch):
    from nodes import OTR_LedgerScriptWriter as writer
    from nodes import _otr_loader_backends as loader_backends

    class _Tensor:
        shape = (1, 65)

    class _Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=_Tensor())

        def to(self, _device):
            return self

    class _Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return "prompt"

        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return _Inputs()

    class _Model:
        device = "cpu"

        def generate(self, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("prompt guard must stop before model.generate")

    monkeypatch.setattr(
        loader_backends, "tokenizer_supports_system_role", lambda _tokenizer: True,
    )
    generate = writer._build_truncating_generate_fn({
        "model": _Model(), "tokenizer": _Tokenizer(), "context_cap": 10,
    })

    with pytest.raises(writer.PromptContextOverflowError, match="requires 65 input tokens"):
        generate(
            lane._PromptMustFitMessages([{"role": "user", "content": "A0"}]),
            temperature=.20, max_new_tokens=2,
        )


def test_advisory_centers_do_not_require_requested_count():
    plan = lane.make_advisory_word_blueprint(719, ["b001", "b002", "b003"])
    assert plan.advisory_total_center == 719
    assert sum(x.advisory_word_center for x in plan.per_beat) == 719
    assert [row.beat_id for row in plan.per_beat] == ["b001", "b002", "b003"]


def test_score_contract_closes_advisory_beats_lines_and_cue_anchors_before_p5():
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    assert lane._validate_radio_score_graph(score, advisory) is None

    missing_beat = score.model_dump(mode="json")
    missing_beat["scenes"][0]["beats"].pop()
    assert lane._validate_radio_score_graph(
        lane.RadioScoreV4.model_validate(missing_beat), advisory,
    ) == "score beat IDs do not exactly match the locked advisory order"

    noncontiguous_line = score.model_dump(mode="json")
    noncontiguous_line["scenes"][0]["beats"][2]["line_ids"] = ["l005"]
    assert lane._validate_radio_score_graph(
        lane.RadioScoreV4.model_validate(noncontiguous_line), advisory,
    ) == "score line IDs must be contiguous canonical IDs in assembly order"

    orphaned_cue = score.model_dump(mode="json")
    orphaned_cue["music_cues"][0]["anchor_line_id"] = "l006"
    assert lane._validate_radio_score_graph(
        lane.RadioScoreV4.model_validate(orphaned_cue), advisory,
    ) == "music cue 'music_open' anchors an unknown score line"


def test_radio_score_draft_surface_is_finite_before_p3_reserves_output_capacity():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    draft = _draft_for_advisory(advisory)
    receipt = lane._radio_score_draft_surface_receipt()

    assert receipt["schema"] == "RadioScoreDraftV4"
    assert receipt["full_result_json_schema_in_prompt"] is False
    assert receipt["max_scenes"] == 3
    assert receipt["max_shots_per_scene"] == 2
    assert receipt["max_beats_per_scene"] == 4
    assert receipt["max_total_beats"] == 12
    assert receipt["max_line_count_per_beat"] == 2
    assert receipt["max_new_tokens"] == lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
    assert receipt["input_token_reservation"] + receipt["max_new_tokens"] == 8192

    overlong_title = draft.model_dump(mode="json")
    overlong_title["title"] = "x" * 65
    with pytest.raises(ValidationError):
        lane.RadioScoreDraftV4.model_validate(overlong_title)

    overlong_visual_prompt = draft.model_dump(mode="json")
    overlong_visual_prompt["scenes"][0]["shots"][0]["visual_prompt"] = "x" * 121
    with pytest.raises(ValidationError):
        lane.RadioScoreDraftV4.model_validate(overlong_visual_prompt)

    invalid_local_anchor = draft.model_dump(mode="json")
    invalid_local_anchor["music_cues"][0]["anchor_line_index"] = 2
    with pytest.raises(ValidationError):
        lane.RadioScoreDraftV4.model_validate(invalid_local_anchor)

    too_many_scenes = draft.model_dump(mode="json")
    too_many_scenes["scenes"] *= 4
    with pytest.raises(ValidationError):
        lane.RadioScoreDraftV4.model_validate(too_many_scenes)

    with pytest.raises(ValidationError):
        lane.AdvisoryWordPlanV4.model_validate({
            "advisory_total_center": 120,
            "per_beat": [
                {"beat_id": f"b{index:03d}", "advisory_word_center": 10}
                for index in range(13)
            ],
        })


def test_max_width_p3_draft_envelopes_fit_the_local_gemma_context():
    """Measure the real draft/base/restart/repair/rewrite chat envelopes."""
    if not _GEMMA_E4B_TOKENIZER_SNAPSHOT.is_dir():
        pytest.skip("local Gemma E4B tokenizer snapshot is unavailable")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        _GEMMA_E4B_TOKENIZER_SNAPSHOT, local_files_only=True,
    )
    advisory = lane.make_advisory_word_blueprint(
        120, [f"b{index:03d}" for index in range(12)],
    )
    cast = _metadata_repair_cast()
    fact_index = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory, max_width=True)
    draft_json = json.dumps(
        draft.model_dump(mode="json"),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    invalid = draft.model_dump(mode="json")
    invalid["music_cues"][1]["cue_id"] = "music_open"
    invalid_json = json.dumps(
        invalid, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    pack = routing.resolve_story_pack("scifi_codex")
    context = _draft_context(advisory, cast, fact_index)
    reservation = lane._radio_score_draft_surface_receipt()

    def run_draft_call(
        pass_id: str,
        responses: list[str],
        *,
        expected_signature: tuple[object, ...] | None = None,
    ) -> list[object]:
        calls: list[object] = []

        def slot(messages, **_kwargs):
            calls.append(messages)
            return responses.pop(0)

        artifact_inputs = dict(context)
        if pass_id == "P3_rewrite":
            artifact_inputs["previous_draft"] = draft.model_dump(mode="json")
            artifact_inputs["review"] = {
                "verdict": "rewrite",
                "issues": [_bounded_prose(120, "issue")],
                "rationale": _bounded_prose(240, "rationale"),
            }
        result = lane._call_radio_score_draft(
            pass_id=pass_id,
            slot_fn=slot,
            pack=pack,
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=artifact_inputs,
            advisory=advisory,
            cast=cast,
            fact_index=fact_index,
            base_temperature=.72 if pass_id == "P3" else .55,
            structural_retry_temperature=.32 if pass_id == "P3" else .20,
            max_new_tokens=reservation["max_new_tokens"],
            call_journal={},
            expected_signature=expected_signature,
        )
        assert isinstance(result, lane.RadioScoreV4)
        return calls

    incomplete = '{"title":"unfinished"'
    base_and_restart = run_draft_call("P3", [incomplete, incomplete, draft_json])
    semantic_repair = run_draft_call("P3", [invalid_json, draft_json])
    rewrite_signature = lane._radio_score_draft_structure_signature(draft)
    rewrite_base = run_draft_call("P3_rewrite", [draft_json], expected_signature=rewrite_signature)
    rewrite_repair = run_draft_call(
        "P3_rewrite", [invalid_json, draft_json], expected_signature=rewrite_signature,
    )

    assert len(base_and_restart) == 3
    assert incomplete not in base_and_restart[2][1]["content"]
    assert "<draft_context>" in base_and_restart[2][1]["content"]
    assert len(semantic_repair) == 2
    assert "<failed_radio_score_draft>" in semantic_repair[1][1]["content"]
    assert invalid_json in semantic_repair[1][1]["content"]
    assert len(rewrite_base) == 1
    assert len(rewrite_repair) == 2
    assert "<failed_radio_score_draft>" in rewrite_repair[1][1]["content"]
    all_calls = [
        *base_and_restart,
        *semantic_repair,
        *rewrite_base,
        *rewrite_repair,
    ]
    assert all(
        "result_json_schema" not in messages[1]["content"]
        for messages in all_calls
    )
    assert all("RadioScoreDraftV4" in messages[0]["content"] for messages in all_calls)

    def token_count(messages) -> int:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    draft_tokens = len(tokenizer(draft_json, add_special_tokens=False)["input_ids"])
    prompt_tokens = [token_count(messages) for messages in all_calls]
    assert draft_tokens == 1576
    assert reservation["max_new_tokens"] == (
        draft_tokens + max(128, math.ceil(draft_tokens * .15)) + 16
    ) == 1829
    assert all(
        prompt_tokens_for_call + reservation["max_new_tokens"] <= 8192
        for prompt_tokens_for_call in prompt_tokens
    ), f"draft P3 prompt tokens {prompt_tokens} exceed the 8192 context cap"

    patch_raw = copy.deepcopy(draft.model_dump(mode="json"))
    patch_loc_caps = [
        (("title",), 64),
        (("premise",), 240),
        (("setting",), 80),
        (("scenes", 0, "env"), 56),
        (("scenes", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "visual_prompt"), 120),
        (("scenes", 0, "shots", 1, "description"), 144),
        (("scenes", 0, "shots", 1, "visual_prompt"), 120),
        (("scenes", 0, "beats", 0, "intent"), 64),
        (("scenes", 0, "beats", 0, "arc_phase"), 28),
        (("scenes", 0, "beats", 1, "intent"), 64),
    ]
    for loc, cap in patch_loc_caps:
        lane._p3_text_patch_set_at(
            patch_raw, loc, "x" * lane._P3_TEXT_PATCH_MAX_SOURCE_CHARS,
        )
    with pytest.raises(ValidationError) as patch_error:
        lane.RadioScoreDraftV4.model_validate(patch_raw)
    patch_targets = lane._derive_p3_text_patch_targets(
        patch_raw, patch_error.value,
    )
    assert patch_targets is not None and len(patch_targets) == 12
    patch_messages = lane._p3_text_patch_messages(pack, patch_targets)
    patch_response = json.dumps(
        {
            "replacements": [
                {
                    "path": target.path,
                    "replacement_text": _bounded_prose(
                        target.max_chars, f"patch{index}",
                    ),
                }
                for index, target in enumerate(patch_targets)
            ],
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    patch_prompt_tokens = token_count(patch_messages)
    patch_response_tokens = len(
        tokenizer(patch_response, add_special_tokens=False)["input_ids"]
    )
    patch_budget = lane._p3_text_patch_output_budget(patch_targets)
    # NOTE what `patch_response` above actually is: a COMPACT object
    # (separators=(",", ":"), no spaces) -- i.e. the shape the local
    # grammar-constrained transport emits. Measuring only that shape is what
    # made the old flat 1024 look safe. A free-form remote model pretty-prints,
    # wraps the object in a ```json fence, and -- if it is a reasoning model --
    # spends part of the SAME max_tokens budget thinking before it writes a
    # character. Hence the transport headroom on top of the measured artifact.
    assert patch_response_tokens <= patch_budget
    assert patch_budget >= patch_response_tokens + (
        lane._P3_TEXT_PATCH_TRANSPORT_HEADROOM_TOKENS // 2
    ), "the budget must leave a free-form/reasoning transport real room, not a hair"
    assert (
        patch_prompt_tokens + patch_budget <= 8192
    ), f"P3 text patch prompt tokens {patch_prompt_tokens} exceed the 8192 context cap"


class TestP3TextPatchOutputBudget:
    """The patch budget is ARITHMETIC over the real targets, not a constant.

    Live failure that forced this (30-word `scifi_codex` Aion leg): P3 exhausted
    with `PostValidationError: draft.text_patch at root: patch_json` -- the
    free-form reasoning model's patch came back cut off mid-JSON, so
    `parse_first_json_object` raised, so the lane's ONLY authored-text repair
    route died, and it took a complete episode with it.
    """

    @staticmethod
    def _target(path: str, max_chars: int):
        return lane._P3TextPatchTarget(
            loc=tuple(path.split(".")), path=path, max_chars=max_chars,
            current_text="x" * max_chars,
        )

    def test_a_small_patch_keeps_the_local_floor(self):
        """One short target must not shrink the budget below the proven floor.

        This is what keeps the LOCAL grammar-constrained path byte-identical.
        """
        targets = (self._target("title", 64),)
        assert (
            lane._p3_text_patch_output_budget(targets)
            == lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
        )

    def test_a_max_width_patch_outgrows_the_old_flat_constant(self):
        """12 targets at their caps need MORE than the 1024 that used to be all
        they could ever get -- which is exactly how the JSON got truncated."""
        targets = tuple(
            self._target(f"scenes.{i}.shots.1.visual_prompt", 120)
            for i in range(lane._P3_TEXT_PATCH_MAX_TARGETS)
        )
        budget = lane._p3_text_patch_output_budget(targets)
        assert budget > lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
        # The artifact's own declared size, plus room for a fence + hidden
        # reasoning tokens -- never less than the raw JSON it must emit.
        raw_json_chars = sum(
            len(t.path) + t.max_chars + lane._P3_TEXT_PATCH_ROW_JSON_CHARS
            for t in targets
        ) + lane._P3_TEXT_PATCH_ROOT_JSON_CHARS
        assert budget >= raw_json_chars // lane._P3_TEXT_PATCH_CHARS_PER_TOKEN

    def test_the_budget_grows_with_the_patch(self):
        small = (self._target("title", 64),)
        big = tuple(
            self._target(f"scenes.{i}.shots.1.visual_prompt", 120)
            for i in range(lane._P3_TEXT_PATCH_MAX_TARGETS)
        )
        assert (
            lane._p3_text_patch_output_budget(big)
            > lane._p3_text_patch_output_budget(small)
        )


def test_draft_compiler_derives_only_mechanical_score_metadata():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    fact_index = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    before = draft.model_dump(mode="json")

    score = lane.compile_radio_score_draft(draft, advisory, cast, fact_index)

    assert score.advisory_word_plan.model_dump(mode="json") == advisory.model_dump(mode="json")
    assert [beat.beat_id for scene in score.scenes for beat in scene.beats] == [
        "b000", "b001", "b002",
    ]
    assert [beat.order for scene in score.scenes for beat in scene.beats] == [1, 2, 3]
    assert [beat.line_ids for scene in score.scenes for beat in scene.beats] == [
        ["l001"], ["l002"], ["l003"],
    ]
    assert [beat.speaker for scene in score.scenes for beat in scene.beats] == [
        "ANNOUNCER", "Iona", "ANNOUNCER",
    ]
    cue = score.music_cues[0]
    assert (cue.placement, cue.anchor_beat_id, cue.anchor_line_id) == (
        "open", "b000", "l001",
    )
    assert lane._validate_radio_score_graph(score, advisory) is None
    assert draft.model_dump(mode="json") == before


def test_draft_compiler_rejects_unowned_or_invalid_runtime_decisions():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    baseline = _draft_for_advisory(advisory).model_dump(mode="json")

    cases = []
    # NOTE: beat_count (a short draft) and cast_coverage (an uncovered planned
    # cast member) are NO LONGER rejections -- they are reconciled/advisory per
    # Gate 3 (counts never gate production). Their cases were removed; the checks
    # below are the integrity gates that stay fatal (dangling references, dup IDs,
    # unknown facts/cast, an unowned declared shot, out-of-range cue anchors).
    one_shot = copy.deepcopy(baseline)
    one_shot["scenes"][0]["shots"].pop()
    one_shot["scenes"][0]["beats"][1]["shot_index"] = 1
    cases.append(("shot_index", one_shot))
    unused_shot = copy.deepcopy(baseline)
    for beat in unused_shot["scenes"][0]["beats"]:
        beat["shot_index"] = 0
    cases.append(("unused_shot", unused_shot))
    unknown_cast = copy.deepcopy(baseline)
    unknown_cast["scenes"][0]["beats"][0]["char_id"] = "c02"
    cases.append(("cast_id", unknown_cast))
    unknown_fact = copy.deepcopy(baseline)
    unknown_fact["scenes"][0]["beats"][0]["fact_ids"] = ["F02"]
    cases.append(("fact_id", unknown_fact))
    duplicate_fact = copy.deepcopy(baseline)
    duplicate_fact["scenes"][0]["beats"][0]["fact_ids"] = ["F01", "F01"]
    cases.append(("fact_id", duplicate_fact))
    duplicate_cue = copy.deepcopy(baseline)
    duplicate_cue["music_cues"].append(copy.deepcopy(duplicate_cue["music_cues"][0]))
    cases.append(("cue_id", duplicate_cue))
    # NOTE: an out-of-range cue anchor is no longer a rejection -- it is a
    # MECHANICAL routing index that Python CLAMPS deterministically to a valid
    # beat/line per Gate 3 (a stale index after a reconciled short draft must not
    # gate production). Its former reject case was removed; see the dedicated
    # clamp test below.

    for expected_code, raw_draft in cases:
        draft = lane.RadioScoreDraftV4.model_validate(raw_draft)
        with pytest.raises(lane.RadioScoreDraftCompileError) as exc_info:
            lane.compile_radio_score_draft(draft, advisory, cast, facts)
        assert exc_info.value.code == expected_code

    duplicate_advisory = advisory.model_copy(update={
        "per_beat": [advisory.per_beat[0], advisory.per_beat[0], advisory.per_beat[2]],
    })
    with pytest.raises(lane.RadioScoreDraftCompileError) as exc_info:
        lane.compile_radio_score_draft(
            lane.RadioScoreDraftV4.model_validate(baseline),
            duplicate_advisory,
            cast,
            facts,
        )
    assert exc_info.value.code == "invalid_advisory"

def test_p3_semantic_repair_uses_minified_draft_and_bounded_receipts():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    bad = draft.model_dump(mode="json")
    bad["music_cues"].append(copy.deepcopy(bad["music_cues"][0]))
    responses = [json.dumps(bad), json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    pack = routing.resolve_story_pack("scifi_codex")
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    repair_system = calls[1]["messages"][0]["content"]
    repair_user = calls[1]["messages"][1]["content"]
    assert "RadioScoreDraftV4" in repair_system
    assert "<failed_radio_score_draft>" in repair_user
    assert "<trusted_draft_context>" in repair_user
    assert "original_request" not in repair_user
    assert '"result_json_schema"' not in repair_user
    attempts = journal["calls"][0]["attempts"]
    assert attempts[0]["compiler_status"] == "cue_id"
    assert attempts[0]["draft_status"] == "compiler_rejected"
    assert attempts[1]["compiler_status"] == "accepted"
    assert journal["calls"][0]["accepted_transport"]["schema"] == "RadioScoreDraftV4"


def test_p3_base_and_repair_bind_locked_total_to_per_scene_cap():
    advisory = lane.make_advisory_word_blueprint(
        42, [f"b{index:03d}" for index in range(6)],
    )
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    valid = _draft_for_advisory(advisory)
    wrong_total = valid.model_dump(mode="json")
    # A per-scene-cap violation (all 6 beats in a single scene, above the
    # max-4-per-scene draft schema) forces the structural retry. NOTE: the old
    # truncated-total variant (scenes[:1]) is now RECONCILED by the compiler
    # (Gate 3), not rejected, so it would no longer trigger a repair.
    all_beats = [beat for scene in wrong_total["scenes"] for beat in scene["beats"]]
    wrong_total["scenes"] = [wrong_total["scenes"][0]]
    wrong_total["scenes"][0]["beats"] = all_beats
    responses = [
        json.dumps(wrong_total),
        json.dumps(valid.model_dump(mode="json")),
    ]
    calls: list[dict[str, object]] = []
    pack = routing.resolve_story_pack("scifi_codex")

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={},
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    for call in calls:
        system = call["messages"][0]["content"]
        assert "contains exactly 6 beat rows" in system
        assert "flattened total" in system
        assert "exactly 6" in system
        assert "Each individual scene may contain at most 4 beats" in system
        assert "across at least 2 scene(s)" in system


@pytest.mark.parametrize(
    ("loc", "cap"),
    [
        (("title",), 64),
        (("premise",), 240),
        (("setting",), 80),
        (("scenes", 0, "env"), 56),
        (("scenes", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "description"), 144),
        (("scenes", 0, "shots", 1, "visual_prompt"), 120),
        (("scenes", 0, "beats", 0, "intent"), 64),
        (("scenes", 0, "beats", 1, "arc_phase"), 28),
        (("music_cues", 0, "description"), 80),
        (("music_cues", 0, "generation_prompt"), 120),
    ],
)
def test_p3_text_patch_gate_covers_each_author_owned_leaf(loc, cap):
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    raw = _draft_for_advisory(advisory).model_dump(mode="json")
    lane._p3_text_patch_set_at(raw, loc, "x" * (cap + 1))

    with pytest.raises(ValidationError) as exc_info:
        lane.RadioScoreDraftV4.model_validate(raw)

    targets = lane._derive_p3_text_patch_targets(raw, exc_info.value)
    assert targets is not None
    assert len(targets) == 1
    assert targets[0].loc == loc
    assert targets[0].max_chars == cap
    assert targets[0].current_text == "x" * (cap + 1)


def test_p3_text_patch_refuses_mixed_or_unbounded_source_errors():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    raw = _draft_for_advisory(advisory).model_dump(mode="json")
    raw["title"] = "x" * 257
    with pytest.raises(ValidationError) as source_exc:
        lane.RadioScoreDraftV4.model_validate(raw)
    assert lane._derive_p3_text_patch_targets(raw, source_exc.value) is None

    mixed = _draft_for_advisory(advisory).model_dump(mode="json")
    mixed["title"] = "x" * 65
    mixed["music_cues"][0]["cue_id"] = "not_a_cue"
    with pytest.raises(ValidationError) as mixed_exc:
        lane.RadioScoreDraftV4.model_validate(mixed)
    assert lane._derive_p3_text_patch_targets(mixed, mixed_exc.value) is None


def test_p3_local_text_patch_repairs_one_leaf_with_one_bounded_call():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    invalid["scenes"][0]["description"] = "x" * 145
    replacement = "A receiver hums under a cold sky."
    responses = [
        json.dumps(invalid),
        json.dumps({"replacements": [{
            "path": "scenes.0.description",
            "replacement_text": replacement,
        }]}),
    ]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "exact_local"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert calls[1]["temperature"] == pytest.approx(lane.REPAIR_TEMPERATURE)
    assert calls[1]["max_new_tokens"] >= lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
    assert getattr(calls[1]["messages"], "_otr_prompt_must_fit", False) is True
    assert "replacements" in calls[1]["messages"][0]["content"]
    assert "source_to_shorten" in calls[1]["messages"][1]["content"]
    patch_request = json.loads(calls[1]["messages"][1]["content"])
    assert set(patch_request) == {"rewrite_tasks"}
    assert patch_request["rewrite_tasks"][0]["max_chars"] == 108
    assert "original_text" not in patch_request["rewrite_tasks"][0]
    assert "Never copy source_to_shorten unchanged" in calls[1]["messages"][0]["content"]

    attempts = journal["calls"][0]["attempts"]
    assert len(attempts) == 2
    assert attempts[1]["repair_kind"] == "p3_authored_text_patch"
    assert attempts[1]["patch_transport"] == "exact_local"
    assert attempts[1]["patch_status"] == "accepted"
    assert attempts[1]["parse_status"] == "decoded"
    assert attempts[1]["schema_status"] == "accepted"
    assert attempts[1]["draft_status"] == "accepted"
    assert attempts[1]["patch_targets"] == [
        {"path": "scenes.0.description", "max_chars": 144},
    ]
    assert "x" * 145 not in json.dumps(attempts, ensure_ascii=False)
    expected = draft.model_dump(mode="json")
    expected["scenes"][0]["description"] = replacement
    expected_wire = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    transport = journal["calls"][0]["accepted_transport"]
    assert transport["schema"] == "RadioScoreDraftV4"
    assert transport["sha256"] == hashlib.sha256(expected_wire.encode()).hexdigest()
    assert result.title == expected["title"]
    assert result.scenes[0].description == replacement
    assert result.scenes[0].shots[0].visual_prompt == (
        expected["scenes"][0]["shots"][0]["visual_prompt"]
    )


def test_p3_rewrite_local_text_patch_preserves_locked_structure():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    score = lane.compile_radio_score_draft(draft, advisory, cast, facts)
    projected = lane.project_radio_score_to_draft(score)
    invalid = projected.model_dump(mode="json")
    invalid["scenes"][0]["beats"][0]["intent"] = "x" * 65
    responses = [
        json.dumps(invalid),
        json.dumps({"replacements": [{
            "path": "scenes.0.beats.0.intent",
            "replacement_text": "A cautious choice sharpens.",
        }]}),
    ]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "exact_local"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3_rewrite", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={
            **_draft_context(advisory, cast, facts),
            "previous_draft": projected.model_dump(mode="json"),
            "review": {"verdict": "rewrite", "issues": ["Tighten the turn."]},
        },
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.55, structural_retry_temperature=.20,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={},
        expected_signature=lane._radio_score_draft_structure_signature(projected),
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert calls[1]["max_new_tokens"] >= lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
    assert getattr(calls[1]["messages"], "_otr_prompt_must_fit", False) is True


def test_p3_text_patch_preflight_falls_back_for_hidden_compiler_defect(caplog):
    caplog.set_level("INFO")
    advisory = lane.make_advisory_word_blueprint(
        60, [f"b{index:03d}" for index in range(6)],
    )
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    invalid["scenes"][0]["description"] = "x" * 145
    invalid["music_cues"][1]["cue_id"] = "music_open"
    responses = [json.dumps(invalid), json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "exact_local"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert calls[1]["max_new_tokens"] == lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
    assert "<failed_radio_score_draft>" in calls[1]["messages"][1]["content"]
    assert "repair_kind" not in journal["calls"][0]["attempts"][1]
    assert "text-patch preflight retained full repair:" in caplog.text
    assert "cue_id" in caplog.text


def test_p3_malformed_text_patch_fails_without_a_third_reroll():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    invalid = _draft_for_advisory(advisory).model_dump(mode="json")
    invalid["scenes"][0]["description"] = "x" * 145
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}
    responses = [json.dumps(invalid), "not JSON"]

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "exact_local"  # type: ignore[attr-defined]

    with pytest.raises(lane.CodexPassError):
        lane._call_radio_score_draft(
            pass_id="P3", slot_fn=slot_fn,
            pack=routing.resolve_story_pack("scifi_codex"),
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=_draft_context(advisory, cast, facts),
            advisory=advisory, cast=cast, fact_index=facts,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
            call_journal=journal,
        )

    assert len(calls) == 2
    attempt = journal["calls"][0]["attempts"][1]
    assert attempt["repair_kind"] == "p3_authored_text_patch"
    assert attempt["patch_status"] == "not_decoded"
    assert attempt["parse_status"] == "not_decoded"
    assert attempt["schema_status"] == "not_run"
    assert attempt["draft_status"] == "not_decoded"
    assert attempt["compiler_status"] == "not_run"
    assert attempt["graph_status"] == "not_run"
    assert "not JSON" not in json.dumps(attempt, ensure_ascii=False)


def test_p3_text_patch_contract_rejects_missing_duplicate_unknown_blank_and_overcap_rows():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    raw = _draft_for_advisory(advisory).model_dump(mode="json")
    raw["scenes"][0]["description"] = "x" * 145
    raw["scenes"][0]["beats"][0]["intent"] = "y" * 65
    with pytest.raises(ValidationError) as exc_info:
        lane.RadioScoreDraftV4.model_validate(raw)
    targets = lane._derive_p3_text_patch_targets(raw, exc_info.value)
    assert targets is not None and len(targets) == 2
    first, second = targets

    def patch(rows):
        return lane._RadioScoreDraftTextPatchV4.model_validate({"replacements": rows})

    rows_by_case = {
        "missing": [{
            "path": first.path,
            "replacement_text": "Short enough.",
        }],
        "duplicate": [
            {"path": first.path, "replacement_text": "Short enough."},
            {"path": first.path, "replacement_text": "Still short."},
        ],
        "unknown": [
            {"path": "scenes.9.description", "replacement_text": "Short enough."},
            {"path": second.path, "replacement_text": "Still short."},
        ],
        "blank": [
            {"path": first.path, "replacement_text": " "},
            {"path": second.path, "replacement_text": "Still short."},
        ],
        "overcap": [
            {"path": first.path, "replacement_text": "z" * (first.max_chars + 1)},
            {"path": second.path, "replacement_text": "Still short."},
        ],
    }
    for rows in rows_by_case.values():
        with pytest.raises(lane.PostValidationError):
            lane._merge_p3_text_patch(raw, targets, patch(rows))

    with pytest.raises(ValidationError):
        patch([])


def test_p3_text_patch_receipt_distinguishes_model_prose_over_schema_cap():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    invalid = _draft_for_advisory(advisory).model_dump(mode="json")
    invalid["premise"] = "x" * 241
    journal: dict[str, object] = {}
    responses = [
        json.dumps(invalid),
        json.dumps({"replacements": [{
            "path": "premise", "replacement_text": "y" * 241,
        }]}),
    ]

    def slot_fn(messages, **kwargs):
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "full_message_remote"  # type: ignore[attr-defined]
    with pytest.raises(lane.CodexPassError, match="replacement_over_schema_cap"):
        lane._call_radio_score_draft(
            pass_id="P3", slot_fn=slot_fn,
            pack=routing.resolve_story_pack("scifi_codex"),
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=_draft_context(advisory, cast, facts),
            advisory=advisory, cast=cast, fact_index=facts,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
            call_journal=journal,
        )

    attempt = journal["calls"][0]["attempts"][1]
    assert attempt["patch_error_code"] == "replacement_over_schema_cap"
    assert "y" * 20 not in json.dumps(attempt)


def test_p3_text_patch_rejects_a_resolved_artifact_wrapper_without_reroll():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    invalid = _draft_for_advisory(advisory).model_dump(mode="json")
    invalid["scenes"][0]["description"] = "x" * 145
    wrapped_patch = {
        "resolved_artifact": {
            "replacements": [{
                "path": "scenes.0.description",
                "replacement_text": "A compact receiver hums.",
            }],
        },
    }
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}
    responses = [json.dumps(invalid), json.dumps(wrapped_patch)]

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_p3_text_patch_transport = "exact_local"  # type: ignore[attr-defined]

    with pytest.raises(lane.CodexPassError):
        lane._call_radio_score_draft(
            pass_id="P3", slot_fn=slot_fn,
            pack=routing.resolve_story_pack("scifi_codex"),
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=_draft_context(advisory, cast, facts),
            advisory=advisory, cast=cast, fact_index=facts,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
            call_journal=journal,
        )

    assert len(calls) == 2
    attempt = journal["calls"][0]["attempts"][1]
    assert attempt["repair_kind"] == "p3_authored_text_patch"
    assert attempt["patch_status"] == "schema_rejected"
    assert attempt["parse_status"] == "decoded"
    assert attempt["schema_status"] == "rejected"
    assert attempt["draft_status"] == "schema_rejected"
    assert attempt["compiler_status"] == "not_run"


def test_p3_scheduler_openrouter_uses_bounded_patch_and_forwards_json_mode(monkeypatch):
    """The production scheduler must preserve remote P3 transport identity.

    A scheduler closure is the actual writer path, unlike an unwrapped test
    slot.  It must advertise OpenRouter before P3 chooses its repair ladder and
    relay the common json_object request mode to the lazily built backend.
    """
    from nodes import OTR_LedgerScriptWriter as writer
    from nodes import _otr_model_catalog as catalog
    from nodes import _otr_model_loader as model_loader

    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    invalid["scenes"][0]["description"] = "x" * 145
    replacement = "A compact receiver hums."
    responses = [json.dumps(invalid), json.dumps({"replacements": [{
        "path": "scenes.0.description",
        "replacement_text": replacement,
    }]})]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    monkeypatch.setattr(
        catalog, "_by_repo_id",
        lambda: {"openrouter:slot-a": SimpleNamespace(provider="openrouter")},
    )
    monkeypatch.setattr(
        model_loader,
        "request_slot",
        lambda slot, model_id, policy=None, load_config=None: {
            "provider": "openrouter",
        },
    )

    def fake_build(cache_entry, **_sampling):
        assert cache_entry["provider"] == "openrouter"

        def base(messages, *, temperature, max_new_tokens, stop=None,
                 response_format=None):
            calls.append({
                "messages": messages,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": stop,
                "response_format": response_format,
            })
            return responses.pop(0)

        return base

    monkeypatch.setattr(writer, "_build_truncating_generate_fn", fake_build)
    scheduler = writer._SlotScheduler(
        creative_id="openrouter:slot-a",
        technical_id="mistralai/Mistral-Nemo-Instruct-2407",
        top_p=.92, min_p=0.0, repetition_penalty=1.0,
    )
    slot_fn = scheduler.for_slot("creative")
    assert slot_fn._otr_openrouter is True  # type: ignore[attr-defined]
    assert slot_fn._otr_p3_text_patch_local is False  # type: ignore[attr-defined]
    assert slot_fn._otr_p3_text_patch_transport == "full_message_remote"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert calls[1]["max_new_tokens"] >= lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
    assert getattr(calls[1]["messages"], "_otr_prompt_must_fit", False) is True
    assert getattr(
        calls[1]["messages"], "_otr_strict_remote_output_budget", False,
    ) is True
    attempt = journal["calls"][0]["attempts"][1]
    assert attempt["repair_kind"] == "p3_authored_text_patch"
    assert attempt["patch_transport"] == "full_message_remote"
    assert attempt["patch_status"] == "accepted"
    assert result.scenes[0].description == replacement


def test_p3_openrouter_repairs_captured_ten_target_shape_with_json_mode():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    target_loc_caps = [
        (("title",), 64),
        (("premise",), 240),
        (("setting",), 80),
        (("scenes", 0, "env"), 56),
        (("scenes", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "visual_prompt"), 120),
        (("scenes", 0, "shots", 1, "description"), 144),
        (("scenes", 0, "shots", 1, "visual_prompt"), 120),
        (("scenes", 0, "beats", 0, "intent"), 64),
    ]
    for loc, cap in target_loc_caps:
        lane._p3_text_patch_set_at(invalid, loc, "x" * (cap + 1))
    patch_rows = [
        {
            "path": ".".join(str(item) for item in loc),
            "replacement_text": f"Short replacement {index}.",
        }
        for index, (loc, _cap) in enumerate(target_loc_caps)
    ]
    responses = [
        json.dumps(invalid),
        json.dumps({"replacements": patch_rows}),
    ]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_openrouter = True  # type: ignore[attr-defined]
    slot_fn._otr_response_format = None  # type: ignore[attr-defined]
    slot_fn._otr_p3_text_patch_transport = "full_message_remote"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert calls[1]["max_new_tokens"] >= lane._P3_TEXT_PATCH_MIN_OUTPUT_TOKENS
    attempt = journal["calls"][0]["attempts"][1]
    assert attempt["patch_transport"] == "full_message_remote"
    assert attempt["patch_status"] == "accepted"
    assert len(attempt["patch_targets"]) == 10
    assert {row["path"] for row in attempt["patch_targets"]} == {
        row["path"] for row in patch_rows
    }


def test_p3_openrouter_thirteen_targets_retain_full_repair():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    target_loc_caps = [
        (("title",), 64), (("premise",), 240), (("setting",), 80),
        (("scenes", 0, "env"), 56),
        (("scenes", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "description"), 144),
        (("scenes", 0, "shots", 0, "visual_prompt"), 120),
        (("scenes", 0, "shots", 1, "description"), 144),
        (("scenes", 0, "shots", 1, "visual_prompt"), 120),
        (("scenes", 0, "beats", 0, "intent"), 64),
        (("scenes", 0, "beats", 0, "arc_phase"), 28),
        (("scenes", 0, "beats", 1, "intent"), 64),
        (("scenes", 0, "beats", 1, "arc_phase"), 28),
    ]
    for loc, cap in target_loc_caps:
        lane._p3_text_patch_set_at(invalid, loc, "x" * (cap + 1))
    responses = [json.dumps(invalid), json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    slot_fn._otr_openrouter = True  # type: ignore[attr-defined]
    slot_fn._otr_response_format = None  # type: ignore[attr-defined]
    slot_fn._otr_p3_text_patch_transport = "full_message_remote"  # type: ignore[attr-defined]

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert calls[1]["max_new_tokens"] == lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
    assert "<failed_radio_score_draft>" in calls[1]["messages"][1]["content"]
    assert "repair_kind" not in journal["calls"][0]["attempts"][1]


def test_p3_compact_contract_names_nested_literal_values_on_base_and_repair():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    invalid["scenes"][0]["beats"][0]["arc_phase"] = 10
    invalid["music_cues"][0]["cue_id"] = "TensionBuild"
    responses = [json.dumps(invalid), json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn, pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={},
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    for call in calls:
        system = call["messages"][0]["content"]
        assert "safe ceilings: title <=48; premise <=108; setting <=60" in system
        assert "env <=42; description <=54" in system
        assert "description <=54 and visual_prompt <=90" in system
        assert "intent <=48; arc_phase <=21" in system
        assert "description <=60; generation_prompt <=90" in system
        assert "arc_phase is a narrative JSON string" in system
        assert "never a number, word count, advisory center, or percentage" in system
        assert "cue_id MUST be exactly one of music_open, music_inter, music_close" in system
        assert "never a descriptive music name" in system


def test_p3_surface_instruction_exposes_hidden_compiler_gates():
    # kibitz r2 (2026-07-17): the compiler enforces unused_shot / cast_coverage /
    # cue_id-unique / cue_anchor<beat_count, none of which were model-visible.
    surface = lane._RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION
    assert "every declared shot MUST be referenced by at least one beat" in surface
    assert "every accepted cast ID MUST own at least one beat, announcer included" in surface
    assert "each cue_id MUST appear at most once" in surface
    assert "anchor_beat_index MUST be less than the total number of beats" in surface


def test_p3_topology_pins_fully_constrained_twelve_beat_layout():
    def topo(n):
        advisory = lane.make_advisory_word_blueprint(
            30, [f"b{i:03d}" for i in range(n)],
        )
        return lane._radio_score_draft_topology_instruction(
            {"advisory_word_plan": advisory.model_dump(mode="json")},
        )

    twelve = topo(12)
    assert "fully constrained" in twelve
    assert "emit exactly 3 scenes, each holding exactly 4 beats" in twelve
    # 6 and 9 beats have layout freedom -> no tight clause.
    for n in (6, 9):
        assert "fully constrained" not in topo(n)


def test_scifi_codex_v4_p3_prompt_contract():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    responses = [json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex_v4"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={},
    )

    assert isinstance(result, lane.RadioScoreV4)
    system = calls[0]["messages"][0]["content"]
    # v4 beat-budget harmonization present; additive-beat wording + the reverted
    # cap-list both absent (the caps live in the tighter surface ceilings only).
    assert "produce exactly as many beats as the advisory plan lists" in system
    assert "include one mandatory COST beat" not in system
    # BUG B / PBUG-20260713-04: the base seam must NOT restate the true 144 cap
    # (exposing the rejection edge makes the model aim at it and cross it -- the
    # cross-check anti-pattern). The conservative target lives in the surface
    # instruction (premise <=108) + the text-patch 75% margin; base seam stays cap-free.
    assert "premise at most 144" not in system
    assert "hard length budget" not in system
    # the newly model-visible coverage gates ride the shared surface instruction.
    assert "every declared shot MUST be referenced by at least one beat" in system
    assert "every accepted cast ID MUST own at least one beat" in system


def test_p3_beat_count_mismatch_is_reconciled_not_rejected():
    # Gate 3 (SOURCE_BANK_PREFLIGHT): "no model-produced or unused count field
    # can gate production". A beat-count mismatch is RECONCILED -- the advisory
    # word plan is rebuilt onto the draft's ACTUAL beat count -- never raised. A
    # 3-beat plan whose draft carries 2 beats compiles to a 2-beat score.
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    baseline = _draft_for_advisory(advisory).model_dump(mode="json")
    baseline["scenes"][0]["beats"].pop()
    draft = lane.RadioScoreDraftV4.model_validate(baseline)
    score = lane.compile_radio_score_draft(draft, advisory, cast, facts)
    assert isinstance(score, lane.RadioScoreV4)
    observed = [beat for scene in score.scenes for beat in scene.beats]
    assert len(observed) == 2
    assert len(score.advisory_word_plan.per_beat) == 2


def test_cue_anchor_out_of_range_is_clamped_not_rejected():
    # Gate 3: a stale cue anchor (index past the beats/lines a reconciled short
    # draft actually carries) is MECHANICAL routing metadata that Python clamps
    # to the last valid beat/line -- never a fatal gate. The cue still lands on a
    # real beat/line, so the executable graph stays closed.
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    baseline = _draft_for_advisory(advisory).model_dump(mode="json")
    baseline["music_cues"][0]["anchor_beat_index"] = 5  # past the 3 beats
    baseline["music_cues"][0]["anchor_line_index"] = 1  # past this beat's 1 line
    draft = lane.RadioScoreDraftV4.model_validate(baseline)
    score = lane.compile_radio_score_draft(draft, advisory, cast, facts)
    assert isinstance(score, lane.RadioScoreV4)
    assert lane._validate_radio_score_graph(
        score, score.advisory_word_plan) is None


def test_p0_source_spans_survive_whitespace_polluted_source():
    # PBUG-20260717: dirty RSS whitespace is normalized at admission so literal
    # source spans index clean word boundaries (the model can reproduce them).
    dirty = "\n\t\t\t\t\t\t\t\tThe Growing Crescent of Mars as NASA approached the planet."
    assert lane._normalize_span_source_text(dirty) == (
        "The Growing Crescent of Mars as NASA approached the planet."
    )
    # idempotent on already-clean text -> no accepted offset shifts vs a control.
    clean_ctrl = "The crescent of Mars grew as the spacecraft approached."
    assert lane._normalize_span_source_text(clean_ctrl) == clean_ctrl

    payload = {
        "headline": "Mars crescent grows in new images",
        "summary": "\t\tThe crescent of Mars grew as the craft approached the planet.",
        "full_text": dirty,
        "source": "Example News",
        "date": "2026-07-17",
        "link": "https://example.test/mars",
        "seed_text": (
            "Mars crescent grows in new images.\n\tThe crescent of Mars grew "
            "as the craft approached the planet."
        ),
    }
    env, _steer = lane.validate_payload_envelope(
        payload, {"seed_source": "custom_premise", "target_words": 30},
    )
    a0 = env.payload.model_dump(mode="json")
    for field in ("headline", "summary", "full_text", "seed_text"):
        assert "\n" not in a0[field]
        assert "\t" not in a0[field]
        assert "  " not in a0[field]
        assert a0[field] == a0[field].strip()
    # a literal span into the CLEANED full_text validates first-try.
    full = a0["full_text"]
    start = full.index("Growing")
    end = start + len("Growing Crescent")
    span = lane.SourceSpanV4(
        field="full_text", start=start, end=end, quote=full[start:end],
    )
    assert lane._span_ok(span, a0) is True


def test_p3_two_decode_failures_restart_only_from_trusted_draft_context():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    incomplete = '{"title":"partial-model-output"'
    responses = [incomplete, incomplete, json.dumps(draft.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn, pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 3
    assert calls[1]["temperature"] == pytest.approx(.32)
    restart_user = calls[2]["messages"][1]["content"]
    assert "<draft_context>" in restart_user
    assert incomplete not in restart_user
    assert "<failed_radio_score_draft>" not in restart_user
    attempts = journal["calls"][0]["attempts"]
    assert [attempt["parse_status"] for attempt in attempts] == [
        "not_decoded", "not_decoded", "decoded",
    ]


def test_project_compile_round_trip_preserves_the_rewrite_structure():
    advisory = lane.make_advisory_word_blueprint(
        120, [f"b{index:03d}" for index in range(12)],
    )
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory, max_width=True)
    score = lane.compile_radio_score_draft(draft, advisory, cast, facts)

    projected = lane.project_radio_score_to_draft(score)
    rebuilt = lane.compile_radio_score_draft(projected, advisory, cast, facts)

    assert lane._radio_score_draft_structure_signature(projected) == (
        lane._radio_score_draft_structure_signature(
            lane.project_radio_score_to_draft(rebuilt)
        )
    )
    assert [cue.anchor_line_id for cue in rebuilt.music_cues] == [
        "l001", "l013", "l024",
    ]


def test_p3_rewrite_rejects_structural_mutation_then_repairs_the_draft():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    score = lane.compile_radio_score_draft(draft, advisory, cast, facts)
    projected = lane.project_radio_score_to_draft(score)
    signature = lane._radio_score_draft_structure_signature(projected)
    changed = projected.model_dump(mode="json")
    changed["scenes"][0]["beats"][0]["char_id"] = "c01"
    responses = [json.dumps(changed), json.dumps(projected.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    artifact_inputs = {
        **_draft_context(advisory, cast, facts),
        "previous_draft": projected.model_dump(mode="json"),
        "review": {"verdict": "rewrite", "issues": ["Tighten the turn."]},
    }

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane._call_radio_score_draft(
        pass_id="P3_rewrite", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=artifact_inputs,
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.55, structural_retry_temperature=.20,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal={}, expected_signature=signature,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    assert "Preserve the previous_draft structural decisions" in calls[0]["messages"][0]["content"]
    assert "preserve every other previous_draft prose leaf byte for byte" in calls[0]["messages"][0]["content"]
    assert "safe ceilings: title <=48; premise <=108; setting <=60" in calls[0]["messages"][0]["content"]
    assert "<failed_radio_score_draft>" in calls[1]["messages"][1]["content"]


def test_p3_exact_resolved_artifact_repair_envelope_is_unwrapped():
    """Live 2026-07-12: typed repair returned one transport wrapper."""
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    invalid = draft.model_dump(mode="json")
    invalid["music_cues"].append(copy.deepcopy(invalid["music_cues"][0]))
    base_raw = json.dumps(invalid)
    wrapped = json.dumps({"resolved_artifact": draft.model_dump(mode="json")})
    responses = [base_raw, wrapped]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    journal: dict[str, object] = {}
    result = lane._call_radio_score_draft(
        pass_id="P3", slot_fn=slot_fn,
        pack=routing.resolve_story_pack("scifi_codex"),
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=_draft_context(advisory, cast, facts),
        advisory=advisory, cast=cast, fact_index=facts,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        call_journal=journal,
    )

    assert isinstance(result, lane.RadioScoreV4)
    assert len(calls) == 2
    attempts = journal["calls"][0]["attempts"]
    assert [a["resolved_artifact_unwrapped"] for a in attempts] == [False, True]
    assert attempts[0]["raw_chars"] == len(base_raw)
    assert attempts[0]["raw_sha256"] == hashlib.sha256(base_raw.encode()).hexdigest()
    assert attempts[1]["raw_chars"] == len(wrapped)
    assert attempts[1]["raw_sha256"] == hashlib.sha256(wrapped.encode()).hexdigest()


def test_default_schema_injection_preserves_wire_receipt_without_unwrap():
    question = lane.DramaticQuestionV4(
        question="What must the observer choose?",
        consequence="The report may change a dangerous decision.",
        ending_direction="Choose caution before dawn.",
    )
    raw = json.dumps(question.model_dump(mode="json"))
    journal = {}
    messages_seen: list[object] = []
    pack = SimpleNamespace(prompt_stages={
        "codex_question_system": "Question.",
    })

    def slot_fn(messages, **kwargs):
        messages_seen.append(messages)
        return raw

    result = lane.invoke_codex_structured(
        pass_id="P1", slot="creative",
        slot_fn=slot_fn,
        pack=pack,
        seam_refs=("codex_question_system",),
        artifact_inputs={}, result_type=lane.DramaticQuestionV4,
        post_validator=lambda candidate: None,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=3600, call_journal=journal,
    )

    assert result == question
    attempt = journal["calls"][0]["attempts"][0]
    assert attempt["resolved_artifact_unwrapped"] is False
    assert attempt["raw_chars"] == len(raw)
    assert attempt["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert attempt["schema_status"] == "accepted"
    assert '"result_json_schema"' in messages_seen[0][1]["content"]
    assert "Its exact top-level keys are: question, consequence, ending_direction" in (
        messages_seen[0][0]["content"]
    )


def test_p1_typed_repair_uses_compact_exact_contract_and_safe_rewrite_margin():
    valid_question = "Must Lina trust the evaluator before the gallery opens?"
    valid_consequence = "Trust may expose bias; refusal may discard a useful pattern."
    overlong_ending = (
        "Before the exhibition closes, Lina must choose whether computational "
        "judgment deserves authority over her own creative decision and its consequences."
    )
    base = json.dumps({
        "question": valid_question,
        "consequence": valid_consequence,
        "ending_direction": overlong_ending,
    })
    repaired_ending = "Lina must choose whether the evaluator deserves authority over her design."
    repaired = json.dumps({
        "question": valid_question,
        "consequence": valid_consequence,
        "ending_direction": repaired_ending,
    })
    responses = [base, repaired]
    calls: list[dict[str, object]] = []
    pack = SimpleNamespace(prompt_stages={
        "codex_question_system": "Write the dramatic question.",
    })

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane.invoke_codex_structured(
        pass_id="P1", slot="creative", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_question_system",), artifact_inputs={
            "fact_index": {"facts": ["large trusted context must stay out of repair"]},
        }, result_type=lane.DramaticQuestionV4,
        post_validator=lambda candidate: None,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=1800, call_journal={}, clamp_overlong_strings=False,
    )

    assert result.ending_direction == repaired_ending
    assert len(calls) == 2
    repair_system = calls[1]["messages"][0]["content"]
    repair_user = json.loads(calls[1]["messages"][1]["content"])
    assert "Return exactly three root keys" in repair_system
    assert "ending_direction is at most 120 characters" in repair_system
    assert "ending_direction at or below 90 characters" in repair_system
    assert "never copy it unchanged" in repair_system
    assert repair_user["failed_dramatic_question"]["ending_direction"] == overlong_ending
    assert "original_request" not in repair_user
    assert "fact_index" not in repair_user


def test_p4_typed_repair_keeps_exact_review_shape_and_only_compact_failed_review():
    # Derive the overflow from the SCHEMA, never a hardcoded literal. This test is
    # about the typed-repair PATH, not about any particular cap: it just needs a
    # rationale that is one character too long. Hardcoding 241 silently pinned the
    # old 240 cap, so raising the cap (2026-07-14, 240 -> 400: the reviewer kept
    # overshooting a limit set to a length it does not write in) made the trigger
    # value VALID and the repair path stopped being exercised at all.
    _rationale_cap = lane.StructureReviewV4.model_fields["rationale"].metadata[0].max_length
    overlong_rationale = "A" * (_rationale_cap + 1)
    base = json.dumps({
        "verdict": "pass",
        "issues": [],
        "rationale": overlong_rationale,
    })
    repaired = json.dumps({
        "verdict": "pass",
        "issues": [],
        "rationale": "The score is coherent, causal, and ready for scripting.",
    })
    calls: list[dict[str, object]] = []
    responses = [base, repaired]
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Review the score.",
        "codex_coda_contract_system": "Keep the coda source-bound.",
    })

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane.invoke_codex_structured(
        pass_id="P4", slot="technical", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={"score": {"large": "accepted score must not be repeated"}},
        result_type=lane.StructureReviewV4, post_validator=lambda _: None,
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=1800, call_journal={}, clamp_overlong_strings=False,
    )

    assert result.verdict == "pass"
    assert result.issues == []
    assert len(calls) == 2
    for call in calls:
        system = call["messages"][0]["content"]
        assert "verdict is the exact JSON string literal pass or rewrite" in system
        assert "issues is a flat JSON array of zero to six strings" in system
        assert "never return fail" in system
    repair_user = json.loads(calls[1]["messages"][1]["content"])
    assert set(repair_user) == {"failed_structure_review", "rejection"}
    assert "original_request" not in calls[1]["messages"][1]["content"]
    assert "accepted score must not be repeated" not in calls[1]["messages"][1]["content"]
    assert "string_too_long" in repair_user["rejection"]


def test_resolved_artifact_envelope_with_sibling_key_stays_fail_loud():
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    draft = _draft_for_advisory(advisory)
    calls = []

    def slot_fn(messages, **kwargs):
        calls.append(1)
        return json.dumps({
            "resolved_artifact": draft.model_dump(mode="json"),
            "unexpected": "must not be discarded",
        })

    with pytest.raises(lane.CodexPassError):
        lane._call_radio_score_draft(
            pass_id="P3", slot_fn=slot_fn,
            pack=routing.resolve_story_pack("scifi_codex"),
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=_draft_context(advisory, cast, facts),
            advisory=advisory, cast=cast, fact_index=facts,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
            call_journal={},
        )
    assert len(calls) == 2


@pytest.mark.parametrize("wrapped_value", [[], "not-an-object", 7])
def test_resolved_artifact_non_object_value_stays_fail_loud(wrapped_value):
    advisory = lane.make_advisory_word_blueprint(30, ["b000", "b001", "b002"])
    cast = _metadata_repair_cast()
    facts = _metadata_repair_fact_index()
    with pytest.raises(lane.CodexPassError):
        lane._call_radio_score_draft(
            pass_id="P3",
            slot_fn=lambda messages, **kwargs: json.dumps({
                "resolved_artifact": wrapped_value,
            }),
            pack=routing.resolve_story_pack("scifi_codex"),
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs=_draft_context(advisory, cast, facts),
            advisory=advisory, cast=cast, fact_index=facts,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
            call_journal={},
        )


def test_scifi_codex_prompt_seams_forbid_envelope_key():
    pack = routing.resolve_story_pack("scifi_codex")
    assert pack.prompt_stages
    assert all(
        "resolved_artifact" not in str(text)
        for text in pack.prompt_stages.values()
    )
    radio_score_seam = pack.prompt_stages["codex_radio_score_system"]
    assert "RadioScoreDraftV4" in radio_score_seam
    assert "never manufacture canonical IDs" in radio_score_seam
    assert "previous_draft locks its structural choices" in radio_score_seam
    assert "RadioScoreV4" not in radio_score_seam


def test_script_output_token_budget_receipts_and_bounds():
    # The reservation scales on BOTH drivers of the serialized artifact: the
    # dialogue word steer AND the accepted line count (every accepted line pays
    # the strict per-line metadata cost, even in a 30-word script).
    assert lane._script_output_token_budget(30, 13) == 2800    # floor holds
    assert lane._script_output_token_budget(300, 13) == 3640   # 1350 + 1690 + 600

    # THE 5,400 CEILING IS GONE, and this test used to enshrine it: it asserted
    # `_script_output_token_budget(720, 13) == 5400  # ceiling holds`. Somebody
    # evaluated the clamp AT 720 WORDS, saw it bite, and wrote down the bitten
    # value as the contract. That is the whole bug, preserved in a green test.
    #
    # The ceiling's comment claimed it "keeps the reservation inside the context
    # cap" -- the false 8,192 cap every remote model was budgeted against
    # (PBUG-20260713-20). P5/P7/P9 run on the CREATIVE slot: aion-3.0-mini has
    # 131,072 tokens. The artifact needs what the artifact needs.
    assert lane._script_output_token_budget(720, 13) == 5530   # 3240 + 1690 + 600
    assert lane._script_output_token_budget(720, 24) == 6960   # 3240 + 3120 + 600
    assert lane._script_output_token_budget(900, 40) == 9850   # 4050 + 5200 + 600

    # The budget must never be capped by a constant that pretends to know the
    # slot's window. The transport owns that: fit_output_tokens clamps the
    # request against the slot's REAL context cap.
    assert lane._script_output_token_budget(720, 24) > 5400, (
        "a constant ceiling under the artifact's real size is how a 720-word "
        "script comes back cut off mid-JSON"
    )

    # A wider accepted graph reserves more output at the SAME word steer --
    # this is the P7 live truncation (generated == max_new_tokens == 2800).
    assert (
        lane._script_output_token_budget(300, 30)
        > lane._script_output_token_budget(300, 13)
    )
    for invalid in (True, 29, 901, 30.0):
        with pytest.raises(lane.CodexTargetRangeError):
            lane._script_output_token_budget(invalid, 13)
    for bad_count in (True, 0, -1, 13.0):
        with pytest.raises(lane.CodexTargetRangeError):
            lane._script_output_token_budget(30, bad_count)


def test_radio_score_draft_output_budget_preserves_the_live_p3_repair_window():
    """The compact draft reserves output while retaining an explicit input cap."""
    context_cap = 8192

    budget = lane._radio_score_draft_output_token_budget(120, 12)
    assert budget == lane._RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
    assert context_cap - budget == lane._radio_score_draft_surface_receipt()["input_token_reservation"]
    assert lane._radio_score_draft_output_token_budget(720, 12) == budget
    assert lane._radio_score_draft_output_token_budget(30, 3) == budget
    for invalid_words in (True, 29, 901, 120.0):
        with pytest.raises(lane.CodexTargetRangeError):
            lane._radio_score_draft_output_token_budget(invalid_words, 12)
    for invalid_beats in (True, 0, -1, 12.0, 13):
        with pytest.raises(lane.CodexTargetRangeError):
            lane._radio_score_draft_output_token_budget(120, invalid_beats)


def test_p3_uses_draft_helper_while_script_passes_keep_dynamic_budget():
    source_path = Path(__file__).parents[1] / "nodes" / "_otr_scifi_codex.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    p3_helpers = []
    script_passes: dict[str, ast.AST | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        pass_node = keywords.get("pass_id")
        if not isinstance(pass_node, ast.Constant) or not isinstance(pass_node.value, str):
            continue
        if node.func.id == "_call_radio_score_draft":
            p3_helpers.append((pass_node.value, keywords))
        elif node.func.id == "invoke_codex_structured":
            script_passes[pass_node.value] = keywords.get("max_new_tokens")

    assert {pass_id for pass_id, _keywords in p3_helpers} == {"P3", "P3_rewrite"}
    for pass_id, keywords in p3_helpers:
        budget = keywords["max_new_tokens"]
        assert isinstance(budget, ast.Name)
        assert budget.id == "score_token_budget"
        assert isinstance(keywords["advisory"], ast.Name)
        assert isinstance(keywords["cast"], ast.Name)
        assert isinstance(keywords["fact_index"], ast.Name)
    for pass_id in ("P5", "P7", "P9"):
        budget = script_passes[pass_id]
        assert isinstance(budget, ast.Name)
        assert budget.id == "script_token_budget"
    assert "_score_graph_contract" not in source
    assert "repair_radio_score_metadata" not in source
    assert "include_result_json_schema=False" in source
    assert "radio_score_draft_token_budget" in source


def test_script_artifact_metadata_repair_normalizes_only_graph_metadata():
    score = _metadata_repair_score()
    raw = _legacy_metadata_artifact()
    preserved = copy.deepcopy(raw)

    repaired = lane.repair_script_artifact_metadata(_fenced_json(raw), score)

    assert repaired is not None
    dumped = repaired.model_dump(mode="json")
    assert dumped["schema_version"] == "scifi_codex.script_artifact.v4"
    assert dumped["scenes"] == preserved["scenes"]
    assert "legacy_artifact_metadata" not in dumped
    assert [line.line_id for line in repaired.lines] == [
        row["line_id"] for row in preserved["lines"]
    ]
    expected_metadata = {
        "l001": ("shot_001", "shot_start"),
        "l002": ("shot_001", "continue"),
        "l003": ("shot_001", "beat_start"),
        "l004": ("shot_002", "shot_start"),
    }
    by_id = {line.line_id: line.model_dump(mode="json") for line in repaired.lines}
    for original in preserved["lines"]:
        line = by_id[original["line_id"]]
        shot_id, boundary = expected_metadata[original["line_id"]]
        assert line["shot_id"] == shot_id
        assert line["boundary"] == boundary
        assert "speaker" not in line
        assert "legacy_line_metadata" not in line
        for field in lane.ScriptLineV4.model_fields:
            if field not in {"shot_id", "boundary"}:
                assert line[field] == original[field]
    assert dumped["music_cues"][0]["description"] == preserved["music_cues"][0]["description"]
    assert "speaker" not in dumped["music_cues"][0]


def test_codex_tail_canon_uses_the_complete_episode_canon_protocol(tmp_path):
    from nodes import _otr_canon

    score = _metadata_repair_score()
    script = lane.repair_script_artifact_metadata(
        _fenced_json(_legacy_metadata_artifact()), score,
    )
    assert script is not None

    locked_premise = "Will the observatory answer the signal?"
    canon = lane._build_codex_episode_canon(
        score, script, premise=locked_premise,
    )

    assert isinstance(canon, _otr_canon.EpisodeCanon)
    assert canon.title == script.title
    assert canon.premise == locked_premise
    assert canon.setting == score.setting
    assert canon.time_of_day == ""
    assert canon.sound_palette == []
    written = _otr_canon.write_episode_canon(tmp_path, canon)
    assert _otr_canon.load_episode_canon(tmp_path) == canon
    assert written.name == _otr_canon.EPISODE_CANON_FILENAME


def test_script_artifact_metadata_repair_fails_closed_for_bad_graph_mappings():
    score = _metadata_repair_score()
    raw = _legacy_metadata_artifact()
    assert lane.repair_script_artifact_metadata('{"lines":[}', score) is None

    unknown_line = copy.deepcopy(raw)
    unknown_line["lines"][0]["line_id"] = "l999"
    assert lane.repair_script_artifact_metadata(_fenced_json(unknown_line), score) is None

    extra_l006 = copy.deepcopy(raw)
    extra_l006["lines"].append(copy.deepcopy(extra_l006["lines"][0]))
    extra_l006["lines"][-1]["line_id"] = "l006"
    preserved_extra_l006 = copy.deepcopy(extra_l006)
    assert lane.repair_script_artifact_metadata(_fenced_json(extra_l006), score) is None
    assert extra_l006 == preserved_extra_l006

    duplicate_line = copy.deepcopy(raw)
    duplicate_line["lines"].append(copy.deepcopy(duplicate_line["lines"][0]))
    assert lane.repair_script_artifact_metadata(_fenced_json(duplicate_line), score) is None

    # `beat_id` is a coordinate the accepted score already owns -- the line_id
    # determines it. It is restored from the same tuple as shot_id/boundary,
    # not refused.
    wrong_beat = copy.deepcopy(raw)
    wrong_beat["lines"][0]["beat_id"] = "b999"
    repaired = lane.repair_script_artifact_metadata(
        _fenced_json(wrong_beat), score,
    )
    assert repaired is not None
    assert repaired.lines[0].beat_id == raw["lines"][0]["beat_id"]

    ambiguous_score = score.model_dump(mode="json")
    ambiguous_score["scenes"][0]["beats"][1]["line_ids"] = ["l002", "l003"]
    assert lane.repair_script_artifact_metadata(
        _fenced_json(raw), lane.RadioScoreV4.model_validate(ambiguous_score),
    ) is None

    missing_shot_score = score.model_dump(mode="json")
    missing_shot_score["scenes"][0]["beats"][0]["shot_id"] = "shot_999"
    assert lane.repair_script_artifact_metadata(
        _fenced_json(raw), lane.RadioScoreV4.model_validate(missing_shot_score),
    ) is None

    missing_authored_field = copy.deepcopy(raw)
    missing_authored_field["lines"][0].pop("text")
    assert lane.repair_script_artifact_metadata(
        _fenced_json(missing_authored_field), score,
    ) is None

    nested_score_scene = copy.deepcopy(raw)
    nested_score_scene["scenes"][0]["shots"] = []
    assert lane.repair_script_artifact_metadata(
        _fenced_json(nested_score_scene), score,
    ) is None


def test_script_post_validator_rejects_schema_valid_wrong_graph_metadata():
    score = _metadata_repair_score()
    repaired = lane.repair_script_artifact_metadata(
        _fenced_json(_legacy_metadata_artifact()), score,
    )
    assert repaired is not None
    wrong_first_line = repaired.lines[0].model_copy(
        update={"shot_id": "shot_002", "boundary": "continue"}
    )
    wrong_shot = repaired.model_copy(update={"lines": [wrong_first_line, *repaired.lines[1:]]})

    assert lane._validate_script_post(wrong_shot, _metadata_repair_cast(), score) == (
        "line l003 does not resolve to its accepted shot"
    )
    wrong_boundary_line = repaired.lines[0].model_copy(update={"boundary": "continue"})
    wrong_boundary = repaired.model_copy(
        update={"lines": [wrong_boundary_line, *repaired.lines[1:]]}
    )
    assert lane._validate_script_post(wrong_boundary, _metadata_repair_cast(), score) == (
        "line l003 has an invalid accepted-order boundary"
    )
    score_shaped_scene = repaired.model_copy(update={"scenes": [{
        "scene_id": "scene_001", "shots": [],
    }]})
    assert lane._validate_script_post(score_shaped_scene, _metadata_repair_cast(), score) == (
        "script scenes contain forbidden score or legacy fields"
    )


def test_script_metadata_repair_preserves_nullable_dialogue_slots_without_fabrication():
    score = _metadata_repair_score()
    raw = _legacy_metadata_artifact()
    for line in raw["lines"]:
        line["dialogue_slot_id"] = None
    repaired = lane.repair_script_artifact_metadata(_fenced_json(raw), score)
    assert repaired is not None
    assert all(line.dialogue_slot_id is None for line in repaired.lines)


def test_script_metadata_repair_uses_only_declared_neutral_metadata_defaults():
    score = _metadata_repair_score()
    raw = _legacy_metadata_artifact()
    for line in raw["lines"]:
        for field in ("skip", "tts_skip_reason", "traits", "compose_flags", "fact_ids"):
            line.pop(field)
    repaired = lane.repair_script_artifact_metadata(_fenced_json(raw), score)
    assert repaired is not None
    assert all(line.skip is False for line in repaired.lines)
    assert all(line.tts_skip_reason is None for line in repaired.lines)
    assert all(line.traits == "" for line in repaired.lines)
    assert all(line.compose_flags == [] and line.fact_ids == [] for line in repaired.lines)


def test_script_metadata_repair_short_circuits_the_typed_repair_model_call():
    score = _metadata_repair_score()
    raw = _fenced_json(_legacy_metadata_artifact())
    pack = SimpleNamespace(prompt_stages={
        "codex_play_system": "Write a ScriptArtifactV4.",
        "codex_coda_contract_system": "Keep the coda cautious.",
    })
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return raw

    result = lane.invoke_codex_structured(
        pass_id="P5", slot="creative", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_play_system", "codex_coda_contract_system"),
        artifact_inputs={"score": score.model_dump(mode="json")},
        result_type=lane.ScriptArtifactV4, post_validator=lambda _: None,
        base_temperature=.78, structural_retry_temperature=.35,
        max_new_tokens=2200, call_journal={}, repair_score=score,
    )

    assert isinstance(result, lane.ScriptArtifactV4)
    assert len(calls) == 1
    assert "SCRIPT ARTIFACT ROOT CONTRACT" in calls[0]["messages"][0]["content"]


def test_script_artifact_inputs_flatten_the_score_without_losing_line_constraints():
    score = _metadata_repair_score()
    fact_index = _metadata_repair_fact_index()
    inputs = lane._script_artifact_inputs(
        score, fact_index, lane.WordSteerV4(requested_words=30),
    )

    assert "score" not in inputs
    assert inputs["story_context"] == {
        "title": score.title,
        "premise": score.premise,
        "setting": score.setting,
        "scenes": [
            {
                "scene_id": "scene_001",
                "env": "Observatory",
                "description": "A receiver clicks under a cold sky.",
            }
        ],
    }
    assert '"shots"' not in json.dumps(inputs["story_context"])
    assert '"beats"' not in json.dumps(inputs["story_context"])
    assert inputs["accepted_line_graph"] == [
        {
            "line_id": "l001", "beat_id": "b001", "shot_id": "shot_001",
            "char_id": "announcer", "speaker_role": "announcer",
            "arc_phase": "arrival", "beat_intent": "Establish the signal.",
            "fact_ids": ["F01"],
        },
        {
            "line_id": "l002", "beat_id": "b001", "shot_id": "shot_001",
            "char_id": "announcer", "speaker_role": "announcer",
            "arc_phase": "arrival", "beat_intent": "Establish the signal.",
            "fact_ids": ["F01"],
        },
        {
            "line_id": "l003", "beat_id": "b002", "shot_id": "shot_001",
            "char_id": "announcer", "speaker_role": "announcer",
            "arc_phase": "test", "beat_intent": "Raise the choice.",
            "fact_ids": ["F01"],
        },
        {
            "line_id": "l004", "beat_id": "b003", "shot_id": "shot_002",
            "char_id": "announcer", "speaker_role": "announcer",
            "arc_phase": "decision", "beat_intent": "Commit to caution.",
            "fact_ids": ["F01"],
        },
    ]
    assert all("speaker" not in row for row in inputs["accepted_line_graph"])
    assert inputs["accepted_line_ids"] == ["l001", "l002", "l003", "l004"]
    assert inputs["accepted_line_count"] == 4
    assert inputs["music_cues"] == [cue.model_dump(mode="json") for cue in score.music_cues]
    assert inputs["fact_index"] == {
        "facts": [{"fact_id": "F01", "claim": "A cautious signal reaches the observatory."}],
        "tone": "cautious",
    }
    assert inputs["initial_draft_word_steer"] == {"requested_words": 30}
