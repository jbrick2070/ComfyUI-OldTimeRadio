from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    assert lane._spoken_error("Iona, listen", "Iona")
    assert lane._spoken_error("The signal arrives", "Iona") is None


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
    assert sum(x["advisory_word_center"] for x in plan.per_beat) == 719
    assert lane._score_graph_contract(plan) == {
        "required_beat_ids": ["b001", "b002", "b003"],
        "required_beat_orders": [
            {"beat_id": "b001", "order": 1},
            {"beat_id": "b002", "order": 2},
            {"beat_id": "b003", "order": 3},
        ],
        "line_id_policy": (
            "Flattened scene/beat line_ids must be unique contiguous canonical "
            "IDs l001, l002, and so on, with no gaps."
        ),
        "music_anchor_policy": (
            "Every music cue anchor_line_id must be one of those line_ids and "
            "anchor_beat_id must be that line's owning beat_id."
        ),
    }


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


def test_score_metadata_repair_derives_only_a_known_cue_anchor_beat():
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    bad = score.model_dump(mode="json")
    bad["music_cues"][0]["anchor_beat_id"] = "b003"
    preserved = copy.deepcopy(bad)

    repaired = lane.repair_radio_score_metadata(_fenced_json(bad), advisory)

    assert repaired is not None
    dumped = repaired.model_dump(mode="json")
    assert dumped["music_cues"][0]["anchor_beat_id"] == "b001"
    assert dumped["music_cues"][0]["anchor_line_id"] == "l001"
    assert {
        key: value for key, value in dumped.items() if key != "music_cues"
    } == {
        key: value for key, value in preserved.items() if key != "music_cues"
    }
    assert bad == preserved

    unknown_anchor = copy.deepcopy(bad)
    unknown_anchor["music_cues"][0]["anchor_line_id"] = "l006"
    assert lane.repair_radio_score_metadata(_fenced_json(unknown_anchor), advisory) is None


def test_p3_cue_anchor_metadata_repair_short_circuits_the_model_call():
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    bad = score.model_dump(mode="json")
    bad["music_cues"][0]["anchor_beat_id"] = "b003"
    calls: list[dict[str, object]] = []
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return _fenced_json(bad)

    result = lane.invoke_codex_structured(
        pass_id="P3", slot="creative", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={"score_graph_contract": lane._score_graph_contract(advisory)},
        result_type=lane.RadioScoreV4,
        post_validator=lambda candidate: lane._validate_radio_score_graph(candidate, advisory),
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=3600, call_journal={}, repair_advisory=advisory,
    )

    assert result.music_cues[0].anchor_beat_id == "b001"
    assert len(calls) == 1


def test_p3_typed_repair_receives_the_locked_score_graph_contract():
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    invalid = score.model_dump(mode="json")
    invalid["scenes"][0]["beats"].pop()
    responses = [json.dumps(invalid), json.dumps(score.model_dump(mode="json"))]
    calls: list[dict[str, object]] = []
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    result = lane.invoke_codex_structured(
        pass_id="P3", slot="creative", slot_fn=slot_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={
            "advisory_word_plan": advisory.model_dump(mode="json"),
            "score_graph_contract": lane._score_graph_contract(advisory),
        },
        result_type=lane.RadioScoreV4,
        post_validator=lambda candidate: lane._validate_radio_score_graph(candidate, advisory),
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=3600, call_journal={},
    )

    assert result == score
    assert len(calls) == 2
    repair_system = calls[1]["messages"][0]["content"]
    assert "score_graph_contract" in repair_system
    assert "every and only required_beat_ids" in repair_system


def test_p3_exact_resolved_artifact_repair_envelope_is_unwrapped():
    """Live 2026-07-12: typed repair returned one transport wrapper."""
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    invalid = score.model_dump(mode="json")
    invalid["scenes"][0]["beats"].pop()
    base_raw = json.dumps(invalid)
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })

    wrapped = json.dumps({
        "resolved_artifact": score.model_dump(mode="json"),
    })
    responses = [base_raw, wrapped]
    calls: list[dict[str, object]] = []

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    journal = {}
    result = lane.invoke_codex_structured(
        pass_id="P3", slot="creative",
        slot_fn=slot_fn,
        pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={
            "advisory_word_plan": advisory.model_dump(mode="json"),
            "score_graph_contract": lane._score_graph_contract(advisory),
        },
        result_type=lane.RadioScoreV4,
        post_validator=lambda candidate: lane._validate_radio_score_graph(
            candidate, advisory
        ),
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=3600, call_journal=journal, repair_advisory=advisory,
    )

    assert result == score
    assert len(calls) == 2
    repair_system = calls[1]["messages"][0]["content"]
    assert "every and only required_beat_ids" in repair_system
    attempts = journal["calls"][0]["attempts"]
    assert [a["resolved_artifact_unwrapped"] for a in attempts] == [False, True]
    assert attempts[0]["raw_chars"] == len(base_raw)
    assert attempts[0]["raw_sha256"] == hashlib.sha256(base_raw.encode()).hexdigest()
    assert attempts[1]["raw_chars"] == len(wrapped)
    assert attempts[1]["raw_sha256"] == hashlib.sha256(wrapped.encode()).hexdigest()


def test_p3_direct_root_preserves_wire_receipt_without_unwrap():
    score = _metadata_repair_score()
    advisory = score.advisory_word_plan
    raw = json.dumps(score.model_dump(mode="json"))
    journal = {}
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })

    result = lane.invoke_codex_structured(
        pass_id="P3", slot="creative",
        slot_fn=lambda messages, **kwargs: raw,
        pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={}, result_type=lane.RadioScoreV4,
        post_validator=lambda candidate: lane._validate_radio_score_graph(
            candidate, advisory
        ),
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=3600, call_journal=journal,
    )

    assert result == score
    attempt = journal["calls"][0]["attempts"][0]
    assert attempt["resolved_artifact_unwrapped"] is False
    assert attempt["raw_chars"] == len(raw)
    assert attempt["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


def test_resolved_artifact_envelope_with_sibling_key_stays_fail_loud():
    score = _metadata_repair_score()
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })
    calls = []
    def slot_fn(messages, **kwargs):
        calls.append(1)
        return json.dumps({
            "resolved_artifact": score.model_dump(mode="json"),
            "unexpected": "must not be discarded",
        })

    with pytest.raises(lane.CodexPassError):
        lane.invoke_codex_structured(
            pass_id="P3", slot="creative",
            slot_fn=slot_fn,
            pack=pack,
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs={}, result_type=lane.RadioScoreV4,
            post_validator=lambda candidate: None,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=3600, call_journal={},
        )
    assert len(calls) == 2


@pytest.mark.parametrize("wrapped_value", [[], "not-an-object", 7])
def test_resolved_artifact_non_object_value_stays_fail_loud(wrapped_value):
    pack = SimpleNamespace(prompt_stages={
        "codex_radio_score_system": "Score.",
        "codex_coda_contract_system": "Coda.",
    })
    with pytest.raises(lane.CodexPassError):
        lane.invoke_codex_structured(
            pass_id="P3", slot="creative",
            slot_fn=lambda messages, **kwargs: json.dumps({
                "resolved_artifact": wrapped_value,
            }),
            pack=pack,
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs={}, result_type=lane.RadioScoreV4,
            post_validator=lambda candidate: None,
            base_temperature=.72, structural_retry_temperature=.32,
            max_new_tokens=3600, call_journal={},
        )


def test_scifi_codex_prompt_seams_forbid_envelope_key():
    pack = routing.resolve_story_pack("scifi_codex")
    assert pack.prompt_stages
    assert all(
        "resolved_artifact" not in str(text)
        for text in pack.prompt_stages.values()
    )


def test_script_output_token_budget_receipts_and_bounds():
    # The reservation scales on BOTH drivers of the serialized artifact: the
    # dialogue word steer AND the accepted line count (every accepted line pays
    # the strict per-line metadata cost, even in a 30-word script).
    assert lane._script_output_token_budget(30, 13) == 2800    # floor holds
    assert lane._script_output_token_budget(300, 13) == 3640   # 1350 + 1690 + 600
    assert lane._script_output_token_budget(720, 13) == 5400   # ceiling holds
    assert lane._script_output_token_budget(900, 40) == 5400   # ceiling holds
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


def test_only_whole_script_passes_use_dynamic_token_budget():
    source_path = Path(__file__).parents[1] / "nodes" / "_otr_scifi_codex.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    seen: dict[str, ast.AST | None] = {}
    repair_scores: dict[str, ast.AST | None] = {}
    repair_advisories: dict[str, ast.AST | None] = {}
    artifact_inputs: dict[str, ast.AST | None] = {}
    post_validators: dict[str, ast.AST | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "invoke_codex_structured":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pass_node = keywords.get("pass_id")
        if isinstance(pass_node, ast.Constant) and isinstance(pass_node.value, str):
            seen[pass_node.value] = keywords.get("max_new_tokens")
            repair_scores[pass_node.value] = keywords.get("repair_score")
            repair_advisories[pass_node.value] = keywords.get("repair_advisory")
            artifact_inputs[pass_node.value] = keywords.get("artifact_inputs")
            post_validators[pass_node.value] = keywords.get("post_validator")
    for pass_id in ("P5", "P7", "P9"):
        budget_node = seen[pass_id]
        assert isinstance(budget_node, ast.Name)
        assert budget_node.id == "script_token_budget"
        repair_score = repair_scores[pass_id]
        assert isinstance(repair_score, ast.Name)
        assert repair_score.id == "score"
    for pass_id in ("P3", "P3_rewrite"):
        validator = post_validators[pass_id]
        assert isinstance(validator, ast.Lambda)
        assert isinstance(validator.body, ast.Call)
        assert isinstance(validator.body.func, ast.Name)
        assert validator.body.func.id == "_validate_radio_score_graph"
        assert len(validator.body.args) == 2
        assert isinstance(validator.body.args[1], ast.Name)
        assert validator.body.args[1].id == "advisory"
        repair_advisory = repair_advisories[pass_id]
        assert isinstance(repair_advisory, ast.Name)
        assert repair_advisory.id == "advisory"
    p5_inputs = artifact_inputs["P5"]
    assert isinstance(p5_inputs, ast.Call)
    assert isinstance(p5_inputs.func, ast.Name)
    assert p5_inputs.func.id == "_script_artifact_inputs"
    for pass_id in ("P7", "P9"):
        inputs = artifact_inputs[pass_id]
        assert isinstance(inputs, ast.Dict)
        assert any(
            key is None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_script_artifact_context"
            for key, value in zip(inputs.keys, inputs.values)
        )
    p0_inputs = artifact_inputs["P0"]
    assert isinstance(p0_inputs, ast.Name)
    assert p0_inputs.id == "p0_inputs"
    p0_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "invoke_codex_structured"
        and any(
            keyword.arg == "pass_id"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "P0"
            for keyword in node.keywords
        )
    )
    p0_keywords = {keyword.arg: keyword.value for keyword in p0_call.keywords if keyword.arg}
    assert isinstance(p0_keywords["prompt_must_fit"], ast.Constant)
    assert p0_keywords["prompt_must_fit"].value is True
    assert isinstance(p0_keywords["post_validator"], ast.Lambda)
    validator_call = p0_keywords["post_validator"].body
    assert isinstance(validator_call, ast.Call)
    validator_keywords = {keyword.arg: keyword.value for keyword in validator_call.keywords if keyword.arg}
    assert isinstance(validator_keywords["allowed_source_fields"], ast.Name)
    assert validator_keywords["allowed_source_fields"].id == "p0_allowed_fields"
    expected_digest = validator_keywords["expected_payload_sha256"]
    assert isinstance(expected_digest, ast.Attribute)
    assert isinstance(expected_digest.value, ast.Name)
    assert expected_digest.value.id == "env"
    assert expected_digest.attr == "source_digest"
    for pass_id, budget_node in seen.items():
        if pass_id not in {"P5", "P7", "P9"}:
            assert not (
                isinstance(budget_node, ast.Name)
                and budget_node.id == "script_token_budget"
            )


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

    wrong_beat = copy.deepcopy(raw)
    wrong_beat["lines"][0]["beat_id"] = "b999"
    assert lane.repair_script_artifact_metadata(_fenced_json(wrong_beat), score) is None

    ambiguous_score = score.model_dump(mode="json")
    ambiguous_score["scenes"][0]["beats"][1]["line_ids"] = ["l002", "l003"]
    assert lane.repair_script_artifact_metadata(
        _fenced_json(raw), lane.RadioScoreV4.model_validate(ambiguous_score),
    ) is None

    missing_shot_score = score.model_dump(mode="json")
    missing_shot_score["scenes"][0]["beats"][0]["shot_id"] = "missing_shot"
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
