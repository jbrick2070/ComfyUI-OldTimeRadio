from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from nodes import _otr_scifi_codex as lane
from nodes import _otr_content_authorship as content_authorship
from nodes import _otr_freeze_cascade as freeze_cascade
from nodes import production_ledger as PL


LineText = Callable[[str, int], str]
ScriptTransform = Callable[[lane.ScriptArtifactV4], lane.ScriptArtifactV4]


def _payload() -> dict[str, str]:
    body = (
        "The Meridian observatory measured a quiet repeating signal from an ice "
        "moon during a cautious orbital survey."
    )
    return {
        "headline": "Observatory measures a quiet signal",
        "summary": "Researchers report a cautious result from Meridian.",
        "full_text": body,
        "source": "Test Wire",
        "date": "2026-07-22",
        "link": "https://example.invalid/meridian",
        "seed_text": body,
    }


def _fact_index_from_p0(artifact_inputs: dict[str, Any]) -> lane.FactIndexV4:
    allowed = artifact_inputs["allowed_source_fields"]
    evidence = artifact_inputs["payload"]["payload"]
    field = allowed[0]
    quote = evidence[field][:24]
    return lane.FactIndexV4(
        facts=[
            lane.FactV4(
                fact_id="F01",
                claim="A quiet repeating signal reached Meridian observatory.",
                source_spans=[
                    lane.SourceSpanV4(
                        field=field, start=0, end=len(quote), quote=quote,
                    )
                ],
            )
        ],
        entities=[],
        numbers=[],
        tone="cautious",
        payload_sha256=artifact_inputs["payload"]["source_digest"],
    )


def _question() -> lane.DramaticQuestionV4:
    return lane.DramaticQuestionV4(
        question="Who should answer the signal before the moon sets?",
        consequence="A hurried reply could expose the observatory.",
        ending_direction="The listeners choose patience over certainty.",
    )


def _cast() -> lane.CastPlanV4:
    return lane.CastPlanV4(
        cast=[
            lane.CastPlanRowV4(
                char_id="announcer", name="ANNOUNCER",
                character_description="A calm witness to the night shift.",
                gender="neutral",
                role_in_conflict="Frames the observatory's choice.",
                voice_slot="announcer",
            ),
            lane.CastPlanRowV4(
                char_id="c01", name="Iona Vale",
                character_description="A careful radio astronomer.",
                gender="female",
                role_in_conflict="Must decide whether the signal deserves an answer.",
                voice_slot="c01",
            ),
        ]
    )


def _score_draft(advisory: lane.AdvisoryWordPlanV4) -> lane.RadioScoreDraftV4:
    scenes: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(advisory.per_beat):
        local_count = min(4, len(advisory.per_beat) - cursor)
        scene_number = len(scenes) + 1
        shot_count = 1 if local_count == 1 else 2
        scenes.append(
            {
                "env": f"Meridian room {scene_number}",
                "description": "Receivers glow beneath the turning dome.",
                "shots": [
                    {
                        "description": f"Receiver station {index + 1}",
                        "visual_prompt": "A quiet receiver under blue instrument light.",
                    }
                    for index in range(shot_count)
                ],
                "beats": [
                    {
                        "shot_index": local_index % shot_count,
                        "char_id": (
                            "announcer" if (cursor + local_index) % 2 == 0 else "c01"
                        ),
                        "line_count": 1,
                        "intent": "Advance the choice without changing the evidence.",
                        "arc_phase": "arrival" if cursor + local_index == 0 else "pressure",
                        "fact_ids": ["F01"],
                    }
                    for local_index in range(local_count)
                ],
            }
        )
        cursor += local_count
    return lane.RadioScoreDraftV4(
        title="Signal at Meridian",
        premise="A quiet signal forces one careful decision.",
        setting="Meridian observatory during the night shift.",
        scenes=scenes,
        music_cues=[
            {
                "cue_id": "music_open",
                "description": "A low radio pulse.",
                "generation_prompt": "Low sustained radio pulse with distant static.",
                "anchor_beat_index": 0,
                "anchor_line_index": 0,
            }
        ],
    )


def _default_line_text(_line_id: str, _index: int) -> str:
    return "A quiet signal crosses the dark observatory."


def _identity_line(
    line_id: str,
    text: str,
    *,
    char_id: str = "c01",
    speaker_role: str = "character",
    skip: bool = False,
    tts_skip_reason: str | None = None,
) -> lane.ScriptLineV4:
    return lane.ScriptLineV4(
        line_id=line_id,
        beat_id="b001",
        shot_id="shot_001",
        char_id=char_id,
        speaker_role=speaker_role,
        text=text,
        skip=skip,
        tts_skip_reason=tts_skip_reason,
        boundary="shot_start",
        arc_phase="arrival",
        beat_intent="Answer the signal.",
        dialogue_slot_id=None,
    )


def _identity_script(
    lines: list[lane.ScriptLineV4],
) -> lane.ScriptArtifactV4:
    return lane.ScriptArtifactV4(
        schema_version="scifi_codex.script_artifact.v4",
        title="Signal at Meridian",
        scenes=[{
            "scene_id": "scene_001",
            "env": "Observatory",
            "description": "Receivers glow.",
        }],
        lines=lines,
        music_cues=[{
            "cue_id": "music_open",
            "placement": "open",
            "description": "A low radio pulse.",
            "generation_prompt": "Low radio pulse.",
            "anchor_line_id": lines[0].line_id,
            "anchor_beat_id": "b001",
        }],
    )


def test_structured_codex_pass_reads_seams_from_prompt_stages(monkeypatch):
    observed: dict[str, Any] = {}
    expected = lane.DramaticQuestionV4(
        question="Who answers the signal?",
        consequence="A rushed reply exposes the observatory.",
        ending_direction="The listeners choose patience.",
    )

    def fake_structured_call(**kwargs):
        observed["system"] = kwargs["prompt"][0]["content"]
        return expected

    monkeypatch.setattr(lane, "structured_call", fake_structured_call)
    result = lane.invoke_codex_structured(
        pass_id="P1",
        slot="creative",
        slot_fn=lambda *_args, **_kwargs: "",
        pack=SimpleNamespace(
            prompt_stages={"codex_question_system": "seam from prompt stages"}
        ),
        seam_refs=("codex_question_system",),
        artifact_inputs={},
        result_type=lane.DramaticQuestionV4,
        post_validator=lambda _value: None,
        base_temperature=0.72,
        structural_retry_temperature=0.32,
        max_new_tokens=None,
        call_journal={},
    )

    assert result == expected
    assert observed["system"].startswith("seam from prompt stages")


def test_p0_alternate_owner_receipt_records_nonce_and_disposition():
    expected = _question()
    primary_calls = 0
    validator_calls = 0

    def primary(_messages, **_kwargs):
        nonlocal primary_calls
        primary_calls += 1
        return expected.model_dump_json()

    def alternate(_messages, **_kwargs):
        return expected.model_dump_json()

    def reject_primary_ladder(value):
        nonlocal validator_calls
        validator_calls += 1
        return "force the alternate owner" if validator_calls < 3 else None

    journal: dict[str, Any] = {}
    result = lane.invoke_codex_structured(
        pass_id="P0",
        slot="technical",
        slot_fn=primary,
        pack=SimpleNamespace(prompt_stages={
            "codex_fact_index_system": "fact seam",
        }),
        seam_refs=("codex_fact_index_system",),
        artifact_inputs={},
        result_type=lane.DramaticQuestionV4,
        post_validator=reject_primary_ladder,
        base_temperature=0.20,
        structural_retry_temperature=0.10,
        max_new_tokens=512,
        call_journal=journal,
        repair_slot_fn=alternate,
        repair_ledger_builder=lambda **kwargs: [{
            "role": "user",
            "content": "bounded repair nonce=" + kwargs["repair_nonce"],
        }],
        primary_backend_id="technical-backend",
        repair_owner_id="creative",
        repair_backend_id="creative-backend",
    )

    assert result == expected
    assert primary_calls == 2
    entry = journal["calls"][0]
    assert entry["terminal_disposition"] == "accepted_after_repair"
    assert entry["repair_attempted"] is True
    assert isinstance(entry["repair_nonce"], str)
    assert entry["repair_nonce"]
    assert [row["rung"] for row in entry["attempts"]] == [
        "primary", "primary", "alternate_repair",
    ]
    assert entry["attempts"][-1]["repair_nonce"] == entry["repair_nonce"]


def _run_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    target_words: int = 120,
    line_text: LineText = _default_line_text,
    transform_script: ScriptTransform | None = None,
    technical_generator: Callable[..., str] | None = None,
    state: dict[str, Any] | None = None,
):
    state = state if state is not None else {}
    state.update(
        trace=[], retry_flags=[], p3_calls=0, p5_calls=0, assemble_calls=0,
    )

    def creative_fn(*_args, **_kwargs) -> str:
        return ""

    if technical_generator is None:
        def technical_fn(*_args, **_kwargs) -> str:
            return ""
    else:
        technical_fn = technical_generator

    state["creative_fn"] = creative_fn
    state["technical_fn"] = technical_fn

    def fake_invoke(**kwargs):
        pass_id = kwargs["pass_id"]
        state["retry_flags"].append(
            (pass_id, kwargs.get("retry_until_valid"))
        )
        state["trace"].append(
            (
                pass_id,
                kwargs["slot"],
                kwargs["seam_refs"],
                kwargs["result_type"],
                kwargs["slot_fn"],
            )
        )
        if pass_id == "P0":
            result = _fact_index_from_p0(kwargs["artifact_inputs"])
        elif pass_id == "P1":
            result = _question()
        elif pass_id == "P2":
            result = _cast()
        elif pass_id == "P3":
            advisory = lane.AdvisoryWordPlanV4.model_validate(
                kwargs["artifact_inputs"]["advisory_word_plan"]
            )
            result = _score_draft(advisory)
        elif pass_id == "P5":
            result = lane.ScriptTextDraftV4(
                lines=[
                    lane.ScriptTextDraftLineV4(
                        line_id=row["line_id"],
                        text=line_text(row["line_id"], index),
                    )
                    for index, row in enumerate(
                        kwargs["artifact_inputs"]["accepted_line_graph"]
                    )
                ]
            )
        else:  # pragma: no cover - the assertion below is the public contract
            raise AssertionError(f"unexpected model pass {pass_id}")

        error = kwargs["post_validator"](result)
        assert error is None, error
        kwargs["call_journal"].setdefault("calls", []).append(
            {
                "pass_id": pass_id,
                "accepted": result.model_dump(mode="json"),
            }
        )
        return result

    real_p3 = lane._call_radio_score_draft
    real_p5 = lane._call_script_text_draft
    real_assemble = lane._assemble_ledger

    def tracked_p3(**kwargs):
        state["p3_calls"] += 1
        state["p3_max_new_tokens"] = kwargs["max_new_tokens"]
        score = real_p3(**kwargs)
        state["p3_score"] = score
        return score

    def tracked_p5(**kwargs):
        state["p5_calls"] += 1
        state["p5_score"] = kwargs["score"]
        script = real_p5(**kwargs)
        if transform_script is not None:
            script = transform_script(script)
        state["p5_script"] = script
        return script

    def tracked_assemble(led, score, cast, script, meta):
        state["assemble_calls"] += 1
        state["assembled_score"] = score
        state["assembled_script"] = script
        return real_assemble(led, score, cast, script, meta)

    monkeypatch.setattr(lane, "invoke_codex_structured", fake_invoke)
    monkeypatch.setattr(lane, "_call_radio_score_draft", tracked_p3)
    monkeypatch.setattr(lane, "_call_script_text_draft", tracked_p5)
    monkeypatch.setattr(lane, "_assemble_ledger", tracked_assemble)

    led = PL.new_ledger(
        episode_id=f"codex_{target_words}",
        out_dir=str(tmp_path / "ledger"),
    )
    meta = led.data.setdefault("meta", {})
    meta["source_bank"] = "scifi_news"
    state["ledger"] = led
    state["meta"] = meta

    parts = lane.run_scifi_codex_episode(
        payload=_payload(),
        pack=SimpleNamespace(),
        resolved={"seed_source": "rss_fetch", "target_words": target_words},
        led=led,
        meta=meta,
        creative_fn=creative_fn,
        technical_fn=technical_fn,
        slot_scheduler=object(),
        source_bank_row=SimpleNamespace(source_bank_id="scifi_news"),
        episode_root=tmp_path,
        episode_id=f"codex_{target_words}",
    )
    state["parts"] = parts
    return state


def test_fixed_topology_calls_exact_passes_and_keeps_one_p3_story_authority(
    monkeypatch, tmp_path,
):
    state = _run_lane(monkeypatch, tmp_path)

    observed = [row[:4] for row in state["trace"]]
    assert observed == [
        (
            "P0",
            "technical",
            ("codex_fact_index_system",),
            lane.FactIndexV4,
        ),
        (
            "P1",
            "creative",
            ("codex_question_system",),
            lane.DramaticQuestionV4,
        ),
        (
            "P2",
            "creative",
            ("codex_pressure_cast_system",),
            lane.CastPlanV4,
        ),
        (
            "P3",
            "creative",
            ("codex_radio_score_system", "codex_coda_contract_system"),
            lane.RadioScoreDraftV4,
        ),
        (
            "P5",
            "creative",
            ("codex_play_system", "codex_coda_contract_system"),
            lane.ScriptTextDraftV4,
        ),
    ]
    assert state["trace"][0][4] is state["technical_fn"]
    assert all(row[4] is state["creative_fn"] for row in state["trace"][1:])
    assert state["retry_flags"] == [
        ("P0", True), ("P1", True), ("P2", True), ("P3", True), ("P5", True),
    ]
    assert state["p3_calls"] == state["p5_calls"] == 1
    assert state["p3_max_new_tokens"] is None
    assert state["p3_score"] is state["p5_score"] is state["assembled_score"]
    assert state["assemble_calls"] == 1
    assert state["parts"].final_title_override == "Signal at Meridian"
    assert state["parts"].run_story_spine is False


def test_spoken_identity_canonicalization_fans_out_from_one_copied_artifact(
    monkeypatch, tmp_path,
):
    raw_text = "(softly) A quiet signal crosses the dark observatory."
    clean_text = "A quiet signal crosses the dark observatory."
    observed: dict[str, Any] = {"calls": 0}
    real_canonicalize = lane._canonicalize_script_spoken_text

    def tracked_canonicalize(
        script: lane.ScriptArtifactV4,
    ) -> lane.ScriptArtifactV4:
        observed["calls"] += 1
        observed["input"] = script
        observed["input_before"] = script.model_dump(mode="json")
        result = real_canonicalize(script)
        observed["result"] = result
        return result

    monkeypatch.setattr(
        lane, "_canonicalize_script_spoken_text", tracked_canonicalize,
    )
    state = _run_lane(
        monkeypatch,
        tmp_path,
        line_text=lambda _line_id, _index: raw_text,
    )

    canonical = observed["result"]
    original = observed["input"]
    assert observed["calls"] == 1
    assert canonical is not original
    assert original.model_dump(mode="json") == observed["input_before"]
    assert {line.text for line in original.lines} == {raw_text}
    assert {line.text for line in canonical.lines} == {clean_text}
    assert state["assembled_script"] is canonical

    expected = {
        str(row["line_id"]): clean_text
        for row in state["ledger"].data["lines"]
        if not row.get("skip")
    }
    for row in state["ledger"].data["lines"]:
        if row.get("line_id") in expected:
            assert row["text"] == clean_text
            assert row["char_count"] == len(clean_text)
            assert row["word_count"] == lane.canonical_word_count(clean_text)
    lane_meta = state["meta"]["scifi_codex"]
    assert lane_meta["script_text_identity_generation"] \
        == "clean_spoken_text.v1"
    assert lane_meta["accepted_lines"] == expected
    assert lane_meta["line_text_sha256"] == {
        line_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for line_id, text in expected.items()
    }
    assert lane_meta["script_digest"] == lane._script_digest(canonical)
    assert lane_meta["script_digest"] != lane._script_digest(original)
    assert lane_meta["actual_split_words"] == sum(
        lane.canonical_word_count(text) for text in expected.values()
    )
    word_receipt = lane_meta["word_receipt"]
    assert word_receipt["actual_character_text_sha256"] \
        == lane._OTRWD.character_text_sha256(state["ledger"].data)
    assert word_receipt["actual_announcer_text_sha256"] \
        == lane._OTRWD.announcer_text_sha256(state["ledger"].data)
    assert word_receipt["actual_total_voiced_text_sha256"] \
        == lane._OTRWD.total_voiced_text_sha256(state["ledger"].data)

    authorship = state["ledger"].data["meta"]["content_authorship"]
    artifact = next(
        row for row in authorship["accepted_artifacts"]
        if row["artifact_id"] == "final_script"
    )
    assert artifact["sha256"] == lane._script_digest(canonical)
    assert {
        row["line_id"]: row["text_sha256"]
        for row in authorship["line_proofs"]
    } == lane_meta["line_text_sha256"]

    finalizer = state["parts"].tail_finalizer
    assert finalizer.expected == expected
    finalizer._proof(state["ledger"].data)
    raw_ledger = copy.deepcopy(state["ledger"].data)
    raw_ledger["lines"][0]["text"] = raw_text
    with pytest.raises(
        lane.CodexPreTailAuditError, match="line receipt mismatch",
    ):
        finalizer._proof(raw_ledger)


def test_spoken_identity_counterfactual_retains_raw_text_hashes(
    monkeypatch, tmp_path,
):
    raw_text = "(softly) A quiet signal crosses the dark observatory."
    monkeypatch.setattr(
        lane, "_canonicalize_script_spoken_text", lambda script: script,
    )
    state = _run_lane(
        monkeypatch,
        tmp_path,
        line_text=lambda _line_id, _index: raw_text,
    )

    raw_expected = {
        str(row["line_id"]): raw_text
        for row in state["ledger"].data["lines"]
        if not row.get("skip")
    }
    lane_meta = state["meta"]["scifi_codex"]
    assert lane_meta["accepted_lines"] == raw_expected
    assert lane_meta["line_text_sha256"] == {
        line_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for line_id, text in raw_expected.items()
    }
    assert lane_meta["script_digest"] == lane._script_digest(
        state["assembled_script"]
    )
    artifact = next(
        row
        for row in state["ledger"].data["meta"]["content_authorship"][
            "accepted_artifacts"
        ]
        if row["artifact_id"] == "final_script"
    )
    assert artifact["sha256"] == lane._script_digest(
        state["assembled_script"]
    )
    assert state["parts"].tail_finalizer.expected == raw_expected


def test_spoken_identity_copy_preserves_skipped_and_music_rows():
    script = _identity_script([
        _identity_line("l001", "(softly) Spoken character line."),
        _identity_line(
            "l002", "(aside) Preserve this skipped row.",
            skip=True, tts_skip_reason="editorial_skip",
        ),
        _identity_line(
            "l003", "(music bed) Preserve this music row.",
            char_id="music_open", speaker_role="music_open",
            skip=True, tts_skip_reason="music_cue",
        ),
        _identity_line(
            "l004", "(calmly) Spoken announcer line.",
            char_id="announcer", speaker_role="announcer",
        ),
    ])
    before = script.model_dump(mode="json")

    canonical = lane._canonicalize_script_spoken_text(script)

    assert script.model_dump(mode="json") == before
    assert canonical is not script
    assert [line.text for line in canonical.lines] == [
        "Spoken character line.",
        "(aside) Preserve this skipped row.",
        "(music bed) Preserve this music row.",
        "Spoken announcer line.",
    ]
    assert all(
        copied is not original
        for copied, original in zip(canonical.lines, script.lines)
    )


def test_frozen_raw_text_grandfather_without_marker_validates_unchanged():
    raw_text = "(softly) A grandfathered raw spoken line."
    raw_script = _identity_script([_identity_line("l001", raw_text)])
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    frozen = {
        "meta": {
            "source_bank": "scifi_news",
            "freeze_verdict": "frozen_clean",
            "scifi_codex": {
                "accepted_lines": {"l001": raw_text},
                "line_text_sha256": {"l001": raw_hash},
            },
        },
        "lines": [{
            "line_id": "l001",
            "text": raw_text,
            "skip": False,
        }],
    }
    content_authorship.stamp_receipt(
        frozen,
        owner_bank="scifi_news",
        accepted_artifacts={"final_script": raw_script},
    )
    before = copy.deepcopy(frozen)

    content_authorship.validate_receipt(frozen)
    assert freeze_cascade._readonly_structural_validation(frozen) == []
    lane._CodexTailFinalizer({"l001": raw_text})._proof(frozen)

    assert frozen == before
    assert "script_text_identity_generation" not in frozen["meta"][
        "scifi_codex"
    ]
    assert frozen["meta"]["scifi_codex"]["line_text_sha256"] == {
        "l001": raw_hash,
    }


@pytest.mark.parametrize(
    ("target_words", "words_per_line"),
    [(120, 1), (320, 80)],
)
def test_arbitrary_spoken_prose_length_publishes_with_telemetry_only(
    monkeypatch, tmp_path, target_words, words_per_line,
):
    def prose(_line_id: str, _index: int) -> str:
        return " ".join(["signal"] * words_per_line)

    state = _run_lane(
        monkeypatch,
        tmp_path,
        target_words=target_words,
        line_text=prose,
    )

    expected_lines = lane._codex_target_beat_count(target_words, 2)
    expected_actual = expected_lines * words_per_line
    receipt = state["meta"]["scifi_codex"]["word_receipt"]
    assert receipt["target_status"] == "valid"
    assert receipt["actual_total_voiced_words"] == expected_actual
    assert receipt["actual_total_voiced_words"] != target_words
    assert len(state["ledger"].data["lines"]) == expected_lines
    assert state["assemble_calls"] == 1
    coverage = state["ledger"].data["meta"]["content_authorship"]["coverage"]
    assert coverage["voiced_line_count"] == expected_lines
    assert coverage["proved_line_count"] == expected_lines
    assert coverage["complete"] is True


def test_divergent_fiction_is_canonical_when_ledger_contract_is_intact(
    monkeypatch, tmp_path,
):
    text = (
        "Far from the observatory, a gardener teaches moonflowers to sing."
    )
    state = _run_lane(
        monkeypatch,
        tmp_path,
        line_text=lambda _line_id, _index: text,
    )

    spoken = [
        row["text"] for row in state["ledger"].data["lines"]
        if not row.get("skip")
    ]
    assert spoken
    assert set(spoken) == {text}
    assert state["assemble_calls"] == 1
    coverage = state["ledger"].data["meta"]["content_authorship"]["coverage"]
    assert coverage["complete"] is True


def test_post_safety_structure_drift_fails_before_assembly(monkeypatch, tmp_path):
    def drift(script: lane.ScriptArtifactV4) -> lane.ScriptArtifactV4:
        first = script.lines[0].model_copy(update={"beat_id": "b999"})
        return script.model_copy(update={"lines": [first, *script.lines[1:]]})

    state: dict[str, Any] = {}
    with pytest.raises(
        lane.CodexGraphError,
        match="safety-cleaned P5 artifact violated structure",
    ):
        _run_lane(
            monkeypatch,
            tmp_path,
            transform_script=drift,
            state=state,
        )

    assert state["p3_calls"] == state["p5_calls"] == 1
    assert state["assemble_calls"] == 0


def test_post_acceptance_spoken_safety_residual_fails_before_assembly(
    monkeypatch, tmp_path,
):
    def inject_post_acceptance_drift(
        script: lane.ScriptArtifactV4,
    ) -> lane.ScriptArtifactV4:
        first = script.lines[0].model_copy(
            update={"text": "A gun rests beside the receiver."}
        )
        return script.model_copy(update={"lines": [first, *script.lines[1:]]})

    def unavailable(*_args, **_kwargs) -> str:
        raise RuntimeError("safety writer unavailable")

    state: dict[str, Any] = {}
    with pytest.raises(
        lane.CodexSpokenTextError,
        match="terminal spoken-safety cleanup failed",
    ):
        _run_lane(
            monkeypatch,
            tmp_path,
            transform_script=inject_post_acceptance_drift,
            technical_generator=unavailable,
            state=state,
        )

    assert state["p3_calls"] == state["p5_calls"] == 1
    assert state["assemble_calls"] == 0
