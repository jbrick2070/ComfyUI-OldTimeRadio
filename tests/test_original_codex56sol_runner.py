import json
from contextlib import contextmanager

import pytest

from nodes import _otr_original_codex56sol as lane
from nodes import _otr_story_routing as routing
from nodes import _otr_story_rules as story_rules
from nodes import production_ledger as ledger_mod


DRAW = {"deck_id":"deck","deck_sha256":"a"*64,"constraint_id":"c01","lost_objects":["stamp","mitten","card"],"acoustic_device":"a grille repeats phrases","helpful_ending":"return every item","device_spoken_anchor":"grille","resolution_spoken_anchor":"every item is returned"}


def _fixtures():
    card = {"possibility_id":"p1","title_seed":"Echo Desk","premise":"A stamp, mitten, and card are traced through one helpful echo.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":DRAW["lost_objects"],"acoustic_device":DRAW["acoustic_device"],"shared_cause":"A grille repeats phrases and carries desk sounds.","clue_plan":["the stamp's shelf number repeats","the mitten makes a wool scrape","the card rustles beside the grille"],"helpful_resolution":"Mara maps the echo and returns every item."}
    truth = {"title":"The Helpful Echo","premise":"A station desk solves three lost-item calls through one echo.","setting":"A small community radio station","desk_operator_name":"Mara Vale","caller_threads":[{"thread_id":"t1","caller_name":"Ivo Reed","lost_object":"stamp","practical_need":"finish a library return"},{"thread_id":"t2","caller_name":"Nell Park","lost_object":"mitten","practical_need":"complete a pair"},{"thread_id":"t3","caller_name":"Oren Bell","lost_object":"card","practical_need":"finish a recipe"}],"causal_steps":[{"step_id":"s1","cause":"The grille is loose.","effect":"Desk sounds travel to the call booth."},{"step_id":"s2","cause":"Objects rest beside the grille.","effect":"Their sounds repeat on calls."}],"audible_clues":[{"clue_id":"q1","thread_id":"t1","sound_or_phrase":"a repeated shelf number","implication":"the stamp is near the desk"},{"clue_id":"q2","thread_id":"t2","sound_or_phrase":"a soft wool scrape","implication":"the mitten is beside the grille"},{"clue_id":"q3","thread_id":"t3","sound_or_phrase":"a crisp card rustle","implication":"the card is beside the grille"}],"reveal":"The loose grille carries sounds from the lost-and-found shelf.","resolution_links":[{"thread_id":"t1","action":"Mara checks the numbered shelf.","result":"Ivo receives the stamp."},{"thread_id":"t2","action":"Mara checks beside the grille.","result":"Nell receives the mitten."},{"thread_id":"t3","action":"Mara checks the card tray.","result":"Oren receives the card."}]}
    cast = [{"char_id":"announcer","name":"Announcer","role":"announcer","character_description":"Brief station host"},{"char_id":"c01","name":"Mara Vale","role":"desk_operator","character_description":"Warm precise desk operator"},{"char_id":"c02","name":"Ivo Reed","role":"caller","character_description":"Patient library volunteer"}]
    scenes = [{"scene_id":"scene_01","description":"Calls reach the desk.","env":"radio station desk"},{"scene_id":"scene_02","description":"Mara resolves the echo.","env":"lost-and-found shelf"}]
    shots = [{"shot_id":"shot_01","scene_id":"scene_01","description":"Mara at the desk.","visual_prompt":"Warm radio desk, Mara listening, amber practical light"},{"shot_id":"shot_02","scene_id":"scene_02","description":"The shelf and grille.","visual_prompt":"Orderly shelf beside a loose grille, soft morning light"}]
    truth["interpretations"] = [
      {"interpretation_id":"i1","clue_ids":["q1"],"explanation":"The shelf number belongs to the stamp.","is_true":True},
      {"interpretation_id":"i2","clue_ids":["q2"],"explanation":"A caller may be brushing a wool coat.","is_true":False},
    ]
    beats = [
      {"beat_id":"b1","shot_id":"shot_01","scene_id":"scene_01","char_id":"announcer","speaker":"Announcer","line_intent":{"intent":"identify the station","arc_phase":"opening","clue_ids":[]}},
      {"beat_id":"b2","shot_id":"shot_01","scene_id":"scene_01","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"name the stamp and its repeated shelf number","arc_phase":"rising","clue_ids":["q1"]}},
      {"beat_id":"b3","shot_id":"shot_01","scene_id":"scene_01","char_id":"c02","speaker":"Ivo Reed","line_intent":{"intent":"name the mitten scrape and card rustle","arc_phase":"rising","clue_ids":["q2","q3"]}},
      {"beat_id":"b4","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"reveal the grille cause","arc_phase":"reveal","clue_ids":[]}},
      {"beat_id":"b5","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"confirm every item is returned","arc_phase":"closing","clue_ids":[]}}]
    score = {"title":truth["title"],"premise":truth["premise"],"setting":truth["setting"],"cast":cast,"scenes":scenes,"shots":shots,"beats":beats,"orientation_beat_id":"b1","reveal_beat_id":"b4","closure_beat_id":"b5","opening_music":{"description":"A curious warm station motif.","generation_prompt":"Warm plucked strings and soft dial tones, no vocals"},"closing_music":{"description":"The motif resolves gently.","generation_prompt":"Gentle resolved plucked strings, no vocals"}}
    script_lines = [{"line_id":f"line_{i:03d}","char_id":b["char_id"],"speaker":b["speaker"],"text":t} for i,(b,t) in enumerate(zip(beats,["Lost and Found Frequency is listening.","The stamp's shelf number repeats after every answer.","The mitten scrapes softly, and the card rustles beside it.","The loose grille carries sounds from this shelf.","The echo is mapped; every item is returned." ]),1)]
    slate = {"possibilities":[card,{**card,"possibility_id":"p2","title_seed":"Whisper Shelf"},{**card,"possibility_id":"p3","title_seed":"Three Notes Home"},{**card,"possibility_id":"p4","title_seed":"Kind Echo"}]}
    script = {"title":"The Helpful Echo","lines":script_lines}
    return {"card":card,"truth":truth,"score":score,"slate":slate,"script":script,
            "triage":{"selected_possibility_id":"p1","findings":[]},
            "fair":{"accepted":True,"findings":[]},
            "listener":{"understood_cause":"The grille carried shelf sounds.","understood_resolution":"Mara returned the objects.","findings":[],"optional_notes":[]},
            "audit":{"accepted":True,"findings":[],"warnings":[]}}


def _responses():
    f = _fixtures()
    return [f["slate"], f["triage"], f["truth"], f["fair"], f["score"],
            f["script"], f["listener"], f["audit"]]


class Scheduler:
    @contextmanager
    def helper_context(self, _name):
        yield


def test_mocked_complete_runner_fills_closed_ledger(tmp_path):
    responses = iter(_responses())
    calls = []
    def generate(_messages, **_kwargs):
        calls.append(1)
        return json.dumps(next(responses))
    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(episode_id="codex56_mock", out_dir=str(tmp_path))
    meta = led.data.setdefault("meta", {})
    meta.update({"source_bank":"original_codex56sol","source_meta":{"constraint_draw":DRAW}})
    parts = lane.run_original_codex56sol_episode(payload={"seed_text":json.dumps(DRAW)},pack=pack,resolved={"target_words":30,"num_characters":3},led=led,meta=meta,creative_fn=generate,technical_fn=generate,slot_scheduler=Scheduler(),source_bank_row=None,story_rules=rules,episode_root=tmp_path,episode_id="codex56_mock")
    assert len(calls) == 8
    assert parts.run_story_spine is False
    assert len(led.data["cast"]) == 3
    assert len(led.data["lines"]) == 5
    assert [(m["cue_id"],m["placement"]) for m in led.data["music"]] == [("opening","opening"),("closing","closing")]
    assert led.data["meta"]["content_authorship"]["coverage"]["complete"] is True
    lane_meta = led.data["meta"]["original_codex56sol"]
    assert lane_meta["grounding_receipt"]["complete"] is True
    assert set(lane_meta["accepted_artifacts"]) == {
        "selected_possibility", "triage", "truth_map", "fair_play_report",
        "grounding_contract", "broadcast_score", "performance_script",
        "blind_listener_report", "final_contract_audit",
    }


def _grounding_fixture():
    draw = lane.ConstraintDraw.model_validate(DRAW)
    truth = lane.AudibleTruthMap.model_validate(_fixtures()["truth"])
    return lane._build_grounding_contract(draw, truth)


def test_script_grounding_requires_objects_on_clue_lines_and_fixed_landmarks():
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    manifest = lane._compile_manifest(score)
    contract = _grounding_fixture()
    script = lane.PerformanceScript.model_validate(fixtures["script"])
    assert lane._validate_script_grounding(contract, manifest, script) is None

    missing_object = script.model_copy(deep=True)
    missing_object.lines[2].text = "I hear two soft sounds beside the shelf."
    assert lane._validate_script_grounding(
        contract, manifest, missing_object,
    ) == (
        "spoken script is missing lost-object anchor 'mitten'"
    )

    wrong_reveal = script.model_copy(deep=True)
    wrong_reveal.lines[3].text = "A mysterious pattern carries the sounds."
    assert lane._validate_script_grounding(
        contract, manifest, wrong_reveal,
    ) == "reveal line must speak exact device anchor 'grille'"

    wrong_closure = script.model_copy(deep=True)
    wrong_closure.lines[4].text = "The desk is quiet again."
    assert lane._validate_script_grounding(
        contract, manifest, wrong_closure,
    ) == (
        "closure line must speak exact resolution anchor "
        "'every item is returned'"
    )


def test_production_muted_melody_detour_is_rejected_before_listener():
    contract = _grounding_fixture()
    manifest_rows = []
    for index in range(1, 8):
        line_id = f"line_{index:03d}"
        if index == 1:
            phase, clue_ids = "opening", []
        elif index == 6:
            phase, clue_ids = "reveal", []
        elif index == 7:
            phase, clue_ids = "closing", []
        else:
            phase = "rising"
            clue_ids = [["q1"], ["q2"], ["q3"], []][index - 2]
        manifest_rows.append({
            "line_id": line_id, "beat_id": f"b{index}",
            "shot_id": f"sh{index}", "scene_id": "s1",
            "char_id": "announcer" if index == 1 else "op1",
            "speaker": "System Voice" if index == 1 else "Elara Vance",
            "speaker_role": "announcer" if index == 1 else "character",
            "boundary": "shot_start", "arc_phase": phase,
            "intent": "production regression", "clue_ids": clue_ids,
        })
    manifest = lane.ClosedLineManifest.model_validate({
        "lines": manifest_rows,
        "orientation_line_id": "line_001",
        "reveal_line_id": "line_006",
        "closure_line_id": "line_007",
    })
    texts = [
        "Initiating artifact sequence analysis protocol alpha.",
        "The primary resonance signature remains stubbornly erratic across all known frequencies.",
        "The isotopic decay rate suggests a highly structured, ancient origin.",
        "The tertiary harmonics indicate decay and instability.",
        "If I stabilize the micro-vibrations, the pattern might lock into place.",
        "It isn't a melody; it is a chromatic key demanding completion.",
        "Stabilizing the micro-vibrations resolves the sequence.",
    ]
    script = lane.PerformanceScript.model_validate({
        "title": "The Muted Melody",
        "lines": [{
            "line_id": row.line_id, "char_id": row.char_id,
            "speaker": row.speaker, "text": text,
        } for row, text in zip(manifest.lines, texts)],
    })
    assert lane._validate_script_grounding(
        contract, manifest, script,
    ) == "spoken script is missing lost-object anchor 'stamp'"


def test_grounding_anchor_matching_is_nfkc_casefolded():
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    manifest = lane._compile_manifest(score)
    contract = _grounding_fixture()
    script = lane.PerformanceScript.model_validate(fixtures["script"])
    script.lines[1].text = script.lines[1].text.replace("stamp", "ＳＴＡＭＰ")
    assert lane._validate_script_grounding(contract, manifest, script) is None


def test_blind_listener_must_infer_a_device_anchor_token():
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    manifest = lane._compile_manifest(score)
    script = lane.PerformanceScript.model_validate(fixtures["script"])
    packet = lane._preceding_lines(manifest, script)
    report = lane.BlindListenerReport.model_validate({
        "understood_cause": "An unstable resonance signature.",
        "understood_resolution": "Unknown.",
        "findings": [], "optional_notes": [],
    })
    blocks = lane._listener_blocks(
        report, {row["line_id"] for row in packet}, _grounding_fixture(),
        packet[-1]["line_id"],
    )
    assert [(row.line_id, row.category) for row in blocks] == [
        (packet[-1]["line_id"], "Cause grounding"),
    ]


def test_blocking_listener_retake_is_rechecked_blind_without_contract(
        tmp_path):
    fixtures = _fixtures()
    responses = _responses()
    responses[6] = {
        "understood_cause": "An unstable resonance signature.",
        "understood_resolution": "Unknown.",
        "findings": [], "optional_notes": [],
    }
    responses.insert(7, fixtures["script"])
    responses.insert(8, fixtures["listener"])
    queued = iter(responses)
    prompts = []

    def generate(messages, **_kwargs):
        prompts.append(messages)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="listener_rerun", out_dir=str(tmp_path),
    )
    meta = led.data.setdefault("meta", {})
    meta.update({"source_bank": "original_codex56sol",
                 "source_meta": {"constraint_draw": DRAW}})
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="listener_rerun",
    )
    assert len(prompts) == 10
    rerun_input = json.loads(prompts[8][1]["content"])
    assert set(rerun_input) == {"preceding_lines"}
    assert "grounding_contract" not in prompts[8][1]["content"]


def test_nonaccepted_final_audit_without_actionable_finding_fails_closed(
        tmp_path):
    responses = _responses()
    responses[-1] = {"accepted": False, "findings": [], "warnings": []}
    queued = iter(responses)

    def generate(_messages, **_kwargs):
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(episode_id="audit_false", out_dir=str(tmp_path))
    meta = led.data.setdefault("meta", {})
    meta.update({"source_bank": "original_codex56sol",
                 "source_meta": {"constraint_draw": DRAW}})
    with pytest.raises(
        lane.OriginalCodex56SolContractError,
        match="without actionable grounded findings",
    ):
        lane.run_original_codex56sol_episode(
            payload={"seed_text": json.dumps(DRAW)}, pack=pack,
            resolved={"target_words": 30, "num_characters": 3}, led=led,
            meta=meta, creative_fn=generate, technical_fn=generate,
            slot_scheduler=Scheduler(), source_bank_row=None,
            story_rules=rules, episode_root=tmp_path,
            episode_id="audit_false",
        )


def test_visual_style_cannot_change_any_codex56_story_message(tmp_path):
    def capture(style_id, out_dir):
        queued = iter(_responses())
        messages = []

        def generate(prompt, **_kwargs):
            messages.append(json.loads(json.dumps(prompt)))
            return json.dumps(next(queued))

        led = ledger_mod.new_ledger(
            episode_id="visual_isolation", out_dir=str(out_dir),
        )
        meta = led.data.setdefault("meta", {})
        meta.update({"source_bank": "original_codex56sol",
                     "source_meta": {"constraint_draw": DRAW}})
        lane.run_original_codex56sol_episode(
            payload={"seed_text": json.dumps(DRAW)},
            pack=routing.resolve_story_pack("original_codex56sol"),
            resolved={"target_words": 30, "num_characters": 3,
                      "visual_style": style_id},
            led=led, meta=meta, creative_fn=generate, technical_fn=generate,
            slot_scheduler=Scheduler(), source_bank_row=None,
            story_rules=story_rules.resolve_story_rules("original_codex56sol"),
            episode_root=out_dir, episode_id="visual_isolation",
        )
        return messages

    routing._REGISTRY = None
    story_rules._clear_caches()
    first = capture("sci_fi_radio", tmp_path / "a")
    second = capture("video_art", tmp_path / "b")
    assert first == second
    serialized = json.dumps(first)
    assert "sci_fi_radio" not in serialized
    assert "video_art" not in serialized


def test_p3_collection_placement_repair_does_not_spend_an_llm_call(tmp_path):
    responses = _responses()
    responses[2]["caller_threads"][0]["causal_steps"] = [{
        "step_id": "nested_extra", "cause": "A redundant detail.",
        "effect": "The top-level graph remains authoritative.",
    }]
    queued = iter(responses)
    calls = []

    def generate(_messages, **_kwargs):
        calls.append(1)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p3_placement", out_dir=str(tmp_path),
    )
    meta = led.data.setdefault("meta", {})
    meta.update({
        "source_bank": "original_codex56sol",
        "source_meta": {"constraint_draw": DRAW},
    })
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="codex56_p3_placement",
    )
    assert len(calls) == 8


def test_cross_artifact_validators_return_retryable_error_strings():
    draw = lane.ConstraintDraw.model_validate(DRAW)
    bad = lane.PossibilitySlate.model_validate({"possibilities": [
        {"possibility_id":f"p{i}","title_seed":"Other","premise":"Other objects.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":["other","items","here"],"acoustic_device":"A storm.","shared_cause":"A storm.","clue_plan":["one","two","three"],"helpful_resolution":"They meet."}
        for i in range(1,5)
    ]})
    error = lane._validate_slate(bad, draw)
    assert isinstance(error, str) and "copied verbatim" in error


def test_ungrounded_fair_play_opinion_is_not_a_fatal_coordinate():
    report = lane.FairPlayReport.model_validate({
        "accepted": False,
        "findings": [{"category":"Helpful Ending","detail":"Could be warmer","blocking":True}],
    })
    truth = lane.AudibleTruthMap.model_validate(_fixtures()["truth"])
    assert lane._corroborated_fair_blocks(report, truth) == []


def test_structural_numeric_ids_canonicalize_without_authored_prose_change():
    card = lane.PossibilityCard.model_validate({
        "possibility_id":1,"title_seed":"Echo","premise":"A premise.",
        "desk_operator":{"name":"Mara Vale"},"callers":[],
        "lost_objects":DRAW["lost_objects"],
        "acoustic_device":DRAW["acoustic_device"],"shared_cause":"Echo",
        "clue_plan":["one","two","three"],"helpful_resolution":"Returned.",
    })
    assert card.possibility_id == "1"

    cast = lane.CastConcept.model_validate({
        "char_id": 2, "name": "Caller", "role": "caller",
        "character_description": "Patient",
    })
    scene = lane.SceneConcept.model_validate({
        "scene_id": 3, "description": "Office", "env": "indoors",
    })
    shot = lane.ShotConcept.model_validate({
        "shot_id": 4, "scene_id": 3, "description": "Desk",
        "visual_prompt": "A warm desk",
    })
    beat = lane.BeatConcept.model_validate({
        "beat_id": 5, "shot_id": 4, "scene_id": 3, "char_id": 2,
        "speaker": "Caller",
        "line_intent": {"intent": "Ask", "arc_phase": "rising",
                        "clue_ids": []},
    })
    assert (cast.char_id, scene.scene_id, shot.shot_id, shot.scene_id,
            beat.beat_id, beat.shot_id, beat.scene_id, beat.char_id) == (
        "2", "3", "4", "3", "5", "4", "3", "2",
    )


def test_freeform_non_owner_cast_role_normalizes_to_caller():
    concept = lane.CastConcept.model_validate({
        "char_id":"c02","name":"Ivo Reed","role":"resident",
        "character_description":"A patient caller",
    })
    assert concept.role == "caller"


def test_shot_environment_is_typed_optional_authored_metadata():
    shot = lane.ShotConcept.model_validate({
        "shot_id":"shot_1","scene_id":"scene_1","description":"Desk",
        "visual_prompt":"Warm desk under amber light","env":"front office",
    })
    assert shot.env == "front office"


def test_manifest_optional_landmark_markers_are_typed_and_checked():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    manifest = lane._compile_manifest(score)
    manifest = manifest.model_copy(update={
        "lines": [manifest.lines[0].model_copy(update={
            "orientation": None, "clue": None, "reveal": None,
            "closure": None,
        }), *manifest.lines[1:]],
    })
    assert lane._validate_manifest(score, manifest) is None
    bad = manifest.model_copy(update={
        "lines": [manifest.lines[0].model_copy(update={"reveal": True}),
                  *manifest.lines[1:]],
    })
    assert "asserts reveal" in lane._validate_manifest(score, bad)


def test_manifest_unknown_enum_value_still_fails_loud():
    data = {
        "line_id":"l1", "beat_id":"b1", "shot_id":"s1",
        "scene_id":"scene1", "char_id":"c1", "speaker":"Caller",
        "speaker_role":"sound_effect", "boundary":"somewhere",
        "arc_phase":"epilogue", "intent":"Speak",
    }
    try:
        lane.ManifestLine.model_validate(data)
    except ValueError as exc:
        message = str(exc)
        assert "speaker_role" in message
        assert "boundary" in message
        assert "arc_phase" in message
    else:
        raise AssertionError("unknown manifest enums must not be accepted")


def test_p6_text_safety_failure_is_a_retryable_error_string():
    script_data = _fixtures()["script"]
    script_data["lines"][2]["text"] = "We must kill the transmitter."
    script = lane.PerformanceScript.model_validate(script_data)
    rules = type("Rules", (), {
        "banned_phrases": ("kill",),
        "stage_business": (),
    })()
    error = lane._validate_text(script, rules)
    assert "forbidden term 'kill'" in error


def test_p6_cross_artifact_graph_failure_is_a_retryable_error_string():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    manifest = lane._compile_manifest(score)
    script_data = _fixtures()["script"]
    script_data["lines"][0]["char_id"] = "wrong_character"
    script = lane.PerformanceScript.model_validate(script_data)
    assert lane._validate_graph(score, manifest, script) == (
        "script roster differs from manifest"
    )


def test_safety_is_caught_before_a_detail_becomes_manifest_immutable():
    score_data = _fixtures()["score"]
    score_data["beats"][2]["line_intent"]["intent"] = "Name To Kill a Mockingbird"
    score = lane.BroadcastScore.model_validate(score_data)
    rules = type("Rules", (), {
        "banned_phrases": ("kill",),
        "stage_business": (),
    })()
    error = lane._validate_authored_surface(score, rules)
    assert error == (
        "authored field 'beats.2.line_intent.intent' contains forbidden term "
        "'kill'; replace every cited authored detail"
    )


def test_safe_score_surface_passes_before_manifest_lock():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    rules = type("Rules", (), {
        "banned_phrases": ("kill",),
        "stage_business": (),
    })()
    assert lane._validate_authored_surface(score, rules) is None


def test_truth_map_interpretations_and_references_are_retryable():
    data = _fixtures()["truth"]
    data["interpretations"][0]["clue_ids"] = ["missing_clue"]
    truth = lane.AudibleTruthMap.model_validate(data)
    assert lane._validate_truth_map(truth) == (
        "every interpretation clue_id must resolve"
    )


def test_truth_map_requires_one_thread_and_resolution_per_selected_object():
    fixtures = _fixtures()
    selected = lane.PossibilityCard.model_validate(fixtures["slate"]["possibilities"][0])
    complete = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    assert lane._validate_truth_map(complete, selected) is None

    incomplete_data = fixtures["truth"]
    incomplete_data["caller_threads"] = incomplete_data["caller_threads"][:2]
    incomplete_data["resolution_links"] = incomplete_data["resolution_links"][:2]
    incomplete = lane.AudibleTruthMap.model_validate(incomplete_data)
    assert lane._validate_truth_map(incomplete, selected) == (
        "caller_threads must contain exactly one row per selected lost object, "
        "with one lost_object field per row"
    )


def test_score_requires_exact_clue_coverage_and_contiguous_shots():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    score_data = fixtures["score"]
    score_data["beats"][2]["line_intent"]["clue_ids"] = ["q2"]
    score = lane.BroadcastScore.model_validate(score_data)
    assert "cover every truth-map clue" in lane._validate_score(score, truth)

    score_data = _fixtures()["score"]
    score_data["beats"][2], score_data["beats"][3] = (
        score_data["beats"][3], score_data["beats"][2])
    score = lane.BroadcastScore.model_validate(score_data)
    assert lane._validate_score(score, truth) == (
        "beats for each shot must form one contiguous block"
    )


def test_score_must_carry_spoken_anchor_intents_before_script_generation():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    grounding = _grounding_fixture()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    assert lane._validate_score(score, truth, grounding) is None

    drifted = score.model_copy(deep=True)
    drifted.beats[1].line_intent.intent = "analyze the first component"
    assert lane._validate_score(drifted, truth, grounding) == (
        "score needs a non-announcer clue intent naming exact lost-object "
        "anchor 'stamp'"
    )

    drifted = score.model_copy(deep=True)
    drifted.beats[3].line_intent.intent = "reveal a resonance signature"
    assert lane._validate_score(drifted, truth, grounding) == (
        "reveal intent must name exact device anchor 'grille'"
    )


def test_duplicate_score_clues_repair_deterministically_at_first_placement():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    score_data = fixtures["score"]
    score_data["beats"][2]["line_intent"]["clue_ids"].insert(0, "q1")
    score = lane.BroadcastScore.model_validate(score_data)
    assert lane._validate_score(score, truth) == (
        "each truth-map clue must be assigned to exactly one line intent"
    )

    repaired = lane._repair_duplicate_score_clues(score, truth)
    assert repaired is not None
    assert repaired.beats[1].line_intent.clue_ids == ["q1"]
    assert repaired.beats[2].line_intent.clue_ids == ["q2", "q3"]
    assert lane._validate_score(repaired, truth) is None
    assert score.beats[2].line_intent.clue_ids == ["q1", "q2", "q3"]


def test_p3_repair_keeps_authoritative_top_level_and_removes_nested_extras():
    fixtures = _fixtures()
    selected = lane.PossibilityCard.model_validate(fixtures["card"])
    data = fixtures["truth"]
    expected_steps = list(data["causal_steps"])
    data["caller_threads"][0]["causal_steps"] = [{
        "step_id": "extra_step", "cause": "An extra detail.",
        "effect": "It is not part of the typed top-level graph.",
    }]
    repaired = lane._repair_truth_map_collection_placement(
        json.dumps(data), selected,
    )
    assert repaired is not None
    assert [row.model_dump() for row in repaired.causal_steps] == expected_steps
    assert all(
        set(row.model_dump()) == {
            "thread_id", "caller_name", "lost_object", "practical_need",
        }
        for row in repaired.caller_threads
    )


def test_p3_repair_lifts_missing_top_level_collection_verbatim():
    fixtures = _fixtures()
    selected = lane.PossibilityCard.model_validate(fixtures["card"])
    data = fixtures["truth"]
    expected_steps = data.pop("causal_steps")
    data["caller_threads"][0]["causal_steps"] = [expected_steps[0]]
    data["caller_threads"][1]["causal_steps"] = [expected_steps[1]]
    repaired = lane._repair_truth_map_collection_placement(
        json.dumps(data), selected,
    )
    assert repaired is not None
    assert [row.model_dump() for row in repaired.causal_steps] == expected_steps


def test_p3_repair_fails_closed_on_unknown_or_graph_invalid_shapes():
    fixtures = _fixtures()
    selected = lane.PossibilityCard.model_validate(fixtures["card"])
    non_list = fixtures["truth"]
    non_list["caller_threads"][0]["causal_steps"] = "not a list"
    assert lane._repair_truth_map_collection_placement(
        json.dumps(non_list), selected,
    ) is None

    duplicate = _fixtures()["truth"]
    duplicate.pop("causal_steps")
    repeated = {
        "step_id": "same", "cause": "Repeated.", "effect": "Ambiguous.",
    }
    duplicate["caller_threads"][0]["causal_steps"] = [repeated]
    duplicate["caller_threads"][1]["causal_steps"] = [dict(repeated)]
    assert lane._repair_truth_map_collection_placement(
        json.dumps(duplicate), selected,
    ) is None

    unknown = _fixtures()["truth"]
    unknown["caller_threads"][0]["unknown_collection"] = []
    unknown["caller_threads"][0]["causal_steps"] = []
    assert lane._repair_truth_map_collection_placement(
        json.dumps(unknown), selected,
    ) is None


def test_p5_repair_keeps_authoritative_top_level_and_removes_nested_extras():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    data = fixtures["score"]
    expected_shots = list(data["shots"])
    data["scenes"][0]["shots"] = [{
        "shot_id": "extra_shot", "scene_id": "scene_01",
        "description": "A redundant nested shot.",
        "visual_prompt": "A redundant nested composition.",
    }]
    repaired = lane._repair_score_collection_placement(
        json.dumps(data), truth,
    )
    assert repaired is not None
    assert [row.model_dump(exclude_defaults=True) for row in repaired.shots] == expected_shots
    assert all("shots" not in row.model_dump() for row in repaired.scenes)


def test_p5_repair_drops_non_authoritative_music_bookkeeping_only():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    data = fixtures["score"]
    expected_opening = dict(data["opening_music"])
    expected_closing = dict(data["closing_music"])
    data["opening_music"]["music_file"] = "opening_music.mp3"
    data["closing_music"]["music_file"] = "closing_music.mp3"
    repaired = lane._repair_score_collection_placement(
        json.dumps(data), truth,
    )
    assert repaired is not None
    assert repaired.opening_music.model_dump() == expected_opening
    assert repaired.closing_music.model_dump() == expected_closing


def test_p5_repair_lifts_missing_top_level_shots_and_beats_verbatim():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    data = fixtures["score"]
    expected_shots = data.pop("shots")
    expected_beats = data.pop("beats")
    beats_by_shot = {
        shot["shot_id"]: [
            beat for beat in expected_beats if beat["shot_id"] == shot["shot_id"]
        ]
        for shot in expected_shots
    }
    for scene in data["scenes"]:
        scene["shots"] = []
    for shot in expected_shots:
        nested_shot = dict(shot)
        nested_shot["beats"] = beats_by_shot[shot["shot_id"]]
        target = next(
            scene for scene in data["scenes"]
            if scene["scene_id"] == shot["scene_id"]
        )
        target["shots"].append(nested_shot)
    repaired = lane._repair_score_collection_placement(
        json.dumps(data), truth,
    )
    assert repaired is not None
    assert [row.model_dump(exclude_defaults=True) for row in repaired.shots] == expected_shots
    assert [row.model_dump() for row in repaired.beats] == expected_beats


def test_p5_repair_fails_closed_on_unknown_or_graph_invalid_shapes():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    non_list = fixtures["score"]
    non_list["scenes"][0]["shots"] = "not a list"
    assert lane._repair_score_collection_placement(
        json.dumps(non_list), truth,
    ) is None

    duplicate = _fixtures()["score"]
    duplicate["shots"] = []
    duplicate["scenes"][0]["shots"] = [
        dict(duplicate["scenes"][0], shot_id="same", scene_id="scene_01",
             visual_prompt="One"),
        dict(duplicate["scenes"][0], shot_id="same", scene_id="scene_01",
             visual_prompt="Two"),
    ]
    assert lane._repair_score_collection_placement(
        json.dumps(duplicate), truth,
    ) is None

    unknown = _fixtures()["score"]
    unknown["scenes"][0]["unknown_collection"] = []
    unknown["scenes"][0]["shots"] = []
    assert lane._repair_score_collection_placement(
        json.dumps(unknown), truth,
    ) is None

    partial = _fixtures()["score"]
    missing_shot = partial["shots"][1]
    partial["shots"] = partial["shots"][:1]
    partial["scenes"][1]["shots"] = [missing_shot]
    assert lane._repair_score_collection_placement(
        json.dumps(partial), truth,
    ) is None


def test_p5_collection_placement_repair_does_not_spend_an_llm_call(tmp_path):
    responses = _responses()
    responses[4]["opening_music"]["music_file"] = "opening_music.mp3"
    responses[4]["closing_music"]["music_file"] = "closing_music.mp3"
    responses[4]["scenes"][0]["shots"] = [{
        "shot_id": "nested_extra", "scene_id": "scene_01",
        "description": "A redundant nested shot.",
        "visual_prompt": "A redundant nested composition.",
    }]
    queued = iter(responses)
    calls = []

    def generate(_messages, **_kwargs):
        calls.append(1)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p5_placement", out_dir=str(tmp_path),
    )
    meta = led.data.setdefault("meta", {})
    meta.update({
        "source_bank": "original_codex56sol",
        "source_meta": {"constraint_draw": DRAW},
    })
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="codex56_p5_placement",
    )
    assert len(calls) == 8


def test_p5_placement_repair_preserves_llm_fallback_for_safety(tmp_path):
    responses = _responses()
    corrected_score = json.loads(json.dumps(responses[4]))
    responses[4]["premise"] = "Kill the clue."
    responses[4]["scenes"][0]["shots"] = [{
        "shot_id": "nested_extra", "scene_id": "scene_01",
        "description": "A redundant nested shot.",
        "visual_prompt": "A redundant nested composition.",
    }]
    responses.insert(5, corrected_score)
    queued = iter(responses)
    calls = []
    repair_prompts = []

    def generate(messages, **_kwargs):
        calls.append(1)
        if any("after structural normalization" in row["content"]
               for row in messages):
            repair_prompts.append(messages)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p5_safety_fallback", out_dir=str(tmp_path),
    )
    meta = led.data.setdefault("meta", {})
    meta.update({
        "source_bank": "original_codex56sol",
        "source_meta": {"constraint_draw": DRAW},
    })
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="codex56_p5_safety_fallback",
    )
    assert len(calls) == 9
    assert len(repair_prompts) == 1
    assert "forbidden term 'kill'" in repair_prompts[0][1]["content"]


def test_duplicate_score_clue_repair_does_not_spend_an_llm_call(tmp_path):
    responses = _responses()
    responses[4]["beats"][2]["line_intent"]["clue_ids"].insert(0, "q1")
    queued = iter(responses)
    calls = []

    def generate(_messages, **_kwargs):
        calls.append(1)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_duplicate_clue", out_dir=str(tmp_path),
    )
    meta = led.data.setdefault("meta", {})
    meta.update({"source_bank": "original_codex56sol",
                 "source_meta": {"constraint_draw": DRAW}})
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="codex56_duplicate_clue",
    )
    assert len(calls) == 8


def test_python_manifest_is_repeatable_closed_and_spoiler_safe():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    first = lane._compile_manifest(score)
    second = lane._compile_manifest(score)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [row.boundary for row in first.lines] == [
        "shot_start", "beat_start", "beat_start", "shot_start", "beat_start",
    ]
    assert first.lines[1].clue_ids == ["q1"]
    script = lane.PerformanceScript.model_validate(_fixtures()["script"])
    packet = lane._preceding_lines(first, script)
    packet_ids = {row["line_id"] for row in packet}
    assert first.reveal_line_id not in packet_ids
    assert first.closure_line_id not in packet_ids
    assert all("manifest" not in row for row in packet)


def test_manifest_rejects_duplicate_beat_and_overlapping_landmark():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    manifest = lane._compile_manifest(score)
    duplicate = manifest.model_copy(update={
        "lines": [manifest.lines[0],
                  manifest.lines[1].model_copy(update={"beat_id":"b1"}),
                  *manifest.lines[2:]],
    })
    assert "exactly once" in lane._validate_manifest(score, duplicate)
    overlap = manifest.model_copy(update={
        "reveal_line_id": manifest.orientation_line_id,
    })
    assert "must be distinct" in lane._validate_manifest(score, overlap)


def test_p7_and_p9_only_ground_real_coordinates_and_literal_spans():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    manifest = lane._compile_manifest(score)
    script = lane.PerformanceScript.model_validate(_fixtures()["script"])
    packet = lane._preceding_lines(manifest, script)
    report = lane.BlindListenerReport.model_validate({
        "understood_cause":"unsure", "understood_resolution":"unsure",
        "findings":[
            {"line_id":"invented", "category":"Clue", "detail":"Missing", "blocking":True},
            {"line_id":packet[0]["line_id"], "category":"Clue", "detail":"Missing", "blocking":True},
        ], "optional_notes":[],
    })
    blocks = lane._listener_blocks(report, {row["line_id"] for row in packet})
    assert [row.line_id for row in blocks] == [packet[0]["line_id"]]

    audit = lane.FinalContractAudit.model_validate({
        "accepted":False,
        "findings":[
            {"field_path":"lines.0.text", "item_id":script.lines[0].line_id,
             "exact_span":"not present", "category":"Safety",
             "allowed_correction":"replace", "blocking":True},
            {"field_path":"lines.0.text", "item_id":script.lines[0].line_id,
             "exact_span":"Lost and Found", "category":"Safety",
             "allowed_correction":"replace", "blocking":True},
        ], "warnings":[],
    })
    assert [row.exact_span for row in lane._audit_blocks(audit, script)] == [
        "Lost and Found",
    ]


def test_safety_reports_all_authored_coordinates_in_one_repair():
    data = _fixtures()["slate"]
    data["possibilities"][0]["premise"] = "Kill the signal."
    data["possibilities"][1]["clue_plan"][0] = "A kill phrase repeats."
    slate = lane.PossibilitySlate.model_validate(data)
    rules = type("Rules", (), {"banned_phrases": ("kill",),
                                "stage_business": ()})()
    error = lane._validate_authored_surface(slate, rules)
    assert "possibilities.0.premise" in error
    assert "possibilities.1.clue_plan.0" in error


def test_p3_safety_repair_keeps_safety_and_collection_rules():
    rules = lane._repair_rules("P3", "forbidden term 'kill'")
    assert "Replace EVERY" in rules
    assert "causal_steps MUST" in rules
    assert "change is_true on one existing interpretation" in rules
    assert "an empty clue_ids list is invalid" in rules
    assert "never invent numbered, secondary, tertiary, or suffixed fields" in rules
    assert "move each extra object into its own caller_threads row" in rules


def test_p5_repair_spells_out_required_fields_and_exact_arc_phases():
    rules = lane._repair_rules("P5", "invalid arc phase")
    assert "exactly 4 scenes" in rules
    assert "at least 5 beats" in rules
    assert "never delete a beat" in rules
    assert "one adjacent contiguous block" in rules
    assert "schema-path pseudo-fields" in rules
    assert "singular clue_id is forbidden" in rules
    assert "scene MUST retain a non-empty env" in rules
    assert "shot MUST retain a non-empty visual_prompt" in rules
    assert "orientation_beat_id beat is `opening`" in rules
    assert "every other beat is `rising`" in rules


def test_listener_and_final_audit_repair_envelopes_are_explicit():
    listener = lane._repair_rules("P7", "optional_notes must be a list")
    assert "optional_notes MUST be a list of strings" in listener
    final = lane._repair_rules("P9", "accepted must be a boolean")
    assert "accepted MUST be one boolean" in final
    assert "Never copy the manifest or script" in final
    assert "blocking MUST be a boolean" in final
