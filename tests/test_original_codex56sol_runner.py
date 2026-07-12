import json
from contextlib import contextmanager

from nodes import _otr_original_codex56sol as lane
from nodes import _otr_story_routing as routing
from nodes import _otr_story_rules as story_rules
from nodes import production_ledger as ledger_mod


DRAW = {"deck_id":"deck","deck_sha256":"a"*64,"constraint_id":"c01","lost_objects":["stamp","mitten","card"],"acoustic_device":"a grille repeats phrases","helpful_ending":"return every item"}


def _fixtures():
    card = {"possibility_id":"p1","title_seed":"Echo Desk","premise":"A stamp, mitten, and card are traced through one helpful echo.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":DRAW["lost_objects"],"acoustic_device":DRAW["acoustic_device"],"shared_cause":"A grille repeats phrases and carries desk sounds.","clue_plan":["the stamp's shelf number repeats","the mitten makes a wool scrape","the card rustles beside the grille"],"helpful_resolution":"Mara maps the echo and returns every item."}
    truth = {"title":"The Helpful Echo","premise":"A station desk solves three lost-item calls through one echo.","setting":"A small community radio station","desk_operator_name":"Mara Vale","caller_threads":[{"thread_id":"t1","caller_name":"Ivo Reed","lost_object":"stamp","practical_need":"finish a library return"},{"thread_id":"t2","caller_name":"Nell Park","lost_object":"mitten","practical_need":"complete a pair"}],"causal_steps":[{"step_id":"s1","cause":"The grille is loose.","effect":"Desk sounds travel to the call booth."},{"step_id":"s2","cause":"Objects rest beside the grille.","effect":"Their sounds repeat on calls."}],"audible_clues":[{"clue_id":"q1","thread_id":"t1","sound_or_phrase":"a repeated shelf number","implication":"the stamp is near the desk"},{"clue_id":"q2","thread_id":"t2","sound_or_phrase":"a soft wool scrape","implication":"the mitten is beside the grille"},{"clue_id":"q3","thread_id":"t2","sound_or_phrase":"three matching notes","implication":"one channel carries every clue"}],"reveal":"The loose grille carries sounds from the lost-and-found shelf.","resolution_links":[{"thread_id":"t1","action":"Mara checks the numbered shelf.","result":"Ivo receives the stamp."},{"thread_id":"t2","action":"Mara checks beside the grille.","result":"Nell receives the mitten."}]}
    cast = [{"char_id":"announcer","name":"Announcer","role":"announcer","character_description":"Brief station host"},{"char_id":"c01","name":"Mara Vale","role":"desk_operator","character_description":"Warm precise desk operator"},{"char_id":"c02","name":"Ivo Reed","role":"caller","character_description":"Patient library volunteer"}]
    scenes = [{"scene_id":"scene_01","description":"Calls reach the desk.","env":"radio station desk"},{"scene_id":"scene_02","description":"Mara resolves the echo.","env":"lost-and-found shelf"}]
    shots = [{"shot_id":"shot_01","scene_id":"scene_01","description":"Mara at the desk.","visual_prompt":"Warm radio desk, Mara listening, amber practical light"},{"shot_id":"shot_02","scene_id":"scene_02","description":"The shelf and grille.","visual_prompt":"Orderly shelf beside a loose grille, soft morning light"}]
    truth["interpretations"] = [
      {"interpretation_id":"i1","clue_ids":["q1"],"explanation":"The shelf number belongs to the stamp.","is_true":True},
      {"interpretation_id":"i2","clue_ids":["q2"],"explanation":"A caller may be brushing a wool coat.","is_true":False},
    ]
    beats = [
      {"beat_id":"b1","shot_id":"shot_01","scene_id":"scene_01","char_id":"announcer","speaker":"Announcer","line_intent":{"intent":"identify the station","arc_phase":"opening","clue_ids":[]}},
      {"beat_id":"b2","shot_id":"shot_01","scene_id":"scene_01","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"orient the practical problem","arc_phase":"rising","clue_ids":["q1"]}},
      {"beat_id":"b3","shot_id":"shot_01","scene_id":"scene_01","char_id":"c02","speaker":"Ivo Reed","line_intent":{"intent":"state the shelf-number clues","arc_phase":"rising","clue_ids":["q2","q3"]}},
      {"beat_id":"b4","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"reveal the grille cause","arc_phase":"reveal","clue_ids":[]}},
      {"beat_id":"b5","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"return the item and close helpfully","arc_phase":"closing","clue_ids":[]}}]
    score = {"title":truth["title"],"premise":truth["premise"],"setting":truth["setting"],"cast":cast,"scenes":scenes,"shots":shots,"beats":beats,"orientation_beat_id":"b1","reveal_beat_id":"b4","closure_beat_id":"b5","opening_music":{"description":"A curious warm station motif.","generation_prompt":"Warm plucked strings and soft dial tones, no vocals"},"closing_music":{"description":"The motif resolves gently.","generation_prompt":"Gentle resolved plucked strings, no vocals"}}
    script_lines = [{"line_id":f"line_{i:03d}","char_id":b["char_id"],"speaker":b["speaker"],"text":t} for i,(b,t) in enumerate(zip(beats,["Lost and Found Frequency is listening.","One echo is joining today's calls.","I hear my shelf number after every answer.","The loose grille carries sounds from this shelf.","Your stamp is here, and the grille is secure." ]),1)]
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
