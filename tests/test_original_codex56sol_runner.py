import json
import pathlib
from contextlib import contextmanager

import pytest
from pydantic import BaseModel, ValidationError

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
      {"beat_id":"b3","shot_id":"shot_01","scene_id":"scene_01","char_id":"c02","speaker":"Ivo Reed","line_intent":{"intent":"name the mitten scrape and card rustle beside the grille","arc_phase":"rising","clue_ids":["q2","q3"]}},
      {"beat_id":"b4","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"reveal the grille cause","arc_phase":"reveal","clue_ids":[]}},
      {"beat_id":"b5","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","line_intent":{"intent":"confirm every item is returned","arc_phase":"closing","clue_ids":[]}}]
    score = {"title":truth["title"],"premise":truth["premise"],"setting":truth["setting"],"cast":cast,"scenes":scenes,"shots":shots,"beats":beats,"orientation_beat_id":"b1","reveal_beat_id":"b4","closure_beat_id":"b5","opening_music":{"description":"A curious warm station motif.","generation_prompt":"Warm plucked strings and soft dial tones, no vocals"},"closing_music":{"description":"The motif resolves gently.","generation_prompt":"Gentle resolved plucked strings, no vocals"}}
    script_lines = [{"line_id":f"line_{i:03d}","char_id":b["char_id"],"speaker":b["speaker"],"text":t} for i,(b,t) in enumerate(zip(beats,["Lost and Found Frequency is listening.","The stamp's shelf number repeats after every answer.","The mitten scrapes softly, and the card rustles beside the grille.","The loose grille carries sounds from this shelf.","The echo is mapped; every item is returned." ]),1)]
    slate = {"possibilities":[card,{**card,"possibility_id":"p2","title_seed":"Whisper Shelf"},{**card,"possibility_id":"p3","title_seed":"Three Notes Home"},{**card,"possibility_id":"p4","title_seed":"Kind Echo"}]}
    script = {"title":"The Helpful Echo","lines":script_lines}
    return {"card":card,"truth":truth,"score":score,"slate":slate,"script":script,
            "triage":{"selected_possibility_id":"p1","findings":[]}}


def _responses():
    f = _fixtures()
    return [f["slate"], f["triage"], f["truth"], f["score"], f["script"]]


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
    assert len(calls) == 5
    assert parts.run_story_spine is False
    assert len(led.data["cast"]) == 3
    assert len(led.data["lines"]) == 5
    assert [(m["cue_id"],m["placement"]) for m in led.data["music"]] == [("opening","opening"),("closing","closing")]
    assert led.data["meta"]["content_authorship"]["coverage"]["complete"] is True
    lane_meta = led.data["meta"]["original_codex56sol"]
    assert lane_meta["grounding_receipt"]["complete"] is True
    assert set(lane_meta["accepted_artifacts"]) == {
        "selected_possibility", "triage", "truth_map",
        "grounding_contract", "broadcast_score", "performance_script",
    }
    # Fair play is proven, and the receipt carries the evidence: the device is
    # heard on a clue line before the reveal explains it.
    plant = next(row for row in lane_meta["grounding_receipt"]["evidence"]
                 if row["kind"] == "device_plant")
    assert plant["line_id"] == "line_003"
    assert plant["anchor"] == "grille"


def test_p6_missing_grounding_anchor_uses_small_line_patch(tmp_path):
    responses = _responses()
    responses[4]["lines"][4]["text"] = "The echo is mapped; the desk is quiet."
    responses.insert(5, {
        "replacements": [{
            "line_id": "line_005",
            "text": "The echo is mapped; every item is returned.",
        }],
    })
    calls = []

    def generate(messages, **_kwargs):
        calls.append(messages)
        return json.dumps(responses.pop(0))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p6_grounding_patch", out_dir=str(tmp_path),
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
        episode_root=tmp_path, episode_id="codex56_p6_grounding_patch",
    )

    assert len(calls) == 6
    assert "ScriptLinePatch" in calls[5][0]["content"]
    assert led.data["meta"]["original_codex56sol"]["call_journal"][5][
        "pass_id"
    ] == "P6_grounding_patch"
    accepted_script = led.data["meta"]["original_codex56sol"][
        "accepted_artifacts"
    ]["performance_script"]
    assert accepted_script["lines"][4]["text"] == (
        "The echo is mapped; every item is returned."
    )


def test_p6_line_patch_preserves_anchor_already_valid_on_target():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    score.beats[1].line_intent.clue_ids = []
    score.beats[3].line_intent.clue_ids = ["q1"]
    manifest = lane._compile_manifest(score)
    script = lane.PerformanceScript.model_validate(_fixtures()["script"])
    contract = _grounding_fixture()
    plan = lane._script_grounding_repair_plan(script, manifest, contract)
    assert plan == [{
        "line_id": "line_004",
        "current_text": "The loose grille carries sounds from this shelf.",
        "required_anchors": ["grille", "stamp"],
    }]

    patch = lane.ScriptLinePatch.model_validate({
        "replacements": [{
            "line_id": "line_004",
            "text": "The stamp proves the loose grille carries shelf sounds.",
        }],
    })
    rules = story_rules.resolve_story_rules("original_codex56sol")
    assert lane._validate_script_line_patch_step(
        patch, script, plan, score, manifest, rules,
    ) is None

    # A replacement that smuggles a banned term goes back to the model, at the
    # rung that can still repair it -- never into the episode.
    unsafe_patch = lane.ScriptLinePatch.model_validate({
        "replacements": [{
            "line_id": "line_004",
            "text": "The stamp proves the loose grille can kill the mystery.",
        }],
    })
    assert "forbidden term 'kill'" in (
        lane._validate_script_line_patch_step(
            unsafe_patch, script, plan, score, manifest, rules,
        ) or ""
    )


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


def test_production_muted_melody_detour_is_rejected_by_grounding():
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
    assert len(calls) == 5


def test_cross_artifact_validators_return_retryable_error_strings():
    """A defect Python CANNOT fix still comes back as a retryable string.

    The echoed draw fields are restored (they are inputs, not authorship), but a
    duplicate possibility id is the model's own structural mistake -- Python may
    not pick which card to keep, so it goes back to the ladder.
    """
    draw = lane.ConstraintDraw.model_validate(DRAW)
    duplicated = lane.PossibilitySlate.model_validate({"possibilities": [
        {"possibility_id":"p1","title_seed":"Other","premise":"Other objects.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":["other","items","here"],"acoustic_device":"A storm.","shared_cause":"A storm.","clue_plan":["one","two","three"],"helpful_resolution":"They meet."}
        for _ in range(4)
    ]})
    error = lane._validate_slate(duplicated, draw)
    assert isinstance(error, str) and "unique" in error


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
    error = lane._validate_truth_map(incomplete, selected)
    assert "exactly one row per selected lost object" in error
    assert "VERBATIM" in error


def test_score_requires_exact_clue_coverage_and_contiguous_shots():
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    score_data = fixtures["score"]
    score_data["beats"][2]["line_intent"]["clue_ids"] = ["q2"]
    score = lane.BroadcastScore.model_validate(score_data)
    # The error NAMES the clue the model dropped: a repair prompt that says only
    # "coverage is wrong" gives the model nothing to act on.
    error = lane._validate_score(score, truth)
    assert "assigned to exactly one line intent" in error
    assert "q3" in error

    score_data = _fixtures()["score"]
    score_data["beats"][2], score_data["beats"][3] = (
        score_data["beats"][3], score_data["beats"][2])
    score = lane.BroadcastScore.model_validate(score_data)
    assert lane._validate_score(score, truth) == (
        "beats for each shot must form one contiguous block"
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


def _score_with_reopened_shot():
    score_data = _fixtures()["score"]
    score_data["shots"].insert(1, {
        "shot_id": "shot_03",
        "scene_id": "scene_01",
        "description": "The return shelf across the room.",
        "visual_prompt": "A return shelf opposite the radio desk.",
    })
    score_data["beats"][1]["shot_id"] = "shot_03"
    return score_data


def test_score_beat_topology_repair_splits_return_without_reordering_beats():
    truth = lane.AudibleTruthMap.model_validate(_fixtures()["truth"])
    score_data = _score_with_reopened_shot()
    score_data["shots"] = [
        score_data["shots"][2], score_data["shots"][0],
        score_data["shots"][1],
    ]
    score = lane.BroadcastScore.model_validate(score_data)
    assert lane._validate_score(score, truth) == (
        "beats for each shot must form one contiguous block"
    )
    authored_beats = [beat.model_dump(mode="json") for beat in score.beats]
    authored_shots = [shot.model_dump(mode="json") for shot in score.shots]
    authored_shot = next(
        shot for shot in score.shots if shot.shot_id == "shot_01"
    )

    repaired = lane._repair_score_beat_topology(score, truth)

    assert repaired is not None
    assert [beat.beat_id for beat in repaired.beats] == [
        beat["beat_id"] for beat in authored_beats
    ]
    assert [beat.shot_id for beat in repaired.beats] == [
        "shot_01", "shot_03", "shot_01_return_2", "shot_02", "shot_02",
    ]
    for index, beat in enumerate(repaired.beats):
        expected = dict(authored_beats[index])
        if index == 2:
            expected["shot_id"] = "shot_01_return_2"
        assert beat.model_dump(mode="json") == expected
    cloned_shot = next(
        shot for shot in repaired.shots
        if shot.shot_id == "shot_01_return_2"
    )
    assert cloned_shot.model_copy(
        update={"shot_id": "shot_01"},
    ).model_dump(mode="json") == authored_shot.model_dump(mode="json")
    assert [
        shot.model_dump(mode="json") for shot in repaired.shots[:-1]
    ] == authored_shots
    assert repaired.shots[-1].shot_id == "shot_01_return_2"
    assert lane._validate_score(repaired, truth) is None
    assert [beat.beat_id for beat in score.beats] == [
        beat["beat_id"] for beat in authored_beats
    ]
    assert [beat.shot_id for beat in score.beats] == [
        "shot_01", "shot_03", "shot_01", "shot_02", "shot_02",
    ]


def test_score_beat_topology_repair_uses_collision_safe_return_id():
    truth = lane.AudibleTruthMap.model_validate(_fixtures()["truth"])
    score_data = _score_with_reopened_shot()
    score_data["shots"].append({
        "shot_id": "shot_01_return_2",
        "scene_id": "scene_02",
        "description": "An unused authored setup with a colliding name.",
        "visual_prompt": "An unused quiet shelf setup.",
    })
    score = lane.BroadcastScore.model_validate(score_data)

    repaired = lane._repair_score_beat_topology(score, truth)

    assert repaired is not None
    assert repaired.beats[2].shot_id == "shot_01_return_2_2"
    assert {shot.shot_id for shot in repaired.shots} >= {
        "shot_01_return_2", "shot_01_return_2_2",
    }


def test_score_beat_topology_repair_fails_closed_on_hidden_graph_error():
    truth = lane.AudibleTruthMap.model_validate(_fixtures()["truth"])
    score_data = _score_with_reopened_shot()
    score_data["beats"][2]["line_intent"]["clue_ids"] = ["unknown"]
    score = lane.BroadcastScore.model_validate(score_data)

    assert lane._repair_score_beat_topology(score, truth) is None


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
    responses[3]["opening_music"]["music_file"] = "opening_music.mp3"
    responses[3]["closing_music"]["music_file"] = "closing_music.mp3"
    responses[3]["scenes"][0]["shots"] = [{
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
    assert len(calls) == 5


def test_p5_schema_boundary_projects_topology_when_raw_projection_misses(
    tmp_path, monkeypatch,
):
    responses = _responses()
    responses[3]["shots"].insert(1, {
        "shot_id": "shot_03",
        "scene_id": "scene_01",
        "description": "The return shelf across the room.",
        "visual_prompt": "A return shelf opposite the radio desk.",
    })
    responses[3]["beats"][1]["shot_id"] = "shot_03"
    monkeypatch.setattr(
        lane, "_repair_score_collection_placement",
        lambda _raw, _truth: None,
    )
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
        episode_id="codex56_p5_topology", out_dir=str(tmp_path),
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
        episode_root=tmp_path, episode_id="codex56_p5_topology",
    )

    assert len(calls) == 5
    assert [row["beat_id"] for row in led.data["beats"]] == [
        "b1", "b2", "b3", "b4", "b5",
    ]
    assert [row["shot_id"] for row in led.data["beats"]] == [
        "shot_01", "shot_03", "shot_01_return_2", "shot_02", "shot_02",
    ]
    assert [row["boundary"] for row in led.data["lines"]] == [
        "shot_start", "shot_start", "shot_start", "shot_start", "beat_start",
    ]


def test_p5_typed_repair_response_gets_schema_boundary_topology_projection(
    tmp_path, monkeypatch,
):
    responses = _responses()
    responses[3]["shots"].insert(1, {
        "shot_id": "shot_03",
        "scene_id": "scene_01",
        "description": "The return shelf across the room.",
        "visual_prompt": "A return shelf opposite the radio desk.",
    })
    responses[3]["beats"][1]["shot_id"] = "shot_03"
    safe_repair_response = json.loads(json.dumps(responses[3]))
    responses[3]["premise"] = "Kill the clue."
    responses.insert(4, safe_repair_response)
    monkeypatch.setattr(
        lane, "_repair_score_collection_placement",
        lambda _raw, _truth: None,
    )
    queued = iter(responses)
    calls = []
    repair_prompts = []

    def generate(messages, **_kwargs):
        calls.append(1)
        if any("forbidden term 'kill'" in row["content"]
               for row in messages):
            repair_prompts.append(messages)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p5_repair_topology", out_dir=str(tmp_path),
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
        episode_root=tmp_path, episode_id="codex56_p5_repair_topology",
    )

    assert len(calls) == 6
    assert len(repair_prompts) == 1
    assert "forbidden term 'kill'" in repair_prompts[0][1]["content"]
    assert [row["beat_id"] for row in led.data["beats"]] == [
        "b1", "b2", "b3", "b4", "b5",
    ]
    assert [row["shot_id"] for row in led.data["beats"]] == [
        "shot_01", "shot_03", "shot_01_return_2", "shot_02", "shot_02",
    ]


def test_p5_schema_boundary_projects_duplicate_clue_ownership(
    tmp_path, monkeypatch,
):
    responses = _responses()
    responses[3]["beats"][2]["line_intent"]["clue_ids"].insert(0, "q1")
    monkeypatch.setattr(
        lane, "_repair_score_collection_placement",
        lambda _raw, _truth: None,
    )
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
    assert len(calls) == 5


def test_p5_typed_repair_response_gets_schema_boundary_clue_projection(
    tmp_path, monkeypatch,
):
    responses = _responses()
    responses[3]["beats"][2]["line_intent"]["clue_ids"].insert(0, "q1")
    safe_repair_response = json.loads(json.dumps(responses[3]))
    responses[3]["premise"] = "Kill the clue."
    responses.insert(4, safe_repair_response)
    monkeypatch.setattr(
        lane, "_repair_score_collection_placement",
        lambda _raw, _truth: None,
    )
    queued = iter(responses)
    calls = []
    repair_prompts = []

    def generate(messages, **_kwargs):
        calls.append(1)
        if any("forbidden term 'kill'" in row["content"]
               for row in messages):
            repair_prompts.append(messages)
        return json.dumps(next(queued))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(
        episode_id="codex56_p5_repair_duplicate", out_dir=str(tmp_path),
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
        episode_root=tmp_path, episode_id="codex56_p5_repair_duplicate",
    )

    assert len(calls) == 6
    assert len(repair_prompts) == 1
    accepted_score = (
        led.data["meta"]["original_codex56sol"]
        ["accepted_artifacts"]["broadcast_score"]
    )
    assert [beat["line_intent"]["clue_ids"]
            for beat in accepted_score["beats"]] == [
        [], ["q1"], ["q2", "q3"], [], [],
    ]


def test_p5_schema_boundary_composes_topology_and_clue_projections(
    tmp_path, monkeypatch,
):
    responses = _responses()
    responses[3]["shots"].insert(1, {
        "shot_id": "shot_03", "scene_id": "scene_01",
        "description": "The return shelf across the room.",
        "visual_prompt": "A return shelf opposite the radio desk.",
    })
    responses[3]["beats"][1]["shot_id"] = "shot_03"
    responses[3]["beats"][2]["line_intent"]["clue_ids"].insert(0, "q1")
    safe_repair_response = json.loads(json.dumps(responses[3]))
    responses[3]["premise"] = "Kill the clue."
    responses.insert(4, safe_repair_response)
    monkeypatch.setattr(
        lane, "_repair_score_collection_placement",
        lambda _raw, _truth: None,
    )
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
        episode_id="codex56_p5_composed_projections", out_dir=str(tmp_path),
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
        episode_root=tmp_path, episode_id="codex56_p5_composed_projections",
    )

    accepted_score = (
        led.data["meta"]["original_codex56sol"]
        ["accepted_artifacts"]["broadcast_score"]
    )
    assert len(calls) == 6
    assert [beat["shot_id"] for beat in accepted_score["beats"]] == [
        "shot_01", "shot_03", "shot_01_return_2", "shot_02", "shot_02",
    ]
    assert [beat["line_intent"]["clue_ids"]
            for beat in accepted_score["beats"]] == [
        [], ["q1"], ["q2", "q3"], [], [],
    ]


def test_python_manifest_is_repeatable_and_closed():
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    first = lane._compile_manifest(score)
    second = lane._compile_manifest(score)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [row.boundary for row in first.lines] == [
        "shot_start", "beat_start", "beat_start", "shot_start", "beat_start",
    ]
    assert first.lines[1].clue_ids == ["q1"]
    # The clue lines the fair-play contract is defined over: non-announcer,
    # clue-carrying, and before the reveal.
    assert [row.line_id for row in lane._pre_reveal_clue_lines(first)] == [
        "line_002", "line_003",
    ]


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


def test_p1_clue_plan_must_cover_every_lost_object():
    """PBUG-20260713-11: a clue-short slate is a model defect, not a Python fix.

    The lost-object count comes from the accepted draw, so Python can prove the
    shortfall -- but a clue is authored story, so the defect returns to the
    model with the exact rule instead of being invented in Python.
    """
    draw = lane.ConstraintDraw.model_validate(DRAW)
    data = _fixtures()["slate"]
    assert lane._validate_slate(
        lane.PossibilitySlate.model_validate(data), draw,
    ) is None

    # A wider draw proves the coverage rule outranks the bare schema minimum:
    # four lost objects still leave a three-clue card one object short.
    wide_draw = lane.ConstraintDraw.model_validate({
        **DRAW, "lost_objects": ["stamp", "mitten", "card", "ledger"],
    })
    wide = json.loads(json.dumps(data))
    for card in wide["possibilities"]:
        card["lost_objects"] = list(wide_draw.lost_objects)
    error = lane._validate_slate(
        lane.PossibilitySlate.model_validate(wide), wide_draw,
    )
    assert "one distinct audible clue for each of the 4 lost objects" in error
    assert "but it has 3" in error

    rules = lane._repair_rules("P1", "list should have at least 3 items")
    assert "one distinct audible clue for EVERY lost object" in rules
    assert "Never merge two objects into one clue" in rules
    assert "never delete one to repair another" in rules

    routing._REGISTRY = None
    seam = routing.resolve_story_pack(
        "original_codex56sol",
    ).prompt_stages["codex56_possibility_slate"]
    assert "one distinct audible clue for EACH lost object" in seam
    assert "Never merge two objects into a single clue" in seam


def test_p1_clue_short_slate_is_repaired_by_the_authoring_model(tmp_path):
    responses = _responses()
    short = json.loads(json.dumps(responses[0]))
    for card in short["possibilities"]:
        card["clue_plan"] = card["clue_plan"][:2]
    responses.insert(0, short)
    prompts = []

    def generate(messages, **_kwargs):
        prompts.append(json.loads(json.dumps(messages)))
        return json.dumps(responses.pop(0))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(episode_id="p1_short", out_dir=str(tmp_path))
    meta = led.data.setdefault("meta", {})
    meta.update({"source_bank": "original_codex56sol",
                 "source_meta": {"constraint_draw": DRAW}})
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led,
        meta=meta, creative_fn=generate, technical_fn=generate,
        slot_scheduler=Scheduler(), source_bank_row=None, story_rules=rules,
        episode_root=tmp_path, episode_id="p1_short",
    )
    assert len(prompts) == 6
    repair_prompt = json.dumps(prompts[1])
    assert "one distinct audible clue for EVERY lost object" in repair_prompt
    assert "at least 3 items" in repair_prompt
    assert len(led.data["lines"]) == 5


def test_device_anchor_must_be_planted_before_the_reveal(tmp_path):
    """Fair play is a contract, not an opinion.

    The device must be audible on a clue line BEFORE the reveal explains it.
    A script that speaks it only at the reveal is repaired by the bounded line
    patch that already exists -- the same rung that repairs any other missing
    immutable anchor.
    """
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    draw = lane.ConstraintDraw.model_validate(DRAW)
    grounding = lane._build_grounding_contract(draw, truth)
    manifest = lane._compile_manifest(score)

    # Strip the plant: "grille" now survives only on the reveal line.
    late = lane.PerformanceScript.model_validate(fixtures["script"])
    late.lines[2].text = "The mitten scrapes softly, and the card rustles."
    assert "before the reveal line" in lane._validate_script_grounding(
        grounding, manifest, late,
    )
    plan = lane._script_grounding_repair_plan(late, manifest, grounding)
    # Both pre-reveal clue lines already speak their own lost-object anchors, so
    # neither is "loaded" by the plan -- the plant takes the earliest, and keeps
    # the anchor that line already speaks.
    assert [(row["line_id"], row["required_anchors"]) for row in plan] == [
        ("line_002", ["grille", "stamp"]),
    ]

    # A score whose every clue lands at or after the reveal is not fair by
    # construction -- only the score author can fix that, so it goes to the P5
    # ladder rather than to a patch that cannot reorder beats.
    unfair = score.model_copy(deep=True)
    for beat in unfair.beats:
        beat.line_intent.clue_ids = []
    unfair.beats[3].line_intent.clue_ids = ["q1", "q2", "q3"]
    assert "BEFORE the reveal beat" in lane._validate_score_clue_ownership(
        unfair, grounding,
    )


def test_a_schema_path_written_into_clue_ids_is_dropped_as_noise():
    """Live prompt ac762dd9: the model wrote
    "grounding_contract.device_anchor" into clue_ids. A string that names no clue
    cannot be a clue -- dropping it invents nothing. But if dropping it would
    leave a real clue uncovered, it was not noise, and the model must answer.
    """
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])

    rules = story_rules.resolve_story_rules("original_codex56sol")

    noisy = lane.BroadcastScore.model_validate(fixtures["score"])
    noisy.beats[1].line_intent.clue_ids = [
        "q1", "grounding_contract.device_anchor",
    ]
    # The whole attempt still validates: the noise is dropped at the boundary.
    assert lane._validate_score_attempt(noisy, truth, rules) is None
    assert noisy.beats[1].line_intent.clue_ids == ["q1"]

    # Noise that hides a dropped clue is not noise -- back to the ladder.
    lossy = lane.BroadcastScore.model_validate(fixtures["score"])
    lossy.beats[1].line_intent.clue_ids = ["grounding_contract.x"]
    assert "q1" in lane._validate_score_attempt(lossy, truth, rules)


def test_a_short_broadcast_cannot_hold_a_long_score():
    """Live prompt 717f3a4f: the seam had a beat FLOOR and no ceiling, so a
    30-word episode got a 19-beat score -- and the script that had to fill all
    nineteen lines blew its token budget and came back as truncated JSON.
    """
    score = lane.BroadcastScore.model_validate(_fixtures()["score"])
    # The floor is the show's own shape: announcer + one beat per caller +
    # reveal + closure. Below that it is a different show, not a shorter one.
    assert lane._beat_ceiling(30) == 8
    assert lane._beat_ceiling(120) == 30
    assert lane._beat_ceiling(420) == 40  # capped: a broadcast is not a novel
    assert lane._validate_score_scale(score, 30) is None

    long_score = score.model_copy(deep=True)
    extra = long_score.beats[1].model_copy(deep=True)
    for index in range(14):
        clone = extra.model_copy(deep=True)
        clone.beat_id = f"bx{index}"
        clone.line_intent.clue_ids = []
        long_score.beats.insert(2, clone)
    error = lane._validate_score_scale(long_score, 30)
    assert "at most 8 beats" in error
    assert "do not delete the clues" in error


def test_arc_phase_is_derived_from_the_landmarks_not_spelled_by_the_model():
    """Live prompt 55756bac: the model wrote `closure` for `closing` and the
    whole score was rejected by the schema. The landmark ids already decide it.
    """
    fixtures = _fixtures()
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    payload = json.loads(json.dumps(fixtures["score"]))
    payload["beats"][4]["line_intent"]["arc_phase"] = "closure"
    payload["beats"][2]["line_intent"]["arc_phase"] = "climax"

    repaired = lane._repair_score_collection_placement(
        json.dumps(payload), truth,
    )
    assert repaired is not None
    assert [b.line_intent.arc_phase for b in repaired.beats] == [
        "opening", "rising", "rising", "reveal", "closing",
    ]

    # And an enum typo must never reach the schema even when the score is
    # otherwise defective -- the ladder can only repair what it can parse.
    broken = json.loads(json.dumps(payload))
    broken["beats"][1]["line_intent"]["clue_ids"] = ["q1", "q1"]
    parsed = lane._project_score_arc_phases_only(json.dumps(broken))
    assert parsed is not None
    assert parsed.beats[4].line_intent.arc_phase == "closing"


def test_a_shortened_anchor_in_a_patch_is_restored_not_rejected():
    """Live prompts 6fe52216 / 522e1581: the patch put the device in exactly the
    right place and called it "the grille" instead of "ventilation grille".

    It decided WHERE the anchor belongs. The exact wording was never its
    decision, so restore it -- and when the model gives Python nothing to work
    with, say precisely which string is missing.
    """
    plan = [{
        "line_id": "line_002",
        "current_text": "The shelf number repeats after every answer.",
        "required_anchors": ["ventilation grille", "library stamp"],
    }]
    patch = lane.ScriptLinePatch.model_validate({"replacements": [{
        "line_id": "line_002",
        "text": "The stamp keeps answering back through the grille.",
    }]})
    assert lane._validate_script_line_patch(patch, plan) is None
    assert patch.replacements[0].text == (
        "The library stamp keeps answering back through the ventilation grille."
    )

    # Nothing to restore from -> a message that names the exact missing string.
    empty = lane.ScriptLinePatch.model_validate({"replacements": [{
        "line_id": "line_002", "text": "She listens to the room.",
    }]})
    error = lane._validate_script_line_patch(empty, plan)
    assert "'ventilation grille'" in error
    assert "She listens to the room." in error


def test_the_plant_never_lands_on_an_already_loaded_line():
    """Live prompt 6fe52216: the patch had to write three verbatim anchors into
    one intent, wrote two, and exhausted the ladder. One line, one new anchor.
    """
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    grounding = lane._build_grounding_contract(
        lane.ConstraintDraw.model_validate(DRAW), truth,
    )
    manifest = lane._compile_manifest(score)

    # line_002 speaks NEITHER its own object nor the device: it is the line the
    # lost-object plan must target. The device plant must therefore go elsewhere.
    stripped = lane.PerformanceScript.model_validate(fixtures["script"])
    stripped.lines[1].text = "The shelf number repeats after every answer."
    stripped.lines[2].text = "The mitten scrapes softly, and the card rustles."

    plan = lane._script_grounding_repair_plan(stripped, manifest, grounding)
    by_line = {row["line_id"]: row["required_anchors"] for row in plan}
    assert by_line["line_002"] == ["stamp"]
    assert by_line["line_003"] == ["card", "grille", "mitten"] or \
        by_line["line_003"] == ["card", "mitten"]
    # No line is asked to carry the device on top of a fresh lost-object anchor.
    assert "grille" not in by_line["line_002"]


def _deck_draws():
    path = (
        pathlib.Path(lane.__file__).parent
        / "story_packs" / "original_codex56sol" / "constraint_deck.json"
    )
    deck = json.loads(path.read_text(encoding="utf-8"))
    # The fetcher stamps deck provenance onto every draw it hands the lane.
    return [
        {**row, "deck_id": deck["deck_id"], "deck_sha256": "a" * 64}
        for row in deck["draws"]
    ]


@pytest.mark.parametrize("draw", _deck_draws(), ids=lambda d: d["constraint_id"])
def test_every_shipped_draw_is_satisfiable_at_the_five_line_minimum(draw):
    """The contract must be WINNABLE on every draw the deck actually ships.

    A 30-word episode compiles to the 5-beat minimum: announcer, two clue lines,
    reveal, closure. With the real multiword anchors, one witness script has to
    exist that speaks every lost-object anchor on a clue line for its thread, the
    device anchor on a pre-reveal clue line AND on the reveal, and the resolution
    anchor on the closure. If no witness exists for some draw, the rule is not a
    contract -- it is an unwinnable gate, and this test is what says so.
    """
    objects = draw["lost_objects"]
    device = draw["device_spoken_anchor"]
    resolution = draw["resolution_spoken_anchor"]
    clue_ids = [f"q{i}" for i in range(1, len(objects) + 1)]

    truth = lane.AudibleTruthMap.model_validate({
        "title": "Deck Witness", "premise": "Three calls, one echo.",
        "setting": "A small station", "desk_operator_name": "Mara Vale",
        "caller_threads": [
            {"thread_id": f"t{i}", "caller_name": f"Caller {i}",
             "lost_object": obj, "practical_need": "finish an errand"}
            for i, obj in enumerate(objects, 1)
        ],
        "causal_steps": [
            {"step_id": "s1", "cause": "The device is loose.",
             "effect": "Desk sounds travel."},
            {"step_id": "s2", "cause": "Objects rest beside it.",
             "effect": "Their sounds repeat."},
        ],
        "audible_clues": [
            {"clue_id": clue_ids[i], "thread_id": f"t{i + 1}",
             "sound_or_phrase": f"a sound from the {obj}",
             "implication": f"the {obj} is near the desk"}
            for i, obj in enumerate(objects)
        ],
        "interpretations": [
            {"interpretation_id": "i1", "clue_ids": [clue_ids[0]],
             "explanation": "The sounds come from the desk.", "is_true": True},
            {"interpretation_id": "i2", "clue_ids": [clue_ids[0]],
             "explanation": "A caller is imagining it.", "is_true": False},
        ],
        "reveal": "The device carries sounds from the shelf.",
        "resolution_links": [
            {"thread_id": f"t{i}", "action": "Mara checks the shelf.",
             "result": f"{obj} goes home."}
            for i, obj in enumerate(objects, 1)
        ],
    })
    grounding = lane._build_grounding_contract(
        lane.ConstraintDraw.model_validate(draw), truth,
    )

    # The witness: the first clue line carries object one AND plants the device;
    # the second carries the remaining objects.
    first_clue, rest = clue_ids[:1], clue_ids[1:]
    beats = [
        {"beat_id": "b1", "shot_id": "shot_01", "scene_id": "scene_01",
         "char_id": "announcer", "speaker": "Announcer",
         "line_intent": {"intent": "open the show", "arc_phase": "opening",
                         "clue_ids": []}},
        {"beat_id": "b2", "shot_id": "shot_01", "scene_id": "scene_01",
         "char_id": "c01", "speaker": "Mara Vale",
         "line_intent": {
             "intent": f"name the {objects[0]} heard through the {device}",
             "arc_phase": "rising", "clue_ids": first_clue}},
        {"beat_id": "b3", "shot_id": "shot_01", "scene_id": "scene_01",
         "char_id": "c02", "speaker": "Ivo Reed",
         "line_intent": {
             "intent": "name " + " and ".join(objects[1:]),
             "arc_phase": "rising", "clue_ids": rest}},
        {"beat_id": "b4", "shot_id": "shot_02", "scene_id": "scene_02",
         "char_id": "c01", "speaker": "Mara Vale",
         "line_intent": {"intent": f"reveal the {device}",
                         "arc_phase": "reveal", "clue_ids": []}},
        {"beat_id": "b5", "shot_id": "shot_02", "scene_id": "scene_02",
         "char_id": "c01", "speaker": "Mara Vale",
         "line_intent": {"intent": f"confirm the desk {resolution}",
                         "arc_phase": "closing", "clue_ids": []}},
    ]
    score = lane.BroadcastScore.model_validate({
        "title": "Deck Witness", "premise": "Three calls, one echo.",
        "setting": "A small station",
        "cast": [
            {"char_id": "announcer", "name": "Announcer", "role": "announcer",
             "character_description": "Brief station host"},
            {"char_id": "c01", "name": "Mara Vale", "role": "desk_operator",
             "character_description": "Warm precise desk operator"},
            {"char_id": "c02", "name": "Ivo Reed", "role": "caller",
             "character_description": "Patient volunteer"},
        ],
        "scenes": [
            {"scene_id": "scene_01", "description": "Calls reach the desk.",
             "env": "radio station desk"},
            {"scene_id": "scene_02", "description": "The echo is mapped.",
             "env": "lost-and-found shelf"},
        ],
        "shots": [
            {"shot_id": "shot_01", "scene_id": "scene_01",
             "description": "Mara at the desk.", "visual_prompt": "Warm desk"},
            {"shot_id": "shot_02", "scene_id": "scene_02",
             "description": "The shelf.", "visual_prompt": "Orderly shelf"},
        ],
        "beats": beats,
        "orientation_beat_id": "b1", "reveal_beat_id": "b4",
        "closure_beat_id": "b5",
        "opening_music": {"description": "Warm motif.",
                          "generation_prompt": "Warm strings, no vocals"},
        "closing_music": {"description": "It resolves.",
                          "generation_prompt": "Gentle strings, no vocals"},
    })
    assert lane._validate_score(score, truth) is None
    assert lane._validate_score_clue_ownership(score, grounding) is None

    manifest = lane._compile_manifest(score)
    texts = [
        "Lost and Found Frequency is listening.",
        f"The {objects[0]} keeps answering back through the {device}.",
        "You can hear " + " and ".join(objects[1:]) + " beside it.",
        f"The {device} has been carrying every sound from that shelf.",
        f"The desk {resolution} before the hour is out.",
    ]
    script = lane.PerformanceScript.model_validate({
        "title": "Deck Witness",
        "lines": [
            {"line_id": row.line_id, "char_id": row.char_id,
             "speaker": row.speaker, "text": text}
            for row, text in zip(manifest.lines, texts)
        ],
    })
    assert lane._validate_script_grounding(grounding, manifest, script) is None
    receipt = lane._grounding_receipt(grounding, manifest, script)
    assert receipt["complete"] is True
    plant = next(r for r in receipt["evidence"] if r["kind"] == "device_plant")
    assert plant["line_id"] == "line_002"


def test_a_shortened_thread_lost_object_is_restored_when_forced():
    """Live prompt 37a3cedf: the model wrote `timetable` for `folded timetable`.

    The thread's real content -- the caller, the need -- is authored. The object
    name is an echo of the draw, so restore it when exactly one draw object can
    be meant. Ambiguity is the model's to resolve, and goes back to the ladder.
    """
    card = lane.PossibilityCard.model_validate({
        **_fixtures()["card"],
        "lost_objects": ["garden key", "folded timetable", "tin whistle"],
    })
    payload = json.loads(json.dumps(_fixtures()["truth"]))
    for row, obj in zip(payload["caller_threads"],
                        ["garden key", "timetable", "whistle"]):
        row["lost_object"] = obj
    truth = lane.AudibleTruthMap.model_validate(payload)

    assert lane._validate_truth_map(truth, card) is None
    assert [row.lost_object for row in truth.caller_threads] == [
        "garden key", "folded timetable", "tin whistle",
    ]
    # The authored content of the thread is untouched.
    assert truth.caller_threads[1].caller_name == "Nell Park"

    # A DROPPED thread is not a coordinate error -- Python may not invent one.
    short = lane.AudibleTruthMap.model_validate(payload)
    short.caller_threads = short.caller_threads[:2]
    assert "VERBATIM" in lane._validate_truth_map(short, card)


def test_a_paraphrased_immutable_draw_field_is_restored_not_fatal():
    """Live prompt efafc6fa: the model wrote the right story about the right
    device and re-worded the field that only echoes the draw. Restore the input.
    """
    draw = lane.ConstraintDraw.model_validate(DRAW)
    slate = lane.PossibilitySlate.model_validate(_fixtures()["slate"])
    slate.possibilities[0].acoustic_device = "a grille that repeats phrases"
    slate.possibilities[1].lost_objects = ["stamp", "mitten", "recipe card"]

    assert lane._validate_slate(slate, draw) is None
    for card in slate.possibilities:
        assert card.acoustic_device == DRAW["acoustic_device"]
        assert card.lost_objects == DRAW["lost_objects"]


def test_announcer_char_id_is_canonicalized_not_rejected(tmp_path):
    """An id is a coordinate, not a story decision.

    Live prompt a89a46a4: the model wrote a valid announcer row and filed it
    under char_id 'a'. Rejecting the score sent a 5,772-token repair prompt into
    a 4,592-token window, PROMPT_GUARD cut the tail off, and the model came back
    with a single cast row. Rename the id at the attempt boundary instead.
    """
    responses = _responses()
    score = responses[3]
    score["cast"][0]["char_id"] = "a"
    for beat in score["beats"]:
        if beat["char_id"] == "announcer":
            beat["char_id"] = "a"
    calls = []

    def generate(messages, **_kwargs):
        calls.append(messages)
        return json.dumps(responses.pop(0))

    routing._REGISTRY = None
    story_rules._clear_caches()
    pack = routing.resolve_story_pack("original_codex56sol")
    rules = story_rules.resolve_story_rules("original_codex56sol")
    led = ledger_mod.new_ledger(episode_id="codex56_announcer", out_dir=str(tmp_path))
    meta = led.data.setdefault("meta", {})
    meta.update({
        "source_bank": "original_codex56sol",
        "source_meta": {"constraint_draw": DRAW},
    })
    lane.run_original_codex56sol_episode(
        payload={"seed_text": json.dumps(DRAW)}, pack=pack,
        resolved={"target_words": 30, "num_characters": 3}, led=led, meta=meta,
        creative_fn=generate, technical_fn=generate, slot_scheduler=Scheduler(),
        source_bank_row=None, story_rules=rules, episode_root=tmp_path,
        episode_id="codex56_announcer",
    )

    # No repair rung was spent: the id was projected at the attempt boundary.
    assert len(calls) == 5
    accepted = led.data["meta"]["original_codex56sol"]["accepted_artifacts"]
    assert accepted["broadcast_score"]["cast"][0]["char_id"] == "announcer"
    assert led.data["cast"][0]["char_id"] == "announcer"


def test_p5_repair_context_is_bounded_to_what_the_model_cannot_infer():
    """The repair prompt must FIT, or PROMPT_GUARD deletes the contract silently."""
    inputs = {
        "truth_map": _fixtures()["truth"],
        "grounding_contract": {
            "lost_object_anchors": ["stamp"], "device_anchor": "grille",
            "resolution_anchor": "every item is returned",
            "expected_cause": "a grille repeats phrases",
            "expected_resolution": "return every item",
            "clues": [{
                "clue_id": "q1", "thread_id": "t1", "lost_object": "stamp",
                "sound_or_phrase": "a repeated shelf number",
                "implication": "the stamp is near the desk",
            }],
        },
        "target_words_advisory": 30,
    }
    bounded = lane._repair_inputs("P5", inputs)
    # The immutable anchors and the clue inventory survive.
    assert bounded["grounding_contract"]["device_anchor"] == "grille"
    assert bounded["truth_map"]["audible_clues"] == [
        {"clue_id": "q1", "thread_id": "t1"},
        {"clue_id": "q2", "thread_id": "t2"},
        {"clue_id": "q3", "thread_id": "t3"},
    ]
    assert bounded["target_words_advisory"] == 30
    # The prose the failed artifact already carries does not.
    assert "caller_threads" not in bounded["truth_map"]
    assert "sound_or_phrase" not in bounded["grounding_contract"]["clues"][0]
    assert len(json.dumps(bounded)) < len(json.dumps(inputs)) / 2
    # A pass with a small input surface keeps all of it.
    assert lane._repair_inputs("P1", {"ingress": {"draw": 1}}) == {
        "ingress": {"draw": 1},
    }


def test_p5_announcer_owned_clue_reaches_the_p5_ladder():
    """The one half of the grounding contract the bounded patch cannot repair.

    A missing anchor WORD is a leaf defect the intent patch fixes. But the patch
    is forbidden from touching clue_ids, so an announcer-owned clue must be
    returned to the pass that authored it -- not hard-aborted with zero calls.
    """
    fixtures = _fixtures()
    score = lane.BroadcastScore.model_validate(fixtures["score"])
    truth = lane.AudibleTruthMap.model_validate(fixtures["truth"])
    draw = lane.ConstraintDraw.model_validate(DRAW)
    grounding = lane._build_grounding_contract(draw, truth)

    assert lane._validate_score_clue_ownership(score, grounding) is None

    # Move every clue for the stamp onto the announcer beat.
    hijacked = score.model_copy(deep=True)
    for beat in hijacked.beats:
        if "q1" in beat.line_intent.clue_ids:
            beat.line_intent.clue_ids = [
                c for c in beat.line_intent.clue_ids if c != "q1"
            ]
    hijacked.beats[0].line_intent.clue_ids = ["q1"]
    error = lane._validate_score_clue_ownership(hijacked, grounding)
    assert "non-announcer beat must carry an audible clue" in error
    assert "stamp" in error

    # The ownership rule belongs in the ladder; the naming rule does not.
    named_only = score.model_copy(deep=True)
    for beat in named_only.beats:
        if "q1" in beat.line_intent.clue_ids:
            beat.line_intent.intent = "mention the shelf number"
    assert lane._validate_score_clue_ownership(named_only, grounding) is None


def test_p2_triage_contract_is_stated_in_seam_and_repair_rules():
    routing._REGISTRY = None
    seam = routing.resolve_story_pack(
        "original_codex56sol",
    ).prompt_stages["codex56_slate_triage"]
    rules = lane._repair_rules("P2", "selected_possibility_id must match")
    for text in (seam, rules):
        assert "copied verbatim" in text
        assert "least compromised" in text
        assert "advisory" in text.lower()


def test_every_schema_bound_is_in_the_shared_model_visible_contract():
    """Every structured pass gets its bounds through the shared ladder contract."""

    passes = [
        ("P1", "codex56_possibility_slate", lane.PossibilitySlate),
        ("P2", "codex56_slate_triage", lane.SlateTriage),
        ("P3", "codex56_audible_truth_map", lane.AudibleTruthMap),
        ("P5", "codex56_broadcast_score", lane.BroadcastScore),
        ("P6", "codex56_performance_script", lane.PerformanceScript),
        ("P6_grounding_patch", "codex56_script_anchor_patch",
         lane.ScriptLinePatch),
    ]

    def bounded(model, prefix="", seen=None, out=None):
        seen = seen if seen is not None else set()
        out = out if out is not None else []
        if model in seen:
            return out
        seen.add(model)
        for name, field in model.model_fields.items():
            if any(hasattr(meta, attr) for meta in field.metadata
                   for attr in ("min_length", "max_length", "pattern")):
                out.append(f"{prefix}{name}")
            annotation = field.annotation
            for sub in (getattr(annotation, "__args__", ()) or (annotation,)):
                if isinstance(sub, type) and issubclass(sub, BaseModel):
                    bounded(sub, f"{prefix}{name}.", seen, out)
        return out

    unstated = []
    for pass_id, _seam_key, schema in passes:
        visible = lane.schema_shape_instruction(schema)
        assert "[OTR_SCHEMA_CONTRACT_V1]" in visible
        for path in bounded(schema):
            leaf = path.split(".")[-1]
            if leaf not in visible:
                unstated.append(f"{pass_id}:{schema.__name__}.{path}")
    assert not unstated, (
        "these fields carry a schema bound the model is never shown, and the "
        f"repair ladder cannot state: {unstated}"
    )


def test_wide_draw_contracts_are_satisfiable():
    """A schema cap must never forbid the only artifact its validator accepts."""
    def max_length(model, field):
        return next(
            getattr(meta, "max_length")
            for meta in model.model_fields[field].metadata
            if getattr(meta, "max_length", None) is not None
        )

    # One caller per lost object, and a draw may carry up to 6 lost objects.
    assert max_length(lane.PossibilityCard, "callers") >= 6
    # The intent patch can target one beat per anchor (<=6) plus reveal and
    # closure = 8, and `_validate_score_intent_patch` demands EVERY planned
    # target -- so a cap below 8 forbids the only patch it would accept.
    assert max_length(lane.ScriptLinePatch, "replacements") >= 8


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
    assert "beats array is chronological and MUST never be reordered" in rules
    assert "new shot row with a new unique shot_id" in rules
    assert "schema-path pseudo-fields" in rules
    assert "singular clue_id is forbidden" in rules
    assert "scene MUST retain a non-empty env" in rules
    assert "shot MUST retain a non-empty visual_prompt" in rules
    assert "orientation_beat_id beat is `opening`" in rules
    assert "every other beat is `rising`" in rules
