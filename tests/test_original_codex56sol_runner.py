import json
from contextlib import contextmanager

from nodes import _otr_original_codex56sol as lane
from nodes import _otr_story_routing as routing
from nodes import _otr_story_rules as story_rules
from nodes import production_ledger as ledger_mod


DRAW = {"deck_id":"deck","deck_sha256":"a"*64,"constraint_id":"c01","lost_objects":["stamp","mitten","card"],"acoustic_device":"a grille repeats phrases","helpful_ending":"return every item"}


def _responses():
    card = {"possibility_id":"p1","title_seed":"Echo Desk","premise":"A stamp, mitten, and card are traced through one helpful echo.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":DRAW["lost_objects"],"acoustic_device":DRAW["acoustic_device"],"shared_cause":"A grille repeats phrases and carries desk sounds.","clue_plan":["the stamp's shelf number repeats","the mitten makes a wool scrape","the card rustles beside the grille"],"helpful_resolution":"Mara maps the echo and returns every item."}
    truth = {"title":"The Helpful Echo","premise":"A station desk solves three lost-item calls through one echo.","setting":"A small community radio station","desk_operator_name":"Mara Vale","caller_threads":[{"thread_id":"t1","caller_name":"Ivo Reed","lost_object":"stamp","practical_need":"finish a library return"},{"thread_id":"t2","caller_name":"Nell Park","lost_object":"mitten","practical_need":"complete a pair"}],"causal_steps":[{"step_id":"s1","cause":"The grille is loose.","effect":"Desk sounds travel to the call booth."},{"step_id":"s2","cause":"Objects rest beside the grille.","effect":"Their sounds repeat on calls."}],"audible_clues":[{"clue_id":"q1","thread_id":"t1","sound_or_phrase":"a repeated shelf number","implication":"the stamp is near the desk"},{"clue_id":"q2","thread_id":"t2","sound_or_phrase":"a soft wool scrape","implication":"the mitten is beside the grille"},{"clue_id":"q3","thread_id":"t2","sound_or_phrase":"three matching notes","implication":"one channel carries every clue"}],"reveal":"The loose grille carries sounds from the lost-and-found shelf.","resolution_links":[{"thread_id":"t1","action":"Mara checks the numbered shelf.","result":"Ivo receives the stamp."},{"thread_id":"t2","action":"Mara checks beside the grille.","result":"Nell receives the mitten."}]}
    cast = [{"char_id":"announcer","name":"Announcer","role":"announcer","character_description":"Brief station host"},{"char_id":"c01","name":"Mara Vale","role":"desk_operator","character_description":"Warm precise desk operator"},{"char_id":"c02","name":"Ivo Reed","role":"caller","character_description":"Patient library volunteer"}]
    scenes = [{"scene_id":"scene_01","description":"Calls reach the desk.","env":"radio station desk"},{"scene_id":"scene_02","description":"Mara resolves the echo.","env":"lost-and-found shelf"}]
    shots = [{"shot_id":"shot_01","scene_id":"scene_01","description":"Mara at the desk.","visual_prompt":"Warm radio desk, Mara listening, amber practical light"},{"shot_id":"shot_02","scene_id":"scene_02","description":"The shelf and grille.","visual_prompt":"Orderly shelf beside a loose grille, soft morning light"}]
    beats = [
      {"beat_id":"b1","shot_id":"shot_01","scene_id":"scene_01","char_id":"announcer","speaker":"Announcer","intent":"identify the station"},
      {"beat_id":"b2","shot_id":"shot_01","scene_id":"scene_01","char_id":"c01","speaker":"Mara Vale","intent":"orient the practical problem"},
      {"beat_id":"b3","shot_id":"shot_01","scene_id":"scene_01","char_id":"c02","speaker":"Ivo Reed","intent":"state the shelf-number clue"},
      {"beat_id":"b4","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","intent":"reveal the grille cause"},
      {"beat_id":"b5","shot_id":"shot_02","scene_id":"scene_02","char_id":"c01","speaker":"Mara Vale","intent":"return the item and close helpfully"}]
    score = {"title":truth["title"],"premise":truth["premise"],"setting":truth["setting"],"cast":cast,"scenes":scenes,"shots":shots,"beats":beats,"opening_music":{"description":"A curious warm station motif.","generation_prompt":"Warm plucked strings and soft dial tones, no vocals"},"closing_music":{"description":"The motif resolves gently.","generation_prompt":"Gentle resolved plucked strings, no vocals"}}
    manifest_lines = [{"line_id":f"l{i}","beat_id":b["beat_id"],"shot_id":b["shot_id"],"scene_id":b["scene_id"],"char_id":b["char_id"],"speaker":b["speaker"],"speaker_role":"announcer" if b["char_id"]=="announcer" else "character","boundary":"shot_start" if i in (1,4) else "beat_start","arc_phase":("opening" if i<3 else "reveal" if i==4 else "closing"),"intent":b["intent"]} for i,b in enumerate(beats,1)]
    script_lines = [{"line_id":f"l{i}","char_id":b["char_id"],"speaker":b["speaker"],"text":t} for i,(b,t) in enumerate(zip(beats,["Lost and Found Frequency is listening.","One echo is joining today's calls.","I hear my shelf number after every answer.","The loose grille carries sounds from this shelf.","Your stamp is here, and the grille is secure." ]),1)]
    return [
      {"draw":DRAW},
      {"possibilities":[card,{**card,"possibility_id":"p2","title_seed":"Whisper Shelf"},{**card,"possibility_id":"p3","title_seed":"Three Notes Home"}]},
      {"selected_possibility_id":"p1","findings":[]}, truth,
      {"accepted":True,"findings":[]}, score,
      {"lines":manifest_lines,"orientation_line_id":"l2","reveal_line_id":"l4","closure_line_id":"l5"},
      {"title":"The Helpful Echo","lines":script_lines},
      {"understood_cause":"The grille carried shelf sounds.","understood_resolution":"Mara returned the objects.","findings":[],"optional_notes":[]},
      {"accepted":True,"findings":[],"warnings":[]},
    ]


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
    assert len(calls) == 10
    assert parts.run_story_spine is False
    assert len(led.data["cast"]) == 3
    assert len(led.data["lines"]) == 5
    assert [(m["cue_id"],m["placement"]) for m in led.data["music"]] == [("opening","opening"),("closing","closing")]
    assert led.data["meta"]["content_authorship"]["coverage"]["complete"] is True


def test_cross_artifact_validators_return_retryable_error_strings():
    draw = lane.ConstraintDraw.model_validate(DRAW)
    bad = lane.PossibilitySlate.model_validate({"possibilities": [
        {"possibility_id":f"p{i}","title_seed":"Other","premise":"Other objects.","desk_operator":{"name":"Mara Vale"},"callers":[{"name":"Ivo Reed"},{"name":"Nell Park"}],"lost_objects":["other","items","here"],"acoustic_device":"A storm.","shared_cause":"A storm.","clue_plan":["one","two","three"],"helpful_resolution":"They meet."}
        for i in range(1,4)
    ]})
    error = lane._validate_slate(bad, draw)
    assert isinstance(error, str) and "copied verbatim" in error


def test_ungrounded_fair_play_opinion_is_not_a_fatal_coordinate():
    report = lane.FairPlayReport.model_validate({
        "accepted": False,
        "findings": [{"category":"Helpful Ending","detail":"Could be warmer","blocking":True}],
    })
    grounded = [f for f in report.findings if f.blocking and f.field_path and f.item_id]
    assert grounded == []


def test_structural_numeric_ids_canonicalize_without_authored_prose_change():
    card = lane.PossibilityCard.model_validate({
        "possibility_id":1,"title_seed":"Echo","premise":"A premise.",
        "desk_operator":{"name":"Mara Vale"},"callers":[],
        "lost_objects":DRAW["lost_objects"],
        "acoustic_device":DRAW["acoustic_device"],"shared_cause":"Echo",
        "clue_plan":["one","two","three"],"helpful_resolution":"Returned.",
    })
    assert card.possibility_id == "1"


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
