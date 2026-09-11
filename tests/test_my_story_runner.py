"""The My Story runner: the pass graph, and the ledger it hands downstream.

Driven by canned slot functions rather than a model, so what is under test is
the LANE -- pass order and slot routing, whether the person's requirements
survive into the finished rows, and whether the ledger it produces satisfies
the contracts the freeze and the voice bus already enforce.

It uses the REAL production ledger against a tmp output root: a fake ledger
would happily accept rows the real one normalizes away, and the row shape is
most of what this runner is responsible for.

CPU only, no model, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_my_story as MS  # noqa: E402
from nodes import _otr_story_input as SI  # noqa: E402
from nodes import _otr_story_routing as RT  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OTR_EPISODE_SEED", "12345")
    yield


# ---------------------------------------------------------------------------
# canned artifacts
# ---------------------------------------------------------------------------

def _interpretation(planned=2, exclusive=False, named=("Ada", "Tom")):
    return {
        "requirements": [
            {"id": "r1", "text": "Ada is the keeper", "kind": "cast",
             "source_field": "characters", "strength": "required"},
        ],
        "named_cast": [
            {"name": n, "notes": "", "stated_gender": "female" if n == "Ada" else "",
             "speaking": True, "required": True} for n in named
        ],
        "cast_plan": {"requested": 2, "planned": planned,
                      "exclusive": exclusive, "reason": "as asked"},
        "setting_brief": "a rock light",
        "assumptions": ["the year is unstated; the story reads as 1890s"],
        "conflicts": [],
    }


def _treatment(acts=1, cast=("Ada", "Tom")):
    return {
        "title": "The Fog Bell",
        "logline": "A keeper hears a voice in the fog.",
        "dramatic_question": "Is anyone out there?",
        "setting": "a rock lighthouse",
        "time_of_day": "midnight",
        "cast": [
            {"name": n, "role": "lead", "character_description": "%s, at the light" % n,
             "gender": "female" if n == "Ada" else "male",
             "age_band": "30s", "register": "plain", "timbre": "warm"}
            for n in cast
        ],
        "acts": [
            {"n": i + 1, "purpose": "turn %d" % (i + 1),
             "scene_setting": "the lamp room", "turns": ["a", "b"],
             "ending_state": "unresolved"} for i in range(acts)
        ],
        "ending": "the bell answers",
    }


def _act(n=1, speakers=("Ada", "Tom")):
    lines = []
    for i, s in enumerate(speakers):
        lines.append({"speaker": s, "text": "Line %d from %s." % (i + 1, s)})
    lines.append({"speaker": speakers[0], "text": "And one more."})
    return {"n": n, "scene_setting": "the lamp room", "lines": lines}


def _frame(attribution, inter=0):
    return {
        "announcer_intro": ["Good evening.", attribution],
        "announcer_outro": ["That was our story."],
        "coda": "Until next time.",
        "music_open": "low strings over a foghorn",
        "music_close": "a single bell, fading",
        "music_inter": ["a held note" for _ in range(inter)],
    }


class Slots:
    """Canned slot functions that record which slot each pass used."""

    def __init__(self, author="A. Listener", acts=1, cast=("Ada", "Tom"),
                 inter=0, interpretation=None):
        self.calls = []
        self._acts = acts
        self._cast = cast
        self._inter = inter
        self._attr = SI.attribution_sentence(author)
        self._interp = interpretation or _interpretation()
        self._act_seen = 0

    def _answer(self, messages):
        system = messages[0]["content"] if messages else ""
        if "work out what they actually want" in system:
            return json.dumps(self._interp)
        if "radio dramatist" in system:
            return json.dumps(_treatment(self._acts, self._cast))
        if "one act of a radio drama" in system:
            self._act_seen += 1
            return json.dumps(_act(self._act_seen, self._cast))
        if "announcer's frame" in system:
            return json.dumps(_frame(self._attr, self._inter))
        raise AssertionError("unexpected system prompt: %r" % system[:80])

    def creative(self, messages, *, temperature=0.0, max_new_tokens=None, **kw):
        self.calls.append(("creative", messages[0]["content"][:40]))
        return self._answer(messages)

    def technical(self, messages, *, temperature=0.0, max_new_tokens=None, **kw):
        self.calls.append(("technical", messages[0]["content"][:40]))
        return self._answer(messages)


def _run(slots, *, author="A. Listener", act_count=1, num_characters=2,
         include_act_breaks=True, idea="a keeper hears a voice", raw_num_characters=None):
    from nodes import production_ledger as PL

    raw_requested = num_characters if raw_num_characters is None else raw_num_characters
    bundle = SI.build_bundle(
        SI.capture_raw(idea=idea, characters="Ada, the keeper. Tom.",
                       author=author),
        SI.StoryRequest(num_characters=raw_requested, act_count=str(act_count),
                        include_act_breaks=include_act_breaks,
                        source_bank_requested="my_story",
                        visual_style_requested="viz_camera"),
    )
    resolved = {
        "act_count": act_count,
        "include_act_breaks": include_act_breaks,
        "num_characters": num_characters,
        "creative_writing_model": "test-creative",
        "technical_model": "test-technical",
        "lemmy_force": None,
        "source_meta": {
            "kind": "user_story",
            "story_input": bundle.as_dict(),
            "draft_digest": bundle.digest,
            "requested_num_characters": raw_requested,
            "story_author": bundle.normalized.author,
        },
    }
    led = PL.new_ledger(episode_id=None)
    parts = MS.run_my_story_episode(
        payload={}, pack=RT.resolve_story_pack("my_story"), resolved=resolved,
        led=led, meta=led.data.setdefault("meta", {}),
        creative_fn=slots.creative, technical_fn=slots.technical,
        slot_scheduler=None,
        source_bank_row=RT.require_runnable_bank("my_story"),
        episode_root=None, episode_id=led.episode_id,
    )
    return led, parts


# ---------------------------------------------------------------------------
# the pass graph
# ---------------------------------------------------------------------------

def test_the_passes_run_in_order_on_the_declared_slots():
    """Interpretation is TECHNICAL (extraction); the writing is CREATIVE."""
    slots = Slots()
    _run(slots)
    kinds = [slot for slot, _ in slots.calls]
    assert kinds[0] == "technical", "interpretation is an extraction pass"
    assert kinds[1:] == ["creative"] * (len(kinds) - 1)


def test_one_call_per_act():
    slots = Slots(acts=3, inter=2)
    _run(slots, act_count=3)
    act_calls = [c for c in slots.calls if "one act of a radio drama" in c[1]
                 or "act" in c[1].lower()]
    # interpretation + treatment + 3 acts + frame
    assert len(slots.calls) == 6, slots.calls


def test_no_source_is_fetched_and_no_spark_is_drawn():
    """The person's words are the whole source. Nothing else is consulted."""
    slots = Slots()
    led, _ = _run(slots)
    meta = led.data["meta"]
    assert meta.get("news") is None
    assert "spark_atoms" not in json.dumps(meta)
    assert meta["my_story"]["draft_digest"]


# ---------------------------------------------------------------------------
# what the person asked for survives
# ---------------------------------------------------------------------------

def test_the_named_cast_reaches_the_finished_cast_rows():
    slots = Slots()
    led, _ = _run(slots)
    names = {r["name"] for r in led.data["cast"]}
    assert {"Ada", "Tom"} <= names


def test_a_stated_gender_is_carried_and_never_guessed_from_a_name():
    slots = Slots()
    led, _ = _run(slots)
    ada = next(r for r in led.data["cast"] if r["name"] == "Ada")
    assert ada["gender"] == "female"


def test_selected_and_delivered_counts_match_while_plan_remains_evidence():
    slots = Slots(cast=("Ada", "Tom", "Mabel"),
                  interpretation=_interpretation(planned=2, exclusive=True))
    led, _ = _run(slots, num_characters=3)
    counts = led.data["meta"]["my_story"]["counts"]
    assert counts["planned_characters"] == 2
    assert counts["requested_characters"] == counts["actual_characters"] == 3
    assert led.data["meta"]["cast_contract"]["num_characters_locked"] == 3


def test_the_assumptions_the_model_made_are_kept_on_the_ledger():
    slots = Slots()
    led, _ = _run(slots)
    interp = led.data["meta"]["my_story"]["interpretation"]
    assert interp["assumptions"], "a material assumption must be recorded"


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------

def test_the_attribution_sentence_is_spoken_verbatim():
    slots = Slots(author="Jeffrey Brick")
    led, _ = _run(slots, author="Jeffrey Brick")
    spoken = " ".join(r["text"] for r in led.data["lines"]
                      if r.get("speaker_role") == "announcer")
    assert "Tonight's story is by Jeffrey Brick." in spoken


def test_an_unattributed_story_says_a_listener_and_names_nobody():
    slots = Slots(author="")
    led, _ = _run(slots, author="")
    spoken = " ".join(r["text"] for r in led.data["lines"]
                      if r.get("speaker_role") == "announcer")
    assert SI.ANONYMOUS_ATTRIBUTION in spoken
    receipt = led.data["meta"]["my_story"]["attribution"]
    assert receipt["author"] == ""


def test_missing_frame_credit_is_appended_once_without_retry():
    from nodes._otr_content_authorship import validate_receipt
    class MissingCredit(Slots):
        def _answer(self, messages):
            if "announcer's frame" in messages[0]["content"]:
                return json.dumps({"announcer_outro": ["Good night."]})
            return super()._answer(messages)
    slots = MissingCredit()
    led, _ = _run(slots)
    text = " ".join(row.get("text", "") for row in led.data["lines"])
    assert text.count(slots._attr) == 1
    assert len(slots.calls) == 4
    assert slots._attr not in str(led.data["meta"]["my_story"]["frame_proposal"])
    validate_receipt(led.data)


def test_every_line_carries_a_role_the_freeze_accepts():
    from nodes._otr_ledger_freeze import ALLOWED_SPEAKER_ROLES

    slots = Slots(acts=2, inter=1)
    led, _ = _run(slots, act_count=2)
    for row in led.data["lines"]:
        assert row["speaker_role"] in ALLOWED_SPEAKER_ROLES, row


@pytest.mark.parametrize("acts,breaks", [(1, True), (3, True), (3, False), (6, True)])
def test_the_music_cues_anchor_to_real_sentinel_rows(acts, breaks):
    from nodes import _otr_freeze_cascade as LFC
    from nodes._otr_writer_tail import _stamp_story_style_receipt

    slots = Slots(acts=acts, inter=acts - 1)
    led, parts = _run(slots, act_count=acts, include_act_breaks=breaks)
    # Normal metadata supplied by the writer tail at this component boundary.
    led.data["meta"]["episode_title"] = parts.final_title_override
    _stamp_story_style_receipt(led.data["meta"], contract=None, scaffold_enabled=False)
    line_ids = {r["line_id"] for r in led.data["lines"]}
    by_id = {r["line_id"]: r for r in led.data["lines"]}
    placements = set()
    for cue in led.data["music"]:
        assert cue["anchor_line_id"] in line_ids, cue
        assert by_id[cue["anchor_line_id"]]["beat_id"] is None
        placements.add(cue["placement"])
    expected = {"opening", "closing"}
    if breaks and acts > 1:
        expected.add("interstitial")
    assert expected == placements
    before_lines = json.loads(json.dumps(led.data["lines"]))
    before_cues = json.loads(json.dumps(led.data["music"]))
    def no_model(*args, **kwargs):
        pytest.fail("read-only freeze must not acquire a model")
    disposition = LFC.run_freeze_cascade(no_model, led)
    assert disposition.verdict == "frozen_clean", disposition.gap_audit_pre
    assert disposition.gap_audit_pre.errors == []
    assert disposition.gap_audit_pre.warnings == []
    assert led.data["lines"] == before_lines
    assert led.data["music"] == before_cues
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert saved["meta"]["freeze_verdict"] == "frozen_clean"


def test_no_interstitial_cue_when_act_breaks_are_off():
    slots = Slots(acts=2, inter=0)
    led, _ = _run(slots, act_count=2, include_act_breaks=False)
    placements = {c["placement"] for c in led.data["music"]}
    assert "interstitial" not in placements


def test_consecutive_lines_from_one_speaker_become_one_turn():
    """A beat is a continuous turn; splitting it would hand the voice bus two
    clips where the audience hears one person still talking."""
    slots = Slots()
    led, _ = _run(slots)
    for beat in led.data["beats"]:
        assert len(beat["line_ids"]) == 1
    character_rows = [r for r in led.data["lines"]
                      if r.get("speaker_role") == "character"]
    speakers = [r["speaker"] for r in character_rows]
    assert all(a != b for a, b in zip(speakers, speakers[1:])), speakers


def test_two_characters_never_share_a_voice():
    slots = Slots()
    led, _ = _run(slots)
    presets = [r["voice_preset"] for r in led.data["cast"]]
    assert len(presets) == len(set(presets)), presets


def test_the_authorship_receipt_validates_against_the_finished_rows():
    """This is what the read-only freeze re-verifies instead of rewriting."""
    from nodes._otr_content_authorship import validate_receipt

    slots = Slots()
    led, _ = _run(slots)
    validate_receipt(led.data)


def test_the_readonly_freeze_accepts_the_assembled_ledger():
    from nodes import _otr_freeze_cascade as LFC

    slots = Slots()
    led, _ = _run(slots)
    policy = LFC.resolve_freeze_policy(led.data["meta"])
    assert policy.name == "content_owned_readonly"
    assert LFC._readonly_structural_validation(led.data) == []


def test_the_word_counts_are_stamped_as_telemetry_only():
    slots = Slots()
    led, _ = _run(slots)
    budget = led.data["meta"]["word_budget"]
    assert budget["owner"] == "my_story"
    assert budget["policy"] == "actual_count_only"
    assert budget["target_status"] == "not_requested"


# ---------------------------------------------------------------------------
# the cameo this lane never rolls
# ---------------------------------------------------------------------------

def test_no_cameo_roll_happens_and_the_decision_is_recorded():
    from nodes._otr_casting import CONTENT_OWNED_NO_CAMEO_ROLL

    slots = Slots()
    led, _ = _run(slots)
    contract = led.data["meta"]["cast_contract"]
    assert contract["lemmy_hit"] is False
    assert contract["lemmy_policy"] == CONTENT_OWNED_NO_CAMEO_ROLL


def test_a_forced_cameo_knob_is_recorded_as_not_applicable_not_obeyed():
    """The cast belongs to the person who described it."""
    from nodes import production_ledger as PL

    slots = Slots(author="")
    bundle = SI.build_bundle(
        SI.capture_raw(idea="x", characters="Ada. Tom."),
        SI.StoryRequest(num_characters=2, act_count="1"))
    led = PL.new_ledger(episode_id=None)
    MS.run_my_story_episode(
        payload={}, pack=RT.resolve_story_pack("my_story"),
        resolved={"act_count": 1, "include_act_breaks": True,
                  "num_characters": 2, "creative_writing_model": "c",
                  "technical_model": "t", "lemmy_force": True,
                  "source_meta": {"story_input": bundle.as_dict(),
                                  "requested_num_characters": 2,
                                  "story_author": ""}},
        led=led, meta=led.data.setdefault("meta", {}),
        creative_fn=slots.creative, technical_fn=slots.technical,
        slot_scheduler=None,
        source_bank_row=RT.require_runnable_bank("my_story"),
        episode_root=None, episode_id=led.episode_id)
    assert led.data["meta"]["my_story"]["lemmy_knob_ignored"] is True
    assert not any(r["name"].upper() == "LEMMY" for r in led.data["cast"])


# ---------------------------------------------------------------------------
# the tail handoff
# ---------------------------------------------------------------------------

def test_the_tail_parts_carry_what_the_writer_tail_reads():
    slots = Slots()
    _, parts = _run(slots)
    assert parts.outline_view.title == "The Fog Bell"
    assert parts.outline_view.premise == "Is anyone out there?"
    assert parts.final_title_override == "The Fog Bell"
    assert parts.run_story_spine is False
    assert parts.canon.title == "The Fog Bell"


# ---------------------------------------------------------------------------
# cast coverage, before the freeze rather than after it
# ---------------------------------------------------------------------------

def test_the_last_act_must_give_every_unheard_character_a_line():
    """A cast of three where only two are heard: the act is otherwise legal,
    so the ONLY thing wrong with it is the character who never speaks."""
    validator = MS._make_act_validator(
        MS.StoryTreatment(**_treatment(1, ("Ada", "Tom", "Mabel"))),
        1, ("Tom",))
    silent = MS.ActScript(**{"n": 1, "scene_setting": "x", "lines": [
        {"speaker": "Ada", "text": "one"}, {"speaker": "Mabel", "text": "two"}]})
    problem = validator(silent)
    assert problem and "Tom" in problem and "last act" in problem


def test_a_speaker_outside_the_cast_is_refused():
    validator = MS._make_act_validator(
        MS.StoryTreatment(**_treatment(1, ("Ada", "Tom"))), 1, ())
    stranger = MS.ActScript(**{"n": 1, "scene_setting": "x", "lines": [
        {"speaker": "Ada", "text": "one"}, {"speaker": "Mabel", "text": "two"}]})
    problem = validator(stranger)
    assert problem and "Mabel" in problem


@pytest.mark.parametrize("planned", [2, 9])
def test_cast_plan_estimates_do_not_reject_a_usable_selected_cast(planned):
    slots = Slots(interpretation=_interpretation(planned=planned, named=tuple("ABCDEFGHI")))
    led, _ = _run(slots)
    story = led.data["meta"]["my_story"]
    assert story["counts"]["actual_characters"] == 2
    assert story["counts"]["planned_characters"] == planned
    assert story["fidelity_discrepancies"]


@pytest.mark.parametrize("failed_save,checkpoint", [(1, "preamble"), (2, "act 1")])
def test_assembly_stops_when_an_incremental_save_fails(monkeypatch, failed_save, checkpoint):
    import random
    from nodes import production_ledger as PL
    led = PL.new_ledger(episode_id=None)
    save = led.save
    calls = []
    def fail_checkpoint():
        calls.append(True)
        return None if len(calls) == failed_save else save()
    monkeypatch.setattr(led, "save", fail_checkpoint)
    treatment = MS.StoryTreatment(**_treatment())
    with pytest.raises(MS.MyStoryError, match=checkpoint):
        MS._assemble(led, treatment, [MS.ActScript(**_act())],
                     MS.StoryFrame(**_frame(SI.attribution_sentence("A. Listener"))),
                     MS._assign_voices(treatment, random.Random(123)),
                     owner_bank="my_story", include_act_breaks=True,
                     interpretation=MS.StoryInterpretation(**_interpretation()))
    assert len(calls) == failed_save


def test_actual_voice_allocation_exhaustion_names_the_character(monkeypatch):
    import random
    monkeypatch.setattr(MS._POOLS, "open_voice_pool", lambda taken: [])
    with pytest.raises(MS.MyStoryCastError, match="character 1.*Ada"):
        MS._assign_voices(MS.StoryTreatment(**_treatment()), random.Random(123))


def test_exclusive_named_cast_can_exceed_the_requested_character_count():
    slots = Slots(interpretation=_interpretation(planned=2, exclusive=True))
    led, _ = _run(slots, num_characters=1)
    assert len(led.data["cast"]) == 3  # two story characters plus announcer
    assert led.data["meta"]["my_story"]["fidelity_discrepancies"] == []
    assert len(slots.calls) == 4  # no cast-count repair


@pytest.mark.parametrize("requested,cast", [
    (1, ("Ada", "Tom", "Mabel")),
    (4, ("Ada", "Tom")),
])
@pytest.mark.parametrize("acts", [1, 3, 6])
def test_flexible_cast_records_requested_and_actual_counts_without_retry(requested, cast, acts):
    from nodes._otr_content_authorship import validate_receipt
    from nodes import _otr_freeze_cascade as FC
    slots = Slots(acts=acts, cast=cast)
    led, _ = _run(slots, act_count=acts, num_characters=requested)
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    counts = saved["meta"]["my_story"]["counts"]
    assert counts["requested_characters"] == requested
    assert counts["accepted_characters"] == counts["actual_characters"] == len(cast)
    assert counts["requested_acts"] == counts["actual_acts"] == acts
    assert len(slots.calls) == acts + 3
    assert FC._readonly_structural_validation(saved) == []
    validate_receipt(saved)


@pytest.mark.parametrize("raw_request", [-1, 0, 12])
def test_character_receipts_keep_raw_api_request_after_hint_normalization(raw_request):
    class ObservedSlots(Slots):
        def _answer(self, messages):
            if "work out what they actually want" in messages[0]["content"]:
                assert "distinct voices available" not in messages[-1]["content"]
            return super()._answer(messages)
    led, _ = _run(ObservedSlots(), num_characters=max(1, min(10, raw_request)),
                  raw_num_characters=raw_request)
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert saved["meta"]["my_story"]["counts"]["requested_characters"] == raw_request
    assert saved["meta"]["cast_contract"]["num_characters_request"] == raw_request
    assert saved["meta"]["my_story"]["counts"]["actual_characters"] == 2


@pytest.mark.parametrize("names", [("Ada", "Ada"), ("ANNOUNCER", "Tom")])
def test_treatment_rejects_ambiguous_or_reserved_cast_identities(names):
    check = MS._make_treatment_validator(1)
    assert check(MS.StoryTreatment(**_treatment(cast=names)))


def test_act_speaker_spelling_normalizes_to_the_accepted_cast():
    check = MS._make_act_validator(MS.StoryTreatment(**_treatment()), 1, ())
    act = MS.ActScript(**_act(speakers=("  ADA  ", "tom")))
    assert check(act) is None
    assert [line.speaker for line in act.lines] == ["Ada", "Tom", "Ada"]


def test_cast_pool_import_is_relative_first_for_comfy_package_loading():
    import ast
    tree = ast.parse(Path(MS.__file__).read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.ImportFrom) and node.level == 2
               and node.module == "config" and any(a.name == "cast_pools" for a in node.names)
               for node in ast.walk(tree))


def test_all_user_fields_reach_treatment_and_the_full_plan_reaches_acts():
    bundle = SI.build_bundle(SI.capture_raw(idea="IDEA-A", characters="CAST-B",
                             plot="PLOT-C", setting="SETTING-D"), SI.StoryRequest())
    captured = []
    def treatment_slot(messages, **kw):
        captured.append(messages[-1]["content"])
        return json.dumps(_treatment())
    treatment = MS._pass_treatment(treatment_slot, RT.resolve_story_pack("my_story"),
        bundle, MS.StoryInterpretation(**_interpretation()), act_count=1,
        requested_characters=2, include_act_breaks=True)
    assert all(value in captured[0] for value in ("IDEA-A", "CAST-B", "PLOT-C", "SETTING-D"))
    def act_slot(messages, **kw):
        captured.append(messages[-1]["content"])
        return json.dumps(_act())
    MS._pass_act(act_slot, RT.resolve_story_pack("my_story"), bundle, treatment,
                 treatment.acts[0], None, None, must_speak=("Ada", "Tom"), is_last=True)
    assert treatment.ending in captured[1] and treatment.dramatic_question in captured[1]
    assert "NOT YET HEARD" in captured[1]


def test_a_failed_act_retries_only_that_act():
    class RetryingSlots(Slots):
        def _answer(self, messages):
            if "one act of a radio drama" in messages[0]["content"]:
                self._act_seen += 1
                if self._act_seen == 2:
                    return "{broken JSON"
                return json.dumps(_act(1 if self._act_seen == 1 else 2, self._cast))
            return super()._answer(messages)
    slots = RetryingSlots(acts=2, inter=1)
    led, _ = _run(slots, act_count=2)
    assert slots._act_seen == 3  # accepted act 1 was not regenerated
    assert led.data["meta"]["my_story"]["acts_accepted"] == 2
    assert len(led.data["meta"]["my_story"]["pass_receipts"]) == 7


def test_capacity_failure_keeps_shared_facts_without_input_field_blame():
    from nodes._otr_generation_budget import GenerationContextOverflowError
    calls = []
    error = GenerationContextOverflowError("prompt 2048 exceeds context 2048", phase="prompt_no_room")
    def slot(*a, **kw):
        calls.append(True)
        raise error
    bundle = SI.build_bundle(SI.capture_raw(idea="x", plot="long plot " * 50), SI.StoryRequest())
    with pytest.raises(GenerationContextOverflowError) as caught:
        MS._pass_interpret(slot, RT.resolve_story_pack("my_story"), bundle,
                          requested=2, act_count=1, include_act_breaks=True)
    assert caught.value is error
    assert calls == [True]


def test_consecutive_speaker_lines_are_merged_without_losing_words():
    class SameSpeakerSlots(Slots):
        def _answer(self, messages):
            if "one act of a radio drama" in messages[0]["content"]:
                return json.dumps(_act(speakers=("Ada", "Ada", "Tom")))
            return super()._answer(messages)
    led, _ = _run(SameSpeakerSlots())
    rows = [r for r in led.data["lines"] if r.get("speaker_role") == "character"]
    assert rows[0]["text"] == "Line 1 from Ada. Line 2 from Ada."


def test_voice_assignment_forwards_the_accepted_age(monkeypatch):
    import random
    seen = []
    real = MS._OTRCAST.python_assign_voice_preset
    def assign(*a, **kw):
        seen.append(kw.get("age_band"))
        return real(*a, **kw)
    monkeypatch.setattr(MS._OTRCAST, "python_assign_voice_preset", assign)
    MS._assign_voices(MS.StoryTreatment(**_treatment()), random.Random(123))
    assert seen == ["30s", "30s"]


def test_real_writer_routes_user_fields_to_a_clean_ledger_and_shared_tail(monkeypatch):
    import importlib
    from nodes import _otr_story_drafts as DR, _otr_freeze_cascade as FC
    from nodes._otr_workflow_validator import WorkflowValidator
    writer = importlib.import_module("nodes.OTR_LedgerScriptWriter")
    slots = Slots()
    monkeypatch.setattr(writer._SlotScheduler, "for_slot",
                        lambda self, name: slots.technical if name == "technical" else slots.creative)
    captured = {}
    def tail(self, ctx, **kw):
        captured["ctx"] = ctx
        return ctx.led.data
    monkeypatch.setattr(writer.OTR_LedgerScriptWriter, "_run_writer_tail", tail)
    inputs = dict(
        source_bank="my_story", custom_premise="a keeper hears a voice",
        story_characters="Ada, the keeper. Tom.", story_plot="The bell answers.",
        story_setting="A rock lighthouse", story_author="A. Listener", act_count="1",
        num_characters=2, include_act_breaks=True, visual_style="roll (any style)")
    monkeypatch.delenv("OTR_SOURCE_SNAPSHOT_MANIFEST", raising=False)
    monkeypatch.setattr(DR, "_executing_context", lambda: DR.SubmissionContext(
        prompt_id="same-prompt", node_id="94"))
    WorkflowValidator._admit_story_input({"1": {
        "class_type": "OTR_LedgerScriptWriter", "inputs": {**inputs, "gate_in": ["94", 0]}
    }}, "94")
    draft_path, = DR.drafts_root().glob("*/input.json")
    queued_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(DR, "_executing_context", lambda: DR.SubmissionContext(
        prompt_id="same-prompt", node_id="1"))
    data = writer.OTR_LedgerScriptWriter().run(**inputs)
    meta = data["meta"]
    assert meta["delivery_intent"]["publication_required"] is True
    assert meta["delivery_intent"]["draft_digest"] == meta["source_meta"]["draft_digest"]
    saved = json.loads(Path(meta["story_draft"]["path"]).read_text(encoding="utf-8"))
    assert saved["request"]["visual_style_requested"] == "roll (any style)"
    assert saved["digest"] == meta["source_meta"]["draft_digest"]
    assert saved == queued_draft
    assert Path(meta["story_draft"]["path"]) == draft_path
    assert meta["credits_source_line"] == SI.credits_source_line("A. Listener")
    assert meta["story_attribution"]["author"] == "A. Listener"
    assert FC._readonly_structural_validation(data) == []
    assert captured["ctx"].run_story_spine is False
    assert len(list(DR.drafts_root().glob("*/input.json"))) == 1


def test_the_runner_refuses_a_run_that_never_went_through_admission():
    from nodes import production_ledger as PL

    led = PL.new_ledger(episode_id=None)
    with pytest.raises(MS.MyStoryError) as caught:
        MS.run_my_story_episode(
            payload={}, pack=RT.resolve_story_pack("my_story"),
            resolved={"source_meta": {}}, led=led, meta={},
            creative_fn=None, technical_fn=None, slot_scheduler=None,
            source_bank_row=RT.require_runnable_bank("my_story"),
            episode_root=None, episode_id=led.episode_id)
    assert "admission step did not execute" in str(caught.value)


def test_errors_name_this_lane_and_not_a_sibling():
    """A message reading '[scifi_news_pro] ...' on a listener's own story
    would send the next reader to the wrong module."""
    error = MS.MyStoryError("interpret", "something went wrong")
    assert str(error).startswith("[my_story] pass 'interpret'")


@pytest.mark.parametrize("acts,characters", [(1, 1), (3, 2), (6, 4)])
def test_full_treatment_repair_preserves_material_and_matches_variable_controls(acts, characters):
    from nodes._otr_content_authorship import validate_receipt
    cast = ("Ada", "Tom", "Mabel", "Ruth")[:characters]
    marker = "THE BELL SOUNDS AND THE FERRY TURNS AWAY"
    class CountRepair(Slots):
        repairs = 0
        def _answer(self, messages):
            if "radio dramatist" in messages[0]["content"]:
                prior = [m for m in messages if m["role"] == "assistant"]
                if prior:
                    self.repairs += 1
                    assert prior[-1]["content"] == self.failed_raw
                    assert marker in prior[-1]["content"]
                    assert prior[-1]["content"].index(marker) > 400
                    assert "a keeper hears a voice" in messages[1]["content"]
                    assert f"exactly {acts} acts" in messages[-1]["content"]
                    assert "character count is flexible" in messages[-1]["content"]
                    result = _treatment(acts, cast)
                else:
                    result = _treatment(acts + 1, (*cast, "Extra"))
                result["ending"] = marker
                result["acts"][-1]["ending_state"] = marker
                raw = "```json\n" + json.dumps(result) + "\n```"
                if not prior:
                    self.failed_raw = raw
                return raw
            return super()._answer(messages)
    slots = CountRepair(acts=acts, cast=cast)
    led, _ = _run(slots, act_count=acts, num_characters=characters)
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    story = saved["meta"]["my_story"]
    assert slots.repairs == 1
    assert story["counts"]["proposed_acts"] == acts + 1
    assert story["counts"]["proposed_characters"] == characters + 1
    assert story["counts"]["actual_acts"] == acts
    assert story["counts"]["actual_characters"] == characters
    assert story["treatment"]["ending"] == marker
    assert len(saved["scenes"]) == acts
    assert len([c for c in saved["music"] if c["placement"] == "interstitial"]) == acts - 1
    assert len({r["shot_id"] for r in saved["shots"]}) == len(saved["shots"])
    assert saved["meta"]["my_story"] == led.data["meta"]["my_story"]
    validate_receipt(saved)


def test_count_repair_exhaustion_remains_an_honest_failed_ledger(tmp_path):
    from nodes._otr_structured_call import StructuredCallFailedError
    class FencedSlots(Slots):
        def _answer(self, messages):
            return "```json\n" + super()._answer(messages) + "\n```"
    slots = FencedSlots(acts=2)
    with pytest.raises(StructuredCallFailedError, match="selected count is 1"):
        _run(slots, act_count=1)
    assert len(slots.calls) == 3  # interpretation, treatment, one typed repair
    path, = tmp_path.rglob("*_ledger.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    story = saved["meta"]["my_story"]
    assert story["counts"]["actual_acts"] is None
    assert story["counts"]["proposed_acts"] == 2
    assert [a["status"] for a in story["attempts"] if a["pass_id"] == "treatment"] == ["failed", "failed"]


@pytest.mark.parametrize("error", [RuntimeError("provider interrupted"), KeyboardInterrupt()])
@pytest.mark.parametrize("raise_on_save", [False, True])
def test_failed_receipt_save_preserves_the_original_failure(monkeypatch, caplog, error, raise_on_save):
    from nodes import production_ledger as PL
    class FailingSlot(Slots):
        def _answer(self, messages):
            def fail_save(self):
                if raise_on_save:
                    raise OSError("storage unavailable")
                return None
            monkeypatch.setattr(PL.Ledger, "save", fail_save)
            raise error
    with pytest.raises(type(error)) as caught:
        _run(FailingSlot())
    assert caught.value is error
    assert "attempt history did not persist" in caplog.text


def test_successful_story_cannot_hide_a_failed_final_receipt_save(monkeypatch):
    from nodes import production_ledger as PL
    real_save = PL.Ledger.save
    def fail_final(self):
        story = self.data.get("meta", {}).get("my_story", {})
        counts = story.get("counts", {})
        if counts.get("actual_acts") and counts.get("proposed_acts"):
            return None
        return real_save(self)
    monkeypatch.setattr(PL.Ledger, "save", fail_final)
    # A caller may already be handling an unrelated error: it must not make a
    # successful runner suppress its own failed save via inherited exc_info().
    try:
        raise ValueError("an earlier caller operation failed")
    except ValueError:
        with pytest.raises(MS.MyStoryError, match="attempt history"):
            _run(Slots())


def test_attribution_already_in_the_coda_is_not_appended_again():
    class CodaCredit(Slots):
        def _answer(self, messages):
            if "announcer's frame" in messages[0]["content"]:
                frame = _frame(self._attr)
                frame["announcer_intro"] = ["Good evening."]
                frame["coda"] = self._attr
                return json.dumps(frame)
            return super()._answer(messages)
    slots = CodaCredit()
    led, _ = _run(slots)
    assert sum(slots._attr in row.get("text", "") for row in led.data["lines"]) == 1


def test_missing_metadata_optional_frame_and_numbering_reach_readonly_freeze():
    from nodes import _otr_freeze_cascade as FC
    class Sparse(Slots):
        def _answer(self, messages):
            system = messages[0]["content"]
            if "radio dramatist" in system:
                return json.dumps({"cast": [{"name": "Ada"}, {"name": "Tom", "gender": "other"}],
                                   "acts": [{"n": 2}, {"n": 3}]})
            if "one act of a radio drama" in system:
                self._act_seen += 1
                return json.dumps(_act(90, ("Ada",) if self._act_seen == 1 else ("Tom",)))
            if "announcer's frame" in system:
                return json.dumps({"announcer_intro": ["", "Hello."],
                                   "announcer_outro": [""], "music_inter": ["", "unused cue"]})
            return super()._answer(messages)
    led, parts = _run(Sparse(acts=2), act_count=2)
    assert parts.final_title_override is None
    assert not next(c for c in led.data["cast"] if c["name"] == "Ada")["gender"]
    assert led.data["meta"]["my_story"]["treatment"]["cast"][0]["gender"] == ""
    story = led.data["meta"]["my_story"]
    assert story["act_number_normalization"]["replies"] == [{"original": 90, "slot": 1}, {"original": 90, "slot": 2}]
    assert story["music_cue_disposition"] == [{"proposal_index": 1, "description": "unused cue", "disposition": "unused_surplus"}]
    intro = next(r for r in led.data["lines"] if r.get("text") == "Hello.")
    assert intro["boundary"] == "shot_start"
    assert FC._readonly_structural_validation(led.data) == []


def test_breaks_off_preserves_all_unused_cue_proposals():
    led, _ = _run(Slots(acts=3, inter=4), act_count=3, include_act_breaks=False)
    assert all(c["placement"] != "interstitial" for c in led.data["music"])
    assert len(led.data["meta"]["my_story"]["music_cue_disposition"]) == 4


def test_speakable_coverage_repairs_only_last_act_with_full_dialogue():
    class CoverageRepair(Slots):
        second_calls = 0
        def _answer(self, messages):
            if "one act of a radio drama" in messages[0]["content"]:
                prior = [m for m in messages if m["role"] == "assistant"]
                if prior:
                    assert "ENDING MARKER" in prior[-1]["content"]
                    assert "Tom" in messages[-1]["content"]
                    self.second_calls += 1
                    return json.dumps(_act(2))
                self._act_seen += 1
                result = _act(self._act_seen, ("Ada",))
                result["lines"][0]["text"] = "The bell is waiting. " * 40 + "ENDING MARKER"
                result["lines"].append({"speaker": "Tom", "text": "(pauses)"})
                return json.dumps(result)
            return super()._answer(messages)
    slots = CoverageRepair(acts=2)
    led, _ = _run(slots, act_count=2)
    assert slots._act_seen == 2 and slots.second_calls == 1
    assert [a["status"] for a in led.data["meta"]["my_story"]["attempts"] if a["pass_id"] == "act_2"] == ["failed", "accepted"]


def test_provider_capacity_contract_reaches_every_slot_and_typed_repair():
    class CapacitySlots(Slots):
        def creative(self, messages, *, max_new_tokens, **kw):
            assert max_new_tokens is None
            assert messages._otr_output_budget_mode == "provider_capacity"
            assert messages._otr_prompt_must_fit is True
            return super().creative(messages, max_new_tokens=max_new_tokens, **kw)
        technical = creative
        def _answer(self, messages):
            if "radio dramatist" in messages[0]["content"]:
                return json.dumps(_treatment(1 if any(m["role"] == "assistant" for m in messages) else 2))
            return super()._answer(messages)
    led, _ = _run(CapacitySlots())
    assert all(p["budget_mode"] == "provider_capacity" and p["max_new_tokens"] is None
               for p in led.data["meta"]["my_story"]["pass_receipts"] if p["model_id"] != "python")
