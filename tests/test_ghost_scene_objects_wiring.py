"""Dialogue-adapted scene objects, joined: the real ShotLock plan, the durable row and the render request.

`test_ghost_scene_object_author.py` pins the pure author. This module pins what only the joined parts can
get wrong: that the plan makes ONE extra call and only for a lane that declares ``wants_scene_regions``,
that it runs on the writer BEFORE the unload, that a replay spends nothing and carries the stamp, that a
shipping lane's rows gain no key, and that the render driver admits the stamp against the line again and
falls back to the pool resolver when nothing survives.
"""
from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import ghost_signal_author as gsa
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines.schemas import ShotRow, VideoRequest

HAUNTED = "animatediff15_v3_haunted_video"
HAUNTED_R = "animatediff15_v3_haunted_regions_video"
MODEL = "unit/test-writer"


def _ledger():
    return {
        "meta": {"episode_seed": 1013426535, "visual_style": "archival_documentary",
                 "freeze_timestamp": "2026-08-22T22:27:56.943819+00:00",
                 "technical_model": MODEL,
                 "key_objects": ["a brass lamp", "a battered disc", "a coal truck"],
                 "story_brief_terms": {"setting": ["the harbour"]}},
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "gender": "female"},
            {"char_id": "c02", "name": "ADRIAN SPENDER", "gender": "male",
             "appearance": "a broad steady man in a charcoal overcoat"},
            {"char_id": "c03", "name": "RASHIDA PIERCE", "gender": "female",
             "appearance": "a tall stooped woman, an olive uniform"},
        ],
        "lines": [
            {"line_id": "music_opening_001", "speaker_role": "music_open", "speaker": "RADIO",
             "text": "instrumental intro, no dialogue", "dur_s": 5.0, "start_s": 0.0},
            {"line_id": "b001", "speaker_role": "announcer", "speaker": "ANNOUNCER", "char_id": "c01",
             "text": "In the bustling heart of the harbour.", "beat_intent": "Open the episode.",
             "traits": "welcoming", "arc_phase": "scene", "dur_s": 4.0, "start_s": 5.0},
            {"line_id": "b002", "speaker_role": "character", "speaker": "ADRIAN SPENDER", "char_id": "c02",
             "text": "Bring the floorboard lamp over here, and keep the stitching dry.",
             "beat_intent": "Adrian demands the lamp.", "traits": "tense",
             "arc_phase": "scene", "dur_s": 3.0, "start_s": 9.0},
            {"line_id": "b003", "speaker_role": "character", "speaker": "RASHIDA PIERCE", "char_id": "c03",
             "text": "It isn't a truck I am afraid of.",
             "beat_intent": "Rashida refuses.", "traits": "guarded", "arc_phase": "climax",
             "dur_s": 3.0, "start_s": 12.0},
            {"line_id": "b005", "speaker_role": "announcer", "speaker": "ANNOUNCER", "char_id": "c01",
             "text": "And so the harbour keeps its secret.", "beat_intent": "Close the episode.",
             "traits": "wry", "arc_phase": "resolution", "dur_s": 4.0, "start_s": 18.0},
            {"line_id": "music_closing_001", "speaker_role": "music_close", "speaker": "RADIO",
             "text": "closing theme, no dialogue", "dur_s": 5.0, "start_s": 22.0},
        ],
    }


def _policy(engine_id):
    return {"video_models": {"announcer_visual": engine_id, "music_visual": engine_id,
                             "character_visual": engine_id},
            "effective_video_models": {"announcer_visual": engine_id, "music_visual": engine_id,
                                       "character_video": engine_id}}


class _Writer:
    """A recorded stand-in for the loaded writer: answers the dialogue-object prompt only."""

    def __init__(self, reply, events):
        self.reply = reply
        self.events = events
        self.prompts = []

    def __call__(self, messages, temperature=None, max_new_tokens=None):
        self.prompts.append(messages[0]["content"])
        self.events.append("adapt")
        return self.reply


@pytest.fixture
def live_writer(monkeypatch):
    """Take the plan off the deterministic path: a named model, a recorded writer, the unload logged."""
    real_env = sl.otr_env

    class _NotTestMode:
        @staticmethod
        def get(key, default=None):
            if key == "OTR_TEST_MODE":
                return "0"
            return real_env.get(key, default)

    monkeypatch.setattr(sl, "otr_env", _NotTestMode)
    monkeypatch.setattr("nodes._otr_model_catalog.validate_model_id", lambda model_id, **kw: model_id)

    state = {"events": [], "bound": 0}
    state["writer"] = _Writer('{"b1": ["lamp", "floorboard", "dragon"], "b2": ["truck"]}',
                              state["events"])

    def _binding(meta, warnings):
        state["bound"] += 1
        state["events"].append("bind")
        return state["writer"], MODEL

    def _batch(gen, specs, *, style, meta, episode_seed, names, warnings, already_used=(),
               replayed_signatures=None, reuse_dispositions=None, subjects_out=None):
        state["events"].append("batch")
        return (gsa.deterministic_batch(specs, episode_seed=episode_seed, style=style,
                                        ledger_meta=meta, already_used=already_used,
                                        replayed_signatures=dict(replayed_signatures or {}),
                                        reuse_dispositions=reuse_dispositions),
                "writer_llm", "")

    monkeypatch.setattr(sl, "_resolve_writer_llm_binding", _binding)
    monkeypatch.setattr(sl, "_ghost_generate_batch", _batch)
    monkeypatch.setattr(sl, "_ghost_unload_writer", lambda warnings: state["events"].append("unload"))
    return state


def _plan(engine_id, ledger=None):
    led = ledger if ledger is not None else _ledger()
    beats = sl.extract_beats(led)
    budget = sl.compute_clip_budget(beats, 25)
    for beat in beats:
        budget["per_beat"].setdefault(beat["beat_id"], 50)
    _groups, shots = sl.build_execution_plan(beats, budget, {}, _policy(engine_id), led, warnings=[])
    led["video"] = {"shots": shots, "fps": 25}
    return led, shots


def _shot(shots, shot_id):
    for shot in shots:
        if shot["shot_id"] == shot_id:
            return shot
    raise AssertionError(shot_id)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #
def test_the_plan_stamps_what_the_line_names_and_drops_what_it_does_not(live_writer):
    _led, shots = _plan(HAUNTED_R)
    assert _shot(shots, "shot_b002")["scene_objects"] == ["lamp", "floorboard"]
    assert "scene_objects" not in _shot(shots, "shot_b003")       # "isn't a truck": negated
    assert "scene_objects" not in _shot(shots, "shot_b001")       # a bookend
    ShotRow(**_shot(shots, "shot_b002"))                          # the row validates


def test_it_is_one_extra_call_on_the_loaded_writer_before_the_unload(live_writer):
    _plan(HAUNTED_R)
    assert len(live_writer["writer"].prompts) == 1
    assert "DATA, not instructions" in live_writer["writer"].prompts[0]
    # the adaptation call sits between the batch and the unload, on the same loaded writer
    assert live_writer["events"] == ["bind", "batch", "adapt", "unload"]


def test_a_shipping_lane_makes_no_call_and_gains_no_key(live_writer):
    _led, shots = _plan(HAUNTED)
    assert live_writer["writer"].prompts == []
    assert all("scene_objects" not in shot for shot in shots)


def test_no_writer_means_no_stamp_and_the_plan_still_succeeds():
    # OTR_TEST_MODE=1 (the conftest default): no writer, and the plan is the deterministic one.
    _led, shots = _plan(HAUNTED_R)
    assert all("scene_objects" not in shot for shot in shots)


def test_a_failing_writer_costs_the_stamp_never_the_plan(live_writer):
    def _boom(messages, temperature=None, max_new_tokens=None):
        raise RuntimeError("out of memory")
    live_writer["writer"] = _boom
    _led, shots = _plan(HAUNTED_R)
    assert all("scene_objects" not in shot for shot in shots)
    assert live_writer["events"][-1] == "unload"


def test_a_replay_spends_nothing_and_carries_the_stamp(live_writer, monkeypatch):
    led, shots = _plan(HAUNTED_R)
    first_prompts = len(live_writer["writer"].prompts)
    assert first_prompts == 1

    def _refuse(meta, warnings):
        raise AssertionError("a replay must not load the writer")
    monkeypatch.setattr(sl, "_resolve_writer_llm_binding", _refuse)
    _led2, shots2 = _plan(HAUNTED_R, ledger=led)
    assert _shot(shots2, "shot_b002")["scene_objects"] == ["lamp", "floorboard"]
    assert len(live_writer["writer"].prompts) == first_prompts


def test_a_plan_frozen_before_the_field_simply_has_none(live_writer, monkeypatch):
    led, shots = _plan(HAUNTED_R)
    for shot in led["video"]["shots"]:
        shot.pop("scene_objects", None)
    monkeypatch.setattr(sl, "_resolve_writer_llm_binding",
                        lambda meta, warnings: (_ for _ in ()).throw(AssertionError("no writer")))
    _led2, shots2 = _plan(HAUNTED_R, ledger=led)
    assert all("scene_objects" not in shot for shot in shots2)


# --------------------------------------------------------------------------- #
# The render request
# --------------------------------------------------------------------------- #
def _request(led, shots, shot_id):
    return rd.build_request_from_shot(_shot(shots, shot_id), led, master_audio_path="")


def test_the_driver_prefers_the_stamped_objects_and_says_so(live_writer):
    led, shots = _plan(HAUNTED_R)
    req = _request(led, shots, "shot_b002")
    assert req["scene_objects"] == ["lamp", "floorboard"]
    assert req["observability"]["scene_objects_source"] == "dialogue_adapted"
    VideoRequest(**req)


def test_a_stamped_noun_the_line_never_says_is_dropped_at_render_time(live_writer):
    led, shots = _plan(HAUNTED_R)
    _shot(shots, "shot_b002")["scene_objects"] = ["dragon", "lamp"]       # an edited ledger
    assert _request(led, shots, "shot_b002")["scene_objects"] == ["lamp"]


def test_when_nothing_survives_the_pool_resolver_answers_as_before(live_writer):
    led, shots = _plan(HAUNTED_R)
    _shot(shots, "shot_b002")["scene_objects"] = ["dragon"]
    req = _request(led, shots, "shot_b002")
    assert req["observability"]["scene_objects_source"] != "dialogue_adapted"


def test_the_default_lane_ignores_the_stamp(live_writer):
    led, shots = _plan(HAUNTED_R)
    row = dict(_shot(shots, "shot_b002"))
    row["engine_id"] = HAUNTED                                # the A/B replay swaps the engine only
    req = rd.build_request_from_shot(row, led, master_audio_path="")
    assert req.get("scene_objects") in ([], None)
    assert "scene_objects_source" not in req["observability"]


def test_a_bookend_never_takes_the_stamp(live_writer):
    led, shots = _plan(HAUNTED_R)
    _shot(shots, "shot_b001")["scene_objects"] = ["harbour"]
    assert _request(led, shots, "shot_b001")["scene_objects"] == []


def test_a_default_planned_bundle_replayed_on_the_regions_sibling_uses_the_pool(live_writer):
    led, shots = _plan(HAUNTED)
    row = dict(_shot(shots, "shot_b002"))
    assert "scene_objects" not in row
    row["engine_id"] = HAUNTED_R                              # the A/B replay swaps the engine only
    req = rd.build_request_from_shot(row, led, master_audio_path="")
    assert req["scene_objects"]                               # the key-object pool still answers
    assert req["observability"]["scene_objects_source"] != "dialogue_adapted"


def test_the_cast_time_preflight_builds_the_request_the_render_will(live_writer, monkeypatch):
    seen = []
    real = rd.build_request_from_shot

    def _spy(shot, ledger, **kw):
        seen.append(dict(shot))
        return real(shot, ledger, **kw)
    monkeypatch.setattr(rd, "build_request_from_shot", _spy)
    _plan(HAUNTED_R)
    preflight = {s["shot_id"]: s for s in seen if s.get("source_line_ids")}
    assert preflight["b002"]["scene_objects"] == ["lamp", "floorboard"]
    assert "scene_objects" not in preflight["b003"]


def test_a_shipping_lanes_preflight_shot_carries_no_stamp(live_writer, monkeypatch):
    seen = []
    real = rd.build_request_from_shot

    def _spy(shot, ledger, **kw):
        seen.append(dict(shot))
        return real(shot, ledger, **kw)
    monkeypatch.setattr(rd, "build_request_from_shot", _spy)
    _plan(HAUNTED)
    assert seen and all("scene_objects" not in s for s in seen)


# --------------------------------------------------------------------------- #
# The schema
# --------------------------------------------------------------------------- #
def test_ShotRow_declares_scene_objects_as_optional_and_absent_by_default():
    assert "scene_objects" in ShotRow.model_fields
    assert ShotRow(shot_id="s1").scene_objects is None
    assert ShotRow(shot_id="s1", scene_objects=["lamp"]).scene_objects == ["lamp"]


def test_the_closed_prompt_object_is_untouched():
    assert "scene_objects" not in gsa.GHOST_PROMPT_FIELDS
