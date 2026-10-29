"""Scene regions on the shipped haunted lane: the graph proven by execution, and the Mac lane left alone.

Regions shipped inside ``animatediff15_v3_haunted_video`` on 2026-10-28 (operator: "ship regions so we
have two animatediff"); the three lab ids are retired. Executes the haunted engine's ``render_clip``
against recorded fake node classes (a source grep cannot tell a branch that runs from one that does not)
and pins:

* the haunted lane declares regions and the Mac's lightning lane does not; the lab ids are retired;
* with objects, one text encode, one ``ConditioningSetMask`` bound to a generated mask and one
  ``ConditioningCombine`` are added per object on top of the UNCHANGED global positive, and the sampler
  receives the final combined conditioning; with none, the encode graph is the pre-regions one, call
  for call;
* the lightning lane never reads the field, never resolves a region class and never gains a receipt key;
* the receipt and the identity tuple move with the objects, the layout and the strength;
* the driver fills ``scene_objects`` for the haunted lane only, from the episode's own key-object pool.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.schemas import VideoRequest

HAUNTED = "animatediff15_v3_haunted_video"      # regions ship here
LIGHTNING = "animatediff15_lightning_video"     # the Mac lane: no regions
RETIRED_LAB_IDS = ("animatediff15_v3_haunted_regions_video", "animatediff15_lightning_regions_video",
                   "animatediff15_v3_stillin_lab_video")


class _Patcher:
    def __init__(self, tag):
        self.tag = tag
        self.detached = False

    def detach(self, unpatch_all=True):
        self.detached = True
        return self


class _Recorder:
    def __init__(self, source_request=16):
        self.calls = []
        self.source_request = source_request
        self.base_model = _Patcher("base")
        self.lora_model = _Patcher("lora")
        self.ade_model = _Patcher("ade")
        self.clip = object()
        self.vae = object()

    def classes(self):
        rec = self

        def _node(tag, result):
            class _N:
                FUNCTION = "go"

                def go(self, **kw):
                    rec.calls.append((tag, dict(kw)))
                    return result(kw) if callable(result) else result
            return _N

        def _sample(kw):
            n = int(kw["latent_image"]["samples"].shape[0])
            return ({"samples": torch.zeros((n, 4, 36, 64))},)

        def _decode(kw):
            n = int(kw["samples"]["samples"].shape[0])
            frames = np.zeros((n, 288, 512, 3), dtype=np.uint8)
            for i in range(n):
                frames[i, 0, 0, 0] = i % 256
            return (frames,)

        counter = {"text": 0}

        def _text(kw):
            counter["text"] += 1
            return ((("cond", kw["text"]),),)

        return {
            "checkpoint": _node("ckpt", (rec.base_model, rec.clip, rec.vae)),
            "text_encode": _node("text_encode", _text),
            "context": _node("context", ("CONTEXT_OPTS",)),
            "lora": _node("lora", (rec.lora_model,)),
            "ade": _node("ade", (rec.ade_model,)),
            "latent": _node("latent", lambda kw: ({"samples": torch.zeros(
                (int(kw["batch_size"]), 4, 36, 64))},)),
            "sampler": _node("sampler", _sample),
            "decode": _node("decode", _decode),
            "cond_mask": _node("cond_mask", lambda kw: (("masked", kw["conditioning"], kw["strength"]),)),
            "cond_combine": _node("cond_combine", lambda kw: (("joined", kw["conditioning_1"], kw["conditioning_2"]),)),
        }


def _request(target=32, scene_objects=(), shot_id="shot_b003"):
    return {
        "shot_id": shot_id,
        "request_id": shot_id,
        "text_prompt": "a brass lamp on a desk, mid-shot or wider",
        "negative_prompt": "text, watermark, caption, lettering, subtitles",
        "timing": {"target_frame_count": target},
        "seed_bundle": {"request_seed": 4242},
        "scene_objects": list(scene_objects),
    }


def _render(monkeypatch, engine_id, **req_kw):
    eng = vreg.get_engine(engine_id)
    rec = _Recorder(source_request=gs.ghost_source_request(32))
    monkeypatch.setattr(eng, "_classes", rec.classes())
    monkeypatch.setattr(eng, "_loaded", True)
    monkeypatch.setattr(eng, "_patchers", [rec.base_model])
    prepared = {"engine_id": eng.name, "lease": None, "patchers": eng._patchers, "session_ctx": {},
                "base_model": (rec.base_model,), "clip": (rec.clip,), "vae": (rec.vae,),
                "recipe": eng._recipe_receipt()}
    monkeypatch.setattr(wb, "encode_frames_to_silent_mp4",
                        lambda frames, out_path, fps, **kw: (out_path, int(np.asarray(frames).shape[0])))
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    monkeypatch.setattr(gs.GhostSignalEngine, "_vae_path", lambda self: "vae")
    raw = eng.render_clip(_request(**req_kw), prepared)
    return rec, eng, raw


# --------------------------------------------------------------------------- #
# Registration: regions on the haunted lane, none on lightning, the lab ids retired
# --------------------------------------------------------------------------- #
def test_the_haunted_lane_wants_regions_and_lightning_does_not():
    assert vreg.get_engine(HAUNTED).wants_scene_regions is True
    assert vreg.get_engine(LIGHTNING).wants_scene_regions is False
    assert vreg.get_engine(HAUNTED).recipe_receipt_id == (
        "animatediff_sd15_v3_haunted_regions_static16_512x288_v1")


@pytest.mark.parametrize("lab_id", RETIRED_LAB_IDS)
def test_the_lab_ids_are_retired_not_registered(lab_id):
    from nodes._otr_shared.public_engines import RETIRED_ENGINE_IDS
    assert lab_id in RETIRED_ENGINE_IDS
    assert not vreg.is_registered(lab_id)


def test_only_the_haunted_lane_resolves_the_two_core_region_classes():
    haunted, lightning = vreg.get_engine(HAUNTED), vreg.get_engine(LIGHTNING)
    assert haunted._node_candidates()["cond_mask"] == ("ConditioningSetMask",)
    assert haunted._node_candidates()["cond_combine"] == ("ConditioningCombine",)
    assert "cond_mask" not in lightning._node_candidates()
    assert "cond_combine" not in lightning._node_candidates()


# --------------------------------------------------------------------------- #
# The graph, proven by execution
# --------------------------------------------------------------------------- #
def test_with_no_objects_the_encode_graph_is_the_parents_call_for_call(monkeypatch):
    rec, _eng, _raw = _render(monkeypatch, HAUNTED, scene_objects=())
    tags = [name for name, _ in rec.calls]
    assert tags.count("text_encode") == 2
    assert tags.count("cond_mask") == 0 and tags.count("cond_combine") == 0


def test_each_object_adds_a_text_a_mask_and_a_combine_on_top_of_the_global_positive(monkeypatch):
    rec, eng, _raw = _render(monkeypatch, HAUNTED, scene_objects=("a brass lamp", "a battered disc"))
    tags = [name for name, _ in rec.calls]
    assert tags.count("text_encode") == 4          # global positive, negative, two objects
    assert tags.count("cond_mask") == 2 and tags.count("cond_combine") == 2
    texts = [kw["text"] for name, kw in rec.calls if name == "text_encode"]
    assert "a brass lamp on a desk, mid-shot or wider" in texts       # the global prompt is UNCHANGED
    assert "a brass lamp" in texts and "a battered disc" in texts
    masks = [kw for name, kw in rec.calls if name == "cond_mask"]
    for kw in masks:
        assert tuple(kw["mask"].shape) == (288, 512) and kw["mask"].dtype == torch.float32
        assert kw["set_cond_area"] == "default"
        assert kw["strength"] == pytest.approx(gs.SCENE_REGION_STRENGTH_DEFAULT)
    # left and right halves: the first object owns the left, the second the right
    assert masks[0]["mask"][:, :32].mean() > 0.9 and masks[0]["mask"][:, -32:].mean() < 0.1
    assert masks[1]["mask"][:, -32:].mean() > 0.9 and masks[1]["mask"][:, :32].mean() < 0.1


def test_the_sampler_receives_the_final_combined_conditioning_not_the_global_one(monkeypatch):
    rec, _eng, _raw = _render(monkeypatch, HAUNTED, scene_objects=("a brass lamp", "a battered disc"))
    sampler = [kw for name, kw in rec.calls if name == "sampler"][0]
    positive = sampler["positive"]
    assert positive[0] == "joined"                       # the last combine, two deep
    assert positive[1][0] == "joined"
    assert positive[1][1][0][1] == "a brass lamp on a desk, mid-shot or wider"   # the global positive
    assert positive[2][0] == "masked" and positive[1][2][0] == "masked"
    assert sampler["negative"][0][0] == "cond"          # the negative is untouched
    assert sampler["negative"][0][1].startswith("text, watermark")


def test_the_lightning_lane_receipt_gains_no_key_and_its_identity_keeps_its_shape():
    eng = vreg.get_engine(LIGHTNING)
    req = _request(scene_objects=("a brass lamp",))
    assert "scene_regions" not in eng.sampler_inputs_for(req)
    assert "scene_objects" not in eng._build_render_request(req)
    assert not any(str(part).startswith("regions:") for part in eng.shot_cache_identity(req))


def test_the_haunted_receipt_and_identity_move_with_the_objects():
    eng = vreg.get_engine(HAUNTED)
    none = eng.sampler_inputs_for(_request(scene_objects=()))
    one = eng.sampler_inputs_for(_request(scene_objects=("a brass lamp",)))
    two = eng.sampler_inputs_for(_request(scene_objects=("a brass lamp", "a battered disc")))
    assert none["scene_regions"]["objects"] == [] and none["scene_regions"]["layout"] == ""
    assert one["scene_regions"]["layout"] == "center" and two["scene_regions"]["layout"] == "halves"
    assert len({one["scene_regions"]["mask_sha8"], two["scene_regions"]["mask_sha8"]}) == 2
    ids = {eng.shot_cache_identity(_request(scene_objects=o))
           for o in ((), ("a brass lamp",), ("a brass lamp", "a battered disc"))}
    assert len(ids) == 3


def test_the_strength_dial_is_read_from_the_environment_and_a_bad_value_falls_back(monkeypatch):
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.delenv(gs.SCENE_REGION_STRENGTH_ENV, raising=False)
    assert eng._scene_region_strength() == gs.SCENE_REGION_STRENGTH_DEFAULT
    monkeypatch.setenv(gs.SCENE_REGION_STRENGTH_ENV, "3.5")
    assert eng._scene_region_strength() == 3.5
    for bad in ("abc", "0", "99", "-1", "nan"):
        monkeypatch.setenv(gs.SCENE_REGION_STRENGTH_ENV, bad)
        assert eng._scene_region_strength() == gs.SCENE_REGION_STRENGTH_DEFAULT, bad
    monkeypatch.setenv(gs.SCENE_REGION_STRENGTH_ENV, "3.5")
    assert vreg.get_engine(HAUNTED).sampler_inputs_for(
        _request(scene_objects=("a brass lamp",)))["scene_regions"]["strength"] == 3.5


@pytest.mark.parametrize("scene_objects", [(), ("a brass lamp",)])
def test_the_encode_externals_are_cleared_so_they_cannot_pin_the_clip_handle(monkeypatch, scene_objects):
    seen = []
    real = wb.run_graph

    def spy(graph, *args, **kwargs):
        external = kwargs.get("external_results")
        if external is not None and "clip" in external:
            seen.append(external)
        return real(graph, *args, **kwargs)

    monkeypatch.setattr(wb, "run_graph", spy)
    _render(monkeypatch, HAUNTED, scene_objects=scene_objects)
    assert seen, "the encode call was not observed"
    assert all(not d for d in seen)


def test_the_phrase_dial_wraps_each_object_and_leaves_the_global_prompt_alone(monkeypatch):
    monkeypatch.setenv(gs.SCENE_REGION_PHRASE_ENV, "a close photograph of {object}")
    rec, eng, _raw = _render(monkeypatch, HAUNTED, scene_objects=("a brass lamp", "a battered disc"))
    texts = [kw["text"] for name, kw in rec.calls if name == "text_encode"]
    assert "a close photograph of a brass lamp" in texts
    assert "a close photograph of a battered disc" in texts
    assert "a brass lamp on a desk, mid-shot or wider" in texts          # the global positive is untouched
    receipt = eng.sampler_inputs_for(_request(scene_objects=("a brass lamp",)))["scene_regions"]
    assert receipt["phrase"] == "a close photograph of {object}"


def test_the_default_phrase_is_the_object_alone(monkeypatch):
    monkeypatch.delenv(gs.SCENE_REGION_PHRASE_ENV, raising=False)
    eng = vreg.get_engine(HAUNTED)
    assert eng._scene_region_phrase("a brass lamp") == "a brass lamp"


@pytest.mark.parametrize("bad", ["no placeholder here", "{object} and {object}", "x" * 130 + " {object}"])
def test_an_unusable_phrase_template_lands_on_the_default(monkeypatch, bad):
    monkeypatch.setenv(gs.SCENE_REGION_PHRASE_ENV, bad)
    eng = vreg.get_engine(HAUNTED)
    assert eng._scene_region_phrase("a brass lamp") == "a brass lamp"


def test_the_strength_default_is_four_and_the_range_is_unchanged():
    assert gs.SCENE_REGION_STRENGTH_DEFAULT == 4.0
    assert (gs.SCENE_REGION_STRENGTH_MIN, gs.SCENE_REGION_STRENGTH_MAX) == (0.1, 10.0)
    assert gs.SCENE_REGION_STRENGTH_MIN <= gs.SCENE_REGION_STRENGTH_DEFAULT <= gs.SCENE_REGION_STRENGTH_MAX


def test_with_no_dial_set_the_haunted_lane_uses_four(monkeypatch):
    monkeypatch.delenv(gs.SCENE_REGION_STRENGTH_ENV, raising=False)
    assert vreg.get_engine(HAUNTED)._scene_region_strength() == 4.0


def test_the_identity_moves_with_the_phrase(monkeypatch):
    eng = vreg.get_engine(HAUNTED)
    monkeypatch.delenv(gs.SCENE_REGION_PHRASE_ENV, raising=False)
    plain = eng.shot_cache_identity(_request(scene_objects=("a brass lamp",)))
    monkeypatch.setenv(gs.SCENE_REGION_PHRASE_ENV, "a photograph of {object}")
    assert eng.shot_cache_identity(_request(scene_objects=("a brass lamp",))) != plain


def test_more_than_three_objects_are_cut_to_three(monkeypatch):
    eng = vreg.get_engine(HAUNTED)
    plan = eng._build_render_request(_request(scene_objects=("a", "b", "c", "d", "e")))
    assert plan["scene_objects"] == ("a", "b", "c")


# --------------------------------------------------------------------------- #
# The driver fills the field for the haunted lane only
# --------------------------------------------------------------------------- #
def _ledger():
    return {
        "meta": {"episode_seed": 1013426535, "visual_style": "archival_documentary",
                 "freeze_timestamp": "2026-08-22T22:27:56.943819+00:00",
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
             "text": "Bring the lamp over here, and keep the disc dry.",
             "beat_intent": "Adrian demands the lamp and the disc.", "traits": "tense",
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


def _plan(engine_id):
    pol = {"video_models": {"announcer_visual": engine_id, "music_visual": engine_id,
                            "character_visual": engine_id},
           "effective_video_models": {"announcer_visual": engine_id, "music_visual": engine_id,
                                      "character_video": engine_id}}
    led = _ledger()
    beats = sl.extract_beats(led)
    budget = sl.compute_clip_budget(beats, 25)
    for beat in beats:
        budget["per_beat"].setdefault(beat["beat_id"], 50)
    _groups, shots = sl.build_execution_plan(beats, budget, {}, pol, led, warnings=[])
    led["video"] = {"shots": shots, "fps": 25}
    return led, shots


def _shot(shots, shot_id):
    for shot in shots:
        if shot["shot_id"] == shot_id:
            return shot
    raise AssertionError(shot_id)


def test_the_driver_fills_scene_objects_for_the_haunted_lane_from_the_pool_the_beat_refers_to():
    led, shots = _plan(HAUNTED)
    req = rd.build_request_from_shot(_shot(shots, "shot_b002"), led, master_audio_path="")
    assert set(req["scene_objects"]) == {"a brass lamp", "a battered disc"}
    assert req["observability"]["scene_objects_source"]
    VideoRequest(**{k: v for k, v in req.items()})          # the declared field validates


def test_a_negated_noun_and_a_bookend_get_nothing():
    led, shots = _plan(HAUNTED)
    negated = rd.build_request_from_shot(_shot(shots, "shot_b003"), led, master_audio_path="")
    assert "a coal truck" not in negated["scene_objects"]
    bookend = rd.build_request_from_shot(_shot(shots, "shot_b001"), led, master_audio_path="")
    assert bookend["scene_objects"] == []


def test_the_driver_leaves_the_lightning_lane_untouched():
    led, shots = _plan(LIGHTNING)
    req = rd.build_request_from_shot(_shot(shots, "shot_b002"), led, master_audio_path="")
    assert req.get("scene_objects") in ([], None)
    assert "scene_objects_source" not in req["observability"]
