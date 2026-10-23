"""Scene Producer stills replay: paint only what changed (chunk 3d, 2026-10-22).

The first producer stills run keeps the payload node 89 handed it and the raw
picture texts it painted from. A later stills run -- a REPLAY of that bundle --
has node 89 rebuild the payload from them, with the direction's picture edits,
scene notes and takes applied, and the dispatcher runs its CACHED path on it,
so every still whose inputs did not change is a cache hit and only the changed
ones are minted. Kept cards keep their verified stills through bulk edits, and
a producer run keeps one ledger row per object.

These tests run the real ``dispatch_images`` twice on a stand-in engine that
counts what it is asked to paint, and node 89's real rebuild in between.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import types

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_producer_direction as PD  # noqa: E402
from nodes import otr_image_gen_dispatcher as D  # noqa: E402
from nodes import otr_meta_brief_image_prompt as MB  # noqa: E402

FIRST = {"version": 1, "project_id": "a1b2c3d4e5f6", "attempt_id": "0f8e1c2d-run1",
         "stage": "stills", "out": "01-stills"}
POLICY = {"policy_version": 2,
          "image_models": {"character_image_model": {"engine_id": "flux_gen1"}},
          "video_models": {"announcer_video_model": {"engine_id": "still_motion"},
                           "music_video_model": {"engine_id": "still_motion"},
                           "character_video_model": {"engine_id": "still_motion"}},
          "seed": {"request_seed": 0}}


def _still(beat, prompt, *, char_id="", kind="scene_beat"):
    obj = {"object_id": "still_%s" % beat, "kind": kind, "role": "character_video",
           "beat_id": beat, "w": 832, "h": 480, "prompt": prompt,
           "prompt_hash": MB._content_hash(prompt), "source": "llm"}
    if char_id:
        obj["char_id"] = char_id
    return obj


PAYLOAD = {"version": 1, "objects": [
    {"object_id": "c1", "kind": "portrait", "role": "character_video", "char_id": "c1",
     "w": 832, "h": 1216, "prompt": "a weathered radio operator",
     "prompt_hash": MB._content_hash("a weathered radio operator"), "source": "llm"},
    _still("b001", "a lamp-lit radio desk"),
    _still("b002", "a rain-streaked window"),
    _still("b003", "a corridor of humming valves"),
]}
BEATS = [{"beat_id": "b001", "scene_id": "s01"}, {"beat_id": "b002", "scene_id": "s01"},
         {"beat_id": "b003", "scene_id": "s02"}]


@pytest.fixture
def studio(tmp_path):
    """The stand-in engine, registered for the test, and what it was asked to paint."""
    import numpy as np

    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_shared import role_compat as rc

    saved = dict(ireg._IMAGE_REGISTRY._registry)
    ireg._IMAGE_REGISTRY._registry.clear()
    ireg.register(types.SimpleNamespace(
        name="flux_gen1", roles=rc.ROLES, default_roles=rc.ROLES, commercial_clean=True,
        requires_flag=None, required_inputs=("text_prompt",), engine_version="1"))
    painted: list = []

    def paint(request):
        painted.append(str(request.get("object_id")))
        return np.full((8, 8, 3), len(painted) % 250, dtype=np.uint8)

    def run(ledger, payload):
        painted.clear()
        led, _done, _report, _warn = D.dispatch_images(
            ledger, POLICY, payload, gen_fn=paint, output_dir=str(tmp_path),
            lockdir=tmp_path / "lease.lockdir")
        return led, sorted(painted)

    yield run
    ireg._IMAGE_REGISTRY._registry.clear()
    ireg._IMAGE_REGISTRY._registry.update(saved)


def _first_run(run):
    ledger = {"episode_id": "ep_repaint", "cast": [{"char_id": "c1"}], "beats": BEATS,
              PD.RUNTIME_KEY: PD.parse_direction(json.dumps(FIRST))}
    led, painted = run(ledger, copy.deepcopy(PAYLOAD))
    assert painted == ["c1", "still_b001", "still_b002", "still_b003"]
    return led


def _again(run, led, **changes):
    """A stills replay of ``led``: node 89's rebuild, then the dispatcher."""
    direction = dict(FIRST, attempt_id="0f8e1c2d-run2", out="02-stills", source="a" * 64)
    direction.update(changes)
    direction = PD.parse_direction(json.dumps(direction))
    wire = copy.deepcopy(led)
    wire[PD.RUNTIME_KEY] = direction
    payload, changed = MB.rebuild_producer_payload(wire, direction)
    led2, painted = run(wire, payload)
    return led2, painted, changed


def _rows(led):
    return {row["object_id"]: row for row in led["images"]["images"]}


def test_an_unchanged_replay_paints_nothing(studio):
    led = _first_run(studio)
    led2, painted, changed = _again(studio, led)
    assert painted == [] and changed == []
    assert len(led2["images"]["images"]) == 4, "one row per object"


def test_a_picture_edit_paints_that_still_alone(studio):
    led = _first_run(studio)
    led2, painted, changed = _again(
        studio, led, cards={"shot_b002": {"picture_text": "a window lashed by a storm"}})
    assert painted == ["still_b002"] and changed == ["shot_b002"]
    assert led2["images"][PD.RAW_TEXTS_KEY]["cards"]["shot_b002"] == "a window lashed by a storm"
    assert led2["images"][PD.PAYLOAD_KEY] == PAYLOAD, "the first payload is kept as it was"
    assert len(led2["images"]["images"]) == 4, "the new still replaced the old row"


def test_an_edit_carries_into_the_next_bundle_without_painting_again(studio):
    led = _first_run(studio)
    led2, _painted, _changed = _again(
        studio, led, cards={"shot_b002": {"picture_text": "a window lashed by a storm"}})
    led3, painted, changed = _again(studio, led2)
    assert painted == [], "the edited still is a cache hit now"
    assert changed == ["shot_b002"], "its prompt still differs from the first payload's"


def test_a_scene_note_repaints_its_scene_and_clearing_it_restores_the_original(studio):
    led = _first_run(studio)
    led2, painted, _changed = _again(studio, led, scenes={"s01": {"note": "at dusk"}})
    assert painted == ["still_b001", "still_b002"], "scene s01 only"
    led3, painted, changed = _again(studio, led2, scenes={"s01": {"note": ""}})
    assert painted == [] and changed == [], "the first paint is back, from the cache"


def test_a_kept_card_keeps_its_still_through_a_scene_note(studio):
    led = _first_run(studio)
    before = _rows(led)["still_b002"]
    led2, painted, _changed = _again(studio, led, scenes={"s01": {"note": "at dusk"}},
                                     kept={"stills": ["shot_b002"]})
    assert painted == ["still_b001"]
    kept = _rows(led2)["still_b002"]
    assert kept["path"] == before["path"] and kept["provenance"] == {"source": "kept"}


def test_takes_redraw_a_still_or_a_whole_character(studio):
    led = _first_run(studio)
    _led2, painted, changed = _again(studio, led, takes={"stills": {"shot_b003": 1}})
    assert painted == ["still_b003"] and changed == []
    _led3, painted, _changed = _again(studio, led, takes={"characters": {"c1": 1}})
    assert painted == ["c1"], "the portrait; these scene stills carry no character"


def test_only_a_beats_own_scene_still_answers_to_its_card():
    assert PD.object_shot_id(_still("b001", "x")) == "shot_b001"
    assert PD.object_shot_id(PAYLOAD["objects"][0]) is None, "a portrait has no card"
    for oid in ("meshfodder_b001", "plate_b001", "still_b001_j1"):
        assert PD.object_shot_id({"object_id": oid, "beat_id": "b001"}) is None


def test_raw_texts_merge_by_name_and_an_empty_note_clears():
    previous = {"cards": {"shot_b001": "old desk", "shot_b002": "old window"},
                "scenes": {"s01": "at dusk", "s02": "in fog"}}
    direction = {"cards": {"shot_b001": {"picture_text": "new desk"},
                           "shot_b003": {"motion_text": "walks"}},
                 "scenes": {"s01": {"note": ""}}}
    assert PD.merged_raw_texts(previous, direction) == {
        "cards": {"shot_b001": "new desk", "shot_b002": "old window"},
        "scenes": {"s01": "", "s02": "in fog"}}
    assert PD.compose_picture("a desk", "at dusk") == "a desk, at dusk"
    assert PD.compose_picture("a desk", " ") == "a desk"


def test_a_blank_picture_text_is_refused_but_a_blank_action_is_not():
    later = dict(FIRST, out="02-stills", source="a" * 64)
    with pytest.raises(PD.DirectionError, match="picture_text is empty"):
        PD.parse_direction(json.dumps(dict(later, cards={"shot_b001": {"picture_text": "  "}})))
    assert PD.parse_direction(json.dumps(dict(later, cards={"shot_b001": {"motion_text": ""}})))


def test_a_replay_with_no_saved_payload_is_refused():
    wire = {"images": {}, PD.RUNTIME_KEY: PD.parse_direction(json.dumps(FIRST))}
    with pytest.raises(ValueError, match="carries no saved prompt payload"):
        MB.rebuild_producer_payload(wire, wire[PD.RUNTIME_KEY])


def test_node_89_rebuilds_only_for_a_producer_stills_replay(monkeypatch):
    direction = PD.parse_direction(json.dumps(dict(FIRST, out="02-stills", source="a" * 64)))
    wire = {"meta": {"replay_from": "bundle"}, "beats": BEATS,
            "images": {PD.PAYLOAD_KEY: copy.deepcopy(PAYLOAD)}, PD.RUNTIME_KEY: direction}
    out, report = MB.OTRMetaBriefImagePromptGen().generate(json.dumps(wire))
    assert json.loads(out) == PAYLOAD and report.startswith("producer stills replay")
    video = dict(direction, stage="video", out="02-video")
    wire[PD.RUNTIME_KEY] = video
    out, report = MB.OTRMetaBriefImagePromptGen().generate(json.dumps(wire))
    assert json.loads(out) == {"replay": True, "objects": []}, "every other replay passes through"


def test_the_dispatcher_paints_on_a_stills_replay_and_verifies_on_any_other(monkeypatch):
    calls = []
    monkeypatch.setattr(D, "dispatch_images", lambda led, *a, **k: (
        calls.append("paint") or (led, "image_done:ok", "made", [])))
    monkeypatch.setattr(D, "verify_replay_images", lambda led: (
        calls.append("verify") or (led, "image_done:replay", "verified")))
    monkeypatch.setattr(D, "_replay_rebase_to_live_episode", lambda led: led)
    monkeypatch.setattr(PD, "freeze_stage", lambda *a, **k: pathlib.Path("b"))
    monkeypatch.setattr(PD, "execution_blocker", lambda: "blocked")
    replay = {"episode_id": "e", "meta": {"replay_from": "bundle"}}
    stills = PD.parse_direction(json.dumps(dict(FIRST, out="02-stills", source="a" * 64)))
    video = PD.parse_direction(json.dumps(dict(FIRST, stage="video", out="02-video",
                                               source="a" * 64)))
    for direction in (stills, video, None):
        wire = dict(replay)
        if direction is not None:
            wire[PD.RUNTIME_KEY] = direction
        D.OTRImageGenDispatcher().dispatch(json.dumps(wire), episode_id="e")
    assert calls == ["paint", "verify", "verify"]
