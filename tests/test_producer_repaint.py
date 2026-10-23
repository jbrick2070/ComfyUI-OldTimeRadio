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

    class Declined(Exception):
        is_model_refusal = True

    def run(ledger, payload, policy=None, refuse=()):
        """Paint, the engine declining every object named in ``refuse``."""
        painted.clear()

        def paint(request):
            oid = str(request.get("object_id"))
            if oid in refuse:
                raise Declined("the model declined %s" % oid)
            painted.append(oid)
            return np.full((8, 8, 3), len(painted) % 250, dtype=np.uint8)

        led, _done, _report, _warn = D.dispatch_images(
            ledger, policy or POLICY, payload, gen_fn=paint, output_dir=str(tmp_path),
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


def _again(run, led, refuse=(), **changes):
    """A stills replay of ``led``: node 89's rebuild, then the dispatcher."""
    direction = dict(FIRST, attempt_id="0f8e1c2d-run2", out="02-stills", source="a" * 64)
    direction.update(changes)
    direction = PD.parse_direction(json.dumps(direction))
    wire = copy.deepcopy(led)
    wire[PD.RUNTIME_KEY] = direction
    payload, changed = MB.rebuild_producer_payload(wire, direction)
    led2, painted = run(wire, payload, refuse=refuse)
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
    assert kept["path"] == before["path"]
    assert kept["provenance"] == {"source": "kept", "painted": before["provenance"]}
    led3, painted, _changed = _again(studio, led2, scenes={"s01": {"note": "at dusk"}},
                                     kept={"stills": ["shot_b002"]})
    assert painted == [] and _rows(led3)["still_b002"]["provenance"] == kept["provenance"], \
        "kept twice, it still names the paint, not the first keep"


def test_takes_redraw_a_still_or_a_whole_character(studio):
    led = _first_run(studio)
    _led2, painted, changed = _again(studio, led, takes={"stills": {"shot_b003": 1}})
    assert painted == ["still_b003"] and changed == []
    _led3, painted, _changed = _again(studio, led, takes={"characters": {"c1": 1}})
    assert painted == ["c1"], "the portrait; these scene stills carry no character"


SHOTS = {"video": {"shots": [
    {"shot_id": "shot_b001", "char_id": "c1", "engine_id": "ltx25_foley_16gb",
     "source_line_ids": ["b001"], "creative": {"motion": "leans in"}},
    {"shot_id": "shot_b002", "char_id": "c1", "engine_id": "ltx25_foley_16gb",
     "source_line_ids": ["b002"], "creative": {"motion": "turns away"}},
    {"shot_id": "shot_b003", "char_id": "", "engine_id": "ltx25_foley_16gb",
     "source_line_ids": ["b003"]}]},
    "lines": [{"line_id": "b001", "text": "Who is there?", "start_s": 1.0, "dur_s": 2.0}],
    "cast": [{"char_id": "c1", "name": "MARA"}]}


def _cards(led, direction=None):
    """The cards a producer freeze of ``led`` would write."""
    ledger = dict(copy.deepcopy(led), **copy.deepcopy(SHOTS))
    files = [{"path": "%s/%s" % (pathlib.Path(row["path"]).parent.name,
                                  pathlib.Path(row["path"]).name)}
             for row in ledger["images"]["images"]]
    fields = PD.manifest_producer_fields(direction or PD.parse_direction(json.dumps(FIRST)))
    return PD.build_cards(ledger, fields, files)


def _video(**changes):
    direction = dict(FIRST, stage="video", out="02-video", source="a" * 64)
    direction.update(changes)
    return PD.parse_direction(json.dumps(direction))


def test_cards_record_what_each_still_was_painted_from(studio):
    led = _first_run(studio)
    cards = _cards(led)
    assert [g["scene_id"] for g in cards["scenes"]] == ["s01", "s02"]
    first = cards["scenes"][0]["shots"][0]
    assert first["shot_id"] == "shot_b001" and first["line"] == "Who is there?"
    assert first["picture_text"] == "a lamp-lit radio desk" and first["motion_text"] == "leans in"
    assert first["state"] == "painted" and first["still"].startswith("stills/")
    assert first["still_input_hash"] == led["images"]["images"][1][PD.ROW_INPUT_HASH]
    assert cards["scenes"][1]["shots"][0]["motion_editable"] is False
    assert cards["cast"] == [{"char_id": "c1", "name": "MARA", "take": 0, "has_portrait": True,
                              "picture": cards["cast"][0]["picture"]}]
    assert cards["cast"][0]["picture"].startswith("stills/c1_"), "portraits paint into stills/"
    assert [g["note_editable"] for g in cards["scenes"]] == [True, True]


def test_a_video_run_refuses_edits_its_stills_do_not_show(studio):
    cards = _cards(_first_run(studio))
    assert PD.unpainted_shots(cards, _video()) == []
    assert PD.unpainted_shots(cards, _video(
        cards={"shot_b001": {"motion_text": "stands up"}})) == [], "an action never repaints"
    assert PD.unpainted_shots(cards, _video(
        cards={"shot_b002": {"picture_text": "a broken window"}})) == ["shot_b002"]
    assert PD.unpainted_shots(cards, _video(
        scenes={"s01": {"note": "at dusk"}})) == ["shot_b001", "shot_b002"]
    assert PD.unpainted_shots(cards, _video(takes={"stills": {"shot_b003": 1}})) == ["shot_b003"]
    assert PD.unpainted_shots(cards, _video(
        scenes={"s01": {"note": "at dusk"}}, kept={"stills": ["shot_b001", "shot_b002"]})) == []


def test_a_kept_still_keeps_the_digest_of_the_paint_it_is(studio):
    led2, painted, _changed = _again(studio, _first_run(studio),
                                     scenes={"s01": {"note": "at dusk"}},
                                     kept={"stills": ["shot_b002"]})
    assert painted == ["still_b001"]
    cards = _cards(led2)
    note = {"s01": {"note": "at dusk"}}
    assert PD.unpainted_shots(cards, _video(scenes=note, kept={"stills": ["shot_b002"]})) == []
    # Un-kept, the still is what it was before the note, and that shows.
    assert PD.unpainted_shots(cards, _video(scenes=note)) == ["shot_b002"]


def test_an_edit_that_paints_the_same_prompt_is_not_unpainted(studio):
    # Codex QA, 2026-10-22: the painter sees only the composed prompt, so
    # whitespace, or a text moved between a card and its scene's note, paints
    # the same pixels and must not hold a Video run back.
    cards = _cards(_first_run(studio))
    assert PD.unpainted_shots(cards, _video(
        cards={"shot_b001": {"picture_text": "  a lamp-lit radio desk "}})) == []
    moved = _video(cards={"shot_b001": {"picture_text": "a lamp-lit radio desk, at dusk"},
                          "shot_b002": {"picture_text": "a rain-streaked window, at dusk"}})
    noted = _video(scenes={"s01": {"note": "at dusk"}})
    assert PD.unpainted_shots(cards, moved) == PD.unpainted_shots(cards, noted)


def test_a_stills_replay_refuses_a_policy_its_project_was_not_painted_with(studio):
    led = _first_run(studio)
    direction = PD.parse_direction(json.dumps(dict(FIRST, attempt_id="0f8e1c2d-run2",
                                                   out="02-stills", source="a" * 64)))
    wire = copy.deepcopy(led)
    wire[PD.RUNTIME_KEY] = direction
    payload, _changed = MB.rebuild_producer_payload(wire, direction)
    rewired = dict(POLICY, image_models={"character_image_model": {"engine_id": "cloud_flux_pro"}})
    with pytest.raises(D.ImageRenderError, match="character_image_model 'cloud_flux_pro'"):
        studio(wire, payload, policy=rewired)
    reseeded = dict(POLICY, seed={"request_seed": 99})
    with pytest.raises(D.ImageRenderError, match="seed"):
        studio(wire, payload, policy=reseeded)


def test_kept_stills_stay_in_the_engine_histogram(studio):
    # Codex QA, 2026-10-22: a replay's readiness plans image_engines_used from
    # this histogram, so a kept still's engine must stay named in it.
    led2, painted, _changed = _again(studio, _first_run(studio),
                                     kept={"stills": ["shot_b001", "shot_b002", "shot_b003"]})
    assert painted == []
    assert led2["meta"]["image_engines"]["by_role"]["character_video"]["flux_gen1"] == 4


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


# --------------------------------------------------------------------------- #
# Sonnet QA on chunk 3 (2026-10-22): each finding pinned
# --------------------------------------------------------------------------- #

def test_a_replay_with_the_shared_pool_cleared_paints_only_the_edit(studio):
    # A replay in a new workspace, or after the pool was cleared, still has
    # every row's episode copy; only the pool copies are gone.
    led = _first_run(studio)
    for row in led["images"]["images"]:
        if row["pool_path"] != row["path"]:
            pathlib.Path(row["pool_path"]).unlink()
    _led2, painted, _changed = _again(
        studio, led, cards={"shot_b002": {"picture_text": "a window lashed by a storm"}})
    assert painted == ["still_b002"]


def test_a_portrait_reference_falls_back_to_the_rows_episode_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", str(tmp_path))
    episode_copy = tmp_path / "stills" / "c1_abc.png"
    episode_copy.parent.mkdir()
    episode_copy.write_bytes(b"\x89PNG\r\n\x1a\nportrait")
    monkeypatch.setattr(D._pl, "portrait_path_for_hash",
                        lambda _h, _o: str(tmp_path / "pool" / "abc.png"))
    row = {"pool_path": str(tmp_path / "pool" / "abc.png"), "path": str(episode_copy)}
    assert D._portrait_png(row, "abc", str(tmp_path)) == str(episode_copy)
    assert D._portrait_png({"pool_path": row["pool_path"]}, "abc", str(tmp_path)) == ""


def test_superseded_rows_are_the_stills_no_longer_shown(studio):
    led = _first_run(studio)
    original = _rows(led)["still_b002"]["image_id"]
    led2, _painted, _changed = _again(
        studio, led, cards={"shot_b002": {"picture_text": "a window lashed by a storm"}})
    edited = _rows(led2)["still_b002"]["image_id"]

    def filed(ledger):
        return [row["image_id"] for row in ledger["images"][PD.SUPERSEDED_KEY]]

    led3, painted, _changed = _again(studio, led2)
    assert painted == [] and filed(led3) == [original], \
        "a cache hit keeps its id, so no copy of a live still is filed"
    led4, painted, _changed = _again(
        studio, led3, cards={"shot_b002": {"picture_text": "a rain-streaked window"}})
    assert painted == [] and _rows(led4)["still_b002"]["image_id"] == original
    assert filed(led4) == [edited], "the undone still is shown again and leaves the list"


def test_a_saved_prompt_that_differs_only_by_whitespace_is_unchanged():
    payload = copy.deepcopy(PAYLOAD)
    payload["objects"][1].update(prompt="a lamp-lit radio desk  ",
                                 prompt_hash=MB._content_hash("a lamp-lit radio desk  "))
    direction = PD.parse_direction(json.dumps(dict(FIRST, out="02-stills", source="a" * 64)))
    wire = {"beats": BEATS, "images": {PD.PAYLOAD_KEY: payload,
                                       PD.RAW_TEXTS_KEY: PD.initial_raw_texts(payload)}}
    rebuilt, changed = MB.rebuild_producer_payload(wire, direction)
    assert changed == [] and rebuilt == payload


def test_an_edit_to_a_failed_still_waits_for_a_stills_run(studio):
    led = _first_run(studio)
    led["images"]["images"] = [r for r in led["images"]["images"]
                               if r["object_id"] != "still_b003"]
    cards = _cards(led)
    failed = cards["scenes"][1]["shots"][0]
    assert (failed["state"], failed["still"], failed["still_input_hash"]) == ("failed", "", "")
    assert PD.unpainted_shots(cards, _video()) == [], "untouched, the video stage floors it"
    assert PD.unpainted_shots(cards, _video(
        cards={"shot_b003": {"picture_text": "a hall of valves"}})) == ["shot_b003"]
    assert PD.unpainted_shots(cards, _video(takes={"stills": {"shot_b003": 1}})) == ["shot_b003"]


def test_a_refused_repaint_shows_failed_and_waits_for_a_new_paint_or_a_kept_clip(studio):
    # The repaint was refused: the old row is still filed, and the receipt
    # names the beat a gap, which the render skips. The card must say so.
    led = _first_run(studio)
    led["images"]["required_scene_targets"] = [
        {"object_id": "still_b002", "status": "sanctioned_gap", "beat_id": "b002"}]
    cards = _cards(led)
    card = {c["shot_id"]: c for g in cards["scenes"] for c in g["shots"]}["shot_b002"]
    assert card["still"], "the old still is still shown"
    assert (card["still_state"], card["state"], card["still_input_hash"]) == \
        ("failed", "failed", "")
    assert PD.unpainted_shots(cards, _video()) == [], "untouched, the video stage floors it"
    publish = _video(stage="publish", out=None)
    assert "shot_b002 (its still was refused)" in PD.unfilmed_shots(cards, publish)
    assert "shot_b002 (its still was refused)" not in PD.unfilmed_shots(
        cards, _video(stage="publish", out=None, kept={"stills": [], "clips": ["shot_b002"]}))


def test_a_character_take_waits_for_a_stills_run_without_a_still_of_its_own(studio):
    # PAYLOAD's scene stills carry no character, so no card digest names c1:
    # only the portrait shows the take.
    led = _first_run(studio)
    retake = _video(takes={"characters": {"c1": 1}})
    assert PD.unpainted_shots(_cards(led), retake) == []
    assert PD.unpainted_characters(_cards(led), retake) == ["c1"]
    led2, painted, _changed = _again(studio, led, takes={"characters": {"c1": 1}})
    assert painted == ["c1"]
    assert _rows(led)["c1"][PD.ROW_IDENTITY_TAKE] == 0
    assert _rows(led2)["c1"][PD.ROW_IDENTITY_TAKE] == 1
    assert PD.unpainted_characters(_cards(led2), retake) == []
    assert PD.unpainted_characters(_cards(led2), _video()) == ["c1"], "takes are sent in full"


def test_a_declined_portrait_retake_is_not_taken_for_a_painted_one(studio):
    # Codex QA, 2026-10-22: the take asked for used to be recorded whatever
    # the model did, so a declined retake reached Video with the old face.
    led = _first_run(studio)
    retake = {"characters": {"c1": 1}}
    led2, painted, _changed = _again(studio, led, refuse=("c1",), takes=retake)
    assert painted == []
    assert _rows(led2)["c1"][PD.ROW_IDENTITY_TAKE] == 0, "the take-0 portrait is still live"
    cards = _cards(led2)
    assert (cards["cast"][0]["take"], cards["cast"][0]["has_portrait"]) == (0, True)
    assert PD.unpainted_characters(cards, _video(takes=retake)) == ["c1"]


def test_a_direction_names_only_what_its_bundles_cards_show(studio):
    led = _first_run(studio)
    cards = _cards(led)
    PD.check_direction_ids(cards, _video(
        cards={"shot_b001": {"picture_text": "a desk", "motion_text": "stands"}},
        scenes={"s02": {"note": "in fog"}}, takes={"characters": {"c1": 2}},
        kept={"stills": ["shot_b002"]}))
    for changes, words in (
            ({"cards": {"shot_b009": {"picture_text": "x"}}}, "no card shot_b009"),
            ({"kept": {"stills": ["shot_b009"]}}, "no card shot_b009"),
            ({"scenes": {"s09": {"note": "x"}}}, "no scene s09"),
            ({"takes": {"characters": {"c9": 1}}}, "no character c9"),
            ({"cards": {"shot_b003": {"motion_text": "walks"}}},
             "card shot_b003 has no action to edit")):
        with pytest.raises(PD.DirectionError, match=words):
            PD.check_direction_ids(cards, _video(**changes))
    # A shot whose lane paints no still: nothing to describe, redraw or keep.
    led["images"][PD.PAYLOAD_KEY]["objects"] = PAYLOAD["objects"][:3]
    led["images"]["images"] = [r for r in led["images"]["images"]
                               if r["object_id"] != "still_b003"]
    stillless = _cards(led)
    for changes, words in (
            ({"cards": {"shot_b003": {"picture_text": "x"}}}, "has no still to describe"),
            ({"takes": {"stills": {"shot_b003": 1}}}, "has no still to take again"),
            ({"kept": {"stills": ["shot_b003"]}}, "has no still to keep")):
        with pytest.raises(PD.DirectionError, match="shot_b003 " + words):
            PD.check_direction_ids(stillless, _video(**changes))
    # Takes are sent in full: a take of 0 is no take, and never refused.
    PD.check_direction_ids(stillless, _video(takes={"stills": {"shot_b003": 0}}))
    # A cast member with no portrait and no still of their own.
    stillless["cast"].append({"char_id": "c2", "name": "OTTO", "picture": "",
                              "has_portrait": False, "take": 0})
    PD.check_direction_ids(stillless, _video(takes={"characters": {"c2": 0}}))
    with pytest.raises(PD.DirectionError, match="character c2 has no picture to take again"):
        PD.check_direction_ids(stillless, _video(takes={"characters": {"c2": 1}}))
