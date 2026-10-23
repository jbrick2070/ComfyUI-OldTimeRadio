"""Scene Producer chunk 2: the direction on the wire, and the stages that stop.

What this file pins (planning/scene-producer/PLAN.md, "Graph contract"):

* Shot Lock puts a filled direction on the ledger it passes downstream under
  one top-level runtime key, and a blank widget leaves no key at all;
* a first-run direction cannot reach a replay run, nor a replay direction a
  fresh one;
* the image dispatcher, at the end of a stills stage, freezes the workspace
  into the bundle the direction names and returns ComfyUI's silent blocker on
  every output, so nothing downstream runs;
* Batch does the same at the end of a video stage, clips and foley included,
  and freezes nothing when a shot is unaccounted for;
* every other run -- blank, or a stage that ends elsewhere -- returns exactly
  what it returned before.

Offline: the Shot Lock cases use the replay suite's own frozen episode; the
freeze and the blocker are recorded instead of run.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_producer_direction as PD  # noqa: E402
from nodes import production_ledger as PL  # noqa: E402

DIRECTION = {"version": 1, "project_id": "a1b2c3d4e5f6",
             "attempt_id": "0f8e1c2d-3b4a-4c5d-9e8f-7a6b5c4d3e2f",
             "stage": "video", "out": "02-video", "source": "a" * 64}


def _direction(**changes) -> str:
    data = dict(DIRECTION)
    data.update(changes)
    return json.dumps({k: v for k, v in data.items() if v is not None})


def _replay_suite():
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_stages",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def replay_json(tmp_path, monkeypatch):
    """A real replay ledger: the replay suite's episode, frozen and imported."""
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    tcr = _replay_suite()
    ep, _ledger = tcr._make_episode(tmp_path / "episodes")
    bundle = tcr._freeze(ep, tmp_path / "bundles")
    led = PL.import_replay_bundle(str(bundle))
    return json.dumps(led.data, ensure_ascii=True, separators=(",", ":"))


@pytest.fixture
def recorded(monkeypatch):
    """Record every freeze and hand out a sentinel instead of ComfyUI's blocker."""
    calls = []
    sentinel = object()
    monkeypatch.setattr(PD, "freeze_stage",
                        lambda direction, ep, include_render_outputs: calls.append(
                            (direction["out"], ep, include_render_outputs)) or pathlib.Path("b"))
    monkeypatch.setattr(PD, "execution_blocker", lambda: sentinel)
    return calls, sentinel


POLICY = json.dumps({"policy_version": 2, "canvas": {"fps": 25}})


# --------------------------------------------------------------------------- #
# Shot Lock: the runtime key
# --------------------------------------------------------------------------- #

def test_a_blank_widget_puts_no_key_on_the_wire(replay_json):
    from nodes.otr_shot_lock import OTRShotLock
    for blank in ("", "   ", None):
        patched, *_ = OTRShotLock().lock(replay_json, video_policy_json=POLICY,
                                         direction_json=blank)
        assert PD.RUNTIME_KEY not in json.loads(patched)


def test_a_blank_widget_removes_a_key_inherited_on_the_wire(replay_json):
    from nodes.otr_shot_lock import OTRShotLock
    inherited = json.loads(replay_json)
    inherited[PD.RUNTIME_KEY] = PD.parse_direction(_direction())
    patched, *_ = OTRShotLock().lock(json.dumps(inherited), video_policy_json=POLICY)
    assert PD.RUNTIME_KEY not in json.loads(patched)


def test_a_filled_direction_rides_the_wire_and_never_the_durable_ledger(replay_json):
    from nodes.otr_shot_lock import OTRShotLock
    patched, *_ = OTRShotLock().lock(replay_json, video_policy_json=POLICY,
                                     direction_json=_direction())
    on_wire = PD.runtime_direction(json.loads(patched))
    assert on_wire == PD.parse_direction(_direction())
    assert PD.RUNTIME_KEY not in PL.peek_ledger().data


def test_a_first_run_direction_cannot_reach_a_replay(replay_json):
    from nodes.otr_shot_lock import OTRShotLock
    first = _direction(stage="stills", out="01-stills", source=None)
    with pytest.raises(ValueError, match="a first-run direction reached a replay run"):
        OTRShotLock().lock(replay_json, video_policy_json=POLICY, direction_json=first)


def test_what_this_build_cannot_honour_is_refused_before_any_work():
    from nodes.otr_shot_lock import OTRShotLock
    edits = _direction(cards={"shot_b001": {"picture_text": "a lamp-lit desk"}})
    with pytest.raises(ValueError, match="picture edits are not available in this build yet"):
        OTRShotLock().lock("{}", direction_json=edits)
    publish = _direction(stage="publish", out=None)
    with pytest.raises(ValueError, match="Publish is not available in this build yet"):
        OTRShotLock().lock("{}", direction_json=publish)


# --------------------------------------------------------------------------- #
# the image dispatcher ends a stills stage
# --------------------------------------------------------------------------- #

def _dispatch(monkeypatch, wire, episode_id="signal_lost_ep_1"):
    from nodes import otr_image_gen_dispatcher as D
    monkeypatch.setattr(D, "dispatch_images",
                        lambda led, policy, prompts, gen_fn=None, episode_id="": (
                            led, "image_done:ok", "made=2 reused=0", []))
    return D.OTRImageGenDispatcher().dispatch(json.dumps(wire), episode_id=episode_id)


def test_a_stills_stage_freezes_then_blocks_every_output(monkeypatch, recorded):
    calls, sentinel = recorded
    wire = {"episode_id": "signal_lost_ep_1", "meta": {},
            PD.RUNTIME_KEY: PD.parse_direction(_direction(stage="stills", out="01-stills",
                                                          source=None))}
    out = _dispatch(monkeypatch, wire)
    assert out == (sentinel, sentinel, sentinel)
    assert calls == [("01-stills", "signal_lost_ep_1", False)]


def test_a_video_stage_passes_the_direction_on_to_batch(monkeypatch, recorded):
    calls, _sentinel = recorded
    direction = PD.parse_direction(_direction())
    wire = {"episode_id": "signal_lost_ep_1", "meta": {}, PD.RUNTIME_KEY: direction}
    patched, done, report = _dispatch(monkeypatch, wire)
    assert PD.runtime_direction(json.loads(patched)) == direction
    assert done == "image_done:ok" and calls == []


def test_an_ordinary_dispatch_is_unchanged(monkeypatch, recorded):
    calls, _sentinel = recorded
    patched, done, report = _dispatch(monkeypatch, {"episode_id": "e", "meta": {}})
    assert PD.RUNTIME_KEY not in json.loads(patched)
    assert (done, report) == ("image_done:ok", "made=2 reused=0") and calls == []


# --------------------------------------------------------------------------- #
# Batch ends a video stage
# --------------------------------------------------------------------------- #

def _render(monkeypatch, wire, report):
    from nodes import otr_video_render_batch as B
    monkeypatch.setattr(B.OTRVideoRenderBatch, "_render_episode",
                        staticmethod(lambda _rd, ledger_json, master_audio_path="": (
                            report, '{"clips": []}', "node_episode_report.json")))
    return B.OTRVideoRenderBatch().render(json.dumps(wire), master_audio_path="")


def test_a_video_stage_freezes_clips_then_blocks(monkeypatch, recorded):
    calls, sentinel = recorded
    wire = {"video": {"shots": [{}]}, PD.RUNTIME_KEY: PD.parse_direction(_direction())}
    out = _render(monkeypatch, wire, {"ok": True, "episode_id": "signal_lost_ep_2"})
    assert out["result"] == (sentinel, sentinel)
    assert calls == [("02-video", "signal_lost_ep_2", True)]


def test_a_video_stage_with_an_unaccounted_shot_freezes_nothing(monkeypatch, recorded):
    calls, _sentinel = recorded
    wire = {"video": {"shots": [{}]}, PD.RUNTIME_KEY: PD.parse_direction(_direction())}
    with pytest.raises(ValueError, match="did not account for every shot"):
        _render(monkeypatch, wire, {"ok": False, "unaccounted_beat_count": 1,
                                    "episode_id": "e"})
    assert calls == []


def test_an_ordinary_render_is_unchanged(monkeypatch, recorded):
    calls, _sentinel = recorded
    out = _render(monkeypatch, {"video": {"shots": [{}]}},
                  {"ok": True, "episode_id": "e"})
    payload, manifest = out["result"]
    assert json.loads(payload)["ok"] is True and manifest == '{"clips": []}'
    assert calls == []


# --------------------------------------------------------------------------- #
# the shared rules
# --------------------------------------------------------------------------- #

def test_the_runtime_key_reader_ignores_anything_malformed():
    assert PD.runtime_direction(None) is None
    assert PD.runtime_direction({PD.RUNTIME_KEY: "stills"}) is None
    assert PD.runtime_direction({PD.RUNTIME_KEY: {"version": 2, "stage": "stills"}}) is None
    assert PD.runtime_direction({PD.RUNTIME_KEY: {"version": 1, "stage": "prepare"}}) is None


def test_the_blocker_is_comfyuis_own_and_silent():
    pytest.importorskip("comfy_execution.graph_utils")
    from comfy_execution.graph_utils import ExecutionBlocker
    blocked = PD.execution_blocker()
    assert isinstance(blocked, ExecutionBlocker) and blocked.message is None


# --------------------------------------------------------------------------- #
# chunk 2b: an edited action reaches the render
# --------------------------------------------------------------------------- #

ACTION = "slams the desk drawer shut"


def _character_shot(replay_json, engine):
    """The replay episode with its one shot made an ordinary character beat."""
    data = json.loads(replay_json)
    shot = data["video"]["shots"][0]
    shot.pop("ghost_prompt", None)
    shot.update({"engine_id": engine, "role": "character_video", "char_id": "c02",
                 "source_line_ids": ["b001"],
                 "creative": {"expression": "worried", "motion": "stands still",
                              "camera": "slow push in", "text_prompt": "frozen prompt",
                              "prompt_hash": "frozen"},
                 "cache_keys": {"prompt_hash": "frozen", "request_hash": "r"}})
    return data


@pytest.mark.parametrize("engine, keeps_line", [("ltx25_foley_16gb", False),
                                                ("humo_1.7B_169", True)])
def test_an_edited_action_reaches_the_lane_request(replay_json, engine, keeps_line):
    from nodes.otr_shot_lock import OTRShotLock
    from nodes._otr_video_engines import render_driver as RD
    data = _character_shot(replay_json, engine)
    direction = _direction(cards={"shot_b001": {"motion_text": ACTION}})
    patched, *_ = OTRShotLock().lock(json.dumps(data), video_policy_json=POLICY,
                                     direction_json=direction)
    ledger = json.loads(patched)
    shot = ledger["video"]["shots"][0]
    creative = shot["creative"]
    assert creative["motion"] == ACTION and ACTION in creative["text_prompt"]
    assert ("sequence is initiated" in creative["text_prompt"]) is keeps_line
    assert creative["prompt_hash"] != "frozen"
    assert shot["cache_keys"]["prompt_hash"] == creative["prompt_hash"]
    request = RD.build_request_from_shot(shot, ledger, phase="render")
    text = request.get("text_prompt") if isinstance(request, dict) else request.text_prompt
    assert ACTION in text
    # ...and the edited plan is the one stamped durably, so a freeze carries it.
    assert PL.peek_ledger().data["video"]["shots"][0]["creative"]["motion"] == ACTION


def test_an_unchanged_action_recomposes_nothing(replay_json):
    from nodes import otr_shot_lock as SL
    data = _character_shot(replay_json, "ltx25_foley_16gb")
    before = json.dumps(data["video"], sort_keys=True)
    assert SL.apply_action_edits(data["video"], data, data["meta"],
                                 {"shot_b001": {"motion_text": " stands still "}}) == []
    assert json.dumps(data["video"], sort_keys=True) == before


def test_an_action_on_an_unknown_or_uneditable_shot_is_refused(replay_json):
    from nodes import otr_shot_lock as SL
    data = _character_shot(replay_json, "ltx25_foley_16gb")
    with pytest.raises(ValueError, match="does not have: shot_zzz"):
        SL.apply_action_edits(data["video"], data, data["meta"],
                              {"shot_zzz": {"motion_text": ACTION}})
    ghost = json.loads(replay_json)          # the replay episode's own Ghost shot
    with pytest.raises(ValueError, match="have none to edit: shot_b001"):
        SL.apply_action_edits(ghost["video"], ghost, ghost["meta"],
                              {"shot_b001": {"motion_text": ACTION}})


def test_a_first_run_carries_no_edits():
    first = {k: v for k, v in DIRECTION.items() if k != "source"}
    first.update(stage="stills", out="01-stills",
                 cards={"shot_b001": {"motion_text": ACTION}})
    with pytest.raises(PD.DirectionError, match="first run has no cards yet"):
        PD.parse_direction(json.dumps(first))
