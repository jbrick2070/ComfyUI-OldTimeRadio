"""Scene Producer clip reuse (chunk 4a, 2026-10-22): the causal key, the
project's clip store, and a Video run that films only what changed.

The render loop runs for real -- ``render_driver.run_episode``, serial and
cloud fan-out -- on a stand-in camera that writes one clip file per shot into
scratch and records which shots it was asked to film. Each workspace holds the
same still and audio bytes at its own paths, as a replay workspace does.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_producer_clips as PC  # noqa: E402
from nodes import _otr_producer_direction as PD  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402
from nodes._otr_video_engines import render_driver as rd  # noqa: E402

PID = "a1b2c3d4e5f6"
PROMPTS = {"shot_b001": "a lamp-lit radio desk", "shot_b002": "a rain-streaked window",
           "shot_b003": "a corridor of humming valves"}


@pytest.fixture
def producer_root(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    monkeypatch.setattr(PD, "otr_episodes_root", lambda: episodes)
    return episodes


class _Camera:
    """A stand-in video engine. ``broken`` names shots it fails on, and
    ``timeouts`` shots whose provider times out -- a cloud floor on a cloud
    lane."""
    family = "abstract"
    roles = ("retired_role_b",)
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    fallback_engine = None
    engine_version = "1"

    def __init__(self, name, scratch):
        self.name = name
        self.scratch = scratch
        self.filmed = []
        self.broken = set()
        self.timeouts = set()

    def load(self):
        pass

    def unload(self):
        pass

    def assert_usable(self, host_caps=None, profile=None, request_template=None):
        return self.name

    def prepare(self, host_caps=None, profile=None, session_ctx=None):
        return {"engine_id": self.name}

    def render_clip(self, request, prepared):
        sid = request["shot_id"]
        if sid in self.broken:
            raise RuntimeError("the camera jammed on %s" % sid)
        if sid in self.timeouts:
            from nodes._otr_shared import cloud_media_backend as cmb
            raise cmb.CloudMediaError(cmb.CloudErrorCode.TIMEOUT,
                                      "the provider timed out on %s" % sid)
        self.filmed.append(sid)
        path = self.scratch / ("otr_engine_tmp_%s_%d.mp4" % (sid, len(self.filmed)))
        path.write_bytes(("%s|%s" % (sid, request.get("text_prompt"))).encode())
        return {"path": str(path)}

    def canonicalize(self, raw, request, profile=None):
        return {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25, "fps": 25, "path": raw["path"]}

    def teardown(self, prepared):
        pass


def _register(tmp_path, name):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    engine = _Camera(name, scratch)
    vreg.register(engine)
    return engine


@pytest.fixture
def cameras(tmp_path):
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    try:
        yield lambda name="stub_camera": _register(tmp_path, name)
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _workspace(root, name, engine_id="stub_camera", prompts=None):
    """An episode workspace and the ledger and request builder a producer
    Video run renders it with. Its stills and audio hold the same bytes in
    every workspace."""
    ep = root / name
    files = {}
    for sid in PROMPTS:
        still = ep / "stills" / ("still_%s.png" % sid)
        audio = ep / "audio" / ("slice_%s.wav" % sid)
        still.parent.mkdir(parents=True, exist_ok=True)
        audio.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(b"\x89PNG still of " + sid.encode())
        audio.write_bytes(b"RIFF audio of " + sid.encode())
        files[sid] = (still, audio)
    words = dict(PROMPTS, **(prompts or {}))
    shots = [{"shot_id": sid, "beat_id": sid[len("shot_"):], "role": "retired_role_b",
              "engine_id": engine_id, "family": "abstract", "group_id": "g%d" % n,
              "target_frame_count": 25, "degradation_trail": [],
              "creative": {"text_prompt": words[sid]}} for n, sid in enumerate(PROMPTS)]
    ledger = {"audio": {"master_audio_sha256": "frozen", "ledger_frozen": True},
              "video": {"video_revision": 1, "fps": 25, "shots": shots}}

    def build(shot, ledger, canvas=None, segment_index=0):
        still, audio = files[shot["shot_id"]]
        request = rd.build_request(shot, {"init_image": str(still), "audio_ref": str(audio)},
                                   25, canvas)
        request["text_prompt"] = shot["creative"]["text_prompt"]
        return request

    return ep, ledger, build


def _film(camera, workspace, bundle_map=None, **reuse):
    ep, ledger, build = workspace
    producer = PC.ProducerClips(project_id=PID, episode_root=ep, bundle_map=bundle_map,
                                **reuse)
    camera.filmed.clear()
    result = rd.run_episode(ledger, request_builder=build, producer=producer)
    return result, sorted(camera.filmed), producer


def _map(result):
    return result["ledger"]["video"][PC.MAP_KEY]


# --------------------------------------------------------------------------- #
# a Video run
# --------------------------------------------------------------------------- #

def test_a_first_video_run_films_every_shot_and_commits_each(producer_root, cameras,
                                                             tmp_path):
    camera = cameras()
    ep, ledger, build = workspace = _workspace(tmp_path, "ep_a")
    result, filmed, _producer = _film(camera, workspace)
    assert filmed == sorted(PROMPTS)
    records = _map(result)
    assert list(records) == list(PROMPTS), "in plan order"
    for sid, record in records.items():
        clip = record["files"]["clip"]
        assert clip["path"] == "clips/" + clip["name"] and clip["name"].startswith(sid)
        assert PC.file_sha256(ep / clip["path"]) == clip["sha256"]
        assert result["clips"][sid]["path"] == str(ep / clip["path"]), "out of scratch"
        assert PC.lookup(PC.store_dir(PID), record["key"])["files"]["clip"] == {
            k: v for k, v in clip.items() if k != "path"}
    assert not ledger["video"].get(PC.MAP_KEY), "the caller's ledger is untouched"


def test_a_fresh_workspace_reuses_every_shot_from_the_store(producer_root, cameras, tmp_path):
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    ep_b, ledger, build = workspace = _workspace(tmp_path, "ep_b")
    again, filmed, producer = _film(camera, workspace)
    assert filmed == [], "the same causes at other paths key the same"
    assert producer.reused == {sid: "store" for sid in PROMPTS}
    assert again["trace"] == first["trace"]
    for sid in PROMPTS:
        assert pathlib.Path(again["clips"][sid]["path"]).parent == ep_b / "clips"
    assert _map(again) == _map(first)


def test_a_replayed_bundle_reuses_its_own_clips_without_the_store(producer_root, cameras,
                                                                   tmp_path):
    camera = cameras()
    ep_a, _ledger, _build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, _ledger, _build))
    shutil.rmtree(str(PC.store_dir(PID)))
    workspace = _workspace(tmp_path, "ep_c")
    shutil.copytree(str(ep_a / "clips"), str(workspace[0] / "clips"))   # as the import does
    again, filmed, producer = _film(camera, workspace, bundle_map=_map(first))
    assert filmed == [] and producer.reused == {sid: "bundle" for sid in PROMPTS}


def test_an_edited_shot_is_the_only_one_filmed_again(producer_root, cameras, tmp_path):
    camera = cameras()
    _film(camera, _workspace(tmp_path, "ep_a"))
    again, filmed, producer = _film(camera, _workspace(
        tmp_path, "ep_b", prompts={"shot_b002": "a window lashed by a storm"}))
    assert filmed == ["shot_b002"]
    assert set(producer.reused) == {"shot_b001", "shot_b003"}
    assert set(_map(again)) == set(PROMPTS)


def test_a_run_that_dies_keeps_every_shot_it_finished(producer_root, cameras, tmp_path):
    camera = cameras()
    camera.broken = {"shot_b002"}
    with pytest.raises(rd.RenderError):
        _film(camera, _workspace(tmp_path, "ep_a"))
    camera.broken.clear()
    _again, filmed, producer = _film(camera, _workspace(tmp_path, "ep_b"))
    assert filmed == ["shot_b002", "shot_b003"]
    assert producer.reused == {"shot_b001": "store"}


def test_a_corrupt_store_entry_is_refused_filmed_again_and_moved_aside(producer_root, cameras,
                                                                     tmp_path):
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    key = _map(first)["shot_b001"]["key"]
    store = PC.store_dir(PID)
    (store / key / "clip.mp4").write_bytes(b"not the clip that was filmed")
    _again, filmed, _producer = _film(camera, _workspace(tmp_path, "ep_b"))
    assert filmed == ["shot_b001"]
    assert PC.lookup(store, key) is not None, "a whole entry is committed again"
    assert [p.name for p in store.iterdir() if p.name.startswith(".%s.refused-" % key)]


def test_a_bundle_record_that_does_not_verify_is_taken_from_the_store(producer_root, cameras,
                                                                     tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    workspace = _workspace(tmp_path, "ep_c")
    shutil.copytree(str(ep_a / "clips"), str(workspace[0] / "clips"))
    name = _map(first)["shot_b003"]["files"]["clip"]["name"]
    (workspace[0] / "clips" / name).write_bytes(b"tampered")
    _again, filmed, producer = _film(camera, workspace, bundle_map=_map(first))
    assert filmed == []
    assert producer.reused == {"shot_b001": "bundle", "shot_b002": "bundle",
                               "shot_b003": "store"}
    assert PC.file_sha256(workspace[0] / "clips" / name) == \
        _map(first)["shot_b003"]["files"]["clip"]["sha256"], "the store's copy replaced it"


def test_a_bundle_record_names_only_episode_files(producer_root, cameras, tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    forged = json.loads(json.dumps(_map(first)))
    forged["shot_b001"]["files"]["clip"]["path"] = "../ep_a/clips/" + \
        forged["shot_b001"]["files"]["clip"]["name"]
    workspace = _workspace(tmp_path, "ep_c")
    _again, filmed, producer = _film(camera, workspace, bundle_map=forged)
    assert filmed == [] and producer.reused["shot_b001"] == "store"


def test_the_cloud_fan_out_reuses_and_commits_on_its_workers(producer_root, cameras, tmp_path):
    camera = cameras("cloud_stub_camera")
    first, filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a",
                                                        engine_id="cloud_stub_camera"))
    assert filmed == sorted(PROMPTS) and list(_map(first)) == list(PROMPTS)
    _again, filmed, producer = _film(camera, _workspace(
        tmp_path, "ep_b", engine_id="cloud_stub_camera",
        prompts={"shot_b001": "a desk under a green lamp"}))
    assert filmed == ["shot_b001"]
    assert producer.reused == {"shot_b002": "store", "shot_b003": "store"}


def test_an_ordinary_run_keeps_no_map_and_commits_nothing(producer_root, cameras, tmp_path):
    camera = cameras()
    ep, ledger, build = _workspace(tmp_path, "ep_a")
    camera.filmed.clear()
    result = rd.run_episode(ledger, request_builder=build)
    assert sorted(camera.filmed) == sorted(PROMPTS)
    assert PC.MAP_KEY not in result["ledger"]["video"]
    assert all(pathlib.Path(c["path"]).parent == camera.scratch for c in result["clips"].values())
    assert not PC.store_dir(PID).exists()


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #

def _key(requests, **changes):
    parts = {"shot_id": "shot_b001", "coverage_plan": None,
             "engine": {"engine_id": "stub_camera", "engine_version": "1"},
             "render_policy": {"dtype_policy": "fp8_ok"}, "settings": {"fear_cape": True}}
    parts.update(changes)
    return PC.clip_reuse_key(requests=requests, **parts)


def test_the_key_sees_what_files_hold_never_where_they_are(tmp_path):
    def request(folder, still=b"still", **extra):
        path = tmp_path / folder / "still.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(still)
        base = {"shot_id": "shot_b001", "request_id": "shot_b001", "text_prompt": "a desk",
                "asset_refs": {"init_image": str(path)}, "seed_bundle": {"request_seed": 7},
                "observability": {"init_image": str(path)}, "plate_path": str(tmp_path / folder)}
        base.update(extra)
        return base

    same = _key([request("a")])
    assert _key([request("b")]) == same, "another workspace, the same bytes"
    assert _key([request("c", request_id="run-2", observability={"x": 1})]) == same
    for changed in (_key([request("d", still=b"a new still")]),
                    _key([request("e", text_prompt="a broken desk")]),
                    _key([request("f", seed_bundle={"request_seed": 8})]),
                    _key([request("g")], coverage_plan={"join_mode": "chain"}),
                    _key([request("h")], engine={"engine_id": "stub_camera",
                                                 "engine_version": "2"}),
                    _key([request("i")], render_policy={"dtype_policy": "bf16"}),
                    _key([request("j")], settings={"fear_cape": False}),
                    _key([request("k"), request("k2")])):
        assert changed != same
    with pytest.raises(TypeError, match="is a dict"):
        PC.clip_reuse_key(shot_id="s", requests=["not a request"], coverage_plan=None,
                          engine={}, render_policy={}, settings={})


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #

def _clip_file(tmp_path, body=b"clip bytes"):
    path = tmp_path / "clips" / "shot_b001_role_engine.mp4"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(body)
    return path


BODY = {"shot_id": "shot_b001", "attempts": ["stub_camera"], "clip": {"frame_count": 25}}


def test_a_store_entry_is_whole_or_absent(tmp_path):
    store, key = tmp_path / "store", "ab" * 32
    store.mkdir()
    (store / (".%s.staging-dead" % key)).mkdir()   # a writer that died mid-commit
    assert PC.lookup(store, key) is None, "a staging folder is never an entry"
    record = PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path))})
    assert PC.lookup(store, key) == record
    assert record["files"]["clip"]["name"] == "shot_b001_role_engine.mp4"
    second = PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path, b"other bytes"))})
    assert second == record, "a committed entry is never changed"


def test_a_tampered_entry_is_refused(tmp_path):
    store, key = tmp_path / "store", "cd" * 32
    PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path))})
    (store / key / "clip.mp4").write_bytes(b"different bytes")
    with pytest.raises(PC.ClipStoreError, match="not the file its record names"):
        PC.lookup(store, key)
    record = json.loads((store / key / PC.RECORD_NAME).read_text(encoding="utf-8"))
    record["key"] = "ef" * 32
    (store / key / PC.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PC.ClipStoreError, match="records key"):
        PC.lookup(store, key)
    with pytest.raises(PC.ClipStoreError, match="not a clip key"):
        PC.lookup(store, "../escape")


# --------------------------------------------------------------------------- #
# chunk 4b: clip takes, floors and kept clips
# --------------------------------------------------------------------------- #

def _direction(**changes):
    data = {"version": 1, "project_id": PID, "attempt_id": "0f8e1c2d-run3",
            "stage": "video", "out": "03-video", "source": "a" * 64}
    data.update(changes)
    return PD.parse_direction(json.dumps(data))


def test_a_clip_take_is_another_draw_and_take_0_is_the_seed_itself(monkeypatch):
    monkeypatch.setattr(rd, "_confined_ledger_path", lambda path, _field: path)
    ledger = {"audio": {"master_audio_sha256": "frozen", "ledger_frozen": True},
              "lines": [{"line_id": "b001", "char_id": "c03", "speaker_role": "character",
                         "start_s": 0.0, "dur_s": 2.0, "bark_wav_path": "X:/a/seg_b001.wav"}],
              "images": {"images": [{"object_id": "c03", "path": "X:/img/c03.png"}]},
              "video": {"video_revision": 1, "fps": 25, "shots": [
                  {"shot_id": "shot_b001", "source_line_ids": ["b001"], "engine_id": "abstract",
                   "target_frame_count": 50, "render_request_hash": "hash_b001",
                   "creative": {"text_prompt": "a desk"}}]}}
    shot = ledger["video"]["shots"][0]

    def seed():
        return rd.build_request_from_shot(shot, ledger)["seed_bundle"]["request_seed"]

    plain = seed()
    ledger[PD.RUNTIME_KEY] = _direction(takes={"clips": {"shot_b001": 0}})
    assert seed() == plain, "take 0 is the seed itself"
    ledger[PD.RUNTIME_KEY] = _direction(takes={"clips": {"shot_b001": 1}})
    one = seed()
    ledger[PD.RUNTIME_KEY] = _direction(takes={"clips": {"shot_b001": 2}})
    assert len({plain, one, seed()}) == 3
    ledger[PD.RUNTIME_KEY] = _direction(takes={"clips": {"shot_b009": 5}})
    assert seed() == plain, "another shot's take leaves this one alone"


def test_a_shot_that_renders_sheds_the_floor_marks_its_replay_carried(producer_root, cameras,
                                                                      tmp_path):
    # A replay plans from the section its source stamped, floors included:
    # without this, a retried beat that rendered was reported floored for good.
    camera = cameras()
    for producer in (True, False):
        ep, ledger, build = _workspace(tmp_path, "ep_%s" % producer)
        ledger["video"]["shots"][1].update(status="sanctioned_gap", cloud_floor="timeout",
                                           floor_render="still_pan")
        if producer:
            result, _filmed, _p = _film(camera, (ep, ledger, build))
        else:
            result = rd.run_episode(ledger, request_builder=build)
        assert not set(rd._FLOOR_MARKS) & set(result["ledger"]["video"]["shots"][1])


@pytest.fixture
def floors(monkeypatch, tmp_path):
    """The still_pan floor drawn as a stand-in file in scratch: these tests are
    about what the producer keeps of a floor, not how ffmpeg pans a still."""
    drawn = []
    scratch = tmp_path / "floor_scratch"
    scratch.mkdir()

    def floor(shot, ledger, **_kw):
        drawn.append(shot["shot_id"])
        clip = scratch / ("otr_floor_%s_%d.mp4" % (shot["shot_id"], len(drawn)))
        clip.write_bytes(b"the scene still of " + shot["shot_id"].encode())
        return {"path": str(clip), "engine_id": "still_pan", "frame_count": 25, "fps": 25}

    monkeypatch.setattr(rd, "_render_still_floor", floor)
    return drawn


def _cloud_workspace(tmp_path, name, **changes):
    return _workspace(tmp_path, name, engine_id="cloud_stub_camera", **changes)


@pytest.mark.parametrize("fanout", ["1", "2"])
def test_a_floored_shot_is_a_failed_card_and_its_retry_comes_back_filmed(
        producer_root, cameras, floors, tmp_path, monkeypatch, fanout):
    monkeypatch.setenv("OTR_CLOUD_FANOUT", fanout)
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    first, filmed, _producer = _film(camera, _cloud_workspace(tmp_path, "ep_a"))
    assert filmed == ["shot_b001", "shot_b003"] and floors == ["shot_b002"]
    floored = first["ledger"]["video"][PC.FLOOR_KEY]
    assert list(floored) == ["shot_b002"] and "shot_b002" not in _map(first)
    assert floored["shot_b002"]["marks"]["cloud_floor"] == "timeout"
    assert floored["shot_b002"]["files"]["clip"]["path"] == \
        "clips/shot_b002_retired_role_b_still_pan.mp4"
    # The provider is back: the floored shot is filmed again, and only it.
    camera.timeouts.clear()
    ep_b, ledger, build = _cloud_workspace(tmp_path, "ep_b")
    ledger["video"]["shots"][1].update(first["ledger"]["video"]["shots"][1])
    again, filmed, _producer = _film(camera, (ep_b, ledger, build),
                                     bundle_map=_map(first), bundle_floors=floored)
    assert filmed == ["shot_b002"]
    assert not set(rd._FLOOR_MARKS) & set(again["ledger"]["video"]["shots"][1])
    assert again["ledger"]["video"][PC.FLOOR_KEY] == {} and "shot_b002" in _map(again)


def test_a_kept_floor_is_used_as_it_is_and_stays_a_floor(producer_root, cameras, floors,
                                                         tmp_path):
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    ep_a, ledger_a, build_a = _cloud_workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger_a, build_a))
    floored = first["ledger"]["video"][PC.FLOOR_KEY]
    camera.timeouts.clear()
    ep_b, ledger, build = _cloud_workspace(tmp_path, "ep_b")
    shutil.copytree(str(ep_a / "clips"), str(ep_b / "clips"))       # as the import does
    ledger["video"]["shots"][1].update(first["ledger"]["video"]["shots"][1])
    again, filmed, producer = _film(camera, (ep_b, ledger, build), bundle_map=_map(first),
                                    bundle_floors=floored, kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept floor"
    shot = again["ledger"]["video"]["shots"][1]
    assert (shot["cloud_floor"], shot["floor_render"]) == ("timeout", "still_pan")
    assert pathlib.Path(again["clips"]["shot_b002"]["path"]).parent == ep_b / "clips"
    assert again["ledger"]["video"][PC.FLOOR_KEY] == floored, "the floor rides on"


def test_a_kept_clip_is_used_whatever_changed_and_a_missing_one_refuses(
        producer_root, cameras, tmp_path):
    camera = cameras()
    ep_a, ledger_a, build_a = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger_a, build_a))
    workspace = _workspace(tmp_path, "ep_c", prompts={"shot_b002": "an edited window"})
    shutil.copytree(str(ep_a / "clips"), str(workspace[0] / "clips"))
    _again, filmed, producer = _film(camera, workspace, bundle_map=_map(first),
                                     kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept"
    with pytest.raises(PC.ClipStoreError, match="card shot_b002 keeps its clip, but"):
        _film(camera, _workspace(tmp_path, "ep_d"), bundle_map=_map(first),
              kept=["shot_b002"])


def test_a_film_again_records_its_take(producer_root, cameras, tmp_path):
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"),
                                      takes={"shot_b001": 3})
    assert _map(first)["shot_b001"]["clip_take"] == 3
    assert _map(first)["shot_b002"]["clip_take"] == 0


def test_cards_show_each_clip_and_its_state(producer_root, cameras, floors, tmp_path):
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    first, _filmed, _producer = _film(camera, _cloud_workspace(tmp_path, "ep_a"),
                                      takes={"shot_b003": 2})
    video = first["ledger"]["video"]
    files = [{"path": r["files"]["clip"]["path"]}
             for r in list(video[PC.MAP_KEY].values()) + list(video[PC.FLOOR_KEY].values())]

    def cards(**kept):
        fields = PD.manifest_producer_fields(_direction(kept=kept))
        return {c["shot_id"]: c for g in PD.build_cards(first["ledger"], fields, files)["scenes"]
                for c in g["shots"]}

    shown = cards()
    assert shown["shot_b001"]["clip"] == video[PC.MAP_KEY]["shot_b001"]["files"]["clip"]["path"]
    assert (shown["shot_b001"]["clip_state"], shown["shot_b001"]["state"]) == ("filmed", "filmed")
    assert shown["shot_b001"]["clip_input_hash"] == video[PC.MAP_KEY]["shot_b001"]["key"]
    assert shown["shot_b003"]["clip_take"] == 2
    floor = shown["shot_b002"]
    assert (floor["clip_state"], floor["state"], floor["clip_floor"]) == \
        ("failed", "failed", "timeout")
    assert floor["clip"] == "clips/shot_b002_retired_role_b_still_pan.mp4"
    kept = cards(clips=["shot_b002"])["shot_b002"]
    assert (kept["clip_state"], kept["state"]) == ("kept", "kept")
    # Node 63 lets a card keep only a clip, or a floor, it has.
    shown_cards = PD.build_cards(first["ledger"], PD.manifest_producer_fields(_direction()),
                                 files)
    PD.check_direction_ids(shown_cards, _direction(kept={"clips": ["shot_b001", "shot_b002"]}))
    for group in shown_cards["scenes"]:
        for card in group["shots"]:
            if card["shot_id"] == "shot_b003":
                card.update(clip=None, clip_state="")
    with pytest.raises(PD.DirectionError, match="card shot_b003 has no clip to keep"):
        PD.check_direction_ids(shown_cards, _direction(kept={"clips": ["shot_b003"]}))


# --------------------------------------------------------------------------- #
# chunk 4c: Publish
# --------------------------------------------------------------------------- #

def _replayed(tmp_path, first_ep, name, **changes):
    """A workspace laid out as a replay of ``first_ep``'s bundle: its clips
    copied in, as the import does."""
    workspace = _workspace(tmp_path, name, **changes)
    shutil.copytree(str(first_ep / "clips"), str(workspace[0] / "clips"))
    return workspace


def test_publish_films_nothing_and_refuses_a_shot_it_has_no_clip_for(producer_root, cameras,
                                                                    tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    published, filmed, producer = _film(camera, _replayed(tmp_path, ep_a, "ep_p"),
                                        bundle_map=_map(first), stage="publish")
    assert filmed == [] and set(producer.reused.values()) == {"bundle"}
    assert _map(published) == _map(first)
    # An action nobody has filmed: Publish refuses it, and renders nothing.
    with pytest.raises(PC.ClipStoreError, match="Publish has no filmed clip for shot shot_b002"):
        _film(camera, _replayed(tmp_path, ep_a, "ep_q",
                                prompts={"shot_b002": "a window nobody filmed"}),
              bundle_map=_map(first), stage="publish")
    assert camera.filmed == []


def test_publish_takes_a_kept_floor_and_refuses_one_that_is_not(producer_root, cameras, floors,
                                                               tmp_path):
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    ep_a, ledger, build = _cloud_workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    floored = first["ledger"]["video"][PC.FLOOR_KEY]
    camera.timeouts.clear()

    def publish(name, **reuse):
        workspace = _replayed(tmp_path, ep_a, name, engine_id="cloud_stub_camera")
        workspace[1]["video"]["shots"][1].update(first["ledger"]["video"]["shots"][1])
        return _film(camera, workspace, bundle_map=_map(first), bundle_floors=floored,
                     stage="publish", **reuse)

    with pytest.raises(PC.ClipStoreError, match="Publish has no filmed clip for shot shot_b002"):
        publish("ep_p")
    published, filmed, producer = publish("ep_q", kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept floor"
    assert published["ledger"]["video"]["shots"][1]["cloud_floor"] == "timeout", \
        "published visibly degraded, never passed off as filmed"


def test_publish_refuses_a_refused_still_gap_unless_its_card_is_kept(producer_root, cameras,
                                                                    tmp_path, monkeypatch):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    monkeypatch.setattr(rd, "sanctioned_gap_beat_ids", lambda ledger: {"b002"})
    with pytest.raises(PC.ClipStoreError, match="Publish refuses shot shot_b002: its still"):
        _film(camera, _replayed(tmp_path, ep_a, "ep_p"), bundle_map=_map(first),
              stage="publish")
    published, filmed, _producer = _film(camera, _replayed(tmp_path, ep_a, "ep_q"),
                                         bundle_map=_map(first), stage="publish",
                                         kept=["shot_b002"])
    assert filmed == [] and "shot_b002" in published["ledger"]["video"][PC.FLOOR_KEY]
    # A Video run films around a gap as it always has.
    _again, filmed, _producer = _film(camera, _replayed(tmp_path, ep_a, "ep_v"),
                                      bundle_map=_map(first))
    assert filmed == []


def test_node_63_refuses_a_publish_over_cards_not_filmed_as_they_stand(
        producer_root, cameras, floors, tmp_path):
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    first, _filmed, _producer = _film(camera, _cloud_workspace(tmp_path, "ep_a"),
                                      takes={"shot_b003": 2})
    video = first["ledger"]["video"]
    files = [{"path": r["files"]["clip"]["path"]}
             for r in list(video[PC.MAP_KEY].values()) + list(video[PC.FLOOR_KEY].values())]
    cards = PD.build_cards(first["ledger"], PD.manifest_producer_fields(_direction()), files)

    def publish(**changes):
        return _direction(stage="publish", out=None, **changes)

    takes = {"clips": {"shot_b003": 2}}
    assert PD.unfilmed_shots(cards, publish(takes=takes)) == ["shot_b002 (failed: timeout)"]
    assert PD.unfilmed_shots(cards, publish(takes=takes, kept={"clips": ["shot_b002"]})) == []
    assert PD.unfilmed_shots(cards, publish(kept={"clips": ["shot_b002"]})) == [
        "shot_b003 (its clip take changed)"], "takes are sent in full"
    edited = publish(takes=takes, kept={"clips": ["shot_b002"]},
                     cards={"shot_b001": {"motion_text": "turns to the door"}})
    assert PD.unfilmed_shots(cards, edited) == ["shot_b001 (its action changed)"]
    for group in cards["scenes"]:
        for card in group["shots"]:
            if card["shot_id"] == "shot_b003":
                card.update(clip=None, clip_state="", clip_take=0)
    assert PD.unfilmed_shots(cards, publish(kept={"clips": ["shot_b002"]})) == [
        "shot_b003 (not filmed)"]
