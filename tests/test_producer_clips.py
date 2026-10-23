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
    """A stand-in video engine. ``broken`` names shots it fails on."""
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


def _film(camera, workspace, bundle_map=None):
    ep, ledger, build = workspace
    producer = PC.ProducerClips(project_id=PID, episode_root=ep, bundle_map=bundle_map)
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
