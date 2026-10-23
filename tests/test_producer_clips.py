"""Scene Producer clip reuse (chunk 4a, 2026-10-22): the causal key, the
project's clip store, and a Video run that films only what changed.

The render loop runs for real -- ``render_driver.run_episode``, serial and
cloud fan-out -- on a stand-in camera that writes one clip file per shot into
scratch and records which shots it was asked to film. Each workspace holds the
same still and audio bytes at its own paths, as a replay workspace does.
"""
from __future__ import annotations

import ast
import json
import os
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
    lane. ``peak_mb``, when set, is the VRAM peak each render reports, and
    ``broke`` names shots the provider refuses for an empty wallet."""
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
        self.broke = set()
        self.peak_mb = None

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
        if sid in self.broke:
            from nodes._otr_shared import cloud_media_backend as cmb
            raise cmb.CloudMediaError(cmb.CloudErrorCode.BUDGET,
                                      "cloud media: budget exhausted on %s" % sid)
        self.filmed.append(sid)
        path = self.scratch / ("otr_engine_tmp_%s_%d.mp4" % (sid, len(self.filmed)))
        path.write_bytes(("%s|%s" % (sid, request.get("text_prompt"))).encode())
        return {"path": str(path)}

    def canonicalize(self, raw, request, profile=None):
        clip = {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25, "fps": 25, "path": raw["path"]}
        if self.peak_mb is not None:
            clip["vram_peak_mb"] = self.peak_mb
        return clip

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


def test_a_miss_against_the_last_bundle_names_the_fields_that_moved(
        producer_root, cameras, tmp_path, caplog):
    """2026-10-23: a Film-again run re-filmed every chained shot of a project
    whose ledger had not changed, and nothing said why. A record now keeps
    the body its key hashes, and a shot the last bundle filmed that this run
    cannot reuse names the fields that moved -- the line is derived from the
    two records' bodies here, never matched against canned text (Codex QA)."""
    import logging
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    old = _map(first)["shot_b002"]["key_body"]
    assert old["shot_id"] == "shot_b002" and old["requests"][0]["text_prompt"] == PROMPTS["shot_b002"]
    with caplog.at_level(logging.WARNING, logger=PC.log.name):
        again, filmed, _producer = _film(camera, _workspace(
            tmp_path, "ep_b", prompts={"shot_b002": "a window lashed by a storm"}),
            bundle_map=_map(first))
    assert filmed == ["shot_b002"]
    new = _map(again)["shot_b002"]["key_body"]
    moved = PC.key_body_diff(old, new)
    assert len(moved) == 1 and moved[0].startswith("requests[0].text_prompt ("), moved
    said = [r.getMessage() for r in caplog.records if "NOT reused" in r.getMessage()]
    assert len(said) == 1 and "shot shot_b002 NOT reused" in said[0], said
    assert "differs in 1 place(s): " + "; ".join(moved) in said[0], said[0]


def test_a_miss_against_a_record_with_no_body_logs_this_runs_body(
        producer_root, cameras, tmp_path, caplog):
    """A record written before bodies were kept cannot be diffed; the run
    logs its own body -- the one its record then keeps -- so the run after
    it can."""
    import logging
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    bundle = {sid: {k: v for k, v in rec.items() if k != "key_body"}
              for sid, rec in _map(first).items()}
    with caplog.at_level(logging.WARNING, logger=PC.log.name):
        again, _filmed, _producer = _film(camera, _workspace(
            tmp_path, "ep_b", prompts={"shot_b002": "a window lashed by a storm"}),
            bundle_map=bundle)
    said = [r.getMessage() for r in caplog.records if "NOT reused" in r.getMessage()]
    assert len(said) == 1 and "keeps no causal body" in said[0], said
    body = _map(again)["shot_b002"]["key_body"]
    assert json.dumps(body, sort_keys=True, ensure_ascii=True, default=str) in said[0]


def test_a_bundle_clip_that_fails_to_verify_is_not_called_a_causal_change(
        producer_root, cameras, tmp_path, caplog):
    """The same key, files that do not verify, no copy in the store: the run
    films again and says that -- it never reports a causal change it did not
    see (Codex QA on daa7adce)."""
    import logging
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    shutil.rmtree(PC.store_dir(PID))
    with caplog.at_level(logging.WARNING, logger=PC.log.name):
        _again, filmed, _producer = _film(camera, _workspace(tmp_path, "ep_b"),
                                          bundle_map=_map(first))
    assert filmed == sorted(PROMPTS)
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "NOT reused" in m], messages
    assert sum("did not verify and the store holds no copy" in m for m in messages) == len(PROMPTS)


def test_a_second_find_with_no_body_clears_the_first(producer_root, tmp_path):
    """Key and body are set together: a later lookup for the same shot with
    no body leaves no earlier body for a commit to pair with the new key."""
    producer = PC.ProducerClips(project_id=PID, episode_root=tmp_path / "ep", bundle_map=None)
    assert producer._find("shot_x", "a" * 64, {"v": 2}) is None
    assert producer._bodies["shot_x"] == {"v": 2}
    assert producer._find("shot_x", "b" * 64, None) is None
    assert producer._keys["shot_x"] == "b" * 64 and producer._bodies["shot_x"] is None


def test_the_key_and_body_come_from_one_projection_and_the_body_is_bounded():
    """The key is clip_reuse_key's hash, unchanged; the body is JSON, long
    strings are kept as hash and length, and a nested non-string key is a
    documented place where the body cannot be re-hashed to the key."""
    kw = dict(shot_id="shot_x", requests=[{"text_prompt": "a", "frames": (1, 2),
                                           "plate_path": "C:/gone", "request_id": "r",
                                           "nested": {2: "two", 10: "ten"}}],
              coverage_plan={"join": "chain"}, engine={"engine_id": "stub"},
              render_policy={"device": "cuda"}, settings={"env": {"OTR_LONG": "x" * 5000}})
    key, body = PC.clip_key_and_body(**kw)
    assert key == PC.clip_reuse_key(**kw)
    assert json.loads(json.dumps(body)) == body
    assert body["requests"][0]["frames"] == [1, 2] and "plate_path" not in body["requests"][0]
    assert body["requests"][0]["nested"] == {"2": "two", "10": "ten"}
    assert body["settings"]["env"]["OTR_LONG"].startswith("sha256:") and "(5000 chars)" in body["settings"]["env"]["OTR_LONG"]
    assert len(json.dumps(body)) < 1500


def test_the_key_is_a_literal_hash_so_a_change_to_it_cannot_slip_by():
    """The key is the sha256 of the raw body, sorted, compact, ASCII. A change
    to that re-films every cached shot in every store, so it is pinned to a
    literal, never to clip_reuse_key itself (Sonnet QA on 0484aeae)."""
    kw = dict(shot_id="shot_x", requests=[{"text_prompt": "a lamp", "frames": (1, 2),
                                           "request_id": "r"}],
              coverage_plan={"join": "chain"}, engine={"engine_id": "stub"},
              render_policy={"device": "cuda"}, settings={"env": {"OTR_A": "1"}})
    assert PC.CLIP_KEY_VERSION == 2
    assert PC.clip_reuse_key(**kw) == "dbed724009153650a9a08e792518ce04d208e23a50ef7da9c7d5580145fdef57"
    assert PC.clip_key_and_body(**kw)[0] == "dbed724009153650a9a08e792518ce04d208e23a50ef7da9c7d5580145fdef57"


def test_a_string_is_kept_at_the_cap_and_hashed_one_past_it():
    kept, hashed = "x" * PC.BODY_STRING_CAP, "x" * (PC.BODY_STRING_CAP + 1)
    body = PC.clip_key_and_body(shot_id="s", requests=[{"a": kept, "b": hashed}],
                                coverage_plan=None, engine={}, render_policy={},
                                settings={})[1]
    assert body["requests"][0]["a"] == kept
    assert body["requests"][0]["b"].startswith("sha256:") and "(1001 chars)" in body["requests"][0]["b"]


def test_the_diff_limit_holds_on_every_branch_and_walks_in_tree_order():
    changed = {"k%03d" % i: i for i in range(1000)}
    assert len(PC.key_body_diff(changed, {k: v + 1 for k, v in changed.items()})) == 40
    assert len(PC.key_body_diff({"l": list(range(1000))}, {"l": list(range(1, 1001))})) == 40
    assert PC.key_body_diff({"b": 1, "z": 1}, {"a": 0, "b": 2, "c": 0}) == [
        "a (new)", "b (1 -> 2)", "c (new)", "z (gone)"]
    assert PC.key_body_diff({}, changed)[:2] == ["k000 (new)", "k001 (new)"]
    # Tree order: a child's paths come before the next sibling's (Codex QA).
    assert PC.key_body_diff({"a": [0], "a0": 0}, {"a": [1], "a0": 1}) == [
        "a[0] (0 -> 1)", "a0 (0 -> 1)"]


def test_a_deep_container_against_a_scalar_never_raises():
    import sys
    deep: dict = {}
    node = deep
    # Deeper than json.dumps can print at this process's recursion limit.
    for _ in range(sys.getrecursionlimit() * 3):
        node["d"] = {}
        node = node["d"]
    moved = PC.key_body_diff({"x": deep}, {"x": 1})
    assert len(moved) == 1 and moved[0].startswith("x (<dict too deep to print> -> 1)")


def test_a_miss_line_says_40_plus_only_past_forty(producer_root, cameras, tmp_path, caplog):
    """Exactly forty moved paths are forty; a forty-first makes it 40+. The
    count is built from the real diff of the two records' bodies, padded
    with gone keys under settings."""
    import logging
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    edit = {"shot_b002": "a window lashed by a storm"}
    again, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_b", prompts=edit),
                                      bundle_map=_map(first))
    old = _map(first)["shot_b002"]["key_body"]
    new = _map(again)["shot_b002"]["key_body"]
    base = len(PC.key_body_diff(old, new, limit=1000))
    assert 0 < base < 40
    for n in (40, 41):
        padded = {sid: rec for sid, rec in _map(first).items()}
        pad = {"OTR_PAD_%03d" % i: "1" for i in range(n - base)}
        padded["shot_b002"] = dict(padded["shot_b002"], key_body=dict(
            old, settings=dict(old["settings"], **pad)))
        caplog.clear()
        # A fresh edit each time: the store already holds the first edit's clip,
        # and a store hit is a reuse, not a miss.
        fresh = {"shot_b002": "a window lashed by a storm, take %d" % n}
        with caplog.at_level(logging.WARNING, logger=PC.log.name):
            _film(camera, _workspace(tmp_path, "ep_%d" % n, prompts=fresh), bundle_map=padded)
        said = [r.getMessage() for r in caplog.records if "shot_b002 NOT reused" in r.getMessage()]
        assert len(said) == 1, said
        expected = "differs in 40 place(s)" if n == 40 else "differs in 40+ place(s)"
        assert expected in said[0], (n, base, said[0][:160])


def test_a_failing_explanation_never_fails_the_run(producer_root, cameras, tmp_path,
                                                    monkeypatch, caplog):
    import logging
    camera = cameras()
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))

    def broken(*_args, **_kw):
        raise RuntimeError("diff exploded")

    monkeypatch.setattr(PC, "key_body_diff", broken)
    with caplog.at_level(logging.WARNING, logger=PC.log.name):
        _again, filmed, _producer = _film(camera, _workspace(
            tmp_path, "ep_b", prompts={"shot_b002": "a window lashed by a storm"}),
            bundle_map=_map(first))
    assert filmed == ["shot_b002"]
    assert any("the explanation failed (RuntimeError: diff exploded)" in r.getMessage()
               for r in caplog.records)


def test_the_body_diff_is_bounded_iterative_and_nan_safe():
    wide = {"k%d" % i: i for i in range(1000)}
    assert len(PC.key_body_diff({}, wide)) == 40
    deep_a: dict = {}
    deep_b: dict = {}
    a, b = deep_a, deep_b
    for _ in range(1200):
        a["d"] = {}
        b["d"] = {}
        a, b = a["d"], b["d"]
    b["leaf"] = 1
    assert PC.key_body_diff(deep_a, deep_b) == ["d" * 0 + ".".join(["d"] * 1200) + ".leaf (new)"]
    assert PC.key_body_diff({"x": float("nan")}, {"x": float("nan")}) == []
    assert PC.key_body_diff({"l": [1, 2]}, {"l": [1, 2, 3]}) == ["l (2 -> 3 items)"]
    assert PC.key_body_diff({"l": [1, 2]}, {"l": [1, 9]}) == ["l[1] (2 -> 9)"]


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


def test_a_bundle_record_that_names_a_path_in_its_clip_is_not_used(producer_root, cameras,
                                                                   tmp_path):
    # A path in the clip body would be handed on unverified: the mux reads a
    # foley_path without checking it against anything.
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    forged = json.loads(json.dumps(_map(first)))
    forged["shot_b001"]["clip"]["foley_path"] = str(tmp_path / "a stem nobody listed.wav")
    again, filmed, producer = _film(camera, _replayed(tmp_path, ep_a, "ep_c"),
                                    bundle_map=forged)
    assert filmed == [] and producer.reused["shot_b001"] == "store"
    assert "foley_path" not in again["clips"]["shot_b001"]


def test_a_reused_shot_reports_the_peak_it_was_filmed_at_and_says_it_was_reused(
        producer_root, cameras, tmp_path):
    camera = cameras()
    camera.peak_mb = 321
    first, _filmed, _producer = _film(camera, _workspace(tmp_path, "ep_a"))
    assert first["vram_peak_mb"] == 321
    again, filmed, _producer = _film(camera, _workspace(tmp_path, "ep_b"))
    assert filmed == [] and again["vram_peak_mb"] == 321, "the credits name what it took"
    assert [r.get("reused_from") for r in again["receipts"]] == ["store"] * len(PROMPTS)
    assert not [r for r in first["receipts"] if "reused_from" in r]
    assert all(r["actual_request_sha"] == f["actual_request_sha"]
               for r, f in zip(again["receipts"], first["receipts"]))
    assert not [r for r in _map(again).values() if "reused_from" in r["clip"]["receipt"]], \
        "the record rides on as it was filmed"


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


def test_the_key_sees_every_engine_setting_in_the_environment(monkeypatch):
    shot = {"shot_id": "shot_b001", "engine_id": "stub_camera"}
    request = {"shot_id": "shot_b001", "text_prompt": "a desk"}

    def key():
        return rd._producer_clip_key(shot, [request], {})

    same = key()
    monkeypatch.setenv("OTR_HUMO_CFG", "7.0")
    assert key() != same, "a sampler setting an engine reads at render time"
    monkeypatch.delenv("OTR_HUMO_CFG")
    assert key() == same
    for name, value in (("OTR_SNAPSHOT_HASH", "f" * 64), ("OTR_ACTIVE_PROFILE", "otr_16gb"),
                        ("OTR_OUTPUT_DIR", "D:/elsewhere"), ("OTR_ENABLE_STYLE_GRAMMAR", "1"),
                        ("OTR_GOOGLE_API_KEY", "a rotated key"),
                        ("OTR_CLOUD_VIDEO_TIMEOUT_S", "900"), ("OTR_CLOUD_FANOUT", "4"),
                        ("OTR_CLOUD_VIDEO_EST_USD", "0.5"), ("OTR_LTX25_ENCODER_CACHE", "0"),
                        # A spend cap raised after a halt must film nothing again.
                        ("OTR_CLOUD_MEDIA_BUDGET_USD", "90"),
                        ("OTR_CLOUD_MAX_CONCURRENCY_LTX25", "3"),
                        ("OTR_COMFY_MAX_RETRIES", "5"), ("OTR_TMP_SWEEP_MAX_AGE_S", "60"),
                        ("OTR_COMFY_API_BASE", "https://example.invalid"),
                        ("OTR_HF_TOKEN_FILE", "D:/token"), ("OTR_RUNTIME_SECRETS_FILE", "D:/s"),
                        ("OTR_VIDEO_BUDGET_MARGIN", "0.2"), ("OTR_VIDEO_COST_PER_FRAME_MB", "9"),
                        ("OTR_UNIFIED_MEMORY_HEADROOM_MB", "512"), ("OTR_TMP", "D:/tmp"),
                        # A health check that raises after the render changes no pixel.
                        ("OTR_LTX_OPEN_STRICT", "1"),
                        ("HF_HOME", "D:/hf")):
        monkeypatch.setenv(name, value)
        assert key() == same, name
    # A folder or root that chooses model files is a cause.
    for name in ("OTR_LTX_8GB_CKPT_DIR", "OTR_COMFYUI_MODELS_ROOT"):
        monkeypatch.setenv(name, "D:/other_weights")
        assert key() != same, name
        monkeypatch.delenv(name)


def test_every_setting_the_pack_writes_while_a_prompt_runs_stays_out_of_the_key():
    # A setting a node writes per prompt can differ between two runs of one
    # project; in the key it would film every shot again on every run. A new
    # one fails here until someone decides which it is: one that cannot change
    # a clip joins _ENV_NOT_CAUSAL, and one that can stays in the key and is
    # named in this test as the exception it is.
    pinned = set()
    for source in (_REPO / "nodes").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # env.pin / env.setdefault / env.unpin("OTR_..."), whatever the alias,
            # and environ["OTR_..."] = / .pop("OTR_...") wherever it is spelled.
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("pin", "setdefault", "unpin", "pop")
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).startswith("OTR_")):
                if node.func.attr != "pop" or "environ" in ast.unparse(node.func.value):
                    pinned.add(node.args[0].value)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and "environ" in ast.unparse(target.value)
                            and isinstance(target.slice, ast.Constant)
                            and str(target.slice.value).startswith("OTR_")):
                        pinned.add(target.slice.value)
    assert pinned, "the scan found the pins"
    assert pinned <= PC._ENV_NOT_CAUSAL, sorted(pinned - PC._ENV_NOT_CAUSAL)


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


def test_a_file_replaced_in_place_never_passes_as_the_one_it_replaced(tmp_path):
    store, key = tmp_path / "store", "be" * 32
    PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path))})
    assert PC.lookup(store, key) is not None
    stored = store / key / "clip.mp4"
    before = stored.stat()
    stored.write_bytes(b"CLIP BYTES")                  # the same length, other bytes
    os.utime(str(stored), ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(PC.ClipStoreError, match="not the file its record names"):
        PC.lookup(store, key)


def test_a_record_of_the_wrong_shape_is_passed_over_never_a_crash(producer_root, cameras,
                                                                  tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    forged = json.loads(json.dumps(_map(first)))
    forged["shot_b001"]["clip"]["receipts"] = 5                # not a list
    forged["shot_b002"]["clip"]["vram_peak_mb"] = 10 ** 400     # no float holds it
    forged["shot_b003"]["clip"]["frame_count"] = "not-a-number"
    again, filmed, producer = _film(camera, _replayed(tmp_path, ep_a, "ep_c"),
                                    bundle_map=forged)
    assert filmed == []
    assert producer.reused["shot_b001"] == "store", "the bad record is passed over"
    assert producer.reused["shot_b002"] == "bundle" and again["vram_peak_mb"] is None
    assert producer.reused["shot_b003"] == "store", "a number that is not one is passed over"
    rd.build_clip_manifest(again, episode_id="ep_c")


def test_a_stored_clip_copied_in_waits_out_a_held_file(producer_root, cameras, tmp_path,
                                                       monkeypatch):
    camera = cameras()
    _film(camera, _workspace(tmp_path, "ep_a"))
    real_replace = os.replace
    refused = []

    def held_once(src, dst):
        if str(src).endswith(".part") and not refused:
            refused.append(src)
            raise PermissionError(13, "the file is held", str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(PC.os, "replace", held_once)
    monkeypatch.setattr(PC.time, "sleep", lambda seconds: None)
    ep_b, ledger, build = workspace = _workspace(tmp_path, "ep_b")
    _again, filmed, producer = _film(camera, workspace)
    assert refused and filmed == [] and set(producer.reused.values()) == {"store"}
    assert not [p.name for p in (ep_b / "clips").iterdir() if p.name.endswith(".part")]


def test_a_copy_in_that_fails_half_way_leaves_nothing_under_the_clips_name(
        producer_root, cameras, tmp_path, monkeypatch):
    camera = cameras()
    real_replace, real_copy = os.replace, shutil.copyfile

    def held(src, dst):
        if "shot_b002" in str(src) and not str(src).endswith(".part"):
            raise PermissionError(13, "the file is held", str(src))
        return real_replace(src, dst)

    def breaks(src, dst, *args, **kwargs):
        if "shot_b002" in str(src) and str(dst).endswith(".part"):
            pathlib.Path(dst).write_bytes(b"half")
            raise OSError("the disk filled up")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(PC.os, "replace", held)
    monkeypatch.setattr(PC.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(shutil, "copyfile", breaks)
    ep, ledger, build = workspace = _workspace(tmp_path, "ep_a")
    result, filmed, _producer = _film(camera, workspace)
    assert filmed == sorted(PROMPTS) and "shot_b002" not in _map(result), "not committed"
    assert not [p.name for p in (ep / "clips").iterdir() if "shot_b002" in p.name]


def test_a_clip_brought_into_the_episode_waits_out_a_held_file(producer_root, cameras,
                                                                tmp_path, monkeypatch):
    # Codex QA, 2026-10-23: the copy-in renamed once, with no wait.
    camera = cameras()
    real_replace = os.replace
    refused = []

    def held_once(src, dst):
        if "shot_b001" in str(src) and not refused:
            refused.append(src)
            raise PermissionError(13, "the file is held", str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(PC.os, "replace", held_once)
    monkeypatch.setattr(PC.time, "sleep", lambda seconds: None)
    ep, ledger, build = workspace = _workspace(tmp_path, "ep_a")
    result, filmed, _producer = _film(camera, workspace)
    assert refused and filmed == sorted(PROMPTS) and "shot_b001" in _map(result)
    assert not [p.name for p in (ep / "clips").iterdir() if p.name.endswith(".part")]


def test_a_record_keeps_files_never_paths(tmp_path):
    store, key = tmp_path / "store", "ad" * 32
    with pytest.raises(PC.ClipStoreError, match="names a path in its clip"):
        PC.commit(store, key, dict(BODY, clip={"frame_count": 25, "path": "x.mp4"}),
                  {"clip": str(_clip_file(tmp_path))})
    PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path))})
    written = store / key / PC.RECORD_NAME
    record = json.loads(written.read_text(encoding="utf-8"))
    for clip, refusal in (({"frame_count": 25, "foley_path": str(tmp_path / "stem.wav")},
                           "names a path in its clip"),
                          ({"frame_count": 25, "foley_sha256": "0" * 64},
                           "a foley receipt for a stem it does not list")):
        written.write_text(json.dumps(dict(record, clip=clip)), encoding="utf-8")
        with pytest.raises(PC.ClipStoreError, match=refusal):
            PC.lookup(store, key)


def test_a_commit_waits_out_a_folder_held_for_a_moment(tmp_path, monkeypatch):
    store, key = tmp_path / "store", "ce" * 32
    real_replace = os.replace
    refused = []

    def held_once(src, dst):
        if ".staging-" in str(src) and not refused:
            refused.append(src)
            raise PermissionError(13, "the folder is held", str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(PC.os, "replace", held_once)
    monkeypatch.setattr(PC.time, "sleep", lambda seconds: None)
    record = PC.commit(store, key, BODY, {"clip": str(_clip_file(tmp_path))})
    assert refused and PC.lookup(store, key) == record
    assert [p.name for p in store.iterdir()] == [key], "no staging folder is left behind"


def test_the_map_describes_the_episodes_own_files(producer_root, tmp_path):
    # Another writer's entry already under the key holds other bytes; the map
    # must still describe the clip this episode holds, or its bundle would not
    # verify on its own. A record whose foley receipt names another stem is
    # never kept, with or without the store.
    producer = PC.ProducerClips(project_id=PID, episode_root=tmp_path / "ep", bundle_map={})
    key = "fa" * 32
    PC.commit(PC.store_dir(PID), key, BODY, {"clip": str(_clip_file(tmp_path, b"theirs"))})
    mine = tmp_path / "ep" / "clips" / "shot_b001_role_engine.mp4"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"mine")
    producer._keys["shot_b001"] = key
    producer.commit("shot_b001", {"path": str(mine), "frame_count": 25}, ["stub_camera"])
    record = producer.records_for(["shot_b001"])["shot_b001"]
    assert record["files"]["clip"]["sha256"] == PC.file_sha256(mine)
    assert PC.lookup(PC.store_dir(PID), key)["files"]["clip"]["sha256"] != \
        record["files"]["clip"]["sha256"], "the store keeps its own entry"
    stem = tmp_path / "ep" / "audio" / "foley" / "shot_b002.wav"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"RIFF a stem")
    producer._keys["shot_b002"] = "fb" * 32
    producer.commit("shot_b002", {"path": str(mine), "foley_path": str(stem),
                                  "foley_sha256": "0" * 64}, [])
    assert producer.records_for(["shot_b002"]) == {}


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


@pytest.mark.parametrize("producer", [True, False])
def test_a_shot_that_floors_again_carries_only_its_new_floors_marks(
        producer_root, cameras, floors, tmp_path, monkeypatch, producer):
    # Its replay carries the marks of the floor it fell to last time. Kept
    # beside the new ones, a timeout read as a budget floor, and a refused
    # still as the timeout it no longer is.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    monkeypatch.setattr(rd, "sanctioned_gap_beat_ids", lambda ledger: {"b003"})
    ep, ledger, build = _cloud_workspace(tmp_path, "ep_%s" % producer)
    ledger["video"]["shots"][1].update(status="sanctioned_gap", budget_floor=True,
                                       floor_render="still_pan")
    ledger["video"]["shots"][2].update(status="sanctioned_gap", cloud_floor="timeout",
                                       floor_render="still_pan")
    if producer:
        result, _filmed, _p = _film(camera, (ep, ledger, build))
        payloads = result["ledger"]["video"][PC.FLOOR_KEY]
        assert payloads["shot_b002"]["marks"] == {
            "status": "sanctioned_gap", "cloud_floor": "timeout", "floor_render": "still_pan"}
        assert payloads["shot_b003"]["marks"] == {"status": "sanctioned_gap"}
    else:
        result = rd.run_episode(ledger, request_builder=build)
    shots = result["ledger"]["video"]["shots"]
    assert "budget_floor" not in shots[1] and shots[1]["cloud_floor"] == "timeout"
    assert not set(rd._FLOOR_MARKS) & set(shots[2]), "a gap row as its first run made it"


@pytest.mark.parametrize("fanout", ["1", "2"])
def test_a_store_refusal_is_never_read_as_an_empty_wallet(
        producer_root, cameras, floors, tmp_path, monkeypatch, fanout):
    # A provider's 402 means an empty wallet; a store refusal that happens to
    # hold the number in its message must still refuse the run.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", fanout)
    camera = cameras("cloud_stub_camera")
    real_find = PC.ProducerClips.find

    def refusing(self, shot_id, key, body=None):
        if shot_id == "shot_b002":
            raise PC.ClipStoreError("Publish has no filmed clip for shot 402")
        return real_find(self, shot_id, key, body)

    monkeypatch.setattr(PC.ProducerClips, "find", refusing)
    with pytest.raises(PC.ClipStoreError, match="no filmed clip for shot 402"):
        _film(camera, _cloud_workspace(tmp_path, "ep_a"))
    assert floors == [], "no floor was drawn for it"


def test_a_keep_that_does_not_verify_refuses_before_any_shot_is_filmed(
        producer_root, cameras, tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    workspace = _replayed(tmp_path, ep_a, "ep_b", prompts={"shot_b001": "an edited desk"})
    name = _map(first)["shot_b003"]["files"]["clip"]["name"]
    (workspace[0] / "clips" / name).write_bytes(b"not the clip that was kept")
    with pytest.raises(PC.ClipStoreError, match="card shot_b003 keeps its clip, but"):
        _film(camera, workspace, bundle_map=_map(first), kept=["shot_b003"])
    assert camera.filmed == [], "nothing is filmed, or paid for, before the keeps verify"


def test_a_kept_shot_never_reaches_its_request_builder(producer_root, cameras, tmp_path):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    ep_b, ledger_b, build_b = _replayed(tmp_path, ep_a, "ep_b")

    def builder(shot, ledger, canvas=None, segment_index=0):
        if shot["shot_id"] == "shot_b002":
            raise RuntimeError("the builder could not make a request for this shot")
        return build_b(shot, ledger, canvas=canvas, segment_index=segment_index)

    again, filmed, producer = _film(camera, (ep_b, ledger_b, builder),
                                    bundle_map=_map(first), kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept"
    assert pathlib.Path(again["clips"]["shot_b002"]["path"]).parent == ep_b / "clips"


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


@pytest.mark.parametrize("fanout", ["1", "2"])
def test_a_spend_cap_halt_still_delivers_the_shots_filmed_before(
        producer_root, cameras, floors, tmp_path, monkeypatch, fanout):
    # Reusing a clip costs nothing, so an empty wallet floors only the shots
    # that would have to be filmed.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", fanout)
    camera = cameras("cloud_stub_camera")
    first, _filmed, _producer = _film(camera, _cloud_workspace(tmp_path, "ep_a"))
    camera.broke = {"shot_b001", "shot_b002"}
    again, filmed, producer = _film(camera, _cloud_workspace(
        tmp_path, "ep_b", prompts={"shot_b001": "an edited desk",
                                   "shot_b002": "an edited window"}))
    assert filmed == [] and producer.reused == {"shot_b003": "store"}
    shots = {s["shot_id"]: s for s in again["ledger"]["video"]["shots"]}
    assert shots["shot_b001"].get("budget_floor") and shots["shot_b002"].get("budget_floor")
    assert not shots["shot_b003"].get("budget_floor")
    assert list(_map(again)) == ["shot_b003"]
    assert _map(again)["shot_b003"] == _map(first)["shot_b003"]


@pytest.mark.parametrize("fanout", ["1", "2"])
def test_a_spend_cap_halt_still_delivers_a_kept_clip(producer_root, cameras, floors,
                                                    tmp_path, monkeypatch, fanout):
    # The kept shot comes third, past both fan-out workers, so the halt
    # reaches it before any worker does.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", fanout)
    camera = cameras("cloud_stub_camera")
    ep_a, ledger_a, build_a = _cloud_workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger_a, build_a))
    camera.broke = {"shot_b001", "shot_b002"}
    edited = {"shot_b001": "an edited desk", "shot_b002": "an edited window",
              "shot_b003": "an edited corridor"}
    workspace = _replayed(tmp_path, ep_a, "ep_b", engine_id="cloud_stub_camera",
                          prompts=edited)
    again, filmed, producer = _film(camera, workspace, bundle_map=_map(first),
                                    kept=["shot_b003"])
    assert filmed == [] and producer.reused == {"shot_b003": "kept"}
    assert list(_map(again)) == ["shot_b003"]
    assert set(again["ledger"]["video"][PC.FLOOR_KEY]) == {"shot_b001", "shot_b002"}


@pytest.mark.parametrize("fanout", ["1", "2"])
def test_a_shot_that_cannot_be_looked_up_after_a_halt_floors(
        producer_root, cameras, floors, tmp_path, monkeypatch, fanout):
    # A request builder that fails for a shot met after the halt floors it,
    # as the halt would; it never stops the episode.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", fanout)
    camera = cameras("cloud_stub_camera")
    _film(camera, _cloud_workspace(tmp_path, "ep_a"))
    camera.broke = {"shot_b001", "shot_b002"}
    ep, ledger, build = _cloud_workspace(tmp_path, "ep_b", prompts={
        "shot_b001": "an edited desk", "shot_b002": "an edited window"})

    def builder(shot, ledger, canvas=None, segment_index=0):
        if shot["shot_id"] == "shot_b003" and len(camera.asked_after_halt) > 0:
            raise RuntimeError("the builder could not make a request for this shot")
        return build(shot, ledger, canvas=canvas, segment_index=segment_index)

    camera.asked_after_halt = []
    real_render = camera.render_clip

    def render(request, prepared):
        camera.asked_after_halt.append(request["shot_id"])
        return real_render(request, prepared)

    camera.render_clip = render
    again, filmed, producer = _film(camera, (ep, ledger, builder))
    shots = {s["shot_id"]: s for s in again["ledger"]["video"]["shots"]}
    assert shots["shot_b003"].get("budget_floor"), "floored as the halt would"
    assert producer.reused == {}


def test_a_kept_clip_on_a_refused_still_is_a_delivered_beat(producer_root, cameras, tmp_path,
                                                           monkeypatch):
    camera = cameras()
    ep_a, ledger, build = _workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger, build))
    monkeypatch.setattr(rd, "sanctioned_gap_beat_ids", lambda ledger: {"b002"})
    ep, ledger_b, build_b = _replayed(tmp_path, ep_a, "ep_b")
    again, filmed, producer = _film(camera, (ep, ledger_b, build_b), bundle_map=_map(first),
                                    kept=["shot_b002"])
    assert producer.reused["shot_b002"] == "kept"
    rows = {r["shot_id"]: r for r in rd.build_clip_manifest(again, episode_id="ep_b")["clips"]}
    assert (rows["shot_b002"]["status"], rows["shot_b002"]["exists"]) == ("ok", True)


def test_a_clip_or_floor_the_episode_cannot_take_by_a_move_is_copied_in_under_its_name(
        producer_root, cameras, floors, tmp_path, monkeypatch):
    # A scanner holding a file it has just seen appear refuses the move. The
    # map and the floor payload must still name files the episode holds, and
    # the store keep the filmed clip.
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    real_move = shutil.move

    def held(src, dst, *args, **kwargs):
        if "shot_b002" in str(src) or "shot_b003" in str(src):
            raise PermissionError(13, "the file is held", str(src))
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", held)
    ep, ledger, build = workspace = _cloud_workspace(tmp_path, "ep_a")
    result, filmed, _producer = _film(camera, workspace)
    record = _map(result)["shot_b003"]
    floor = result["ledger"]["video"][PC.FLOOR_KEY]["shot_b002"]
    assert record["files"]["clip"]["path"] == \
        "clips/shot_b003_retired_role_b_cloud_stub_camera.mp4"
    assert floor["files"]["clip"]["path"] == "clips/shot_b002_retired_role_b_still_pan.mp4"
    for sid, item in (("shot_b003", record["files"]["clip"]),
                      ("shot_b002", floor["files"]["clip"])):
        assert PC.file_sha256(ep / item["path"]) == item["sha256"]
        assert result["clips"][sid]["path"] == str(ep / item["path"])
    assert PC.lookup(PC.store_dir(PID), record["key"]) is not None


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
    assert [row["shot_id"] for row in again["trace"]] == ["shot_b001", "shot_b003"], \
        "nothing rendered a kept floor, as nothing renders a fresh one"


def test_a_floor_payload_cannot_rewrite_its_shot(producer_root, cameras, floors, tmp_path):
    camera = cameras("cloud_stub_camera")
    camera.timeouts = {"shot_b002"}
    ep_a, ledger_a, build_a = _cloud_workspace(tmp_path, "ep_a")
    first, _filmed, _producer = _film(camera, (ep_a, ledger_a, build_a))
    floored = json.loads(json.dumps(first["ledger"]["video"][PC.FLOOR_KEY]))
    floored["shot_b002"]["marks"].update(shot_id="shot_b001", target_frame_count=1)
    camera.timeouts.clear()
    again, _filmed, _producer = _film(camera, _replayed(tmp_path, ep_a, "ep_b",
                                                        engine_id="cloud_stub_camera"),
                                      bundle_map=_map(first), bundle_floors=floored,
                                      kept=["shot_b002"])
    shots = again["ledger"]["video"]["shots"]
    assert [s["shot_id"] for s in shots] == list(PROMPTS)
    assert shots[1]["target_frame_count"] == 25 and shots[1]["cloud_floor"] == "timeout"


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
    unfiled = {c["shot_id"]: c for g in PD.build_cards(
        first["ledger"], PD.manifest_producer_fields(_direction()),
        [f for f in files if "shot_b001" not in f["path"]])["scenes"] for c in g["shots"]}
    assert (unfiled["shot_b001"]["clip"], unfiled["shot_b001"]["clip_state"]) == (None, ""), \
        "filmed only with its clip in the bundle"
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
    # Kept, the card publishes the clip it shows, filmed before its still was
    # refused.
    published, filmed, producer = _film(camera, _replayed(tmp_path, ep_a, "ep_q"),
                                        bundle_map=_map(first), stage="publish",
                                        kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept"
    assert "shot_b002" in _map(published)
    assert not published["ledger"]["video"][PC.FLOOR_KEY]
    # A Video run films around a gap as it always has, and a card that was a
    # gap when last filmed publishes, kept, as the gap it is.
    gapped, filmed, _producer = _film(camera, _replayed(tmp_path, ep_a, "ep_v"),
                                      bundle_map=_map(first))
    assert filmed == [] and "shot_b002" in gapped["ledger"]["video"][PC.FLOOR_KEY]
    published, filmed, producer = _film(
        camera, _replayed(tmp_path, ep_a, "ep_r"), bundle_map=_map(gapped),
        bundle_floors=gapped["ledger"]["video"][PC.FLOOR_KEY], stage="publish",
        kept=["shot_b002"])
    assert filmed == [] and producer.reused["shot_b002"] == "kept floor"
    assert "shot_b002" in published["ledger"]["video"][PC.FLOOR_KEY]


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
    # A still the image model refused: the render would skip it as a gap.
    for group in cards["scenes"]:
        for card in group["shots"]:
            if card["shot_id"] == "shot_b001":
                card.update(still=None, still_state="failed", picture_editable=True)
    assert PD.unfilmed_shots(cards, publish(kept={"clips": ["shot_b002"]}))[0] == \
        "shot_b001 (its still was refused)"
    assert "shot_b001 (its still was refused)" not in PD.unfilmed_shots(
        cards, publish(kept={"clips": ["shot_b001", "shot_b002"]}))
