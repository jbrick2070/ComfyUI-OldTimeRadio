"""The Scene Producer in Studio: projects, drafts, a stage run and its bundle,
with the SDK and ComfyUI's output folder faked. The stand-in client freezes a
bundle the way the pack does, from the direction it was sent, so a run is
followed from the page's draft to the bundle it adopts."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading

import pytest

from radio_play_studio import producer
from radio_play_studio import runner as rps_runner
from radio_play_studio.form import EPISODE_NODE_TYPE, SHOT_LOCK
from radio_play_studio.runner import Studio
from radio_play_studio.server import make_server

TEXTS = {"shot_b001": "a lamp-lit radio desk", "shot_b002": "a rain-streaked window",
         "shot_b003": "a corridor of humming valves"}


# --------------------------------------------------------------------------- #
# stand-ins
# --------------------------------------------------------------------------- #

class Progress:
    """Named as the SDK's event: the app matches on the class name."""

    def __init__(self, node, step, steps):
        self.value, self.message = step / steps, ""
        self.nodes_total, self.nodes_done = 0, 0
        self.current_node, self.step, self.steps = node, step, steps


class JobFailed(Exception):
    """Named as the SDK's: the runner matches it by name."""


class ReadTimeout(Exception):
    """Named as httpx's: a submit that may have reached the server."""


class NotFound(Exception):
    """Named as the SDK's: the bridge has no record of a job."""


class Refused(Exception):
    """A submit the server answered with a refusal, as the SDK reports one."""
    http_status = 400


class FakeOutput:
    def __init__(self, node_id, name, payload=b"episode-bytes"):
        self.node_id, self.name, self._payload = node_id, name, payload

    def to_file(self, path):
        with open(path, "wb") as handle:
            handle.write(self._payload)


class StageJob:
    def __init__(self, job_id, events, fail=None, outputs=()):
        self.id, self._events, self._fail, self.outputs = job_id, events, fail, list(outputs)

    def events(self):
        yield from self._events

    def wait(self, timeout=None):
        return self

    def result(self):
        if self._fail:
            raise self._fail
        return self

    def get_outputs(self, node_id):
        return [o for o in self.outputs if o.node_id == node_id]


class StageClient:
    """Takes a stage's graph and, as the pack's freeze would, writes the bundle
    its direction names into ``output`` -- ``{(subfolder, name): bytes}``,
    ComfyUI's output folder as ``/view`` serves it."""

    def __init__(self, output):
        self.output = output
        self.workflows = self
        self.directions = []
        self.texts = dict(TEXTS)
        self.filmed = {}            # shot id -> the picture text its clip shows
        self.refuse = None          # an exception submit raises
        self.fail = None            # the job ends with this
        self.freezes = True         # whether the run leaves its bundle
        self.on_submit = None       # called with the graph before anything else
        self.events = [Progress("41", 1, 4), Progress("41", 4, 4)]
        self.jobs = self            # client.jobs.get(id)
        self.made = {}              # job id -> the job, as the bridge recalls it

    def from_json(self, graph):
        return graph

    def submit(self, graph):
        if self.on_submit:
            self.on_submit(graph)
        if self.refuse:
            raise self.refuse
        lock = next(n for n in graph.values() if n.get("class_type") == SHOT_LOCK)
        direction = json.loads(lock["inputs"]["direction_json"])
        self.directions.append(direction)
        for sid, edit in direction.get("cards", {}).items():
            self.texts[sid] = edit.get("picture_text", self.texts[sid])
        if direction["stage"] == "video":
            self.filmed = dict(self.texts)
        if direction["stage"] != "publish" and self.freezes:
            freeze(self.output, direction, self.texts, self.filmed)
        outputs = ([FakeOutput("9", "otr_episode.mp4")]
                   if direction["stage"] == "publish" else [])
        job = StageJob("job-%d" % len(self.directions), self.events, self.fail, outputs)
        self.made[job.id] = job
        return job

    def get(self, job_id):
        if job_id not in self.made:
            raise NotFound(job_id)
        return self.made[job_id]


def freeze(output, direction, texts, filmed=None):
    """What the pack's freeze leaves for one stage: each still, each clip
    filmed so far (``filmed``: shot id -> the picture its clip shows; a Stills
    bundle carries them over), cards.json, and the manifest listing them."""
    folder = f"{producer.REPLAY_ROOT}/{direction['project_id']}/{direction['out']}"
    files = []

    def put(rel, data):
        sub, _, name = rel.rpartition("/")
        output[("/".join(part for part in (folder, sub) if part), name)] = data
        files.append({"path": rel, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})

    cards = []
    for sid, text in texts.items():
        still = "stills/still_%s.png" % sid[len("shot_"):]
        put(still, b"PNG of " + text.encode())
        clip = None
        if sid in (filmed or {}):
            clip = "clips/%s.mp4" % sid
            put(clip, b"MP4 of " + filmed[sid].encode())
        cards.append({"shot_id": sid, "picture_text": text, "motion_text": "turns to the door",
                      "picture_editable": True, "motion_editable": sid != "shot_b003",
                      "still": still, "clip": clip, "state": "painted",
                      "clip_state": "filmed" if clip else ""})
    view = {"version": 1, "project_id": direction["project_id"],
            "attempt_id": direction["attempt_id"], "stage": direction["stage"],
            "bundle": direction["out"], "source": direction.get("source"),
            "cast": [{"char_id": "c01", "name": "Ann", "picture": cards[0]["still"],
                      "take": 0}],
            "scenes": [{"scene_id": "scene_1", "note": "", "shots": cards}]}
    put("cards.json", json.dumps(view).encode())
    record = {"version": 1, "project_id": direction["project_id"],
              "attempt_id": direction["attempt_id"], "stage": direction["stage"],
              "out": direction["out"], "source": direction.get("source")}
    output[(folder, "manifest.json")] = json.dumps({"schema_version": 1, "files": files,
                                                   "producer": record}).encode()


def reader(output):
    def read(subfolder, name):
        try:
            return output[(subfolder, name)]
        except KeyError:
            raise producer.BundleMissing(f"no {subfolder}/{name}") from None
    return read


@pytest.fixture
def stage_schemas(schemas):
    schemas[SHOT_LOCK] = {"input": {"required": {}, "optional": {
        "direction_json": ("STRING", {"default": "", "multiline": True})}}}
    return schemas


@pytest.fixture
def stage_settings(settings, workflow):
    workflow["nodes"].append({"id": 90, "type": SHOT_LOCK, "inputs": [], "widgets_values": [""]})
    settings.workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    return settings


@pytest.fixture
def studio(stage_settings, stage_schemas):
    output = {}
    client = StageClient(output)
    comfy = {"/queue": {"queue_running": [], "queue_pending": []}, "/history": {}}
    slept = []

    def comfy_json(path):
        if comfy.get("down"):
            raise producer.ServerUnreachable("ComfyUI is not answering")
        if path.startswith("/history/"):
            job_id = path[len("/history/"):]
            return {k: v for k, v in comfy["/history"].items() if k == job_id}
        return comfy[path.split("?")[0]]

    made = Studio(stage_settings, client_factory=lambda url: client,
                  fetch=lambda url: stage_schemas, read_output=reader(output),
                  comfy_json=comfy_json, sleep=slept.append)
    made.client, made.output, made.comfy, made.slept = client, output, comfy, slept
    return made


def _project(studio, choices=None):
    return producer.new_project(studio.settings.out_dir,
                                workflow=str(studio.settings.workflow_path),
                                choices=choices or {"bank": "my_story"})


def _run(studio, project_id, stage):
    return list(studio.stream_stage(studio.prepare_stage(project_id, stage)))


def _load(studio, project_id):
    return producer.load_project(studio.settings.out_dir, project_id)


# --------------------------------------------------------------------------- #
# the project file and the draft
# --------------------------------------------------------------------------- #

def test_a_project_is_saved_whole_and_listed(tmp_path):
    project = producer.new_project(tmp_path, workflow="otr_app.json", choices={"bank": "my_story"})
    assert producer.load_project(tmp_path, project["project_id"]) == project
    assert producer.list_projects(tmp_path) == [{
        "project_id": project["project_id"], "created": project["created"], "bundles": [],
        "busy": False, "published": False}]
    with pytest.raises(producer.NoProject):
        producer.load_project(tmp_path, "0000aaaa0000")
    with pytest.raises(producer.NoProject):
        producer.load_project(tmp_path, "../escape")


def test_a_draft_saves_over_the_revision_it_read_and_refuses_a_stale_one(tmp_path):
    pid = producer.new_project(tmp_path, workflow="w", choices={})["project_id"]
    draft = {"cards": {"shot_b001": {"picture_text": "a brass desk lamp"}},
             "takes": {"clips": {"shot_b002": 2}}, "kept": {"clips": ["shot_b003", "shot_b003"]}}
    saved = producer.save_draft(tmp_path, pid, draft, 0)
    assert saved["draft"]["revision"] == 1
    assert saved["draft"]["kept"] == {"stills": [], "clips": ["shot_b003"]}
    with pytest.raises(producer.StaleDraft):
        producer.save_draft(tmp_path, pid, draft, 0)
    assert producer.load_project(tmp_path, pid)["draft"]["revision"] == 1


@pytest.mark.parametrize("draft", [
    {"cards": {"../shot": {"picture_text": "x"}}},
    {"cards": {"shot_b001": {"picture_text": "x" * (producer.MAX_CARD_TEXT_CHARS + 1)}}},
    {"cards": {"shot_b001": {"sound": "a door"}}},
    {"cards": {"shot_b001": {"picture_text": 7}}},
    {"scenes": {"scene_1": {"note": "x" * (producer.MAX_NOTE_CHARS + 1)}}},
    {"takes": {"clips": {"shot_b001": producer.MAX_TAKE + 1}}},
    {"takes": {"clips": {"shot_b001": True}}},
    {"takes": {"clips": {"shot_b001": -1}}},
    {"kept": {"clips": "shot_b001"}},
    {"kept": {"clips": ["shot b001"]}},
    {"takes": {"characters": {"con": 0}}},
    {"kept": {"stills": ["LPT1"]}},
    ["not", "an", "object"],
])
def test_a_draft_the_pack_would_refuse_never_reaches_the_disk(tmp_path, draft):
    pid = producer.new_project(tmp_path, workflow="w", choices={})["project_id"]
    with pytest.raises(producer.ProjectError):
        producer.save_draft(tmp_path, pid, draft, 0)
    assert producer.load_project(tmp_path, pid)["draft"]["revision"] == 0


# --------------------------------------------------------------------------- #
# directions and the graph
# --------------------------------------------------------------------------- #

def test_a_projects_first_run_paints_its_stills_and_carries_no_edits(tmp_path):
    project = producer.new_project(tmp_path, workflow="w", choices={})
    direction = producer.build_direction(project, "stills")
    assert (direction["stage"], direction["out"]) == ("stills", "01-stills")
    assert set(direction) == {"version", "project_id", "attempt_id", "stage", "out"}
    for stage in ("video", "publish"):
        with pytest.raises(producer.ProjectError, match="starts with its stills"):
            producer.build_direction(project, stage)


def _shown(stage="stills"):
    """A project as it stands after one bundle, read as a page reads it."""
    cards = [{"shot_id": sid, "picture_text": text, "motion_text": "turns to the door",
              "picture_editable": True, "motion_editable": sid != "shot_b003"}
             for sid, text in TEXTS.items()]
    project = {"project_id": "a1b2c3d4e5f6", "last_number": 1, "draft": producer.empty_draft(),
               "bundles": [{"out": "01-%s" % stage, "stage": stage, "digest": "d" * 64}],
               "view": {"scenes": [{"scene_id": "scene_1", "note": "rain", "shots": cards}]}}
    return project


def test_a_replay_carries_only_what_differs_from_the_bundle_shown():
    project = _shown()
    project["draft"].update(
        cards={"shot_b001": {"picture_text": TEXTS["shot_b001"], "motion_text": "sits"},
               "shot_b002": {"picture_text": "a window in a storm"},
               "shot_b003": {"motion_text": "an action the card cannot take"},
               "shot_b404": {"picture_text": "a card the bundle never had"}},
        scenes={"scene_1": {"note": "rain"}, "scene_9": {"note": "nowhere"}},
        takes={"characters": {"c01": 1}, "stills": {}, "clips": {"shot_b002": 3}},
        kept={"stills": [], "clips": ["shot_b001"]})
    with pytest.raises(producer.ProjectError, match="paint the changed pictures first: shot_b002"):
        producer.build_direction(project, "video")
    direction = producer.build_direction(project, "stills")
    assert (direction["out"], direction["source"]) == ("02-stills", "d" * 64)
    assert direction["cards"] == {"shot_b001": {"motion_text": "sits"},
                                  "shot_b002": {"picture_text": "a window in a storm"}}
    assert direction["scenes"] == {}
    assert direction["takes"]["clips"] == {"shot_b002": 3}, "takes and keeps go in full"
    assert direction["kept"]["clips"] == ["shot_b001"]
    project["draft"]["scenes"] = {"scene_1": {}}
    assert producer.build_direction(project, "stills")["scenes"] == {}, "no note, no clear"
    project["draft"]["scenes"] = {"scene_1": {"note": ""}}
    assert producer.build_direction(project, "stills")["scenes"] == {"scene_1": {"note": ""}}
    project["draft"]["cards"] = {"shot_b002": {"picture_text": "  "}}
    with pytest.raises(producer.ProjectError, match="shot_b002 has no picture text"):
        producer.build_direction(project, "stills")


def test_a_new_portrait_waits_for_a_stills_run_whether_or_not_a_still_names_it():
    # Codex QA, 2026-10-23: no card's still names c1, so only the portrait
    # shows the take; node 63 refuses Video until a Stills run paints it.
    project = _shown()
    project["view"]["cast"] = [{"char_id": "c1", "has_portrait": True, "take": 0},
                               {"char_id": "c2", "has_portrait": False, "take": 0}]
    project["draft"]["takes"]["characters"] = {"c1": 1, "c2": 1}
    assert producer.recast(project) == ["c1"], "a cast member with no portrait is the cards' to say"
    with pytest.raises(producer.ProjectError, match="paint the new portraits first: c1"):
        producer.build_direction(project, "video")
    assert producer.build_direction(project, "stills")["takes"]["characters"] == {"c1": 1, "c2": 1}
    project["view"]["cast"][0]["take"] = 1          # the Stills run painted it
    assert producer.recast(project) == []


def test_publish_follows_a_video_run_and_bundle_numbers_never_repeat():
    with pytest.raises(producer.ProjectError, match="film them first"):
        producer.build_direction(_shown("stills"), "publish")
    direction = producer.build_direction(_shown("video"), "publish")
    assert "out" not in direction and direction["source"] == "d" * 64
    project = _shown()
    project["last_number"] = 4          # attempts 02 to 04 never made a bundle
    assert producer.build_direction(project, "video")["out"] == "05-video"


def test_the_graph_takes_the_direction_and_a_replay_empties_the_story(studio):
    project = _project(studio, {"bank": "my_story", "my_characters": "Ann (she)"})
    graph, ids = studio.build_graph_from(studio.settings.workflow_path, project["choices"])
    direction = producer.build_direction(project, "stills")
    producer.patch_graph(graph, ids, direction)
    assert json.loads(graph[ids[SHOT_LOCK]]["inputs"]["direction_json"]) == direction
    writer = graph[ids["OTR_LedgerScriptWriter"]]["inputs"]
    assert writer["story_characters"] == "Ann (she)" and "replay_from" not in writer
    producer.patch_graph(graph, ids, direction, "otr/episodes/_replay/p/x/01-stills")
    assert writer["replay_from"] == "otr/episodes/_replay/p/x/01-stills"
    assert (writer["source_bank"], writer["source_ref"]) == (producer.ROLL_BANK, "")
    assert writer["story_characters"] == writer["story_plot"] == ""
    writer["source_ref"] = ["12", 0]
    with pytest.raises(producer.ProjectError, match="source_ref is wired"):
        producer.patch_graph(graph, ids, direction, "otr/episodes/_replay/p/x/01-stills")


# --------------------------------------------------------------------------- #
# a bundle
# --------------------------------------------------------------------------- #

def _frozen(tmp_path, **direction):
    output = {}
    project = producer.new_project(tmp_path, workflow="w", choices={})
    direction = dict({"project_id": project["project_id"], "attempt_id": "a" * 32,
                      "stage": "stills", "out": "01-stills", "source": None}, **direction)
    freeze(output, direction, TEXTS)
    return project, output, direction


def test_a_bundle_is_adopted_only_as_it_was_frozen(tmp_path):
    project, output, direction = _frozen(tmp_path)
    view = producer.adopt_bundle(tmp_path, project, "01-stills", reader(output),
                                 attempt_id="a" * 32, source=None, stage="stills")
    manifest = output[(f"{producer.REPLAY_ROOT}/{project['project_id']}/01-stills",
                       "manifest.json")]
    assert view["digest"] == hashlib.sha256(manifest).hexdigest()
    card = view["scenes"][0]["shots"][0]
    sha = hashlib.sha256(b"PNG of " + TEXTS["shot_b001"].encode()).hexdigest()
    assert card["still"] == f"/episodes/projects/{project['project_id']}/media/{sha}.png"
    assert view["cast"][0]["picture"] == card["still"]
    kept = tmp_path / "projects" / project["project_id"] / "media" / f"{sha}.png"
    assert kept.read_bytes() == b"PNG of " + TEXTS["shot_b001"].encode()
    for attempt, stage in (("b" * 32, "stills"), ("a" * 32, "video")):
        with pytest.raises(producer.ProjectError, match="not the one this run made"):
            producer.adopt_bundle(tmp_path, project, "01-stills", reader(output),
                                  attempt_id=attempt, source=None, stage=stage)
    # A copy kept here is checked as well: one altered on disk is fetched again.
    kept.write_bytes(b"not the picture")
    producer.adopt_bundle(tmp_path, project, "01-stills", reader(output),
                          attempt_id="a" * 32, source=None, stage="stills")
    assert kept.read_bytes() == b"PNG of " + TEXTS["shot_b001"].encode()
    # A picture copied in once is kept by its hash and never fetched again;
    # one not yet copied is checked against the manifest as it is fetched.
    other, output, _direction = _frozen(tmp_path / "other")
    folder = f"{producer.REPLAY_ROOT}/{other['project_id']}/01-stills/stills"
    output[(folder, "still_b002.png")] = b"a different picture"
    with pytest.raises(producer.ProjectError, match="not the file its manifest lists"):
        producer.adopt_bundle(tmp_path / "other", other, "01-stills", reader(output),
                              attempt_id="a" * 32, source=None, stage="stills")
    # The manifest's own producer record must name the run too.
    folder = f"{producer.REPLAY_ROOT}/{other['project_id']}/01-stills"
    manifest = json.loads(output[(folder, "manifest.json")])
    manifest["producer"]["attempt_id"] = "c" * 32
    output[(folder, "manifest.json")] = json.dumps(manifest).encode()
    with pytest.raises(producer.ProjectError, match="not the one this run made"):
        producer.adopt_bundle(tmp_path / "other", other, "01-stills", reader(output),
                              attempt_id="a" * 32, source=None, stage="stills")


def test_a_bundle_cannot_name_media_outside_itself_or_unlisted(tmp_path):
    project, output, direction = _frozen(tmp_path)
    folder = f"{producer.REPLAY_ROOT}/{project['project_id']}/01-stills"
    for still, refusal in (("../02-stills/stills/still_b001.png", "outside itself"),
                           ("stills/never_frozen.png", "does not list"),
                           ("audio/master.wav", "not a picture or a clip")):
        manifest = json.loads(output[(folder, "manifest.json")])
        cards = json.loads(output[(folder, "cards.json")])
        cards["scenes"][0]["shots"][0]["still"] = still
        body = json.dumps(cards).encode()
        output[(folder, "cards.json")] = body
        for row in manifest["files"]:
            if row["path"] == "cards.json":
                row["sha256"] = hashlib.sha256(body).hexdigest()
        if still.startswith("../"):
            manifest["files"].append({"path": still, "sha256": "e" * 64})
        output[(folder, "manifest.json")] = json.dumps(manifest).encode()
        with pytest.raises(producer.ProjectError, match=refusal):
            producer.adopt_bundle(tmp_path, project, "01-stills", reader(output),
                                  attempt_id="a" * 32, source=None, stage="stills")


def test_a_path_outside_the_bundle_is_refused_even_when_its_file_is_already_here(tmp_path):
    # Codex QA, 2026-10-23: a file already copied in by its hash was never
    # fetched, so its path was never checked; and a backslash is one part to
    # a POSIX path, so "..\x" slipped past the ".." test.
    project, clean, _direction = _frozen(tmp_path)
    folder = f"{producer.REPLAY_ROOT}/{project['project_id']}/01-stills"
    producer.adopt_bundle(tmp_path, project, "01-stills", reader(clean),
                          attempt_id="a" * 32, source=None, stage="stills")
    for name in ("../02-stills/stills/still_b001.png", r"..\02-stills\still_b001.png",
                 "C:/stills/still_b001.png", "stills//still_b001.png", "./stills/still_b001.png"):
        output = dict(clean)
        manifest = json.loads(output[(folder, "manifest.json")])
        cards = json.loads(output[(folder, "cards.json")])
        real = cards["scenes"][0]["shots"][0]["still"]
        digest = next(r["sha256"] for r in manifest["files"] if r["path"] == real)
        cards["scenes"][0]["shots"][0]["still"] = name
        body = json.dumps(cards).encode()
        output[(folder, "cards.json")] = body
        for row in manifest["files"]:
            if row["path"] == "cards.json":
                row["sha256"] = hashlib.sha256(body).hexdigest()
        manifest["files"].append({"path": name, "sha256": digest})
        output[(folder, "manifest.json")] = json.dumps(manifest).encode()
        with pytest.raises(producer.ProjectError, match="outside itself"):
            producer.adopt_bundle(tmp_path, project, "01-stills", reader(output),
                                  attempt_id="a" * 32, source=None, stage="stills")


# --------------------------------------------------------------------------- #
# a stage run
# --------------------------------------------------------------------------- #

def test_a_stills_run_is_saved_before_it_is_sent_and_adopts_its_bundle(studio):
    project = _project(studio)
    pid = project["project_id"]
    seen = {}

    def before_sending(graph):
        pending = _load(studio, pid)["pending"]
        seen["pending"] = dict(pending)
        attempt = (studio.settings.out_dir / "projects" / pid / "attempts"
                   / f"{pending['attempt_id']}.json")
        seen["graph"] = json.loads(attempt.read_text(encoding="utf-8"))["graph"]
        assert seen["graph"] == graph, "the exact graph it sends"

    studio.client.on_submit = before_sending
    events = _run(studio, pid, "stills")
    assert seen["pending"]["state"] == "submitting" and seen["pending"]["job_id"] is None
    assert [e["type"] for e in events] == ["submitted", "step", "step", "done"]
    assert [e.get("node") for e in events[1:3]] == ["41", "41"], "the stage bar's node"
    assert events[-1] == {"type": "done", "stage": "stills", "project_id": pid,
                          "bundle": "01-stills"}
    after = _load(studio, pid)
    assert after["pending"] is None and after["last_error"] is None
    assert [b["out"] for b in after["bundles"]] == ["01-stills"]
    assert after["view"]["digest"] == after["bundles"][0]["digest"]
    assert after["last_number"] == 1


def test_a_replay_sends_its_edits_and_the_previous_bundle(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window in a storm"}}}, 0)
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "done"
    sent = studio.client.directions[-1]
    first = _load(studio, pid)["bundles"][0]
    assert sent["source"] == first["digest"] and sent["out"] == "02-stills"
    assert sent["cards"] == {"shot_b002": {"picture_text": "a window in a storm"}}
    shown = {c["shot_id"]: c["picture_text"]
             for c in _load(studio, pid)["view"]["scenes"][0]["shots"]}
    assert shown["shot_b002"] == "a window in a storm"
    assert producer.build_direction(_load(studio, pid), "video")["cards"] == {}, \
        "an edit the bundle now shows is an edit no more"


def test_each_clip_is_noted_with_the_still_it_was_filmed_from(studio):
    # A still painted again after its clip was filmed leaves that clip
    # showing the old picture: the page must say it needs filming again.
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    _run(studio, pid, "video")
    cards = {c["shot_id"]: c for c in _load(studio, pid)["view"]["scenes"][0]["shots"]}
    assert all(c["clip_still"] == c["still"] for c in cards.values())
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window in a storm"}}}, 0)
    _run(studio, pid, "stills")
    cards = {c["shot_id"]: c for c in _load(studio, pid)["view"]["scenes"][0]["shots"]}
    assert cards["shot_b002"]["clip"] and cards["shot_b002"]["clip_still"] != \
        cards["shot_b002"]["still"]
    assert cards["shot_b001"]["clip_still"] == cards["shot_b001"]["still"]


def test_a_draft_saved_while_a_stage_runs_stays_an_edit(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window in a storm"}}}, 0)
    plan = studio.prepare_stage(pid, "stills")
    stream = studio.stream_stage(plan)
    assert next(stream)["type"] == "submitted"
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window at dawn"}}}, 1)
    assert list(stream)[-1]["type"] == "done"
    after = _load(studio, pid)
    assert after["draft"]["revision"] == 2
    assert producer.build_direction(after, "stills")["cards"] == {
        "shot_b002": {"picture_text": "a window at dawn"}}


def test_a_run_sends_the_project_as_it_stands_when_its_attempt_is_saved(studio):
    # Prepared, then overtaken: a draft saved and a whole stage run since.
    # What is sent is built when the attempt is saved, so it carries the
    # newest draft and the next bundle number, never a stale plan.
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    stale = studio.prepare_stage(pid, "stills")
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window in a storm"}}}, 0)
    _run(studio, pid, "stills")                                   # 02-stills
    producer.save_draft(studio.settings.out_dir, pid,
                        {"cards": {"shot_b002": {"picture_text": "a window at dawn"}}}, 1)
    events = list(studio.stream_stage(stale))
    assert events[-1]["bundle"] == "03-stills"
    sent = studio.client.directions[-1]
    assert sent["source"] == _load(studio, pid)["bundles"][1]["digest"]
    assert sent["cards"] == {"shot_b002": {"picture_text": "a window at dawn"}}


def test_a_project_whose_workflow_has_gone_is_refused_readably(studio, tmp_path):
    # A pack update can rename the workflow a project started on.
    project = producer.new_project(studio.settings.out_dir,
                                   workflow=str(tmp_path / "retired_workflow.json"),
                                   choices={"bank": "my_story"})
    with pytest.raises(producer.ProjectError, match="workflow is not at"):
        studio.prepare_stage(project["project_id"], "stills")


def test_a_client_that_cannot_be_built_sends_nothing_and_frees_the_project(studio):
    pid = _project(studio)["project_id"]

    def broken(url):
        raise RuntimeError("the Comfy SDK is not installed")

    studio._client_factory = broken
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "error" and "not installed" in events[-1]["message"]
    assert _load(studio, pid)["pending"] is None


def test_a_refused_submit_frees_the_project_and_an_unclear_one_holds_it(studio):
    pid = _project(studio)["project_id"]
    studio.client.refuse = Refused("node 63 refused the direction")
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "error" and "node 63 refused" in events[-1]["message"]
    project = _load(studio, pid)
    assert project["pending"] is None and project["last_error"]["stage"] == "stills"
    studio.client.refuse = ReadTimeout("the bridge did not answer in time")
    events = _run(studio, pid, "stills")
    assert "may have taken this run" in events[-1]["message"]
    assert _load(studio, pid)["pending"]["state"] == "uncertain"
    with pytest.raises(producer.StageInFlight):
        studio.prepare_stage(pid, "stills")


def test_a_failed_job_frees_the_project_unless_its_bundle_was_frozen(studio):
    pid = _project(studio)["project_id"]
    studio.client.fail = JobFailed("job ended failed")
    studio.client.freezes = False
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "error" and "job ended failed" in events[-1]["message"]
    assert _load(studio, pid)["pending"] is None
    studio.client.freezes = True        # failed after its bundle was written
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "done"
    assert [b["out"] for b in _load(studio, pid)["bundles"]] == ["02-stills"]


def test_a_finished_job_whose_bundle_is_not_there_yet_is_held(studio):
    pid = _project(studio)["project_id"]
    studio.client.freezes = False
    events = _run(studio, pid, "stills")
    assert events[-1]["type"] == "error" and "not complete yet" in events[-1]["message"]
    pending = _load(studio, pid)["pending"]
    assert (pending["state"], pending["job_id"]) == ("finishing", "job-1")


def test_publish_saves_the_episode_and_records_it(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    _run(studio, pid, "video")
    events = _run(studio, pid, "publish")
    done = events[-1]
    assert (done["type"], done["stage"], done["project_id"]) == ("done", "publish", pid)
    saved = studio.settings.out_dir / done["run"] / done["file"]
    assert saved.read_bytes() == b"episode-bytes"
    project = _load(studio, pid)
    assert project["published"]["run"] == done["run"] and project["pending"] is None
    assert [b["out"] for b in project["bundles"]] == ["01-stills", "02-video"]


def test_a_publish_with_no_video_back_frees_the_project(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    _run(studio, pid, "video")
    studio.client.on_submit = None
    original = studio.client.submit

    def no_video(graph):
        job = original(graph)
        job.outputs = []
        return job

    studio.client.submit = no_video
    events = _run(studio, pid, "publish")
    assert events[-1]["type"] == "error" and "no video" in events[-1]["message"]
    assert _load(studio, pid)["pending"] is None


# --------------------------------------------------------------------------- #
# over HTTP
# --------------------------------------------------------------------------- #

@pytest.fixture
def app(studio):
    server = make_server(studio, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], studio
    server.shutdown()
    server.server_close()


def _call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, None if body is None else json.dumps(body), headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data) if data else None


def _events(port, run_id):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", f"/api/episodes/{run_id}/events")
    data = conn.getresponse().read()
    conn.close()
    return [json.loads(line[6:]) for line in data.decode().splitlines()
            if line.startswith("data: ")]


def test_a_project_over_http(app):
    port, studio = app
    status, made = _call(port, "POST", "/api/projects", {"choices": {"bank": "my_story"}})
    assert status == 202
    pid = made["project_id"]
    assert _events(port, made["id"])[-1] == {"type": "done", "stage": "stills",
                                             "project_id": pid, "bundle": "01-stills"}
    status, listing = _call(port, "GET", "/api/projects")
    assert status == 200 and listing["projects"][0]["project_id"] == pid
    status, read = _call(port, "GET", f"/api/projects/{pid}")
    assert status == 200 and read["project"]["bundles"][0]["out"] == "01-stills"
    media = read["project"]["view"]["scenes"][0]["shots"][0]["still"]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", media)
    resp = conn.getresponse()
    assert resp.status == 200 and resp.read().startswith(b"PNG of ")
    conn.close()
    draft = {"cards": {"shot_b001": {"motion_text": "sits down"}}}
    assert _call(port, "POST", f"/api/projects/{pid}/draft",
                 {"draft": draft, "revision": 0}) == (200, {"revision": 1})
    status, body = _call(port, "POST", f"/api/projects/{pid}/draft",
                         {"draft": draft, "revision": 0})
    assert status == 409 and "edited elsewhere" in body["error"]
    status, body = _call(port, "POST", f"/api/projects/{pid}/draft",
                         {"draft": {"cards": "no"}, "revision": 1})
    assert status == 400
    status, started = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "video"})
    assert status == 202
    assert _events(port, started["id"])[-1]["bundle"] == "02-video"
    assert studio.client.directions[-1]["cards"] == {"shot_b001": {"motion_text": "sits down"}}


def test_project_routes_refuse_what_cannot_run(app):
    port, studio = app
    status, _ = _call(port, "GET", "/api/projects/0000aaaa0000")
    assert status == 404
    status, _ = _call(port, "POST", "/api/projects/0000aaaa0000/stages", {"stage": "stills"})
    assert status == 404
    pid = _project(studio)["project_id"]
    status, body = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "video"})
    assert status == 400 and "starts with its stills" in body["error"]
    status, body = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "edit"})
    assert status == 400
    studio.client.refuse = ReadTimeout("the bridge did not answer in time")
    status, started = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    _events(port, started["id"])
    status, body = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    assert status == 409 and "in flight" in body["error"]


def test_a_run_that_raises_still_ends_and_frees_the_gpu(app, monkeypatch):
    port, studio = app

    def broken(plan):
        yield {"type": "submitted"}
        raise RuntimeError("a bug in the stage")

    monkeypatch.setattr(studio, "stream_stage", broken)
    pid = _project(studio)["project_id"]
    status, started = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    assert status == 202
    last = _events(port, started["id"])[-1]
    assert last["type"] == "error" and "a bug in the stage" in last["message"]
    monkeypatch.undo()
    status, _ = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    assert status == 202, "the guard was released"


def test_only_loopback_hosts_and_bounded_bodies_are_answered(app):
    port, _studio = app
    for host in ("[::1].evil.example", "[::1]x", "127.0.0.1.evil.example"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/api/projects", headers={"Host": host})
        assert conn.getresponse().status == 403, host
        conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/api/projects", headers={"Host": "[::1]:8765"})
    assert conn.getresponse().status == 200
    conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/api/projects")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(10 ** 30))
    conn.endheaders()
    assert conn.getresponse().status == 400
    conn.close()


def test_a_stage_waits_its_turn_behind_an_episode(app, monkeypatch):
    port, studio = app
    pid = _project(studio)["project_id"]
    gate = threading.Event()

    def slow(choices):
        gate.wait(5)
        yield {"type": "done", "path": "", "run": "", "file": ""}

    monkeypatch.setattr(studio, "stream", slow)
    status, episode = _call(port, "POST", "/api/episodes", {"choices": {}})
    assert status == 202
    status, body = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    assert status == 409 and "already being made" in body["error"]
    gate.set()
    _events(port, episode["id"])
    status, _ = _call(port, "POST", f"/api/projects/{pid}/stages", {"stage": "stills"})
    assert status == 202


# --------------------------------------------------------------------------- #
# restart and reconciliation (chunk 5c)
# --------------------------------------------------------------------------- #

def _sent(studio, pid, stage, record=True):
    """A stage sent and taken, as if Studio stopped right after: the attempt
    saved, the job made, its id saved only when ``record``."""
    plan = studio.prepare_stage(pid, stage)
    direction = producer.begin_attempt(studio.settings.out_dir, pid, stage, plan["graph"],
                                       plan["node_ids"])
    job = studio.client.submit(plan["graph"])
    if record:
        producer.record_job(studio.settings.out_dir, pid, direction["attempt_id"], job.id)
    return direction, job, plan["graph"]


def test_a_run_studio_lost_touch_with_is_followed_to_its_end_and_never_sent_again(studio):
    pid = _project(studio)["project_id"]
    _sent(studio, pid, "stills")
    events = list(studio.reconcile_stage(pid))
    assert events[-1]["type"] == "done" and events[-1]["stage"] == "stills"
    project = _load(studio, pid)
    assert project["pending"] is None and [b["out"] for b in project["bundles"]] == ["01-stills"]
    assert len(studio.client.directions) == 1, "sent once, in all"


def test_a_lost_submit_is_found_by_its_attempt_in_comfyuis_queue(studio):
    pid = _project(studio)["project_id"]
    direction, job, graph = _sent(studio, pid, "stills", record=False)
    assert _load(studio, pid)["pending"]["state"] == "submitting"
    other = json.loads(json.dumps(graph))
    lock = next(n for n in other.values() if n.get("class_type") == SHOT_LOCK)
    lock["inputs"]["direction_json"] = json.dumps(dict(direction, attempt_id="f" * 32))
    studio.comfy["/queue"]["queue_running"] = [[1, job.id, graph, {}, []]]
    studio.comfy["/queue"]["queue_pending"] = [[2, "someone-else", other, {}, []]]
    events = list(studio.reconcile_stage(pid))
    assert events[-1]["type"] == "done"
    assert _load(studio, pid)["bundles"][0]["attempt_id"] == direction["attempt_id"]
    assert len(studio.client.directions) == 1


def test_a_lost_submit_comfyui_never_took_frees_the_project_only_after_every_look(studio):
    pid = _project(studio)["project_id"]
    plan = studio.prepare_stage(pid, "stills")
    producer.begin_attempt(studio.settings.out_dir, pid, "stills", plan["graph"], plan["node_ids"])
    events = list(studio.reconcile_stage(pid))
    assert events[-1]["type"] == "error" and "never took" in events[-1]["message"]
    assert studio.slept == [rps_runner.LOST_SUBMIT_GAP_S] * (rps_runner.LOST_SUBMIT_LOOKS - 1)
    project = _load(studio, pid)
    assert project["pending"] is None and "never took" in project["last_error"]["message"]
    assert studio.client.directions == [], "nothing was ever sent"


@pytest.mark.parametrize("trouble", ["two", "down"])
def test_a_lost_submit_stays_pending_when_comfyui_cannot_say_which_run_it_is(studio, trouble):
    pid = _project(studio)["project_id"]
    _direction, job, graph = _sent(studio, pid, "stills", record=False)
    if trouble == "two":
        studio.comfy["/queue"]["queue_running"] = [[1, job.id, graph, {}, []]]
        studio.comfy["/history"] = {"job-x": {"prompt": [0, "job-x", graph, {}, []]}}
    else:
        studio.comfy["down"] = True
    events = list(studio.reconcile_stage(pid))
    assert events[-1]["type"] == "error"
    pending = _load(studio, pid)["pending"]
    assert (pending["state"], pending["job_id"]) == ("uncertain", None)
    assert studio.slept == [], "an answer that is not empty is never waited out"


def test_a_job_the_bridge_forgot_is_read_from_comfyuis_own_history(studio):
    pid = _project(studio)["project_id"]
    _direction, job, _graph = _sent(studio, pid, "stills")
    studio.client.made.clear()                   # the bridge restarted
    studio.comfy["/history"] = {job.id: {"status": {"status_str": "success", "completed": True}}}
    assert list(studio.reconcile_stage(pid))[-1]["type"] == "done"
    assert _load(studio, pid)["bundles"][0]["out"] == "01-stills"


def test_a_job_neither_bridge_nor_comfyui_knows_with_no_bundle_lets_the_project_try_again(studio):
    pid = _project(studio)["project_id"]
    studio.client.freezes = False
    _sent(studio, pid, "stills")
    studio.client.made.clear()
    events = list(studio.reconcile_stage(pid))
    assert events[-1]["type"] == "error" and "no longer has this run" in events[-1]["message"]
    assert _load(studio, pid)["pending"] is None


def test_a_job_comfyui_is_still_making_stays_pending(studio):
    pid = _project(studio)["project_id"]
    _direction, job, graph = _sent(studio, pid, "stills")
    studio.client.made.clear()
    studio.comfy["/queue"]["queue_running"] = [[1, job.id, graph, {}, []]]
    assert list(studio.reconcile_stage(pid))[-1]["type"] == "error"
    assert _load(studio, pid)["pending"]["state"] == "running"


def test_a_publish_made_but_not_saved_is_saved_without_making_it_again(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    _run(studio, pid, "video")
    direction, job, _graph = _sent(studio, pid, "publish")
    producer.hold_attempt(studio.settings.out_dir, pid, direction["attempt_id"], "finishing",
                          "the download failed")
    done = list(studio.reconcile_stage(pid))[-1]
    assert (done["type"], done["stage"]) == ("done", "publish")
    assert (studio.settings.out_dir / done["run"] / done["file"]).read_bytes() == b"episode-bytes"
    assert _load(studio, pid)["published"]["attempt_id"] == direction["attempt_id"]
    assert [d["stage"] for d in studio.client.directions] == ["stills", "video", "publish"]


def test_a_publish_the_bridge_forgot_is_fetched_from_where_comfyui_saved_it(studio):
    pid = _project(studio)["project_id"]
    _run(studio, pid, "stills")
    _run(studio, pid, "video")
    _direction, job, _graph = _sent(studio, pid, "publish")
    studio.client.made.clear()
    # "9": the stand-in workflow's episode node, as the job's own outputs name it.
    studio.output[("otr/obs", "radio_play.mp4")] = b"episode-from-history"
    studio.comfy["/history"] = {job.id: {
        "status": {"status_str": "success", "completed": True},
        "outputs": {"9": {"gifs": [{"filename": "radio_play.mp4", "subfolder": "otr/obs",
                                    "type": "output"}]}}}}
    done = list(studio.reconcile_stage(pid))[-1]
    assert done["type"] == "done", done
    assert (studio.settings.out_dir / done["run"] / done["file"]).read_bytes() == \
        b"episode-from-history"
    assert len(studio.client.directions) == 3


def test_a_project_comfyui_may_still_be_working_for_keeps_every_run_waiting(studio):
    from radio_play_studio.server import Busy, Runs
    pid = _project(studio)["project_id"]
    direction, job, graph = _sent(studio, pid, "stills", record=False)
    producer.hold_attempt(studio.settings.out_dir, pid, direction["attempt_id"], "uncertain",
                          "the answer was lost")
    runs = Runs(studio)
    with pytest.raises(Busy, match=pid):
        runs.start({"bank": "my_story"})
    other = _project(studio)["project_id"]
    with pytest.raises(Busy, match=pid):
        runs.start_stage(other, "stills")
    with pytest.raises(producer.StageInFlight):
        runs.start_stage(pid, "stills")
    studio.comfy["/queue"]["queue_running"] = [[1, job.id, graph, {}, []]]
    run = runs.start_reconcile(pid)
    with run.cond:
        run.cond.wait_for(lambda: run.finished, timeout=30)
    assert run.events[-1]["type"] == "done" and _load(studio, pid)["pending"] is None
    assert run.label == {"project_id": pid, "stage": "stills", "reconcile": True}


def test_a_run_given_up_by_hand_frees_the_project(studio):
    from radio_play_studio.server import Runs
    pid = _project(studio)["project_id"]
    direction, _job, _graph = _sent(studio, pid, "stills", record=False)
    runs = Runs(studio)
    with pytest.raises(producer.ProjectError):
        runs.forget(pid, "0" * 32)                 # not the pending attempt
    project = runs.forget(pid, direction["attempt_id"])
    assert project["pending"] is None and "forgotten" in project["last_error"]["message"]


def test_studio_settles_what_was_left_pending_when_it_starts(studio):
    from radio_play_studio.server import Runs
    pid = _project(studio)["project_id"]
    _sent(studio, pid, "stills")
    Runs(studio).resume_pending(wait=30)
    assert _load(studio, pid)["pending"] is None
    assert len(studio.client.directions) == 1
