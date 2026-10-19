"""Driving a run through the SDK, with the SDK and the server faked."""
from __future__ import annotations

import os

import pytest

from radio_play_studio.graph import ServerUnreachable
from radio_play_studio.runner import (
    EpisodeMissing, Studio, describe_failure, make_client, normalize_event, pick_episode_output,
)


# Named like the SDK's event classes: the app matches on the class name.
class Progress:
    def __init__(self, value, message=None):
        self.value, self.message = value, message


class StatusChange:
    def __init__(self, status):
        self.status = status


class Preview:
    pass


class FakeOutput:
    def __init__(self, name, payload=b"video-bytes"):
        self.name, self._payload = name, payload

    def to_file(self, path):
        with open(path, "wb") as handle:
            handle.write(self._payload)


class OutputReady:
    def __init__(self, output):
        self.output = output


class FakeJob:
    def __init__(self, events, outputs, fail=None):
        self._events, self.outputs, self._fail = events, outputs, fail
        self.seen_result = False

    def events(self):
        yield from self._events

    def result(self):
        if self._fail:
            raise self._fail
        self.seen_result = True
        return self

    def get_outputs(self, node_id):
        assert self.seen_result, "outputs are only read after the job is confirmed finished"
        return [o for o in self.outputs if getattr(o, "node_id", node_id) == node_id]


class FakeClient:
    def __init__(self, job):
        self._job, self.submitted = job, None
        self.workflows = self

    def from_json(self, graph):
        self.submitted = graph
        return graph

    def submit(self, workflow):
        return self._job


def _studio(settings, schemas, job):
    client = FakeClient(job)
    studio = Studio(settings, client_factory=lambda url: client, fetch=lambda url: schemas)
    return studio, client


def _good_job():
    video = FakeOutput("otr_episode.mp4")
    video.node_id = "9"
    poster = FakeOutput("poster.png")
    poster.node_id = "9"
    return FakeJob([StatusChange("running"), Progress(0.5, "writing"), Preview(), OutputReady(poster)],
                   [poster, video])


def test_normalize_event_keeps_what_the_page_shows():
    assert normalize_event(Progress(0.25, "casting")) == {"type": "progress", "value": 0.25, "message": "casting"}
    assert normalize_event(Progress(0.5)) == {"type": "progress", "value": 0.5, "message": ""}
    assert normalize_event(StatusChange("running")) == {"type": "status", "status": "running"}
    assert normalize_event(OutputReady(FakeOutput("a.png"))) == {"type": "output", "name": "a.png"}
    assert normalize_event(Preview()) is None


def test_normalize_event_reads_the_real_sdk_classes():
    sdk = pytest.importorskip("comfy_sdk")
    from comfy_sdk.events import OutputReady as RealOutput, Progress as RealProgress, StatusChange as RealStatus
    assert normalize_event(RealProgress(value=0.4, message=None)) == {"type": "progress", "value": 0.4, "message": ""}
    assert normalize_event(RealStatus(status="succeeded")) == {"type": "status", "status": "succeeded"}
    assert normalize_event(RealOutput(output=FakeOutput("e.mp4")))["name"] == "e.mp4"
    assert sdk is not None


def test_a_run_streams_events_then_saves_the_video(settings, schemas):
    studio, client = _studio(settings, schemas, _good_job())
    events = list(studio.stream({"bank": "shakespeare"}))
    kinds = [e["type"] for e in events]
    assert kinds == ["submitted", "status", "progress", "output", "done"]
    done = events[-1]
    saved = settings.out_dir / done["run"] / done["file"]
    assert done["file"] == "otr_episode.mp4" and saved.read_bytes() == b"video-bytes"
    assert client.submitted["7"]["inputs"]["source_bank"] == "shakespeare"


def test_the_client_is_made_for_the_local_bridge(settings, schemas):
    seen = []
    client = FakeClient(_good_job())
    studio = Studio(settings, client_factory=lambda url: seen.append(url) or client, fetch=lambda url: schemas)
    list(studio.stream({}))
    assert seen == ["http://bridge.invalid"]


def test_make_client_never_leaves_the_sdk_on_comfy_cloud(monkeypatch):
    pytest.importorskip("comfy_sdk")
    monkeypatch.delenv("COMFY_BASE_URL", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    make_client("http://127.0.0.1:8189")
    assert os.environ["COMFY_BASE_URL"] == "http://127.0.0.1:8189"


def test_a_failed_job_ends_the_stream_with_an_error(settings, schemas):
    class JobFailed(Exception):
        error = {"node": "OTR_Foo"}
    job = FakeJob([StatusChange("running")], [], fail=JobFailed("job x ended failed"))
    studio, _ = _studio(settings, schemas, job)
    events = list(studio.stream({}))
    assert events[-1]["type"] == "error" and "OTR_Foo" in events[-1]["message"]
    assert not any(e["type"] == "done" for e in events)


def test_a_finished_job_with_no_video_is_not_a_finished_episode(settings, schemas):
    poster = FakeOutput("poster.png")
    poster.node_id = "9"
    studio, _ = _studio(settings, schemas, FakeJob([], [poster]))
    last = list(studio.stream({}))[-1]
    assert last["type"] == "error" and "no video" in last["message"]


def test_an_empty_file_is_not_a_saved_episode(settings, schemas):
    video = FakeOutput("otr_episode.mp4", payload=b"")
    video.node_id = "9"
    studio, _ = _studio(settings, schemas, FakeJob([], [video]))
    last = list(studio.stream({}))[-1]
    assert last["type"] == "error" and "was not saved" in last["message"]


def test_bad_choices_end_the_stream_before_anything_is_submitted(settings, schemas):
    studio, client = _studio(settings, schemas, _good_job())
    last = list(studio.stream({"bank": "not a bank"}))[-1]
    assert last["type"] == "error" and client.submitted is None


def test_pick_episode_output_prefers_the_mux_node_then_any_video():
    mux_video = FakeOutput("a.mp4")
    mux_video.node_id = "9"
    other = FakeOutput("b.webm")
    other.node_id = "4"

    class Job:
        outputs = [other]
        def get_outputs(self, node_id):
            return [mux_video] if node_id == "9" else []
    assert pick_episode_output(Job(), "9") is mux_video
    assert pick_episode_output(Job(), "55") is other
    Job.outputs = []
    with pytest.raises(EpisodeMissing):
        pick_episode_output(Job(), "55")


def test_options_come_from_the_live_server(settings, schemas):
    studio, _ = _studio(settings, schemas, _good_job())
    assert {r["key"] for r in studio.options()} >= {"bank", "acts", "cast_size"}


def test_an_unreachable_server_surfaces_as_such(settings):
    def down(url):
        raise ServerUnreachable("no ComfyUI")
    studio = Studio(settings, fetch=down)
    with pytest.raises(ServerUnreachable):
        studio.options()
    assert list(studio.stream({}))[-1] == {"type": "error", "message": "ServerUnreachable: no ComfyUI"}


def test_describe_failure_includes_node_detail_when_the_sdk_gives_it():
    class E(Exception):
        error = "boom"
    assert describe_failure(E("failed")) == "E: failed (boom)"
    assert describe_failure(ValueError("x")) == "ValueError: x"
