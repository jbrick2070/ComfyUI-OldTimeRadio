"""Driving a run through the SDK, with the SDK and the server faked."""
from __future__ import annotations

import os

import pytest

from radio_play_studio.graph import ServerUnreachable
from radio_play_studio.runner import (
    EpisodeMissing, Studio, describe_failure, is_transient, make_client, normalize_event,
    pick_episode_output, settle,
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

    def wait(self, timeout=None):
        return self

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


def test_pick_episode_output_takes_only_the_mux_nodes_video():
    mux_video = FakeOutput("a.mp4")
    mux_video.node_id = "9"
    silent_cut = FakeOutput("silent_cut.mp4")
    silent_cut.node_id = "4"

    class Job:
        outputs = [silent_cut, mux_video]
        def get_outputs(self, node_id):
            return [o for o in self.outputs if o.node_id == node_id]
    assert pick_episode_output(Job(), "9") is mux_video
    with pytest.raises(EpisodeMissing, match="OTR_OBS_DIR"):
        pick_episode_output(Job(), "55")


def test_the_silent_cut_is_never_reported_as_the_episode(settings, schemas):
    """Seen against the real SDK: with the published file outside ComfyUI's
    output folder the mux node offers no video, and the silent cut's mp4 was
    the only one left."""
    silent_cut = FakeOutput("silent_cut.mp4")
    silent_cut.node_id = "4"
    poster = FakeOutput("poster.png")
    poster.node_id = "9"
    studio, _ = _studio(settings, schemas, FakeJob([], [silent_cut, poster]))
    events = list(studio.stream({}))
    assert events[-1]["type"] == "error" and not any(e["type"] == "done" for e in events)
    assert not settings.out_dir.exists() or not any(settings.out_dir.rglob("silent_cut.mp4"))


class Flaky(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.http_status = status


class PollingJob(FakeJob):
    """A job whose first polls fail the way a restarting bridge does."""
    def __init__(self, failures, still_running=0, **kw):
        super().__init__([], [], **kw)
        self._failures, self._running = list(failures), still_running

    def wait(self, timeout=None):
        if self._failures:
            raise self._failures.pop(0)
        if self._running:
            self._running -= 1
            raise TimeoutError("not terminal yet")
        return self


def test_settle_rides_out_a_restarting_bridge_and_a_long_render():
    pauses = []
    job = PollingJob([Flaky(503), ConnectionError("reset"), Flaky(429)], still_running=3)
    assert settle(job, sleep=pauses.append) is job
    assert pauses == [2.0, 4.0, 8.0] and job.seen_result


def test_settle_gives_up_on_a_bridge_that_stays_down():
    job = PollingJob([Flaky(503)] * 50)
    with pytest.raises(Flaky):
        settle(job, sleep=lambda s: None)


def test_settle_does_not_retry_a_verdict():
    class JobFailed(Exception):
        error = "node died"
    job = PollingJob([], fail=JobFailed("failed"))
    with pytest.raises(JobFailed):
        settle(job, sleep=lambda s: pytest.fail("a failed job must not be retried"))
    with pytest.raises(Flaky):                       # a 404 is the job being gone, not a blip
        settle(PollingJob([Flaky(404)]), sleep=lambda s: None)


def test_is_transient_reads_status_and_connection_errors():
    assert is_transient(Flaky(502)) and is_transient(Flaky(429)) and is_transient(ConnectionResetError())
    assert not is_transient(Flaky(404)) and not is_transient(ValueError("x"))


def test_a_lost_event_stream_does_not_stop_a_healthy_run(settings, schemas):
    class DeadStream(FakeJob):
        def events(self):
            raise Flaky(503)
            yield  # pragma: no cover - makes this a generator
    video = FakeOutput("otr_episode.mp4")
    video.node_id = "9"
    studio, _ = _studio(settings, schemas, DeadStream([], [video]))
    events = list(studio.stream({}))
    assert [e["type"] for e in events] == ["submitted", "note", "done"]


def test_make_client_refuses_a_blank_bridge_address(monkeypatch):
    monkeypatch.setenv("COMFY_BASE_URL", "http://127.0.0.1:8189")
    for bad in ("", " ", "/", "cloud.comfy.org", "ftp://host"):
        with pytest.raises(ValueError):
            make_client(bad)
    assert os.environ["COMFY_BASE_URL"] == "http://127.0.0.1:8189"


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
