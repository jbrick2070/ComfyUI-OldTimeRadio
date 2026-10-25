"""The headless runner's live view: which node is running, and how far along.

(PLAN row 1.1, planning/creatoros-concepts/PLAN.md.) `poll_history` only reads
/history, so a leg log said `pending` for the whole of a video walk. The frame
shapes below are the ones ComfyUI sends on /ws (`executing`, `progress`,
`progress_state`, binary previews); the socket test runs a real aiohttp
websocket server on a loopback port.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import struct
import threading
import time

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


@pytest.fixture(scope="module")
def api():
    """Load `scripts/otr_api.py` directly -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "otr_api_live_view_under_test", SCRIPTS / "otr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def labels(api):
    return api.load_phase_labels()


PROMPT = {
    "1": {"class_type": "OTR_LedgerScriptWriter", "inputs": {}},
    "7": {"class_type": "OTR_VideoRenderBatch", "inputs": {}},
    "9": {"class_type": "KSampler", "inputs": {}},
    "12": {"class_type": "SomeoneElsesNode", "inputs": {}},
}


def _state(api, labels, prompt_id=None):
    class_by_node = {n: spec["class_type"] for n, spec in PROMPT.items()}
    return api.LiveState(class_by_node, labels, prompt_id=prompt_id)


def _text(kind, **data):
    return {"type": kind, "data": data}


# --- the label table ---------------------------------------------------------

def test_every_node_class_in_node_list_has_a_label(labels):
    """A new node cannot run unnamed."""
    classes = json.loads((REPO / "node_list.json").read_text(encoding="utf-8"))
    missing = sorted(name for name in classes if not labels["classes"].get(name))
    assert not missing, (
        "add a plain label for these classes to config/node_phase_labels.json: "
        + ", ".join(missing))


def test_labels_are_plain_single_lines(labels):
    for name, label in labels["classes"].items():
        assert label == label.strip() and "\n" not in label, name
        label.encode("ascii")


def test_label_order_is_exact_then_pattern_then_the_class_name(api, labels):
    assert api.label_for_class("OTR_LedgerScriptWriter", labels) == "Writing the script"
    assert api.label_for_class("KSampler", labels) == "Sampling"
    assert api.label_for_class("VAEDecodeTiled", labels) == "Decoding"
    assert api.label_for_class("CheckpointLoaderSimple", labels) == "Loading"
    # No invented phrase for a class nobody labelled.
    assert api.label_for_class("SomeoneElsesNode", labels) == "SomeoneElsesNode"
    assert api.label_for_class(None, labels) is None


def test_an_unreadable_label_file_is_reported_not_hidden(api, tmp_path, monkeypatch):
    view = api.LiveView("c", PROMPT)
    assert "note" not in view.snapshot()
    broken = tmp_path / "labels.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(api, "PHASE_LABELS_PATH", str(broken))
    view = api.LiveView("c", PROMPT)
    note = view.snapshot()["note"]
    assert note.startswith("labels unreadable:")
    assert "[labels unreadable:" in api.format_live(view.snapshot())


# --- frame parsing -----------------------------------------------------------

def test_executing_then_progress_names_the_node_and_its_progress(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("execution_start", prompt_id="p"))
    state.apply_text(_text("executing", node="7", prompt_id="p"))
    state.apply_text(_text("progress", value=34, max=96, node="7", prompt_id="p"))
    snap = state.snapshot()
    assert (snap["state"], snap["node"], snap["class"]) == (
        "running", "7", "OTR_VideoRenderBatch")
    assert (snap["label"], snap["value"], snap["max"]) == ("Rendering the video", 34, 96)
    assert api.format_live(snap) == "node 7 Rendering the video 34/96"


def test_a_new_node_resets_the_progress_of_the_last_one(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("executing", node="7"))
    state.apply_text(_text("progress", value=96, max=96, node="7"))
    state.apply_text(_text("executing", node="1"))
    snap = state.snapshot()
    assert (snap["node"], snap["value"], snap["max"]) == ("1", None, None)
    assert api.format_live(snap) == "node 1 Writing the script"


def test_progress_alone_can_move_the_node(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("progress", value=2, max=20, node="9"))
    snap = state.snapshot()
    assert (snap["node"], snap["label"]) == ("9", "Sampling")


def test_executing_null_means_finished(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("executing", node="7"))
    state.apply_text(_text("executing", node=None))
    snap = state.snapshot()
    assert snap["state"] == "finished" and snap["node"] is None
    assert api.format_live(snap) == "finished"


def test_progress_state_names_the_running_node(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("progress_state", prompt_id="p", nodes={
        "1": {"state": "finished", "value": 1, "max": 1, "node_id": "1"},
        "7": {"state": "running", "value": 5, "max": 50, "node_id": "7"},
    }))
    snap = state.snapshot()
    assert (snap["node"], snap["value"], snap["max"]) == ("7", 5, 50)


def test_frames_for_another_prompt_are_ignored_once_ours_is_known(api, labels):
    state = _state(api, labels, prompt_id="mine")
    state.apply_text(_text("executing", node="7", prompt_id="theirs"))
    assert state.snapshot()["state"] == "waiting"
    state.apply_text(_text("executing", node="1", prompt_id="mine"))
    assert state.snapshot()["node"] == "1"


def test_early_state_from_another_prompt_is_dropped_when_ours_is_known(api, labels):
    """The socket opens before `submit_prompt` returns an id, so until then
    every frame is taken. A broadcast about ANOTHER client's prompt (ComfyUI
    sends `execution_interrupted` to everyone) must not leave this view
    saying `finished` for a prompt that has not started."""
    state = _state(api, labels)
    state.apply_text(_text("executing", node="9", prompt_id="theirs"))
    state.apply_text(_text("execution_interrupted", prompt_id="theirs"))
    assert state.snapshot()["state"] == "finished"
    state.adopt_prompt_id("mine")
    snap = state.snapshot()
    assert (snap["state"], snap["node"], snap["value"], snap["max"]) == (
        "waiting", None, None, None)
    # ...and from here on theirs is ignored and ours is taken.
    state.apply_text(_text("executing", node="7", prompt_id="theirs"))
    assert state.snapshot()["state"] == "waiting"
    state.apply_text(_text("executing", node="1", prompt_id="mine"))
    assert state.snapshot()["node"] == "1"


def test_early_state_from_our_own_prompt_is_kept_when_its_id_arrives(api, labels):
    """Our first frames can land before `set_prompt_id` runs; they are ours."""
    state = _state(api, labels)
    state.apply_text(_text("execution_start", prompt_id="mine"))
    state.apply_text(_text("executing", node="7", prompt_id="mine"))
    state.apply_text(_text("progress", value=3, max=9, node="7", prompt_id="mine"))
    state.adopt_prompt_id("mine")
    snap = state.snapshot()
    assert (snap["state"], snap["node"], snap["value"], snap["max"]) == (
        "running", "7", 3, 9)


def test_early_frames_that_name_no_prompt_are_kept(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("executing", node="1"))
    state.adopt_prompt_id("mine")
    assert state.snapshot()["node"] == "1"


def test_live_view_set_prompt_id_adopts_it(api):
    view = api.LiveView("c", PROMPT)
    view._on_text(json.dumps(_text("execution_interrupted", prompt_id="theirs")))
    assert view.snapshot()["state"] == "finished"
    view.set_prompt_id("mine")
    assert view.snapshot()["state"] == "waiting"
    assert view._state.prompt_id == "mine"


def test_a_subgraph_child_is_named_by_its_parent(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("executing", node="7:3"))
    assert state.snapshot()["class"] == "OTR_VideoRenderBatch"


def test_an_unknown_node_is_shown_by_id_only(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("executing", node="404"))
    snap = state.snapshot()
    assert snap["label"] is None
    assert api.format_live(snap) == "node 404"


@pytest.mark.parametrize("junk", [
    "not json", "[]", "null", "{}", '{"type": "executing"}',
    '{"type": "executing", "data": 5}', {"type": "progress", "data": []},
    b"\xff\xfe",
])
def test_malformed_frames_are_ignored(api, labels, junk):
    state = _state(api, labels)
    state.apply_text(junk)
    assert state.snapshot()["state"] == "waiting"


def test_non_numeric_progress_does_not_raise(api, labels):
    state = _state(api, labels)
    state.apply_text(_text("progress", value="x", max=None, node="7"))
    snap = state.snapshot()
    assert (snap["node"], snap["value"], snap["max"]) == ("7", None, None)


def test_binary_preview_frames_are_counted_and_other_binary_is_not(api, labels):
    state = _state(api, labels)
    state.apply_binary(struct.pack(">II", 1, 1) + b"jpeg-bytes")
    state.apply_binary(struct.pack(">II", 4, 2) + b"meta+png")
    state.apply_binary(struct.pack(">II", 3, 0) + b"text output")
    state.apply_binary(b"abc")
    snap = state.snapshot()
    assert snap["preview_frames"] == 2
    assert snap["preview_bytes"] == len(struct.pack(">II", 4, 2) + b"meta+png")
    assert state.other_binary_frames == 2


def test_format_live_for_every_state(api):
    assert api.format_live(None) == "none"
    assert api.format_live({"state": "waiting", "connected": False}) == "connecting"
    assert api.format_live({"state": "waiting", "connected": True}) == (
        "connected, no node reported yet")
    assert api.format_live({"state": "unavailable", "reason": "refused"}) == (
        "unavailable (refused)")
    assert api.format_live({"state": "running", "node": "3", "label": None,
                            "value": 1, "max": None}) == "node 3"
    assert api.format_live({"state": "running", "node": "3", "label": "Sampling",
                            "value": None, "max": 8}) == "node 3 Sampling 0/8"


def test_websocket_url(api):
    assert api.websocket_url("http://127.0.0.1:8000", "abc") == (
        "ws://127.0.0.1:8000/ws?clientId=abc")
    assert api.websocket_url("https://pod.example/", "a b") == (
        "wss://pod.example/ws?clientId=a%20b")


def test_wrap_adds_the_snapshot_then_calls_the_heartbeat(api):
    view = api.LiveView("c", PROMPT)
    seen = []
    tick = view.wrap(lambda elapsed, status: seen.append((elapsed, dict(status))))
    tick(5.0, {"status_str": "running"})
    assert seen[0][0] == 5.0
    assert seen[0][1]["status_str"] == "running"
    assert seen[0][1]["live"]["state"] == "waiting"
    view.wrap(None)(1.0, {})  # no heartbeat to call is fine


# --- a real websocket --------------------------------------------------------

class FakeComfy:
    """A loopback aiohttp server speaking ComfyUI's /ws: it records the clientId
    it was given, sends the scripted frames, then holds the socket open."""

    def __init__(self, frames):
        self.frames = frames
        self.client_ids: list = []
        self.port = None
        self._ready = threading.Event()
        self._loop = None
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    async def _handler(self, request):
        from aiohttp import web
        self.client_ids.append(request.query.get("clientId"))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        for kind, payload in self.frames:
            if kind == "text":
                await ws.send_str(json.dumps(payload))
            else:
                await ws.send_bytes(payload)
        async for _ in ws:
            pass
        return ws

    async def _main(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/ws", self._handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        self.port = runner.addresses[0][1]
        self._stop = asyncio.Event()
        self._ready.set()
        await self._stop.wait()
        await runner.cleanup()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._main())

    def start(self):
        self._thread.start()
        assert self._ready.wait(10), "the fake ComfyUI did not start"
        return self

    def close(self):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)


def _wait_for(view, predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    snap = view.snapshot()
    while time.monotonic() < deadline:
        snap = view.snapshot()
        if predicate(snap):
            return snap
        time.sleep(0.05)
    pytest.fail("the live view never reached the expected state; last snapshot: %r" % (snap,))


def test_live_view_follows_a_real_websocket(api):
    import aiohttp  # noqa: F401 -- a ComfyUI dependency; missing means FAIL, not skip
    server = FakeComfy([
        ("text", _text("status", status={"exec_info": {"queue_remaining": 1}})),
        ("text", _text("execution_start", prompt_id="p1")),
        ("text", _text("executing", node="7", prompt_id="p1")),
        ("text", _text("progress", value=3, max=10, node="7", prompt_id="p1")),
        ("binary", struct.pack(">II", 1, 1) + b"jpeg-bytes"),
    ]).start()
    view = api.LiveView("client-abc", PROMPT,
                        base_url="http://127.0.0.1:%d" % server.port)
    try:
        view.start()
        snap = _wait_for(view, lambda s: s["preview_frames"] == 1 and s["value"] == 3)
        assert snap["connected"] is True
        assert (snap["state"], snap["node"], snap["label"], snap["max"]) == (
            "running", "7", "Rendering the video", 10)
        assert api.format_live(snap) == "node 7 Rendering the video 3/10"
        assert server.client_ids == ["client-abc"]
        assert "preview_frames=1" in view.summary()
    finally:
        view.stop()
        server.close()
    assert not view._thread.is_alive()


def test_live_view_says_why_when_it_cannot_connect(api):
    import aiohttp  # noqa: F401 -- a ComfyUI dependency; missing means FAIL, not skip
    view = api.LiveView("client-abc", PROMPT, base_url="http://127.0.0.1:1")
    began = None
    try:
        view.start()
        snap = _wait_for(view, lambda s: s["state"] == "unavailable")
        assert snap["connected"] is False
        assert snap["reason"].startswith("websocket ")
        assert api.format_live(snap).startswith("unavailable (websocket ")
        began = time.monotonic()
    finally:
        view.stop()
    assert began is not None and time.monotonic() - began < 4, (
        "stop() must not wait out the back-off")
    assert not view._thread.is_alive()


def test_live_view_without_aiohttp_says_so(api, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("aiohttp is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    view = api.LiveView("client-abc", PROMPT, base_url="http://127.0.0.1:1")
    view.start()
    view._thread.join(10)
    monkeypatch.undo()
    snap = view.snapshot()
    assert snap["state"] == "unavailable"
    assert snap["reason"].startswith("aiohttp is not importable (ImportError")


# --- the preview probe is opt-in --------------------------------------------

class _Resp:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"prompt_id": "pid-1", "node_errors": {}}


def test_preview_method_is_sent_only_when_asked_for(api, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json=json) or _Resp())
    monkeypatch.delenv("OTR_COMFY_API_KEY", raising=False)
    monkeypatch.delenv("OTR_PREVIEW_METHOD", raising=False)
    api.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, client_id="c1")
    assert "extra_data" not in captured["json"]
    assert captured["json"]["client_id"] == "c1"

    monkeypatch.setenv("OTR_PREVIEW_METHOD", "LATENT2RGB")
    monkeypatch.setenv("OTR_COMFY_API_KEY", "k")
    api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})
    assert captured["json"]["extra_data"]["preview_method"] == "latent2rgb"
    assert captured["json"]["extra_data"]["api_key_comfy_org"] == "k"


def test_a_mistyped_preview_method_fails_loudly(api, monkeypatch):
    monkeypatch.setenv("OTR_PREVIEW_METHOD", "latent2rbg")
    with pytest.raises(ValueError, match="OTR_PREVIEW_METHOD='latent2rbg'"):
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})


# --- the preview file (PLAN row 1.1, second half) ---------------------------

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(range(32)) + b"\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(32))


def _preview_frame(image=JPEG, image_type=1):
    """A type-1 binary frame as ComfyUI sends it: event, image type, image."""
    return struct.pack(">II", 1, image_type) + image


def test_the_image_is_cut_out_of_each_preview_frame_layout(api):
    # Type 1: event, image type (1 = JPEG, 2 = PNG), then the image.
    assert api._image_from_preview_frame(_preview_frame(JPEG, 1)) == JPEG
    assert api._image_from_preview_frame(_preview_frame(PNG, 2)) == PNG
    # Type 4: event, metadata length, metadata JSON, then the image.
    meta = json.dumps({"node_id": "9", "image_type": "image/jpeg"}).encode("utf-8")
    framed = struct.pack(">II", 4, len(meta)) + meta + JPEG
    assert api._image_from_preview_frame(framed) == JPEG
    # No image in it, or too short to hold one: nothing to keep.
    assert api._image_from_preview_frame(_preview_frame(b"jpeg-bytes")) is None
    assert api._image_from_preview_frame(b"abc") is None


def test_the_newest_preview_image_and_its_sending_node_are_kept(api, labels):
    state = _state(api, labels)
    state.apply_binary(_preview_frame(JPEG + b"before-any-node"))
    state.apply_text(_text("executing", node="9"))
    state.apply_binary(_preview_frame(JPEG + b"first"))
    serial = state.preview_serial
    state.apply_binary(_preview_frame(JPEG + b"second"))
    assert state.last_preview == JPEG + b"second"
    assert state.preview_serial == serial + 1
    state.apply_text(_text("executing", node="7"))
    state.apply_binary(_preview_frame(JPEG))
    assert state.preview_by_node == {"?": 1, "9": 2, "7": 1}
    # A preview frame with no image in it is counted and keeps the last image.
    state.apply_binary(_preview_frame(b"no-image-here"))
    assert state.preview_frames == 5
    assert state.last_preview == JPEG
    assert state.preview_serial == serial + 2


def test_summary_names_the_nodes_that_sent_previews(api):
    view = api.LiveView("c", PROMPT)
    assert view.summary().endswith("preview_nodes=none")
    view._on_text(json.dumps(_text("executing", node="9")))
    view._on_binary(_preview_frame())
    view._on_binary(_preview_frame())
    view._on_text(json.dumps(_text("executing", node="7")))
    view._on_binary(_preview_frame())
    assert view.summary().endswith("preview_nodes=7:1,9:2")


def test_no_preview_path_means_nothing_is_written(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view = api.LiveView("c", PROMPT)
    view._on_binary(_preview_frame())
    view.stop()
    assert list(tmp_path.iterdir()) == []
    assert "note" not in view.snapshot()


def test_the_preview_file_is_the_newest_frame_and_never_half_written(api, tmp_path):
    target = tmp_path / "shots" / "live.jpg"
    view = api.LiveView("c", PROMPT, preview_path=str(target))
    view._on_binary(_preview_frame())
    assert target.read_bytes() == JPEG
    # Replaced in place: no scratch file is left beside it.
    assert [p.name for p in target.parent.iterdir()] == ["live.jpg"]


def test_a_png_preview_is_saved_as_png_whatever_name_was_asked_for(api, tmp_path):
    view = api.LiveView("c", PROMPT, preview_path=str(tmp_path / "live.jpg"))
    view._on_binary(_preview_frame(PNG, 2))
    assert (tmp_path / "live.png").read_bytes() == PNG
    assert not (tmp_path / "live.jpg").exists()


def test_the_preview_file_is_written_once_a_second_and_flushed_on_stop(api, tmp_path, monkeypatch):
    target = tmp_path / "live.jpg"
    monkeypatch.setattr(api, "_PREVIEW_WRITE_EVERY_S", 3600.0)
    view = api.LiveView("c", PROMPT, preview_path=str(target))
    view._on_binary(_preview_frame(JPEG + b"1"))
    assert target.read_bytes() == JPEG + b"1", "the first frame is written at once"
    view._on_binary(_preview_frame(JPEG + b"2"))
    view._on_binary(_preview_frame(JPEG + b"3"))
    assert target.read_bytes() == JPEG + b"1", "later frames wait out the pause"
    view.stop()
    assert target.read_bytes() == JPEG + b"3", "stop() writes the newest frame"
    view.stop()  # nothing newer: no second write, no error
    assert target.read_bytes() == JPEG + b"3"


def test_with_no_pause_every_frame_is_written(api, tmp_path, monkeypatch):
    target = tmp_path / "live.jpg"
    monkeypatch.setattr(api, "_PREVIEW_WRITE_EVERY_S", 0.0)
    view = api.LiveView("c", PROMPT, preview_path=str(target))
    for tag in (b"a", b"b", b"c"):
        view._on_binary(_preview_frame(JPEG + tag))
        assert target.read_bytes() == JPEG + tag


def test_a_failed_preview_write_is_reported_retried_and_never_raised(api, tmp_path, monkeypatch):
    target = tmp_path / "live.jpg"
    monkeypatch.setattr(api, "_PREVIEW_WRITE_EVERY_S", 0.0)
    view = api.LiveView("c", PROMPT, preview_path=str(target))
    view._note = "labels unreadable: earlier"
    real_replace = api.os.replace

    def viewer_has_the_file_open(source, destination):
        raise PermissionError("a viewer has the file open")

    monkeypatch.setattr(api.os, "replace", viewer_has_the_file_open)
    view._on_binary(_preview_frame())  # must not raise
    snap = view.snapshot()
    assert snap["note"] == (
        "labels unreadable: earlier; preview file not written: "
        "PermissionError: a viewer has the file open")
    assert "[labels unreadable: earlier; preview file not written" in api.format_live(snap)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "no half-written file is left behind"
    # The next frame tries again, and the note clears once a write lands.
    monkeypatch.setattr(api.os, "replace", real_replace)
    view._on_binary(_preview_frame(JPEG + b"again"))
    assert target.read_bytes() == JPEG + b"again"
    assert view.snapshot()["note"] == "labels unreadable: earlier"


def test_live_view_writes_the_preview_file_from_a_real_websocket(api, tmp_path):
    import aiohttp  # noqa: F401 -- a ComfyUI dependency; missing means FAIL, not skip
    target = tmp_path / "live.jpg"
    server = FakeComfy([
        ("text", _text("execution_start", prompt_id="p1")),
        ("text", _text("executing", node="9", prompt_id="p1")),
        ("binary", _preview_frame()),
    ]).start()
    view = api.LiveView("client-abc", PROMPT,
                        base_url="http://127.0.0.1:%d" % server.port,
                        preview_path=str(target))
    try:
        view.start()
        _wait_for(view, lambda s: s["preview_frames"] == 1)
    finally:
        view.stop()
        server.close()
    assert target.read_bytes() == JPEG
    assert view.summary().endswith("preview_nodes=9:1")


# --- the runner's --preview-file flag ---------------------------------------

def _recording_live(built):
    """Stands in for LiveView; ``built`` collects the extra arguments main()
    gave it, so a run without the flag is seen to pass none."""
    class _Live:
        def __init__(self, client_id, prompt, **kwargs):
            built.append(kwargs)

        def start(self):
            return self

        def set_prompt_id(self, prompt_id):
            pass

        def wrap(self, on_tick):
            return on_tick

        def stop(self):
            pass

        def summary(self):
            return "text_frames=0"
    return _Live


def _run_runner(canonical, monkeypatch, tmp_path, args):
    import contextlib
    import io
    sent, built = {}, []
    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(
        canonical, "submit_prompt",
        lambda prompt, client_id=None, **extra: sent.update(extra) or "prompt-live")
    monkeypatch.setattr(canonical, "LiveView", _recording_live(built))
    monkeypatch.setattr(canonical, "poll_history", lambda *a, **k: ("SUCCESS", ""))
    monkeypatch.setattr(canonical, "episode_of_prompt", lambda _pid: None)
    monkeypatch.delenv("OTR_NOTIFY_URL", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = canonical.main([*args, "--dump-prompt", str(tmp_path / "prompt.json")])
    return rc, buf.getvalue(), sent, built


@pytest.fixture(scope="module")
def canonical():
    import sys
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import otr_canonical_api_run
    return otr_canonical_api_run


def test_preview_plan_says_nothing_without_the_flag(canonical):
    assert canonical.preview_plan(None, None) == (None, None)
    assert canonical.preview_plan("", "latent2rgb") == (None, None)


def test_preview_plan_asks_for_latent2rgb_unless_the_operator_chose(canonical):
    method, line = canonical.preview_plan("C:/x/live.jpg", None)
    assert method == "latent2rgb"
    assert "preview file: on -> C:/x/live.jpg (or C:/x/live.png)" in line
    assert canonical.preview_plan("live.jpg", "   ")[0] == "latent2rgb"
    method, line = canonical.preview_plan("live.jpg", "TAESD")
    assert method is None and "OTR_PREVIEW_METHOD=taesd decides the frames" in line
    method, line = canonical.preview_plan("live.jpg", "none")
    assert method is None and "no file will appear" in line


def test_the_flag_reaches_the_live_view_and_the_submit(canonical, monkeypatch, tmp_path):
    monkeypatch.delenv("OTR_PREVIEW_METHOD", raising=False)
    shot = str(tmp_path / "live.jpg")
    rc, out, sent, built = _run_runner(canonical, monkeypatch, tmp_path,
                                       ["--preview-file", shot])
    assert rc == 0
    assert sent == {"preview_method": "latent2rgb"}
    assert built == [{"preview_path": shot}]
    assert "[canonical-api] preview file: on ->" in out


def test_the_operators_preview_method_stands_when_the_flag_is_given(canonical, monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_PREVIEW_METHOD", "taesd")
    shot = str(tmp_path / "live.jpg")
    rc, out, sent, built = _run_runner(canonical, monkeypatch, tmp_path,
                                       ["--preview-file", shot])
    assert rc == 0
    assert sent == {}, "the env var is read by submit_prompt, not overridden here"
    assert built == [{"preview_path": shot}]
    assert "OTR_PREVIEW_METHOD=taesd decides the frames" in out


def test_a_run_without_the_flag_submits_and_listens_as_it_always_did(canonical, monkeypatch, tmp_path):
    monkeypatch.delenv("OTR_PREVIEW_METHOD", raising=False)
    rc, out, sent, built = _run_runner(canonical, monkeypatch, tmp_path, [])
    assert rc == 0
    assert sent == {} and built == [{}]
    assert "preview file" not in out


# --- submit_prompt takes the method as an argument too ----------------------

def test_the_preview_method_argument_wins_over_the_environment(api, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json=json) or _Resp())
    monkeypatch.delenv("OTR_COMFY_API_KEY", raising=False)
    monkeypatch.setenv("OTR_PREVIEW_METHOD", "taesd")
    api.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, preview_method="Latent2RGB")
    assert captured["json"]["extra_data"]["preview_method"] == "latent2rgb"
    api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})
    assert captured["json"]["extra_data"]["preview_method"] == "taesd"


def test_a_mistyped_preview_method_argument_fails_loudly(api, monkeypatch):
    monkeypatch.delenv("OTR_PREVIEW_METHOD", raising=False)
    with pytest.raises(ValueError, match="preview_method='latent2rbg'"):
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, preview_method="latent2rbg")
