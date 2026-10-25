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
