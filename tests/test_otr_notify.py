"""The phone ping: scripts/otr_notify.py, the canonical runner's end-of-leg ping,
and the watchdog's DEAD ping. (PLAN row 1.3, planning/creatoros-concepts/PLAN.md.)

Every network test talks to a loopback HTTP stub that records what an ntfy
server would receive. The rules under test: it is never a hidden switch, it
reports and never decides, "published" is said only for an obs file confirmed
on disk, and the topic (a secret) is never printed.
"""
from __future__ import annotations

import base64
import contextlib
import http.server
import io
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import otr_canonical_api_run as canonical  # noqa: E402
import otr_notify  # noqa: E402

TOPIC = "secret-topic-4f9c1a"


class NtfyStub:
    """Records every request; answers with ``status``, after ``delay`` seconds,
    and with a ``Location`` header when ``location`` is given."""

    def __init__(self, status=200, delay=0.0, location=None):
        self.received: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.received.append({
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": self.rfile.read(length),
                })
                if delay:
                    time.sleep(delay)
                try:
                    self.send_response(status)
                    if location:
                        self.send_header("Location", location)
                    self.end_headers()
                    self.wfile.write(b"{}")
                except OSError:
                    pass  # the caller gave up on us (the deadline test)

            do_POST = _answer  # noqa: N815 -- http.server's names
            do_GET = _answer   # noqa: N815

            def log_message(self, *args):  # silence the test output
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return "http://127.0.0.1:%d/%s" % (self.server.server_address[1], TOPIC)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture(autouse=True)
def _notify_off_by_default(monkeypatch):
    monkeypatch.delenv(otr_notify.NOTIFY_ENV, raising=False)


# --- scripts/otr_notify.py ---------------------------------------------------

def test_state_line_says_off_when_unset():
    assert otr_notify.state_line() == "notify: off (OTR_NOTIFY_URL unset)"


def test_state_line_names_the_host_and_never_the_topic(monkeypatch):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "https://ntfy.example.org/" + TOPIC)
    line = otr_notify.state_line()
    assert line == "notify: on -> ntfy.example.org"
    assert TOPIC not in line


def test_notify_posts_the_message_to_the_topic(monkeypatch):
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        assert otr_notify.notify("OTR episode published", "ep_20261024_120000\nfile.mp4",
                                 priority="high") is True
    (request,) = stub.received
    assert request["path"] == "/" + TOPIC
    assert request["headers"]["title"] == "OTR episode published"
    assert request["headers"]["priority"] == "high"
    assert request["headers"]["content-type"] == "text/plain; charset=utf-8"
    assert request["body"].decode("utf-8") == "ep_20261024_120000\nfile.mp4"


def test_a_non_ascii_title_does_not_stop_the_ping(monkeypatch):
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        assert otr_notify.notify("Episode \u00e9\u4e2d done", "body \u00e9\u4e2d") is True
    (request,) = stub.received
    assert request["headers"]["title"] == "Episode ?? done"
    assert request["body"].decode("utf-8") == "body \u00e9\u4e2d"


def test_notify_does_nothing_and_says_nothing_when_off(capsys):
    assert otr_notify.notify("t", "b") is False
    assert capsys.readouterr().out == ""


def test_a_server_error_is_reported_and_returns_false(monkeypatch, capsys):
    with NtfyStub(status=500) as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert "[notify] FAILED:" in out
    assert TOPIC not in out


def test_an_unreachable_server_is_reported_without_the_topic(monkeypatch, capsys):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://127.0.0.1:1/" + TOPIC)
    assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert "[notify] FAILED:" in out
    assert TOPIC not in out


def test_cli_prints_its_state_and_exits_nonzero_when_off(capsys):
    assert otr_notify.main([]) == 2
    assert "notify: off" in capsys.readouterr().out


def test_cli_sends_a_test_ping(monkeypatch, capsys):
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        assert otr_notify.main(["--title", "T", "--body", "B"]) == 0
    assert stub.received[0]["headers"]["title"] == "T"
    assert "[notify] sent" in capsys.readouterr().out


def test_a_blank_value_is_off_here_as_in_the_watchdog(monkeypatch, capsys):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "   ")
    assert otr_notify.state_line() == "notify: off (OTR_NOTIFY_URL unset)"
    assert otr_notify.notify("t", "b") is False
    assert capsys.readouterr().out == ""


def test_state_line_never_shows_a_password_or_the_topic(monkeypatch):
    monkeypatch.setenv(
        otr_notify.NOTIFY_ENV, "https://alice:hunter2@ntfy.example.org:8443/" + TOPIC)
    line = otr_notify.state_line()
    assert line == "notify: on -> ntfy.example.org:8443"
    for secret in ("alice", "hunter2", TOPIC):
        assert secret not in line


@pytest.mark.parametrize("unusable", [
    "ntfy.example.org/topic",            # no scheme
    "ftp://ntfy.example.org/topic",      # not http(s)
    "http://",                           # no host
    "http://[bad/topic",                 # urlsplit itself raises
    "http://host:notaport/topic",        # a port that is not a number
    '"https://ntfy.example.org/topic"',  # quotes that came along with `set X="..."`
])
def test_an_unusable_url_is_named_not_raised_and_not_echoed(monkeypatch, capsys, unusable):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, unusable)
    state = otr_notify.state_line()
    assert state.startswith("notify: ON BUT UNUSABLE")
    assert "no ping will be sent" in state
    assert "topic" not in state
    assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert out.strip() == (
        "[notify] FAILED: OTR_NOTIFY_URL is not an http:// or https:// URL with a host")
    assert "topic" not in out


def test_a_url_with_credentials_is_sent_as_a_basic_header_not_in_the_url(monkeypatch, capsys):
    with NtfyStub() as stub:
        host_and_topic = stub.url.split("//", 1)[1]            # 127.0.0.1:PORT/TOPIC
        # Percent-escapes in the userinfo are decoded before they are encoded.
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://al%40ice:p%3Ass@" + host_and_topic)
        assert otr_notify.notify("t", "b") is True
    (request,) = stub.received
    assert request["path"] == "/" + TOPIC
    assert request["headers"]["authorization"] == (
        "Basic " + base64.b64encode(b"al@ice:p:ss").decode("ascii"))
    assert capsys.readouterr().out == ""


def test_a_failed_ping_never_prints_credentials_or_the_topic(monkeypatch, capsys):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://alice:hunter2@127.0.0.1:1/" + TOPIC)
    assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert "[notify] FAILED:" in out
    for secret in ("alice", "hunter2", TOPIC):
        assert secret not in out


def test_an_error_that_quotes_the_request_path_shows_its_type_alone(monkeypatch, capsys):
    """`http.client` refuses a space in the request path and QUOTES the path in
    its message -- and the path is the topic."""
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url + " tail")
        assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert "[notify] FAILED: InvalidURL (details withheld" in out
    assert TOPIC not in out
    assert stub.received == []


# --- scrubbing, redirects, the deadline -------------------------------------

def test_a_network_error_that_quotes_the_url_is_scrubbed(monkeypatch, capsys):
    """Some OSErrors quote what they were given. The topic and the credentials
    are replaced before anything is printed; without the scrub this prints them."""
    class _Quoting:
        def open(self, request, timeout=None):
            raise OSError("cannot reach %s (user alice, password hunter2, topic %s)"
                          % (request.full_url, TOPIC))

    monkeypatch.setattr(otr_notify, "_OPENER", _Quoting())
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://alice:hunter2@127.0.0.1:9/" + TOPIC)
    assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert out.strip() == (
        "[notify] FAILED: OSError: cannot reach <hidden> (user <hidden>, "
        "password <hidden>, topic <hidden>)")
    for secret in ("alice", "hunter2", TOPIC):
        assert secret not in out


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_a_redirect_is_a_failure_and_is_never_followed(monkeypatch, capsys, code):
    """urllib would answer a 301/302/303 to a POST with a bodiless GET, which the
    target answers 200: reported as sent, never delivered."""
    with NtfyStub() as landing:
        with NtfyStub(status=code, location=landing.url) as moved:
            monkeypatch.setenv(otr_notify.NOTIFY_ENV, moved.url)
            assert otr_notify.notify("t", "b") is False
    out = capsys.readouterr().out
    assert "[notify] FAILED: HTTP %d: the server redirected the message" % code in out
    assert "put the final URL" in out
    assert TOPIC not in out
    assert [r["method"] for r in moved.received] == ["POST"]
    assert landing.received == [], "the redirect target was never contacted"


def test_a_server_that_never_answers_cannot_hold_the_end_of_a_run(monkeypatch, capsys):
    monkeypatch.setattr(otr_notify, "_DEADLINE_S", 0.4)
    with NtfyStub(delay=3) as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        started = time.monotonic()
        assert otr_notify.notify("t", "b") is False
        took = time.monotonic() - started
    assert took < 2.0
    out = capsys.readouterr().out
    assert "[notify] FAILED: no answer within 0.4 s" in out
    assert TOPIC not in out


def test_a_worker_that_ends_without_an_answer_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(otr_notify, "_post", lambda request, outcome: None)
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://127.0.0.1:9/" + TOPIC)
    assert otr_notify.notify("t", "b") is False
    assert "[notify] FAILED: the send ended without an answer" in capsys.readouterr().out


# --- the runner's end-of-leg ping -------------------------------------------

def test_compose_success_with_a_confirmed_obs_file():
    title, body, priority = canonical.compose_result_ping(
        "SUCCESS", "", "tale_20261024_101500", "C:/x/otr/obs/tale_20261024_101500__a_final.mp4",
        "", 3725)
    assert title == "OTR episode published"
    assert body == "tale_20261024_101500\ntale_20261024_101500__a_final.mp4\ntook 1h02m"
    assert priority is None
    assert "C:/x" not in body, "no local path beyond the file name"


def test_compose_success_with_no_obs_file_pings_at_high_priority():
    title, body, priority = canonical.compose_result_ping(
        "SUCCESS", "", None, None, "the obs file is zero bytes", 45)
    assert title == "OTR leg finished, nothing in obs"
    assert body == "episode id not read\nthe obs file is zero bytes\ntook 45s"
    assert priority == "high"


def test_compose_failure_carries_the_headline_and_the_hint_not_the_traceback():
    err = ("node 9 (OTR_VideoRenderBatch) raised torch.OutOfMemoryError\n"
           "Allocation on device C:\\Users\\someone\\secret\\path\n"
           "-- hint: the GPU ran out of memory: close other GPU programs\n"
           "traceback (last 2 frames):\n  File \"a.py\"")
    title, body, priority = canonical.compose_result_ping(
        "FAIL", err, "tale_20261024_101500", None, "", 725)
    assert title == "OTR leg FAILED"
    assert body.splitlines() == [
        "node 9 (OTR_VideoRenderBatch) raised torch.OutOfMemoryError",
        "-- hint: the GPU ran out of memory: close other GPU programs",
        "after 12m05s",
    ]
    assert priority == "high"
    assert "someone" not in body


def test_compose_failure_finds_a_hint_that_ends_a_line_and_keeps_it_out_of_the_headline():
    """A refusal, a SERVER-GONE message and the raw fallback end their line with
    the hint instead of giving it a line of its own."""
    err = ("SERVER-GONE: the ComfyUI server at http://127.0.0.1:8000 stopped "
           "answering -- hint: the ComfyUI server is gone or was restarted: "
           "read the watchdog verdict")
    title, body, priority = canonical.compose_result_ping("FAIL", err, "ep", None, "", 61)
    assert body.splitlines() == [
        "SERVER-GONE: the ComfyUI server at http://127.0.0.1:8000 stopped answering",
        "-- hint: the ComfyUI server is gone or was restarted: read the watchdog verdict",
        "after 1m01s",
    ]
    assert priority == "high"


def test_compose_failure_without_a_hint_or_text_is_still_a_message():
    _, body, _ = canonical.compose_result_ping("FAIL", "", None, None, "", 5)
    assert body.splitlines() == ["no error text", "after 5s"]
    _, body, _ = canonical.compose_result_ping("FAIL", "Boom: it broke", None, None, "", 5)
    assert body.splitlines() == ["Boom: it broke", "after 5s"]


def test_compose_timeout_says_the_render_may_still_be_running():
    title, body, priority = canonical.compose_result_ping("TIMEOUT", "", None, None, "", 5400)
    assert title == "OTR watch timed out"
    assert "may still be running" in body and "1h30m" in body
    assert priority is None


def _history(prompt_id, node_id, text):
    return {prompt_id: {"outputs": {node_id: {"text": [text]}}}}


MUX_PROMPT = {"85": {"class_type": "OTR_MasterAudioMux", "inputs": {}},
              "1": {"class_type": "OTR_LedgerScriptWriter", "inputs": {}}}


def test_obs_file_is_confirmed_only_when_it_exists_and_is_not_empty(tmp_path):
    good = tmp_path / "ep__style_final.mp4"
    good.write_bytes(b"x" * 10)
    path, why = canonical.obs_file_from_history(MUX_PROMPT, _history("p", "85", str(good)), "p")
    assert (path, why) == (str(good), "")

    empty = tmp_path / "empty_final.mp4"
    empty.write_bytes(b"")
    assert canonical.obs_file_from_history(
        MUX_PROMPT, _history("p", "85", str(empty)), "p") == (None, "the obs file is zero bytes")

    gone = tmp_path / "gone_final.mp4"
    path, why = canonical.obs_file_from_history(MUX_PROMPT, _history("p", "85", str(gone)), "p")
    assert path is None and "cannot see" in why


def test_an_unpublished_verdict_is_passed_on_without_its_path():
    path, why = canonical.obs_file_from_history(
        MUX_PROMPT,
        _history("p", "85", "not published -- archived at C:\\private\\ep_final.mp4"), "p")
    assert path is None
    assert why == "not published -- archived"


def test_no_mux_node_or_no_output_is_named():
    path, why = canonical.obs_file_from_history({"1": {"class_type": "X"}}, {}, "p")
    assert path is None and "no OTR_MasterAudioMux node" in why
    path, why = canonical.obs_file_from_history(MUX_PROMPT, {"p": {"outputs": {}}}, "p")
    assert (path, why) == (None, "the mux node reported no obs path")
    path, why = canonical.obs_file_from_history(MUX_PROMPT, None, "p")
    assert (path, why) == (None, "the mux node reported no obs path")


def test_announce_is_silent_and_inert_when_notifications_are_off(monkeypatch, capsys):
    monkeypatch.setattr(canonical, "published_obs_file",
                        lambda *a: pytest.fail("must not look for obs when notify is off"))
    canonical.announce_result(MUX_PROMPT, "p", "SUCCESS", "", "ep", 10)
    assert capsys.readouterr().out == ""


def test_announce_sends_one_ping_for_a_confirmed_success(monkeypatch, tmp_path, capsys):
    obs = tmp_path / "tale_20261024_101500__a_final.mp4"
    obs.write_bytes(b"x")
    monkeypatch.setattr(canonical, "published_obs_file", lambda *a: (str(obs), ""))
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        canonical.announce_result(MUX_PROMPT, "p", "SUCCESS", "", "tale_20261024_101500", 61)
    (request,) = stub.received
    assert request["headers"]["title"] == "OTR episode published"
    assert obs.name in request["body"].decode("utf-8")
    assert "[canonical-api] notify sent: OTR episode published" in capsys.readouterr().out


def test_a_failed_ping_changes_nothing_about_the_run(monkeypatch, capsys):
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://127.0.0.1:1/" + TOPIC)
    canonical.announce_result(MUX_PROMPT, "p", "FAIL", "boom", None, 5)
    out = capsys.readouterr().out
    assert "notify NOT sent: OTR leg FAILED" in out
    assert TOPIC not in out


def test_a_bug_inside_announce_is_reported_and_never_raised(monkeypatch, capsys):
    monkeypatch.setattr(canonical, "published_obs_file",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("lookup broke")))
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://127.0.0.1:1/" + TOPIC)
    canonical.announce_result(MUX_PROMPT, "p", "SUCCESS", "", "ep", 5)
    assert "notify skipped: RuntimeError: lookup broke" in capsys.readouterr().out


class _StubLive:
    """Stands in for LiveView so main() never opens a socket in a unit test."""

    last = None  # the most recent instance, so a test can read what main() did to it

    def __init__(self, client_id, prompt):
        self.client_id = client_id
        self.prompt_id = None
        self.stopped = False
        type(self).last = self

    def start(self):
        return self

    def set_prompt_id(self, prompt_id):
        self.prompt_id = prompt_id

    def wrap(self, on_tick):
        def tick(elapsed_s, status):
            status["live"] = {"state": "running", "node": "7",
                              "label": "Rendering the video", "value": 3, "max": 10}
            on_tick(elapsed_s, status)
        return tick

    def stop(self):
        self.stopped = True

    def summary(self):
        return "text_frames=4 preview_frames=0 last_preview_bytes=0 other_binary_frames=0"


def _run_main(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = canonical.main(args)
    return rc, buf.getvalue()


def test_the_runner_prints_the_notify_state_the_live_view_and_pings_once(monkeypatch, tmp_path):
    obs = tmp_path / "tale_20261024_101500__a_final.mp4"
    obs.write_bytes(b"x")
    seen = {}

    def fake_submit(prompt, client_id=None):
        seen["client_id"] = client_id
        return "prompt-live"

    def fake_poll(prompt_id, timeout_s, poll_s, on_tick=None):
        on_tick(0.0, {})
        on_tick(5.9, {"status_str": "running"})
        return "SUCCESS", ""

    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: (MUX_PROMPT, []))
    monkeypatch.setattr(canonical, "LiveView", _StubLive)
    monkeypatch.setattr(canonical, "submit_prompt", fake_submit)
    monkeypatch.setattr(canonical, "poll_history", fake_poll)
    monkeypatch.setattr(canonical, "episode_of_prompt", lambda _pid: "tale_20261024_101500")
    monkeypatch.setattr(canonical, "published_obs_file", lambda *a: (str(obs), ""))
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        rc, out = _run_main(["--dump-prompt", str(tmp_path / "prompt.json")])
    assert rc == 0
    assert "[canonical-api] notify: on -> 127.0.0.1:" in out
    assert TOPIC not in out
    # The SAME client id reaches the submit (the listener's socket id).
    assert seen["client_id"]
    # The heartbeat keeps its `t=<N>s ... status=` prefix; live text follows it.
    assert ('t=5s prompt_id=prompt-live status=running '
            'live="node 7 Rendering the video 3/10"') in out
    assert "[canonical-api] LIVE text_frames=4 preview_frames=0" in out
    assert out.index("RESULT SUCCESS") < out.index("notify sent: OTR episode published")
    assert len(stub.received) == 1
    assert _StubLive.last.prompt_id == "prompt-live", "the view adopted the prompt id"
    assert _StubLive.last.stopped, "the view was stopped"


def test_the_runner_exit_code_does_not_depend_on_the_ping(monkeypatch, tmp_path):
    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _StubLive)
    monkeypatch.setattr(canonical, "submit_prompt", lambda p, client_id=None: "prompt-live")
    monkeypatch.setattr(canonical, "poll_history",
                        lambda *a, **k: ("FAIL", "node 1 (X) raised Boom"))
    monkeypatch.setattr(canonical, "episode_of_prompt", lambda _pid: None)
    monkeypatch.setenv(otr_notify.NOTIFY_ENV, "http://127.0.0.1:1/" + TOPIC)
    rc, out = _run_main(["--dump-prompt", str(tmp_path / "prompt.json")])
    assert rc == 1
    assert "notify NOT sent: OTR leg FAILED" in out
    assert "notify: off" not in out


def test_a_refused_submission_pings_then_raises_unchanged(monkeypatch, tmp_path):
    """The commonest unattended failure ends before any prompt id exists: a
    bad dropdown value, a server that is not up. It must reach the phone, and
    the exception must still be the one the caller would have seen."""
    refusal = RuntimeError(
        "POST /prompt -> HTTP 400: {\"error\": {}} -- hint: ComfyUI does not list "
        "that option or model file")

    def refuse(prompt, client_id=None):
        raise refusal

    live = []

    class _TrackedLive(_StubLive):
        def stop(self):
            live.append("stopped")

    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _TrackedLive)
    monkeypatch.setattr(canonical, "submit_prompt", refuse)
    monkeypatch.setattr(canonical, "poll_history",
                        lambda *a, **k: pytest.fail("a refused prompt is never polled"))
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        with pytest.raises(RuntimeError) as caught:
            _run_main(["--dump-prompt", str(tmp_path / "prompt.json")])
    assert caught.value is refusal
    assert live == ["stopped"], "the listener is still stopped"
    assert _TrackedLive.last.prompt_id is None, "no prompt id was adopted"
    (request,) = stub.received
    assert request["headers"]["title"] == "OTR leg FAILED"
    assert request["headers"]["priority"] == "high"
    lines = request["body"].decode("utf-8").splitlines()
    assert lines[0].startswith(
        "RuntimeError before the prompt was queued: POST /prompt -> HTTP 400")
    assert lines[1] == "-- hint: ComfyUI does not list that option or model file"
    assert lines[2].startswith("after ")


def test_a_crash_while_watching_pings_then_raises_unchanged(monkeypatch, tmp_path):
    def crash(*args, **kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _StubLive)
    monkeypatch.setattr(canonical, "submit_prompt", lambda p, client_id=None: "prompt-live")
    monkeypatch.setattr(canonical, "poll_history", crash)
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        with pytest.raises(OSError, match="disk went away"):
            _run_main(["--dump-prompt", str(tmp_path / "prompt.json")])
    (request,) = stub.received
    assert request["body"].decode("utf-8").splitlines()[0] == (
        "OSError while watching the render: disk went away")
    assert _StubLive.last.stopped, "the view was stopped"


def test_ctrl_c_while_watching_sends_no_ping_and_still_stops_the_view(monkeypatch, tmp_path):
    """An operator's Ctrl-C is not a failure to report."""
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _StubLive)
    monkeypatch.setattr(canonical, "submit_prompt", lambda p, client_id=None: "prompt-live")
    monkeypatch.setattr(canonical, "poll_history", interrupt)
    with NtfyStub() as stub:
        monkeypatch.setenv(otr_notify.NOTIFY_ENV, stub.url)
        with pytest.raises(KeyboardInterrupt):
            _run_main(["--dump-prompt", str(tmp_path / "prompt.json")])
    assert stub.received == []
    assert _StubLive.last.stopped, "the view was stopped"


def test_a_refused_submission_with_notify_off_just_raises(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _StubLive)

    def refuse(prompt, client_id=None):
        raise RuntimeError("refused")

    monkeypatch.setattr(canonical, "submit_prompt", refuse)
    with pytest.raises(RuntimeError, match="refused"):
        canonical.main(["--dump-prompt", str(tmp_path / "prompt.json")])
    assert "notify sent" not in capsys.readouterr().out


# --- the watchdog's DEAD ping (a PowerShell harness) -------------------------

def _run_watchdog(tmp_path, notify_url=None, extra=()):
    leg_log = tmp_path / "leg.log"
    leg_log.write_text(
        "[canonical-api] t=5s prompt_id=p status=running\n", encoding="utf-8")
    env = dict(os.environ)
    env.pop(otr_notify.NOTIFY_ENV, None)
    if notify_url:
        env[otr_notify.NOTIFY_ENV] = notify_url
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(SCRIPTS / "otr_render_watchdog.ps1"),
         "-LegLog", str(leg_log), "-PollSeconds", "0", "-StallSeconds", "0",
         "-QueueFails", "99", "-QueueUrl", "http://127.0.0.1:1/queue", *extra],
        check=False, capture_output=True, text=True, timeout=60, env=env)
    return completed


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_pings_when_it_declares_a_leg_dead(tmp_path):
    with NtfyStub() as stub:
        completed = _run_watchdog(tmp_path, stub.url)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify: on -> 127.0.0.1:" in completed.stdout
    assert "notify sent" in completed.stdout
    assert TOPIC not in completed.stdout
    (request,) = stub.received
    assert request["headers"]["title"] == "OTR leg dead: stalled"
    body = request["body"].decode("utf-8")
    assert body.startswith("leg dead: heartbeat stalled")
    assert body.endswith("leg.log")
    assert str(tmp_path) not in body, "the leg log's folder is a local path"


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_verdict_is_unchanged_by_a_failed_ping(tmp_path):
    completed = _run_watchdog(tmp_path, "http://127.0.0.1:1/" + TOPIC)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify FAILED" in completed.stdout
    assert TOPIC not in completed.stdout
    assert "DEAD" in (tmp_path / "leg.log.watchdog").read_text(encoding="ascii")


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_says_the_ping_is_off_when_unset(tmp_path):
    completed = _run_watchdog(tmp_path)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify: off (OTR_NOTIFY_URL unset)" in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_reads_a_blank_url_as_off_like_the_runner_does(tmp_path):
    completed = _run_watchdog(tmp_path, "   ")
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify: off (OTR_NOTIFY_URL unset)" in completed.stdout
    assert "notify sent" not in completed.stdout and "notify FAILED" not in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_sends_credentials_as_a_basic_header_and_never_prints_them(tmp_path):
    with NtfyStub() as stub:
        host_and_topic = stub.url.split("//", 1)[1]
        completed = _run_watchdog(tmp_path, "http://alice:hunter2@" + host_and_topic)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify sent" in completed.stdout
    (request,) = stub.received
    assert request["path"] == "/" + TOPIC
    assert request["headers"]["authorization"] == (
        "Basic " + base64.b64encode(b"alice:hunter2").decode("ascii"))
    for secret in ("alice", "hunter2", TOPIC):
        assert secret not in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_failure_line_hides_credentials_and_the_topic(tmp_path):
    completed = _run_watchdog(tmp_path, "http://alice:hunter2@127.0.0.1:1/" + TOPIC)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify FAILED" in completed.stdout
    for secret in ("alice", "hunter2", TOPIC):
        assert secret not in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
@pytest.mark.parametrize("code", [301, 302, 307])
def test_watchdog_never_follows_a_redirect(tmp_path, code):
    with NtfyStub() as landing:
        with NtfyStub(status=code, location=landing.url) as moved:
            completed = _run_watchdog(tmp_path, moved.url)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "notify FAILED" in completed.stdout
    assert "notify sent" not in completed.stdout
    assert "put the final URL" in completed.stdout
    assert TOPIC not in completed.stdout
    assert [r["method"] for r in moved.received] == ["POST"]
    assert landing.received == [], "the redirect target was never contacted"


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
@pytest.mark.parametrize("unusable", [
    "ntfy.example.org/topic",
    "ftp://ntfy.example.org/topic",
    "http://",
    '"https://ntfy.example.org/topic"',
])
def test_watchdog_names_an_unusable_url_loudly(tmp_path, unusable):
    completed = _run_watchdog(tmp_path, unusable)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert ("notify: ON BUT UNUSABLE (OTR_NOTIFY_URL is not an http:// or https:// "
            "URL with a host), no ping will be sent") in completed.stdout
    assert otr_notify._UNUSABLE_STATE in completed.stdout, "same wording as otr_notify"
    assert "notify sent" not in completed.stdout
    assert "topic" not in completed.stdout
