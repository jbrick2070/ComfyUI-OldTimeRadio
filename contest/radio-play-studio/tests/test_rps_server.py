"""The local web app, end to end over real HTTP with the SDK faked."""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from radio_play_studio.runner import Studio
from radio_play_studio.server import make_server

from test_rps_runner import FakeClient, _good_job  # noqa: E402 - sibling test module


@pytest.fixture
def app(settings, schemas):
    studio = Studio(settings, client_factory=lambda url: FakeClient(_good_job()), fetch=lambda url: schemas)
    server = make_server(studio, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], settings
    server.shutdown()
    server.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def _post(port, choices, content_type="application/json"):
    return _request(port, "POST", "/api/episodes", json.dumps({"choices": choices}),
                    {"Content-Type": content_type})


def _events(port, run_id):
    status, headers, data = _request(port, "GET", f"/api/episodes/{run_id}/events")
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    return [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith("data: ")]


def test_the_page_is_served(app):
    port, _ = app
    status, _, body = _request(port, "GET", "/")
    assert status == 200 and b"Radio Play Studio" in body


def test_options_list_the_form(app):
    port, _ = app
    status, _, body = _request(port, "GET", "/api/options")
    keys = [f["key"] for f in json.loads(body)["fields"]]
    assert status == 200 and keys[0] == "bank" and "acts" in keys


def test_a_whole_episode_over_http(app):
    port, settings = app
    status, _, body = _post(port, {"bank": "shakespeare", "acts": 2})
    assert status == 202
    events = _events(port, json.loads(body)["id"])
    assert [e["type"] for e in events][0] == "submitted" and events[-1]["type"] == "done"
    done = events[-1]
    status, headers, video = _request(port, "GET", f"/episodes/{done['run']}/{done['file']}")
    assert status == 200 and video == b"video-bytes" and headers["Accept-Ranges"] == "bytes"
    status, headers, part = _request(port, "GET", f"/episodes/{done['run']}/{done['file']}",
                                     headers={"Range": "bytes=2-5"})
    assert (status, part, headers["Content-Range"]) == (206, b"deo-", "bytes 2-5/11")
    status, headers, tail = _request(port, "GET", f"/episodes/{done['run']}/{done['file']}",
                                     headers={"Range": "bytes=-3"})
    assert (status, tail) == (206, b"tes")
    status, _, _ = _request(port, "GET", f"/episodes/{done['run']}/{done['file']}",
                            headers={"Range": "bytes=99-"})
    assert status == 416


def test_a_bad_choice_is_a_400_before_any_run_starts(app):
    port, _ = app
    status, _, body = _post(port, {"bank": "not a bank"})
    assert status == 400 and "Story bank" in json.loads(body)["error"]
    status, _, body = _post(port, {"vram": 8})
    assert status == 400 and "unknown field" in json.loads(body)["error"]


def test_only_json_posts_are_taken(app):
    port, _ = app
    status, _, _ = _post(port, {}, content_type="application/x-www-form-urlencoded")
    assert status == 415
    status, _, _ = _request(port, "POST", "/api/episodes", "[1]", {"Content-Type": "application/json"})
    assert status == 400


def test_a_request_for_another_host_is_refused(app):
    port, _ = app
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("GET", "/api/options", skip_host=True)
    conn.putheader("Host", "evil.example")
    conn.endheaders()
    assert conn.getresponse().status == 403
    conn.close()


def test_files_outside_the_episode_folder_are_not_served(app, tmp_path):
    port, settings = app
    (settings.out_dir).mkdir(parents=True, exist_ok=True)
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    for path in ("/episodes/../secret.txt", "/episodes/%2e%2e/secret.txt", "/episodes//etc/passwd"):
        status, _, body = _request(port, "GET", path)
        assert status == 404 and b"nope" not in body


def test_an_unknown_episode_is_a_404(app):
    port, _ = app
    status, _, _ = _request(port, "GET", "/api/episodes/000000000000/events")
    assert status == 404
