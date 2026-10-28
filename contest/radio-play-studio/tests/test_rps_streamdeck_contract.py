"""What the Stream Deck plugin (../streamdeck) asks this server, asked of the
real server: the same requests the plugin's client sends, over real HTTP, with
the SDK faked as test_rps_server does. The plugin's own tests run against a
stand-in of this server; these hold the real one to the same answers, so a
change here is felt here first. streamdeck/API-CONTRACT.md is the prose form."""
from __future__ import annotations

import http.client
import json
import pathlib
import re
import threading
import time
from urllib.parse import quote

import pytest

from radio_play_studio import runner
from radio_play_studio.form import FIELDS_BY_KEY, MY_STORY_BANK
from radio_play_studio.runner import Studio
from radio_play_studio.server import make_server

from test_rps_runner import FakeClient, _good_job  # noqa: E402 - sibling test module

PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "streamdeck"
CONTRACT_TS = (PLUGIN / "src" / "types" / "contract.ts").read_text(encoding="utf-8")
SETTINGS_TS = (PLUGIN / "src" / "types" / "settings.ts").read_text(encoding="utf-8")
CHOICE_MATH_TS = (PLUGIN / "src" / "actions" / "choice-math.ts").read_text(encoding="utf-8")
CLIENT_TS = (PLUGIN / "src" / "studio" / "studio-client.ts").read_text(encoding="utf-8")
STORE_TS = (PLUGIN / "src" / "state" / "studio-store.ts").read_text(encoding="utf-8")
RUNNER_PY = pathlib.Path(runner.__file__).read_text(encoding="utf-8")

#: The four fields the deck's dials post (settings.ts DIAL_FIELDS).
DIAL_FIELDS = re.findall(r'"([a-z_]+)"', re.search(r"DIAL_FIELDS = \[([^\]]+)\]", SETTINGS_TS).group(1))


@pytest.fixture
def app(settings, schemas):
    studio = Studio(settings, client_factory=lambda url: FakeClient(_good_job()), fetch=lambda url: schemas)
    server = make_server(studio, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _request(port, method, path, body=None, headers=None):
    """A request as node:http sends one to 127.0.0.1: Host 127.0.0.1:<port>,
    and for a body a JSON content type and a declared length."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def _post_json(port, path, payload):
    raw = json.dumps(payload).encode("utf-8")
    return _request(port, "POST", path, raw,
                    {"Content-Type": "application/json", "Content-Length": str(len(raw))})


def _stream(port, run_id, last_event_id=None):
    """The (id, event) pairs the event stream sends, as the plugin's SSE
    parser reads them: `id:` then `data:`, keep-alive comments skipped."""
    headers = {"Accept": "text/event-stream"}
    if last_event_id is not None:
        headers["Last-Event-ID"] = str(last_event_id)
    status, resp_headers, data = _request(port, "GET", f"/api/episodes/{run_id}/events", headers=headers)
    assert status == 200 and resp_headers["Content-Type"] == "text/event-stream"
    out, current = [], None
    for line in data.decode("utf-8").splitlines():
        if line.startswith("id: "):
            current = int(line[4:])
        elif line.startswith("data: "):
            out.append((current, json.loads(line[6:])))
    return out


def _deck_choices(port):
    """What the deck posts for an untouched set of dials: each dial field the
    form has, at the form's default."""
    status, _, body = _request(port, "GET", "/api/options", headers={"Accept": "application/json"})
    assert status == 200
    form = json.loads(body)
    fields = {f["key"]: f for f in form["fields"]}
    return {k: fields[k]["default"] for k in DIAL_FIELDS if k in fields}, form


def _until_finished(port, run_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        _, _, body = _request(port, "GET", "/api/episodes")
        row = next(r for r in json.loads(body)["episodes"] if r["id"] == run_id)
        if row["finished"]:
            return row
        time.sleep(0.05)
    raise AssertionError("the run never finished")


def test_the_deck_is_answered_with_the_host_node_sends(app):
    status, _, _ = _request(app, "GET", "/api/episodes")
    assert status == 200


@pytest.mark.parametrize("host", ["studio.local:8765", "192.168.1.9:8765"])
def test_a_foreign_host_is_still_refused(app, host):
    status, _, body = _request(app, "GET", "/api/episodes", headers={"Host": host})
    assert status == 403 and json.loads(body)["error"]


def test_the_form_has_every_field_the_dials_post(app):
    assert DIAL_FIELDS == ["language", "bank", "style", "acts"]
    for key in DIAL_FIELDS:
        assert key in FIELDS_BY_KEY, key
    choices, form = _deck_choices(app)
    assert set(choices) == set(DIAL_FIELDS)
    for row in form["fields"]:
        for key in ("key", "label", "group", "help", "kind", "default"):
            assert key in row, (row.get("key"), key)
    # The bank the dial skips is the one that takes typed text.
    assert re.search(r'MY_STORY_BANK = "([a-z_]+)"', CHOICE_MATH_TS).group(1) == MY_STORY_BANK
    # `workflow` routes the run; it is not a form field.
    assert runner.WORKFLOW_CHOICE == "workflow" and "workflow" not in FIELDS_BY_KEY


def test_the_deck_post_starts_an_episode_and_a_second_is_refused_busy(app):
    choices, _ = _deck_choices(app)
    status, _, body = _post_json(app, "/api/episodes", {"choices": choices})
    assert status == 202
    run_id = json.loads(body)["id"]
    assert re.fullmatch(r"[0-9a-f]{12}", run_id)
    _until_finished(app, run_id)


def test_a_post_that_is_not_json_is_refused(app):
    status, _, _ = _request(app, "POST", "/api/episodes", b"choices=1",
                            {"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 415


def test_the_listing_rows_and_their_order(app):
    choices, _ = _deck_choices(app)
    ids = []
    for _ in range(2):
        _, _, body = _post_json(app, "/api/episodes", {"choices": choices})
        ids.append(json.loads(body)["id"])
        _until_finished(app, ids[-1])
    _, _, body = _request(app, "GET", "/api/episodes")
    rows = json.loads(body)["episodes"]
    # Newest first, each an episode row (no project_id) with exactly these keys.
    assert [r["id"] for r in rows][:2] == ids[::-1]
    for row in rows:
        assert set(row) == {"id", "finished", "started"}
        assert row["finished"] is True and isinstance(row["started"], float)


def test_the_stream_numbers_events_resumes_after_last_event_id_and_ends_on_a_served_file(app):
    choices, _ = _deck_choices(app)
    _, _, body = _post_json(app, "/api/episodes", {"choices": choices})
    run_id = json.loads(body)["id"]
    full = _stream(app, run_id)
    assert [i for i, _ in full] == list(range(len(full)))
    types = [e["type"] for _, e in full]
    assert types[0] == "submitted" and types[-1] == "done"
    # The plugin reconnects with Last-Event-ID = (events seen) - 1; the server
    # resumes at the next one, so nothing is sent twice and nothing is lost.
    assert _stream(app, run_id, last_event_id=0) == full[1:]
    assert _stream(app, run_id, last_event_id=len(full) - 2) == full[-1:]
    done = full[-1][1]
    assert done["run"] and done["file"]
    status, _, video = _request(app, "GET", f"/episodes/{quote(done['run'])}/{quote(done['file'])}")
    assert status == 200 and video


def test_an_unknown_run_is_a_404_the_deck_reads_as_a_restart(app):
    status, _, body = _request(app, "GET", "/api/episodes/0123456789ab/events")
    assert status == 404 and json.loads(body)["error"]


def test_every_event_type_the_run_yields_is_one_the_deck_types():
    yielded = set(re.findall(r'\{"type": "([a-z]+)"', RUNNER_PY))
    union = CONTRACT_TS[CONTRACT_TS.index("export type StudioEvent"):CONTRACT_TS.index("/** The choices")]
    typed = set(re.findall(r'type: "([a-z]+)"', union))
    assert {"submitted", "progress", "step", "status", "output", "note", "done", "error"} <= yielded
    assert yielded <= typed, sorted(yielded - typed)


def test_the_deck_calls_these_routes_and_no_others():
    """Every path the plugin's client builds, whole: the three API routes and
    nothing else (not presence, keys or projects -- API-CONTRACT.md says why)."""
    paths = set(re.findall(r"[`\"](/api/[^`\"?]*)", CLIENT_TS))
    assert paths == {"/api/options${query}", "/api/episodes", "/api/episodes/${encodeURIComponent(runId)}/events"}
    # ... and the only query it adds is the workflow, encoded.
    assert "`?workflow=${encodeURIComponent(workflow)}`" in CLIENT_TS
    # The two addresses it opens in the browser: the finished file, and the page on a run.
    assert "/episodes/${encodeURIComponent(done.run)}/${encodeURIComponent(done.file)}" in STORE_TS
    assert "#run=${encodeURIComponent(this.run.id)}" in STORE_TS


def test_the_plugin_never_ships_in_the_registry_zip():
    ignore = (PLUGIN.parents[2] / ".comfyignore").read_text(encoding="utf-8")
    assert re.search(r"^contest/\s*$", ignore, re.M)
    manifest = json.loads((PLUGIN / "com.jbrick2070.otr.sdPlugin" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["UUID"] == "com.jbrick2070.otr"
