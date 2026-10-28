"""What the Stream Deck plugin (../streamdeck) reads from this server, pinned on
the server's side: the routes it calls, the two rules it must meet (a loopback
Host, JSON posts), the event types it follows, and the choices it posts. The
plugin's own tests run against a stand-in of this server; these make sure the
stand-in and the real thing agree, so a change here is felt here first.
streamdeck/API-CONTRACT.md is the prose form of the same agreement."""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from radio_play_studio import runner, server
from radio_play_studio.form import FIELDS, FIELDS_BY_KEY, MY_STORY_BANK

PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "streamdeck"
CONTRACT_TS = (PLUGIN / "src" / "types" / "contract.ts").read_text(encoding="utf-8")
SETTINGS_TS = (PLUGIN / "src" / "types" / "settings.ts").read_text(encoding="utf-8")
CHOICE_MATH_TS = (PLUGIN / "src" / "actions" / "choice-math.ts").read_text(encoding="utf-8")
CLIENT_TS = (PLUGIN / "src" / "studio" / "studio-client.ts").read_text(encoding="utf-8")
STORE_TS = (PLUGIN / "src" / "state" / "studio-store.ts").read_text(encoding="utf-8")
FAKE_TS = (PLUGIN / "tests" / "helpers" / "fake-studio-server.ts").read_text(encoding="utf-8")
SERVER_PY = pathlib.Path(server.__file__).read_text(encoding="utf-8")
RUNNER_PY = pathlib.Path(runner.__file__).read_text(encoding="utf-8")


class _Headers(dict):
    def get(self, key, default=None):  # http.server's headers are case-insensitive
        for k, v in self.items():
            if k.lower() == str(key).lower():
                return v
        return default


def _host_ok(host: str) -> bool:
    handler = server.Handler.__new__(server.Handler)
    handler.headers = _Headers({"Host": host})
    return server.Handler._host_ok(handler)


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "127.0.0.1", "localhost:8765", "[::1]:8765"])
def test_the_host_a_node_client_sends_to_loopback_passes(host):
    """node:http sends `Host: 127.0.0.1:<port>` for a request to that address;
    the plugin relies on that passing, with the server unchanged."""
    assert _host_ok(host)


@pytest.mark.parametrize("host", ["studio.local:8765", "192.168.1.9:8765", "", "[::1]x"])
def test_a_foreign_host_is_still_refused(host):
    assert not _host_ok(host)


def test_the_dials_name_fields_the_form_has():
    """DIAL_FIELDS in settings.ts are form.FIELDS keys, and `workflow` is the
    one extra key, read by runner.WORKFLOW_CHOICE and not a field."""
    dial_fields = re.search(r"DIAL_FIELDS = \[([^\]]+)\]", SETTINGS_TS).group(1)
    keys = re.findall(r'"([a-z_]+)"', dial_fields)
    assert keys == ["language", "bank", "style", "acts"]
    for key in keys:
        assert key in FIELDS_BY_KEY, key
    assert runner.WORKFLOW_CHOICE == "workflow"
    assert "workflow" not in FIELDS_BY_KEY


def test_the_bank_the_dial_skips_is_the_one_that_takes_typed_text():
    assert re.search(r'MY_STORY_BANK = "([a-z_]+)"', CHOICE_MATH_TS).group(1) == MY_STORY_BANK
    assert FIELDS_BY_KEY["bank"].key == "bank"


def test_the_event_types_the_plugin_follows_are_the_ones_the_run_yields():
    """Every `type` an episode run yields (Studio.stream and normalize_event)
    is in contract.ts's StudioEvent, and the fields the deck reads are the
    ones written."""
    yielded = set(re.findall(r'\{"type": "([a-z]+)"', RUNNER_PY))
    union = CONTRACT_TS[CONTRACT_TS.index("export type StudioEvent"):CONTRACT_TS.index("/** The choices")]
    typed = set(re.findall(r'type: "([a-z]+)"', union))
    assert {"submitted", "progress", "step", "status", "output", "note", "done", "error"} <= yielded
    assert yielded <= typed, sorted(yielded - typed)
    # done carries run and file, and progress may carry phase and minutes_left
    assert '"run": dest.parent.name, "file": dest.name' in RUNNER_PY
    assert '"minutes_left": round(' in RUNNER_PY
    assert 'note["phase"] = phase' in RUNNER_PY


def test_the_stream_is_shaped_as_the_plugin_parses_it():
    """id: then data: then a blank line, and `: keep-alive` comments;
    Last-Event-ID resumes. The fake server in the plugin's tests writes the
    same shape."""
    assert 'f"id: {sent}\\ndata: {json.dumps(event)}\\n\\n"' in SERVER_PY
    assert 'b": keep-alive\\n\\n"' in SERVER_PY
    assert 'self.headers.get("Last-Event-ID", "-1")' in SERVER_PY
    assert "res.write(`id: ${sent}\\ndata: ${JSON.stringify(run.events[sent])}\\n\\n`)" in FAKE_TS
    assert 'res.write(": keep-alive\\n\\n")' in FAKE_TS
    assert "Last-Event-ID" in CLIENT_TS


def test_the_routes_the_plugin_calls_exist_and_no_other():
    """The plugin's client names exactly these paths; each is a route here."""
    called = set(re.findall(r'[`"](/api/[a-z]+)', CLIENT_TS))
    assert called == {"/api/options", "/api/episodes"}
    assert 'if path == "/api/options":' in SERVER_PY
    assert 'if path == "/api/episodes":' in SERVER_PY
    assert 're.fullmatch(r"/api/episodes/([0-9a-f]{12})/events", path)' in SERVER_PY
    assert 'path.startswith("/episodes/")' in SERVER_PY
    # The deck opens the finished file and the page on a run, as the page does.
    assert "/episodes/${encodeURIComponent(done.run)}/${encodeURIComponent(done.file)}" in STORE_TS
    assert "#run=${encodeURIComponent(this.run.id)}" in STORE_TS
    assert '#run=" + body.id' in (PLUGIN.parent / "radio_play_studio" / "static" / "index.html").read_text(
        encoding="utf-8")
    # Not called, on purpose (API-CONTRACT.md says why).
    assert "/api/presence" not in CLIENT_TS
    assert "/api/keys" not in CLIENT_TS
    assert "/api/projects" not in CLIENT_TS


def test_the_listing_carries_what_the_deck_sorts_runs_by():
    """Runs.listing rows: id, finished, started, and the label (project_id,
    stage) for a stage run, which the deck treats as busy."""
    runs = server.Runs.__new__(server.Runs)
    runs._runs, runs._lock = {}, __import__("threading").Lock()
    episode = server.Run("a" * 12)
    stage = server.Run("b" * 12, {"project_id": "p1", "stage": "stills"})
    runs._runs = {episode.id: episode, stage.id: stage}
    rows = runs.listing()
    assert {tuple(sorted(r)) for r in rows} == {
        ("finished", "id", "started"), ("finished", "id", "project_id", "stage", "started")}
    assert "r.project_id" in STORE_TS


def test_a_post_without_json_is_refused_before_it_is_read():
    """The plugin sends application/json; the fake server refuses anything
    else the way this one does."""
    assert '!= "application/json"' in SERVER_PY
    assert "UNSUPPORTED_MEDIA_TYPE" in SERVER_PY
    assert 'contentType !== "application/json"' in FAKE_TS
    assert '"Content-Type"] = "application/json"' in CLIENT_TS


def test_the_fields_the_fake_server_offers_are_shaped_like_describe_form():
    """The stand-in's rows carry the keys describe_form writes, so the plugin's
    tests exercise the real shape."""
    rows = re.search(r"SAMPLE_FIELDS = \[(.*?)\n\];", FAKE_TS, re.S).group(1)
    for key in ("key", "label", "group", "help", "kind", "default"):
        assert f"{key}:" in rows
    import radio_play_studio.form as form
    src = pathlib.Path(form.__file__).read_text(encoding="utf-8")
    block = src[src.index("def describe_form"):src.index("def apply_choices")]
    for key in ("key", "label", "group", "help", "kind", "default", "choices", "min", "max"):
        assert f'"{key}":' in block, key
    assert [f.key for f in FIELDS[:2]] == ["language", "bank"]


def test_the_plugin_never_ships_in_the_registry_zip():
    ignore = (pathlib.Path(server.__file__).resolve().parents[3] / ".comfyignore").read_text(encoding="utf-8")
    assert re.search(r"^contest/\s*$", ignore, re.M)
    manifest = json.loads((PLUGIN / "com.jbrick2070.otr.sdPlugin" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["UUID"] == "com.jbrick2070.otr"
