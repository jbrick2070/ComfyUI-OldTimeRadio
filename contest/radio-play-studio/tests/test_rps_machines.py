"""Your machine, how it's filmed, the writer panel and the keys (operator,
2026-10-23: "first you choose ... AMD / Mac / NVIDIA 8/16/24 ... then it
prepopulates those", "don't forget cloud too", "people can mix and match").

The menu picks among workflows the pack ships, the writer is one choice whose
cloud models need a model picked, and a Comfy key rides with each run without
ever coming back out."""
from __future__ import annotations

import json
import threading

import pytest
from test_rps_runner import FakeClient, _good_job
from test_rps_server import _request

from radio_play_studio.config import (MACHINES, WORKFLOW_NAME, Settings, detect_machine,
                                      menu_lookup)
from radio_play_studio.graph import GraphError
from radio_play_studio.runner import Studio
from radio_play_studio.server import make_server

GIB = 2 ** 30


def _device(kind, name, gib):
    return {"devices": [{"type": kind, "name": name, "vram_total": int(gib * GIB)}]}


@pytest.mark.parametrize("stats,machine", [
    (_device("cuda", "cuda:0 NVIDIA GeForce RTX 5080 Laptop GPU", 15.9), "nvidia16"),
    (_device("cuda", "cuda:0 NVIDIA GeForce RTX 4060 Laptop GPU", 8.0), "nvidia8"),
    (_device("cuda", "cuda:0 NVIDIA GeForce RTX 4090", 24.0), "nvidia24"),
    (_device("cuda", "cuda:0 AMD Radeon AI PRO R9700", 32.0), "amd"),
    (_device("mps", "mps", 16.0), "mac16"),
    (_device("cpu", "cpu", 0), None),
    ({}, None),
])
def test_the_machine_comfyui_reports_is_the_one_the_page_starts_on(stats, machine):
    assert detect_machine(stats) == machine


def test_every_menu_entry_is_a_workflow_the_pack_ships():
    from radio_play_studio.config import _checkout_workflow
    folder = _checkout_workflow().parent
    for machine, ways in MACHINES:
        assert ways, machine
        for way, name in ways:
            assert way in ("foley", "video", "stills"), (machine, way)
            assert (folder / name).is_file(), name
            assert menu_lookup(name) == (machine, way)
    # The contest default is the 16 GB machine's first way.
    assert menu_lookup(WORKFLOW_NAME) == ("nvidia16", "foley")


@pytest.fixture
def menu_settings(tmp_path, workflow):
    """A workflows folder holding the 16 GB trio and one 8 GB graph, with the
    contest default's reference twin beside it."""
    folder = tmp_path / "workflows"
    folder.mkdir()
    for name in (WORKFLOW_NAME, "otr_16gb_video.json", "otr_16gb_still.json", "otr_8gb_still.json",
                 WORKFLOW_NAME.replace(".json", "_flux3_ref.json")):
        (folder / name).write_text(json.dumps(workflow), encoding="utf-8")
    return Settings(comfyui_url="http://comfy.invalid", api_url="http://bridge.invalid",
                    workflow_path=folder / WORKFLOW_NAME, out_dir=tmp_path / "episodes")


def _menu_studio(settings, schemas, **kw):
    stats = _device("cuda", "NVIDIA GeForce RTX 5080 Laptop GPU", 15.9)
    return Studio(settings, client_factory=lambda url: None, fetch=lambda url: schemas,
                  comfy_json=lambda path: stats, **kw)


def test_the_menu_lists_only_the_ways_whose_workflow_is_there(menu_settings, schemas):
    machines = {m["key"]: [w["key"] for w in m["ways"]] for m in _menu_studio(menu_settings, schemas).menu()}
    assert machines == {"nvidia16": ["foley", "video", "stills"], "nvidia8": ["stills"]}


def test_a_server_on_a_graph_off_the_menu_offers_no_menu(settings, schemas):
    studio = Studio(settings, client_factory=lambda url: None, fetch=lambda url: schemas)
    assert studio.menu() is None
    assert "menu" not in studio.form()
    with pytest.raises(ValueError):
        studio.resolve_workflow("otr_16gb_still.json")


def test_the_form_is_drawn_from_the_way_it_names(menu_settings, schemas):
    studio = _menu_studio(menu_settings, schemas)
    body = studio.form("otr_8gb_still.json")
    assert body["menu"]["current"] == {"machine": "nvidia8", "way": "stills",
                                       "workflow": "otr_8gb_still.json"}
    assert body["menu"]["detected"] == "nvidia16"
    assert studio.form()["menu"]["current"]["workflow"] == WORKFLOW_NAME


def test_the_machine_is_asked_until_it_answers_then_remembered(menu_settings, schemas):
    calls = []

    def stats(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("busy")
        return _device("cuda", "NVIDIA GeForce RTX 4060", 8.0)
    studio = Studio(menu_settings, client_factory=lambda url: None, fetch=lambda url: schemas,
                    comfy_json=stats)
    assert studio.detected_machine() is None
    assert studio.detected_machine() == "nvidia8"
    assert studio.detected_machine() == "nvidia8" and len(calls) == 2


@pytest.mark.parametrize("name", ["otr_24gb_foley.json", "../otr_16gb_still.json", "otr_app.json"])
def test_only_a_menu_workflow_beside_the_servers_own_is_accepted(menu_settings, schemas, name):
    with pytest.raises(ValueError):
        _menu_studio(menu_settings, schemas).resolve_workflow(name)


def test_a_run_uses_the_way_it_names_and_my_story_still_gets_the_twin(menu_settings, schemas):
    studio = _menu_studio(menu_settings, schemas)
    folder = menu_settings.workflow_path.parent
    assert studio.workflow_for({"workflow": "otr_16gb_still.json"}) == folder / "otr_16gb_still.json"
    assert studio.workflow_for({"bank": "my_story"}).name.endswith("_flux3_ref.json")
    # A way with no twin runs as named for My Story too.
    assert studio.workflow_for({"bank": "my_story", "workflow": "otr_16gb_still.json"}).name == \
        "otr_16gb_still.json"
    # The routing choice never reaches the graph as a field.
    graph, _ = studio.build_graph({"workflow": "otr_16gb_still.json", "acts": "2"})
    assert "workflow" not in json.dumps(graph)


@pytest.mark.parametrize("writer,slot,model", [
    ("openrouter:slot-a", "writer_openrouter", "(enable OpenRouter)"),
    ("google_api:slot-a", "writer_google", "(select Google API model)"),
    ("comfy:slot-a", "writer_comfy", "(enable Comfy Credits)"),
])
def test_a_cloud_writer_with_no_model_picked_is_refused_before_it_is_sent(settings, schemas,
                                                                           writer, slot, model):
    studio = Studio(settings, client_factory=lambda url: None, fetch=lambda url: schemas)
    with pytest.raises(GraphError, match="has no model picked"):
        studio.build_graph({"writer": writer, slot: model})
    picked = {"writer_openrouter": "~anthropic/claude-sonnet-latest", "writer_google": "gemini-pro-latest",
              "writer_comfy": "anthropic/claude-sonnet-5"}[slot]
    graph, ids = studio.build_graph({"writer": writer, slot: picked})
    assert graph[ids["OTR_LedgerScriptWriter"]]["inputs"]["creative_writing_model"] == writer


def test_the_writer_offers_one_cloud_choice_per_service(settings, schemas):
    studio = Studio(settings, client_factory=lambda url: None, fetch=lambda url: schemas)
    writer = next(f for f in studio.options() if f["key"] == "writer")
    assert not [c for c in writer["choices"] if c.endswith(":slot-b")], "slot B is the technical passes'"
    assert {"openrouter:slot-a", "google_api:slot-a", "comfy:slot-a"} <= set(writer["choices"])


class KeyedClient(FakeClient):
    def __init__(self, job):
        super().__init__(job)
        self.keys = []

    def submit(self, workflow, api_key=None):
        self.keys.append(api_key)
        return self._job


def test_a_comfy_key_rides_with_the_run_and_only_then(settings, schemas):
    client = KeyedClient(_good_job())
    studio = Studio(settings, client_factory=lambda url: client, fetch=lambda url: schemas,
                    comfy_key="  sk-test-key  ")
    list(studio.stream({}))
    studio.comfy_key = None
    list(studio.stream({}))
    assert client.keys == ["sk-test-key", None]


def test_the_key_status_names_files_and_never_a_key(menu_settings, schemas, tmp_path):
    studio = _menu_studio(menu_settings, schemas, comfy_key="sk-secret-value")
    pack = menu_settings.workflow_path.parent.parent
    (pack / "google.secret").write_text("AIza-secret-value\n", encoding="utf-8")
    status = studio.key_status()
    assert status["comfy"]["set"] is True
    assert status["google"]["file"] is True and status["google"]["path"].endswith("google.secret")
    assert status["openrouter"]["file"] is False
    assert "secret-value" not in json.dumps(status)


def test_the_page_can_hand_over_a_comfy_key_but_never_read_it_back(settings, schemas):
    studio = Studio(settings, client_factory=lambda url: FakeClient(_good_job()), fetch=lambda url: schemas)
    server = make_server(studio, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, _, body = _request(port, "POST", "/api/keys/comfy", json.dumps({"key": "sk-page-key"}),
                                   {"Content-Type": "application/json"})
        assert status == 200 and json.loads(body)["comfy"]["set"] is True
        assert studio.comfy_key == "sk-page-key"
        for path in ("/api/keys", "/api/options"):
            assert b"sk-page-key" not in _request(port, "GET", path)[2]
        status, _, body = _request(port, "POST", "/api/keys/comfy", json.dumps({"key": ""}),
                                   {"Content-Type": "application/json"})
        assert json.loads(body)["comfy"]["set"] is False and studio.comfy_key is None
        status, _, _ = _request(port, "POST", "/api/keys/comfy", json.dumps({"key": 7}),
                                {"Content-Type": "application/json"})
        assert status == 400
        status, _, _ = _request(port, "GET", "/api/options?workflow=otr_24gb_foley.json")
        assert status == 400
    finally:
        server.shutdown()


@pytest.mark.parametrize("slot", [None, "", "   ", "(enable Comfy Credits)"])
def test_a_cloud_writer_whose_slot_is_missing_or_blank_is_refused(slot):
    """Codex QA of 1e77e064: an absent slot reached the graph as None and passed."""
    from radio_play_studio.form import _check_writer
    inputs = {"creative_writing_model": "comfy:slot-a"}
    if slot is not None:
        inputs["comfy_slot_a_model"] = slot
    with pytest.raises(GraphError, match="has no model picked"):
        _check_writer({"7": {"inputs": inputs}}, {"OTR_LedgerScriptWriter": "7"})
    _check_writer({"7": {"inputs": {"creative_writing_model": "google/gemma-4-12b-it"}}},
                  {"OTR_LedgerScriptWriter": "7"})


def test_a_menu_name_that_links_outside_the_folder_is_refused(menu_settings, schemas, tmp_path):
    """Codex QA of 1e77e064: an allowed file name must still be a file in the folder."""
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    link = menu_settings.workflow_path.parent / "otr_mac16_still.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot make a symlink")
    with pytest.raises(ValueError):
        _menu_studio(menu_settings, schemas).resolve_workflow("otr_mac16_still.json")
