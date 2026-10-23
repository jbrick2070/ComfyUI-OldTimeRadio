"""The Scene Producer's direction module (chunk 1, 2026-10-22).

``nodes/_otr_producer_direction.py`` is the one place a direction is read: its
version-1 schema, the names that become paths, and the queue-time admission the
Workflow Validator runs. These tests pin, in order of how much they matter:

* a BLANK widget (empty, whitespace, or absent from an older graph) is an
  ordinary run, and nothing refuses or changes for it;
* a direction is untrusted input: strict schema, strict names, paths confined;
* the stage rules that tie a direction to the writer's ``replay_from``;
* a replay must name the exact bundle it continues, verified file by file;
* a bundle name that already exists refuses (the same run submitted twice).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_producer_direction as PD  # noqa: E402
from nodes import _otr_replay_freezer as RF  # noqa: E402
from nodes import production_ledger as PL  # noqa: E402

FIRST = {"version": 1, "project_id": "a1b2c3d4e5f6",
         "attempt_id": "0f8e1c2d-3b4a-4c5d-9e8f-7a6b5c4d3e2f",
         "stage": "stills", "out": "01-stills"}


def _raw(**changes) -> str:
    data = copy.deepcopy(FIRST)
    for key, value in changes.items():
        if value is _DROP:
            data.pop(key, None)
        else:
            data[key] = value
    return json.dumps(data)


_DROP = object()


@pytest.fixture
def producer_root(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    monkeypatch.setattr(PD, "otr_episodes_root", lambda: episodes)
    return episodes / "_replay" / "p"


# --------------------------------------------------------------------------- #
# blank is an ordinary run
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_a_blank_widget_is_an_ordinary_run(raw):
    assert PD.is_blank(raw)
    assert PD.parse_direction(raw) is None


def test_a_prompt_without_a_direction_is_never_refused():
    assert PD.admit_prompt_direction(None) is None
    assert PD.admit_prompt_direction({}) is None
    old_graph = {"90": {"class_type": "OTR_ShotLock", "inputs": {"script_json": ["62", 1]}}}
    assert PD.admit_prompt_direction(old_graph) is None
    blank = {"90": {"class_type": "OTR_ShotLock", "inputs": {"direction_json": ""}}}
    assert PD.admit_prompt_direction(blank) is None
    # Two Shot Locks and two writers are someone's own graph: only a FILLED
    # direction makes them ambiguous.
    two = {"90": {"class_type": "OTR_ShotLock", "inputs": {"direction_json": " "}},
           "190": {"class_type": "OTR_ShotLock", "inputs": {}},
           "1": {"class_type": "OTR_LedgerScriptWriter", "inputs": {}},
           "101": {"class_type": "OTR_LedgerScriptWriter", "inputs": {}}}
    assert PD.admit_prompt_direction(two) is None


# --------------------------------------------------------------------------- #
# the schema
# --------------------------------------------------------------------------- #

def test_a_minimal_first_stills_direction_is_normalized():
    got = PD.parse_direction(_raw())
    assert got["stage"] == "stills" and got["out"] == "01-stills" and got["source"] is None
    assert got["scenes"] == {} and got["cards"] == {}
    assert got["takes"] == {"characters": {}, "stills": {}, "clips": {}}
    assert got["kept"] == {"stills": [], "clips": []}


def test_a_full_direction_round_trips():
    raw = _raw(stage="video", out="03-video", source="a" * 64,
               scenes={"s01": {"note": "rain on the windows"}},
               cards={"shot_b001": {"picture_text": "a lamp-lit desk", "motion_text": "leans in"}},
               takes={"characters": {"c02": 1}, "stills": {"shot_b001": 2}, "clips": {"shot_b001": 1}},
               kept={"stills": ["shot_b002"], "clips": []})
    got = PD.parse_direction(raw)
    assert got["cards"]["shot_b001"] == {"motion_text": "leans in", "picture_text": "a lamp-lit desk"}
    assert got["takes"]["stills"] == {"shot_b001": 2}
    assert got["kept"]["stills"] == ["shot_b002"]


@pytest.mark.parametrize("raw, words", [
    (["90", 0], "literal text"),
    ("{not json", "not valid JSON"),
    ("[1, 2]", "must be an object"),
    (_raw(extra=1), "unknown field"),
    (_raw(version=2), "version must be 1"),
    (_raw(version=True), "version must be 1"),
    (_raw(project_id="Project1"), "project_id"),
    (_raw(project_id="../etc"), "project_id"),
    (_raw(project_id="abc"), "project_id"),
    (_raw(attempt_id="short"), "attempt_id"),
    (_raw(stage="prepare"), "stage must be"),
    (_raw(out=_DROP), "out must look like"),
    (_raw(out="1-stills"), "out must look like"),
    (_raw(out="01-video"), "does not match stage"),
    (_raw(out="01-stills/../x"), "out must look like"),
    (_raw(stage="publish", out="01-stills", source="a" * 64), "takes no out"),
    (_raw(stage="video", out="02-video"), "needs a source"),
    (_raw(source="ABC"), "lowercase sha256"),
    (_raw(scenes={"bad id!": {"note": "x"}}), "not a valid id"),
    (_raw(scenes={"s01": {"note": "x", "light": "y"}}), "unknown field"),
    (_raw(cards={"shot_b001": {"camera": "pan"}}), "unknown field"),
    (_raw(cards={"shot_b001": {"picture_text": 5}}), "must be text"),
    (_raw(cards={"shot_b001": {"picture_text": "x" * 4001}}), "longer than"),
    (_raw(takes={"stills": {"shot_b001": True}}), "whole number"),
    (_raw(takes={"stills": {"shot_b001": -1}}), "between 0"),
    (_raw(takes={"clips": {"shot_b001": 10000}}), "between 0"),
    (_raw(takes={"portraits": {}}), "unknown field"),
    (_raw(kept={"stills": "shot_b001"}), "must be a list"),
    (_raw(kept={"stills": ["shot_b001", "shot_b001"]}), "twice"),
])
def test_a_malformed_direction_is_refused_with_its_reason(raw, words):
    with pytest.raises(PD.DirectionError, match=words):
        PD.parse_direction(raw)


def test_the_size_limit_holds():
    with pytest.raises(PD.DirectionError, match="larger than"):
        PD.parse_direction(" " + "x" * PD.MAX_DIRECTION_CHARS)


# --------------------------------------------------------------------------- #
# stage rules against replay_from
# --------------------------------------------------------------------------- #

def test_the_first_stills_run_starts_fresh_and_later_runs_replay():
    first = PD.parse_direction(_raw())
    PD.check_stage_rules(first, "")
    with pytest.raises(PD.DirectionError, match="replay_from must be empty"):
        PD.check_stage_rules(first, "otr/episodes/_replay/p/a1b2c3d4e5f6/01-stills")
    later = PD.parse_direction(_raw(out="02-stills", source="b" * 64))
    with pytest.raises(PD.DirectionError, match="replay_from must name it"):
        PD.check_stage_rules(later, "  ")
    PD.check_stage_rules(later, "otr/episodes/_replay/p/a1b2c3d4e5f6/01-stills")


# --------------------------------------------------------------------------- #
# names and paths
# --------------------------------------------------------------------------- #

def test_bundle_paths_are_confined_to_the_producer_folder(producer_root):
    assert PD.bundle_path("a1b2c3d4e5f6", "03-video") == producer_root / "a1b2c3d4e5f6" / "03-video"
    for project, out in [("..", "01-stills"), ("a1b2c3d4e5f6", "../01-stills"),
                         ("a1b2c3d4e5f6", "clips"), ("A1B2", "01-stills")]:
        with pytest.raises(PD.DirectionError):
            PD.bundle_path(project, out)


def test_the_manifest_digest_is_the_sha256_of_its_bytes(tmp_path):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "manifest.json").write_bytes(b'{"x": 1}\n')
    assert PD.manifest_digest(str(bundle)) == hashlib.sha256(b'{"x": 1}\n').hexdigest()
    with pytest.raises(PD.DirectionError, match="no manifest.json"):
        PD.manifest_digest(str(tmp_path))


# --------------------------------------------------------------------------- #
# the queue-time admission
# --------------------------------------------------------------------------- #

def _prompt(direction, replay_from=""):
    return {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
            "1": {"class_type": "OTR_LedgerScriptWriter",
                  "inputs": {"source_bank": "my_story", "replay_from": replay_from}},
            "90": {"class_type": "OTR_ShotLock",
                   "inputs": {"script_json": ["62", 1], "direction_json": direction}}}


def _frozen_bundle(tmp_path, monkeypatch, dest):
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_direction",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    ep, _ledger = tcr._make_episode(tmp_path / "source_episodes")
    return RF.freeze(ep, None, dest=dest)


def test_the_first_stills_run_is_admitted(producer_root):
    got = PD.admit_prompt_direction(_prompt(_raw()))
    assert got["direction"]["stage"] == "stills" and got["manifest"] is None


def test_a_filled_direction_refuses_ambiguous_or_wired_graphs(producer_root):
    wired = _prompt(["200", 0])
    with pytest.raises(PD.DirectionError, match="must type it into the widget"):
        PD.admit_prompt_direction(wired)
    two_locks = _prompt(_raw())
    two_locks["190"] = {"class_type": "OTR_ShotLock", "inputs": {}}
    with pytest.raises(PD.DirectionError, match="exactly one OTR_ShotLock"):
        PD.admit_prompt_direction(two_locks)
    two_writers = _prompt(_raw())
    two_writers["101"] = {"class_type": "OTR_LedgerScriptWriter", "inputs": {}}
    with pytest.raises(PD.DirectionError, match="exactly one OTR_LedgerScriptWriter"):
        PD.admit_prompt_direction(two_writers)
    linked_replay = _prompt(_raw(out="02-stills", source="c" * 64), replay_from=["300", 0])
    with pytest.raises(PD.DirectionError, match="replay_from from another node"):
        PD.admit_prompt_direction(linked_replay)


def test_a_first_run_into_an_existing_project_is_refused(producer_root):
    (producer_root / "a1b2c3d4e5f6").mkdir(parents=True)
    with pytest.raises(PD.DirectionError, match="project a1b2c3d4e5f6 already exists"):
        PD.admit_prompt_direction(_prompt(_raw()))


def test_a_later_run_whose_bundle_name_is_taken_is_refused(producer_root, tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path("a1b2c3d4e5f6", "01-stills"))
    digest = PD.manifest_digest(str(bundle))
    (bundle.parent / "02-stills").mkdir()
    with pytest.raises(PD.DirectionError, match="bundle 02-stills of project a1b2c3d4e5f6 already exists"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source=digest),
                                          replay_from=str(bundle)))


def test_a_replay_must_name_the_exact_bundle_it_continues(producer_root, tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path("a1b2c3d4e5f6", "01-stills"))
    digest = PD.manifest_digest(str(bundle))
    ok = PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source=digest),
                                           replay_from=str(bundle)))
    assert ok["manifest"]["source_episode_id"] and ok["replay_from"] == str(bundle)
    with pytest.raises(PD.DirectionError, match="is not the bundle this direction continues"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source="d" * 64),
                                          replay_from=str(bundle)))


def test_a_tampered_bundle_is_refused_before_anything_runs(producer_root, tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path("a1b2c3d4e5f6", "01-stills"))
    digest = PD.manifest_digest(str(bundle))
    still = next((bundle / "stills").glob("*.png"))
    still.write_bytes(b"\x89PNG swapped after freezing")
    with pytest.raises(PD.DirectionError, match="cannot be trusted"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source=digest),
                                          replay_from=str(bundle)))


# --------------------------------------------------------------------------- #
# the wiring: node 63 asks, Shot Lock carries the widget
# --------------------------------------------------------------------------- #

def test_the_validator_asks_before_either_readiness_gate():
    import inspect
    from nodes import _otr_workflow_validator as wv
    src = inspect.getsource(wv.WorkflowValidator.validate)
    ask = src.index("self._admit_producer_direction(prompt)")
    assert src.index("_mux_gap = master_mux_gap_for_prompt(prompt)") < ask
    assert ask < src.index("if not validate_anyway:")
    assert ask < src.index("_queue_time_readiness_gates(")


def test_the_validator_admits_a_plain_direction_and_refuses_what_is_not_built(producer_root):
    from nodes import _otr_workflow_validator as wv
    admit = wv.WorkflowValidator._admit_producer_direction
    assert admit(_prompt("")) is None
    assert admit({"90": {"class_type": "OTR_ShotLock", "inputs": {}}}) is None
    assert admit(_prompt(_raw())) is None
    with pytest.raises(ValueError, match="direction refused -- .*version must be 1"):
        admit(_prompt(_raw(version=7)))
    with pytest.raises(ValueError, match="card edits are not available in this build yet"):
        admit(_prompt(_raw(cards={"shot_b001": {"picture_text": "a desk"}})))
    publish = PD.parse_direction(_raw(stage="publish", out=_DROP, source="a" * 64))
    assert "Publish is not available" in PD.not_yet_supported(publish)


def test_shot_lock_carries_the_widget_as_an_advanced_input():
    # By identity, never by position: a later widget may legitimately follow it.
    from nodes.otr_shot_lock import OTRShotLock
    optional = OTRShotLock.INPUT_TYPES()["optional"]
    assert "direction_json" in optional
    kind, opts = optional["direction_json"]
    assert kind == "STRING" and opts["default"] == ""
    assert opts["advanced"] is True and opts["multiline"] is True
    assert "socketless" not in opts and "forceInput" not in opts


def test_shot_lock_refuses_what_this_build_cannot_honour_before_any_work():
    from nodes.otr_shot_lock import OTRShotLock
    with pytest.raises(ValueError, match="takes are not available in this build yet"):
        OTRShotLock().lock("{}", direction_json=_raw(takes={"clips": {"shot_b001": 1}}))


def test_every_shipped_graph_saves_the_widget_blank():
    # By identity: find the descriptor by name and its value by its place among
    # Shot Lock's WIDGET descriptors (its other inputs are sockets).
    for path in sorted((_REPO / "workflows").glob("otr_*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        locks = [n for n in graph.get("nodes", []) if n.get("type") == "OTR_ShotLock"]
        for node in locks:
            by_name = {i["name"]: i for i in node["inputs"]}
            desc = by_name.get("direction_json")
            assert desc and desc.get("widget") == {"name": "direction_json"}, path.name
            assert desc.get("link") is None, path.name
            widget_names = [i["name"] for i in node["inputs"] if "widget" in i]
            value = node["widgets_values"][widget_names.index("direction_json")]
            assert value == "", path.name


def test_the_harness_sends_a_validated_direction_as_text(tmp_path):
    sys.path.insert(0, str(_REPO / "scripts"))
    import otr_canonical_api_run as canonical
    import otr_api
    from nodes._otr_workflow_apply import CREATIVE_WHITELIST
    assert "direction_json" in CREATIVE_WHITELIST and "direction_json" in otr_api.CREATIVE_WHITELIST
    # The harness's own loader, offline schemas and converter: what a run does.
    schemas = canonical._schemas(True)
    workflow = canonical.load_workflow(str(_REPO / "workflows" / "otr_canonical.json"))
    good = tmp_path / "direction.json"
    good.write_text(_raw(), encoding="utf-8")
    applied = canonical._apply_direction(workflow, schemas, str(good))
    assert applied == ["OTR_ShotLock.direction_json=<stage stills, project a1b2c3d4e5f6, out 01-stills>"]
    prompt = canonical.workflow_to_api_prompt(workflow, schemas)
    lock = [n for n in prompt.values() if n["class_type"] == "OTR_ShotLock"][0]
    assert lock["inputs"]["direction_json"] == _raw()
    bad = tmp_path / "bad.json"
    bad.write_text(_raw(stage="video", out="02-video"), encoding="utf-8")
    with pytest.raises(SystemExit, match="needs a source"):
        canonical._apply_direction(workflow, schemas, str(bad))


# --------------------------------------------------------------------------- #
# Codex QA on chunk 1 (2026-10-22): each finding pinned
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("changes, words", [
    ({"project_id": "abcd\n"}, "project_id"),
    ({"out": "01-stills\n"}, "out must look like"),
    ({"attempt_id": "0f8e1c2d-3b4a\n"}, "attempt_id"),
    ({"out": "02-stills", "source": "a" * 64 + "\n"}, "lowercase sha256"),
    ({"out": chr(0x0660) + chr(0x0661) + "-stills"}, "out must look like"),  # Arabic-Indic digits
    ({"version": 1.0}, "version must be 1"),
    ({"out": "99-stills"}, "first run writes 01-stills"),
])
def test_names_are_matched_whole_and_the_version_is_an_integer(changes, words):
    with pytest.raises(PD.DirectionError, match=words):
        PD.parse_direction(_raw(**changes))


def test_a_replay_must_stay_inside_its_own_project(producer_root, tmp_path, monkeypatch):
    other = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path("bbbbcccc1111", "01-stills"))
    digest = PD.manifest_digest(str(other))
    with pytest.raises(PD.DirectionError, match="is not one of project a1b2c3d4e5f6's bundles"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source=digest),
                                          replay_from=str(other)))


def test_an_ordinary_bundle_outside_the_producer_folder_is_refused(producer_root, tmp_path, monkeypatch):
    loose = _frozen_bundle(tmp_path, monkeypatch, tmp_path / "elsewhere" / "01-stills")
    digest = PD.manifest_digest(str(loose))
    with pytest.raises(PD.DirectionError, match="is not one of project"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source=digest),
                                          replay_from=str(loose)))


@pytest.mark.parametrize("falsy", [0, False, [], {}])
def test_shot_lock_and_the_validator_agree_on_what_blank_means(falsy):
    from nodes.otr_shot_lock import OTRShotLock
    assert not PD.is_blank(falsy)
    with pytest.raises(ValueError, match="Scene Producer direction refused -- .*literal text"):
        OTRShotLock().lock("{}", direction_json=falsy)


def test_set_cannot_write_the_direction():
    sys.path.insert(0, str(_REPO / "scripts"))
    import otr_canonical_api_run as canonical
    schemas = canonical._schemas(True)
    workflow = canonical.load_workflow(str(_REPO / "workflows" / "otr_canonical.json"))
    with pytest.raises(SystemExit, match="use --direction-json"):
        canonical._apply_set(workflow, schemas, "OTR_ShotLock.direction_json=")
