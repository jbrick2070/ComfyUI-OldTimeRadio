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

import codecs
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
PID = FIRST["project_id"]


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

#: The image engines every test bundle is stamped with, and every test graph's
#: video director picks.
ENGINES = {"announcer_image_model": "z_image_turbo", "music_image_model": "z_image_turbo",
           "character_image_model": "z_image_turbo"}


#: ...and the image seed configuration they are stamped with.
SEED = {"mode": "request_hash", "request_seed": 0}


def _prompt(direction, replay_from="", picks=None):
    director = dict(picks if picks is not None else ENGINES, custom_models_json="{}")
    return {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
            "1": {"class_type": "OTR_LedgerScriptWriter",
                  "inputs": {"source_bank": "my_story", "replay_from": replay_from}},
            "87": {"class_type": "OTR_VideoDirector", "inputs": director},
            "88": {"class_type": "OTR_ImageDirector",
                   "inputs": {"seed_mode": SEED["mode"], "request_seed": SEED["request_seed"]}},
            "90": {"class_type": "OTR_ShotLock",
                   "inputs": {"script_json": ["62", 1], "direction_json": direction}}}


def _stamp_engines(ep, engines=None):
    """Give a fixture episode's ledger the stamps a producer stills run writes:
    its image engines and its image seed configuration."""
    led_path = next((pathlib.Path(ep) / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    led.setdefault("meta", {})[PD.META_IMAGE_ENGINES] = dict(engines or ENGINES)
    led["meta"][PD.META_IMAGE_SEED] = dict(SEED)
    led_path.write_text(json.dumps(led), encoding="utf-8")


def _producer_freeze(ep, dest, *, stamp=True):
    """A producer freeze of ``ep`` into ``dest``, as a stage end makes one."""
    if stamp:
        _stamp_engines(ep)
    return RF.freeze(ep, None, dest=dest,
                     producer=PD.manifest_producer_fields(PD.parse_direction(_raw())))


def _frozen_bundle(tmp_path, monkeypatch, dest, *, producer=True):
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_direction",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    ep, _ledger = tcr._make_episode(tmp_path / "source_episodes")
    if not producer:
        return RF.freeze(ep, None, dest=dest)
    return _producer_freeze(ep, dest)


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


def test_a_project_starts_with_its_first_bundle_not_its_folder(producer_root, tmp_path,
                                                              monkeypatch):
    # Sonnet QA, 2026-10-22: an empty folder, or one holding only a crashed
    # freeze's building sibling, is not a started project, and refusing it
    # refused the retry the plan promises a failed first run.
    (producer_root / PID / ".01-stills.building").mkdir(parents=True)
    assert PD.admit_prompt_direction(_prompt(_raw()))
    # Its first bundle is: the same first run again is the same run submitted
    # twice. (The freeze clears the stale building sibling on its way.)
    _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    assert not (producer_root / PID / ".01-stills.building").exists()
    with pytest.raises(PD.DirectionError, match="already exists: this run was already submitted"):
        PD.admit_prompt_direction(_prompt(_raw()))


def test_a_first_run_into_a_project_with_any_bundle_is_refused(producer_root):
    (producer_root / PID / "02-video").mkdir(parents=True)
    with pytest.raises(PD.DirectionError, match="already has bundle 02-video"):
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
    with pytest.raises(ValueError, match="first run has no cards yet"):
        admit(_prompt(_raw(cards={"shot_b001": {"picture_text": "a desk"}})))
    # Since chunk 3 picture edits, notes, still and character takes and kept
    # stills are real; clip takes and kept clips wait for clip reuse.
    later = PD.parse_direction(_raw(out="02-stills", source="a" * 64,
                                    cards={"shot_b001": {"picture_text": "a desk"}},
                                    scenes={"s01": {"note": "at dusk"}},
                                    takes={"characters": {"c01": 1}, "stills": {"shot_b001": 2}},
                                    kept={"stills": ["shot_b002"]}))
    assert PD.not_yet_supported(later) is None
    clips = PD.parse_direction(_raw(out="02-stills", source="a" * 64,
                                    takes={"clips": {"shot_b001": 1}}, kept={"clips": ["shot_b002"]}))
    assert "clip takes, kept clips are not available" in PD.not_yet_supported(clips)
    action_only = PD.parse_direction(_raw(out="02-stills", source="a" * 64,
                                          cards={"shot_b001": {"motion_text": "leans in"}}))
    assert PD.not_yet_supported(action_only) is None
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
        OTRShotLock().lock("{}", direction_json=_raw(out="02-stills", source="a" * 64,
                                                     takes={"clips": {"shot_b001": 1}}))


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


# --------------------------------------------------------------------------- #
# Sonnet QA on chunk 1 (2026-10-22): each finding pinned
# --------------------------------------------------------------------------- #

def _continue(bundle, out, *, project_id=PID):
    """A later-run prompt that continues ``bundle`` into ``out``."""
    raw = _raw(project_id=project_id, stage=out.split("-", 1)[1], out=out,
               source=PD.manifest_digest(str(bundle)))
    return _prompt(raw, replay_from=str(bundle))


def test_a_refused_first_freeze_does_not_burn_the_project(producer_root, tmp_path, monkeypatch):
    good = _frozen_bundle(tmp_path, monkeypatch, tmp_path / "scratch" / "01-stills")
    ep = pathlib.Path(PL.load_replay_manifest(str(good))["source_episode_root"])
    led_path = next((ep / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    led["images"]["images"].append({"image_id": "gone", "path": str(ep / "stills" / "gone.png")})
    led_path.write_text(json.dumps(led), encoding="utf-8")
    assert PD.admit_prompt_direction(_prompt(_raw()))
    with pytest.raises(PL.ReplayBundleError, match="image rows not found"):
        RF.freeze(ep, None, dest=PD.bundle_path(PID, "01-stills"))
    assert not producer_root.exists(), "the refused freeze removed every folder it created"
    assert PD.admit_prompt_direction(_prompt(_raw())), "so the retry is admitted"


def test_a_later_run_continues_the_latest_bundle_into_the_next_number(producer_root, tmp_path,
                                                                     monkeypatch):
    first = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    ep = pathlib.Path(PL.load_replay_manifest(str(first))["source_episode_root"])
    second = _producer_freeze(ep, PD.bundle_path(PID, "02-stills"))
    with pytest.raises(PD.DirectionError, match="latest bundle is 02-stills"):
        PD.admit_prompt_direction(_continue(first, "03-video"))
    with pytest.raises(PD.DirectionError, match="does not follow .* this run writes 03-video"):
        PD.admit_prompt_direction(_continue(second, "04-video"))
    assert PD.admit_prompt_direction(_continue(second, "03-video"))["replay_from"] == str(second)


def test_nothing_to_continue_and_a_shared_number_are_refused(producer_root, tmp_path, monkeypatch):
    nowhere = PD.bundle_path(PID, "01-stills")
    with pytest.raises(PD.DirectionError, match="has no bundle yet"):
        PD.admit_prompt_direction(_prompt(_raw(out="02-stills", source="a" * 64),
                                          replay_from=str(nowhere)))
    first = _frozen_bundle(tmp_path, monkeypatch, nowhere)
    (producer_root / PID / "01-video").mkdir()
    with pytest.raises(PD.DirectionError, match="two bundles numbered 01"):
        PD.admit_prompt_direction(_continue(first, "02-stills"))


def test_a_key_named_twice_is_refused():
    with pytest.raises(PD.DirectionError, match="names 'project_id' twice"):
        PD.parse_direction(_raw()[:-1] + ', "project_id": "bbbbcccc1111"}')
    later = _raw(out="02-stills", source="a" * 64)[:-1]
    with pytest.raises(PD.DirectionError, match="names 'shot_001' twice"):
        PD.parse_direction(later + ', "cards": {"shot_001": {"motion_text": "a"}, '
                                   '"shot_001": {"motion_text": "b"}}}')


def test_a_deeply_nested_direction_is_refused_not_raised():
    with pytest.raises(PD.DirectionError, match="nested too deeply"):
        PD.parse_direction(_raw()[:-1] + ', "cards": ' + "[" * 100_000 + "]" * 100_000 + "}")


def test_a_refusal_quotes_a_bad_value_briefly():
    with pytest.raises(PD.DirectionError) as info:
        PD.parse_direction(_raw(project_id="x" * 900_000))
    assert len(str(info.value)) < 300


@pytest.mark.parametrize("bad", [".", "..", "a:b", "a.b", "__proto__", "-x", "", "con",
                                 "COM1", "lpt9", "nul"])
def test_ids_that_could_misname_a_file_are_refused(bad):
    with pytest.raises(PD.DirectionError, match="not a valid id"):
        PD.parse_direction(_raw(out="02-stills", source="a" * 64,
                                cards={bad: {"motion_text": "x"}}))


def test_every_id_shape_the_ledger_mints_is_accepted():
    ids = ["shot_shot_001_b2", "shot_b000_music_open", "shot_music_opening_001", "s01", "c03"]
    got = PD.parse_direction(_raw(out="02-stills", source="a" * 64,
                                  cards={i: {"motion_text": "x"} for i in ids}))
    assert sorted(got["cards"]) == sorted(ids)


@pytest.mark.parametrize("fault", ["truncated", "bom", "not_an_object_row"])
def test_a_corrupt_manifest_is_refused_as_untrusted(producer_root, tmp_path, monkeypatch, fault):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    path = bundle / PD.MANIFEST_NAME
    if fault == "truncated":
        payload = path.read_bytes()[:40]
    elif fault == "bom":
        payload = codecs.BOM_UTF8 + path.read_bytes()
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["files"].append(1)
        payload = json.dumps(data).encode("utf-8")
    path.write_bytes(payload)
    prompt = _prompt(_raw(out="02-stills", source=hashlib.sha256(payload).hexdigest()),
                     replay_from=str(bundle))
    with pytest.raises(PD.DirectionError, match="cannot be trusted"):
        PD.admit_prompt_direction(prompt)


_junctions = pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows feature")


def _junction(link: pathlib.Path, target: pathlib.Path) -> None:
    import _winapi
    _winapi.CreateJunction(str(target), str(link))


@_junctions
def test_a_linked_bundle_or_project_cannot_borrow_another_projects_bundle(producer_root, tmp_path,
                                                                         monkeypatch):
    other = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path("bbbbcccc1111", "01-stills"))
    # A bundle-named link inside this project, leading to the other project's bundle.
    (producer_root / PID).mkdir()
    _junction(producer_root / PID / "01-stills", other)
    with pytest.raises(PD.DirectionError, match="is not one of project a1b2c3d4e5f6's bundles"):
        PD.admit_prompt_direction(_continue(producer_root / PID / "01-stills", "02-stills"))
    # A whole project folder that is a link to the other project.
    _junction(producer_root / "dddd0000aaaa", producer_root / "bbbbcccc1111")
    with pytest.raises(PD.DirectionError, match="is not one of project dddd0000aaaa's bundles"):
        PD.admit_prompt_direction(_continue(producer_root / "dddd0000aaaa" / "01-stills",
                                            "02-stills", project_id="dddd0000aaaa"))


@_junctions
def test_the_real_spelling_of_a_linked_output_root_names_the_same_bundle(tmp_path, monkeypatch):
    # The Comfy Desktop layout reaches the output folder through a junction; the
    # pack's own spelling and the real one are the same bundle.
    real = tmp_path / "real_output"
    (real / "episodes").mkdir(parents=True)
    _junction(tmp_path / "linked_output", real)
    monkeypatch.setattr(PD, "otr_episodes_root", lambda: tmp_path / "linked_output" / "episodes")
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    spelled_real = real / "episodes" / "_replay" / "p" / PID / "01-stills"
    assert spelled_real.is_dir()
    for spelling in (bundle, spelled_real):
        assert PD.admit_prompt_direction(_continue(spelling, "02-stills"))["manifest"]


def test_the_harness_sends_the_file_text_verbatim_and_explains_utf16(tmp_path):
    sys.path.insert(0, str(_REPO / "scripts"))
    import otr_canonical_api_run as canonical
    schemas = canonical._schemas(True)
    workflow = canonical.load_workflow(str(_REPO / "workflows" / "otr_canonical.json"))
    text = _raw().replace(", ", "," + chr(13) + chr(10))
    with_bom = tmp_path / "bom.json"
    with_bom.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))
    canonical._apply_direction(workflow, schemas, str(with_bom))
    prompt = canonical.workflow_to_api_prompt(workflow, schemas)
    lock = [n for n in prompt.values() if n["class_type"] == "OTR_ShotLock"][0]
    assert lock["inputs"]["direction_json"] == text, "CRLF kept, the BOM dropped"
    utf16 = tmp_path / "utf16.json"
    utf16.write_bytes(_raw().encode("utf-16"))
    with pytest.raises(SystemExit, match="not UTF-8 text"):
        canonical._apply_direction(workflow, schemas, str(utf16))


def test_an_empty_direction_option_is_refused_not_ignored(tmp_path):
    sys.path.insert(0, str(_REPO / "scripts"))
    import otr_canonical_api_run as canonical
    with pytest.raises(SystemExit, match="direction-json missing"):
        canonical.main(["--offline-schemas", "--dry-run", "--direction-json", "",
                        "--dump-prompt", str(tmp_path / "prompt.json")])


# --------------------------------------------------------------------------- #
# chunk 3a: the image-engine stamp and its check
# --------------------------------------------------------------------------- #

def test_a_producer_freeze_records_its_identity_and_the_stills_engines(tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, tmp_path / "b" / "01-stills")
    producer = PL.load_replay_manifest(str(bundle))[PD.MANIFEST_PRODUCER_KEY]
    assert producer == {"version": 1, "project_id": PID, "attempt_id": FIRST["attempt_id"],
                        "stage": "stills", "out": "01-stills", "source": None,
                        "image_engines": ENGINES, "image_seed": SEED,
                        # chunk 3b: what a replay of it renders with (the
                        # fixture's histogram is empty; its one shot's lane)
                        "image_engines_used": [],
                        "video_engines": ["animatediff15_v3_haunted_video"],
                        # chunk 3e: what the direction kept
                        "kept": {"stills": [], "clips": []}}


def test_an_ordinary_freeze_writes_no_producer_section_and_an_unstamped_one_is_refused(
        tmp_path, monkeypatch):
    plain = _frozen_bundle(tmp_path, monkeypatch, tmp_path / "plain" / "ep", producer=False)
    assert PD.MANIFEST_PRODUCER_KEY not in PL.load_replay_manifest(str(plain))
    ep = pathlib.Path(PL.load_replay_manifest(str(plain))["source_episode_root"])
    with pytest.raises(PL.ReplayBundleError, match="records no engines"):
        _producer_freeze(ep, tmp_path / "unstamped" / "01-stills", stamp=False)
    assert not (tmp_path / "unstamped").exists()


def test_a_later_run_must_paint_with_the_bundles_engines(producer_root, tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    assert PD.admit_prompt_direction(_continue(bundle, "02-stills"))["manifest"]
    swapped = _continue(bundle, "02-stills")
    swapped["87"]["inputs"]["character_image_model"] = "flux_gen1"
    with pytest.raises(PD.DirectionError,
                       match="character_image_model 'flux_gen1' \\(the bundle used 'z_image_turbo'\\)"):
        PD.admit_prompt_direction(swapped)
    wired = _continue(bundle, "02-stills")
    wired["87"]["inputs"]["music_image_model"] = ["300", 0]
    with pytest.raises(PD.DirectionError, match="takes music_image_model from another node"):
        PD.admit_prompt_direction(wired)
    headless = _continue(bundle, "02-stills")
    del headless["87"]
    with pytest.raises(PD.DirectionError, match="exactly one OTR_VideoDirector"):
        PD.admit_prompt_direction(headless)


def test_a_later_run_must_keep_the_projects_image_seed(producer_root, tmp_path, monkeypatch):
    # Codex QA, 2026-10-22: every still's seed and cache key derive from it, so
    # a changed widget would repaint every still with no edit at all.
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"))
    assert PL.load_replay_manifest(str(bundle))[PD.MANIFEST_PRODUCER_KEY]["image_seed"] == SEED
    reseeded = _continue(bundle, "02-stills")
    reseeded["88"]["inputs"]["request_seed"] = 7
    with pytest.raises(PD.DirectionError, match="image director's seed .* differs"):
        PD.admit_prompt_direction(reseeded)
    wired = _continue(bundle, "02-stills")
    wired["88"]["inputs"]["seed_mode"] = ["400", 0]
    with pytest.raises(PD.DirectionError, match="takes seed_mode from another node"):
        PD.admit_prompt_direction(wired)


def test_the_run_time_check_reads_the_policy_actually_wired():
    meta = {PD.META_IMAGE_ENGINES: dict(ENGINES), PD.META_IMAGE_SEED: dict(SEED)}
    policy = {"image_models": {slot: {"engine_id": engine} for slot, engine in ENGINES.items()},
              "seed": {"mode": "request_hash", "request_seed": 0}}
    assert PD.check_policy_matches_stamp(policy, meta) is None
    swapped = dict(policy, image_models=dict(policy["image_models"],
                                             music_image_model={"engine_id": "flux_gen1"}))
    assert "music_image_model 'flux_gen1'" in PD.check_policy_matches_stamp(swapped, meta)
    reseeded = dict(policy, seed={"mode": "fixed", "request_seed": 0})
    assert "seed" in PD.check_policy_matches_stamp(reseeded, meta)
    assert "no image-engine and seed stamp" in PD.check_policy_matches_stamp(policy, {})


def test_a_bundle_without_the_stamp_cannot_be_continued(producer_root, tmp_path, monkeypatch):
    bundle = _frozen_bundle(tmp_path, monkeypatch, PD.bundle_path(PID, "01-stills"),
                            producer=False)
    with pytest.raises(PD.DirectionError, match="carries no image-engine stamp"):
        PD.admit_prompt_direction(_continue(bundle, "02-stills"))


def test_the_check_resolves_picks_as_the_two_directors_do():
    # An empty pick takes the image director's default; a custom sentinel takes
    # its custom_models_json entry, exactly as the video director resolves it.
    prompt = _prompt(_raw(), picks={"announcer_image_model": "",
                                    "music_image_model": " z_image_turbo ",
                                    "character_image_model": "+ Add Custom Model"})
    prompt["87"]["inputs"]["custom_models_json"] = json.dumps(
        {"character_image_model": "my_house_engine"})
    assert PD.prompt_image_engines(prompt) == {
        "announcer_image_model": "flux_gen1", "music_image_model": "z_image_turbo",
        "character_image_model": "my_house_engine"}
    # Sonnet QA, 2026-10-22: a sentinel the video director leaves unresolved
    # is resolved by the image director from its own map, as it paints it.
    prompt["87"]["inputs"]["custom_models_json"] = "{}"
    prompt["88"]["inputs"]["custom_models_json"] = json.dumps(
        {"character_image_model": "my_house_engine"})
    assert PD.prompt_image_engines(prompt)["character_image_model"] == "my_house_engine"
    prompt["88"]["inputs"]["custom_models_json"] = "{}"
    assert PD.prompt_image_engines(prompt)["character_image_model"] == "", \
        "declared nowhere, it paints with nothing, and the stamp records that"
    prompt["88"]["inputs"]["custom_models_json"] = ["500", 0]
    with pytest.raises(PD.DirectionError, match="OTR_ImageDirector node 88 takes "
                                                "custom_models_json from another node"):
        PD.prompt_image_engines(prompt)


def test_the_stamp_reads_the_policy_the_dispatcher_paints_with():
    policy = {"image_models": {"announcer_image_model": {"engine_id": "z_image_turbo"},
                               "character_image_model": "flux_gen1"}}
    assert PD.policy_image_engines(policy) == {
        "announcer_image_model": "z_image_turbo", "music_image_model": "",
        "character_image_model": "flux_gen1"}


# --------------------------------------------------------------------------- #
# chunk 3b: readiness -- what a producer replay must have ready
# --------------------------------------------------------------------------- #

USED = {"announcer_visual": {"cloud_flux_pro": 2}, "character_video": {"z_image_turbo": 9}}


def _bundle_painted_with(tmp_path, monkeypatch, by_role):
    """A producer 01-stills bundle whose stills the dispatcher's histogram says
    were painted with ``by_role``."""
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_readiness",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    ep, _ledger = tcr._make_episode(tmp_path / "source_episodes")
    led_path = next((ep / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    led["meta"]["image_engines"] = {"by_role": by_role, "image_revision": 1}
    led_path.write_text(json.dumps(led), encoding="utf-8")
    return _producer_freeze(ep, PD.bundle_path(PID, "01-stills"))


def _gated(prompt):
    """The prompt with its writer gated by node 63, as the canonical graph is."""
    prompt["1"]["inputs"]["gate_in"] = ["63", 0]
    prompt["87"]["inputs"]["gate_in"] = ["63", 0]
    return prompt


def test_the_resolver_names_what_each_producer_replay_renders(producer_root, tmp_path,
                                                             monkeypatch):
    bundle = _bundle_painted_with(tmp_path, monkeypatch, USED)
    producer = PL.load_replay_manifest(str(bundle))[PD.MANIFEST_PRODUCER_KEY]
    assert producer["image_engines_used"] == ["cloud_flux_pro", "z_image_turbo"]
    assert PD.producer_replay_needs(_continue(bundle, "02-stills")) == {
        "stage": "stills", "image_engines": ["cloud_flux_pro", "z_image_turbo"],
        "video_engines": []}
    assert PD.producer_replay_needs(_continue(bundle, "02-video")) == {
        "stage": "video", "image_engines": [],
        "video_engines": ["animatediff15_v3_haunted_video"]}
    publish = _prompt(_raw(stage="publish", out=None, source=PD.manifest_digest(str(bundle))),
                      replay_from=str(bundle))
    assert PD.producer_replay_needs(publish) == {
        "stage": "publish", "image_engines": [], "video_engines": []}


def test_the_resolver_leaves_every_other_prompt_alone(producer_root):
    assert PD.producer_replay_needs(_prompt("")) is None, "a blank widget"
    assert PD.producer_replay_needs(_prompt(_raw())) is None, "a project's first run is live"
    assert PD.producer_replay_needs({"1": {"class_type": "OTR_LedgerScriptWriter"}}) is None


def test_a_video_replay_plans_its_frozen_lanes_not_the_live_dropdowns(producer_root, tmp_path,
                                                                    monkeypatch):
    from nodes import _otr_visual_assets as VA
    bundle = _bundle_painted_with(tmp_path, monkeypatch, USED)

    def no_live_resolver(*_a, **_k):
        raise AssertionError("a producer replay never reads the live dropdowns")

    plan = VA.plan_prompt(_gated(_continue(bundle, "02-video")), "63",
                          resolve_video=no_live_resolver, freeze_video=no_live_resolver)
    assert plan["replay"] and plan["engines"] == {"animatediff15_v3_haunted_video"}
    plan = VA.plan_prompt(_gated(_continue(bundle, "02-stills")), "63",
                          resolve_video=no_live_resolver, freeze_video=no_live_resolver)
    assert plan["engines"] == {"cloud_flux_pro", "z_image_turbo"}


def test_the_cloud_gates_check_and_price_what_the_replay_renders(producer_root, tmp_path,
                                                               monkeypatch):
    from nodes._otr_shared import cloud_balance_preflight as CBP
    from nodes._otr_shared import cloud_slug_preflight as CSP
    bundle = _bundle_painted_with(tmp_path, monkeypatch, USED)
    stills = _gated(_continue(bundle, "02-stills"))
    looked_up = []
    CSP.collect_findings(stills, "63", resolve_engine=lambda eid: looked_up.append(eid),
                         walk_fn=lambda prompt, uid: list(prompt.values()))
    assert sorted(looked_up) == ["cloud_flux_pro", "z_image_turbo"]
    lines = CBP.estimate_lines(stills, "63", resolve_engine=lambda eid: None,
                               unit_usd_fn=lambda label, eng: 0.5,
                               walk_fn=lambda prompt, uid: list(prompt.values()))
    assert [line.label for line in lines] == ["cloud_flux_pro"], "the paid image engine only"


# --------------------------------------------------------------------------- #
# chunk 3e: cards.json and the unpainted-edit check, through a real freeze
# --------------------------------------------------------------------------- #

def _bundle_with_a_painted_card(tmp_path, monkeypatch):
    """A producer 01-stills bundle whose one card still, shot_b001, records
    the digest of what it was painted from, as the dispatcher stamps it."""
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_cards",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    ep, _ledger = tcr._make_episode(tmp_path / "source_episodes")
    led_path = next((ep / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    obj = {"object_id": "still_b001", "beat_id": "b001", "kind": "scene_beat",
           "role": "character_video", "prompt": "a lamp-lit desk", "prompt_hash": "h"}
    raw = {"cards": {"shot_b001": "a lamp-lit desk"}, "scenes": {}}
    takes = {"characters": {}, "stills": {}}
    row = led["images"]["images"][0]
    row["object_id"] = "still_b001"
    row[PD.ROW_INPUT_HASH] = PD.card_still_digest(obj, "", raw, takes, {})
    led["images"].update({PD.PAYLOAD_KEY: {"version": 1, "objects": [obj]},
                          PD.RAW_TEXTS_KEY: raw, PD.TAKES_KEY: takes})
    led_path.write_text(json.dumps(led), encoding="utf-8")
    return _producer_freeze(ep, PD.bundle_path(PID, "01-stills"))


def test_a_producer_freeze_writes_a_verified_cards_json(producer_root, tmp_path, monkeypatch):
    bundle = _bundle_with_a_painted_card(tmp_path, monkeypatch)
    assert PD.CARDS_NAME in {f["path"] for f in PL.load_replay_manifest(str(bundle))["files"]}
    cards = json.loads((bundle / PD.CARDS_NAME).read_text(encoding="utf-8"))
    card = cards["scenes"][0]["shots"][0]
    assert (cards["version"], cards["project_id"], cards["bundle"]) == (1, PID, "01-stills")
    assert card["shot_id"] == "shot_b001" and card["picture_text"] == "a lamp-lit desk"
    assert card["still"] == "stills/scene_b001.png" and card["still_input_hash"]
    plain = _frozen_bundle(tmp_path / "plain", monkeypatch, tmp_path / "plain_b" / "ep",
                           producer=False)
    assert not (plain / PD.CARDS_NAME).exists(), "an ordinary freeze writes no cards"


def test_node_63_refuses_a_video_run_over_edits_the_stills_do_not_show(producer_root, tmp_path,
                                                                      monkeypatch):
    bundle = _bundle_with_a_painted_card(tmp_path, monkeypatch)
    digest = PD.manifest_digest(str(bundle))

    def video(**changes):
        return _prompt(_raw(stage="video", out="02-video", source=digest, **changes),
                       replay_from=str(bundle))

    assert PD.admit_prompt_direction(video())["manifest"], "nothing pending"
    edit = {"shot_b001": {"picture_text": "a broken desk"}}
    with pytest.raises(PD.DirectionError,
                       match="1 card\\(s\\) carry edits their stills do not show yet: shot_b001"):
        PD.admit_prompt_direction(video(cards=edit))
    assert PD.admit_prompt_direction(video(cards=edit, kept={"stills": ["shot_b001"]}))
    with pytest.raises(PD.DirectionError, match="do not show yet: shot_b001"):
        PD.admit_prompt_direction(video(takes={"stills": {"shot_b001": 1}}))
    # A stills replay is how the edit gets painted: it is never refused for it.
    assert PD.admit_prompt_direction(_prompt(
        _raw(stage="stills", out="02-stills", source=digest, cards=edit),
        replay_from=str(bundle)))


def test_node_63_refuses_what_the_bundle_does_not_show(producer_root, tmp_path, monkeypatch):
    # Sonnet QA, 2026-10-22: each of these used to be admitted and then do
    # nothing at all.
    bundle = _bundle_with_a_painted_card(tmp_path, monkeypatch)
    digest = PD.manifest_digest(str(bundle))

    def run(stage, **changes):
        return _prompt(_raw(stage=stage, out="02-%s" % stage, source=digest, **changes),
                       replay_from=str(bundle))

    for changes, words in (
            ({"cards": {"shot_nope": {"picture_text": "x"}}}, "no card shot_nope"),
            ({"scenes": {"s99": {"note": "x"}}}, "no scene s99"),
            ({"takes": {"characters": {"c99": 1}}}, "no character c99"),
            # Ghost composes from its own prompt: the fixture's one shot has no
            # action to edit, which Shot Lock alone used to catch, at run time.
            ({"cards": {"shot_b001": {"motion_text": "turns"}}},
             "card shot_b001 has no action to edit")):
        for stage in ("stills", "video"):
            with pytest.raises(PD.DirectionError,
                               match="does not fit bundle 01-stills: .*" + words):
                PD.admit_prompt_direction(run(stage, **changes))
    # The fixture's still carries no character, so only the cast's take shows
    # c02's: a Video run waits for the Stills run that paints it.
    with pytest.raises(PD.DirectionError, match="the take of c02 differs"):
        PD.admit_prompt_direction(run("video", takes={"characters": {"c02": 1}}))
    assert PD.admit_prompt_direction(run("stills", takes={"characters": {"c02": 1}}))
