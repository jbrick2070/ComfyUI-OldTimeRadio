"""tests/test_visual_smoke_fixture.py -- S-F visual smoke fixture (accelerator).

Offline guardrails for scripts/otr_visual_smoke.py. No live server / GPU:
the schemas come from a captured fixture (tests/fixtures/smoke_schemas.json)
and the node-92 widget mapping is read from the REAL workflow JSON.

Covers:
  * the node-92 serialized slot map matches the saved widgets_values length;
  * build_pruned_prompt emits ONLY node 92 (and 63 when asked), with the three
    forceInput sockets baked to constant literals + mode=episode + NO link refs;
  * assert_prompt_pruned rejects a stray node / a residual link;
  * verify_executed enforces the exact {92} / {63,92} acceptance set;
  * bake_bundle copies + rewrites + hashes every referenced asset and raises
    LOUD on a missing one;
  * load_capture rejects a flat (non-node-92-shaped) ledger.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import otr_visual_smoke as vs  # noqa: E402
from otr_api import load_workflow, _serialized_slot_names  # noqa: E402

_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"
_SCHEMAS_FIXTURE = _REPO / "tests" / "fixtures" / "smoke_schemas.json"


def _schemas() -> dict:
    with open(_SCHEMAS_FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _workflow() -> dict:
    return load_workflow(str(_WORKFLOW))


# ---------------------------------------------------------------------------
# Node-92 slot mapping
# ---------------------------------------------------------------------------
def test_render_node_slots_match_saved_widgets():
    schemas = _schemas()
    slots = _serialized_slot_names("OTR_VideoRenderBatch", schemas)
    # the 3 forceInput sockets are excluded; 7 widget slots remain
    assert slots == ["mode", "beats", "oom_index", "frame_count",
                     "engine", "portrait_path", "audio_path"]
    wf = _workflow()
    node = next(n for n in wf["nodes"] if n.get("id") == vs.DEFAULT_RENDER_NODE)
    assert node["type"] == "OTR_VideoRenderBatch"
    assert len(node.get("widgets_values") or []) == len(slots)


# ---------------------------------------------------------------------------
# build_pruned_prompt
# ---------------------------------------------------------------------------
def test_pruned_prompt_only_render_node():
    wf, schemas = _workflow(), _schemas()
    prompt = vs.build_pruned_prompt(
        wf, schemas, ledger_json='{"video":{"shots":[1]}}',
        master_audio_path=r"C:\fake\master.wav", engine="ltx_video")
    assert set(prompt.keys()) == {"92"}
    inp = prompt["92"]["inputs"]
    assert prompt["92"]["class_type"] == "OTR_VideoRenderBatch"
    assert inp["mode"] == "episode"
    assert inp["engine"] == "ltx_video"
    assert inp["patched_ledger_json"] == '{"video":{"shots":[1]}}'
    assert inp["master_audio_path"] == r"C:\fake\master.wav"
    assert inp["image_done"] == vs.SMOKE_IMAGE_DONE
    # no input may be a [src_node, slot] link ref
    for v in inp.values():
        assert not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, str))
                    and str(v[0]).isdigit())


def test_pruned_prompt_keep_validator():
    wf, schemas = _workflow(), _schemas()
    prompt = vs.build_pruned_prompt(
        wf, schemas, ledger_json="{}", master_audio_path="",
        keep_validator=True)
    assert set(prompt.keys()) == {"63", "92"}
    assert prompt["63"]["class_type"] == "OTR_WorkflowValidator"


def test_pruned_prompt_engine_defaults_to_ledger():
    wf, schemas = _workflow(), _schemas()
    prompt = vs.build_pruned_prompt(
        wf, schemas, ledger_json="{}", master_audio_path="", engine=None)
    # engine widget keeps the workflow's saved value (not overridden)
    assert "engine" in prompt["92"]["inputs"]


# ---------------------------------------------------------------------------
# assert_prompt_pruned
# ---------------------------------------------------------------------------
def test_assert_prompt_pruned_accepts_clean():
    prompt = {"92": {"class_type": "OTR_VideoRenderBatch",
                     "inputs": {"mode": "episode", "patched_ledger_json": "{}"}}}
    assert vs.assert_prompt_pruned(prompt) is True


def test_assert_prompt_pruned_rejects_extra_node():
    prompt = {"92": {"class_type": "X", "inputs": {}},
              "1": {"class_type": "OTR_LedgerScriptWriter", "inputs": {}}}
    with pytest.raises(ValueError):
        vs.assert_prompt_pruned(prompt)


def test_assert_prompt_pruned_rejects_residual_link():
    prompt = {"92": {"class_type": "X",
                     "inputs": {"patched_ledger_json": ["260", 0]}}}
    with pytest.raises(ValueError):
        vs.assert_prompt_pruned(prompt)


# ---------------------------------------------------------------------------
# verify_executed
# ---------------------------------------------------------------------------
def test_verify_executed_exact_render_only():
    ok, d = vs.verify_executed({"92"})
    assert ok and d["render_ran"] and not d["unexpected"]


def test_verify_executed_rejects_writer_ran():
    ok, d = vs.verify_executed({"92", "1"})
    assert not ok
    assert "1" in d["unexpected"]


def test_verify_executed_rejects_no_render():
    ok, d = vs.verify_executed(set())
    assert not ok and not d["render_ran"]


def test_verify_executed_validator_allowed_when_kept():
    ok, _ = vs.verify_executed({"63", "92"}, keep_validator=True)
    assert ok
    ok2, d2 = vs.verify_executed({"63", "92"}, keep_validator=False)
    assert not ok2 and "63" in d2["unexpected"]


def test_executed_node_set_reads_outputs():
    entry = {"outputs": {"92": {"text": ["..."]}}}
    assert vs.executed_node_set(entry) == {"92"}


# ---------------------------------------------------------------------------
# bake_bundle
# ---------------------------------------------------------------------------
def _make_asset(p: Path, data=b"asset"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)


def test_bake_bundle_rewrites_copies_hashes(tmp_path):
    src = tmp_path / "src"
    portrait = _make_asset(src / "portrait.png", b"portrait-bytes")
    still = _make_asset(src / "still.png", b"still-bytes")
    wav = _make_asset(src / "beat.wav", b"wav-bytes")
    master = _make_asset(src / "master.wav", b"master-bytes")
    ledger = {
        "episode_id": "ref",
        "video": {"shots": [{"shot_id": "shot_b000"}]},
        "images": {"images": [
            {"kind": "portrait", "object_id": "c1", "path": portrait},
            {"kind": "scene_open", "beat_id": "b000", "path": still},
        ]},
        "lines": [{"line_id": "b000", "audio_wav_path": wav}],
    }
    bundle = tmp_path / "bundle"
    baked, baked_master, manifest = vs.bake_bundle(ledger, master, str(bundle))

    # paths rewritten into the bundle, originals untouched
    new_portrait = baked["images"]["images"][0]["path"]
    new_wav = baked["lines"][0]["audio_wav_path"]
    assert str(bundle) in new_portrait and os.path.exists(new_portrait)
    assert str(bundle) in new_wav and os.path.exists(new_wav)
    assert ledger["images"]["images"][0]["path"] == portrait  # deep-copied, intact
    assert os.path.exists(baked_master) and str(bundle) in baked_master
    # manifest carries a sha for every copied asset + the master
    assert len(manifest["assets"]) == 3
    assert manifest["master"]["sha256"]
    assert not manifest["missing"]
    # baked_ledger.json + bundle_manifest.json written
    assert os.path.exists(bundle / "baked_ledger.json")
    ok, problems = vs.preflight_bundle(str(bundle))
    assert ok, problems


def test_bake_bundle_missing_asset_raises(tmp_path):
    ledger = {
        "video": {"shots": [1]},
        "images": {"images": [
            {"kind": "portrait", "object_id": "c1",
             "path": str(tmp_path / "does_not_exist.png")}]},
        "lines": [],
    }
    with pytest.raises(FileNotFoundError):
        vs.bake_bundle(ledger, "", str(tmp_path / "bundle"))


def test_bake_bundle_dedupes_shared_asset(tmp_path):
    shared = _make_asset(tmp_path / "shared.png", b"x")
    ledger = {
        "video": {"shots": []},
        "images": {"images": [
            {"kind": "portrait", "object_id": "c1", "path": shared},
            {"kind": "portrait", "object_id": "c2", "path": shared}]},
        "lines": [],
    }
    baked, _, manifest = vs.bake_bundle(ledger, "", str(tmp_path / "bundle"))
    a = baked["images"]["images"][0]["path"]
    b = baked["images"]["images"][1]["path"]
    assert a == b  # same source -> same bundle copy (deduped)


def test_iter_asset_slots_counts():
    ledger = {
        "images": {"images": [{"path": "a"}, {"path": "b"}, {"no": "path"}]},
        "lines": [{"audio_wav_path": "c"}, {"bark_wav_path": "d", "char_id": "x"}],
        "video": {"shots": [{"init_image": "e"}]},
    }
    slots = list(vs.iter_asset_slots(ledger))
    assert sorted(cur for _, cur in slots) == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# load_capture
# ---------------------------------------------------------------------------
def test_load_capture_rejects_flat_ledger(tmp_path):
    cap = tmp_path / "cap.json"
    cap.write_text(json.dumps({"ledger": {"shots": [1], "clips": []},
                               "master_audio_path": ""}), encoding="utf-8")
    with pytest.raises(ValueError):
        vs.load_capture(str(cap))


def test_load_capture_accepts_node92_shaped(tmp_path):
    cap = tmp_path / "cap.json"
    cap.write_text(json.dumps(
        {"ledger": {"video": {"shots": [{"shot_id": "s"}]}},
         "master_audio_path": "m.wav"}), encoding="utf-8")
    led, master, src = vs.load_capture(str(cap))
    assert (led.get("video") or {}).get("shots")
    assert master == "m.wav"
