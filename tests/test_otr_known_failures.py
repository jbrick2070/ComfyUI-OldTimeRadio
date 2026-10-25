"""scripts/otr_known_failures.py -- platform failures get a hint that points at a
real OTR tool, and the hint never costs the diagnosis it rides on.

(PLAN row 1.2, planning/creatoros-concepts/PLAN.md.) Fixtures are the strings
ComfyUI, torch, safetensors and requests really print for each failure class.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import otr_known_failures as kf  # noqa: E402


@pytest.fixture(scope="module")
def api():
    """Load `scripts/otr_api.py` directly -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "otr_api_known_failures_under_test", SCRIPTS / "otr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: One real-shaped string per failure class, keyed by the row's key.
FIXTURES = {
    "server_gone": [
        "SERVER-GONE: the ComfyUI server at http://127.0.0.1:8000 stopped "
        "answering -- 12 consecutive refused connections to /history",
        "requests.exceptions.ConnectionError: HTTPConnectionPool("
        "host='127.0.0.1', port=8000): Max retries exceeded with url: /prompt",
    ],
    "out_of_memory": [
        "torch.OutOfMemoryError: Allocation on device",
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA error: out of memory",
    ],
    "value_not_in_list": [
        "Value not in list: ckpt_name: 'x.safetensors' not in ['a.safetensors']",
        "{'7': {'errors': [{'type': 'value_not_in_list', 'message': 'Value "
        "not in list'}]}}",
    ],
    "truncated_weights": [
        "safetensors_rust.SafetensorError: Error while deserializing header: "
        "MetadataIncompleteBuffer",
        "RuntimeError: PytorchStreamReader failed reading zip archive: failed "
        "finding central directory",
    ],
    "cuda_device_assert": [
        "RuntimeError: CUDA error: device-side assert triggered",
    ],
    "cuda_unavailable": [
        "AssertionError: Torch not compiled with CUDA enabled",
        "RuntimeError: Found no NVIDIA driver on your system. CUDA "
        "initialization: The NVIDIA driver on your system is too old",
    ],
    "missing_module": [
        "ModuleNotFoundError: No module named 'accelerate'",
    ],
    "model_family_mismatch": [
        "RuntimeError: Error(s) in loading state_dict for AutoencoderKL: "
        "size mismatch for decoder.conv_in.weight",
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied (77x768 and "
        "1024x320)",
    ],
}


def test_every_row_has_a_fixture_and_every_fixture_has_a_row():
    assert set(FIXTURES) == {row.key for row in kf.KNOWN_FAILURES}


@pytest.mark.parametrize(
    "key,text",
    [(key, text) for key, texts in FIXTURES.items() for text in texts],
)
def test_each_fixture_matches_its_own_row(key, text):
    row = kf.match_for(text)
    assert row is not None, "no row matched: %r" % text
    assert row.key == key
    assert kf.hint_for(text).startswith(row.hint)


def test_first_match_wins_so_a_dead_server_is_not_called_out_of_memory():
    both = ("SERVER-GONE: the server stopped answering. The last node raised "
            "torch.OutOfMemoryError before the process died.")
    assert kf.match_for(both).key == "server_gone"


def test_out_of_memory_beats_a_module_name_in_the_same_traceback():
    both = ("torch.OutOfMemoryError: Allocation on device\n"
            "  File \"x.py\", line 1\n    # No module named 'x' is not the cause")
    assert kf.match_for(both).key == "out_of_memory"


def test_text_that_matches_nothing_gets_no_hint():
    assert kf.hint_for("TypeError: Object of type method is not JSON serializable") is None
    assert kf.hint_for("") is None
    assert kf.hint_for(None) is None


def test_hint_is_one_plain_line():
    for row in kf.KNOWN_FAILURES:
        assert "\n" not in row.hint, row.key
        row.hint.encode("ascii")
        assert row.hint == row.hint.strip()


def test_every_tool_a_hint_names_exists():
    """A hint that sends a person to a script that is not there is worse than
    no hint."""
    for row in kf.KNOWN_FAILURES:
        for name in re.findall(r"scripts/([A-Za-z0-9_]+\.(?:py|ps1))", row.hint):
            assert (SCRIPTS / name).is_file(), (row.key, name)
        for pattern in re.findall(r"(workflows/[A-Za-z0-9_*]+\.json)", row.hint):
            assert list(REPO.glob(pattern)), (row.key, pattern)


# --- refs resolve: never an invented id ------------------------------------

def _bible_path() -> pathlib.Path:
    override = os.environ.get("OTR_BUG_BIBLE_YAML", "").strip()
    return (pathlib.Path(override) if override else
            REPO.parents[1] / "comfyui-custom-node-survival-guide" / "BUG_BIBLE.yaml")


def _bible_ids() -> set[str]:
    path = _bible_path()
    if not path.is_file():
        pytest.fail(
            "the Bug Bible is not at %s: its repo (comfyui-custom-node-survival-"
            "guide) must sit beside ComfyUI, or set OTR_BUG_BIBLE_YAML. Every "
            "ref in scripts/otr_known_failures.py is checked against it, so "
            "this test cannot pass without it." % path)
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*-\s*id:\s*"?([0-9]+\.[0-9]+)"?\s*$', text, re.M))


def _pbug_ids() -> set[str]:
    text = (REPO / "apple" / "PROD_BUG_LOG.md").read_text(encoding="utf-8")
    return set(re.findall(r"^##\s+(PBUG-\d{8}-\d+)\b", text, re.M))


def test_every_ref_is_a_real_bible_or_pbug_id():
    bible = pbug = None
    for row in kf.KNOWN_FAILURES:
        for ref in row.refs:
            if re.fullmatch(r"\d+\.\d+", ref):
                bible = _bible_ids() if bible is None else bible
                assert ref in bible, "%s: Bible id %s does not exist" % (row.key, ref)
            elif re.fullmatch(r"PBUG-\d{8}-\d+", ref):
                pbug = _pbug_ids() if pbug is None else pbug
                assert ref in pbug, "%s: %s is not in PROD_BUG_LOG.md" % (row.key, ref)
            else:
                pytest.fail("%s: %r is neither a Bible id nor a PBUG id" % (row.key, ref))


def test_refs_appear_in_the_hint_line():
    for row in kf.KNOWN_FAILURES:
        line = kf.hint_for(next(iter(FIXTURES[row.key])))
        for ref in row.refs:
            assert ref in line


# --- wiring in scripts/otr_api.py ------------------------------------------

def _execution_error(exception_type, message, frames=None):
    return [
        ["execution_start", {"prompt_id": "p"}],
        ["execution_error", {
            "prompt_id": "p", "node_id": "9", "node_type": "OTR_VideoRenderBatch",
            "exception_type": exception_type, "exception_message": message,
            "traceback": frames or ['  File "a.py", line 1, in run', "    boom()"],
        }],
    ]


def test_execution_error_carries_the_hint_under_the_headline(api):
    got = api.describe_execution_error(
        _execution_error("torch.OutOfMemoryError", "Allocation on device"))
    lines = got.splitlines()
    assert lines[0] == "node 9 (OTR_VideoRenderBatch) raised torch.OutOfMemoryError"
    assert lines[1] == "Allocation on device"
    assert lines[2].startswith("-- hint: the GPU ran out of memory")
    assert lines[3].startswith("traceback (last 2 frames):")


def test_execution_error_without_a_known_cause_is_unchanged(api):
    got = api.describe_execution_error(
        _execution_error("TypeError", "Object of type method is not JSON serializable"))
    assert "hint" not in got
    assert got.splitlines()[2].startswith("traceback (last 2 frames):")


def test_the_hint_can_come_from_the_traceback_tail(api):
    got = api.describe_execution_error(_execution_error(
        "RuntimeError", "load failed",
        ['  File "loader.py", line 4', "    safetensors_rust.SafetensorError: "
         "Error while deserializing header: HeaderTooLarge"]))
    assert "otr_fetch_lane_weights.py" in got


class _Post:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


def test_a_refused_prompt_with_a_bad_combo_value_prints_its_hint(api, monkeypatch):
    """HTTP 400 is how ComfyUI refuses a dropdown value it does not list. The
    500-character cut of the body lands before `node_errors`, so the hint has to
    be read from the whole body."""
    models = ["model_%03d.safetensors" % n for n in range(60)]
    body = {
        "error": {"type": "prompt_outputs_failed_validation",
                  "message": "Prompt outputs failed validation",
                  "details": "", "extra_info": {}},
        "node_errors": {
            "2": {"errors": [{"type": "required_input_missing",
                              "message": "Required input is missing",
                              "details": "x" * 400}]},
            "7": {"errors": [{
                "type": "value_not_in_list", "message": "Value not in list",
                "details": "ckpt_name: 'gone.safetensors' not in %r" % models}]},
        },
    }
    monkeypatch.setattr(api.requests, "post", lambda *a, **k: _Post(400, body))
    with pytest.raises(RuntimeError) as caught:
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})
    text = str(caught.value)
    assert text.startswith("POST /prompt -> HTTP 400:")
    assert "value_not_in_list" not in text.split(" -- hint: ")[0], (
        "fixture must put the cause past the 500-character cut")
    assert " -- hint: ComfyUI does not list that option or model file" in text
    assert "otr_asset_index.py" in text


def test_node_errors_refusal_and_error_refusal_carry_the_hint(api, monkeypatch):
    node_errors = {"3": {"errors": [{"type": "value_not_in_list"}]}}
    monkeypatch.setattr(api.requests, "post",
                        lambda *a, **k: _Post(200, {"node_errors": node_errors}))
    with pytest.raises(RuntimeError, match=r"node_errors.* -- hint: ComfyUI does not list"):
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})

    monkeypatch.setattr(api.requests, "post", lambda *a, **k: _Post(
        200, {"error": {"type": "x", "message": "CUDA out of memory"}}))
    with pytest.raises(RuntimeError, match=r"returned error.* -- hint: the GPU ran out"):
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})


def test_a_refusal_with_no_known_cause_has_no_hint(api, monkeypatch):
    monkeypatch.setattr(api.requests, "post",
                        lambda *a, **k: _Post(500, "internal server error"))
    with pytest.raises(RuntimeError) as caught:
        api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})
    assert str(caught.value) == "POST /prompt -> HTTP 500: internal server error"


def test_a_broken_hint_table_is_reported_and_keeps_the_diagnosis(api, monkeypatch):
    def boom(_text):
        raise ValueError("table is broken")

    monkeypatch.setattr(kf, "hint_for", boom)
    got = api.describe_execution_error(
        _execution_error("torch.OutOfMemoryError", "Allocation on device"))
    assert "node 9 (OTR_VideoRenderBatch) raised torch.OutOfMemoryError" in got
    assert "-- hint: unavailable (ValueError: table is broken)" in got
    assert "traceback (last 2 frames):" in got
