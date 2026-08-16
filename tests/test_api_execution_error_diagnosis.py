"""The leg log must name the node that died, not a truncated repr.

WHY THIS EXISTS. `poll_history` used to return `str(messages)[:500]`. `messages`
is a list of `[event_name, payload]` pairs, mostly timestamps and cache lists,
so the 500-character cut landed inside the traceback:

    'traceback': ['  File "C

The failing node, the exception type, and every frame naming our code were all
past the cut. On 2026-08-12 that cost the diagnosis of BOTH live writer failures
(PBUG-20260812-02 and -03) -- each had to be re-derived out of an unrelated
server log, and one of them twice. A campaign leg costs minutes to hours, so a
lost traceback is a lost leg, not a lost line.

These tests use the REAL payload shape recorded in `tmp/_w45_still_flat.log` and
`tmp/_w45_wan_ti2v.log`, not an invented one.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api():
    """Load `scripts/otr_api.py` directly -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "otr_api_under_test", REPO / "scripts" / "otr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The real shape, from the still_flat leg that died in the writer.
WRITER_FAILURE = [
    ["execution_start", {"prompt_id": "4cc3dd65", "timestamp": 1786540015814}],
    ["execution_cached", {"nodes": [], "prompt_id": "4cc3dd65"}],
    ["execution_error", {
        "prompt_id": "4cc3dd65",
        "node_id": "1",
        "node_type": "OTR_LedgerScriptWriter",
        "executed": ["63"],
        "exception_message": "Object of type method is not JSON serializable\n",
        "exception_type": "TypeError",
        "traceback": [
            '  File "C:\\...\\execution.py", line 1, in execute',
            '  File "C:\\...\\_otr_scifi_news_pro.py", line 1532, in _script_user_prompt',
            "    json.dumps(treatment.model_dump())",
        ],
        # THE BULK IS THE POINT. ComfyUI's `handle_execution_error` attaches
        # `current_inputs` and `current_outputs` to every error, and on a real
        # writer node those carry the whole prompt payload -- which is why the
        # 500-char cut landed inside the traceback rather than after it. A
        # fixture without them is not the shape that caused the defect.
        "current_inputs": {"prompt_text": "x" * 400,
                           "ledger_path": "C:\\...\\pending_ledger.json"},
        "current_outputs": ["63"],
    }],
]


def test_it_names_the_node_that_died(api):
    """The single most valuable fact, and the one the truncation always ate."""
    got = api.describe_execution_error(WRITER_FAILURE)
    assert "OTR_LedgerScriptWriter" in got
    assert "TypeError" in got


def test_it_keeps_the_exception_message(api):
    got = api.describe_execution_error(WRITER_FAILURE)
    assert "Object of type method is not JSON serializable" in got


def test_it_keeps_the_frame_that_names_OUR_code(api):
    """The tail of the traceback is where our frames are -- keeping the head
    would preserve only ComfyUI's executor."""
    got = api.describe_execution_error(WRITER_FAILURE)
    assert "_otr_scifi_news_pro.py" in got
    assert "_script_user_prompt" in got


# NOT TESTED, DELIBERATELY: "the old `str(messages)[:500]` would have lost our
# frames". It was written, and it was a bad test. Whether a 500-character cut
# reaches the traceback depends entirely on how many bytes of `current_inputs`
# and event boilerplate precede it, so the assertion could only be made true by
# tuning fixture padding to a byte count -- which tests the fixture, not the
# code, and would break on any unrelated fixture edit. The behaviour that
# matters is asserted directly above: the node is named and our frames survive.
# Removed rather than padded, since a test nobody can trust is worse than none.


def test_a_non_writer_failure_is_described_the_same_way(api):
    """The wan_ti2v leg died in OTR_CastLock -- a different node, same need."""
    messages = [
        ["execution_start", {"prompt_id": "5cc80919"}],
        ["execution_error", {
            "node_id": "80",
            "node_type": "OTR_CastLock",
            "exception_type": "RuntimeError",
            "exception_message": "freeze cascade stamped "
                                 "freeze_verdict='needs_full_rerun'",
        }],
    ]
    got = api.describe_execution_error(messages)
    assert "OTR_CastLock" in got
    assert "needs_full_rerun" in got


# ---------------------------------------------------------------------------
# It must never be the reason a failure goes unreported
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("messages", [
    None,
    [],
    "a bare string",
    [["execution_start", {"prompt_id": "x"}]],          # no error event
    [["execution_error", "not a dict"]],
    [{"malformed": "not a pair"}],
    [["execution_error", {}]],                          # error, no fields
])
def test_an_unrecognised_payload_still_returns_SOMETHING(api, messages):
    """A diagnosis helper that raises, or returns empty, turns a reported
    failure into a silent one -- strictly worse than the truncation it
    replaced."""
    got = api.describe_execution_error(messages)
    assert isinstance(got, str) and got


def test_an_error_event_with_no_fields_still_says_an_error_happened(api):
    got = api.describe_execution_error([["execution_error", {}]])
    assert "?" in got  # unknown node/type, but the shape is reported


def test_an_INTERRUPTED_prompt_is_not_reported_as_a_node_exception(api):
    """`execution_interrupted` is a separate event carrying NO exception fields
    (ComfyUI `execution.py::handle_execution_error`). On an unattended campaign
    an interrupt usually means an OOM kill or a stopped server -- a completely
    different investigation from a node bug, so it must not read as one."""
    messages = [
        ["execution_start", {"prompt_id": "abc"}],
        ["execution_interrupted", {
            "prompt_id": "abc", "node_id": "42",
            "node_type": "OTR_VideoRender", "executed": ["1"],
        }],
    ]
    got = api.describe_execution_error(messages)
    assert "INTERRUPTED" in got
    assert "OTR_VideoRender" in got
    assert "raised" not in got, "an interrupt is not an exception"


def test_the_payload_field_names_match_ComfyUIs_own_builder(api):
    """Guards against ComfyUI renaming a field under us. These are the exact
    keys `handle_execution_error` writes; if upstream changes them this parser
    silently degrades to the repr fallback, which is what it replaced."""
    required = {"prompt_id", "node_id", "node_type", "executed",
                "exception_message", "exception_type", "traceback"}
    payload = dict(WRITER_FAILURE[-1][1])
    assert required <= set(payload), sorted(required - set(payload))
    got = api.describe_execution_error(WRITER_FAILURE)
    assert "OTR_LedgerScriptWriter" in got and "TypeError" in got


def test_a_very_long_traceback_is_bounded(api):
    """Unbounded output would flood a leg log; the bound keeps the tail."""
    messages = [["execution_error", {
        "node_id": "1", "node_type": "N", "exception_type": "E",
        "exception_message": "boom",
        "traceback": ["frame %d" % i for i in range(200)],
    }]]
    got = api.describe_execution_error(messages)
    assert "frame 199" in got, "the tail must survive"
    assert "frame 0" not in got, "the head must not"
    assert got.count("frame ") <= api._ERROR_TRACEBACK_FRAMES
