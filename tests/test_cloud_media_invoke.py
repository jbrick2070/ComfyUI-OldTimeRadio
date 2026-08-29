"""S0 invoke_partner_node bridge -- offline contract tests.

No network, no comfy, no torch: the partner class, interrupt flag and
heartbeat are all monkeypatched through the module's named seams.
Build doc: docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md sec 3.
"""
import asyncio
import sys
import time
import types
import uuid

import pytest

from nodes._otr_shared import cloud_media_backend as backend
from nodes._otr_shared import cloud_media_invoke as invoke
from nodes._otr_shared.cloud_media_backend import (
    CloudErrorCode,
    CloudMediaError,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _fake_row(**over):
    row = {
        "api_node": True,
        "auth_hidden_present": ["auth_token_comfy_org", "api_key_comfy_org"],
        "class_name": "FakePartnerNode",
        "function": "EXECUTE_NORMALIZED_ASYNC",
        "import_path": "tests.fake_partner_module",
        "inputs": {
            "hidden": {
                "api_key_comfy_org": "API_KEY_COMFY_ORG",
                "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG",
                "comfy_usage_source": "COMFY_USAGE_SOURCE",
                "unique_id": "UNIQUE_ID",
            },
            "optional": {"prompt": "STRING"},
            "required": {},
        },
        "is_async": True,
        "provider_id": "TESTPROV",
        "return_types": ["VIDEO"],
        "selected_output": 0,
        "status": "OK",
    }
    row.update(over)
    return row


class _RecordingNode:
    """Stands in for a pinned comfy_api partner class."""

    last_kwargs = None
    behavior = None  # callable(kwargs) -> value, set per test

    async def EXECUTE_NORMALIZED_ASYNC(self, **kwargs):
        type(self).last_kwargs = kwargs
        return type(self).behavior(kwargs)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Cloud env (no enable flag -- removed 2026-07-02) + fake row + fake
    node class + bound prompt."""
    monkeypatch.setenv("OTR_CLOUD_MEDIA_BUDGET_USD", "5.00")
    monkeypatch.setenv("OTR_COMFY_API_KEY", "test-key-123")
    monkeypatch.setenv("OTR_CLOUD_MEDIA_CACHE_DIR", str(tmp_path / "cache"))

    node_key = "cloud_fake_video"
    invoke._set_rows_for_tests({node_key: _fake_row()})
    monkeypatch.setattr(invoke, "_resolve_node_class",
                        lambda row: _RecordingNode)
    monkeypatch.setattr(invoke, "_interrupt_requested", lambda: False)
    beats = []
    monkeypatch.setattr(invoke, "_emit_heartbeat",
                        lambda nk, el, to: beats.append(el))
    _RecordingNode.last_kwargs = None

    src = tmp_path / "provider_output.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42-fake-payload")
    _RecordingNode.behavior = lambda kwargs: (str(src),)

    prompt_id = f"test-prompt-{uuid.uuid4().hex[:8]}"
    with invoke.bind_prompt_id(prompt_id):
        yield {
            "node_key": node_key, "prompt_id": prompt_id,
            "src": src, "beats": beats, "tmp": tmp_path,
        }
    backend.teardown_session(prompt_id)
    invoke._set_rows_for_tests(None)


def _session(rig_data):
    return backend.peek_session(rig_data["prompt_id"])


# ---------------------------------------------------------------------------
# Gate + config failures
# ---------------------------------------------------------------------------


def test_no_enable_flag_gate(monkeypatch, rig):
    """Operator directive 2026-07-02: the dropdown pick is the enable --
    invoke must NOT gate on OTR_ENABLE_COMFY_CLOUD_MEDIA. With credentials
    present the call proceeds regardless of the (removed) flag."""
    monkeypatch.delenv("OTR_ENABLE_COMFY_CLOUD_MEDIA", raising=False)
    result = invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    assert result["path"]


def test_missing_credentials_fail_loud(monkeypatch, rig):
    """A dropdown pick without ANY credentials fails LOUD at auth,
    naming the sources (env key + hidden inputs)."""
    monkeypatch.delenv("OTR_COMFY_API_KEY", raising=False)
    backend.teardown_session(rig["prompt_id"])  # drop the keyed session
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    assert ei.value.code is CloudErrorCode.AUTH
    assert "OTR_COMFY_API_KEY" in str(ei.value)


def test_unknown_node_key(rig):
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node("cloud_nonexistent", {}, timeout_s=5)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG


def test_non_ok_row_status(rig):
    invoke._set_rows_for_tests(
        {rig["node_key"]: _fake_row(status="DRIFTED")})
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    assert ei.value.code is CloudErrorCode.UNSUPPORTED_SCHEMA


def test_bad_timeout_rejected(rig):
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=0)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG


def test_missing_required_input_fails_before_partner(rig):
    row = _fake_row(inputs={
        "hidden": _fake_row()["inputs"]["hidden"],
        "optional": {},
        "required": {"prompt": "STRING"},
    })
    invoke._set_rows_for_tests({rig["node_key"]: row})
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG
    assert "omitted required Partner input" in str(ei.value)
    assert _RecordingNode.last_kwargs is None


def test_undeclared_input_fails_before_partner(rig):
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {"prompt": "radio", "unsupported": 1},
            timeout_s=5)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG
    assert "undeclared Partner input" in str(ei.value)
    assert _RecordingNode.last_kwargs is None


def test_non_dict_inputs_fail_before_partner(rig):
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], [], timeout_s=5)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG
    assert "partner inputs must be a dict" in str(ei.value)
    assert _RecordingNode.last_kwargs is None


def test_no_prompt_context_fails_closed(monkeypatch, rig):
    """Outside a ComfyUI execution and without bind_prompt_id the
    bridge must fail closed, naming the smoke escape hatch."""
    token = invoke._PROMPT_ID_OVERRIDE.set(None)
    try:
        with pytest.raises(CloudMediaError) as ei:
            invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    finally:
        invoke._PROMPT_ID_OVERRIDE.reset(token)
    assert ei.value.code is CloudErrorCode.MALFORMED_CONFIG
    assert "bind_prompt_id" in str(ei.value)


# ---------------------------------------------------------------------------
# Happy path + hidden-input injection + budget billing
# ---------------------------------------------------------------------------


def test_happy_path_returns_partner_result(rig):
    res = invoke.invoke_partner_node(
        rig["node_key"], {"prompt": "radio"}, timeout_s=30,
        estimated_usd=0.25)
    assert res["path"] == str(rig["src"])
    assert res["content_type"] == "video/mp4"
    assert res["raw_meta"]["provider_id"] == "TESTPROV"
    sess = _session(rig)
    assert sess is not None
    assert sess.spent_usd() == pytest.approx(0.25)
    assert sess.open_reservations() == []


def test_hidden_auth_and_usage_injected(rig):
    invoke.invoke_partner_node(
        rig["node_key"], {"prompt": "radio"}, timeout_s=30)
    kw = _RecordingNode.last_kwargs
    assert kw["prompt"] == "radio"
    assert kw["api_key_comfy_org"] == "test-key-123"
    assert kw["comfy_usage_source"] == invoke._USAGE_SOURCE
    assert kw["unique_id"].startswith("otr-cloud-")
    assert "auth_token_comfy_org" not in kw  # only the matching kind


def test_caller_inputs_never_overridden(rig):
    invoke.invoke_partner_node(
        rig["node_key"],
        {"prompt": "radio", "unique_id": "caller-owned"},
        timeout_s=30)
    assert _RecordingNode.last_kwargs["unique_id"] == "caller-owned"


def test_row_without_matching_auth_hidden_fails_auth(rig):
    row = _fake_row()
    row["inputs"]["hidden"] = {"auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG"}
    invoke._set_rows_for_tests({rig["node_key"]: row})
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=5)
    assert ei.value.code is CloudErrorCode.AUTH


def test_budget_ceiling_blocks_reserve(rig):
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {}, timeout_s=5, estimated_usd=9.99)
    assert ei.value.code is CloudErrorCode.BUDGET


# ---------------------------------------------------------------------------
# Watchdog: interrupt, timeout, heartbeat
# ---------------------------------------------------------------------------


def test_interrupt_cancels_and_bills(monkeypatch, rig):
    async def _sleepy(self, **kwargs):
        await asyncio.sleep(120)

    monkeypatch.setattr(_RecordingNode, "EXECUTE_NORMALIZED_ASYNC", _sleepy)
    monkeypatch.setattr(invoke, "WATCHDOG_TICK_S", 0.05)
    flag = {"hit": False}

    def _interrupted():
        flag["hit"] = True
        return True

    monkeypatch.setattr(invoke, "_interrupt_requested", _interrupted)
    start = time.monotonic()
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {}, timeout_s=60, estimated_usd=0.10)
    assert ei.value.code is CloudErrorCode.INTERRUPTED
    assert time.monotonic() - start < 10
    assert flag["hit"]
    # ambiguous outcome -> billed estimate, exposure closed
    sess = _session(rig)
    assert sess.spent_usd() == pytest.approx(0.10)
    assert sess.open_reservations() == []


def test_timeout_cancels_and_bills(monkeypatch, rig):
    async def _sleepy(self, **kwargs):
        await asyncio.sleep(120)

    monkeypatch.setattr(_RecordingNode, "EXECUTE_NORMALIZED_ASYNC", _sleepy)
    monkeypatch.setattr(invoke, "WATCHDOG_TICK_S", 0.05)
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {}, timeout_s=0.3, estimated_usd=0.10)
    assert ei.value.code is CloudErrorCode.TIMEOUT
    sess = _session(rig)
    assert sess.spent_usd() == pytest.approx(0.10)


def test_heartbeat_emitted_during_long_call(monkeypatch, rig):
    async def _briefly_slow(self, **kwargs):
        await asyncio.sleep(0.4)
        return (str(rig["src"]),)

    monkeypatch.setattr(_RecordingNode, "EXECUTE_NORMALIZED_ASYNC",
                        _briefly_slow)
    monkeypatch.setattr(invoke, "WATCHDOG_TICK_S", 0.05)
    monkeypatch.setattr(invoke, "HEARTBEAT_EVERY_S", 0.1)
    invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert len(rig["beats"]) >= 2


def test_heartbeat_cadence_within_requirement():
    assert invoke.HEARTBEAT_EVERY_S <= 30.0  # pass04: tick <= 30s


def test_headless_prompt_server_progress_sink_installs_stub(monkeypatch):
    class _PromptServer:
        instance = None

    monkeypatch.setitem(
        sys.modules, "server", types.SimpleNamespace(PromptServer=_PromptServer))

    with invoke._headless_prompt_server_progress_sink():
        assert _PromptServer.instance is not None
        _PromptServer.instance.send_progress_text("progress", "node-1")

    assert _PromptServer.instance is None


def test_headless_prompt_server_progress_sink_preserves_real_server(monkeypatch):
    class _RealServer:
        def __init__(self):
            self.calls = []

        def send_progress_text(self, *args):
            self.calls.append(args)

    class _PromptServer:
        instance = _RealServer()

    existing = _PromptServer.instance
    monkeypatch.setitem(
        sys.modules, "server", types.SimpleNamespace(PromptServer=_PromptServer))

    with invoke._headless_prompt_server_progress_sink():
        assert _PromptServer.instance is existing
        _PromptServer.instance.send_progress_text("progress", "node-1")

    assert _PromptServer.instance is existing
    assert existing.calls == [("progress", "node-1")]


# ---------------------------------------------------------------------------
# Failure mapping + settlement
# ---------------------------------------------------------------------------


def test_provider_rejection_releases_budget(monkeypatch, rig):
    async def _rejects(self, **kwargs):
        raise RuntimeError("provider says: content policy")

    monkeypatch.setattr(_RecordingNode, "EXECUTE_NORMALIZED_ASYNC", _rejects)
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {}, timeout_s=30, estimated_usd=0.50)
    assert ei.value.code is CloudErrorCode.PROVIDER_REJECTED
    sess = _session(rig)
    assert sess.spent_usd() == 0.0  # released, never billed
    assert sess.open_reservations() == []


def test_auth_error_text_maps_to_auth(monkeypatch, rig):
    async def _unauthorized(self, **kwargs):
        raise RuntimeError("HTTP 401 Unauthorized")

    monkeypatch.setattr(_RecordingNode, "EXECUTE_NORMALIZED_ASYNC",
                        _unauthorized)
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert ei.value.code is CloudErrorCode.AUTH


def test_comfy_processing_interrupted_maps_to_interrupted():
    ProcessingInterrupted = type("ProcessingInterrupted", (Exception,), {})
    err = invoke._map_exception(
        ProcessingInterrupted("Task cancelled"), "cloud_fake_video")
    assert err.code is CloudErrorCode.INTERRUPTED
    assert "Task cancelled" in str(err)


def test_missing_output_file_is_corrupt_output(rig):
    _RecordingNode.behavior = lambda kwargs: (
        str(rig["tmp"] / "never_written.mp4"),)
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(
            rig["node_key"], {}, timeout_s=30, estimated_usd=0.10)
    assert ei.value.code is CloudErrorCode.CORRUPT_OUTPUT
    # ambiguous (provider may have charged) -> billed
    assert _session(rig).spent_usd() == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


def test_url_output_streams_to_temp(monkeypatch, rig):
    def _fake_stream(url, dest):
        assert url == "https://provider.example/out.mp4"
        dest.write_bytes(b"streamed-bytes")

    monkeypatch.setattr(invoke, "_stream_to_temp", _fake_stream)
    _RecordingNode.behavior = lambda kwargs: (
        "https://provider.example/out.mp4",)
    res = invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert res["content_type"] == "video/mp4"
    assert "partner_tmp" in res["path"]
    assert rig["node_key"] in res["path"]


def test_side_channel_string_becomes_job_id(rig):
    invoke._set_rows_for_tests({rig["node_key"]: _fake_row(
        return_types=["VIDEO", "STRING"], selected_output=0)})
    _RecordingNode.behavior = lambda kwargs: (str(rig["src"]), "job-abc-123")
    res = invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert res["provider_job_id"] == "job-abc-123"
    assert res["raw_meta"]["extra_outputs"]["output_1"] == "job-abc-123"


def test_node_output_object_unwrapped(rig):
    class _NodeOutput:
        def __init__(self, *args):
            self.args = args

    _RecordingNode.behavior = lambda kwargs: _NodeOutput(str(rig["src"]))
    res = invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert res["path"] == str(rig["src"])


def test_save_to_object_materialized(rig):
    class _FakeVideo:
        def save_to(self, path):
            with open(path, "wb") as fh:
                fh.write(b"container-bytes")

    _RecordingNode.behavior = lambda kwargs: (_FakeVideo(),)
    res = invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert res["path"].endswith(".mp4")
    assert res["content_type"] == "video/mp4"


def test_unrecognized_output_type_is_corrupt(rig):
    _RecordingNode.behavior = lambda kwargs: (object(),)
    with pytest.raises(CloudMediaError) as ei:
        invoke.invoke_partner_node(rig["node_key"], {}, timeout_s=30)
    assert ei.value.code is CloudErrorCode.CORRUPT_OUTPUT


# ---------------------------------------------------------------------------
# V3 IO.ComfyNode partner contract -- hidden via PREPARE_CLASS_CLONE, NOT kwargs
# (regression for: IdeogramV4.execute() got an unexpected keyword argument
# 'api_key_comfy_org' -- every comfy_api_nodes partner is a V3 node)
# ---------------------------------------------------------------------------


class _V3HiddenHolder:
    """Minimal stand-in for comfy_api's HiddenHolder: attribute access by
    lowercase name resolves the value stored under the Hidden ENUM VALUE (the
    uppercase TYPE) key, mirroring HiddenHolder.from_dict."""

    _MAP = {
        "unique_id": "UNIQUE_ID",
        "api_key_comfy_org": "API_KEY_COMFY_ORG",
        "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG",
        "comfy_usage_source": "COMFY_USAGE_SOURCE",
    }

    def __init__(self, hidden_inputs):
        self._d = dict(hidden_inputs or {})

    def __getattr__(self, name):
        type_key = _V3HiddenHolder._MAP.get(name)
        return self._d.get(type_key) if type_key else None


class _FakeV3Node:
    """Stands in for a V3 IO.ComfyNode partner: hidden is delivered via
    PREPARE_CLASS_CLONE -> cls.hidden; execute() only ever sees real inputs."""

    hidden = None
    last_kwargs = None
    captured_hidden = None
    behavior = None

    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data):
        clone = type("FakeV3Clone", (cls,), {})
        clone.hidden = _V3HiddenHolder((v3_data or {}).get("hidden_inputs"))
        return clone

    @classmethod
    async def EXECUTE_NORMALIZED_ASYNC(cls, **kwargs):
        _FakeV3Node.last_kwargs = kwargs
        _FakeV3Node.captured_hidden = cls.hidden
        return _FakeV3Node.behavior(kwargs)


def test_v3_partner_hidden_via_clone_not_kwargs(monkeypatch, rig):
    """The V3 fix: hidden auth/usage/id must reach the node via cls.hidden
    (PREPARE_CLASS_CLONE), and execute() must receive ONLY the real inputs --
    passing api_key_comfy_org as a kwarg is the exact bug that crashed cloud."""
    monkeypatch.setattr(invoke, "_resolve_node_class", lambda row: _FakeV3Node)
    _FakeV3Node.last_kwargs = None
    _FakeV3Node.captured_hidden = None
    _FakeV3Node.behavior = lambda kwargs: (str(rig["src"]),)

    res = invoke.invoke_partner_node(
        rig["node_key"], {"prompt": "radio"}, timeout_s=30, estimated_usd=0.10)
    assert res["path"] == str(rig["src"])

    kw = _FakeV3Node.last_kwargs
    assert kw == {"prompt": "radio"}  # ONLY real inputs reach execute()
    assert "api_key_comfy_org" not in kw
    assert "unique_id" not in kw
    assert "comfy_usage_source" not in kw

    holder = _FakeV3Node.captured_hidden
    assert holder is not None
    assert holder.api_key_comfy_org == "test-key-123"
    assert str(holder.unique_id).startswith("otr-cloud-")
    assert holder.comfy_usage_source == invoke._USAGE_SOURCE


# ---------------------------------------------------------------------------
# Real pin table loads + shape (uses the checked-in yaml, no network)
# ---------------------------------------------------------------------------


def test_real_partner_table_loads_and_has_kling(rig):
    invoke._set_rows_for_tests(None)  # drop the fake, read the real yaml
    rows = invoke.partner_rows()
    assert "cloud_kling_avatar" in rows
    row = rows["cloud_kling_avatar"]
    assert row["provider_id"] == "KLING"
    assert row["function"] == "EXECUTE_NORMALIZED_ASYNC"
    assert "sound_file" in row["inputs"]["required"]
    hidden = row["inputs"]["hidden"]
    assert "api_key_comfy_org" in hidden
    not_ok = {rid: r.get("status") for rid, r in rows.items()
              if r.get("status") != "OK"}
    assert not_ok in ({}, {"cloud_stability_audio": "MISSING"})
