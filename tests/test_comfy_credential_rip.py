"""Rip 2026-09-19: the Comfy credential is ONLY the api_key_comfy_org hidden
input ComfyUI injects -- app sign-in, or a headless submitter's extra_data.

Before this the pack had three credential sources (env var, hidden input,
pack key file) resolved in two different orders by two resolvers, and the
lane flag alone satisfied the writer's gate. These tests pin the new shape
at its real wiring sites, not just on the helpers (a helper test proves the
helper; only source inspection proves the call exists where it must).
"""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import pytest

from nodes import _otr_comfy_backend as occ
from nodes._otr_shared import cloud_media_backend as cmb
from nodes._otr_shared import cloud_media_invoke as invoke
from nodes._otr_shared import cloud_balance_preflight as cbp
from nodes._otr_shared import api_key_files as keys

REPO = Path(__file__).resolve().parents[1]
HIDDEN_NAME = "api_key_comfy_org"
HIDDEN_TYPE = "API_KEY_COMFY_ORG"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    occ.clear_auth()
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()
        cmb._PROMPT_API_KEYS.clear()
    yield
    occ.clear_auth()
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()
        cmb._PROMPT_API_KEYS.clear()


# --- every host node that can spend Comfy credits declares the hidden input

def _host_classes():
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
    from nodes.otr_image_gen_dispatcher import OTRImageGenDispatcher
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    from nodes.stable_audio_theme import StableAudioTheme
    from nodes.announcer_voice import AnnouncerVoice
    from nodes.batch_character_voices import BatchCharacterVoices
    from nodes._otr_workflow_validator import WorkflowValidator
    from nodes.otr_shot_lock import OTRShotLock
    from nodes.otr_meta_brief_image_prompt import OTRMetaBriefImagePromptGen
    return [OTR_LedgerScriptWriter, OTRImageGenDispatcher, OTRVideoRenderBatch,
            StableAudioTheme, AnnouncerVoice, BatchCharacterVoices,
            WorkflowValidator, OTRShotLock, OTRMetaBriefImagePromptGen]


def test_every_credit_spending_host_declares_the_hidden_key():
    for cls in _host_classes():
        hidden = cls.INPUT_TYPES().get("hidden") or {}
        assert hidden.get(HIDDEN_NAME) == HIDDEN_TYPE, cls.__name__
        fn = getattr(cls, cls.FUNCTION)
        assert HIDDEN_NAME in inspect.signature(fn).parameters, cls.__name__


def test_media_hosts_stash_the_key_at_the_top_of_execute():
    """The wiring, at its real site: each media host's execute calls
    stash_comfy_api_key(api_key_comfy_org) before anything can invoke a
    partner node."""
    from nodes.otr_image_gen_dispatcher import OTRImageGenDispatcher
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    from nodes.stable_audio_theme import StableAudioTheme
    from nodes._otr_voice_node_common import OTRVoiceNodeBase
    for cls in (OTRImageGenDispatcher, OTRVideoRenderBatch, StableAudioTheme,
                OTRVoiceNodeBase):
        src = inspect.getsource(getattr(cls, "generate", None)
                                or getattr(cls, cls.FUNCTION))
        assert "stash_comfy_api_key(api_key_comfy_org)" in src, cls.__name__


def test_llm_hosts_capture_the_key_through_set_auth():
    """The writer and ShotLock spend through the LLM backend, not the media
    session: each hands its hidden input to set_auth at the top of execute."""
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
    from nodes.otr_shot_lock import OTRShotLock
    from nodes.otr_meta_brief_image_prompt import OTRMetaBriefImagePromptGen
    for cls in (OTR_LedgerScriptWriter, OTRShotLock, OTRMetaBriefImagePromptGen):
        src = inspect.getsource(getattr(cls, cls.FUNCTION))
        assert "set_auth(api_key=api_key_comfy_org)" in src, cls.__name__


def test_set_auth_replaces_so_a_bare_queue_cannot_spend_a_stale_key():
    """codex finding 2026-09-19: queue A injects a key, queue B injects none.
    B's set_auth(None) must CLEAR A's key, not keep it."""
    occ.set_auth(api_key="queue-a-key")
    assert occ._bearer() == "queue-a-key"
    occ.set_auth(api_key=None)
    assert occ._bearer() is None
    occ.set_auth(api_key="queue-a-key")
    occ.set_auth(api_key="   ")
    assert occ._bearer() is None


def test_no_host_declares_the_session_bearer():
    """PBUG-20260902-04: the registry scan flags the session-bearer hidden
    input. The rip must not have reintroduced it anywhere."""
    banned = "auth_token" + "_comfy_org"
    for cls in _host_classes():
        assert banned not in (cls.INPUT_TYPES().get("hidden") or {}), cls.__name__


# --- the stash reaches the session that the partner call opens

def test_stash_binds_the_key_to_the_current_prompt():
    with invoke.bind_prompt_id("prompt-rip"):
        assert invoke.stash_comfy_api_key("queue-key")
    sess = cmb.get_or_create_session("prompt-rip")
    assert sess.auth.kind == "api_key_hidden"
    assert sess.auth.value == "queue-key"


def test_stash_without_a_prompt_context_is_a_quiet_no_op():
    assert invoke.stash_comfy_api_key("queue-key") is False
    assert invoke.stash_comfy_api_key(None) is False


# --- the writer's backend reads nothing but the injected key

def test_writer_bearer_is_env_blind(monkeypatch):
    monkeypatch.setenv("OTR_COMFY_API_KEY", "server-env-key")
    assert occ._bearer() is None
    occ.set_auth(api_key="queue-key")
    assert occ._bearer() == "queue-key"


def test_key_files_have_no_comfy_lane():
    assert "comfy" not in keys.LANES
    src = inspect.getsource(keys)
    assert "comfy.secret" not in src.replace("No \"comfy\" lane", "")
    assert "resolve_lane_key(\"comfy\")" not in inspect.getsource(occ)
    assert "resolve_lane_key(\"comfy\")" not in inspect.getsource(cmb)


# --- the balance preflight measures the queue's own wallet

def test_validator_threads_the_hidden_key_into_the_balance_gate():
    from nodes import _otr_workflow_validator as v
    src = inspect.getsource(v._queue_time_readiness_gates)
    assert "ensure_prompt_cloud_balance(prompt, unique_id, comfy_api_key=comfy_api_key)" in src
    validate_src = inspect.getsource(v.WorkflowValidator.validate)
    assert validate_src.count("_queue_time_readiness_gates(prompt, unique_id, api_key_comfy_org)") == 2


def test_comfy_balance_uses_the_queue_key_and_warns_without_one():
    seen = {}

    def get_json(url, token):
        seen["token"] = token
        return 200, {"effective_balance_micros": 12345}

    result = cbp.comfy_balance(get_json=get_json, api_key="queue-key")
    assert seen["token"] == "queue-key"
    assert result.remaining_usd == pytest.approx(123.45)

    bare = cbp.comfy_balance(get_json=get_json, api_key=None)
    assert bare.severity == "warn"
    assert "api_key_comfy_org" in bare.error


def test_collect_verdicts_routes_the_comfy_wallet_through_the_queue_key(monkeypatch):
    seen = {}

    def fake_comfy_balance(*, api_key=None, **_kw):
        seen["api_key"] = api_key
        return cbp.BalanceResult(500.0, "", cbp.COMFY_HOST, "ok", "")

    monkeypatch.setattr(cbp, "comfy_balance", fake_comfy_balance)
    monkeypatch.setattr(cbp, "estimate_lines", lambda *a, **k: [
        cbp.CostLine(cbp.WALLET_COMFY, "writer", 1.0, 1.0, 1.0, "fixture"),
    ])
    verdicts = cbp.collect_verdicts({}, "1", comfy_api_key="queue-key")
    assert seen["api_key"] == "queue-key"
    assert verdicts and verdicts[0].wallet == cbp.WALLET_COMFY


# --- the headless submitter is the ONLY place the env var is read

def test_submit_prompt_packs_the_submitter_env_into_extra_data(monkeypatch):
    import scripts.otr_api as api

    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"prompt_id": "pid-1", "node_errors": {}}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(api.requests, "post", fake_post)
    monkeypatch.setenv("OTR_COMFY_API_KEY", "submitter-key")
    assert api.submit_prompt({"1": {"class_type": "X", "inputs": {}}}) == "pid-1"
    assert captured["json"]["extra_data"] == {HIDDEN_NAME: "submitter-key"}

    monkeypatch.delenv("OTR_COMFY_API_KEY", raising=False)
    api.submit_prompt({"1": {"class_type": "X", "inputs": {}}})
    assert "extra_data" not in captured["json"]


def test_server_side_pack_never_reads_the_env_var():
    """Grep receipt for the rip: under nodes/ the variable may be NAMED in
    a hint or a comment, but never READ."""
    # Any env read spelled with the literal name: os.environ[...],
    # os.environ.get(...), os.getenv(...), otr_env.get(...), or a lane
    # tuple that names it. A read through an assembled name is out of
    # reach of a grep -- the LANES assertion below covers the one indirect
    # reader the pack has.
    read_pattern = re.compile(
        r"(?:environ(?:\.get)?|getenv|otr_env\.get)\s*[\[(]\s*[\"']OTR_COMFY_API_KEY")
    offenders = []
    for path in (REPO / "nodes").rglob("*.py"):
        if read_pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []
    assert not any(k == "OTR_COMFY_API_KEY"
                   for spec in keys.LANES.values() for k in spec["env"])
