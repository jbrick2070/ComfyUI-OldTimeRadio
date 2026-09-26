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


# --- plan 0k: ONE node declares the hidden key, and it cannot raise

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


def test_no_credit_spending_host_declares_the_hidden_key():
    """A V1 node that declares API_KEY_COMFY_ORG has the key formatted into
    /history's error record whenever it raises (ComfyUI v0.36.0
    execution.py: get_input_data -> format_value -> current_inputs). The nine
    hosts used to; several refuse on purpose."""
    for cls in _host_classes():
        hidden = cls.INPUT_TYPES().get("hidden") or {}
        assert HIDDEN_NAME not in hidden, cls.__name__
        fn = getattr(cls, cls.FUNCTION)
        assert HIDDEN_NAME not in inspect.signature(fn).parameters, cls.__name__


def test_exactly_one_registered_node_receives_the_key():
    """Every class the pack registers, not just the nine it used to be: the
    credential node is the only receiver, whatever gets added later."""
    from nodes.otr_comfy_credential import OTR_ComfyCredential
    receivers = []
    for path in sorted((REPO / "nodes").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if '"' + HIDDEN_TYPE + '"' in text and '"hidden"' in text:
            receivers.append(path.relative_to(REPO).as_posix())
    assert receivers == ["nodes/otr_comfy_credential.py"], receivers
    assert (OTR_ComfyCredential.INPUT_TYPES()["hidden"]
            == {HIDDEN_NAME: HIDDEN_TYPE})


def test_the_credential_node_never_raises(monkeypatch):
    """The whole point: a node that cannot raise never has its inputs written
    into an error record. Every failure below its surface is swallowed."""
    from nodes import otr_comfy_credential as cred
    node = cred.OTR_ComfyCredential()
    assert node.bind("queue-key") == (cred.KEY_PRESENT,)
    assert node.bind(None) == (cred.KEY_ABSENT,)
    assert node.bind("   ") == (cred.KEY_ABSENT,)
    assert node.bind(12345) == (cred.KEY_ABSENT,)

    def _boom(**_kw):
        raise RuntimeError("queue-key would be in this message")
    monkeypatch.setattr(occ, "set_auth", _boom)
    # A key that could not be bound is reported ABSENT (Cursor, 2e78e2b6):
    # the token must never claim a credential the spend paths do not have.
    assert node.bind("queue-key") == (cred.KEY_ABSENT,)
    import math
    assert math.isnan(cred.OTR_ComfyCredential.IS_CHANGED())


def test_the_credential_node_output_never_carries_the_key():
    from nodes import otr_comfy_credential as cred
    (token,) = cred.OTR_ComfyCredential().bind("sk-secret-queue-key")
    assert "sk-secret" not in token
    assert token in (cred.KEY_PRESENT, cred.KEY_ABSENT)


def test_the_credential_node_binds_both_spend_paths():
    """One call feeds the media sessions (the per-prompt stash) and the
    Comfy Credits writer backend (set_auth, bound to the same prompt)."""
    from nodes import otr_comfy_credential as cred
    with invoke.bind_prompt_id("prompt-0k"):
        cred.OTR_ComfyCredential().bind("queue-key")
    assert cmb.prompt_api_key("prompt-0k") == "queue-key"
    assert occ._auth == {"api_key": "queue-key", "prompt_id": "prompt-0k"}


def test_every_host_runs_after_the_credential_node_in_every_shipped_workflow():
    """ORDER IS THE MECHANISM: the hosts read what the credential node bound,
    so each must descend from it. One link into the Workflow Validator -- the
    only root -- is what does that; a rewire that breaks it fails here."""
    from tests._support.shipped_graphs import shipped_graphs
    host_types = {"OTR_LedgerScriptWriter", "OTR_WorkflowValidator",
                  "OTR_ImageGenDispatcher", "OTR_MetaBriefImagePromptGen",
                  "OTR_ShotLock", "OTR_VideoRenderBatch", "OTR_StableAudioTheme",
                  "OTR_BatchCharacterVoices", "OTR_AnnouncerVoice"}
    for path in shipped_graphs():
        wf = json.loads(Path(path).read_text(encoding="utf-8"))
        creds = [n["id"] for n in wf["nodes"] if n["type"] == "OTR_ComfyCredential"]
        assert len(creds) == 1, path.name
        down = {}
        for link in wf["links"]:
            down.setdefault(link[1], set()).add(link[3])
        seen, todo = set(), [creds[0]]
        while todo:
            for nxt in down.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        hosts = {n["id"] for n in wf["nodes"] if n["type"] in host_types}
        assert hosts and hosts <= seen, (path.name, sorted(hosts - seen))


def test_a_key_bound_to_an_earlier_prompt_is_never_spent(monkeypatch):
    """M1 (plan 0k): only the credential node sets auth now, so a later graph
    that lacks the node would still find the previous queue's key here. The
    binding refuses it once another prompt is executing."""
    occ.set_auth(api_key="queue-a-key", prompt_id="prompt-a")
    monkeypatch.setattr(cmb, "live_prompt_id", lambda: "prompt-a")
    assert occ._bearer() == "queue-a-key"
    monkeypatch.setattr(cmb, "live_prompt_id", lambda: "prompt-b")
    assert occ._bearer() is None
    # Cannot read the executing prompt: a BOUND key is refused, not trusted.
    monkeypatch.setattr(cmb, "live_prompt_id", lambda: None)
    assert occ._bearer() is None
    # A key set with no prompt id (a direct call outside any queue) is kept.
    occ.set_auth(api_key="direct-key")
    assert occ._bearer() == "direct-key"


def test_the_executing_prompt_is_never_swept(monkeypatch):
    """S1 (plan 0k): the key is stashed ONCE, at the start of the queue; a
    render longer than the six-hour sweep must not lose it."""
    now = 1_000_000.0
    with cmb._TABLE_LOCK:
        cmb._PROMPT_API_KEYS["running"] = ("k1", now - cmb.SESSION_SWEEP_MAX_AGE_S - 10)
        cmb._PROMPT_API_KEYS["finished"] = ("k2", now - cmb.SESSION_SWEEP_MAX_AGE_S - 10)
    monkeypatch.setattr(cmb, "live_prompt_id", lambda: "running")
    with cmb._TABLE_LOCK:
        cmb._sweep_locked(now)
    assert cmb.prompt_api_key("running") == "k1"
    assert cmb.prompt_api_key("finished") is None


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

def test_validator_threads_the_bound_key_into_the_balance_gate():
    from nodes import _otr_workflow_validator as v
    src = inspect.getsource(v._queue_time_readiness_gates)
    assert "ensure_prompt_cloud_balance(prompt, unique_id, comfy_api_key=comfy_api_key)" in src
    validate_src = inspect.getsource(v.WorkflowValidator.validate)
    assert validate_src.count("_queue_time_readiness_gates(prompt, unique_id, _queue_api_key())") == 2
    with invoke.bind_prompt_id("prompt-v"):
        cmb.stash_prompt_api_key("prompt-v", "queue-key")
        assert v._queue_api_key() == "queue-key"
    assert v._queue_api_key() is None           # no executing prompt


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
