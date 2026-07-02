"""S0 offline tests for nodes/_otr_shared/cloud_media_backend.py.

No network, no torch, no ComfyUI server -- pure control-plane logic.
Build doc: docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md.
"""
from __future__ import annotations

import json
import threading

import pytest

from nodes._otr_shared import cloud_media_backend as cmb


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """Isolate env + session table + cache root per test."""
    for var in (
        "OTR_ENABLE_COMFY_CLOUD_MEDIA",
        "OTR_COMFY_API_KEY",
        "OTR_CLOUD_MEDIA_BUDGET_USD",
        "OTR_CLOUD_MEDIA_CACHE_DIR",
        "OTR_VIDEO_MUTE_OK_ROLES",
        "OTR_CLOUD_MAX_CONCURRENCY_KLING",
        "OTR_CLOUD_MAX_CONCURRENCY_TESTPROV",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OTR_CLOUD_MEDIA_CACHE_DIR", str(tmp_path / "cache"))
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()
    yield
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()


# -- flags -----------------------------------------------------------------


def test_flag_default_off():
    assert cmb.is_cloud_media_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_COMFY_CLOUD_MEDIA", "1")
    assert cmb.is_cloud_media_enabled() is True


def test_mute_ok_roles_default_empty():
    """Operator amendment: reactivity default-on; opt-down list EMPTY."""
    assert cmb.mute_ok_roles() == frozenset()


def test_mute_ok_roles_parses(monkeypatch):
    monkeypatch.setenv("OTR_VIDEO_MUTE_OK_ROLES", "music_visual, character_video")
    assert cmb.mute_ok_roles() == {"music_visual", "character_video"}


# -- auth broker -----------------------------------------------------------


def test_auth_env_beats_hidden(monkeypatch):
    monkeypatch.setenv("OTR_COMFY_API_KEY", "comfyui-envkey")
    auth = cmb.resolve_auth("hidden-key", "hidden-token")
    assert auth.kind == "api_key_env" and auth.value == "comfyui-envkey"


def test_auth_hidden_api_key_beats_bearer():
    auth = cmb.resolve_auth("hidden-key", "hidden-token")
    assert auth.kind == "api_key_hidden" and auth.value == "hidden-key"


def test_auth_bearer_last():
    auth = cmb.resolve_auth(None, "hidden-token")
    assert auth.kind == "bearer_hidden"


def test_auth_missing_fails_closed():
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.resolve_auth(None, None)
    assert ei.value.code is cmb.CloudErrorCode.AUTH
    assert "OTR_COMFY_API_KEY" in str(ei.value)


def test_auth_repr_never_leaks():
    auth = cmb.resolve_auth("supersecret", None)
    assert "supersecret" not in repr(auth)


# -- provider ids + semaphores ----------------------------------------------


def test_provider_id_normalizes():
    assert cmb.normalize_provider_id("kling") == "KLING"
    assert cmb.normalize_provider_id("byte-dance") == "BYTE_DANCE"


def test_provider_id_invalid_fails():
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.normalize_provider_id("bad id!")
    assert ei.value.code is cmb.CloudErrorCode.MALFORMED_CONFIG


def test_semaphore_defaults():
    assert cmb.provider_semaphore_size("kling") == 1  # pinned default
    assert cmb.provider_semaphore_size("recraft") == 2  # global default


def test_semaphore_env_override(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_MAX_CONCURRENCY_KLING", "3")
    assert cmb.provider_semaphore_size("kling") == 3


def test_semaphore_env_invalid(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_MAX_CONCURRENCY_TESTPROV", "zero")
    with pytest.raises(cmb.CloudMediaError):
        cmb.provider_semaphore_size("testprov")


# -- session table ------------------------------------------------------------


def _mk_session(budget="5.00", monkeypatch=None):
    import os
    os.environ["OTR_CLOUD_MEDIA_BUDGET_USD"] = budget
    return cmb.get_or_create_session("prompt-1", hidden_api_key="k")


def test_session_create_requires_auth():
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.get_or_create_session("prompt-x")
    assert ei.value.code is cmb.CloudErrorCode.AUTH


def test_session_reused_by_prompt_id():
    s1 = _mk_session()
    s2 = cmb.get_or_create_session("prompt-1")  # no auth needed on reuse
    assert s1 is s2


def test_session_teardown_logs_orphans(capsys):
    s = _mk_session()
    rid = s.reserve(1.0)
    s.submit(rid, "job-9")
    cmb.teardown_session("prompt-1")
    out = capsys.readouterr().out
    assert "ORPHANED_JOB" in out and "job-9" in out
    assert cmb.peek_session("prompt-1") is None


def test_session_sweep_evicts_stale(capsys, monkeypatch):
    s = _mk_session()
    s.created_at -= cmb.SESSION_SWEEP_MAX_AGE_S + 60
    rid = s.reserve(0.5)  # leave an open reservation
    cmb.get_or_create_session("prompt-2", hidden_api_key="k")  # triggers sweep
    out = capsys.readouterr().out
    assert "LEAKED_SESSION" in out and "prompt-1" in out
    assert cmb.peek_session("prompt-1") is None


# -- budget state machine -----------------------------------------------------


def test_budget_unset_fails_closed():
    import os
    os.environ.pop("OTR_CLOUD_MEDIA_BUDGET_USD", None)
    s = cmb.get_or_create_session("prompt-b0", hidden_api_key="k")
    with pytest.raises(cmb.CloudMediaError) as ei:
        s.reserve(0.01)
    assert ei.value.code is cmb.CloudErrorCode.BUDGET
    assert "OTR_CLOUD_MEDIA_BUDGET_USD" in str(ei.value)


def test_budget_reserve_bill_flow():
    s = _mk_session(budget="2.00")
    rid = s.reserve(1.25)
    s.submit(rid, "job-1")
    s.bill(rid, actual_usd=1.10)
    assert s.spent_usd() == pytest.approx(1.10)
    # actuals freed headroom vs the estimate
    rid2 = s.reserve(0.80)
    s.submit(rid2, "job-2")
    s.bill(rid2)  # estimate-billed
    assert s.spent_usd() == pytest.approx(1.90)


def test_budget_open_exposure_counts():
    s = _mk_session(budget="1.00")
    s.reserve(0.60)
    with pytest.raises(cmb.CloudMediaError) as ei:
        s.reserve(0.60)
    assert ei.value.code is cmb.CloudErrorCode.BUDGET


def test_budget_release_frees_exposure():
    s = _mk_session(budget="1.00")
    rid = s.reserve(0.9)
    s.release(rid)
    s.reserve(0.9)  # fits again


def test_budget_double_bill_rejected():
    s = _mk_session(budget="5.00")
    rid = s.reserve(1.0)
    s.submit(rid, None)
    s.bill(rid)
    with pytest.raises(cmb.CloudMediaError):
        s.bill(rid)


def test_budget_bill_without_submit_rejected():
    s = _mk_session(budget="5.00")
    rid = s.reserve(1.0)
    with pytest.raises(cmb.CloudMediaError):
        s.bill(rid)


def test_budget_concurrent_reserves_never_exceed():
    s = _mk_session(budget="10.00")
    errors = []

    def worker():
        for _ in range(20):
            try:
                rid = s.reserve(0.40)
                s.submit(rid, None)
                s.bill(rid)
            except cmb.CloudMediaError as e:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.spent_usd() <= 10.00 + 1e-9
    assert errors, "ceiling should have rejected some reserves"
    assert all(e.code is cmb.CloudErrorCode.BUDGET for e in errors)


# -- ledger --------------------------------------------------------------------


def test_ledger_appends_jsonl(tmp_path):
    s = _mk_session()
    s.episode_id = "ep_test"
    s.ledger_append({"event": "CACHE_HIT", "request_id": "r1"})
    s.ledger_append({"event": "BILLED", "request_id": "r2",
                     "estimated_usd": 0.5})
    lines = s.ledger_path().read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["prompt_id"] == "prompt-1"
    assert rec["episode_id"] == "ep_test"
    assert rec["event"] == "CACHE_HIT"


# -- error taxonomy -------------------------------------------------------------


def test_error_codes_canonical_spelling():
    expected = {
        "malformed_config", "unsupported_schema", "incompatible_profile",
        "gated_by_flag", "auth", "budget", "retryable_transport",
        "provider_rejected", "timeout", "interrupted", "corrupt_output",
        "orphaned_job",
    }
    assert {c.value for c in cmb.CloudErrorCode} == expected
