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
        "OTR_CLOUD_MAX_CONCURRENCY_ACME",
        "OTR_CLOUD_MAX_CONCURRENCY_TESTPROV",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OTR_CLOUD_MEDIA_CACHE_DIR", str(tmp_path / "cache"))
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()
        cmb._PROMPT_API_KEYS.clear()
    yield
    with cmb._TABLE_LOCK:
        cmb._SESSIONS.clear()
        cmb._PROMPT_API_KEYS.clear()


# -- flags -----------------------------------------------------------------


def test_enable_flag_removed():
    """Operator directive 2026-07-02: no hidden enable switch -- the
    dropdown pick IS the enable (same clean break as OpenRouter C6)."""
    assert not hasattr(cmb, "is_cloud_media_enabled")
    assert "is_cloud_media_enabled" not in cmb.__all__


def test_mute_ok_roles_default_empty():
    """Operator amendment: reactivity default-on; opt-down list EMPTY."""
    assert cmb.mute_ok_roles() == frozenset()


def test_mute_ok_roles_parses(monkeypatch):
    monkeypatch.setenv("OTR_VIDEO_MUTE_OK_ROLES", "music_visual, character_video")
    assert cmb.mute_ok_roles() == {"music_visual", "character_video"}


# -- auth broker -----------------------------------------------------------


def test_auth_is_only_the_injected_key(monkeypatch):
    """Rip 2026-09-19: the hidden api_key_comfy_org is the ONE credential.
    An env var on the server box is ignored; there is no env kind, no file
    kind and no bearer kind left to resolve to."""
    monkeypatch.setenv("OTR_COMFY_API_KEY", "comfyui-envkey")
    auth = cmb.resolve_auth("hidden-key")
    assert auth.kind == "api_key_hidden" and auth.value == "hidden-key"


def test_auth_missing_fails_closed_naming_both_real_paths(monkeypatch):
    monkeypatch.setenv("OTR_COMFY_API_KEY", "ignored-on-the-server")
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.resolve_auth(None)
    assert ei.value.code is cmb.CloudErrorCode.AUTH
    text = str(ei.value)
    assert "api_key_comfy_org" in text
    assert "extra_data.api_key_comfy_org" in text
    assert "ignored-on-the-server" not in text


def test_auth_blank_is_missing():
    with pytest.raises(cmb.CloudMediaError):
        cmb.resolve_auth("   ")


def test_auth_repr_never_leaks():
    auth = cmb.resolve_auth("supersecret")
    assert "supersecret" not in repr(auth)


def test_stashed_prompt_key_opens_the_session():
    """The host node stashes its hidden input; the first partner call under
    that prompt opens the session with it and needs no explicit key."""
    assert cmb.stash_prompt_api_key("prompt-stash", "stashed-key")
    sess = cmb.get_or_create_session("prompt-stash")
    assert sess.auth.value == "stashed-key"
    cmb.teardown_session("prompt-stash")
    with cmb._TABLE_LOCK:
        assert "prompt-stash" not in cmb._PROMPT_API_KEYS


def test_stash_ignores_empty_and_non_string():
    assert not cmb.stash_prompt_api_key("prompt-e", None)
    assert not cmb.stash_prompt_api_key("prompt-e", "")
    assert not cmb.stash_prompt_api_key("prompt-e", 42)
    assert not cmb.stash_prompt_api_key("", "key")
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.get_or_create_session("prompt-e")
    assert ei.value.code is cmb.CloudErrorCode.AUTH


# -- provider ids + semaphores ----------------------------------------------


def test_provider_id_normalizes():
    assert cmb.normalize_provider_id("vidu") == "VIDU"
    assert cmb.normalize_provider_id("byte-dance") == "BYTE_DANCE"


def test_provider_id_invalid_fails():
    with pytest.raises(cmb.CloudMediaError) as ei:
        cmb.normalize_provider_id("bad id!")
    assert ei.value.code is cmb.CloudErrorCode.MALFORMED_CONFIG


def test_semaphore_defaults():
    assert cmb.provider_semaphore_size("vidu") == 8  # pinned default
    assert cmb.provider_semaphore_size("seedream") == 8  # same overlap as Vidu
    assert cmb.provider_semaphore_size("luma") == 8
    assert cmb.provider_semaphore_size("elevenlabs") == 8


def test_semaphore_env_override(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_MAX_CONCURRENCY_ACME", "3")
    assert cmb.provider_semaphore_size("acme") == 3


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


def test_budget_unset_has_no_local_ceiling():
    """Operator 2026-09-16: no fake USD cap. Unset means unlimited;
    the wallet 402 is the stop. Explicit 0 remains spend-off."""
    import os
    os.environ.pop("OTR_CLOUD_MEDIA_BUDGET_USD", None)
    s = cmb.get_or_create_session("prompt-b0", hidden_api_key="k")
    assert s.budget_ceiling_usd is None
    rid = s.reserve(10_000.0)
    s.release(rid)


def test_budget_explicit_zero_fails_closed():
    """An EXPLICIT 0 remains a deliberate spend-off: every reserve fails."""
    import os
    os.environ["OTR_CLOUD_MEDIA_BUDGET_USD"] = "0"
    s = cmb.get_or_create_session("prompt-b0z", hidden_api_key="k")
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


def test_is_cloud_budget_error_walks_cause_chain():
    budget = cmb.CloudMediaError(
        cmb.CloudErrorCode.BUDGET, "reserve $0.5000")
    wrap = RuntimeError("shot failed; fallbacks are disabled")
    wrap.__cause__ = budget
    assert cmb.is_cloud_budget_error(budget)
    assert cmb.is_cloud_budget_error(wrap)
    assert not cmb.is_cloud_budget_error(RuntimeError("timeout"))
    assert not cmb.is_cloud_budget_error(None)
    rejected = cmb.CloudMediaError(
        cmb.CloudErrorCode.PROVIDER_REJECTED,
        "cloud_ltx25_i2v: HTTP 402 Payment Required")
    assert cmb.is_cloud_budget_error(rejected)
    wrap402 = RuntimeError("shot failed")
    wrap402.__cause__ = rejected
    assert cmb.is_cloud_budget_error(wrap402)
    assert cmb.is_wallet_empty_message("HTTP 402: Payment Required")
    assert cmb.is_wallet_empty_message("HTTP 402 Unauthorized")
    assert cmb.is_wallet_empty_message("insufficient credits")
    assert not cmb.is_wallet_empty_message("HTTP 500 upstream")
    assert not cmb.is_wallet_empty_message("job 1402 payment pending")
    assert not cmb.is_wallet_empty_message("HTTP 401 Unauthorized")
    assert cmb.is_auth_failure_message("HTTP 401 Unauthorized")
    assert cmb.is_auth_failure_message("HTTP 401")
    assert not cmb.is_auth_failure_message("Failed on request 1401-abc")
    assert not cmb.is_auth_failure_message("HTTP 500 upstream")


def test_error_codes_canonical_spelling():
    expected = {
        "malformed_config", "unsupported_schema", "incompatible_profile",
        "gated_by_flag", "auth", "budget", "retryable_transport",
        "provider_rejected", "content_refused", "timeout", "interrupted",
        "corrupt_output", "orphaned_job",
    }
    assert {c.value for c in cmb.CloudErrorCode} == expected


# -- content-policy refusals ----------------------------------------------------


def test_is_content_policy_message_matches_every_provider_spelling():
    """One verdict, five separator dialects -- see the needle note."""
    # LTX 2.5, the proven live 2026-09-16 refusal.
    assert cmb.is_content_policy_message(
        'Task failed: {"error": {"type": "content_filtered_error", '
        '"message": "Content filtered due to policy restrictions"}}')
    # BFL statuses, Gemini, ByteDance, Comfy's own violation code.
    assert cmb.is_content_policy_message('{"status": "Content Moderated"}')
    assert cmb.is_content_policy_message('{"status": "Request Moderated"}')
    assert cmb.is_content_policy_message("IMAGE_PROHIBITED_CONTENT")
    assert cmb.is_content_policy_message(
        "OutputAudioSensitiveContentDetected.PolicyViolation")
    assert cmb.is_content_policy_message("image_content_policy_violation")
    assert cmb.is_content_policy_message("flagged by content moderation")


def test_content_policy_needles_cannot_swallow_an_ordinary_fault():
    """A false match launders a broken render into a publishable episode.

    Every string here matched a needle that was CUT on 2026-09-16 review. A
    miss only restores the old fail-loud behavior; a false match floors a beat
    that a real fix would have rendered, which is the worse direction.
    """
    benign = [
        # cut needle 'policy restriction' -- systemic, would floor EVERY beat
        "ProxyError: 403 Forbidden - request blocked by policy restriction",
        "AccessDenied: denied by a policy restriction on the account",
        # cut needle 'safety system' -- a retryable outage, not a verdict
        '{"message": "our safety system is temporarily unavailable, retry"}',
        # cut needle 'safety filter' -- a MODEL FILE whose name contains it
        "FileNotFoundError: upscale_models/safety_filter_v2.pth",
        # cut needle 'sensitive content' -- ordinary English, echo-reachable
        '{"raw": "prompt: keep sensitive content out of frame"}',
        # ordinary faults that must always fail loud
        "HTTP 500 upstream", "CUDA out of memory",
        "HTTP 402 Payment Required", "HTTP 401 Unauthorized",
        "ffmpeg exited with code 1", "", None,
    ]
    for text in benign:
        assert not cmb.is_content_policy_message(text), text


def test_the_exact_live_ltx_refusal_maps_to_content_refused():
    """The 2026-09-16 bytes, verbatim -- this is why no credits are needed.

    A live one-shot LTX call would prove exactly one thing: that this string
    becomes CONTENT_REFUSED. The string is already in hand, so the test proves
    it for free and keeps proving it after the wallet is empty.
    """
    from nodes._otr_shared import cloud_media_invoke as cmi
    live = (
        "Polling aborted due to error: Task failed: "
        '{"id": "70d5715281164ec880371706c11fe8d9", '
        '"created_at": "2026-09-16T20:32:39.717Z", "status": "failed", '
        '"completed_at": "2026-09-16T20:32:44.904Z", '
        '"error": {"type": "content_filtered_error", '
        '"message": "Content filtered due to policy restrictions"}}')
    err = cmi._map_exception(Exception(live), "cloud_ltx25_i2v")
    assert err.code is cmb.CloudErrorCode.CONTENT_REFUSED
    assert cmb.is_content_refusal(err)


def test_is_content_refusal_prefers_the_stamped_code_over_prose():
    """A stamped verdict is a recorded fact; prose is the last resort."""
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED,
        "cloud_ltx25_i2v: Content filtered due to policy restrictions")
    assert cmb.is_content_refusal(refused)
    # render_shot wraps every failure, and the wrap message CONCATENATES the
    # inner chain -- so the walk is the contract, not a nicety.
    wrap = RuntimeError(
        "shot shot_shot_001_b40 engine 'cloud_ltx25_foley_plus' failed to "
        "render; fallbacks are disabled")
    wrap.__cause__ = refused
    assert cmb.is_content_refusal(wrap)
    # The code alone settles it with no policy words in the message at all.
    assert cmb.is_content_refusal(
        cmb.CloudMediaError(cmb.CloudErrorCode.CONTENT_REFUSED, "beat 40"))
    # No code anywhere -> prose may speak (a BYO engine, or a pre-stamp raise).
    assert cmb.is_content_refusal(RuntimeError("Content Moderated"))
    assert not cmb.is_content_refusal(RuntimeError("timeout"))
    assert not cmb.is_content_refusal(None)


def test_a_different_stamped_code_vetoes_the_prose():
    """The invoke boundary already adjudicated; prose must not overturn it.

    A timeout or transport blip whose body QUOTES the refusal words of some
    other job would otherwise floor a beat the boundary called retryable.
    """
    for code in (cmb.CloudErrorCode.RETRYABLE_TRANSPORT,
                 cmb.CloudErrorCode.TIMEOUT,
                 cmb.CloudErrorCode.PROVIDER_REJECTED,
                 cmb.CloudErrorCode.BUDGET):
        noisy = cmb.CloudMediaError(
            code, "cloud_ltx25_i2v: content_filtered_error seen on job 41")
        assert not cmb.is_content_refusal(noisy), code
        wrap = RuntimeError("fallbacks are disabled")
        wrap.__cause__ = noisy
        assert not cmb.is_content_refusal(wrap), code
    # A refusal is not a spend halt, and a spend halt is not a refusal.
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")
    assert not cmb.is_cloud_budget_error(refused)
    assert not cmb.is_content_refusal(
        cmb.CloudMediaError(cmb.CloudErrorCode.BUDGET, "reserve $0.50"))


def test_content_refused_releases_its_reservation():
    """A 5-second verdict produced no media; billing it charges for nothing."""
    from nodes._otr_shared import cloud_media_invoke as cmi
    assert cmb.CloudErrorCode.CONTENT_REFUSED in cmi._RELEASE_CODES


# -- job-scoped vs run-scoped (operator 2026-09-16) -----------------------------


def test_job_scoped_and_run_scoped_codes_partition_the_taxonomy():
    """Every code is job-scoped, run-scoped, or deliberately neither."""
    job, run = cmb.JOB_SCOPED_CODES, cmb.RUN_SCOPED_CODES
    assert not (job & run)
    # INTERRUPTED is in NEITHER on purpose: a cancel is the operator saying
    # stop, and turning that into 40 floored beats would be obscene.
    assert set(cmb.CloudErrorCode) - job - run == {
        cmb.CloudErrorCode.INTERRUPTED}
    # The run-scoped set is the "nothing will ever render" set.
    assert cmb.CloudErrorCode.AUTH in run
    assert cmb.CloudErrorCode.BUDGET in run
    assert cmb.CloudErrorCode.CONTENT_REFUSED in job
    assert cmb.CloudErrorCode.TIMEOUT in job


def test_cloud_job_failure_code_reads_only_a_stamped_verdict():
    """A failed output on ONE cloud job must not break the system."""
    for code in cmb.JOB_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "cloud_ltx25_i2v: no clip")
        assert cmb.cloud_job_failure_code(err) is code
        wrap = RuntimeError("fallbacks are disabled")
        wrap.__cause__ = err
        assert cmb.cloud_job_failure_code(wrap) is code
    # Run-scoped codes, a cancel, and anything that never reached the cloud
    # boundary at all get NOTHING -- those still fail LOUD.
    for code in cmb.RUN_SCOPED_CODES | {cmb.CloudErrorCode.INTERRUPTED}:
        assert cmb.cloud_job_failure_code(
            cmb.CloudMediaError(code, "x")) is None
    assert cmb.cloud_job_failure_code(RuntimeError("torch blew up")) is None
    assert cmb.cloud_job_failure_code(MemoryError("cuda oom")) is None
    assert cmb.cloud_job_failure_code(None) is None
    # No prose fallback here, unlike is_content_refusal: "an exception during
    # a cloud beat" is far too wide a net to floor a beat on.
    assert cmb.cloud_job_failure_code(RuntimeError("Content filtered")) is None
