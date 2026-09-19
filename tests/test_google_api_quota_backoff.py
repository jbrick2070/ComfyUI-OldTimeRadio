"""HTTP 429 is a rate limit before it is a billing fault (measured 2026-09-19
on the third live google_veo_low_1act leg: two of four Veo clips came back
429 while the other two rendered). The client retries a 429 with a bounded,
Retry-After-honouring backoff at every poster (post_json, get_json,
create_interaction); 401/403 and everything else pass straight through.
"""
from __future__ import annotations

import pytest

from nodes._otr_google_api import client as gc

# captured before the autouse fixture replaces it, for the one test of the
# real slicing sleep
_REAL_INTERRUPTIBLE_SLEEP = gc._interruptible_sleep


def _quota(status, retry_after=None):
    exc = gc.GoogleAPIBillingOrQuotaError("HTTP %d" % status)
    exc.http_status = status
    exc.retry_after_s = retry_after
    return exc


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """Record each requested 429 wait instead of sleeping. The fake keeps
    the real contract: it reports False when Cancel is pending."""
    slept = []

    def fake_wait(seconds):
        slept.append(float(seconds))
        return not gc._processing_interrupted()

    monkeypatch.setattr(gc, "_interruptible_sleep", fake_wait)
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)   # the outer loop's tiny sleeps
    # pin the Cancel probe (codex: an importable ComfyUI stub reporting True
    # would abort the ordinary backoff tests); cancel tests override it
    monkeypatch.setattr(gc, "_processing_interrupted", lambda: False)
    monkeypatch.delenv("OTR_GOOGLE_QUOTA_RETRIES", raising=False)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")
    return slept


def test_interruptible_sleep_slices_and_stops_on_cancel(monkeypatch):
    slices = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: slices.append(s))
    monkeypatch.setattr(gc, "_processing_interrupted", lambda: False)
    assert _REAL_INTERRUPTIBLE_SLEEP(2.5) is True
    assert slices == [1.0, 1.0, 0.5]
    ticks = {"n": 0}

    def interrupted():
        ticks["n"] += 1
        return ticks["n"] >= 3

    slices.clear()
    monkeypatch.setattr(gc, "_processing_interrupted", interrupted)
    assert _REAL_INTERRUPTIBLE_SLEEP(30.0) is False
    assert len(slices) == 2


def test_429_is_retried_with_doubling_backoff_then_succeeds(_fast):
    calls = {"n": 0}

    def post(path, payload, api_key, timeout_s):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _quota(429)
        return {"ok": True}

    assert gc.post_json("/x", {"a": 1}, _post=post) == {"ok": True}
    assert calls["n"] == 3
    assert _fast == [5.0, 10.0]


def test_retry_after_header_wins_and_is_capped(_fast):
    calls = {"n": 0}

    def post(path, payload, api_key, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _quota(429, retry_after=17.0)
        if calls["n"] == 2:
            raise _quota(429, retry_after=900.0)
        return {}

    gc.post_json("/x", {}, _post=post)
    assert _fast == [17.0, 60.0]


def test_429_gives_up_after_the_retry_budget(_fast, monkeypatch):
    monkeypatch.setenv("OTR_GOOGLE_QUOTA_RETRIES", "2")

    def post(path, payload, api_key, timeout_s):
        raise _quota(429)

    with pytest.raises(gc.GoogleAPIBillingOrQuotaError):
        gc.post_json("/x", {}, _post=post)
    assert len(_fast) == 2


def test_401_and_403_never_retry(_fast):
    for status in (401, 403):
        calls = {"n": 0}

        def post(path, payload, api_key, timeout_s, _s=status):
            calls["n"] += 1
            raise _quota(_s)

        with pytest.raises(gc.GoogleAPIBillingOrQuotaError):
            gc.post_json("/x", {}, _post=post)
        assert calls["n"] == 1
    assert _fast == []


def test_create_interaction_keeps_one_quota_budget_across_its_transport_loop(_fast, monkeypatch):
    """codex 2026-09-19: a fresh 429 budget per outer transport attempt let
    (429 x4 -> 500) x2 -> 429 x4 run 12 waits. One budget for the whole
    call: after four 429 waits the fifth 429 raises, whatever 500s came
    between."""
    monkeypatch.setenv("OTR_GOOGLE_MAX_RETRIES", "2")
    script = [429, 429, 500, 429, 429, 500, 429]
    calls = {"n": 0}

    def post(path, payload, api_key, timeout_s):
        calls["n"] += 1
        status = script[min(calls["n"] - 1, len(script) - 1)]
        if status == 429:
            raise _quota(429)
        exc = gc.GoogleAPIError("HTTP 500")
        exc.http_status = 500
        raise exc

    with pytest.raises(gc.GoogleAPIBillingOrQuotaError):
        gc.create_interaction({"model": "m", "input": "x"}, _post=post)
    quota_waits = [s for s in _fast if s >= 5.0]
    assert len(quota_waits) == 4
    assert calls["n"] == 7


def test_cancel_during_the_wait_sends_no_further_request(_fast, monkeypatch):
    """codex 2026-09-19: Cancel must stop the retries, not just shorten one
    sleep. The wait is sliced and re-checks the interrupt; a Cancel re-raises
    the pending 429 without another POST."""
    # first check (before the wait) passes; the check inside the wait cancels
    state = {"cancel_after": 1, "checks": 0}

    def interrupted():
        state["checks"] += 1
        return state["checks"] > state["cancel_after"]

    monkeypatch.setattr(gc, "_processing_interrupted", interrupted)
    calls = {"n": 0}

    def post(path, payload, api_key, timeout_s):
        calls["n"] += 1
        raise _quota(429, retry_after=30.0)

    with pytest.raises(gc.GoogleAPIBillingOrQuotaError):
        gc.post_json("/x", {}, _post=post)
    assert calls["n"] == 1            # cancelled inside the first wait: no second POST
    assert _fast == [30.0]            # one wait was requested, then cut short


def test_download_media_shares_the_backoff(_fast):
    calls = {"n": 0}

    def get(path, api_key, timeout_s, accept=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _quota(429)
        return b"mp4bytes"

    assert gc.download_media("https://x/clip.mp4", _get=get) == b"mp4bytes"
    assert calls["n"] == 2 and _fast == [5.0]


def test_real_transport_classifies_429_and_reads_retry_after(_fast, monkeypatch):
    """Through the real _post_json: an HTTPError 429 with Retry-After lands
    as http_status 429 / retry_after_s, is waited exactly that long, and the
    next 200 body is returned."""
    import io
    import urllib.error
    from email.message import Message

    calls = {"n": 0}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = Message()
            hdrs["Retry-After"] = "7"
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", hdrs,
                io.BytesIO(b'{"error": {"message": "You exceeded your current quota"}}'))
        return _Resp(b'{"name": "operations/abc"}')

    monkeypatch.setattr(gc, "urlopen", fake_urlopen)
    out = gc.post_json("/v1beta/models/veo:predictLongRunning", {"instances": []})
    assert out == {"name": "operations/abc"}
    assert calls["n"] == 2
    assert _fast == [7.0]


def test_get_json_and_create_interaction_share_the_backoff(_fast):
    seen = {"get": 0, "post": 0}

    def get(path, api_key, timeout_s, accept=None):
        seen["get"] += 1
        if seen["get"] == 1:
            raise _quota(429)
        return b'{"done": true}'

    assert gc.get_json("/op", _get=get) == {"done": True}

    def post(path, payload, api_key, timeout_s):
        seen["post"] += 1
        if seen["post"] == 1:
            raise _quota(429)
        return {"status": "completed"}

    assert gc.create_interaction({"model": "m", "input": "x"}, _post=post)["status"] == "completed"
    assert seen == {"get": 2, "post": 2}
    assert _fast == [5.0, 5.0]
