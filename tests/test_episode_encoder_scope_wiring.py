"""The EPISODE boundary that owns an adapter's cross-beat residency.

WHY THE DRIVER OWNS THIS AND THE ENGINE CANNOT. The video registry builds each
adapter ONCE at import and hands back that same instance forever
(``engine_registry_base.py:148-155``), so an adapter that cached a heavy handle
on ``self`` would never release it -- the cache would outlive the episode, the
lane switch, and every later workflow in that server.

Every ENGINE-level hook was measured during the 2026-08-20 arc and all of them
run too often to be the release point: ``free_otr_pipeline_residue`` is the
per-SHOT load preflight, and ``teardown``/``unload`` run once per BEAT through
``BeatSession.close``. Releasing on any of those drops the cache immediately
before the thing that needed it. ``run_episode`` is the only boundary that
matches the lifetime, which is why the hooks live there.

THE TESTS THAT MATTER MOST ARE THE CLEANUP ONES. A scope that opens and never
closes is worse than no cache at all: it pins 8.86 GiB for the life of the
server, and it does so most readily on the FAILURE path, where an OOM is
exactly the state in which holding the encoder into the retry is worst.

CPU-safe: stub engines, no GPU, no model loads.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


#: A SHARED, ORDERED log across every stub in one episode.
#:
#: Per-engine event lists are not enough to catch the bug the panel found. With
#: the release nested inside the reclaim predicate, a local -> cloud hand-off
#: never released early -- but the episode's ``finally`` closed the scope at the
#: end anyway, so the local engine's OWN list still read ``["begin", "end"]``
#: and a per-engine assertion passed either way. What distinguishes the two is
#: WHEN the close happened relative to the next engine starting, and that is a
#: cross-engine ordering question.
_EPISODE_EVENTS = []


class _ScopeStubBase:
    """A stub that RECORDS its episode-scope lifecycle."""

    family = "abstract"
    roles = ("retired_role_b",)
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    fallback_engine = None

    def __init__(self):
        self.events = []
        self.open_scopes = 0
        self.generation = 0
        self.tokens_seen = []

    def load(self):
        pass

    def unload(self):
        pass

    def assert_usable(self, host_caps=None, profile=None, request_template=None):
        return self.name

    def prepare(self, host_caps=None, profile=None, session_ctx=None):
        return {"engine_id": self.name}

    def canonicalize(self, raw, request, profile=None):
        return {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25, "path": ""}

    def teardown(self, prepared):
        pass

    def begin_encoder_scope(self):
        self.events.append("begin")
        _EPISODE_EVENTS.append(("begin", self.name))
        self.open_scopes += 1
        self.generation += 1
        return self.generation

    def end_encoder_scope(self, token=None):
        """Mirrors the real adapter: a stale token is a no-op."""
        self.tokens_seen.append(token)
        if token is not None and token != self.generation:
            return
        self.events.append("end")
        _EPISODE_EVENTS.append(("end", self.name))
        self.open_scopes -= 1

    def release_encoder_cache(self):
        """The kill-switch path. Present on the stub because the real adapter
        has it, and because one test asserts the driver does NOT reach for it
        on a failing close -- an assertion that would be vacuous if the stub
        could not offer it in the first place."""
        self.events.append("release")
        _EPISODE_EVENTS.append(("release", self.name))
        self.open_scopes = 0

    def render_clip(self, request, prepared):
        _EPISODE_EVENTS.append(("render", self.name))
        return {"raw": True}


class _ScopeOK(_ScopeStubBase):
    name = "scope_ok"


class _ScopeOther(_ScopeStubBase):
    name = "scope_other"


class _ScopeCloud(_ScopeStubBase):
    """A CLOUD engine. The ``cloud_`` prefix is what
    ``_is_cloud_video_engine`` keys on, and it is the whole point of the
    local -> cloud regression test below."""

    name = "cloud_scope_stub"


class _ScopeFail(_ScopeStubBase):
    name = "scope_fail"

    def render_clip(self, request, prepared):
        raise RuntimeError("boom")


class _NoScopeStub(_ScopeStubBase):
    """An adapter that declares NOTHING. It must be untouched."""

    name = "scope_absent"
    begin_encoder_scope = None
    end_encoder_scope = None

    def render_clip(self, request, prepared):
        return {"raw": True}


@pytest.fixture(autouse=True)
def _fresh_event_log():
    """The shared ordering log is module state; never inherit another test's."""
    _EPISODE_EVENTS.clear()
    yield
    _EPISODE_EVENTS.clear()


@pytest.fixture
def stub_registry():
    """Snapshot the global video registry, add the stubs, restore after."""
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    stubs = {}
    for cls in (_ScopeOK, _ScopeOther, _ScopeFail, _NoScopeStub, _ScopeCloud):
        inst = cls()
        vreg.register(inst)
        stubs[inst.name] = inst
    try:
        yield stubs
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _ledger(*engine_ids):
    shots = []
    for i, eid in enumerate(engine_ids):
        shots.append({"shot_id": "shot_%04d" % i, "beat_id": "b%d" % i,
                      "role": "retired_role_b", "engine_id": eid,
                      "family": "abstract", "group_id": "g%d" % i,
                      "target_frame_count": 25, "degradation_trail": []})
    return rd.build_full_ledger({"video_revision": 1, "fps": 25,
                                 "shots": shots})


# --- the happy path -------------------------------------------------------- #
def test_an_episode_opens_a_scope_once_and_closes_it(stub_registry):
    eng = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok", "scope_ok", "scope_ok"))
    assert eng.events == ["begin", "end"], (
        "three beats on one engine must open ONE scope, not three")
    assert eng.open_scopes == 0


def test_the_scope_spans_the_beats_rather_than_each_beat(stub_registry):
    """The whole point: the handle must still be open while beat 2 renders.
    A per-beat open/close would show begin,end,begin,end and buy nothing."""
    eng = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok", "scope_ok"))
    assert eng.events.count("begin") == 1
    assert eng.events.index("begin") < eng.events.index("end")


# --- the cleanup paths, which are the ones that actually matter ------------ #
def test_a_FAILED_episode_still_closes_the_scope(stub_registry):
    """A raise anywhere in the loop must not strand the episode's handles.
    Without the ``finally`` this leaks 8.86 GiB for the life of the server,
    and it leaks it precisely when the failure was an OOM.

    THE ENGINE THAT FAILS IS THE ENGINE THAT OPENED THE SCOPE, deliberately.
    An earlier version of this test used ``("scope_ok", "scope_fail")``, and a
    QA pass caught that it proved the wrong thing: the ENGINE SWITCH closes
    ``scope_ok``'s scope at the hand-off, so the assertion passed via the
    hand-off path and never exercised the ``finally`` at all. One engine, two
    shots, second one fails -- now nothing but the ``finally`` can close it.
    """
    bad = stub_registry["scope_fail"]
    with pytest.raises(rd.RenderError):
        rd.run_episode(_ledger("scope_fail", "scope_fail"))
    assert bad.open_scopes == 0, "the failed episode stranded a scope"
    assert bad.events == ["begin", "end"]
    # No hand-off happened, so the close can only have come from the finally.
    assert [e for e in _EPISODE_EVENTS if e[0] == "begin"] == [
        ("begin", "scope_fail")], "a second engine would muddy the proof"


def test_the_failing_engines_own_scope_is_closed_too(stub_registry):
    bad = stub_registry["scope_fail"]
    with pytest.raises(rd.RenderError):
        rd.run_episode(_ledger("scope_fail"))
    assert bad.open_scopes == 0
    assert bad.events == ["begin", "end"]


def test_a_second_episode_starts_COLD(stub_registry):
    """The contract that makes this bounded rather than merely quiet: episode
    two must not inherit episode one's residency."""
    eng = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok"))
    rd.run_episode(_ledger("scope_ok"))
    assert eng.events == ["begin", "end", "begin", "end"]
    assert eng.open_scopes == 0


# --- switching engines mid-episode ----------------------------------------- #
def test_switching_engines_closes_the_outgoing_scope(stub_registry):
    """The driver already drains the prior engine's residue when an episode
    crosses engines. An open scope holding that engine's handles IS residue,
    so it must be released BEFORE that reclaim rather than survive it."""
    first = stub_registry["scope_ok"]
    second = stub_registry["scope_other"]
    rd.run_episode(_ledger("scope_ok", "scope_other"))
    assert first.events == ["begin", "end"]
    assert second.events == ["begin", "end"]
    assert first.open_scopes == second.open_scopes == 0


def test_a_local_to_CLOUD_switch_still_closes_the_local_scope(stub_registry):
    """THE REGRESSION THE PANEL CAUGHT, and the switch test above could not see.

    The release was originally nested inside
    ``_should_reclaim_between_engines``, which ends in
    ``and not _is_cloud_video_engine(this)`` (``render_driver.py:1885-1888``)
    because a VRAM reclaim only matters before a LOCAL engine loads. So a
    local -> CLOUD hand-off skipped the predicate entirely and never released
    the local engine's 8.86 GiB.

    The original switch test used two LOCAL stubs, so it passed either way.
    Whether to drain VRAM and who owns a handle are different questions that
    happened to share a line.
    """
    local = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok", "cloud_scope_stub"))
    assert local.open_scopes == 0

    # THE ASSERTION HAS TO BE ABOUT *ORDER*, NOT ABOUT THE FINAL STATE.
    # ``local.events == ["begin", "end"]`` passes under the OLD code too,
    # because the episode's ``finally`` closes every scope at the end anyway.
    # What actually distinguishes the fix is that the local scope closes
    # BEFORE the cloud engine renders -- under the old code it closed after.
    assert ("end", "scope_ok") in _EPISODE_EVENTS
    closed_at = _EPISODE_EVENTS.index(("end", "scope_ok"))
    rendered_at = _EPISODE_EVENTS.index(("render", "cloud_scope_stub"))
    assert closed_at < rendered_at, (
        "the local engine's 8.86 GiB was still held while the cloud engine "
        "rendered -- the release is nested inside the VRAM-reclaim predicate "
        "again, which skips cloud targets")


def test_local_to_cloud_to_local_reopens_a_fresh_scope(stub_registry):
    local = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok", "cloud_scope_stub", "scope_ok"))
    assert local.events == ["begin", "end", "begin", "end"]
    assert local.open_scopes == 0


def test_a_single_engine_episode_opens_exactly_one_scope(stub_registry):
    """REPLACES A VACUOUS TEST. The old assertion was ``open_scopes == 0``
    after a one-engine episode, which is ALSO true when the hooks are never
    called at all -- it passed with the whole feature removed. Assert the
    positive shape instead: opened once, closed once, in that order."""
    ok = stub_registry["scope_ok"]
    rd.run_episode(_ledger("scope_ok"))
    assert ok.events == ["begin", "end"]
    assert ok.open_scopes == 0
    assert [e for e in _EPISODE_EVENTS if e[0] in ("begin", "end")] == [
        ("begin", "scope_ok"), ("end", "scope_ok")]


# --- an adapter that declares nothing -------------------------------------- #
def test_an_adapter_without_the_hooks_is_untouched(stub_registry):
    """Duck-typed like ``session_identity`` and the optional
    ``frame_contract`` -- NOT like ``prepare``, which IS declared on the
    ``VideoEngine`` protocol (``registry.py:109``); an earlier version of this
    docstring cited it as the precedent and was wrong. Legacy wording kept
    below for the shape of the claim:
    ``frame_contract``: declaring nothing must be a no-op, not an error."""
    plain = stub_registry["scope_absent"]
    out = rd.run_episode(_ledger("scope_absent", "scope_absent"))
    assert out["trace"][0]["final_engine"] == "scope_absent"
    assert plain.events == []


def test_a_raising_begin_hook_never_fails_the_episode(stub_registry,
                                                      monkeypatch):
    """Residency is best-effort by long-standing contract. An adapter whose
    scope hook throws degrades to no caching; it does not lose the render."""
    eng = stub_registry["scope_ok"]

    def _boom():
        raise RuntimeError("scope refused")

    monkeypatch.setattr(eng, "begin_encoder_scope", _boom)
    out = rd.run_episode(_ledger("scope_ok"))
    assert out["trace"][0]["final_engine"] == "scope_ok"


def test_a_raising_end_hook_never_fails_the_episode(stub_registry,
                                                    monkeypatch):
    eng = stub_registry["scope_ok"]

    def _boom(token=None):
        raise RuntimeError("release refused")

    monkeypatch.setattr(eng, "end_encoder_scope", _boom)
    out = rd.run_episode(_ledger("scope_ok"))
    assert out["trace"][0]["final_engine"] == "scope_ok"


def test_a_raising_end_hook_does_NOT_globally_release(stub_registry,
                                                     monkeypatch):
    """THIS TEST ASSERTED THE OPPOSITE ONE ROUND AGO, and the r4 panel showed
    the old behaviour was the more dangerous one.

    An earlier fix reached for ``release_encoder_cache()`` when the polite
    close raised -- belt and braces. But that call drops the cache for EVERY
    owner, so one run's failing close would destroy a CONCURRENT run's live
    cache. Losing this run's share of a cache is a slow render; taking someone
    else's is a broken one.

    So a failed close now leaks this run's claim and says so, and must NOT
    reach for the global release.
    """
    eng = stub_registry["scope_ok"]
    released = []

    def _boom(token=None):
        raise RuntimeError("release refused")

    monkeypatch.setattr(eng, "end_encoder_scope", _boom)
    monkeypatch.setattr(eng, "release_encoder_cache",
                        lambda: released.append(True))
    rd.run_episode(_ledger("scope_ok"))
    assert released == [], (
        "a failing close evicted every other owner's cache as well")


def test_a_raising_end_AND_a_raising_release_still_finish_the_episode(
        stub_registry, monkeypatch):
    """Last resort must be loud, not fatal: the episode's frames are real and
    a residency failure may never destroy them."""
    eng = stub_registry["scope_ok"]

    def _boom(token=None):
        raise RuntimeError("refused")

    monkeypatch.setattr(eng, "end_encoder_scope", _boom)
    monkeypatch.setattr(eng, "release_encoder_cache", _boom)
    out = rd.run_episode(_ledger("scope_ok"))
    assert out["trace"][0]["final_engine"] == "scope_ok"
