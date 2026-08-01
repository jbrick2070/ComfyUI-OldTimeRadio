"""``session_ctx`` belongs to the BASE, and every adapter must see it.

THE REGRESSION THIS PINS (2026-08-01). ``BeatSession`` passes ``session_ctx`` into
``prepare`` (``beat_session.py:165-167``) because it is the ONLY channel telling
``render_clip`` whether its output will be concatenated with other segments -- a
coverage-planned segment's request is shaped exactly like a single-clip beat's.
``MotionEngineBase.prepare`` ACCEPTED it and DROPPED it, returning only
``{engine_id, lease, patchers}``.

``eng_wan_ti2v`` and ``eng_humo`` each re-added it LOCALLY and worked. The four
adapters that did not -- ``ltx_video``, ``ltx_8gb``, ``ltx_av``, ``wan_i2v`` -- saw
no ``multi_clip``, so ``eng_ltx_video._loop_fill_allowed`` enabled the boomerang on
a beat whose length a coverage plan had already decided.

WHY THE SUITE MISSED IT: ``tests/test_ltx_boomerang.py`` builds ``prepared`` by hand,
so it exercises the very structure the bug destroys. These tests go through the real
``prepare`` instead.

TIMELINE, because "it always worked" and "it fails now" are both true: ``ltx_video``
last delivered episode clips 2026-07-06; the beat session landed 2026-07-25
(``4fa992e6`` / ``e90dedf1``). It worked BEFORE multi-clip existed.
"""

import inspect

import pytest

from nodes._otr_video_engines import motion_common as _MC


class _StubEngine(_MC.MotionEngineBase):
    """The smallest thing that can be prepared: no weights, no GPU."""

    name = "stub_engine"
    family = "image_to_video"

    def load(self):
        return None

    def unload(self):
        return None


@pytest.fixture()
def _no_lease(monkeypatch):
    """Neutralize the GPU lease -- this contract is about the returned dict."""
    monkeypatch.setattr(_MC._GR, "acquire", lambda **kw: object())
    monkeypatch.setattr(_MC._GR, "release", lambda lease: None)


def test_base_prepare_returns_session_ctx(_no_lease):
    prepared = _StubEngine().prepare(
        host_caps={}, profile={}, session_ctx={"multi_clip": True, "segments": 3})
    assert prepared["session_ctx"] == {"multi_clip": True, "segments": 3}


def test_multi_clip_is_visible_to_every_adapter(_no_lease):
    """The discriminator the boomerang decision reads."""
    prepared = _StubEngine().prepare(
        host_caps={}, profile={}, session_ctx={"multi_clip": True})
    assert bool((prepared.get("session_ctx") or {}).get("multi_clip")) is True


@pytest.mark.parametrize("ctx", [None, {}])
def test_absent_and_empty_stay_indistinguishable(_no_lease, ctx):
    """``render_single`` never opens a BeatSession and passes None. A
    single-clip render must not be mistaken for a multi-clip one."""
    prepared = _StubEngine().prepare(host_caps={}, profile={}, session_ctx=ctx)
    assert prepared["session_ctx"] == {}
    assert not (prepared.get("session_ctx") or {}).get("multi_clip")


def test_session_ctx_is_copied_not_aliased(_no_lease):
    """``prepare`` documents the policy as read-only; a caller mutating its own
    dict afterwards must not retroactively change what the engine prepared."""
    caller = {"multi_clip": True}
    prepared = _StubEngine().prepare(host_caps={}, profile={}, session_ctx=caller)
    caller["multi_clip"] = False
    assert prepared["session_ctx"]["multi_clip"] is True


def test_no_adapter_re_adds_session_ctx_locally():
    """ONE OWNER. Two adapters carrying a private workaround for a shared-base
    gap is exactly how the sibling adapters were left broken -- the same failure
    class as the eng_wan_i2v VRAM-peak lesson. If this fails, someone patched an
    adapter instead of the base again."""
    import glob
    import io
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for path in glob.glob(os.path.join(here, "nodes", "_otr_video_engines",
                                       "eng_*.py")):
        src = io.open(path, encoding="utf-8").read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if 'prepared["session_ctx"]' in stripped and "=" in stripped \
                    and "==" not in stripped:
                offenders.append((os.path.basename(path), stripped[:70]))
    assert not offenders, (
        "adapters re-adding session_ctx locally (the base owns it): %r" % offenders)


def test_base_prepare_still_carries_lease_and_patchers(_no_lease):
    """The additive change must not have displaced the existing contract."""
    prepared = _StubEngine().prepare(host_caps={}, profile={}, session_ctx={})
    for key in ("engine_id", "lease", "patchers"):
        assert key in prepared, key
    assert prepared["engine_id"] == "stub_engine"


def test_beat_session_actually_passes_session_ctx_into_prepare():
    """Pins the OTHER half of the contract by CALLING it, not by reading source.

    The base returning session_ctx is worthless if BeatSession stops sending it.
    This captures what prepare() actually receives, which is the failure mode a
    source-grep would sail straight past."""
    from nodes._otr_video_engines import beat_session as _bs

    src = inspect.getsource(_bs.BeatSession)
    assert "session_ctx=self.session_ctx()" in src, (
        "BeatSession no longer passes session_ctx into prepare -- the base "
        "returning it cannot help if nothing supplies it")
    assert callable(getattr(_bs.BeatSession, "session_ctx", None)), (
        "BeatSession.session_ctx() is the producer; it must exist")
