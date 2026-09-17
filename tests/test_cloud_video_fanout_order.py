"""Cloud video fan-out: request many clips, assemble in ledger order.

Local-GPU episodes stay on the historical serial walk -- they cannot
share VRAM. This file pins the two facts that would be easy to lose:

* clips may finish in any order
* ``new_shots`` / the clip map still follow ShotLock order
"""
from __future__ import annotations

import threading
import time

import pytest

from nodes._otr_shared import cloud_fanout as cf
from nodes._otr_shared import cloud_media_backend as cmb
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


_FINISH = []
_LOCK = threading.Lock()


class _CloudFanStub:
    name = "cloud_fanout_stub"
    family = "abstract"
    roles = ("retired_role_b",)
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    fallback_engine = None
    provider_side = True

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
                "family": self.family, "frame_count": 25,
                "path": "otr_beat_%s.mp4" % request["shot_id"]}

    def teardown(self, prepared):
        pass

    def render_clip(self, request, prepared):
        sid = str(request.get("shot_id") or "")
        # Later ledger rows finish FIRST so a completion-order commit
        # would scramble the episode.
        delay = {"shot_0000": 0.18, "shot_0001": 0.08, "shot_0002": 0.0}.get(
            sid, 0.0)
        time.sleep(delay)
        with _LOCK:
            _FINISH.append(sid)
        return {"raw": True}


class _LocalFanStub(_CloudFanStub):
    name = "local_fanout_stub"
    provider_side = False


class _BoomCloudStub(_CloudFanStub):
    name = "cloud_fanout_boom"

    def render_clip(self, request, prepared):
        sid = str(request.get("shot_id") or "")
        if sid == "shot_0000":
            time.sleep(0.12)
            raise RuntimeError("first-beat-failed")
        time.sleep(0.0)
        with _LOCK:
            _FINISH.append(sid)
        return {"raw": True}


@pytest.fixture
def stub_registry():
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    stubs = {}
    for cls in (_CloudFanStub, _LocalFanStub, _BoomCloudStub):
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
        shots.append({
            "shot_id": "shot_%04d" % i,
            "beat_id": "b%d" % i,
            "role": "retired_role_b",
            "engine_id": eid,
            "family": "abstract",
            "group_id": "g%d" % i,
            "target_frame_count": 25,
            "degradation_trail": [],
        })
    return rd.build_full_ledger({
        "video_revision": 1, "fps": 25, "shots": shots,
    })


@pytest.fixture(autouse=True)
def _clear_finish():
    _FINISH.clear()
    yield
    _FINISH.clear()


def test_vidu_default_concurrency_is_eight():
    assert cmb.provider_semaphore_size("vidu") == 8
    assert cmb.provider_semaphore_size("kling") == 1
    assert cmb.provider_semaphore_size("luma") == 8
    assert cmb.provider_semaphore_size("elevenlabs") == 8
    assert cmb.provider_semaphore_size("sonilo") == 8


def test_cloud_fanout_knob_prefers_new_name(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "3")
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "5")
    assert cf.cloud_fanout_workers() == 5
    monkeypatch.delenv("OTR_CLOUD_FANOUT", raising=False)
    assert cf.cloud_fanout_workers() == 3


def test_cloud_fanout_unset_defaults_to_four(monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_FANOUT", raising=False)
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    assert cf.cloud_fanout_workers() == 4


def test_cloud_only_episode_fans_out_but_commits_in_ledger_order(
        stub_registry, monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    out = rd.run_episode(_ledger(
        "cloud_fanout_stub", "cloud_fanout_stub", "cloud_fanout_stub"))
    assert _FINISH[0] == "shot_0002", (
        "the last shot must be allowed to land first -- otherwise this "
        "test is not actually proving out-of-order completion")
    assert [s["shot_id"] for s in out["ledger"]["video"]["shots"]] == [
        "shot_0000", "shot_0001", "shot_0002"]
    assert list(out["clips"]) == ["shot_0000", "shot_0001", "shot_0002"]
    assert [row["shot_id"] for row in out["trace"]] == [
        "shot_0000", "shot_0001", "shot_0002"]


def test_a_local_engine_keeps_the_serial_walk(stub_registry, monkeypatch):
    seen = []

    class _SerialPool:
        def __init__(self, *a, **k):
            raise AssertionError(
                "ThreadPoolExecutor must not run on a local-GPU episode")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cf.concurrent.futures, "ThreadPoolExecutor", _SerialPool)
    out = rd.run_episode(_ledger("local_fanout_stub", "local_fanout_stub"))
    assert [s["shot_id"] for s in out["ledger"]["video"]["shots"]] == [
        "shot_0000", "shot_0001"]


def test_mixed_local_and_cloud_stays_serial(stub_registry, monkeypatch):
    class _SerialPool:
        def __init__(self, *a, **k):
            raise AssertionError(
                "a mixed local+cloud episode must not fan out")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cf.concurrent.futures, "ThreadPoolExecutor", _SerialPool)
    out = rd.run_episode(_ledger("local_fanout_stub", "cloud_fanout_stub"))
    assert [s["engine_id"] for s in out["ledger"]["video"]["shots"]] == [
        "local_fanout_stub", "cloud_fanout_stub"]


def test_intra_beat_chain_still_allows_episode_fanout():
    """CHAIN last-frame starts stay serial INSIDE that beat. Other beats
    can still fly -- that is the later cloud parallel path."""
    section = {
        "shots": [
            {"engine_id": "cloud_vidu_q2_pro_fast_720p", "beat_id": "b0",
             "coverage_plan": {"join_mode": "jump"}},
            {"engine_id": "cloud_vidu_q2_pro_fast_720p", "beat_id": "b1",
             "coverage_plan": {"join_mode": "chain", "segment_count": 2}},
        ],
    }
    assert rd._shot_is_first_to_last_chain(section["shots"][0]) is False
    assert rd._shot_is_first_to_last_chain(section["shots"][1]) is True
    assert rd._should_fanout_cloud_episode(section, set()) is True


def test_cross_beat_last_frame_waits_on_the_predecessor():
    """When first-to-last is added later: fire everyone whose first frame
    is already on disk; hold the beat that starts on another clip's last
    frame until that clip lands."""
    a = {"shot_id": "shot_a", "engine_id": "cloud_vidu_q2_pro_fast_720p"}
    b = {"shot_id": "shot_b", "engine_id": "cloud_vidu_q2_pro_fast_720p",
         "starts_on_last_frame_of": "shot_a"}
    c = {"shot_id": "shot_c", "engine_id": "cloud_vidu_q2_pro_fast_720p"}
    assert rd.cloud_frame_predecessors(a) == ()
    assert rd.cloud_frame_predecessors(b) == ("shot_a",)
    ready0 = [s["shot_id"] for s in rd.cloud_shots_ready_now([a, b, c], ())]
    assert ready0 == ["shot_a", "shot_c"]
    ready1 = [s["shot_id"] for s in rd.cloud_shots_ready_now(
        [a, b, c], ("shot_a",))]
    assert ready1 == ["shot_a", "shot_b", "shot_c"]


def test_jump_and_single_cloud_shots_may_fanout():
    section = {
        "shots": [
            {"engine_id": "cloud_vidu_q2_pro_fast_720p", "beat_id": "b0",
             "coverage_plan": {"join_mode": "single"}},
            {"engine_id": "cloud_vidu_q2_pro_fast_720p", "beat_id": "b1",
             "coverage_plan": {"join_mode": "jump"}},
        ],
    }
    assert rd._should_fanout_cloud_episode(section, set()) is True


def test_fanout_one_forces_serial_cloud(stub_registry, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")
    assert rd._should_fanout_cloud_episode(
        {"shots": [
            {"engine_id": "cloud_fanout_stub", "beat_id": "b0"},
            {"engine_id": "cloud_fanout_stub", "beat_id": "b1"},
        ]},
        set(),
    ) is False


def test_first_ledger_failure_wins_even_if_a_later_clip_already_landed(
        stub_registry, monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    with pytest.raises(rd.RenderError):
        rd.run_episode(_ledger("cloud_fanout_boom", "cloud_fanout_boom"))
    assert "shot_0001" in _FINISH


def test_run_episode_holds_a_last_frame_dependent_until_the_pred_lands(
        stub_registry, monkeypatch):
    """The later first-to-last path: B cannot even START until A has
    landed. Independent C still flies in wave 1."""
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    started = []
    started_lock = threading.Lock()
    a_done = threading.Event()

    class _ChainStub(_CloudFanStub):
        name = "cloud_fanout_readyq"

        def render_clip(self, request, prepared):
            sid = str(request.get("shot_id") or "")
            with started_lock:
                started.append(sid)
            if sid == "shot_0001":
                assert a_done.is_set(), (
                    "dependent beat started before its predecessor landed")
            if sid == "shot_0000":
                time.sleep(0.12)
                a_done.set()
            with _LOCK:
                _FINISH.append(sid)
            return {"raw": True}

    inst = _ChainStub()
    vreg.register(inst)
    ledger = _ledger(
        "cloud_fanout_readyq", "cloud_fanout_readyq", "cloud_fanout_readyq")
    ledger["video"]["shots"][1]["starts_on_last_frame_of"] = "shot_0000"
    out = rd.run_episode(ledger)
    assert "shot_0000" in started
    assert "shot_0001" in started
    assert "shot_0002" in started
    assert started.index("shot_0001") > started.index("shot_0000")
    assert list(out["clips"]) == ["shot_0000", "shot_0001", "shot_0002"]
    assert _FINISH[0] in ("shot_0002", "shot_0000")


def test_sanctioned_gap_beat_is_not_submitted_on_fanout(
        stub_registry, monkeypatch):
    """Gap beats stay in the ledger and never enter the pool."""
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    ledger = _ledger(
        "cloud_fanout_stub", "cloud_fanout_stub", "cloud_fanout_stub")
    ledger["images"] = {"required_scene_targets": [{
        "object_id": "scene_b1",
        "status": "sanctioned_gap",
        "beat_id": "b1",
        "reason": "model_refusal",
    }]}
    out = rd.run_episode(ledger)
    assert "shot_0001" not in _FINISH
    assert list(out["clips"]) == ["shot_0000", "shot_0002"]
    assert [s["shot_id"] for s in out["ledger"]["video"]["shots"]] == [
        "shot_0000", "shot_0001", "shot_0002"]


def test_ready_queue_cycle_is_loud(stub_registry, monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    ledger = _ledger("cloud_fanout_stub", "cloud_fanout_stub")
    ledger["video"]["shots"][0]["starts_on_last_frame_of"] = "shot_0001"
    ledger["video"]["shots"][1]["starts_on_last_frame_of"] = "shot_0000"
    with pytest.raises(rd.RenderError, match="never became ready"):
        rd.run_episode(ledger)


def test_fanout_workers_bind_the_comfy_prompt_id(stub_registry, monkeypatch):
    """Partner invoke is executor-thread local. Workers must see the
    snapshot bind_prompt_id, not 'no prompt context'."""
    from nodes._otr_shared import cloud_media_invoke as inv

    seen = []

    class _CtxStub(_CloudFanStub):
        name = "cloud_fanout_ctx"

        def render_clip(self, request, prepared):
            seen.append(inv.current_prompt_id())
            with _LOCK:
                _FINISH.append(str(request.get("shot_id") or ""))
            return {"raw": True}

    vreg.register(_CtxStub())
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    with inv.bind_prompt_id("fanout-live-prompt"):
        rd.run_episode(_ledger("cloud_fanout_ctx", "cloud_fanout_ctx"))
    assert seen
    assert set(seen) == {"fanout-live-prompt"}
