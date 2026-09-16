"""Cloud stills / helper fan-out: request many, assemble in object order."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_shared import cloud_fanout as cf
from nodes._otr_shared import cloud_media_backend as cmb
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines.render_errors import RenderError
from nodes import otr_image_gen_dispatcher as disp


def test_run_cloud_fanout_finishes_out_of_order_returns_in_item_order():
    finish = []
    lock = threading.Lock()
    items = ["a", "b", "c"]
    delays = {"a": 0.12, "b": 0.06, "c": 0.0}

    def execute(item):
        time.sleep(delays[item])
        with lock:
            finish.append(item)
        return item.upper()

    out = cf.run_cloud_fanout(
        items, item_id=lambda x: x, execute=execute, workers=3)
    assert finish[0] == "c", finish
    assert [out.results[i] for i in items] == ["A", "B", "C"]
    assert out.errors == {}
    assert out.stuck_ids == []


def test_run_cloud_fanout_keeps_first_item_error_while_later_jobs_land():
    finish = []
    lock = threading.Lock()

    def execute(item):
        if item == "a":
            time.sleep(0.08)
            raise RuntimeError("first-failed")
        time.sleep(0.0)
        with lock:
            finish.append(item)
        return item

    out = cf.run_cloud_fanout(
        ["a", "b"], item_id=lambda x: x, execute=execute, workers=2)
    assert "b" in finish
    assert "first-failed" in str(out.errors["a"])
    assert "b" in out.results
    assert out.halted_ids == []


def test_run_cloud_fanout_caps_in_flight_to_workers():
    inflight = {"n": 0, "peak": 0}
    lock = threading.Lock()

    def execute(item):
        with lock:
            inflight["n"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["n"])
        time.sleep(0.04)
        with lock:
            inflight["n"] -= 1
        return item

    out = cf.run_cloud_fanout(
        ["a", "b", "c", "d"], item_id=lambda x: x, execute=execute, workers=2)
    assert inflight["peak"] <= 2, inflight
    assert set(out.results) == {"a", "b", "c", "d"}
    assert out.halted_ids == []


def test_run_cloud_fanout_halts_remaining_on_budget_error():
    started = []
    lock = threading.Lock()

    def execute(item):
        with lock:
            started.append(item)
        if item == "a":
            raise cmb.CloudMediaError(
                cmb.CloudErrorCode.BUDGET, "reserve $0.5000")
        return item

    out = cf.run_cloud_fanout(
        ["a", "b", "c"], item_id=lambda x: x, execute=execute, workers=1)
    assert started == ["a"], started
    assert "a" in out.errors
    assert cmb.is_cloud_budget_error(out.errors["a"])
    assert out.halted_ids == ["b", "c"]
    assert out.stuck_ids == []
    assert out.results == {}


def test_run_cloud_fanout_workers_gt1_does_not_submit_after_budget():
    started = []
    lock = threading.Lock()

    def execute(item):
        with lock:
            started.append(item)
        if item == "a":
            time.sleep(0.02)
            raise cmb.CloudMediaError(
                cmb.CloudErrorCode.BUDGET, "reserve $0.5000")
        time.sleep(0.04)
        return item

    out = cf.run_cloud_fanout(
        ["a", "b", "c", "d"], item_id=lambda x: x, execute=execute, workers=2)
    assert "c" not in started and "d" not in started, started
    assert set(out.halted_ids) == {"c", "d"}
    assert "a" in out.errors
    assert out.stuck_ids == []


def test_cloud_budget_floor_sid_sees_wrapped_render_error():
    inner = cmb.CloudMediaError(
        cmb.CloudErrorCode.BUDGET, "reserve $0.5000")
    wrap = RenderError("shot s1 engine failed")
    wrap.__cause__ = inner
    assert rd._cloud_budget_floor_sid("s1", {"s1": wrap}, [])
    assert rd._cloud_budget_floor_sid("s2", {}, ["s2"])
    assert not rd._cloud_budget_floor_sid(
        "s3", {"s3": RuntimeError("boom")}, [])


def test_run_cloud_fanout_halts_on_partner_402_text():
    started = []
    lock = threading.Lock()

    def execute(item):
        with lock:
            started.append(item)
        if item == "a":
            raise cmb.CloudMediaError(
                cmb.CloudErrorCode.PROVIDER_REJECTED,
                "cloud_ltx25_i2v: HTTP 402 Payment Required")
        return item

    out = cf.run_cloud_fanout(
        ["a", "b", "c"], item_id=lambda x: x, execute=execute, workers=1)
    assert started == ["a"], started
    assert cmb.is_cloud_budget_error(out.errors["a"])
    assert out.halted_ids == ["b", "c"]


def test_render_driver_stamps_budget_floors_on_fanout_and_serial_paths():
    src = Path(rd.__file__).read_text(encoding="utf-8")
    assert src.count("new_shots.append(_stamp_budget_floor_shot(shot))") >= 3
    assert "cloud_spend_halt" in src
    assert "_shot_is_budget_floor(shot)" in src


def _img_stub(**kw):
    import types
    base = dict(
        name="img_stub", roles=rc.ROLES, default_roles=rc.ROLES,
        commercial_clean=True, requires_flag=None,
        required_inputs=("text_prompt",), engine_version="1",
        native=True,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pobj(cid, prompt, prompt_hash):
    return {"object_id": cid, "kind": "portrait", "role": "character_video",
            "char_id": cid, "w": 832, "h": 1216,
            "prompt": prompt, "prompt_hash": prompt_hash, "source": "llm"}


def _payload(*objs):
    return {"version": 1, "objects": list(objs)}


def _complete_video_models():
    return {
        "announcer_video_model": {"engine_id": "viz_mxc_cpu"},
        "music_video_model": {"engine_id": "viz_mxc_mandala"},
        "character_video_model": {"engine_id": "still_motion"},
    }


def _np_pixels(val):
    import numpy as np
    return np.full((8, 8, 3), int(val), dtype=np.uint8)


@pytest.fixture
def clean_image_registry():
    saved = dict(ireg._IMAGE_REGISTRY._registry)
    try:
        yield ireg._IMAGE_REGISTRY
    finally:
        ireg._IMAGE_REGISTRY._registry.clear()
        ireg._IMAGE_REGISTRY._registry.update(saved)


def test_cloud_stills_fan_out_but_commit_in_object_order(
        clean_image_registry, tmp_path, monkeypatch):
    monkeypatch.delenv("OTR_CLOUD_FANOUT", raising=False)
    monkeypatch.delenv("OTR_CLOUD_VIDEO_FANOUT", raising=False)
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="cloud_luma_photon_flash", native=False))
    finish = []
    lock = threading.Lock()
    delays = {"c0": 0.12, "c1": 0.06, "c2": 0.0}

    def render(req):
        oid = str(req.get("object_id") or "")
        time.sleep(delays[oid])
        with lock:
            finish.append(oid)
        return _np_pixels(ord(oid[-1]))

    policy = {
        "policy_version": 2,
        "image_models": {
            "character_image_model": {"engine_id": "cloud_luma_photon_flash"},
        },
        "video_models": _complete_video_models(),
        "seed": {"request_seed": 0},
    }
    ledger, _done, _report, _warnings = disp.dispatch_images(
        {"episode_id": "ep_cloud_still_fan",
         "cast": [{"char_id": "c0"}, {"char_id": "c1"}, {"char_id": "c2"}]},
        policy,
        _payload(
            _pobj("c0", "boot one", "h0"),
            _pobj("c1", "boot two", "h1"),
            _pobj("c2", "boot three", "h2"),
        ),
        gen_fn=render, output_dir=str(tmp_path),
        lockdir=tmp_path / "lease.lockdir")
    assert finish[0] == "c2", finish
    assert [row["object_id"] for row in ledger["images"]["images"]] == [
        "c0", "c1", "c2"]


def test_local_stills_never_open_the_cloud_pool(
        clean_image_registry, tmp_path, monkeypatch):
    class _SerialPool:
        def __init__(self, *a, **k):
            raise AssertionError(
                "ThreadPoolExecutor must not run on a local still mint")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cf.concurrent.futures, "ThreadPoolExecutor", _SerialPool)
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    policy = {
        "policy_version": 2,
        "image_models": {"character_image_model": {"engine_id": "flux_gen1"}},
        "video_models": _complete_video_models(),
        "seed": {"request_seed": 0},
    }
    ledger, _done, _report, _warnings = disp.dispatch_images(
        {"episode_id": "ep_local_still", "cast": [{"char_id": "c1"}]},
        policy, _payload(_pobj("c1", "weathered spacer", "ph1")),
        gen_fn=lambda _req: _np_pixels(8), output_dir=str(tmp_path),
        lockdir=tmp_path / "lease.lockdir")
    assert ledger["images"]["images"][0]["object_id"] == "c1"
