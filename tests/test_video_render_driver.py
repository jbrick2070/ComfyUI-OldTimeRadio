"""CPU tests for the in-process render driver's PURE logic (A-S7.5).

The live engine forwards (render_shot / run_episode / _render_one) are the GPU
soak gate and never run in pytest -- exactly as the engine adapters' forwards
don't. Here we prove the model-agnostic glue the driver is responsible for: the
fallback resolution terminates at the radio floor for every engine, failures
classify HARD, the fixture matches the shipped CPU-soak shape, requests are
deterministic, and assert_soak_ok enforces every A-S7.5 invariant.
"""
import copy

import pytest

from nodes._otr_video_engines import cheap_families  # noqa: F401 (register floor)
from nodes._otr_video_engines import eng_humo         # noqa: F401 (register humo)
from nodes._otr_video_engines import eng_latentsync    # noqa: F401 (register)
from nodes._otr_video_engines import eng_ltx_video      # noqa: F401 (register)
from nodes._otr_video_engines import eng_wan_i2v        # noqa: F401 (register)
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_shared.fallback import resolve_fallback_chain


def test_fallback_chain_character3d_converges_to_floor():
    fb = rd.make_fallback_of()
    # W7-pre: triposg_talk is the v1 character_3d lane.
    chain = resolve_fallback_chain("triposg_talk", fb)
    assert chain == ["triposg_talk", "humo", "humo_1.7B",
                     "latentsync", "still_kenburns"]
    # The deferred-toolkit engine keeps the same resolvable chain shape.
    chain = resolve_fallback_chain("hunyuan3d_talk", fb)
    assert chain == ["hunyuan3d_talk", "humo", "humo_1.7B",
                     "latentsync", "still_kenburns"]


def test_fallback_dangling_engine_gets_universal_floor():
    fb = rd.make_fallback_of()
    # ltx_video / wan_i2v declare no fallback_engine -> the driver appends the
    # registered radio floor so the chain never dangles (survival BUG 12.23).
    assert resolve_fallback_chain("ltx_video", fb) == ["ltx_video", "still_kenburns"]
    assert resolve_fallback_chain("wan_i2v", fb) == ["wan_i2v", "still_kenburns"]


def test_fallback_floor_is_terminal():
    fb = rd.make_fallback_of()
    assert fb("still_kenburns") is None
    assert resolve_fallback_chain("still_kenburns", fb) == ["still_kenburns"]


def test_classify_failure_is_always_hard():
    assert rd.classify_failure(rd.OomSignal("x")) is rt.FailureKind.OOM
    assert rt.block_class_of(rd.classify_failure(LookupError())) is rt.BlockClass.HARD
    assert rt.block_class_of(
        rd.classify_failure(RuntimeError("boom"))) is rt.BlockClass.HARD


def test_fixture_matches_shipped_soak_shape():
    section, meta = rd.build_soak_fixture(n_beats=40, oom_index=20)
    assert len(section["shots"]) == 40
    assert section["video_revision"] == 1
    oom = section["shots"][20]
    assert (oom["engine_id"], oom["family"]) == ("triposg_talk", "character_3d")
    assert meta["oom_shot_id"] == "shot_0020"


def test_build_request_is_deterministic_per_shot():
    shot = {"shot_id": "shot_0007"}
    a = rd.build_request(shot, {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    b = rd.build_request(shot, {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    assert a == b
    assert a["seed_bundle"]["request_seed"] == (7 * 1009 + 7) & 0x7FFFFFFF
    assert a["audio_ref"] == {"path": "a.wav"}


def _passing_episode(n, oom_sid):
    decisions = [
        {"shot_id": oom_sid, "from_engine": "triposg_talk",
         "to_engine": "humo", "failure_kind": "oom", "block_class": "hard",
         "video_revision": 1},
        {"shot_id": oom_sid, "from_engine": "humo", "to_engine": "latentsync",
         "failure_kind": "oom", "block_class": "hard", "video_revision": 1},
        {"shot_id": oom_sid, "from_engine": "latentsync",
         "to_engine": "still_kenburns", "failure_kind": "oom",
         "block_class": "hard", "video_revision": 1},
    ]
    trace = [{"shot_id": "shot_%04d" % i, "attempts": ["x"],
              "final_engine": "x"} for i in range(n)]
    return {
        "n_clips": n, "all_clips_real": True,
        "oom_final_engine": "still_kenburns", "oom_trail": rd.EXPECTED_OOM_TRAIL,
        "decisions": decisions, "video_revision": 1,
        "audio_sha": rd.FROZEN_AUDIO_SHA, "humo_rendered": 2,
        "vram_peak_mb": 10000, "trace": trace, "clips": {},
    }


def _passing_report(n=6, oom_sid="shot_0002"):
    ep = _passing_episode(n, oom_sid)
    return {
        "meta": {"n_beats": n, "oom_shot_id": oom_sid, "oom_index": 2},
        "vram_ceiling_mb": 14500,
        "episode_1": copy.deepcopy(ep), "episode_2": copy.deepcopy(ep),
        "input_oom_engine": "triposg_talk", "input_oom_trail": [],
    }


def test_assert_soak_ok_passes_on_a_valid_report():
    checks = rd.assert_soak_ok(_passing_report())
    assert any("converged" in c for c in checks)
    assert any("determinism" in c for c in checks)


@pytest.mark.parametrize("mutate", [
    lambda r: r["episode_1"].__setitem__("humo_rendered", 0),
    lambda r: r["episode_1"].__setitem__("audio_sha", "tampered"),
    lambda r: r["episode_1"].__setitem__("oom_final_engine", "humo"),
    lambda r: r["episode_1"].__setitem__("vram_peak_mb", 15000),
    lambda r: r["episode_1"].__setitem__("all_clips_real", False),
    lambda r: r["episode_2"]["trace"].append({"shot_id": "x"}),
])
def test_assert_soak_ok_rejects_violations(mutate):
    report = _passing_report()
    mutate(report)
    with pytest.raises(rd.SoakError):
        rd.assert_soak_ok(report)
