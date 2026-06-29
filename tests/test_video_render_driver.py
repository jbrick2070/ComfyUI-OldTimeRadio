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
from nodes._otr_video_engines import eng_ltx_video      # noqa: F401 (register)
from nodes._otr_video_engines import eng_wan_i2v        # noqa: F401 (register)
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_shared.fallback import resolve_fallback_chain


def test_fallback_chain_character3d_converges_to_floor():
    fb = rd.make_fallback_of()
    # soak_oom_3d is the synthetic soak stub (C3 -- the real 3D scaffolds are
    # unregistered): it resolves to humo via the overlay, then the registry walks
    # humo -> humo_1.7B -> still_motion.
    chain = resolve_fallback_chain("soak_oom_3d", fb)
    assert chain == ["soak_oom_3d", "humo", "humo_1.7B", "still_motion"]


def test_fallback_dangling_engine_gets_universal_floor():
    fb = rd.make_fallback_of()
    # ltx_video / wan_i2v declare no fallback_engine -> the driver appends the
    # registered radio floor so the chain never dangles (survival BUG 12.23).
    assert resolve_fallback_chain("ltx_video", fb) == ["ltx_video", "still_motion"]
    assert resolve_fallback_chain("wan_i2v", fb) == ["wan_i2v", "still_motion"]


def test_fallback_floor_is_terminal():
    fb = rd.make_fallback_of()
    assert fb("still_motion") is None
    assert resolve_fallback_chain("still_motion", fb) == ["still_motion"]


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
    assert (oom["engine_id"], oom["family"]) == ("soak_oom_3d", "character_3d")
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
        {"shot_id": oom_sid, "from_engine": "soak_oom_3d",
         "to_engine": "humo", "failure_kind": "oom", "block_class": "hard",
         "video_revision": 1},
        {"shot_id": oom_sid, "from_engine": "humo_1.7B",
         "to_engine": "still_motion", "failure_kind": "oom",
         "block_class": "hard", "video_revision": 1},
    ]
    trace = [{"shot_id": "shot_%04d" % i, "attempts": ["x"],
              "final_engine": "x"} for i in range(n)]
    return {
        "n_clips": n, "all_clips_real": True,
        "oom_final_engine": "still_motion", "oom_trail": rd.EXPECTED_OOM_TRAIL,
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
        "input_oom_engine": "soak_oom_3d", "input_oom_trail": [],
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


def test_ltx_renders_native_832x480_others_keep_landscape(monkeypatch):
    """BUG-LOCAL-412 (6/5 parity): ltx_video renders at its native 832x480
    (LTX-2B mushes above 480p; the composite scales it up) while still_pan
    keeps the full 1472x832 landscape canvas. Both env-overridable."""
    monkeypatch.delenv("OTR_LTX_RENDER_CANVAS", raising=False)
    monkeypatch.delenv("OTR_VIDEO_LANDSCAPE_CANVAS", raising=False)
    ledger = {"video": {"video_revision": 1, "shots": []},
              "lines": [{"line_id": "b001", "start_s": 0.0, "dur_s": 2.0}],
              "images": {"images": []}}

    def shot(engine, family):
        return {"shot_id": "shot_b001", "beat_id": "b001",
                "engine_id": engine, "family": family,
                "target_frame_count": 169, "source_line_ids": ["b001"],
                "char_id": "", "creative": {}}

    req_ltx = rd.build_request_from_shot(shot("ltx_video", "text_to_video"),
                                         ledger)
    assert (req_ltx["canvas"]["w"], req_ltx["canvas"]["h"]) == (832, 480)
    req_flux = rd.build_request_from_shot(shot("still_pan", "static_image_gen"),
                                          ledger)
    assert (req_flux["canvas"]["w"], req_flux["canvas"]["h"]) == (1472, 832)
    monkeypatch.setenv("OTR_LTX_RENDER_CANVAS", "768x432")
    req2 = rd.build_request_from_shot(shot("ltx_video", "text_to_video"), ledger)
    assert (req2["canvas"]["w"], req2["canvas"]["h"]) == (768, 432)


def test_still_flat_character_beat_uses_landscape_scene_still_not_portrait():
    """BUG 1 (2026-06-20 operator directive): still-only (still_flat / still_pan)
    are LANDSCAPE engines -- a CHARACTER beat conditions on its per-beat 16:9
    CHARACTER scene still (kind=scene_character, minted in the image phase),
    NEVER the 832x1216 vertical portrait (which pillarboxed -> the radio-booth
    floor filled the sides). Only HuMo / audio_driven_face use the portrait.
    Image-model agnostic -- the scene still is whatever image engine rendered it.
    Non-character beats keep their scene still."""
    ledger = {
        "video": {"video_revision": 1, "shots": []},
        "lines": [
            {"line_id": "b002", "beat_id": "b002", "char_id": "c01",
             "speaker_role": "character", "start_s": 0.0, "dur_s": 2.0},
            {"line_id": "b003", "beat_id": "b003", "char_id": "",
             "speaker_role": "music", "start_s": 2.0, "dur_s": 2.0},
        ],
        "images": {"images": [
            {"object_id": "c01", "kind": "portrait", "path": "portrait_c01.png"},
            {"beat_id": "b002", "kind": "scene_character", "path": "scene_b002.png"},
            {"beat_id": "b003", "kind": "scene_default", "path": "scene_b003.png"},
        ]},
    }

    def shot(beat, cid):
        return {"shot_id": "shot_" + beat, "beat_id": beat,
                "engine_id": "still_flat", "family": "static_image_gen",
                "target_frame_count": 50, "source_line_ids": [beat],
                "char_id": cid, "creative": {}}

    # CHARACTER beat: the 16:9 character scene still, NOT the vertical portrait.
    req_char = rd.build_request_from_shot(shot("b002", "c01"), ledger)
    assert req_char["asset_refs"].get("init_image") == "scene_b002.png"
    assert req_char["asset_refs"].get("init_image") != "portrait_c01.png"
    req_music = rd.build_request_from_shot(shot("b003", ""), ledger)
    assert req_music["asset_refs"].get("init_image") == "scene_b003.png"


def test_still_flat_character_beat_missing_still_clears_init_no_portrait_leak():
    """BUG 1: when a CHARACTER beat has NO scene still, the landscape still engine
    degrades to its dark floor (init_image cleared) -- the vertical portrait must
    NEVER leak into the 1472x832 frame."""
    ledger = {
        "video": {"video_revision": 1, "shots": []},
        "lines": [
            {"line_id": "b002", "beat_id": "b002", "char_id": "c01",
             "speaker_role": "character", "start_s": 0.0, "dur_s": 2.0},
        ],
        "images": {"images": [
            {"object_id": "c01", "kind": "portrait", "path": "portrait_c01.png"},
        ]},
    }
    shot = {"shot_id": "shot_b002", "beat_id": "b002",
            "engine_id": "still_flat", "family": "static_image_gen",
            "target_frame_count": 50, "source_line_ids": ["b002"],
            "char_id": "c01", "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    # No still -> empty init (cheap-family floor); the portrait does NOT leak.
    assert req["asset_refs"].get("init_image", "") == ""
