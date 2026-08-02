"""CPU tests for the in-process render driver's PURE logic (A-S7.5).

The live engine forwards (render_shot / run_episode / _render_one) are the GPU
soak gate and never run in pytest -- exactly as the engine adapters' forwards
don't. Here we prove the model-agnostic glue the driver is responsible for:
NO FALLBACKS (2026-07-02 rip) -- no chain machinery exists and every registered
video engine declares fallback_engine=None, failures classify HARD, the fixture
matches the shipped CPU-soak shape, requests are deterministic, and
assert_soak_ok enforces every A-S7.5 invariant (incl. the LOUD-failure
contract).
"""
import copy

import pytest

from nodes._otr_video_engines import cheap_families  # noqa: F401 (register)
from nodes._otr_video_engines import eng_humo         # noqa: F401 (register humo)
from nodes._otr_video_engines import eng_ltx_video      # noqa: F401 (register)
from nodes._otr_video_engines import eng_wan_i2v        # noqa: F401 (register)
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_shared import retry_taxonomy as rt


def test_no_fallback_machinery_exists():
    """Sprint A rip (2026-07-02): the chain machinery is GONE from the driver."""
    for name in ("make_fallback_of", "FLOOR_NAMES", "UNIVERSAL_FLOOR",
                 "SYNTH_FALLBACKS", "EXPECTED_OOM_TRAIL"):
        assert not hasattr(rd, name), "%s must stay ripped (NO FALLBACKS)" % name


def test_every_registered_engine_declares_no_fallback():
    for name in vreg.all_engine_names():
        eng = vreg.get_engine(name)
        assert getattr(eng, "fallback_engine", None) is None, (
            "engine %r declares fallback_engine=%r -- NO FALLBACKS "
            "(2026-07-02): every adapter must declare None"
            % (name, eng.fallback_engine))


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


def _passing_episode(n):
    trace = [{"shot_id": "shot_%04d" % i, "attempts": ["x"],
              "final_engine": "x"} for i in range(n)]
    return {
        "n_clips": n, "all_clips_real": True,
        "video_revision": 1,
        "audio_sha": rd.FROZEN_AUDIO_SHA, "humo_rendered": 2,
        "vram_peak_mb": 10000, "trace": trace, "clips": {},
    }


def _passing_report(n=6):
    ep = _passing_episode(n)
    return {
        "meta": {"n_beats": n, "oom_shot_id": None, "oom_index": None},
        "episode_1": copy.deepcopy(ep), "episode_2": copy.deepcopy(ep),
        "input_shot_count": n,
        "oom_contract": {"raised": True, "error_type": "RenderError",
                         "detail": "shot shot_0002 engine 'soak_oom_3d' ..."},
    }


def test_assert_soak_ok_passes_on_a_valid_report():
    checks = rd.assert_soak_ok(_passing_report())
    assert any("determinism" in c for c in checks)
    assert any("LOUD-failure contract" in c for c in checks)


@pytest.mark.parametrize("mutate", [
    lambda r: r["episode_1"].__setitem__("humo_rendered", 0),
    lambda r: r["episode_1"].__setitem__("audio_sha", "tampered"),
    lambda r: r["episode_1"].__setitem__("all_clips_real", False),
    lambda r: r["episode_2"]["trace"].append({"shot_id": "x"}),
    # LOUD-failure contract: the forced OOM must RAISE RenderError.
    lambda r: r["oom_contract"].__setitem__("raised", False),
    lambda r: r["oom_contract"].__setitem__("error_type", "OomSignal"),
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
    # This test isolates canvas selection. Explicitly exercise LTX's documented
    # text-only opt-out so the required-still contract is covered by its own
    # regression rather than by an unrelated canvas fixture.
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
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
    # ltx_video's canvas STOPPED being env-overridable on 2026-08-02, and that
    # is the fix rather than a regression. Its frame contract is now the single
    # length 169, which is only true at the canvas the decode floor was measured
    # at -- so an env-moved canvas would invalidate a STATIC contract the beat
    # was already partitioned against, with no code change and no warning.
    # A declared render canvas wins (it is applied last, by design), and the
    # engine REFUSES at render time rather than quietly handing back a canvas
    # the operator did not ask for. still_pan, which declares nothing, keeps its
    # env branch untouched -- see the assertion above.
    monkeypatch.setenv("OTR_LTX_RENDER_CANVAS", "768x432")
    req2 = rd.build_request_from_shot(shot("ltx_video", "text_to_video"), ledger)
    assert (req2["canvas"]["w"], req2["canvas"]["h"]) == (832, 480), (
        "a declared render canvas must beat OTR_LTX_RENDER_CANVAS")

    from nodes._otr_video_engines import eng_ltx_video as _lv
    from nodes._otr_video_engines import frame_contract as _fc
    with pytest.raises(_fc.ContractEnvConflict):
        _lv.assert_env_matches_contract(_lv.LtxVideoEngine.frame_contract)


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
