"""CPU tests for the in-process render driver's PURE logic (A-S7.5).

The live engine forwards (render_shot / run_episode / _render_one) are proven
by a live canonical episode and never run in pytest -- exactly as the engine
adapters' forwards don't; render_shot runs here only with _render_one stubbed.
Here we prove the model-agnostic glue the driver is responsible for: a failed
shot is labelled and fails once, LOUD, and requests are deterministic.
"""

import pytest

from nodes._otr_video_engines import cheap_families  # noqa: F401 (register)
from nodes._otr_video_engines import eng_humo         # noqa: F401 (register humo)
from nodes._otr_video_engines import eng_ltx25          # noqa: F401 (register)
from nodes._otr_video_engines import eng_ltx_8gb        # noqa: F401 (register)
from nodes._otr_video_engines import render_driver as rd


def test_classify_failure_labels_every_exception():
    assert rd.classify_failure(RuntimeError("CUDA out of memory")) is rd.FailureKind.OOM
    assert rd.classify_failure(LookupError()) is rd.FailureKind.DEPENDENCY_MISSING
    assert rd.classify_failure(RuntimeError("boom")) is rd.FailureKind.CRASH_BEFORE_LOAD


@pytest.mark.parametrize("code, kind", [
    ("content_refused", "CONTENT_REFUSED"),
    ("timeout", "TIMEOUT"),
    ("corrupt_output", "CORRUPT_OUTPUT"),
    ("retryable_transport", "TRANSIENT_IO"),
])
def test_a_cloud_job_verdict_is_labelled_with_its_kind(code, kind):
    """Each job-scoped cloud verdict names its own kind. The kind is looked up
    BY NAME (getattr), so a FailureKind member removed by mistake would turn a
    cloud timeout into an AttributeError instead of a RenderError."""
    from nodes._otr_shared import cloud_media_backend as cmb

    assert set(rd._JOB_CODE_TO_KIND.values()) <= set(rd.FailureKind.__members__)
    exc = cmb.CloudMediaError(code, "test verdict")
    assert rd.classify_failure(exc) is rd.FailureKind[kind]


def test_a_failed_shot_is_tried_once_and_named_in_the_error(monkeypatch):
    """No retry and no fallback: the engine runs once, the shot fails LOUD,
    and the error names the kind classify_failure gave it."""
    calls = []

    def render(engine_name, request, **kwargs):
        calls.append(engine_name)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(rd, "_render_one", render)
    shot = {"shot_id": "shot_0001", "engine_id": "ltx25_video"}
    with pytest.raises(rd.RenderError) as excinfo:
        rd.render_shot(shot, {"shot_id": "shot_0001"})
    assert calls == ["ltx25_video"]
    assert str(rd.FailureKind.OOM) in str(excinfo.value)


def test_build_request_is_deterministic_per_shot():
    shot = {"shot_id": "shot_0007"}
    a = rd.build_request(shot, {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    b = rd.build_request(shot, {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    assert a == b
    assert a["seed_bundle"]["request_seed"] == (7 * 1009 + 7) & 0x7FFFFFFF
    assert a["audio_ref"] == {"path": "a.wav"}


def test_ltx_renders_its_DECLARED_canvas_others_keep_landscape(monkeypatch, tmp_path):
    """ltx25_video renders at the canvas it DECLARES while still_pan keeps the
    full 1472x832 landscape canvas.

    Asserted against the DECLARATION rather than a literal, so the number can
    move without this test lying (lesson L10).

    still_pan stays env-overridable through OTR_VIDEO_LANDSCAPE_CANVAS. The
    LTX lane is NOT: it declares a render_canvas, which is applied last and
    wins, because its frame contract is canvas-dependent and an env-moved
    canvas would invalidate it with no code change and no warning.
    """
    monkeypatch.delenv("OTR_VIDEO_LANDSCAPE_CANVAS", raising=False)
    # This test isolates canvas selection. The lane requires a still, always,
    # so the fixture SUPPLIES one rather than dodging the requirement.
    still = tmp_path / "scene_b001.png"
    still.write_bytes(bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)))
    ledger = {"video": {"video_revision": 1, "shots": []},
              "lines": [{"line_id": "b001", "start_s": 0.0, "dur_s": 2.0}],
              "images": {"images": [{"object_id": "scene_b001",
                                     "kind": "scene_wide", "beat_id": "b001",
                                     "path": str(still)}]}}

    def shot(engine, family):
        return {"shot_id": "shot_b001", "beat_id": "b001",
                "engine_id": engine, "family": family,
                "target_frame_count": 169, "source_line_ids": ["b001"],
                "char_id": "", "creative": {}}

    from nodes._otr_video_engines.eng_ltx25 import Ltx25VideoEngine
    declared = tuple(Ltx25VideoEngine.render_canvas)
    assert declared != (1472, 832), (
        "the declared canvas must differ from the landscape default or this "
        "test cannot tell the two apart")
    req_ltx = rd.build_request_from_shot(shot("ltx25_video", "image_to_video"),
                                         ledger)
    assert (req_ltx["canvas"]["w"], req_ltx["canvas"]["h"]) == declared
    req_flux = rd.build_request_from_shot(shot("still_pan", "static_image_gen"),
                                          ledger)
    assert (req_flux["canvas"]["w"], req_flux["canvas"]["h"]) == (1472, 832)
    # A declared render canvas wins (it is applied last, by design). still_pan,
    # which declares nothing, follows the env lever.
    monkeypatch.setenv("OTR_VIDEO_LANDSCAPE_CANVAS", "768x432")
    req2 = rd.build_request_from_shot(shot("ltx25_video", "image_to_video"),
                                      ledger)
    assert (req2["canvas"]["w"], req2["canvas"]["h"]) == declared, (
        "a declared render canvas must beat OTR_VIDEO_LANDSCAPE_CANVAS")
    req_flux2 = rd.build_request_from_shot(
        shot("still_pan", "static_image_gen"), ledger)
    assert (req_flux2["canvas"]["w"], req_flux2["canvas"]["h"]) == (768, 432)


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
