"""Two section-3 rows of the go-forward plan, each with one answer under the bar.

The bar (operator, 2026-09-11): "as long as it doesn't crash when it's not
supposed to" -- only an out-of-memory, a traceback or a hang may end a leg, and
never reduce how many episodes reach otr/obs.

1. CAPTION BURN. A probe-confirmed capability gap on the HOST (no ffmpeg, or an
   ffmpeg without libass) used to refuse the whole episode whenever a hero title
   card was planned. The gap is the host's shape, not the episode's fault, and a
   refusal repairs nothing: the clean master now passes through, LOUD and receipted.
   An UNCLASSIFIED burn failure with a planned title still refuses, exactly as
   before -- the classification (`CaptionCapabilityGapError`, a ValueError subclass)
   is what lets the two be told apart.

2. GOOGLE IMAGE. A 200 OK carrying no image block is how Gemini surfaces a content
   block. It used to reach the dispatcher as an engine failure and hard-fail the
   episode (operator 2026-08-22: "why is refusing card killing the episode"). It is
   now flagged `is_model_refusal`, so the dispatcher skips the one card as a
   sanctioned gap with evidence; the honest `failure_kind = "empty_response"`
   label is unchanged.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from nodes import otr_caption_burn as burn_mod  # noqa: E402
from nodes.otr_caption_burn import CaptionCapabilityGapError, OTRCaptionBurn  # noqa: E402
from nodes.video_engine import _CRTRenderer  # noqa: E402

W, H, FPS, TOTAL = 1920, 1080, 25, 200


@pytest.fixture()
def plan_json():
    vol = np.linspace(0.0, 1.0, TOTAL).astype("float32")
    freqs = [np.zeros(32, dtype="float32") for _ in range(TOTAL)]
    waves = [np.zeros(64, dtype="float32") for _ in range(TOTAL)]
    r = _CRTRenderer(W, H, "The Probe", vol, freqs, waves, FPS, timing={
        "music_open_start_f": 10, "music_open_end_f": 60, "first_dialogue_f": 80,
    })
    return json.dumps(r.title_card_plan(main_frames=TOTAL))


class TestACapabilityGapPassesTheCleanMasterThrough:
    def test_a_planned_title_card_on_a_host_without_libass_publishes_the_clean_master(
            self, plan_json, tmp_path, monkeypatch, caplog):
        vid = tmp_path / "ep_silent.mp4"
        vid.write_bytes(b"\x00\x00")

        def _gap(*a, **k):
            raise CaptionCapabilityGapError(
                "OTR_CaptionBurn: ffmpeg has no libass -- captions cannot be burned")
        monkeypatch.setattr(burn_mod, "burn_captions_on_video", _gap)
        out, report = OTRCaptionBurn().burn(
            str(vid), burn_captions=True, title_card_plan_json=plan_json)
        assert out == str(vid), "the clean master must pass through, never be lost"
        assert "capability gap" in report and "title card NOT burned" in report
        assert any("CAPABILITY GAP" in rec.getMessage() for rec in caplog.records), \
            "the passthrough must be LOUD, not a cheerful log line"

    def test_an_UNCLASSIFIED_burn_failure_with_a_planned_title_still_refuses(
            self, plan_json, tmp_path, monkeypatch):
        vid = tmp_path / "ep_silent.mp4"
        vid.write_bytes(b"\x00\x00")

        def _hard(*a, **k):
            raise ValueError("ffmpeg exited 1 while burning")
        monkeypatch.setattr(burn_mod, "burn_captions_on_video", _hard)
        with pytest.raises(RuntimeError, match="hero title card was planned"):
            OTRCaptionBurn().burn(str(vid), burn_captions=True,
                                  title_card_plan_json=plan_json)

    def test_without_a_plan_a_gap_still_passes_through_as_before(
            self, tmp_path, monkeypatch):
        vid = tmp_path / "ep_silent.mp4"
        vid.write_bytes(b"\x00\x00")
        led = tmp_path / "ep_ledger.json"
        led.write_text("{}", encoding="utf-8")

        def _gap(*a, **k):
            raise CaptionCapabilityGapError("OTR_CaptionBurn: ffmpeg not found ('ffmpeg')")
        monkeypatch.setattr(burn_mod, "burn_captions_on_video", _gap)
        out, report = OTRCaptionBurn().burn(
            str(vid), burn_captions=True, ledger_path=str(led))
        assert out == str(vid) and "passthrough" in report


class TestAnEmptyGoogleImageResponseIsARefusalForRouting:
    def test_a_200_with_no_image_block_carries_the_refusal_flag(self):
        from nodes._otr_image_engines import eng_google_image as G
        from nodes._otr_image_engines.eng_google_image import GoogleAPIRequestShapeError
        with pytest.raises(GoogleAPIRequestShapeError) as info:
            G._extract_image_data({"steps": [{"content": [{"type": "text", "text": "no"}]}]})
        exc = info.value
        assert getattr(exc, "is_model_refusal", False) is True
        assert exc.failure_kind == "empty_response", "the honest label is kept"
        assert exc.http_status is None

    def test_the_dispatcher_routes_on_that_flag(self):
        import inspect
        from nodes import otr_image_gen_dispatcher as D
        src = inspect.getsource(D)
        assert 'getattr(exc, "is_model_refusal", False)' in src, \
            "the dispatcher must still split failures by that attribute"
