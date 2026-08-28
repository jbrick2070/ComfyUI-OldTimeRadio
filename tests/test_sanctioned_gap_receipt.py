"""C0 + C1 of the sanctioned-gap control path: the receipt and the spine.

WHAT THIS PINS, and why each half would be useless alone: C1 makes a tolerated
model refusal MINT a receipt row instead of vanishing into a function-local
dict, and C0 makes the still-spine validator ACCEPT that row instead of
killing the episode over the missing file. Before both, a refused card was
logged, dropped, and then re-discovered by the spine as an unexplained
absence -- which is how a 30-minute render died for one declined image the
operator had already ruled survivable.

The negative assertions matter as much as the positive ones: an absence with
NO gap row must still fail, or this work would convert a crashed render into
a publishable episode. That inversion is exactly what the 2026-08-28 review
panel caught in the first draft of the plan.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import still_receipt as _receipt
from nodes._otr_video_engines import render_driver as rd


def _ledger_with_receipt(rows, images=(), shots=()):
    return {
        # The spine refuses to run without a durable episode id -- it resolves
        # the stills root from it -- so the fixture supplies one.
        "episode_id": "test_ep_sanctioned_gap",
        "images": {"images": list(images), "required_scene_targets": list(rows)},
        "video": {"video_revision": 1, "shots": list(shots)},
        "lines": [{"line_id": "b001", "start_s": 0.0, "dur_s": 2.0}],
    }


OK_ROW = {"object_id": "scene_b001", "status": _receipt.STATUS_OK,
          "kind": "scene_wide", "role": "narration", "beat_id": "b001",
          "path": "X:/stills/scene_b001.png", "content_hash": "abc"}
GAP_ROW = {"object_id": "scene_b002", "status": _receipt.STATUS_SANCTIONED_GAP,
           "kind": "scene_wide", "role": "narration", "beat_id": "b002",
           "reason": "model_refusal", "engine_id": "ideogram_v4",
           "seed": 7, "prompt": "a lighthouse at dusk",
           "prompt_hash": "deadbeef", "detail": "declined",
           "image_revision": 3}


class TestVocabulary:
    def test_gap_is_explicit_never_inferred_from_a_missing_path(self):
        """The whole control path rests on this: no path is NOT a gap."""
        assert _receipt.is_sanctioned_gap(GAP_ROW)
        assert not _receipt.is_sanctioned_gap(OK_ROW)
        # A row with no status and no path is an unexplained absence.
        assert not _receipt.is_sanctioned_gap({"object_id": "x", "beat_id": "b"})
        assert not _receipt.is_sanctioned_gap(None)
        assert not _receipt.is_sanctioned_gap("sanctioned_gap")

    def test_only_model_refusal_is_sanctionable(self):
        """Narrow on purpose -- widening this re-opens the gate it guards."""
        assert _receipt.SANCTIONABLE_SKIP_REASON == "model_refusal"
        assert _receipt.RECEIPT_STATUSES == {"ok", "sanctioned_gap"}


class TestGapPredicates:
    def test_beat_and_object_sets_come_from_the_receipt(self):
        led = _ledger_with_receipt([OK_ROW, GAP_ROW])
        assert rd.sanctioned_gap_beat_ids(led) == {"b002"}
        assert rd.sanctioned_gap_object_ids(led) == {"scene_b002"}

    def test_no_receipt_is_no_gaps_not_a_crash(self):
        assert rd.sanctioned_gap_beat_ids({}) == set()
        assert rd.sanctioned_gap_object_ids({"images": {}}) == set()

    def test_an_absent_still_without_a_gap_row_is_not_a_gap(self):
        """The negative that keeps a crashed render failing."""
        absent = {"object_id": "scene_b003", "beat_id": "b003",
                  "kind": "scene_wide", "role": "narration"}
        led = _ledger_with_receipt([absent])
        assert rd.sanctioned_gap_beat_ids(led) == set()


class TestSpineAcceptsTheGap:
    """C0: the validator declines to raise on a sanctioned beat."""

    def _shot(self, beat_id, shot_id):
        return {"shot_id": shot_id, "beat_id": beat_id,
                "engine_id": "still_flat", "family": "static_motion",
                "target_frame_count": 50, "source_line_ids": [beat_id],
                "char_id": "", "creative": {}}

    def test_gapped_beat_does_not_raise_and_is_not_validated(self):
        led = _ledger_with_receipt(
            [GAP_ROW],
            images=[],
            shots=[self._shot("b002", "shot_b002")])
        led["lines"] = [{"line_id": "b002", "start_s": 0.0, "dur_s": 2.0}]
        # Before C0 this raised "still-spine handoff missing materialized
        # scene still". The episode must now survive it.
        rd.validate_and_repair_still_spine(led)

    def test_a_stale_row_for_a_refused_object_cannot_satisfy_the_spine(
            self, tmp_path):
        """The mask: a prior revision's file still sitting on disk."""
        stale = tmp_path / "scene_b002.png"
        stale.write_bytes(bytes((0x89, 0x50, 0x4E, 0x47,
                                 0x0D, 0x0A, 0x1A, 0x0A)))
        led = _ledger_with_receipt(
            [GAP_ROW],
            images=[{"object_id": "scene_b002", "beat_id": "b002",
                     "kind": "scene_wide", "path": str(stale)}],
            shots=[self._shot("b002", "shot_b002")])
        led["lines"] = [{"line_id": "b002", "start_s": 0.0, "dur_s": 2.0}]
        # The stale row is dropped, so the beat stays a gap rather than
        # validating a still this dispatch never produced.
        rd.validate_and_repair_still_spine(led)
        assert rd.sanctioned_gap_object_ids(led) == {"scene_b002"}

    def test_an_unexplained_missing_still_STILL_kills_the_episode(
            self, tmp_path):
        """The guard rail. Without this, C0 would launder every failure."""
        no_gap = {"object_id": "scene_b009", "status": _receipt.STATUS_OK,
                  "kind": "scene_wide", "role": "narration",
                  "beat_id": "b009", "path": str(tmp_path / "gone.png"),
                  "content_hash": "x"}
        led = _ledger_with_receipt(
            [no_gap], images=[], shots=[self._shot("b009", "shot_b009")])
        led["lines"] = [{"line_id": "b009", "start_s": 0.0, "dur_s": 2.0}]
        with pytest.raises(rd.RenderError, match="still-spine handoff"):
            rd.validate_and_repair_still_spine(led)


class TestEpisodeLoopSkipsTheGap:
    """C2: ``run_episode`` renders nothing for a sanctioned beat, and keeps it.

    The beat surviving the loop is the point. ``run_episode`` REPLACES
    ``video.shots`` with the shots it collected, so a beat that merely fails
    to render would be ERASED from the episode rather than gapped -- and an
    erased beat cannot be floored by the composite, because nothing downstream
    knows it existed.
    """

    def test_gapped_beat_is_kept_unrendered_and_never_reaches_an_engine(self):
        rendered = []

        def _explode(*_a, **_kw):
            # Reaching an engine at all is the failure this test exists for.
            rendered.append(_a)
            raise AssertionError(
                "render_beat_coverage was called for a sanctioned-gap beat")

        led = _ledger_with_receipt([GAP_ROW], images=[], shots=[])
        led["video"]["shots"] = [
            {"shot_id": "shot_b002", "beat_id": "b002",
             "engine_id": "still_flat", "family": "static_motion",
             "target_frame_count": 50, "source_line_ids": ["b002"],
             "char_id": "", "creative": {}, "start_s": 0.0, "dur_s": 2.0}]
        led["lines"] = [{"line_id": "b002", "start_s": 0.0, "dur_s": 2.0}]

        import unittest.mock as _mock
        with _mock.patch.object(rd, "render_beat_coverage", _explode):
            out = rd.run_episode(led)

        assert not rendered, "the gapped beat reached the renderer"
        shots = (out["ledger"]["video"] or {}).get("shots") or []
        ids = [s.get("shot_id") for s in shots]
        assert ids == ["shot_b002"], (
            "the gapped beat must SURVIVE collection, not be erased: %r" % ids)
        assert not out["clips"], "a gapped beat must contribute no clip"
