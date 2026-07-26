"""A jump segment's init image is resolved BY OBJECT ID, never by beat.

Multi-clip coverage chunk 6b. The chunk-4 QA panel found this and it was
carried forward on purpose: ``render_driver._still_index`` filters to rows
whose ``kind`` starts with ``scene_`` and keys them BY BEAT, while a
jump-segment still deliberately wears ``jump_segment`` so it stays invisible to
every beat-keyed consumer. A per-segment loop that reused the existing lookup
would hand EVERY segment segment-0's still -- the held-frame degradation this
whole build exists to remove, with the correct stills sitting unused on disk.

The differential test at the bottom is the point: it shows the OLD lookup
returning the wrong image for the same shot and segment.
"""

from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import render_driver as rd


def _shot(segments=(1, 2)):
    return {
        "shot_id": "shot_b001",
        "role": "character_video",
        "engine_id": "wan_i2v",
        "family": "image_to_video",
        "target_frame_count": 50,
        "jump_still_requests": [
            {"object_id": cp.jump_still_object_id("b001", index),
             "kind": cp.JUMP_STILL_KIND,
             "beat_id": "b001",
             "segment_index": index,
             "role": "character_video",
             "engine_id": "wan_i2v"}
            for index in segments
        ],
    }


def _ledger(paths=("/stills/seg1.png", "/stills/seg2.png"),
            beat_still="/stills/beat_scene.png"):
    validated = [
        {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "wan_i2v",
         "kind": "scene_character", "path": beat_still},
    ]
    for offset, path in enumerate(paths):
        index = offset + 1
        validated.append({
            "shot_id": "shot_b001", "beat_id": "b001", "engine_id": "wan_i2v",
            "kind": cp.JUMP_STILL_KIND, "segment_index": index,
            "object_id": cp.jump_still_object_id("b001", index),
            "path": path,
        })
    return {
        "episode_id": "ep_seg",
        "images": {
            "still_spine_receipt": {"version": 1, "episode_id": "ep_seg",
                                    "validated": validated},
            # ``_still_index`` reads THIS list: the beat-keyed scene rows.
            # Note there is deliberately no jump_segment row here -- that is
            # the whole point, they are invisible to the beat-keyed lookup.
            "images": [
                {"object_id": "scene_character_b001", "kind": "scene_character",
                 "beat_id": "b001", "path": beat_still},
            ],
        },
        "video": {"shots": [_shot()]},
    }


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

def test_each_segment_gets_ITS_OWN_still():
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    assert rd.jump_segment_still_path(ledger, shot, 1) == "/stills/seg1.png"
    assert rd.jump_segment_still_path(ledger, shot, 2) == "/stills/seg2.png"


def test_segment_0_keeps_the_BEATS_still():
    """Segment 0 is the one segment the beat's own still still covers."""
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    assert rd.jump_segment_still_path(ledger, shot, 0) is None


def test_a_segment_with_no_request_is_TERMINAL():
    """NO FALLBACK to the beat's still: the identical image at both ends of a
    cut is the degradation, not a recovery."""
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    with pytest.raises(rd.RenderError, match="no jump-still request"):
        rd.jump_segment_still_path(ledger, shot, 3)


def test_a_request_the_SPINE_never_proved_is_TERMINAL():
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    ledger["images"]["still_spine_receipt"]["validated"] = [
        row for row in ledger["images"]["still_spine_receipt"]["validated"]
        if row.get("segment_index") != 2
    ]
    with pytest.raises(rd.RenderError, match="does not carry a materialized"):
        rd.jump_segment_still_path(ledger, shot, 2)


def test_a_receipt_entry_with_an_EMPTY_path_is_TERMINAL():
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    for row in ledger["images"]["still_spine_receipt"]["validated"]:
        if row.get("segment_index") == 2:
            row["path"] = ""
    with pytest.raises(rd.RenderError, match="does not carry a materialized"):
        rd.jump_segment_still_path(ledger, shot, 2)


def test_a_missing_receipt_entirely_is_TERMINAL():
    """Reaching a render with no receipt means the spine was bypassed."""
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]
    ledger["images"].pop("still_spine_receipt")
    with pytest.raises(rd.RenderError, match="does not carry a materialized"):
        rd.jump_segment_still_path(ledger, shot, 1)


# ---------------------------------------------------------------------------
# THE DIFFERENTIAL -- what the old lookup would have done
# ---------------------------------------------------------------------------

def test_the_BEAT_KEYED_lookup_would_have_returned_the_WRONG_image():
    """The carry-forward, demonstrated rather than asserted.

    ``_still_index`` cannot see a ``jump_segment`` row at all, so for every
    segment of this shot it returns the SAME beat still -- segment 0's image.
    The failure would have been silent: a real path, a real file, the wrong
    picture, and nothing in the log to say the correct stills went unused.
    """
    ledger = _ledger()
    shot = ledger["video"]["shots"][0]

    beat_keyed = rd._still_index(ledger).get("b001", "")
    assert beat_keyed == "/stills/beat_scene.png"

    by_object_id = [rd.jump_segment_still_path(ledger, shot, index)
                    for index in (1, 2)]
    assert by_object_id == ["/stills/seg1.png", "/stills/seg2.png"]
    assert beat_keyed not in by_object_id, (
        "the beat-keyed lookup and the per-segment lookup agree, so this test "
        "no longer demonstrates anything -- check the fixture")
