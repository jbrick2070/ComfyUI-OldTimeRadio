"""C7: the sanctioned-gap control path, end to end, through the REAL seams.

THE LAST OPEN PIECE OF THE HARDENED r2 PLAN. C0-C6 each proved one layer; this
proves the CHAIN, because the panel's whole finding was that unit-proven layers
composed into dead code (the spine raised before anything downstream could
run). The fixture is the operator's nightmare scenario: EVERY required still
refused. Per the 2026-08-27 ruling, that episode PUBLISHES degraded -- so this
chain must end in renderable floor segments, not an exception.

The plan predicted a hole here: "all-gap with empty new_shots makes
assemble_silent_timeline raise 'manifest has no renderable beats' -- that hole
IS the work." C2 is what was built against that prediction (the loop KEEPS the
gapped beat instead of dropping it), and this file is the proof the hole is
actually closed -- if C2 ever regresses to dropping beats, the planner goes
empty and the assertions here name the exact layer that broke.

CPU-only and honest about it: `assemble_silent_timeline` itself needs ffmpeg
and a floor video, but its "no renderable beats" raise is literally
`if not segments or total <= 0` over `plan_timeline_segments(...)` -- so
proving the PURE planner emits floor segments for an all-gap manifest is
proving the raise cannot fire, without spending an encode.
"""
from __future__ import annotations

import unittest.mock as mock

import pytest

from nodes._otr_shared import still_receipt as _receipt
from nodes._otr_video_engines import render_driver as rd
from nodes.otr_silent_composite import plan_timeline_segments
from nodes.otr_video_render_batch import _build_render_engines_payload

FPS = 25
N_BEATS = 3
FRAMES_PER_BEAT = 50
TOTAL_FRAMES = N_BEATS * FRAMES_PER_BEAT


def _all_refused_ledger():
    """Three beats, every required still refused by the image model."""
    receipt, shots, lines = [], [], []
    for i in range(1, N_BEATS + 1):
        bid = "b%03d" % i
        receipt.append({
            "object_id": "scene_%s" % bid,
            "status": _receipt.STATUS_SANCTIONED_GAP,
            "kind": "scene_wide", "role": "narration", "beat_id": bid,
            "reason": "model_refusal", "engine_id": "ideogram_v4",
            "seed": i, "prompt": "a refused card %d" % i,
            "prompt_hash": "hash%d" % i, "detail": "declined",
            "image_revision": 1,
        })
        shots.append({
            "shot_id": "shot_%s" % bid, "beat_id": bid,
            "engine_id": "still_flat", "family": "static_motion",
            "target_frame_count": FRAMES_PER_BEAT,
            "source_line_ids": [bid], "char_id": "", "creative": {},
            "start_s": (i - 1) * (FRAMES_PER_BEAT / FPS),
            "dur_s": FRAMES_PER_BEAT / FPS,
        })
        lines.append({"line_id": bid,
                      "start_s": (i - 1) * (FRAMES_PER_BEAT / FPS),
                      "dur_s": FRAMES_PER_BEAT / FPS})
    return {
        "episode_id": "test_ep_all_refused",
        "images": {"image_revision": 1, "images": [],
                   "required_scene_targets": receipt},
        "video": {"video_revision": 1, "fps": FPS, "shots": shots},
        "lines": lines,
    }


def _explode(*_a, **_k):
    raise AssertionError("an all-refused episode must render NOTHING")


def _run_chain():
    """Spine -> episode loop -> manifest -> payload, all real code."""
    led = _all_refused_ledger()
    rd.validate_and_repair_still_spine(led)          # C0: must not raise
    with mock.patch.object(rd, "render_beat_coverage", _explode):
        result = rd.run_episode(led)                 # C2: skips, keeps beats
    manifest = rd.build_clip_manifest(result, episode_id="test_ep_all_refused")
    payload = _build_render_engines_payload(manifest, None)
    return led, result, manifest, payload


def test_the_spine_and_loop_survive_a_fully_refused_episode():
    led, result, manifest, payload = _run_chain()
    shots = (result["ledger"]["video"] or {}).get("shots") or []
    assert [s["shot_id"] for s in shots] == [
        "shot_b001", "shot_b002", "shot_b003"], (
        "the gapped beats must SURVIVE collection -- a dropped beat cannot be "
        "floored, and this regressing is exactly the hole the plan predicted")
    assert not result["clips"], "nothing may render on an all-refused episode"


def test_the_manifest_marks_every_row_sanctioned_and_counts_zero_clips():
    _led, _result, manifest, payload = _run_chain()
    rows = manifest["clips"]
    assert len(rows) == N_BEATS
    assert all(not r["exists"] for r in rows)
    assert all(_receipt.is_sanctioned_gap(r) for r in rows), (
        "C3: the sanction must ride the manifest row -- absence alone is what "
        "would let a crashed render impersonate a degraded one")
    assert manifest["clip_count"] == 0

    # C5: the payload counts them as SANCTIONED, none unexplained.
    assert payload["sanctioned_gap_count"] == N_BEATS
    assert payload["unsanctioned_gap_count"] == 0

    # C4's predicate over these rows: every beat accounted for -> ok+degraded.
    n = len(rows)
    sanctioned = sum(1 for r in rows if _receipt.is_sanctioned_gap(r))
    delivered = sum(1 for r in rows if r.get("exists"))
    assert n > 0 and (n - delivered - sanctioned) == 0, (
        "the publish predicate's inputs must say: publishable, degraded")


def test_the_planner_floors_every_beat_so_the_composite_cannot_raise():
    """THE HOLE THE PLAN PREDICTED, proven closed.

    `assemble_silent_timeline` raises 'manifest has no renderable beats' when
    `plan_timeline_segments` returns nothing. For the all-gap manifest it must
    instead return one floor/black segment per beat covering the full master
    length -- the shape of a publishable degraded episode.
    """
    _led, _result, manifest, _payload = _run_chain()
    segments, total = plan_timeline_segments(
        manifest, floor_available=True, floor_frames=TOTAL_FRAMES,
        target_total_frames=TOTAL_FRAMES, fps=FPS)
    assert segments, (
        "EMPTY segment plan for an all-gap episode: this is the exact hole "
        "the r2 plan predicted (assemble_silent_timeline would raise "
        "'manifest has no renderable beats'), which means a beat was DROPPED "
        "upstream instead of kept -- check C2's skip in run_episode")
    assert total == TOTAL_FRAMES
    assert all(seg["source"] in ("floor", "black") for seg in segments), (
        "an all-refused episode has no clip to place; every segment must be "
        "floor or black, got: %r" % sorted({s["source"] for s in segments}))
    assert sum(s["n_frames"] for s in segments) == TOTAL_FRAMES


def test_without_the_floor_the_plan_still_covers_with_black():
    """A box with no floor video still publishes -- black is the last resort
    and the ruling does not depend on procgen assets existing."""
    _led, _result, manifest, _payload = _run_chain()
    segments, total = plan_timeline_segments(
        manifest, floor_available=False, floor_frames=0,
        target_total_frames=TOTAL_FRAMES, fps=FPS)
    assert segments and total == TOTAL_FRAMES
    assert all(seg["source"] == "black" for seg in segments)


def test_the_credits_payload_guard_accepts_the_all_gap_episode():
    """C5's other half: `OTR_CreditsRoll._require` rejects empty payloads --
    'the exact outcome the sanctioned gap exists to prevent'. The all-gap
    payload must therefore be non-empty and carry the gap accounting."""
    from nodes.otr_credits_roll import _require

    _led, _result, _manifest, payload = _run_chain()
    meta = {"render_engines": payload}
    assert _require(meta, "render_engines", "meta") is payload
    assert payload["sanctioned_gap_shot_ids"] == [
        "shot_b001", "shot_b002", "shot_b003"]


def test_an_UNSANCTIONED_absence_still_poisons_the_predicate():
    """The guard rail, end to end: strip ONE gap row's status and the same
    chain must classify that beat as unexplained -- the crashed-render case
    the panel proved the first draft would have laundered."""
    led = _all_refused_ledger()
    led["images"]["required_scene_targets"][1].pop("status")
    # The spine now sees b002 as an ordinary missing still and fails loud.
    with pytest.raises(rd.RenderError, match="still-spine"):
        rd.validate_and_repair_still_spine(led)
