"""
test_otr_shot_duration_calculator.py
=====================================

Validates the shot -> clip -> frame expansion math without requiring
torch or ComfyUI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------------------------
# Pure math helpers
# ------------------------------------------------------------------


def test_clips_per_shot_under_cap():
    from nodes.otr_shot_duration_calculator import clips_per_shot
    assert clips_per_shot(6.0) == 1
    assert clips_per_shot(9.0) == 1
    assert clips_per_shot(9.9) == 1
    assert clips_per_shot(10.0) == 1    # exactly at cap still 1 clip


def test_clips_per_shot_over_cap():
    from nodes.otr_shot_duration_calculator import clips_per_shot
    # Over cap => ceil(dur / 9)
    assert clips_per_shot(10.5) == 2   # ceil(10.5/9) = 2
    assert clips_per_shot(15.0) == 2   # ceil(15/9) = 2
    assert clips_per_shot(18.0) == 2
    assert clips_per_shot(19.0) == 3   # ceil(19/9) = 3
    assert clips_per_shot(27.0) == 3   # ceil(27/9) = 3
    assert clips_per_shot(27.1) == 4   # ceil(27.1/9) = 4
    assert clips_per_shot(36.0) == 4
    assert clips_per_shot(45.0) == 5


def test_clips_per_shot_edge_cases():
    from nodes.otr_shot_duration_calculator import clips_per_shot
    assert clips_per_shot(0.0) == 1    # defensive default
    assert clips_per_shot(-5.0) == 1   # negative guard
    assert clips_per_shot(None) == 1   # None guard


def test_clips_per_shot_custom_target_cap():
    from nodes.otr_shot_duration_calculator import clips_per_shot
    # Change target + cap and the boundaries shift
    assert clips_per_shot(10.0, target_s=5.0, cap_s=6.0) == 2   # 10>6 -> ceil(10/5) = 2
    assert clips_per_shot(6.0, target_s=5.0, cap_s=6.0) == 1    # 6<=6 -> 1


def test_clip_duration_even_split():
    from nodes.otr_shot_duration_calculator import clip_duration_s
    assert clip_duration_s(15.0, 2) == 7.5
    assert clip_duration_s(27.0, 3) == 9.0
    assert clip_duration_s(19.0, 3) == pytest.approx(6.3333, rel=1e-3)
    assert clip_duration_s(12.0, 2) == 6.0


def test_clip_duration_zero_guards():
    from nodes.otr_shot_duration_calculator import clip_duration_s
    assert clip_duration_s(0.0, 3) == 0.0
    assert clip_duration_s(15.0, 0) == 0.0
    assert clip_duration_s(15.0, -1) == 0.0
    assert clip_duration_s(None, 3) == 0.0


def test_total_frames_math():
    """FLF chain: N clips = N+1 frames (0 clips = 0 frames edge case)."""
    from nodes.otr_shot_duration_calculator import total_frames_for_clip_counts
    assert total_frames_for_clip_counts([]) == 0
    assert total_frames_for_clip_counts([0]) == 0
    assert total_frames_for_clip_counts([1]) == 2           # 1 clip, 2 frames
    assert total_frames_for_clip_counts([1, 1, 1]) == 4     # 3 clips, 4 frames
    assert total_frames_for_clip_counts([1, 2, 3]) == 7     # 6 clips, 7 frames
    assert total_frames_for_clip_counts([2, 3, 1, 4]) == 11


# ------------------------------------------------------------------
# expand_plan_with_durations
# ------------------------------------------------------------------


def _minimal_plan(shot_count: int = 3) -> dict:
    """Hand-craft a minimal OTR_VideoPlan output with N shots, 1 clip each."""
    shots = []
    tokens = []
    for i in range(shot_count):
        shot_id = f"shot_{i+1:03d}"
        start_f = f"frame_{i:04d}"
        end_f = f"frame_{i+1:04d}"
        shots.append({
            "shot_id": shot_id,
            "scene_id": f"scene_{(i // 2) + 1}",
            "shot_duration_s": None,
            "segments": [{
                "clip_id": f"{shot_id}_c1",
                "start_frame_id": start_f,
                "end_frame_id": end_f,
            }],
            "start_frame_id": start_f,
            "end_frame_id": end_f,
        })
        tokens.append({
            "type": "environment",
            "description": f"composite prompt for {shot_id}",
            "frame_id": start_f,
            "shot_id": start_f,
            "scene_id": f"scene_{(i // 2) + 1}",
            "starts_shot": shot_id,
            "ends_shot": shots[-2]["shot_id"] if i > 0 else None,
            "kind": "start" if i == 0 else "bridge",
        })
    # final end frame
    final_frame = f"frame_{shot_count:04d}"
    tokens.append({
        "type": "environment",
        "description": f"composite prompt for last shot end",
        "frame_id": final_frame,
        "shot_id": final_frame,
        "scene_id": shots[-1]["scene_id"],
        "starts_shot": None,
        "ends_shot": shots[-1]["shot_id"],
        "kind": "end",
    })
    return {
        "tokens": tokens,
        "shots": shots,
        "source": "OTR_VideoPlan",
        "focus_character": "LEMMY",
        "shots_per_scene": 1,
        "scenes_covered": (shot_count + 1) // 2,
        "total_shots": shot_count,
        "total_prompts": len(tokens),
    }


def test_expand_default_single_clip_per_shot():
    """All shots under cap => each stays 1 clip; structure preserved."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(3)
    expanded = expand_plan_with_durations(plan, [6.0, 8.0, 7.0])
    assert expanded["total_shots"] == 3
    assert expanded["total_clips"] == 3
    assert expanded["total_prompts"] == 4   # N+1 = 3+1
    for s in expanded["shots"]:
        assert s["clips_per_shot"] == 1
        assert len(s["segments"]) == 1


def test_expand_multi_clip_for_long_shot():
    """A 27s shot => 3 clips; frame count expands accordingly."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(3)
    # shots: 6s, 27s, 7s  -> 1 + 3 + 1 = 5 clips -> 6 frames
    expanded = expand_plan_with_durations(plan, [6.0, 27.0, 7.0])
    assert expanded["total_clips"] == 5
    assert expanded["total_prompts"] == 6

    shots = expanded["shots"]
    assert shots[0]["clips_per_shot"] == 1
    assert shots[1]["clips_per_shot"] == 3
    assert shots[2]["clips_per_shot"] == 1
    # Middle shot should have 3 segments
    assert len(shots[1]["segments"]) == 3
    # Each segment in the 27s shot should be 9.0s
    for seg in shots[1]["segments"]:
        assert seg["duration_s"] == 9.0


def test_expand_mixed_durations():
    """Worked example from documentation: 7, 15, 8, 19 => 1+2+1+3 = 7 clips, 8 frames."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(4)
    expanded = expand_plan_with_durations(plan, [7.0, 15.0, 8.0, 19.0])
    assert expanded["total_clips"] == 7
    assert expanded["total_prompts"] == 8


def test_expand_frame_sharing_invariant():
    """Critical FLF property: adjacent clips share boundary frames."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(4)
    expanded = expand_plan_with_durations(plan, [7.0, 15.0, 8.0, 19.0])

    # Across ALL shots and ALL clips: clip N+1 start == clip N end
    all_segments = []
    for shot in expanded["shots"]:
        all_segments.extend(shot["segments"])
    for i in range(len(all_segments) - 1):
        assert all_segments[i]["end_frame_id"] == all_segments[i + 1]["start_frame_id"], (
            f"FLF chain broken between clip {i} and clip {i+1}: "
            f"{all_segments[i]['end_frame_id']} != {all_segments[i + 1]['start_frame_id']}"
        )


def test_expand_frame_ids_are_global_sequential():
    """Frame IDs should be frame_0000, 0001, 0002, ... unique, sequential."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(3)
    expanded = expand_plan_with_durations(plan, [8.0, 15.0, 8.0])
    # Collect all frame IDs referenced by segments
    all_frames = set()
    for s in expanded["shots"]:
        for seg in s["segments"]:
            all_frames.add(seg["start_frame_id"])
            all_frames.add(seg["end_frame_id"])
    # 4 clips total -> 5 unique frames: frame_0000..frame_0004
    assert all_frames == {
        "frame_0000", "frame_0001", "frame_0002", "frame_0003", "frame_0004"
    }


def test_expand_clip_ids_indexed_from_one():
    """Clip IDs are shot_NNN_c1, shot_NNN_c2, ..."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(2)
    expanded = expand_plan_with_durations(plan, [8.0, 27.0])  # 1 + 3 = 4 clips
    shot2 = expanded["shots"][1]
    clip_ids = [seg["clip_id"] for seg in shot2["segments"]]
    assert clip_ids == [
        f"{shot2['shot_id']}_c1",
        f"{shot2['shot_id']}_c2",
        f"{shot2['shot_id']}_c3",
    ]


def test_expand_preserves_shot_metadata():
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(2)
    expanded = expand_plan_with_durations(plan, [9.0, 15.0])
    # Scene IDs preserved
    assert expanded["shots"][0]["scene_id"] == plan["shots"][0]["scene_id"]
    assert expanded["shots"][1]["scene_id"] == plan["shots"][1]["scene_id"]
    # shot_duration_s filled in
    assert expanded["shots"][0]["shot_duration_s"] == 9.0
    assert expanded["shots"][1]["shot_duration_s"] == 15.0
    # durations_applied flag set
    assert expanded["durations_applied"] is True


def test_expand_tokens_reuse_shot_composes():
    """Expanded tokens should carry the owning shot's composite prompt."""
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(2)
    expanded = expand_plan_with_durations(plan, [8.0, 15.0])
    # 3 clips -> 4 frames -> 4 tokens
    assert len(expanded["tokens"]) == 4
    # Token for first frame should reference shot_001's compose
    first = expanded["tokens"][0]
    assert "shot_001" in first["description"]
    # All tokens should have the env-token shape for BatchFluxRender
    for tok in expanded["tokens"]:
        assert tok["type"] == "environment"
        assert tok.get("description")
        assert tok.get("frame_id")


def test_expand_empty_plan_raises():
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    with pytest.raises(ValueError):
        expand_plan_with_durations({"shots": [], "tokens": []}, [5.0])


def test_expand_shot_duration_mismatch_raises():
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(3)
    with pytest.raises(ValueError, match="mismatch"):
        expand_plan_with_durations(plan, [5.0])     # 1 dur vs 3 shots


def test_expand_does_not_mutate_input():
    from nodes.otr_shot_duration_calculator import expand_plan_with_durations
    plan = _minimal_plan(2)
    original_shots_copy = [dict(s) for s in plan["shots"]]
    _ = expand_plan_with_durations(plan, [8.0, 15.0])
    # Input plan should be untouched
    for orig, cur in zip(original_shots_copy, plan["shots"]):
        assert orig == cur


# ------------------------------------------------------------------
# ComfyUI node surface area
# ------------------------------------------------------------------


def test_node_registered():
    from nodes.otr_shot_duration_calculator import (
        NODE_CLASS_MAPPINGS, OTRShotDurationCalculator,
    )
    assert "OTR_ShotDurationCalculator" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["OTR_ShotDurationCalculator"] is OTRShotDurationCalculator


def test_input_types_schema():
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    schema = OTRShotDurationCalculator.INPUT_TYPES()
    assert "pass3_compose_prompts_json" in schema["required"]
    assert "shot_durations_json" in schema["required"]
    assert schema["required"]["pass3_compose_prompts_json"][0] == "STRING"
    assert schema["required"]["shot_durations_json"][0] == "STRING"


def test_return_types():
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    assert OTRShotDurationCalculator.RETURN_TYPES == ("STRING", "INT", "INT", "STRING")
    assert OTRShotDurationCalculator.RETURN_NAMES == (
        "expanded_pass3_json", "total_clips", "total_frames", "debug_summary"
    )


def test_calculate_method_end_to_end():
    """Feed a minimal plan + durations, get expanded output back."""
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    node = OTRShotDurationCalculator()
    plan = _minimal_plan(3)
    plan_json = json.dumps(plan)
    durations_json = json.dumps([6.0, 15.0, 27.0])

    out_json, total_clips, total_frames, summary = node.calculate(
        pass3_compose_prompts_json=plan_json,
        shot_durations_json=durations_json,
    )
    out = json.loads(out_json)
    # 1 + 2 + 3 = 6 clips, 7 frames
    assert total_clips == 6
    assert total_frames == 7
    assert out["total_clips"] == 6
    assert out["total_prompts"] == 7
    # Summary should mention clips/frames
    assert "total clips" in summary.lower()
    assert "total frames" in summary.lower()


def test_calculate_method_empty_durations_defaults():
    """Empty durations_json -> default 8.0s per shot."""
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    node = OTRShotDurationCalculator()
    plan = _minimal_plan(3)
    out_json, total_clips, total_frames, _ = node.calculate(
        pass3_compose_prompts_json=json.dumps(plan),
        shot_durations_json="",
    )
    # 3 shots at 8.0s each = 1 clip each = 3 clips, 4 frames
    assert total_clips == 3
    assert total_frames == 4


def test_calculate_method_output_batchfluxrender_shaped():
    """Smoke: output must match the schema BatchFluxRender reads."""
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    node = OTRShotDurationCalculator()
    plan = _minimal_plan(2)
    out_json, _, _, _ = node.calculate(
        pass3_compose_prompts_json=json.dumps(plan),
        shot_durations_json=json.dumps([8.0, 15.0]),
    )
    payload = json.loads(out_json)
    assert isinstance(payload.get("tokens"), list)
    for tok in payload["tokens"]:
        assert tok["type"] == "environment"
        assert isinstance(tok.get("description"), str)


def test_calculate_method_custom_target_and_cap():
    """Caller can override segment target and hard cap."""
    from nodes.otr_shot_duration_calculator import OTRShotDurationCalculator
    node = OTRShotDurationCalculator()
    plan = _minimal_plan(1)
    # With target=5 cap=6, a 10s shot becomes 2 clips (ceil(10/5))
    out_json, total_clips, _, _ = node.calculate(
        pass3_compose_prompts_json=json.dumps(plan),
        shot_durations_json=json.dumps([10.0]),
        segment_target_s=5.0,
        segment_hard_cap_s=6.0,
    )
    assert total_clips == 2
