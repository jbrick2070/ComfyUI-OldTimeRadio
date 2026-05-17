"""
nodes.otr_shot_duration_calculator  --  OTR_ShotDurationCalculator
==================================================================

MIT-licensed, OTR-original. Pure-stdlib, no torch.

Expands a single-clip-per-shot plan from OTR_VideoPlan into the correct
multi-clip-per-shot structure based on actual shot durations. Math:

    clips_per_shot(dur) =
        1                               if dur <= SEGMENT_HARD_CAP_S
        ceil(dur / SEGMENT_TARGET_S)    otherwise
    clip_duration_s    = dur / clips_per_shot(dur)

    total_frames = sum(clips_per_shot for each shot) + 1   (FLF chain)

FLF (First-Last-Frame) invariant preserved: adjacent clips share their
boundary frames. Clip N's end_frame_id == clip N+1's start_frame_id.

Input:
    pass3_compose_prompts_json  -- JSON envelope from OTR_VideoPlan.
                                    Must contain "shots" and "tokens".
    shot_durations_json         -- STRING, JSON array of floats, one per
                                    shot in the plan (same order).

Output:
    expanded_pass3_json         -- Same envelope shape, but with:
                                    * shots[i].shot_duration_s filled in
                                    * shots[i].segments[] expanded to N
                                      clips per shot
                                    * frame_ids renumbered globally
                                    * tokens[] regenerated (N+1 total)
    total_clips                 -- INT sum of clips
    total_frames                -- INT count of unique FLUX frames
    debug_summary               -- human-readable breakdown

Duration calculator is a TBA pipeline step that runs AFTER Bark audio
renders (we need real audio timings). For now the node takes the
durations JSON as a widget so it can be exercised with hand-crafted
values and validated before the audio-timeline wiring lands.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

log = logging.getLogger("OTR.nodes.otr_shot_duration_calculator")


# Same constants as OTR_VideoPlan (locked in that module, duplicated here
# so this node can be imported standalone without a side dependency).
SEGMENT_TARGET_DURATION_S = 9.0
SEGMENT_HARD_CAP_S = 10.0


# ---------------------------------------------------------------------------
# Pure-math helpers (unit-testable without ComfyUI)
# ---------------------------------------------------------------------------


def clips_per_shot(
    duration_s: float,
    *,
    target_s: float = SEGMENT_TARGET_DURATION_S,
    cap_s: float = SEGMENT_HARD_CAP_S,
) -> int:
    """How many Wan-call-sized clips does a shot of this duration need?

    Rules:
      * Zero / negative / missing duration  -> 1 (defensive default)
      * <= cap_s (10s)                       -> 1 clip at the shot's full length
      * > cap_s                              -> ceil(duration / target_s)
                                                (even split, never tiny trailing
                                                clips, all clips <= cap_s)

    Worked examples at target=9.0 cap=10.0:
        6s   -> 1     15s  -> 2     27s  -> 3
        9s   -> 1     18s  -> 2     36s  -> 4
        10s  -> 1     19s  -> 3     45s  -> 5
        10.5 -> 2     20s  -> 3
    """
    if duration_s is None or duration_s <= 0:
        return 1
    if duration_s <= cap_s:
        return 1
    return math.ceil(duration_s / target_s)


def clip_duration_s(shot_duration_s: float, n_clips: int) -> float:
    """Even-split the shot's duration across N clips.

    All clips in the same shot share the same duration. Smart-divide
    (no tiny trailing clips). Returns 0.0 defensively if n_clips <= 0.
    """
    if n_clips <= 0 or shot_duration_s is None or shot_duration_s <= 0:
        return 0.0
    return shot_duration_s / n_clips


def total_frames_for_clip_counts(clip_counts: list[int]) -> int:
    """FLF chain: N total clips -> N + 1 unique frames (boundary sharing)."""
    total_clips = sum(clip_counts)
    if total_clips <= 0:
        return 0
    return total_clips + 1


# ---------------------------------------------------------------------------
# Core expansion — standalone, unit-testable, torch-free
# ---------------------------------------------------------------------------


def _load_plan(plan_json: str) -> dict:
    if not plan_json or not plan_json.strip():
        raise ValueError("FixedShotDurationStub: empty plan JSON")
    plan = json.loads(plan_json)
    if not isinstance(plan, dict):
        raise ValueError(
            f"FixedShotDurationStub: plan JSON root must be dict, got {type(plan).__name__}"
        )
    return plan


def _load_durations(durations_json: str, expected_count: int) -> list[float]:
    if not durations_json or not durations_json.strip():
        # Empty => default all shots to 8s (mid-range single-clip)
        log.info(
            "FixedShotDurationStub: empty durations, defaulting %d shots to 8.0s",
            expected_count,
        )
        return [8.0] * expected_count
    parsed = json.loads(durations_json)
    if not isinstance(parsed, list):
        raise ValueError(
            f"shot_durations_json must be a JSON array of floats, "
            f"got {type(parsed).__name__}"
        )
    if len(parsed) != expected_count:
        raise ValueError(
            f"shot_durations count mismatch: plan has {expected_count} shots, "
            f"durations has {len(parsed)}"
        )
    return [float(d) for d in parsed]


def _build_shot_composes(original_tokens: list[dict]) -> dict[str, str]:
    """Map shot_id -> composite prompt text from the original plan's tokens.

    OTR_VideoPlan tokens carry ``starts_shot`` / ``ends_shot`` pointers.
    We prefer the START shot's token for each shot's compose text since
    the start frame is where the shot's visual identity is established.
    """
    composes: dict[str, str] = {}
    for tok in original_tokens:
        desc = (tok.get("description") or "").strip()
        if not desc:
            continue
        starts = tok.get("starts_shot")
        ends = tok.get("ends_shot")
        # Prefer the start-of-shot token as canonical for each shot's compose.
        if starts and starts not in composes:
            composes[starts] = desc
        elif ends and ends not in composes:
            composes[ends] = desc
    return composes


def expand_plan_with_durations(
    plan: dict,
    shot_durations: list[float],
    *,
    target_s: float = SEGMENT_TARGET_DURATION_S,
    cap_s: float = SEGMENT_HARD_CAP_S,
) -> dict:
    """Re-emit a VideoPlan envelope with multi-clip shots based on durations.

    Input plan has ``shots[]`` with ``segments[]`` where each segment is
    a single clip (the default from OTR_VideoPlan pre-duration-calculator).
    This function:

      * Fills in ``shot_duration_s`` per shot from the durations array
      * Computes ``clips_per_shot`` per shot
      * Rebuilds ``segments[]`` as N clips per shot with even-split durations
      * Renumbers all frames globally as ``frame_NNNN``
      * Maintains FLF sharing: clip N's ``end_frame_id`` == clip N+1's ``start_frame_id``
      * Regenerates ``tokens[]`` — one per unique frame, carrying the
        owning shot's composite prompt as its description
      * Returns the expanded envelope (new dict, does not mutate input)
    """
    original_shots = plan.get("shots") or []
    original_tokens = plan.get("tokens") or []
    if len(original_shots) != len(shot_durations):
        raise ValueError(
            f"shot count mismatch: plan has {len(original_shots)} shots, "
            f"durations has {len(shot_durations)}"
        )

    # Per-shot compose text (shot_id -> description) from the original plan.
    # Multi-clip expansion reuses the same compose text for every frame
    # inside a shot (the per-frame shot_hint variation that OTR_VideoPlan
    # applied at generation time is baked in; we don't re-vary here).
    shot_composes = _build_shot_composes(original_tokens)

    new_shots: list[dict[str, Any]] = []
    frame_idx = 0  # global frame counter

    for shot_idx, (shot, dur) in enumerate(zip(original_shots, shot_durations)):
        n_clips = clips_per_shot(dur, target_s=target_s, cap_s=cap_s)
        per_clip_s = clip_duration_s(dur, n_clips)

        segments: list[dict[str, Any]] = []
        for clip_idx in range(n_clips):
            # Start frame: reuse the previous frame (shared boundary) except
            # for the very first clip of the episode which gets a fresh one.
            if shot_idx == 0 and clip_idx == 0:
                start_frame_id = f"frame_{frame_idx:04d}"
                frame_idx += 1
            elif clip_idx == 0:
                # First clip of this shot reuses the previous shot's last
                # end frame (seamless cross-shot chain).
                start_frame_id = new_shots[-1]["segments"][-1]["end_frame_id"]
            else:
                # Subsequent clip of this shot reuses this shot's previous
                # clip's end frame.
                start_frame_id = segments[-1]["end_frame_id"]

            # End frame is always a new slot.
            end_frame_id = f"frame_{frame_idx:04d}"
            frame_idx += 1

            segments.append({
                "clip_id": f"{shot['shot_id']}_c{clip_idx + 1}",
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id,
                "duration_s": round(per_clip_s, 3),
            })

        # Copy forward the shot's existing fields + overwrite with expanded.
        expanded_shot = {
            "shot_id": shot.get("shot_id"),
            "scene_id": shot.get("scene_id"),
            "shot_duration_s": round(float(dur), 3) if dur is not None else None,
            "clips_per_shot": n_clips,
            "segments": segments,
            "start_frame_id": segments[0]["start_frame_id"] if segments else None,
            "end_frame_id": segments[-1]["end_frame_id"] if segments else None,
        }
        new_shots.append(expanded_shot)

    # Build tokens: one per unique frame, in global frame order. Each
    # frame's description = the owning shot's composite prompt.
    # Assignment rule: a frame belongs to the shot where it's a
    # segment's START, except the very last frame of the episode which
    # belongs to the last shot (as its final end_frame).
    unique_frame_ids: list[str] = []
    seen_frames: set[str] = set()
    frame_to_shot: dict[str, dict] = {}
    for shot in new_shots:
        for seg in shot["segments"]:
            if seg["start_frame_id"] not in seen_frames:
                seen_frames.add(seg["start_frame_id"])
                unique_frame_ids.append(seg["start_frame_id"])
            # Claim the start frame for this shot if unclaimed.
            frame_to_shot.setdefault(seg["start_frame_id"], shot)
        # Also include the last end frame of each shot so end-of-episode
        # appears in the list.
        last_end = shot["segments"][-1]["end_frame_id"] if shot["segments"] else None
        if last_end and last_end not in seen_frames:
            seen_frames.add(last_end)
            unique_frame_ids.append(last_end)
            frame_to_shot.setdefault(last_end, shot)

    new_tokens: list[dict[str, Any]] = []
    for frame_id in unique_frame_ids:
        owning = frame_to_shot.get(frame_id)
        shot_id = owning["shot_id"] if owning else None
        scene_id = owning["scene_id"] if owning else None
        desc = shot_composes.get(shot_id, "") if shot_id else ""
        # `shot_id: frame_id` alias deleted S27 (cleanbreak-tail Phase 2
        # / QA-3). Audit found zero callers reading `shot_id` from these
        # envelope tokens; the canonical `frame_id` key is what every
        # consumer reads (3 in tests, none elsewhere). The `owning_shot`
        # key below is the real shot reference -- not the alias.
        new_tokens.append({
            "type": "environment",
            "description": desc,
            "role": "char_scene_composite",
            "frame_id": frame_id,
            "scene_id": scene_id,
            "owning_shot": shot_id,
            "focus_character": plan.get("focus_character", ""),
            "kind": "composite_expanded",
        })

    # Emit the expanded envelope (new dict, does not mutate input).
    out = dict(plan)
    out["tokens"] = new_tokens
    out["shots"] = new_shots
    out["total_shots"] = len(new_shots)
    out["total_clips"] = sum(s["clips_per_shot"] for s in new_shots)
    out["total_prompts"] = len(new_tokens)
    out["durations_applied"] = True
    out["segment_target_s"] = target_s
    out["segment_hard_cap_s"] = cap_s
    return out


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------


class OTRFixedShotDurationStub:
    """OTR_FixedShotDurationStub  --  expand 1-clip-per-shot plan into
    correct multi-clip structure based on real shot durations.

    Sprint E E8 rename from OTR_ShotDurationCalculator (M10). The old
    name implied dynamic computation; the current implementation
    operates on a hand-crafted JSON array of durations until the Bark
    audio-timeline wiring lands. The rename makes the stub-nature of
    the current behavior surface-visible. Per CLAUDE.md no-back-compat
    directive: no alias for the old name; pre-Sprint-E saved workflow
    JSONs must be rewritten.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pass3_compose_prompts_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "OTR_VideoPlan pass3_compose_prompts_json output. "
                            "Must contain 'shots' and 'tokens'."
                        ),
                    },
                ),
                "shot_durations_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[8.0, 9.0, 15.0, 6.0, 27.0, 8.0]",
                        "tooltip": (
                            "JSON array of floats, one per shot in the plan. "
                            "Order matches shots[] in the input plan. Until "
                            "the Bark audio-timeline wiring lands, this is "
                            "hand-crafted for testing. Empty string = default "
                            "all shots to 8.0s."
                        ),
                    },
                ),
            },
            "optional": {
                "segment_target_s": (
                    "FLOAT",
                    {"default": SEGMENT_TARGET_DURATION_S, "min": 2.0, "max": 10.0, "step": 0.5},
                ),
                "segment_hard_cap_s": (
                    "FLOAT",
                    {"default": SEGMENT_HARD_CAP_S, "min": 2.0, "max": 15.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "expanded_pass3_json",
        "total_clips",
        "total_frames",
        "debug_summary",
    )
    FUNCTION = "calculate"
    CATEGORY = "OldTimeRadio/video"

    def calculate(
        self,
        pass3_compose_prompts_json: str,
        shot_durations_json: str,
        segment_target_s: float = SEGMENT_TARGET_DURATION_S,
        segment_hard_cap_s: float = SEGMENT_HARD_CAP_S,
    ):
        plan = _load_plan(pass3_compose_prompts_json)
        original_shots = plan.get("shots") or []
        durations = _load_durations(shot_durations_json, len(original_shots))
        expanded = expand_plan_with_durations(
            plan,
            durations,
            target_s=segment_target_s,
            cap_s=segment_hard_cap_s,
        )

        total_clips = expanded["total_clips"]
        total_frames = expanded["total_prompts"]

        # Per-shot breakdown for the summary
        shot_lines = []
        for s in expanded["shots"]:
            dur = s.get("shot_duration_s")
            nc = s.get("clips_per_shot", 1)
            per_clip = (s["segments"][0]["duration_s"] if s["segments"] else 0.0)
            shot_lines.append(
                f"  {s['shot_id']:<10}  {str(dur)+'s':<8}  ->  "
                f"{nc} clip(s) x {per_clip}s each"
            )

        summary = "\n".join([
            f"total shots:   {len(original_shots)}",
            f"total clips:   {total_clips}",
            f"total frames:  {total_frames}  (N+1 via FLF sharing)",
            f"target/cap:    {segment_target_s}s / {segment_hard_cap_s}s",
            "",
            "per-shot breakdown:",
            *shot_lines,
        ])

        log.info(
            "OTR_FixedShotDurationStub READY: shots=%d clips=%d frames=%d",
            len(original_shots),
            total_clips,
            total_frames,
        )

        return (
            json.dumps(expanded, indent=2),
            total_clips,
            total_frames,
            summary,
        )


# ---------------------------------------------------------------------------
# Module-level registration
# ---------------------------------------------------------------------------


NODE_CLASS_MAPPINGS = {
    "OTR_FixedShotDurationStub": OTRFixedShotDurationStub,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_FixedShotDurationStub": " OTR Fixed Shot Duration Stub",
}

__all__ = [
    "OTRFixedShotDurationStub",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "clips_per_shot",
    "clip_duration_s",
    "total_frames_for_clip_counts",
    "expand_plan_with_durations",
    "SEGMENT_TARGET_DURATION_S",
    "SEGMENT_HARD_CAP_S",
]
