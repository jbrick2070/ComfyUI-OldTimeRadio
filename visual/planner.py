"""
otr_v2.visual.planner  --  orchestration timeline planner (Day 9)
==================================================================

Given a parsed outline_json (scenes + beats + characters + timing),
the planner produces a non-repeating sidecar job list that covers the
full target runtime.  Each job in the output names exactly one of the
seven Day 1-7 backends and carries everything the bridge + sidecar
need to execute it:

    * ``shot_id``      -- stable identifier
    * ``backend``      -- flux_anchor / pulid_portrait / flux_keyframe /
                          ltx_motion / wan21_loop / florence2_sdxl_comp /
                          placeholder_test
    * ``prompt``       -- per-backend prompt string (motion_prompt /
                          loop_prompt / mask_prompt / insert_prompt as
                          appropriate)
    * ``duration_s``   -- per-shot runtime in seconds (respects C4
                          10 s cap for motion/loop backends)
    * ``refs``         -- optional identity refs (PuLID characters)
    * ``handoff_from`` -- upstream shot whose still is the seed for
                          motion/loop backends (Day 5-6 handoff)
    * ``scene_id``     -- parent scene identifier
    * ``prompt_hash``  -- SHA256(backend + prompt + refs) 12-char prefix
                          used by the non-repetition window

Non-repetition:  no ``(backend, prompt_hash)`` tuple may appear twice
inside a configurable sliding window of consecutive jobs.  Default
window = 3 jobs.

No torch / diffusers imports.  Pure stdlib.  Safe to import from
unit tests and from the bridge.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger("OTR.visual.planner")


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MAX_MOTION_DURATION_S: float = 10.0
DEFAULT_BEAT_DURATION_S: float = 6.0
DEFAULT_NONREPEAT_WINDOW: int = 3

BEAT_KIND_TO_BACKENDS: dict[str, tuple[str, ...]] = {
    "establishing": ("flux_anchor",),
    "wide": ("flux_anchor",),
    "master": ("flux_anchor",),
    "anchor": ("flux_anchor",),
    "close_up": ("pulid_portrait", "flux_keyframe"),
    "two_shot": ("pulid_portrait", "flux_keyframe"),
    "portrait": ("pulid_portrait",),
    "keyframe": ("flux_keyframe",),
    "scene": ("flux_keyframe",),
    "interior": ("flux_keyframe",),
    "motion": ("ltx_motion",),
    "action": ("ltx_motion",),
    "camera": ("ltx_motion",),
    "loop": ("wan21_loop",),
    "ambient": ("wan21_loop",),
    "idle": ("wan21_loop",),
    "insert": ("florence2_sdxl_comp",),
    "composite": ("florence2_sdxl_comp",),
    "overlay": ("florence2_sdxl_comp",),
    "hud": ("florence2_sdxl_comp",),
}

_KNOWN_BACKENDS: frozenset[str] = frozenset({
    "placeholder_test",
    "flux_anchor",
    "pulid_portrait",
    "flux_keyframe",
    "ltx_motion",
    "wan21_loop",
    "florence2_sdxl_comp",
})

HANDOFF_BACKENDS: frozenset[str] = frozenset({"ltx_motion", "wan21_loop"})
IDENTITY_BACKENDS: frozenset[str] = frozenset({"pulid_portrait"})
COMPOSITE_BACKENDS: frozenset[str] = frozenset({"florence2_sdxl_comp"})


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerJob:
    shot_id: str
    backend: str
    scene_id: str
    prompt: str
    duration_s: float
    refs: tuple[str, ...] = ()
    character: str | None = None
    handoff_from: str | None = None
    mask_prompt: str | None = None
    insert_prompt: str | None = None
    prompt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "shot_id": self.shot_id,
            "backend": self.backend,
            "scene_id": self.scene_id,
            "prompt": self.prompt,
            "duration_s": round(self.duration_s, 3),
            "prompt_hash": self.prompt_hash,
        }
        if self.refs:
            payload["refs"] = list(self.refs)
        if self.character:
            payload["character"] = self.character
        if self.handoff_from:
            payload["handoff_from"] = self.handoff_from
        if self.mask_prompt:
            payload["mask_prompt"] = self.mask_prompt
        if self.insert_prompt:
            payload["insert_prompt"] = self.insert_prompt
        return payload


@dataclass
class PlannerResult:
    jobs: list = field(default_factory=list)
    total_duration_s: float = 0.0
    target_runtime_s: float = 0.0
    scenes_covered: int = 0
    warnings: list = field(default_factory=list)
    repetition_window: int = DEFAULT_NONREPEAT_WINDOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobs": [j.to_dict() for j in self.jobs],
            "total_duration_s": round(self.total_duration_s, 3),
            "target_runtime_s": round(self.target_runtime_s, 3),
            "scenes_covered": self.scenes_covered,
            "job_count": len(self.jobs),
            "warnings": list(self.warnings),
            "repetition_window": self.repetition_window,
        }


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


def _prompt_hash(backend: str, prompt: str, refs: Sequence[str]) -> str:
    ref_blob = "|".join(sorted(refs))
    key = f"{backend}|{prompt}|{ref_blob}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _infer_backend(beat: dict, previous_backend):
    explicit = (beat.get("backend") or "").strip().lower()
    if explicit:
        if explicit not in _KNOWN_BACKENDS:
            raise ValueError(
                f"beat {beat.get('beat_id')!r} names unknown backend {explicit!r}"
            )
        return explicit

    kind = (beat.get("kind") or "").strip().lower()
    candidates = BEAT_KIND_TO_BACKENDS.get(kind, ())
    for name in candidates:
        if name == "pulid_portrait":
            if not beat.get("character") or not beat.get("refs"):
                continue
        if name == "florence2_sdxl_comp":
            if not beat.get("mask_prompt") or not beat.get("insert_prompt"):
                continue
        return name

    return "flux_keyframe"


def _clamp_duration(backend: str, duration_s: float):
    if duration_s <= 0:
        return (DEFAULT_BEAT_DURATION_S, f"duration_s<=0 replaced with default")
    if backend in HANDOFF_BACKENDS and duration_s > MAX_MOTION_DURATION_S:
        return (
            MAX_MOTION_DURATION_S,
            f"{backend} duration {duration_s:.1f}s clamped to C4 cap "
            f"{MAX_MOTION_DURATION_S:.0f}s",
        )
    return (float(duration_s), None)


def _pick_handoff_from(jobs: Sequence[PlannerJob], scene_id: str):
    for prior in reversed(jobs):
        if prior.scene_id != scene_id:
            continue
        if prior.backend in HANDOFF_BACKENDS:
            continue
        if prior.backend in (
            "flux_anchor", "pulid_portrait", "flux_keyframe",
            "florence2_sdxl_comp",
        ):
            return prior.shot_id
    return None


def _violates_nonrepeat(pending: PlannerJob, window_jobs: Sequence[PlannerJob]) -> bool:
    key = (pending.backend, pending.prompt_hash)
    for j in window_jobs:
        if (j.backend, j.prompt_hash) == key:
            return True
    return False


def _nudge_prompt_for_uniqueness(backend: str, prompt: str, refs: Sequence[str], attempt: int):
    suffix = f" [variant {attempt}]"
    nudged = prompt + suffix
    return (nudged, _prompt_hash(backend, nudged, refs))


def _build_job_from_beat(scene: dict, beat: dict, shot_counter: int, prior_jobs: Sequence[PlannerJob], warnings: list) -> PlannerJob:
    scene_id = str(scene.get("scene_id") or f"scene_{len(prior_jobs):03d}")
    beat_id = str(beat.get("beat_id") or f"beat_{shot_counter:04d}")

    previous_backend = prior_jobs[-1].backend if prior_jobs else None
    backend = _infer_backend(beat, previous_backend)

    # When _infer_backend silently skipped florence2_sdxl_comp or pulid_portrait
    # due to missing fields, surface a degradation warning.
    explicit_backend = (beat.get("backend") or "").strip().lower()
    if not explicit_backend and backend == "flux_keyframe":
        kind = (beat.get("kind") or "").strip().lower()
        candidates = BEAT_KIND_TO_BACKENDS.get(kind, ())
        if candidates:
            first = candidates[0]
            if first == "florence2_sdxl_comp" and (
                not beat.get("mask_prompt") or not beat.get("insert_prompt")
            ):
                warnings.append(
                    f"beat {beat_id!r} needed florence2_sdxl_comp but "
                    f"missing mask_prompt/insert_prompt; degraded to "
                    f"flux_keyframe"
                )
            elif first == "pulid_portrait" and (
                not beat.get("character") or not beat.get("refs")
            ):
                warnings.append(
                    f"beat {beat_id!r} needed pulid_portrait but missing "
                    f"character/refs; degraded to flux_keyframe"
                )

    prompt = str(beat.get("prompt") or "").strip()
    if not prompt:
        prompt = f"{scene.get('location', 'scene')} -- {backend} shot"
        warnings.append(
            f"beat {beat_id!r} had empty prompt; synthesised from scene location"
        )

    refs_raw = beat.get("refs") or ()
    refs = tuple(str(r) for r in refs_raw)

    character = beat.get("character")
    if backend in IDENTITY_BACKENDS:
        if not character or not refs:
            warnings.append(
                f"beat {beat_id!r} needed pulid_portrait but missing "
                f"character/refs; degraded to flux_keyframe"
            )
            backend = "flux_keyframe"
            character = None
            refs = ()

    mask_prompt = beat.get("mask_prompt")
    insert_prompt = beat.get("insert_prompt")
    if backend in COMPOSITE_BACKENDS:
        if not mask_prompt or not insert_prompt:
            warnings.append(
                f"beat {beat_id!r} needed florence2_sdxl_comp but missing "
                f"mask_prompt/insert_prompt; degraded to flux_keyframe"
            )
            backend = "flux_keyframe"
            mask_prompt = None
            insert_prompt = None

    duration_raw = float(beat.get("duration_s", DEFAULT_BEAT_DURATION_S))
    duration_s, dur_warning = _clamp_duration(backend, duration_raw)
    if dur_warning:
        warnings.append(f"beat {beat_id!r}: {dur_warning}")

    handoff_from = None
    if backend in HANDOFF_BACKENDS:
        handoff_from = _pick_handoff_from(prior_jobs, scene_id)
        if handoff_from is None:
            warnings.append(
                f"beat {beat_id!r} ({backend}) has no upstream still in scene "
                f"{scene_id!r}; bridge will route this to stub mode"
            )

    shot_id = f"{scene_id}_{beat_id}_{shot_counter:04d}"

    prompt_hash = _prompt_hash(backend, prompt, refs)

    return PlannerJob(
        shot_id=shot_id,
        backend=backend,
        scene_id=scene_id,
        prompt=prompt,
        duration_s=duration_s,
        refs=refs,
        character=character if backend in IDENTITY_BACKENDS else None,
        handoff_from=handoff_from,
        mask_prompt=mask_prompt if backend in COMPOSITE_BACKENDS else None,
        insert_prompt=insert_prompt if backend in COMPOSITE_BACKENDS else None,
        prompt_hash=prompt_hash,
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def plan_episode(
    outline,
    *,
    target_runtime_s: float | None = None,
    nonrepeat_window: int = DEFAULT_NONREPEAT_WINDOW,
    default_beat_duration_s: float = DEFAULT_BEAT_DURATION_S,
) -> PlannerResult:
    parsed = _coerce_outline(outline)

    runtime = float(target_runtime_s if target_runtime_s is not None
                    else parsed.get("runtime_s", 0.0))
    if runtime <= 0:
        runtime = sum(
            float(b.get("duration_s", default_beat_duration_s))
            for scene in parsed.get("scenes", [])
            for b in scene.get("beats", [])
        )

    result = PlannerResult(
        target_runtime_s=runtime,
        repetition_window=max(1, int(nonrepeat_window)),
    )

    scenes = list(parsed.get("scenes", []))
    if not scenes:
        result.warnings.append("outline has zero scenes; emitting empty timeline")
        return result

    shot_counter = 0
    scene_cursor = 0
    scenes_seen = set()
    used_scene_ids_in_order = []

    while result.total_duration_s < runtime and len(result.jobs) < 10_000:
        scene = scenes[scene_cursor % len(scenes)]
        scene_id = str(scene.get("scene_id") or f"scene_{scene_cursor:03d}")
        if scene_id not in scenes_seen:
            scenes_seen.add(scene_id)
            used_scene_ids_in_order.append(scene_id)

        beats = list(scene.get("beats", []))
        if not beats:
            result.warnings.append(f"scene {scene_id!r} has no beats; skipping")
            scene_cursor += 1
            if scene_cursor >= len(scenes) * 20:
                result.warnings.append(
                    "bail: too many empty scene rotations; aborting plan"
                )
                break
            continue

        for beat in beats:
            if result.total_duration_s >= runtime:
                break

            pending = _build_job_from_beat(
                scene=scene,
                beat=beat,
                shot_counter=shot_counter,
                prior_jobs=result.jobs,
                warnings=result.warnings,
            )

            attempt = 0
            window_start = max(0, len(result.jobs) - result.repetition_window)
            window_jobs = result.jobs[window_start:]
            while _violates_nonrepeat(pending, window_jobs):
                attempt += 1
                nudged_prompt, nudged_hash = _nudge_prompt_for_uniqueness(
                    pending.backend, pending.prompt, pending.refs, attempt
                )
                pending = PlannerJob(
                    shot_id=pending.shot_id,
                    backend=pending.backend,
                    scene_id=pending.scene_id,
                    prompt=nudged_prompt,
                    duration_s=pending.duration_s,
                    refs=pending.refs,
                    character=pending.character,
                    handoff_from=pending.handoff_from,
                    mask_prompt=pending.mask_prompt,
                    insert_prompt=pending.insert_prompt,
                    prompt_hash=nudged_hash,
                )
                if attempt > 32:
                    result.warnings.append(
                        f"beat {beat.get('beat_id')!r}: could not clear "
                        f"non-repeat window after 32 nudges; accepting"
                    )
                    break

            result.jobs.append(pending)
            result.total_duration_s += pending.duration_s
            shot_counter += 1

        scene_cursor += 1

    result.scenes_covered = len(used_scene_ids_in_order)

    if result.total_duration_s < runtime:
        result.warnings.append(
            f"emitted timeline {result.total_duration_s:.1f}s < target "
            f"{runtime:.1f}s -- outline has insufficient content even after "
            f"scene rotation"
        )
    for job in result.jobs:
        if job.backend not in _KNOWN_BACKENDS:
            result.warnings.append(
                f"emitted job {job.shot_id!r} names unregistered backend "
                f"{job.backend!r} -- bridge will reject it"
            )

    return result


# ------------------------------------------------------------------
# Outline coercion + JSON I/O
# ------------------------------------------------------------------


def _coerce_outline(outline):
    if isinstance(outline, dict):
        return outline
    if isinstance(outline, (str, Path)):
        raw = str(outline)
        stripped = raw.lstrip()
        # JSON fast-path: a payload starting with { or [ is never a path.
        # Avoids Path.exists() raising "File name too long" on long JSON.
        if stripped.startswith("{") or stripped.startswith("["):
            text = raw
        else:
            try:
                as_path = Path(raw)
                is_file = as_path.exists() and as_path.is_file()
            except OSError:
                is_file = False
            if is_file:
                text = as_path.read_text(encoding="utf-8")
            else:
                text = raw
        return json.loads(text)
    raise TypeError(f"unsupported outline type: {type(outline).__name__}")


def emit_shotlist_json(result: PlannerResult) -> dict:
    return {
        "shots": [
            {
                **job.to_dict(),
            }
            for job in result.jobs
        ],
        "target_runtime_s": round(result.target_runtime_s, 3),
        "total_duration_s": round(result.total_duration_s, 3),
        "job_count": len(result.jobs),
        "warnings": list(result.warnings),
    }


def write_shotlist(result: PlannerResult, path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(emit_shotlist_json(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


__all__ = [
    "MAX_MOTION_DURATION_S",
    "DEFAULT_BEAT_DURATION_S",
    "DEFAULT_NONREPEAT_WINDOW",
    "BEAT_KIND_TO_BACKENDS",
    "HANDOFF_BACKENDS",
    "IDENTITY_BACKENDS",
    "COMPOSITE_BACKENDS",
    "PlannerJob",
    "PlannerResult",
    "plan_episode",
    "emit_shotlist_json",
    "write_shotlist",
]
