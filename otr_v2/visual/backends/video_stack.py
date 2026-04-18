"""
otr_v2.visual.backends.video_stack  --  composite still+motion pipeline
========================================================================

Chains two backends in sequence within a single sidecar spawn so each
shot produces BOTH a unique still (render.png) AND a motion clip
(render.mp4).  Without this, single-backend dispatch gives either
stills OR motion, never both, and the legacy Ken Burns "auto" path was
the only code path producing per-shot motion MP4s -- which is why the
2026-04-17 overnight run looked identical to v1.7.

Sequence (each backend may run in real or stub mode independently):

    1. flux_anchor -> writes <shot>/render.png + <shot>/meta.json
    2. wan21_loop  -> consumes render.png, writes <shot>/loop.mp4
    3. promote     -> copy loop.mp4 -> render.mp4 (renderer.py looks
                       for render.mp4 first, render.png as fallback)

Why wan21_loop and not ltx_motion?
    - LTX-Video weights aren't in the download manifest; it would
      always stub and produce non-playable MP4 bytes.
    - Wan-AI/Wan2.1-T2V-1.3B-Diffusers IS in the manifest and the
      wan21_loop backend falls back gracefully to the T2V pipeline
      when WanImageToVideoPipeline isn't available (see
      wan21_loop._try_load_wan_pipeline).

Stub-safety:
    - If FLUX weights are missing/gated, flux_anchor writes
      per-shot deterministic colored PNGs (distinct by shot_id).
    - If Wan2.1 weights are missing, wan21_loop writes a minimal
      MP4 ftyp+mdat skeleton keyed on the input-still hash.  That
      file is NOT playable, so the promote step SKIPS the copy when
      loop.mp4 is not a real video -- renderer falls through to the
      still, and VisualRenderer._still_to_clip Ken-Burns it.

This keeps the worker.run_stub legacy path available (backend="auto")
while giving the workflow a single new "video_stack" option that
produces visible unique per-shot output at every weight-availability
tier.

Per-shot seed base 0x56_53_54_4B spells "VSTK" in ASCII so composite
metadata doesn't collide with any individual backend's meta.json.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ._base import (
    Backend,
    STATUS_ERROR,
    STATUS_READY,
    STATUS_RUNNING,
    write_status,
)


# A real (playable) Wan2.1 MP4 will be much larger than the stub's
# 40-byte ftyp+mdat skeleton.  We use 4 KB as the promotion threshold:
# anything smaller is assumed to be the stub and left in place so the
# renderer falls through to the still frame.
_REAL_MP4_MIN_BYTES = 4096


def _promote_loop_to_render(shot_dir: Path) -> tuple[bool, str]:
    """Copy loop.mp4 -> render.mp4 when loop.mp4 is a real clip.

    Returns (promoted, reason).  Reason is logged to meta.json for
    post-run debugging when motion is missing from the final episode.
    """
    loop = shot_dir / "loop.mp4"
    if not loop.exists():
        return (False, "no_loop_mp4")

    try:
        size = loop.stat().st_size
    except OSError as exc:
        return (False, f"stat_error:{exc}")

    if size < _REAL_MP4_MIN_BYTES:
        return (False, f"loop_is_stub:{size}B<{_REAL_MP4_MIN_BYTES}B")

    render = shot_dir / "render.mp4"
    try:
        shutil.copy2(loop, render)
    except OSError as exc:
        return (False, f"copy_error:{exc}")
    return (True, f"promoted:{size}B")


class VideoStackBackend(Backend):
    """Composite: flux_anchor still -> wan21_loop motion -> promote to render.mp4."""

    name = "video_stack"

    def run(self, job_dir: Path) -> None:
        out_dir = self.out_dir_for(job_dir)
        try:
            shots = self.load_shotlist(job_dir)
        except (FileNotFoundError, ValueError) as exc:
            write_status(out_dir, STATUS_ERROR, f"{type(exc).__name__}: {exc}")
            return

        if not shots:
            write_status(out_dir, STATUS_ERROR, "shotlist has zero shots")
            return

        t0 = time.time()
        write_status(
            out_dir, STATUS_RUNNING,
            f"video_stack: stage 1/3 flux_anchor on {len(shots)} shots",
            backend=self.name, stage="flux_anchor",
        )

        # Stage 1: stills.  Lazy import so import-time stays torch-free.
        try:
            from . import flux_anchor
            flux_anchor.FluxAnchorBackend().run(job_dir)
        except Exception as exc:  # noqa: BLE001
            write_status(
                out_dir, STATUS_ERROR,
                f"video_stack: flux_anchor raised {type(exc).__name__}: {exc}",
                backend=self.name, stage="flux_anchor",
            )
            return

        # Stage 2: motion.  Consumes <shot>/render.png or keyframe.png.
        write_status(
            out_dir, STATUS_RUNNING,
            f"video_stack: stage 2/3 wan21_loop on {len(shots)} shots",
            backend=self.name, stage="wan21_loop",
        )
        try:
            from . import wan21_loop
            wan21_loop.Wan21LoopBackend().run(job_dir)
        except Exception as exc:  # noqa: BLE001
            # Motion failure is NOT fatal -- renderer still picks up
            # the stills from stage 1 and Ken Burns them.  Log and
            # continue to promotion.
            write_status(
                out_dir, STATUS_RUNNING,
                f"video_stack: wan21_loop raised {type(exc).__name__}: {exc}; "
                f"continuing with stills only",
                backend=self.name, stage="wan21_loop",
            )

        # Stage 3: promote loop.mp4 -> render.mp4 so renderer.py picks
        # it up.  Skips stub MP4s (they'd fail ffmpeg concat).
        write_status(
            out_dir, STATUS_RUNNING,
            f"video_stack: stage 3/3 promote loop.mp4 -> render.mp4",
            backend=self.name, stage="promote",
        )
        promoted = 0
        stub_skipped = 0
        missing = 0
        for shot in shots:
            shot_id = shot.get("shot_id", "")
            if not shot_id:
                continue
            shot_dir = out_dir / shot_id
            if not shot_dir.is_dir():
                missing += 1
                continue
            ok, reason = _promote_loop_to_render(shot_dir)
            if ok:
                promoted += 1
            elif reason.startswith("loop_is_stub"):
                stub_skipped += 1
            else:
                missing += 1

        detail = (
            f"video_stack READY: {len(shots)} shots, "
            f"motion_promoted={promoted}, stub_skipped={stub_skipped}, "
            f"missing={missing}, wall_clock_s={time.time() - t0:.1f}"
        )
        write_status(
            out_dir, STATUS_READY, detail,
            backend=self.name,
            mode="composite",
            motion_promoted=promoted,
            stub_skipped=stub_skipped,
            missing=missing,
        )
