"""
otr_v2.visual.backends.video_stack  --  composite still+motion pipeline
========================================================================

Chains motion backends in sequence within a single sidecar spawn so
each shot produces BOTH a unique still (render.png) AND a motion clip
(render.mp4).  Without this, single-backend dispatch gives either
stills OR motion, never both, and the legacy Ken Burns "auto" path was
the only code path producing per-shot motion MP4s -- which is why the
2026-04-17 overnight run looked identical to v1.7.

Sequence (each backend may run in real or stub mode independently):

    1. flux_anchor -> writes <shot>/render.png + <shot>/meta.json
    2. ltx_motion  -> consumes render.png, writes <shot>/motion.mp4
                      (PREFERRED -- LTX-2.3 I2V, 24fps, 10s clips,
                      fp8_e4m3fn Blackwell-native)
    3. wan21_loop  -> fallback when ltx_motion produced no real clip;
                      consumes render.png, writes <shot>/loop.mp4
    4. promote     -> copy motion.mp4 OR loop.mp4 -> render.mp4
                      (renderer.py looks for render.mp4 first,
                      render.png as fallback)

Why LTX first, Wan2.1 as fallback?
    - LTX-2.3 was purpose-built for I2V handoff from stills; prompt
      adherence is stronger and Blackwell-native fp8 fits the 16 GB
      VRAM ceiling without heavy CPU-offload churn.
    - Wan2.1-T2V-1.3B remains the fallback because the T2V pipeline
      loads even when the I2V variant isn't available, guaranteeing
      we still attempt motion if LTX weights are missing or the
      pipeline errors out mid-shot.

Stub-safety:
    - If FLUX weights are missing/gated, flux_anchor writes
      per-shot deterministic colored PNGs (distinct by shot_id).
    - If LTX weights are missing, ltx_motion writes a minimal MP4
      ftyp+mdat skeleton keyed on the input-still hash.  That file
      is NOT playable; the promote step SKIPS it and falls through
      to the Wan2.1 stage.
    - If Wan2.1 weights are ALSO missing, wan21_loop writes its own
      stub.  Promote sees both stubs, writes nothing, and renderer
      falls through to the still which _still_to_clip Ken-Burnses
      with a procgen overlay (scan-lines, signal-lost static).

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


def _promote_motion_to_render(shot_dir: Path) -> tuple[bool, str]:
    """Copy motion.mp4 (LTX) or loop.mp4 (Wan2.1) -> render.mp4.

    Checks LTX first; falls through to Wan2.1 if LTX produced nothing
    or only a stub.  Returns (promoted, reason).  Reason is logged to
    meta.json for post-run debugging when motion is missing.
    """
    render = shot_dir / "render.mp4"
    candidates = [
        ("motion.mp4", shot_dir / "motion.mp4"),  # LTX output
        ("loop.mp4",   shot_dir / "loop.mp4"),    # Wan2.1 output
    ]
    first_reason = ""
    for tag, src in candidates:
        if not src.exists():
            if not first_reason:
                first_reason = f"no_{tag}"
            continue
        try:
            size = src.stat().st_size
        except OSError as exc:
            if not first_reason:
                first_reason = f"stat_error:{tag}:{exc}"
            continue
        if size < _REAL_MP4_MIN_BYTES:
            if not first_reason:
                first_reason = f"{tag}_is_stub:{size}B<{_REAL_MP4_MIN_BYTES}B"
            continue
        try:
            shutil.copy2(src, render)
        except OSError as exc:
            return (False, f"copy_error:{tag}:{exc}")
        return (True, f"promoted_{tag}:{size}B")
    return (False, first_reason or "no_motion_candidates")


# Back-compat alias — older tests / imports may still reference
# _promote_loop_to_render.  Same behaviour; new name is accurate.
_promote_loop_to_render = _promote_motion_to_render


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

        # Stage 2a: LTX-2.3 motion (preferred).  Consumes
        # <shot>/render.png and writes <shot>/motion.mp4.
        write_status(
            out_dir, STATUS_RUNNING,
            f"video_stack: stage 2a/4 ltx_motion on {len(shots)} shots",
            backend=self.name, stage="ltx_motion",
        )
        ltx_ok = False
        try:
            from . import ltx_motion
            ltx_motion.LtxMotionBackend().run(job_dir)
            ltx_ok = True
        except Exception as exc:  # noqa: BLE001
            # Motion failure is NOT fatal -- Wan2.1 fallback + still
            # fallback still cover the output.  Log and continue.
            write_status(
                out_dir, STATUS_RUNNING,
                f"video_stack: ltx_motion raised {type(exc).__name__}: {exc}; "
                f"falling through to wan21_loop",
                backend=self.name, stage="ltx_motion",
            )

        # Decide whether Wan2.1 fallback is worth running.  If LTX
        # produced at least one real (non-stub) motion.mp4 for every
        # shot, we can skip Wan2.1 and save VRAM/wall-clock.  If any
        # shot is missing real motion, run Wan2.1 to backfill.
        needs_wan_backfill = False
        for shot in shots:
            shot_id = shot.get("shot_id", "")
            if not shot_id:
                continue
            motion_mp4 = out_dir / shot_id / "motion.mp4"
            if not motion_mp4.exists():
                needs_wan_backfill = True
                break
            try:
                if motion_mp4.stat().st_size < _REAL_MP4_MIN_BYTES:
                    needs_wan_backfill = True
                    break
            except OSError:
                needs_wan_backfill = True
                break

        # Stage 2b: Wan2.1 1.3B fallback (only if LTX left gaps).
        if needs_wan_backfill:
            write_status(
                out_dir, STATUS_RUNNING,
                f"video_stack: stage 2b/4 wan21_loop fallback on shots "
                f"without real LTX motion",
                backend=self.name, stage="wan21_loop",
            )
            try:
                from . import wan21_loop
                wan21_loop.Wan21LoopBackend().run(job_dir)
            except Exception as exc:  # noqa: BLE001
                write_status(
                    out_dir, STATUS_RUNNING,
                    f"video_stack: wan21_loop raised {type(exc).__name__}: "
                    f"{exc}; continuing with whatever LTX produced + stills",
                    backend=self.name, stage="wan21_loop",
                )
        else:
            write_status(
                out_dir, STATUS_RUNNING,
                f"video_stack: stage 2b/4 skipped (LTX covered all shots)",
                backend=self.name, stage="wan21_loop",
            )

        # Stage 3: promote motion.mp4 or loop.mp4 -> render.mp4 so
        # renderer.py picks it up.  Skips stub MP4s (they'd fail
        # ffmpeg concat) and falls through to the still.
        write_status(
            out_dir, STATUS_RUNNING,
            f"video_stack: stage 3/4 promote motion.mp4|loop.mp4 -> render.mp4",
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
            ok, reason = _promote_motion_to_render(shot_dir)
            if ok:
                promoted += 1
            elif "_is_stub" in reason:
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
