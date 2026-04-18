"""
otr_v2.hyworld.postproc  --  video post-processing filters
==========================================================

Day 8 of the 14-day video stack sprint.  This package contains
post-processing filters that run AFTER per-shot backends (Days 2-7)
and BEFORE the HyworldRenderer final mux.

Current members:
    * ``vhs`` -- VHS aesthetic filter (scanlines, chroma bleed,
      RGB shift, vignette, grain) via ffmpeg filter_complex.

The renderer's existing ``crt_postfx`` switch is the UI hook; this
package provides the actual filter pipeline behind that switch.

Contract with the renderer:
    * Transforms per-shot video clips in place at
      ``io/hyworld_out/<job_id>/<shot_id>/render.mp4``.
    * Does NOT touch audio streams (C7 byte-identical guarantee).
    * Auto-skips composite.png stills (renderer handles still-to-clip
      conversion downstream).
    * Stub mode (OTR_VHS_STUB=1 / ffmpeg missing / no input) is a
      byte-identical passthrough copy.

No torch / diffusers imports.  Safe to import from unit tests.
"""

from __future__ import annotations

__all__ = ["vhs"]
