"""Cheap radio-floor video families -- the no-heavy-engine adapters (A-S3 / CW-4).

The first concrete video engines registered in the platform: cheap, CPU/ffmpeg
families that need NO heavy model, so a watchable episode (M1) renders before any
GPU engine exists. Each registers exactly like a future heavy engine (HuMo / LTX
/ Wan) -- model-agnostic, selected per role; no model is "primary".

Families (schemas.FAMILIES): ``abstract`` (procedural pattern), ``static_motion``
(still_kenburns -- a still with a slow pan), ``static_image_gen`` (station_card /
flux_still). Each produces an ALWAYS-SILENT ``CanonicalClip`` (``has_audio`` is
always False -- audio is added ONLY by ``OTR_MasterAudioMux``, V-1).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary. ffmpeg / PIL / numpy / torch are imported LAZILY inside
``render_clip`` (the render slice wires up with the interactive episode smoke;
the platform's selection / role-filter / usability logic is fully CPU-tested
without rendering).
"""
from __future__ import annotations

import logging

from .registry import register

log = logging.getLogger("OTR.video.cheap_families")


class _CheapFamilyBase:
    """Shared shell for a cheap, no-heavy-engine video family. Render lifecycle
    is the ffmpeg/CPU slice (lazy); the registry only reads the core metadata."""

    name = "cheap"
    roles: tuple = ()
    default_roles: tuple = ()
    commercial_clean = True
    requires_flag = None            # cheap families are always available (no opt-in)
    family = "abstract"
    required_inputs: tuple = ()
    engine_version = "1"
    #: True when this family animates a provided still (asset_refs.init_image /
    #: still) with a Ken Burns pan; False families always synthesize a procedural
    #: floor. Either way render_clip ALWAYS produces a valid silent clip.
    uses_still = False

    def load(self) -> None:  # cheap families hold no resident weights
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """Cheap families are usable wherever ffmpeg exists; the real ffmpeg
        check runs in render_clip. Returns the validated name (no heavy dep)."""
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover
        return {"engine_id": self.name}

    # ---- canvas / input resolution (pure; CPU-testable) ----
    @staticmethod
    def _get(request):
        return request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))

    def _canvas_dims(self, request):
        """(width, height, fps) from the request canvas, with platform defaults
        (832x480 @ 25) when a field is absent. The cheap floor never stretches:
        a still is scaled to cover with one uniform scale (bridge), a synth source
        fills the canvas exactly."""
        get = self._get(request)
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        w = int(c_get("w", 0) or 0) or 832
        h = int(c_get("h", 0) or 0) or 480
        fps = int(c_get("fps", 0) or 0) or 25
        return w, h, fps

    def _frame_count(self, request, fps):
        """The clip length in frames (timing.target_frame_count), defaulting to one
        second (``fps`` frames) when unspecified so the floor always renders."""
        get = self._get(request)
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        n = int(t_get("target_frame_count", 0) or 0)
        return n if n > 0 else int(fps)

    def _still_path(self, request):
        """A provided still/portrait path from asset_refs (or "" when none)."""
        get = self._get(request)
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            for key in ("still", "init_image", "image"):
                v = assets.get(key)
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, dict) and v.get("path"):
                    return v["path"]
        return ""

    def _lavfi_source(self, w, h, fps):
        """The libavfilter source for this family's synthesized floor (a dark
        radio-slate field by default; a family may override for its own look)."""
        return "color=c=0x0A0E14:s=%dx%d:r=%d" % (int(w), int(h), int(fps))

    def render_clip(self, request, prepared=None):
        """Render ONE always-silent CanonicalClip via ffmpeg (the CPU radio floor).

        A provided still (asset_refs) is animated with a gentle Ken Burns pan when
        ``uses_still``; otherwise the family's libavfilter source is synthesized.
        Output is the platform's silent bt709 / yuv420p contract (V-1: only
        OTR_MasterAudioMux ever adds audio). This family's canonicalize() is
        identity, so the canonical clip is returned here directly. With valid
        inputs it ALWAYS succeeds -- the fallback-chain terminus the A-S6 chain
        humo -> latentsync -> still_kenburns converges on."""
        from . import wrapper_bridge as _wb       # lazy import: cold-import clean
        import os
        import tempfile
        w, h, fps = self._canvas_dims(request)
        n = self._frame_count(request, fps)
        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_floor_%s_" % self.name)
        still = self._still_path(request) if self.uses_still else ""
        if still and os.path.exists(still):
            cmd = _wb.ffmpeg_still_motion_cmd(still, out_path, w, h, fps, n)
        else:
            cmd = _wb.ffmpeg_lavfi_floor_cmd(
                out_path, w, h, fps, n, source=self._lavfi_source(w, h, fps))
        _wb.run_ffmpeg(cmd)
        return self._floor_clip(request, out_path, fps, n)

    def _floor_clip(self, request, out_path, fps, n):
        """Pure: shape the rendered floor mp4 into the silent CanonicalClip dict
        (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = self._get(request)
        return {
            "clip_id": get("shot_id") or get("request_id") or ("%s_clip" % self.name),
            "type": "video", "path": out_path,
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(fps), "frame_count": int(n),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }

    def canonicalize(self, raw, request, profile):
        """Identity: render_clip already emits the canonical silent clip dict."""
        return raw

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


@register
class AbstractFamily(_CheapFamilyBase):
    name = "abstract"
    family = "abstract"
    roles = ("background_abstract", "music_visual")
    default_roles = ("background_abstract",)
    required_inputs = ()            # procedural -- needs nothing


@register
class StillKenBurnsFamily(_CheapFamilyBase):
    name = "still_kenburns"
    family = "static_motion"
    roles = ("scene_broll", "background_abstract", "announcer_visual")
    default_roles = ("scene_broll",)
    required_inputs = ("text_prompt",)
    uses_still = True               # Ken Burns pan over a provided still when present


@register
class StationCardFamily(_CheapFamilyBase):
    name = "station_card"
    family = "static_image_gen"
    roles = ("announcer_visual", "background_abstract")
    default_roles = ("announcer_visual",)
    required_inputs = ("text_prompt",)
    uses_still = True               # show a provided card still when present


@register
class VisualizerFamily(_CheapFamilyBase):
    name = "visualizer"
    family = "abstract"
    roles = ("music_visual", "background_abstract")
    default_roles = ("music_visual",)
    required_inputs = ()            # audio-reactive procedural


@register
class FluxStillFamily(_CheapFamilyBase):
    name = "flux_still"
    family = "static_image_gen"
    roles = ("announcer_visual", "scene_broll", "character_video")
    default_roles = ()              # selectable peer; not the in-stack default
    required_inputs = ("text_prompt",)
    commercial_clean = False        # Flux.1-dev = BFL non-commercial (migration)
    uses_still = True               # animate a provided flux still when present


__all__ = [
    "AbstractFamily", "StillKenBurnsFamily", "StationCardFamily",
    "VisualizerFamily", "FluxStillFamily",
]
