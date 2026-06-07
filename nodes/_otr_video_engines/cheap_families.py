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

    def render_clip(self, request, prepared):  # pragma: no cover - ffmpeg/interactive
        """Render ONE always-silent CanonicalClip. The ffmpeg/CPU implementation
        lands with the interactive episode wiring; until then the M1 episode uses
        the OTR_SignalLostVideo radio-floor base through OTR_SilentComposite."""
        raise NotImplementedError(
            f"{self.name}.render_clip is the CW-4 ffmpeg render slice (wired with "
            f"the interactive episode smoke); the platform selection/role logic is "
            f"CPU-tested without it"
        )

    def canonicalize(self, raw, request, profile):  # pragma: no cover
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


@register
class StationCardFamily(_CheapFamilyBase):
    name = "station_card"
    family = "static_image_gen"
    roles = ("announcer_visual", "background_abstract")
    default_roles = ("announcer_visual",)
    required_inputs = ("text_prompt",)


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


__all__ = [
    "AbstractFamily", "StillKenBurnsFamily", "StationCardFamily",
    "VisualizerFamily", "FluxStillFamily",
]
