"""The SCENE-REGIONS lab peers of the two AnimateDiff lanes.

``animatediff15_v3_haunted_regions_video`` and ``animatediff15_lightning_regions_video`` are the haunted
and lightning lanes with ONE addition: the objects the beat's own dialogue refers to each get a region of
the frame and their own text, on top of the unchanged global prompt. Nothing else moves: same
checkpoint, motion module, adapter, decoder, sampler cells, canvas, cadence, context, prompt profile and
frame contract, which is what makes each a Ghost sibling of its parent and lets a frozen bundle planned
on the parent replay on it (``otr_shot_lock._apply_replay_engine_override``).

WHY IT EXISTS (operator, 2026-10-27, relayed): what is on screen must come from the beat's dialogue, the
concrete objects, the setting and the situation; style is a light touch. The global prompt gives the
dialogue's noun one clause among several. A region gives it its own text and its own place in the
frame, which is the recipe-part-2 technique (regional text conditioning) in core ComfyUI nodes: no node
pack, no model, no download.

WHERE THE OBJECTS COME FROM (2026-10-28, operator: "the prompts need to talk about real objects ... it has to
adapt dialogue"): when ShotLock PLANS an episode on one of these lanes it makes one extra short call on the
writer it already has loaded, which picks the physical things each character line itself names
(``ghost_scene_object_author``; every phrase is a contiguous run of the line's own words, so the model can
only choose, never invent). The picks ride the frozen plan as ``scene_objects`` and replay on any sibling.
A beat with none, a plan frozen before the field, or a failed call falls back to the episode's key-object
pool, as the first cut of this lane did.

OPT-IN, DEFAULT OFF. Neither id is a default of any role and neither appears in a shipping profile; the
three AnimateDiff workflows keep their lanes and an operator picks one of these from the existing video
dropdown. With no object for a beat the encode graph is the parent's, node for node.

THE MECHANISM LIVES IN THE BASE (``GhostSignalEngine``): ``wants_scene_regions`` is False there, so
every existing lane renders as it always did. These subclasses only set it.

CUDA ONLY until proven. Nothing here has rendered on Apple silicon.
"""
from __future__ import annotations

from .eng_ghost_signal_lightning import GhostSignalLightningEngine
from .eng_ghost_signal_official import GhostSignalV3HauntedEngine
from .registry import register


@register
class GhostSignalV3HauntedRegionsEngine(GhostSignalV3HauntedEngine):
    """``animatediff15_v3_haunted_regions_video`` -- the haunted lane plus scene regions."""

    name = "animatediff15_v3_haunted_regions_video"
    recipe_receipt_id = "animatediff_sd15_v3_haunted_regions_static16_512x288_lab_v1"
    wants_scene_regions = True


@register
class GhostSignalLightningRegionsEngine(GhostSignalLightningEngine):
    """``animatediff15_lightning_regions_video`` -- the lightning lane plus scene regions."""

    name = "animatediff15_lightning_regions_video"
    recipe_receipt_id = (
        "animatediff_sd15_lightning8_ftmse_regions_static16_hold3_512x288_lab_v1")
    wants_scene_regions = True


__all__ = ["GhostSignalV3HauntedRegionsEngine", "GhostSignalLightningRegionsEngine"]
