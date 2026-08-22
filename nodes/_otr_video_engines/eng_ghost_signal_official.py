"""Ghost Signal PEER LANES on the OFFICIAL AnimateDiff motion modules.

ADDITIVE, NOT A REPLACEMENT (operator, 2026-08-22: *"we clone the current
ghost's lane so if it doesn't work we have our golden ghost untouched"*, and
*"a peer lane"*). ``animatediff15_video`` is unchanged, still default, still the
lane that rendered the published episode. These two sit BESIDE it in the
dropdown and are chosen the same way any other lane is.

WHY THEY EXIST -- TWO REASONS, AND THE FIRST IS A BLOCKER
---------------------------------------------------------
**1. LICENCE.** The golden lane's ``mm-p_0.5.pth`` comes from
``manshoety/beta_testing_models``, which publishes NO LICENCE GRANT for it. That
is why ``animatediff15_video`` declares ``commercial_clean = False``, and it is
the single thing standing between Ghost Signal and being submittable to a
ComfyUI community template pack. Both modules here are from the OFFICIAL
``guoyww/AnimateDiff`` release under **Apache-2.0**.

**2. IT WAS NEVER A CONSIDERED CHOICE.** The spec records ``mm-p-0.5`` as the
operator's OPENING SUGGESTION -- *"Operator's start: SD1.5 + the mm-p-0.5 motion
module"* -- and left the question explicitly open: *"whether to hold mm-p-0.5 or
let the panel argue AnimateLCM"*. Its own Phase-0 step 3 called for an inventory
of *"v2/v3, AnimateLCM, Lightning"* with the expected answer *"v3, then
AnimateLCM"*. That inventory was skipped when the lane was built. These lanes
are it, done properly: same episode, same seed, three modules, operator's eye.

The public id carries no ``low``/``high`` token (G7.4: those come from a
measurement receipt, never a guess) and the engine id was deliberately made
module-agnostic in the spec's NAME section so a swap could never make it lie.

WHAT IS DIFFERENT FROM THE GOLDEN LANE: the motion module and the recipe
receipt. NOTHING ELSE. Canvas, cadence, context, sampler, prompt profile,
delivery, still policy and every guard are inherited unchanged, which is what
makes a comparison between them mean anything.

AnimateLCM is deliberately absent and stays absent: its CFG 1-2 regime kills the
live negative, and the Ghost lettering defence needs real unconditional
conditioning (coding plan, section 4.1 item 5).
"""
from __future__ import annotations

from .eng_ghost_signal import GhostSignalEngine
from .registry import register

#: The official v3 module. Documented as the smoother one -- less "boiling" and
#: flicker than v2 -- which matters here because flicker on this project is a
#: recipe problem, never a prompt problem.
MM_V3_NAME = "v3_sd15_mm.ckpt"

#: The official v2 module. Balanced, and the only one of the three with
#: MotionLoRA support (eight camera moves). Ghost wires no Motion LoRA today --
#: that is excluded by the coding plan -- so this is a future capability, not a
#: current one, and it is recorded rather than claimed.
MM_V2_NAME = "mm_sd_v15_v2.ckpt"


@register
class GhostSignalV3Engine(GhostSignalEngine):
    """``animatediff15_v3_video`` -- Ghost Signal on the official v3 module."""

    name = "animatediff15_v3_video"

    motion_module_name = MM_V3_NAME
    recipe_receipt_id = "animatediff_sd15_v3_static16_512x288_v1"

    #: APACHE-2.0, AND THAT IS THE POINT OF THIS LANE. The code is Apache-2.0
    #: and the SD1.5 checkpoint is CreativeML Open RAIL-M (hosted by Comfy-Org
    #: themselves), so with a licensed motion module every artifact in the lane
    #: is redistributable. The golden lane cannot say that.
    #:
    #: Still declared conservatively rather than True: `commercial_clean` is a
    #: legal claim, RAIL-M carries use restrictions, and this build has not had
    #: that reviewed. Apache-2.0 removes the BLOCKER; it does not by itself make
    #: the lane commercially clean, and saying so would be exactly the kind of
    #: overclaim the admission rules exist to stop.
    commercial_clean = False


@register
class GhostSignalV2Engine(GhostSignalEngine):
    """``animatediff15_v2_video`` -- Ghost Signal on the official v2 module."""

    name = "animatediff15_v2_video"

    motion_module_name = MM_V2_NAME
    recipe_receipt_id = "animatediff_sd15_v2_static16_512x288_v1"

    #: See the v3 note above -- same reasoning, same conservative declaration.
    #:
    #: WORTH KNOWING WHEN READING A COMPARISON: this module is 1,817,888,431
    #: bytes and the golden lane's `mm-p_0.5.pth` is 1,817,894,327 -- 5,896
    #: bytes apart. `mm-p_0.5` is almost certainly a v2 DERIVATIVE, so expect
    #: this lane to look closer to the golden one than v3 does. If the operator
    #: likes the golden look and cannot ship it, THIS is the likeliest licensed
    #: substitute, and that is the practical reason it is here at all.
    commercial_clean = False


__all__ = ["GhostSignalV3Engine", "GhostSignalV2Engine",
           "MM_V3_NAME", "MM_V2_NAME"]
