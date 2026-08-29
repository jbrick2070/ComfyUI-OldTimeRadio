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

import os

from .eng_ghost_signal import GhostSignalEngine
from .registry import register

#: The official v3 module. Documented as the smoother one -- less "boiling" and
#: flicker than v2 -- which matters here because flicker on this project is a
#: recipe problem, never a prompt problem.
MM_V3_NAME = "v3_sd15_mm.ckpt"

#: Its real size, 1,673,262,583 bytes. The floor sits just under it, because a
#: floor inherited from a DIFFERENT artifact is a false accusation -- this lane's
#: first live leg died on exactly that, refused for being "truncated" while the
#: file was byte-perfect.
MM_V3_MIN_BYTES = 1_600_000_000

#: The official v2 module. Balanced, and the only one of the three with
#: MotionLoRA support (eight camera moves). Ghost wires no Motion LoRA today --
#: that is excluded by the coding plan -- so this is a future capability, not a
#: current one, and it is recorded rather than claimed.
MM_V2_NAME = "mm_sd_v15_v2.ckpt"

#: 1,817,888,431 bytes -- within 6 KB of mm-p_0.5, which is the clearest single
#: piece of evidence that mm-p_0.5 is a v2 derivative.
MM_V2_MIN_BYTES = 1_700_000_000


# UNREGISTERED 2026-08-23. The class survives because the haunted lane inherits
# it -- the v3 module, its byte floor and its recipe receipt are that lane's own
# machinery. The public id `animatediff15_v3_video` is tombstoned in
# RETIRED_ENGINE_IDS. Do not re-add @register to "fix" a missing engine.
class GhostSignalV3Engine(GhostSignalEngine):
    """The official-v3 base. UNREGISTERED -- inherited by the haunted lane."""

    name = "animatediff15_v3_video"

    motion_module_name = MM_V3_NAME
    motion_min_bytes = MM_V3_MIN_BYTES
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


#: The v3 DOMAIN ADAPTER -- a LoRA on the IMAGE model, not the motion module.
#: 102,134,097 bytes, and the smallest artifact in this lane by two orders of
#: magnitude.
ADAPTER_V3_NAME = "v3_sd15_adapter.ckpt"

#: G1.3: below its own artifact, and within 15% of it so a truncated fetch is
#: still caught. 95 MB against a real 97.4 MB.
ADAPTER_V3_MIN_BYTES = 95_000_000

#: THE DIAL'S DIRECTION, and it is the opposite of what "adapter" suggests.
#: Upstream trains this LoRA to ABSORB the video training set's own defects so
#: they can be taken back out at inference: remove it, or scale it down, to get
#: LESS of the dataset's grime. So 0.0 is the pristine image-model domain and
#: 1.0 is the full video-dataset domain -- which is exactly the haunted end.
#:
#: 1.0 is the adapter's own full scale and therefore the honest default for a
#: lane whose entire purpose is to be haunted. IT IS UNMEASURED: no strength
#: has been qualified by eye yet, which is what the environment override below
#: is for. Sweep it, look at the output, then freeze the number here.
ADAPTER_V3_STRENGTH = 1.0

#: The sweep knob, following the eng_fastwan_8gb recipe-override pattern.
ADAPTER_V3_STRENGTH_ENV = "OTR_GHOST_HAUNTED_LORA_STRENGTH"


@register
class GhostSignalV3HauntedEngine(GhostSignalV3Engine):
    """``animatediff15_v3_haunted_video`` -- v3 plus the removable adapter.

    A THIRD peer, additive exactly as v3 and v2 were. It inherits everything
    from the clean v3 lane and adds one thing: the domain adapter on the MODEL
    path. The clean v3 lane sits beside it as the reference and the golden lane
    is untouched, so a comparison between them is a one-variable comparison.

    WHY THE ADAPTER IS INTERESTING HERE. Ghost's worst tendency is lettering --
    SD1.5 volunteers text into anything resembling a sign, a poster or a radio
    dial, which is most of this show, and the lane fights that with
    negative-prompt tokens on every single beat. The adapter is a different
    mechanism aimed at the same class of defect: rather than arguing with the
    model at inference, upstream parks the training set's learned artifacts in
    a removable component. Turned DOWN it should mean less learned grime;
    turned UP, more of it, deliberately, as a look.

    IT LOADS ON THE STOCK NODE. The artifact carries 256 UNet attention tensors
    and ZERO text-encoder tensors -- read off the checkpoint's own key list,
    not its documentation -- which is why this is ``LoraLoaderModelOnly`` and
    why CLIP keeps its untouched path. The keys are the legacy diffusers
    attn-processor spelling, which ComfyUI maps natively, so there is no
    conversion step and no custom loader anywhere in this lane.

    WHAT IS NOT HERE YET: nothing schedules the strength across an episode. The
    dial is one frozen number per render. Driving it from the ledger's arc so
    the picture degrades as the story tightens needs a declared field on
    ``VideoRequest`` -- that model is ``extra="forbid"`` and its one open dict
    is documented "NEVER conditioning" -- and a schema change is a design item,
    not a mechanical one. Prove the dial does something first.
    """

    name = "animatediff15_v3_haunted_video"

    lora_name = ADAPTER_V3_NAME
    lora_min_bytes = ADAPTER_V3_MIN_BYTES
    recipe_receipt_id = "animatediff_sd15_v3_haunted_static16_512x288_v1"

    @property
    def lora_strength(self):
        """Frozen default, overridable for the strength sweep.

        A property rather than a plain constant because the point of the first
        renders is to try several values without editing code between them.
        An unreadable or absent value lands on the frozen default rather than
        on zero: zero would render the CLEAN lane while stamping a haunted
        receipt, which is the one outcome this lane may never produce.
        """
        raw = os.environ.get(ADAPTER_V3_STRENGTH_ENV)
        if raw is None or not str(raw).strip():
            return ADAPTER_V3_STRENGTH
        try:
            return float(raw)
        except (TypeError, ValueError):
            return ADAPTER_V3_STRENGTH


__all__ = ["GhostSignalV3Engine",
           "GhostSignalV3HauntedEngine",
           "MM_V3_NAME", "MM_V2_NAME",
           "MM_V3_MIN_BYTES", "MM_V2_MIN_BYTES",
           "ADAPTER_V3_NAME", "ADAPTER_V3_MIN_BYTES", "ADAPTER_V3_STRENGTH",
           "ADAPTER_V3_STRENGTH_ENV"]
