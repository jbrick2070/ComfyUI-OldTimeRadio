"""Ghost Signal CADENCE peers -- the same picture, traversed at a calmer rate.

ADDITIVE. ``animatediff15_video`` is untouched and still renders the dailies;
these sit beside it and are chosen from the dropdown like any other lane.

WHY THEY EXIST. The operator, watching a published episode: *"the animation of
animatediff is heavily fast paced, you could probably bring it down to 5fps and
might even look better"*, and *"like stop action animation"*. A second,
independent look (Gemini, on two unrelated episodes) reached the same reading
without prompting -- "rapidly shifting". Two observers, one complaint.

WHAT IS ACTUALLY WRONG, and it is not the frame rate. Standard Static slides a
length-16 window with overlap 4 -- stride 12 -- so a beat traverses
``1 + ceil((U - 16) / 12)`` windows, each contributing its own motion and
pyramid-fused into its neighbour. A one-second beat gets ONE clean gesture
because ``use_on_equal_length=False`` runs the module directly at 16. A
twelve-second beat gets THIRTEEN fused gestures. Real episodes run 2444-2944
delivered frames across 8 beats -- about twelve seconds each -- so they sit at
the bottom of that table. Motion energy per beat is therefore an uncontrolled
function of how long the writer's sentence happened to be, which nobody chose.

Lowering the hold lowers U, which lowers the window count, which is the lever
that actually exists:

    hold-2 (today)  12s beat -> U=150 -> 13 windows   100% render cost
    hold-3           "       -> U=100 ->  8 windows    68%
    hold-5           "       -> U= 60 ->  5 windows    40%

THE AUDIO SYNC SURVIVES ALL OF IT, by construction. A beat's audio duration sets
``target_frame_count``; the hold only decides how many fresh frames fill that
fixed T. The gesture and the line still start and end together. What changes is
how much motion is traversed inside the window, and how finely -- so these lanes
read slower AND steppier, which is the trade the operator asked to see rather
than a defect to apologise for. For a 1930s radio drama, deliberate stop-action
may be the house style; that is an eye question and this is how it gets asked.

The docs/2026-08-22-GHOST-CADENCE-PROBLEM-STATEMENT.md carries the full
derivation, the levers not taken (context overlap, ffmpeg minterpolate) and the
things that remain unmeasured -- including that AnimateDiff SD1.5's native frame
rate is NOT documented upstream, so nothing here rests on the commonly repeated
8fps figure.
"""
from __future__ import annotations

from .eng_ghost_signal import GhostSignalEngine
from .registry import register

#: One-third energy. The conservative arm: is a modest cut already enough,
#: before committing to the stepped look of hold-5.
HOLD_THREE = 3

#: The operator's own suggestion -- "bring it down to 5fps". 5 fresh frames per
#: second, and 40% of a normal leg's render cost.
HOLD_FIVE = 5


@register
class GhostSignalHold3Engine(GhostSignalEngine):
    """``animatediff15_h3_video`` -- Ghost Signal at hold-3 (8.33 fresh fps)."""

    name = "animatediff15_h3_video"
    hold_factor = HOLD_THREE
    recipe_receipt_id = "animatediff_sd15_mmp05_static16_hold3_512x288_v1"


@register
class GhostSignalHold5Engine(GhostSignalEngine):
    """``animatediff15_h5_video`` -- Ghost Signal at hold-5 (5 fresh fps).

    A SHORT BEAT BEHAVES DIFFERENTLY HERE and the test episode should include
    one. At hold-5 a one-second beat asks for U=5, below the 16-frame source
    floor, so it still generates a full window and shows only its first third --
    a deliberately partial gesture. That is worth seeing before this becomes a
    house style.
    """

    name = "animatediff15_h5_video"
    hold_factor = HOLD_FIVE
    recipe_receipt_id = "animatediff_sd15_mmp05_static16_hold5_512x288_v1"


__all__ = ["GhostSignalHold3Engine", "GhostSignalHold5Engine",
           "HOLD_THREE", "HOLD_FIVE"]
