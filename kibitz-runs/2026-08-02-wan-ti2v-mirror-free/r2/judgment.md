# r2 judgment -- wan_ti2v mirror-free recipe (driver grounding)

Panel: codex `gpt-5.6-sol` (high). Antigravity lane pending at time of writing.
Operator ruling being served: "kibitz for a solution, a proven workflow recipe,
and then measure the coverage."

## THE ONE SCOPING ERROR, and it changes the ordering argument

**codex MUST-FIX 1 / CUT 1 -- "cost-correction-only cannot work, because the
profile pins 17."** The CITED FACTS ARE CORRECT and I verified both:

    config/profiles/otr_8gb_wan.json   video.max_render_frames = 17
                                       launch.env OTR_WAN_TI2V_MAX_FRAMES = 17

**But that is not the profile that failed.** The 268-minute leg ran
`otr_g4_wan_ti2v`, which pins NOTHING:

    config/profiles/otr_g4_wan_ti2v.json  video.max_render_frames  = None
                                          render.max_render_frames = None
                                          launch.env WAN cap       = None

and the failure was `static frame budget 173 ... affordable 24`, with no 17-cap
anywhere in it. So there are TWO wan_ti2v configurations and they have different
binding constraints:

| profile | binding constraint | cost-correction alone sufficient? |
|---|---|---|
| `otr_8gb_wan` | the 17-frame ceiling, clamped BEFORE the predictor | NO -- codex is right |
| `otr_g4_wan_ti2v` | the cost model only | POSSIBLY -- codex's cut does not apply |

CONSEQUENCE: codex's "CUT the cost-correction-only branch" is ACCEPTED for the
8 GB tier and REJECTED as stated for the campaign lane. Note also that
`otr_8gb_wan` is already independently recommended for retirement as the source
of the 17==17 landmine; retiring it would remove the conflict rather than
resolve it.

The ORDERING codex argues for -- measure first, then decide topology -- SURVIVES
this correction, because a measurement is required either way and it is exactly
what the operator ruled. Only the justification narrows.

## ACCEPTED, and directly actionable

**A1. The existing bench may not authorise a refit.** CONFIRMED by its own
contract: `scripts/run_video_arm_bakeoff.py` states its results may not refit
`FRAME_COST_MODEL` and that the estimator and adapter are bypassed. So a
measurement mode has to be added or the campaign contract amended -- the numbers
already on disk (6563/6531/6563 at 17/49/81) cannot legally become the new row.
This is the single most useful thing the panel produced.

**A2. The unit contract must be ONE quantity.** `motion_common` fits from
machine-wide absolute peak, `compute_real_frame_budget` compares against live
FREE VRAM, and the adapter adds hoist cost back to free. Fitting raw NVML peak
would mix desktop baseline, engine demand and hoisted residency. Define
"additional engine demand above quiescent baseline" and make the fit, the hoist
correction, the budget comparison and the CPU tests all use it.

**A3. Fit a conservative UPPER envelope, not a mean.** The predictor is the ONLY
enforcing guard; `VramPeakProbe` samples and never enforces. A mean fit converts
normal run-to-run variance into an unguarded CUDA OOM.

**A4. The allowlist alone does not remove the mirror.** `extend_frames_to_target`
is still reachable on the single-clip path. Deleting the allowlist entry is not
the same as deleting the mirror; both are required, plus
`extension_mode == "none"` and `native_frame_count == frame_count` as acceptance
invariants.

**A5. Store a legal 4n+1 rung**, not an arbitrary measured integer, so the
profile value and the stamped effective maximum cannot differ.

**A6. Seam measurement needs a predeclared threshold**, and `OTR_FEAR_CAPE` must
be OFF for it -- the default 4-segment behaviour deliberately inverts the final
handoff, which would be measured as a discontinuity.

**A7. Bench cannot ship behaviour.** Any coefficient the bench selects must be
re-proved through `workflows/otr_canonical.json` before it counts.

## DEFERRED TO THE WIRING REVIEW (agy + Sonnet 5, per operator sequencing)

codex MUST-FIX 5 rewrites three tests and MUST-FIX 7 asks whether the canonical
`max_render_frames` widget or profile application owns the measured value. Both
are wiring questions and the operator ordered the wiring review FIRST. Held.

## REJECTED

**Turning `VramPeakProbe` into a second admission controller.** codex cuts this
and is right; it is deliberately telemetry-only. One preflight authority.

**Adding a new ComfyUI node or node input.** codex cuts this and is right: the
seams already exist in profile, contract, adapter, planner and render loop.
