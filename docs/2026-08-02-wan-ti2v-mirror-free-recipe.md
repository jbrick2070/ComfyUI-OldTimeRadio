# wan_ti2v must cover a beat with REAL clips and no mirror. Find the recipe.

**Operator ruling 2026-08-02:** kibitz for the solution -- a PROVEN WORKFLOW
RECIPE -- and then measure the coverage. Panel before code; measurement before
any constant. Do not guess a VRAM number.

## THE HARD FACT, from a live 268-minute leg

    MotionBudgetError: engine wan_ti2v: static frame budget 173 (snapped 173)
    exceeds the cost-model's affordable 24 frames (free=13481 MB, margin=0.85).
    NO silent resize -- lower the frame_count widget, free VRAM, or pick a
    lighter engine.

Not an OOM, not a leak: free VRAM was 13.5 GB and post-render residency was
stable at 4160-4485 MB across eight renders that night.

    FRAME_COST_MODEL["wan_ti2v"] = (7000.0, 185.0)     # overhead MB, per-frame MB
    budget     = 13481 * 0.85          = 11459 MB
    affordable = (11459 - 7000) / 185  = 24 frames

**So wan_ti2v is priced at ~24 frames per render on this box, and the
adapter-side ping-pong is what has always made it appear to deliver 173-frame
beats.** The operator's standing ruling forbids the mirror ("no mirror, it's
1/1, one and done, no re-using video"), and without it the 25-177 frame band
cannot be covered at all.

## WHY THAT BAND IS A TRAP (the mechanism, grounded)

* Beat ABOVE the contract max (177): already splits. A planned segment goes
  through `_planned_length`, which DELIBERATELY does not consult the VRAM
  predictor. This is why `fastwan_8gb` passes -- its beats are ~245 frames.
* Beat BELOW ~25: affordable outright, single clip, fine.
* Beat BETWEEN (173 here): too small to trigger a split, too big for the
  predictor. Single-clip path, predictor consulted, REFUSED.

## WHAT THE CODEBASE DELIBERATELY PROTECTS

`tests/test_multiclip_effective_contract.py` pins all three:

    assert "wan_ti2v" not in fc.PLANNING_CAP_ENGINES
    test_a_pinned_ceiling_does_not_move_wan_ti2v_coverage_topology
    test_wan_ti2v_keeps_its_adapter_side_ping_pong

and `frame_contract.py:289` says adding an id to `PLANNING_CAP_ENGINES` is "a
per-engine decision with a LIVE PROOF attached, never a convenience".

So this is a DESIGN CHANGE the operator has ruled for, not a defect fix. The
panel's job is the recipe that makes it correct and provable.

## THE EVIDENCE ALREADY ON DISK

* `fastwan_8gb` shares base weights with the incumbent -- the adapter comment
  says "base weights bit-identical to the incumbent's, which is why the two
  measure the same peak" -- and its four-arm bench measured VRAM **FLAT**:
  6563.1 / 6531.1 / 6563.1 MiB at 17 / 49 / 81 frames.
* That flat curve CONTRADICTS `(7000, 185)`, which predicts 7000 + 81*185 =
  22 GB for 81 frames where the bench measured 6.5 GB.
* Live wan_ti2v peaks the same night: 10648-12128 MB over eight renders, post
  4160-4485 MB. Peaks, not a per-frame curve.
* `wan_ti2v`'s RECIPE IS FROZEN and does not move (CLAUDE.md). Only coverage
  topology and the cost row are in scope.

## WHAT THE PANEL MUST DELIVER

1. **The recipe.** Coverage-plan `wan_ti2v` (add to `PLANNING_CAP_ENGINES` with
   a measured ceiling), or correct the cost model so the predictor affords whole
   beats, or both -- and say WHICH ORDER, since a correct cost row may make the
   topology change unnecessary or may change the right ceiling.
2. **The measurement protocol**, exactly: which frame counts, which canvas,
   what to read (NVML peak? `torch.cuda.max_memory_allocated`? the existing
   `VramPeakProbe`, which is telemetry-only and enforces nothing?), how many
   repeats, and how to separate per-frame cost from fixed overhead so the
   fitted `(overhead, per_frame)` is honest rather than a two-point guess.
3. **The seam risk.** A 173-frame beat split into ~81-frame segments under
   `strict_first_frame` chaining: does the existing terminal-frame handoff hold
   at this engine's canvas, and what is the visible cost of 2-3 joins where
   there used to be one mirrored clip?
4. **The three tests.** They encode the OLD ruling. Say precisely what each
   should assert under the new one -- especially
   `test_wan_ti2v_keeps_its_adapter_side_ping_pong`, whose whole premise is
   inverted.
5. **What could make this WORSE.** The predictor is the only guard on this path
   (`VramPeakProbe` enforces nothing -- "the peak is sampled + logged, never
   enforced"), so any change that lets more frames through converts a clean
   preflight refusal into an unguarded CUDA OOM. Name the failure mode.

## CONSTRAINTS

Every second of audio gets video; no mirror / ping-pong / re-used frames; fail
loud, never silently degrade; `wan_ti2v`'s sampler recipe frozen; the only
workflow JSON is `workflows/otr_canonical.json`; 16 GB RTX 5080 laptop,
14.5 GB real-world ceiling; 100% local. **Do not launch renders or boot a
server** -- read the files and reason.
