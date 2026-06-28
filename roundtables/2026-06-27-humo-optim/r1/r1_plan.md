# HuMo quality bakeoff -- r1-HARDENED plan (Codex + AntiGravity CONVERGED; Claude judge)

The open-ended review CORRECTED the brief's core premise. All accepted claims verified vs the code.

## PREMISE CORRECTION (load-bearing, VERIFIED by Claude)
There is NO silent 14B->1.7B downgrade. `render_shot()` (render_driver.py:1468-1495): "NO FALLBACKS
(operator 2026-06-16) ... a HARD render failure RAISES RenderError LOUD ... the degrade chain is
gone." The `fallback_engine="humo_1.7B"` in eng_humo.py:106 is VESTIGIAL. So HuMo either renders 14B
or FAILS LOUD. The operator's "lost quality" is therefore NOT a quiet tier drop -- it is either (a)
the 14B OUTPUT quality (the 6-step lightx2v distill, or the de-blue cfg), or (b) 14B failing/raising
under VRAM pressure (no episode). The bakeoff is reframed accordingly.

## GOAL (corrected)
(1) Measure 14B output QUALITY across the few settings that matter; (2) prove 14B FITS <=14.5 GB
under real production pressure so it doesn't RAISE; (3) verify the RIGHT unet/tier actually loads.

## PHASE A -- isolated quality sweep (clean boot per leg; fixed still+audio+seed)
Mirror scripts/run_ltx_av_q_bakeoff.py. Fixed assets = the LTX pair `c02_466a19906ccb.png` +
`c02_b002_line.wav` (visible mouth + plosives/sibilants; one short lip-sync clip, fixed frames/seed).
FAIL-LOUD per-leg MANIFEST (the LTX #1 risk -- measuring the wrong graph): record + assert the
RESOLVED unet / lora / steps / cfg / shift / tier AND the tier/UNET that ACTUALLY loaded; abort if
it isn't the intended one. Legs (one lever per leg, staged, carry winners):
- HEADLINE 3-way: (a) 14B + lightx2v distill @ 6 steps / cfg 1.0 (the fast default); (b) 14B
  NO-LoRA (`OTR_HUMO_LORA_NAME=none`) @ ~25 steps / cfg ~5 (max-quality); (c) 1.7B (control only).
  This answers "is the 6-step distill the perceived quality loss, and what does no-LoRA cost?"
- CFG sweep ONLY on the no-LoRA legs (distill is trained for cfg 1.0; higher cfg on a distill leg
  => blue saturation). 
- SHIFT: hardcoded 8.0 (eng_humo.py:271) -> add an `OTR_HUMO_SHIFT` env knob (small engine change,
  wired into the workflow JSON SAME change per CLAUDE.md S0) OR have the runner mutate the API-graph
  shift directly (cf. the existing scripts/_otr_humo_shift_sweep.py). Decide one; manifest asserts it.
- RESOLUTION: NATIVE only -- portrait 480x832 (default) + optional wide 832x480
  (HuMo14BLandscapeEngine, eng_humo.py:541-558). No arbitrary sizes (off-res => interpolation blur).
- HOLD FIXED: still+audio+seed, frames. Also classify ALL knobs (OTR_HUMO_STEPS/CFG/UNET_NAME/
  LORA_NAME/WIDTH/HEIGHT/SHIFT/NEGATIVE) as production-env vs bakeoff-only-API-mutation vs not-allowed.

## PHASE B -- production-pressure sentinel (the REAL concern; NO reboot)
A clean per-leg boot ERASES the cross-engine residency that is the suspected failure. So a SECOND
phase loads the AV stack (LTX-AV + Whisper) IMMEDIATELY before the HuMo leg in ONE resident session
(no reboot) and asks: does 14B still fit <=14.5 GB, or RAISE? Test pre-emptive `--reserve-vram`
protection (post-decode eviction already exists at eng_humo.py:361; the open lever is a STARTUP
reserve so 14B never spills under the stack). This is what actually answers "HuMo under AV pressure."

## METRICS
VRAM peak + s/it (feasibility) + side-by-side clips to `otr/episodes/_bakeoff_humo/<leg>.mp4` for the
OPERATOR'S EYEBALL (primary). Objective proxies ONLY IF the GPU host has the libs (face-detect
confidence, mouth-landmark motion vs audio energy, lip-area SSIM, blue-cast color delta vs the source
still) -- VERIFY OpenCV / a face-landmark stack is installed before promising these; else eyeball-only.

## CUT (both agents) / OPTIONAL
CUT: NEGATIVE-prompt sweep (subjective, secondary); broad FRAMES sweep (33-177 already clamped at
eng_humo.py:53-54 -- one short clip + one max sentinel only). OPTIONAL: `--dry-validate` manifest-only
preflight (the LTX rail). Invariants: single resident <=14.5 GB; selective box reset per leg; 100%
local; LOUD; UTF-8 no BOM; SFW; don't touch the audio spine; quality = the operator's eyeball.

## Open for r2/r3/r4
The SHIFT knob (env vs API-mutate) decision + its workflow-JSON wiring; the exact Phase-B sequence
(which AV models, in what order); the objective-proxy library availability; the staged carry-forward.
