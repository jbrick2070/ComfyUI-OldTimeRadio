# LTX-AV bakeoff winner -- production WIRING PLAN (r4 convergence target)

Repo: this directory. Branch v2.0-alpha, HEAD 08540ceb. Hard ceiling 14500 MB for the
single resident heavy engine (`wrapper_bridge.VRAM_CEILING_MB`). 100% local.

## Goal
Wire the LTX-AV quality-bakeoff winner into production. The isolated bakeoff
(scripts/run_ltx_av_q_bakeoff.py + scripts/build_ltx_av_q_bakeoff_workflow.py +
scripts/otr_ltx_av_q_bakeoff_distilled_native.json; results in
otr/episodes/_bakeoff_ltxq/ltqx_bakeoff_results.md) measured the distilled_native
ltx_audio_in path that the overnight soak flagged as soft + with a temporal "flash" + an
init-hold micro-stutter. Two prior review rounds (AntiGravity/Gemini + Codex, both
GO-WITH-FIXES) and a Claude code-grounded grounding pass have already hardened this.

## Validated findings (objective, from the bakeoff)
- TEMPORAL SEAM ("flash"): caused by VAE temporal-tiling. Whole-clip decode (VAEDecodeTiled
  temporal_size 4096 / overlap 8) eliminates it (seam p99 0.2353 -> 0.0). Best TILED decode
  = 128/32 (seam p99 0.0321, ratio 0.57 = jump below the frame-to-frame noise floor =
  imperceptible).
- SOFTNESS: the composite scaler is the lever, and it is the UNSHARP mask, not the resampler.
  Measured at the common 1472x832: bilinear 23.69; lanczos 23.99 (+1.3%); spline 23.74
  (+0.2%); lanczos+unsharp(0.4) 25.79 (+8.9%); lanczos+unsharp(0.8) 27.17 (+14.7%). So the
  resampler choice is ~irrelevant; the unsharp AMOUNT is the real lever.
- CANVAS: 512x288 is the production clamp (render_driver.py ~1116). 640x384 fit (14476 MB,
  8.98 s/it) but introduced freezes=5; 704x384 needed reserve 5 to fit. Bigger native canvas
  is NOT worth it (output is upscaled to 1472x832 either way).
- STUTTER (init-hold): NOT validated by this bakeoff -- freezedetect read 0 freezes at the
  512 baseline, so the fixture never reproduced it. i2v 0.62-vs-0.75 + native-vs-respaced
  sigmas did not separate on any objective metric. Treat the stutter as an OPEN eyeball item,
  not "fixed."

## The decision (what to wire)
- DECODE = **128/32 tiled** as the production default. Rationale: seam is imperceptible
  (ratio 0.57) and it leaves ~228 MB headroom (peak 14272), vs whole-clip's 27-162 MB (the
  same whole-clip config ran 14338 and 14473 on two legs -- run-to-run variance ~135 MB
  alone, which on a desktop with apps eating 3-5 GB can cross 14500). Whole-clip (4096/8) is
  the documented "max-quality when the box is quiet" alternative, gated on a production smoke.
- SCALER = lanczos + **unsharp amount 0.4** default (+8.9%); operator may raise to 0.5-0.6 or
  0.8 after eyeballing rain/glass/face for halos. Resampler stays whatever is simplest.
- CANVAS = stay 512x288.

## Code changes (one commit)
1. `nodes/_otr_video_engines/eng_ltx_av.py` ~line 556-559: change the decode dict
   `temporal_size` 64 -> 32, `temporal_overlap` 8 -> 32 (i.e. 128/32 -> actually
   temporal_size 128, temporal_overlap 32). [VERIFY exact param names/values against the file.]
2. `nodes/otr_silent_composite.py` `_seg_vf` ~line 319-325: add `:flags=lanczos` to the scale
   and append `,unsharp=5:5:0.4:5:5:0.0`.
No canonical-workflow-JSON edit: the decode params and the scaler are hardcoded Python engine
constants, NOT node widgets in workflows/otr_scifi_16gb_full.json (confirmed across both prior
reviewers). CLAUDE.md S0 governs node/wiring/widget changes; these are neither.

## Pre-ship validation (before the production commit)
1. Add companion-drift asserts to the bakeoff manifest (fail loud if the DEV video VAE / DEV
   audio VAE / DEV projection / Gemma encoder ever swap).
2. Run ONE real canonical-workflow `ltx_audio_in` smoke under normal desktop load; confirm
   peak VRAM < 14500. If whole-clip is chosen and the smoke runs hot, fall back to 128/32.
3. Full regression suite must be green vs the 5 pre-existing 267a53e workflow-pin fails
   (test_capability_profiles / test_workflow_apply x2 / test_workflow_live /
   test_full_workflow_v2) -- zero NEW fails. Bug Bible 16/7/3. AST/no-BOM. B7 sweep +
   OTR_WorkflowValidator.
4. Commit + push v2.0-alpha. Do NOT touch eng_humo.py / eng_wan_ti2v.py. prod/main + tags
   GATED.

## Open question for this round
Is 128/32 the right production default over whole-clip given the 14500 ceiling and desktop
contention, or should whole-clip ship gated behind the live smoke? Any residual build-blocker
or fix-introduced regression in the change above?
