# LTX-AV bakeoff winner -- HARDENED wiring plan (r4 CONVERGED)

Panel: Codex r4 (local, this round) + AntiGravity/Gemini + Codex r1-r2 (operator-pasted) +
Claude code-grounded anchor/judge. VERDICT across all voices: **yes-with-fixes -> converged**.
The fixes are plan-clarity (lock values, scope, specify the filter), not new architecture.

## Judgment on Codex r4 (every claim grounded against the real files)
- MF1 decode contradiction ("64->32" vs "128/32") -- CONFIRMED (my plan text was garbled).
  LOCK: temporal_size **128**, temporal_overlap **32** (the L1d leg).
- MF2 distilled_native is not the code default (`eng_ltx_av._unet_name()` -> dev ->
  sharp_lora, eng_ltx_av.py:240-249) -- CONFIRMED + important SCOPE point. Resolution: the
  decode + scaler fixes are RECIPE-AGNOSTIC (the decode dict is shared in `_build_graph` after
  the recipe branch; the scaler is post-engine), so they ship independent of recipe. Making
  distilled_native the DEFAULT is a SEPARATE, already-deferred operator decision (tier/ship) --
  NOT this commit.
- MF3 companion-drift asserts under-specified + file not listed -- CONFIRMED. Re-scoped as an
  OPTIONAL harness hardening (it hardens the bakeoff, not the production path), with exact
  values if done.
- SF1 `_seg_vf` order ambiguous -- CONFIRMED. LOCK the exact string (below).
- SF2 "operator may raise unsharp" implies a widget that doesn't exist
  (OTR_SilentComposite.INPUT_TYPES has none, otr_silent_composite.py:628-663) -- CONFIRMED.
  Reworded as a future CODE tweak, not an operator control.
- SF3 evidence-path typo `ltqx` -> `ltxq_bakeoff_results.md` -- CONFIRMED, fixed.
- SF4 lock the open question -- ACCEPTED: 128/32 is the default; whole-clip is documented manual.

## LOCKED decision (the winner to wire)
- **DECODE = temporal_size 128, temporal_overlap 32** (spatial stays tile_size 512 / overlap 64).
  Seam imperceptible (p99 0.0321, jump 0.57x the local median), ~228 MB headroom (peak 14272 <
  14500). Whole-clip (4096/8) is a DOCUMENTED manual max-quality option only (true-zero seam but
  27-162 MB headroom + run-to-run variance ~135 MB) -- NOT the default.
- **SCALER (`otr_silent_composite._seg_vf`)** final filter order (unsharp right after the scale,
  before pad): `scale=W:H:force_original_aspect_ratio=decrease:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black,fps=F,tpad=stop_mode=clone:stop_duration=3600`.
  Unsharp AMOUNT 0.4 (+8.9%; the resampler choice is ~irrelevant, the unsharp is the sharpener).
  Raising it (0.5-0.8) is a future CODE tweak, not a runtime widget.
- **CANVAS = 512x288** (render_driver.py ~1116, unchanged).
- **RECIPE = out of scope** -- the fix is recipe-agnostic; distilled_native-as-default stays the
  separately-deferred operator decision.
- **No canonical-workflow-JSON edit** -- decode params + scaler are hardcoded engine constants,
  not node widgets (verify item 3).

## Code changes (one commit)
1. `nodes/_otr_video_engines/eng_ltx_av.py` ~556-559: decode dict ->
   `tile_size:512, overlap:64, temporal_size:128, temporal_overlap:32`.
2. `nodes/otr_silent_composite.py` `_seg_vf` ~319-325: scale gets `:flags=lanczos`, append
   `unsharp=5:5:0.4:5:5:0.0` immediately after the scale (before pad).
3. Tests: add a CPU graph-build assert (decode == 512/64/128/32) + a `_seg_vf` string assert.

## Pre-ship VERIFY-AT-BUILD (run before the commit)
1. `wrapper_bridge.py:37` still `VRAM_CEILING_MB = 14500`.
2. built `ltx_audio_in` graph decode == 512/64/128/32.
3. `workflows/otr_scifi_16gb_full.json` has NO temporal_size/temporal_overlap widget (engine
   constant, not wiring).
4. `_seg_vf` emits `flags=lanczos` + `unsharp=5:5:0.4:5:5:0.0` in the locked order.
5. ONE real canonical-workflow `ltx_audio_in` smoke under normal desktop load -> peak VRAM <
   14500 AND the output asset exists in the canonical tree (otr/episodes or obs).
6. Full regression suite green vs ONLY the 5 pre-existing 267a53e workflow-pin fails (zero new);
   Bug Bible 16/7/3; AST/no-BOM/no-0-byte; JSON round-trip; OTR_WorkflowValidator + link/widget
   audit.
7. Commit + push v2.0-alpha; HEAD == origin. Do NOT touch eng_humo.py / eng_wan_ti2v.py.
   prod/main + tags GATED.

## Optional (not blocking)
- Harden the bakeoff manifest with companion-drift asserts (Gemma encoder / projection / video
  VAE / audio VAE == the build_ltx_av_q_bakeoff_workflow.py:62-65 DEV values).
