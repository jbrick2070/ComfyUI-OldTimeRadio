# LTX upgrade wiring plan -- r1-HARDENED (Claude judge: anchor + Codex + AntiGravity, all code-grounded)

Panel this round: Codex (gpt-5.5/high, read-only) + AntiGravity (gemini-3.5-pro, file-handoff) +
Claude anchor/judge. Every accepted claim VERIFIED against the real files.

## What r1 changed (verified findings)
- **SCALER is NOT a one-line `_seg_vf` edit (Codex + AntiGravity, CONFIRMED).** The composite has
  THREE scale paths: `_seg_vf` (real clips AND the procgen floor; otr_silent_composite.py:319 ->
  `_encode_segment`:348) and `_encode_segment_from_dir`'s `bg_filter`/`fg_filter` (3D character
  directory-clips; :428-444) -- the latter has NO lanczos/unsharp. So editing only `_seg_vf`
  would (a) over-sharpen the procgen FLOOR and (b) leave CHARACTER clips blurrier than the rest.
  RESOLUTION (judge): a shared `_scale_filter(w,h,*,sharpen)` helper applied CONSISTENTLY to the
  real-video scale points (LTX clip via `_seg_vf` + character via `_encode_segment_from_dir`
  bg/fg), with `sharpen=False` for generated black/floor-only segments. Exact floor gating ->
  r2/r3. This supersedes "edit `_seg_vf`, append unsharp."
- **VRAM verify used the wrong ceiling + misses the peak (Codex, CONFIRMED).** The active runtime
  guard for ltx_av is `_MC.dynamic_vram_ceiling_mb()` / `_MC.assert_vram_within_ceiling()`
  (eng_ltx_av.py:356/633), NOT the static `wrapper_bridge.py:37`. And the assert is INSTANTANEOUS
  after run_graph+reclaim, so it misses the in-render peak -- use `motion_common.VramPeakProbe`
  (motion_common.py:242-295) to sample NVML ACROSS run_graph. Verify item rewritten accordingly.
- **Make decode + unsharp ENV-OVERRIDABLE (AntiGravity, matches the OTR pattern).** Hardcoding
  `temporal_size:128/overlap:32` forces a source edit to try whole-clip. Define
  `OTR_LTX_AV_DECODE_TEMPORAL_SIZE` (128) + `_OVERLAP` (32) + `OTR_COMPOSITE_UNSHARP_AMOUNT` (0.4)
  module-level and wire them in. Whole-clip becomes a config flip, not a code edit.
- **RECIPE scope (Codex, CONFIRMED).** Production default auto-resolves to sharp_lora
  (eng_ltx_av.py:240-294 -> dev Q3 -> RECIPE_SHARP_LORA); the bakeoff measured distilled_native.
  The decode change IS recipe-agnostic (shared dict after the recipe branch), but the pre-ship
  smoke MUST record `recipe=...` actually run (no-env default = sharp_lora) + its peak.
- **CUT the companion-drift manifest hardening (BOTH agents converged).** It couples tests to
  env-specific model filenames; it hardens the bakeoff, not the production path.
- **2-of-3, stated honestly (Claude anchor).** This upgrade ships SEAM (decode) + SOFTNESS
  (scaler); the init-hold STUTTER is NOT addressed (the bakeoff's freezedetect read 0 at baseline,
  never reproduced it). i2v 0.62 stays an open eyeball item.

## The locked decision (unchanged where it survived grounding)
- DECODE = temporal_size 128 / overlap 32 (env-overridable; whole-clip 4096/8 via env). 128/32
  seam imperceptible (0.57x local median), ~228 MB headroom vs whole-clip 27-162 MB.
- SCALER = lanczos + unsharp amount 0.4 (env), applied via a shared helper at the real-video
  scale points (NOT blanket `_seg_vf`).
- CANVAS = 512x288. No canonical-workflow-JSON edit (engine constants, not widgets). Don't touch
  eng_humo.py / eng_wan_ti2v.py. prod/main + tags GATED.

## Open for r2/r3 (coding/wiring)
- Exact shape of the shared `_scale_filter` helper + which segment kinds get `sharpen=True`
  (real clips yes; generated black/floor -- decide).
- Wiring the env constants + the VramPeakProbe smoke + the recipe-record into the verify steps.

## Pre-ship (revised)
1. ffmpeg supports `unsharp` (preflight/test). 2. `_scale_filter` string contract test (LTX +
character paths sharpened; black/floor per decision). 3. built graph decode == 128/32 (+ env
override test). 4. selective box reset (CLAUDE.md S4) BEFORE the smoke. 5. one canonical
`ltx_audio_in` smoke: NVML peak sampled ACROSS run_graph < `dynamic_vram_ceiling_mb()`, record the
recipe run, asset exists. 6. suite green vs the 5 pre-existing 267a53e fails; Bug Bible 16/7/3;
AST/no-BOM. 7. commit+push v2.0-alpha; HEAD==origin. Operator may bump
`OTR_LTX_AV_RESERVE_VRAM_GB` 4.0->4.5 on spill.
