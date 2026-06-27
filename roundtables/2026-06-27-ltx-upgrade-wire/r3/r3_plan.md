# LTX upgrade wiring plan -- r3-HARDENED (wiring/sequencing; Codex + AntiGravity CONVERGED)

Builds on r2_plan.md. r3 fixed the wiring contracts; agy verdict = "build-ready-with-fixes".
All claims code-grounded + accepted by the judge.

## Scaler helper -- final signature + per-path sharpen/pad/bg rules
`_scale_filter(w, h, fps, *, sharpen, pad=True)` (fps IS needed -- the chain appends fps, agy
catch). Returns `scale=w:h:force_original_aspect_ratio=decrease[:flags=lanczos]
[,unsharp=5:5:<amt>:5:5:0.0][,pad=w:h:(ow-iw)/2:(oh-ih)/2:color=black],fps=fps`; amt =
`OTR_COMPOSITE_UNSHARP_AMOUNT` (0.4). PRESERVE `_seg_vf`'s order: trim -> scale/pad/unsharp/fps ->
tpad (moving tpad before fps changes frame budgeting -- Codex).
Per call site (all grounded):
- `_seg_vf`/`_encode_segment` (:319/:336/:579-587): `sharpen=True` for `kind=="clip"`, `False`
  for procgen floor. `_encode_black` UNCHANGED.
- `_encode_segment_from_dir` (:403-461): FOREGROUND fg (:444, RGBA) -> `sharpen=True, pad=False`
  (pad=color=black destroys the straight-alpha edges -> opaque borders block the plate). BACKGROUND
  (:570-578): `bg_is_still=True` (real still plate) -> `sharpen=True`; `base_video_path` procgen
  floor (workflow link 246) and black -> `sharpen=False`. (Resolves the open still-plate decision.)
- `normalize_to_silent_canonical` (:72-105): `_scale_filter(..., sharpen=False)` (floor-like).

## Decode env (validated, runtime-read)
`OTR_LTX_AV_DECODE_TEMPORAL_SIZE` (128) + `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` (32), read INSIDE
`_build_graph` (testable via monkeypatch w/o reload). Parse with try/except ValueError -> default
128/32, and fail-loud/clamp if overlap>=size or <=0 BEFORE VAEDecodeTiled. Spatial tile 512 /
overlap 64 fixed.

## VRAM verify (real peak, leak-safe, cleanup-safe)
In `render_clip` (eng_ltx_av.py:621-633): `probe = _MC.VramPeakProbe().start()`; inside `try:`
run_graph + `images = results[self._TERMINAL][0]` (extraction INSIDE try so a KeyError/IndexError
can't leak the probe thread -- mirror eng_wan_ti2v.py:456-461); `finally: peak = probe.stop()` +
do `_retain_model_patchers` / encode / `reclaim_idle_models` in a guaranteed block; THEN
`_MC.assert_peak_within_ceiling(peak, ceiling=_MC.dynamic_vram_ceiling_mb())` (the assert must NOT
short-circuit cleanup). Replaces the instantaneous post-render `assert_vram_within_ceiling`.

## Smoke -- generalize per-recipe (records the recipe ACTUALLY run; default sharp_lora)
Generalize `_preflight_distilled_native_graph` (run_otr_30word_smoke.py:196-245) to switch on
`LtxAudioInEngine()._recipe()`, KEEPING the live-workflow + Z-Image/LTX enablement checks (:202-212):
- sharp_lora: lora + sigmas PRESENT; modelsampling/sched ABSENT; guider.model <- lora.
- distilled_native: sigmas PRESENT; lora/modelsampling/sched ABSENT; guider.model <- unet.
- m0_base: modelsampling + sched PRESENT; lora/sigmas ABSENT; guider.model <- modelsampling.
Record recipe + decode knobs + NVML render-phase peak.

## Tests (final)
- `_scale_filter` contracts: -vf form (clip/floor) + labeled `[in]..[out]` form (bg/fg); lanczos+
  unsharp present iff sharpen=True; pad absent iff pad=False.
- Alpha: a SEMI-transparent/transparent-edge RGBA fixture (not just opaque alpha=255) -> assert
  edges stay (partially) transparent after format=rgba->scale->unsharp->overlay.
- decode env override (runtime read, monkeypatch no-reload) + bad-value guard.
- ffmpeg `unsharp` capability via the RESOLVED `fb` (_ffmpeg_bin :79-81), sharpen-paths only, run
  in the test suite (NOT at import/startup).

## Unchanged (survived all rounds)
DECODE 128/32 default; CANVAS 512x288; NO canonical-JSON edit (env-only, no new widgets -- if a
widget is ever added, JSON must change same-commit + revalidate); don't touch eng_humo/eng_wan_ti2v;
companion-drift manifest hardening CUT (schema unchanged); 2-of-3 (stutter open); prod/main GATED.
