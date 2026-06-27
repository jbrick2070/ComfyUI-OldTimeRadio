# LTX upgrade -- FINAL build-ready wiring plan (r1->r4 CONVERGED)

Local roundtable: Codex (gpt-5.5/high, read-only) + AntiGravity (gemini-3.5-pro, file-handoff) x
r1->r4, Claude grounded anchor + sole judge. r4 CONVERGED -- both agents independently agreed on
the same residual fixes. Every accepted claim was verified against the real files.
Goal: wire the bakeoff winner. prod/main + tags GATED. Don't touch eng_humo.py / eng_wan_ti2v.py.

## 1. SCALER -- shared helper, alpha-safe, per-path (NOT a one-line `_seg_vf` edit)
`_scale_filter(w, h, fps, *, sharpen, pad=True)` in otr_silent_composite.py. Emits, IN ORDER:
`scale=w:h:force_original_aspect_ratio=decrease[:flags=lanczos]` then (if sharpen)
`,unsharp=5:5:<amt>:5:5:0.0` then (if pad) `,pad=w:h:(ow-iw)/2:(oh-ih)/2:color=black` then `,fps=fps`.
`<amt>` = `OTR_COMPOSITE_UNSHARP_AMOUNT` (default 0.4). Returns a chain usable as `-vf` OR a labeled
`[in]...[out]` for `-filter_complex`.
Wire into ALL composite scale paths (all CONFIRMED):
- `_seg_vf(w,h,fps,start_frame,sharpen=True)` + `_encode_segment(..., sharpen=True)` (:319/:336);
  `assemble_silent_timeline` (:579-587) passes `sharpen=True` for `kind=="clip"`, `False` for floor.
  PRESERVE `_seg_vf`'s `trim -> scale/unsharp/pad/fps -> tpad` ordering. `_encode_black` UNCHANGED.
- `_encode_segment_from_dir` (:403-461): FOREGROUND fg (:444, straight-RGBA) -> `sharpen=True,
  pad=False` (pad color=black destroys the alpha edges -> opaque borders). BACKGROUND (:570-578):
  `bg_is_still=True` (real still plate) -> `sharpen=True`; `base_video_path` procgen floor (workflow
  link 246) and black -> `sharpen=False`.
- `normalize_to_silent_canonical` (:72-105): `_scale_filter(..., sharpen=False)`.

## 2. DECODE -- env-overridable, fail-loud (NO clamp)
Read INSIDE `_build_graph` (eng_ltx_av.py:447-560, so monkeypatch works w/o reload):
`OTR_LTX_AV_DECODE_TEMPORAL_SIZE` (128) + `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` (32). Absent ->
defaults. Present but non-int / <=0 / `overlap >= size` -> RAISE a named ValueError BEFORE building
VAEDecodeTiled (fail-loud; clamp is CUT). Spatial `tile_size=512`/`overlap=64` fixed. Whole-clip =
set the env to 4096/8 (config flip, no source edit).

## 3. VRAM -- real peak, leak/cleanup-safe, threaded to the report
In `render_clip` (eng_ltx_av.py:621-634): `results = images = None`;
`probe = _MC.VramPeakProbe(interval_s=0.1).start()`; `try:` (with `_ltx_av_vram_reserve()`)
`results = _wb.run_graph(...)`, `images = results[self._TERMINAL][0]`; `finally:`
`peak = probe.stop()`, `_retain_model_patchers` ONLY if `results is not None`, encode ONLY if
`images is not None`, ALWAYS `reclaim_idle_models`. Then
`_MC.assert_peak_within_ceiling(peak, _MC.dynamic_vram_ceiling_mb())` AFTER cleanup (replaces the
instantaneous post-render assert). Return `"vram_peak_mb": peak`.
THREAD it to the report: `_clip_from_raw` copies `vram_peak_mb`; `render_shot`
(render_driver.py:1496) returns `clip.get("vram_peak_mb") or _mc.vram_used_mb()` -> the episode
report (otr_video_render_batch.py:222) now records the REAL render-phase peak.

## 4. SMOKE -- generalize per recipe (records the recipe ACTUALLY run; default sharp_lora)
Generalize `_preflight_distilled_native_graph` (run_otr_30word_smoke.py:196-245): drop the strict
`OTR_LTX_AV_UNET==DISTILLED_UNET` equality + header; switch on `LtxAudioInEngine()._recipe()`,
KEEP the live-workflow + Z-Image/LTX enablement checks (:202-212):
- sharp_lora: lora+sigmas PRESENT; modelsampling/sched ABSENT; guider.model<-lora.
- distilled_native: sigmas PRESENT; lora/modelsampling/sched ABSENT; guider.model<-unet.
- m0_base: modelsampling+sched PRESENT; lora/sigmas ABSENT; guider.model<-modelsampling.
Record recipe + decode knobs + the propagated NVML render-phase peak.

## 5. TESTS
- `_scale_filter` contract: `-vf` (clip/floor) + labeled (bg/fg) forms; order scale->unsharp
  (iff sharpen)->pad(iff pad)->fps; lanczos+unsharp present iff sharpen; pad absent iff pad=False.
- ALPHA = source-over BLEND MATH (output is flattened yuv420p :455-457, NO alpha): a
  semi-transparent-edge RGBA fg over a CONTRASTING still plate -> assert the flattened edge pixels
  show the expected fg/bg blend, NOT an alpha channel.
- decode env override (runtime read, monkeypatch no-reload) + invalid-value raises.
- ffmpeg `unsharp` capability on the RESOLVED `fb` (_ffmpeg_bin :79-81), sharpen-paths only, run in
  the test suite (NOT at import/startup).

## 6. PRE-SHIP (operator/GPU)
Selective box reset (CLAUDE.md S4) -> one canonical `ltx_audio_in` smoke: real render-phase NVML
peak (propagated) < `dynamic_vram_ceiling_mb()`, recipe recorded, asset under
`<output>/otr/episodes/<slug>/`. Full suite green vs the 5 pre-existing 267a53e fails; Bug Bible
16/7/3; AST/no-BOM/no-0-byte. NO canonical-JSON edit (env-only, no new widgets; if a widget is ever
added -> update otr_scifi_16gb_full.json same-commit + OTR_WorkflowValidator + round-trip +
link/widget audit). Commit+push v2.0-alpha; HEAD==origin. Operator may bump
`OTR_LTX_AV_RESERVE_VRAM_GB` 4.0->4.5 on spill.

## Carried (survived all 4 rounds)
DECODE 128/32 default (whole-clip via env); CANVAS 512x288; companion-drift manifest hardening
CUT (both agents; schema unchanged); this upgrade closes 2 of 3 -- SEAM (decode) + SOFTNESS
(scaler) ship; the init-hold STUTTER (i2v 0.62) was never reproduced by the bakeoff -> OPEN eyeball.

## Judgment log (accepted / cut)
ACCEPTED (grounded): shared `_scale_filter` w/ sharpen+pad; the 4 scale paths incl
normalize_to_silent_canonical; alpha pad=False on fg; floor (base_video_path) NOT sharpened;
fail-loud decode env; VramPeakProbe start/stop + threading to the report; per-recipe smoke;
blend-math alpha test; ffmpeg preflight on resolved bin. CUT (both agents): decode-env clamp;
companion-drift manifest hardening. OPEN (operator): still-plate-bg sharpen confirmed True;
stutter (i2v 0.62) eyeball.
