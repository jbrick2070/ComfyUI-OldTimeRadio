# LTX upgrade wiring plan -- r2-HARDENED (coding plan; Codex + AntiGravity CONVERGED, Claude judge)

Builds on r1_plan.md. r2 turned the scaler refactor into a precise, code-grounded coding spec.
Both agents converged; the new headline is alpha preservation.

## Scaler: a shared helper with sharpen AND pad controls (the 4 scale paths)
Define `_scale_filter(w, h, *, sharpen, pad=True)` in otr_silent_composite.py returning the inner
`scale[:flags=lanczos][,unsharp=...],[pad=...,]fps` chain (composable into a `-vf` string OR a
labeled `-filter_complex` `[in]...[out]`). unsharp amount = `OTR_COMPOSITE_UNSHARP_AMOUNT` (0.4).
Apply per call site (all CONFIRMED in the code):
- `_seg_vf` / `_encode_segment` (:319, :336, callers :579-587): add `sharpen` param -> `True` for
  real clips (`kind=="clip"`), `False` for the procgen floor. `_encode_black` UNCHANGED.
- `_encode_segment_from_dir` (:403-461): foreground `fg_filter` (:444, RGBA) -> `sharpen=True,
  pad=False` (CRITICAL: pad with color=black destroys the RGBA alpha -> breaks overlay). bg per
  kind (real bg sharpen=True; generated black bg sharpen=False). [ASSUMPTION] textured still-plate
  bg = character lane -> sharpen=True unless operator says floor-like.
- `normalize_to_silent_canonical` (:72-105, the 4th path, called when no clip manifest :704-707):
  refactor its ad-hoc `vf` to `_scale_filter(..., sharpen=False)` for consistency (floor-like).

## Decode env (read at RUNTIME for testability)
`OTR_LTX_AV_DECODE_TEMPORAL_SIZE` (128) + `OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP` (32) -- explicit
TEMPORAL names; keep spatial `tile_size=512` / `overlap=64` fixed. Read them INSIDE `_build_graph`
(eng_ltx_av.py:447-560), NOT module-load, so `monkeypatch.setenv` works without a module reload.
Whole-clip = set the env to 4096/8 (config flip, no source edit).

## VRAM verify (real peak, right ceiling)
`VramPeakProbe` is NOT a context manager (motion_common.py:242-295) -- use start/stop:
`probe = _MC.VramPeakProbe().start(); try: with _ltx_av_vram_reserve(): results = _wb.run_graph(...)
finally: peak = probe.stop()` then `_MC.assert_peak_within_ceiling(peak, ...)` (replaces the
instantaneous post-render `assert_vram_within_ceiling` at eng_ltx_av.py:621-633). Ceiling =
`_MC.dynamic_vram_ceiling_mb()`, not `wrapper_bridge.py:37`.

## Smoke: record the ACTUAL recipe (don't hardcode distilled_native)
`scripts/run_otr_30word_smoke.py:188-204` `_preflight_distilled_native_graph` REJECTS the no-env
default (sharp_lora). Generalize it to read `LtxAudioInEngine()._recipe()` and validate the graph
for whichever recipe runs (default = sharp_lora), recording recipe + decode knobs + NVML peak.

## Tests (r2)
- `_scale_filter` string contract: -vf form (clip/floor) + labeled form (bg/fg); assert
  lanczos+unsharp present where sharpen=True, ABSENT where False, and pad ABSENT where pad=False.
- Alpha: fg path keeps RGBA (no opaque pad) -- reuse the opaque-mesh pattern in
  tests/test_3d_image_streams.py:197-226.
- decode env override (runtime read; monkeypatch without reload).
- ffmpeg `unsharp` capability preflight (parse `ffmpeg -filters`, fail loud if missing).

## Unchanged from r1 (survived grounding)
DECODE 128/32 default; CANVAS 512x288; no canonical-JSON edit; don't touch eng_humo/eng_wan_ti2v;
CUT the companion-drift manifest hardening (both agents); 2-of-3 (stutter open); prod/main GATED.

## Open for r3/r4 (wiring/convergence)
- Confirm the floor-vs-character `sharpen` booleans against real episode segment routing.
- Sequence the edits so the suite stays green per chunk; the exact still-plate-bg decision.
