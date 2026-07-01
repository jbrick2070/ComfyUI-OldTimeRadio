# r2 JUDGMENT -- mandala engine (coding plan)

Panel present at pause: Codex (VERDICT no -- fallback path + open build choices). Antigravity + Claude-Code
r2 still crawling when the operator paused (fold on resume if they add anything). Claude anchor + Codex
grounded vs the real files.

## CORRECTION TO r1 (grounded, important)
- **r1 lock #3 was WRONG.** I locked `fallback_engine="viz_mxc_cpu"` on an L158 chain helper. Codex
  grounded the ACTUAL production path: `render_shot` (render_driver.py:1531-1558) has the operator
  directive *"NO FALLBACKS (2026-06-16, 'this is art, not a space shuttle')"* -- `fallback_of` is
  accepted but IGNORED, a hard failure RAISES `RenderError` loud, no swap, no still floor. Tests lock it
  (test_video_render_driver_additive.py:159,733). => The L158 `fallback_engine` helper is dead for this
  path.
- **CORRECTED DECISION:** viz_mxc_mandala declares `fallback_engine=None` (mirror viz_mxc_cpu +
  ltx_audio_in) and FAILS LOUD. assert_usable import-probes cairo -> loud "pip install pycairo" at
  eligibility; a cairo render crash raises loud like any engine. No graceful degrade. This supersedes r1
  #3 and matches the operator's no-fallback contract.

## LOCKED FOR THE BUILD (r2)
1. Registration is CONCRETE (Codex #2): create `nodes/_otr_video_engines/eng_viz_mandala.py` with
   `@register class VizMxcMandalaEngine(name="viz_mxc_mandala", family="abstract", required_inputs=(),
   accepts_still=False, fallback_engine=None)`; ADD the import row in
   `nodes/_otr_video_engines/__init__.py`; ADD `registry.CAPABILITIES["viz_mxc_mandala"]`
   (cpu_ok, required_toolchain=None, model_requirements=[]); UPDATE `render_driver.ENGINE_FAMILY`,
   `content_oracle._FAMILY_FALLBACK`, and `render_driver._uses_ambient_master_audio` -- the exact map set
   viz_mxc_cpu touched. (Invariant: test_capability_profiles.py:215 -- every registered engine needs a row.)
2. CRT glue = PIL roundtrip, DECIDED (Codex CUT native-cairo; my lean). scope_draw grain is NOT a public
   API (private `_rng`, embedded in paint fns). Add a small PUBLIC helper `apply_crt_post_rgb(rgb, ...,
   rng_key)` in scope_draw that applies scanlines+vignette+seed-keyed grain to an RGB array, and call it
   from BOTH the mandala engine and (optionally later) viz_mxc_cpu. Deterministic via rng_key.
3. Painter into scope_draw: promote paint_mandala + _band/_centroid/_hue + a `mandala_surface_to_rgb`
   that reads `surface.get_stride()` (Codex/Antigravity/anchor) into scope_draw for shared testing.
4. Radius caps LOCKED (Codex #4): `outer_max = min(cx, cy, w-cx, h-cy) - margin` bounds the eye + rings;
   only the named OUTER SPECTRUM BAND may bleed to the edges (operator asked for denser/thicker 2026-06-30).
5. Perf: numeric budget (<= ~40 ms/frame @ 1472x832; a 25-frame beat < ~1s paint). ONE local benchmark
   script/log gate comparing mandala vs viz_mxc_cpu -- NOT a unit-test timing assert (Codex CUT #2).
6. Tests (Codex SHOULD): hash RAW frames not the mp4 (monkeypatch encode_silent_mp4, capture RGB,
   hash np.ascontiguousarray(frame).tobytes()); cold-import subprocess test for eng_viz_mandala (the
   __init__ swallows adapter import errors at L138 -> a module-scope `import cairo` would SILENTLY
   unregister the engine); separate assert_usable-missing-cairo test (patch import to raise); keep
   accepts_still=False no-image assertion; a 96x64 frame fixture for nonblack-ratio + frame-delta.

## STATUS: r1+r2 converged into a build-ready coding+wiring plan (with the fallback correction).
Remaining kibitz value (r3 wiring / r4 convergence) is LOW -- wiring mirrors the already-shipped
viz_mxc_cpu exactly and r2 already nailed the concrete registration + map set. On resume, either fold any
late Antigravity/Claude r2 notes + do a quick r3 wiring confirm, or proceed straight to the build.
