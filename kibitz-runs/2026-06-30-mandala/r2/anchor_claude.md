# Claude anchor -- r2 (coding plan)

Grounded vs eng_viz_rainbow.py (the pattern to mirror), scope_draw.py, render_driver.py,
registry.py, capability_profiles.py, mandala_proto.py.

## VERDICT: build-ready coding shape; 3 coding calls to lock.

## MODULE SHAPE (mirror eng_viz_rainbow.py exactly)
- New `nodes/_otr_video_engines/eng_viz_mandala.py`: class `VizMxcMandalaEngine`, name="viz_mxc_mandala",
  family="abstract", required_inputs=(), accepts_still=False, engine_version="1",
  **fallback_engine="viz_mxc_cpu"** (r1 lock -- differs from viz_mxc_cpu which is None).
- render_clip mirrors eng_viz_rainbow.render_clip: resolve canvas/frames/seed, optional audio ->
  analyze_audio_np + dual_ema + onset; loop paint -> encode_silent_mp4; qc stamps mode(reactive|idle) +
  audio_used. The ONLY body difference: paint via cairo (paint_mandala) not PIL (paint_rainbow_frame).
- `import cairo` LAZILY at the top of render_clip (NOT module scope) -> cold-import test passes.
- assert_usable override: try import cairo; on ImportError raise EngineUnusable/RenderError with
  "pip install pycairo". (Check how viz_mxc_cpu/base does assert_usable -- likely inherits; add the probe.)

## PAINTER -> move into scope_draw (shared, testable)
- Promote paint_mandala + _band/_centroid/_hue + surface_to_rgb from mandala_proto.py into scope_draw.py
  (where paint_rainbow_frame + build_scanlines + _rng already live) so tests import the same module and
  the CRT helpers are in-reach. Keep signature seed-keyed (rng_key) for grain.
- surface_to_rgb: use surface.get_stride(); if stride==w*4 the fast [:, :, [2,1,0]] path; else copy
  row-by-row honoring stride. Assert opaque bg.

## CRT GLUE -- my lean: PIL-roundtrip, reuse scope_draw as-is
- cairo surface -> numpy RGB -> reuse the SAME scanline/vignette/grain application viz_mxc_cpu already
  uses (proven, one code path, identical period look across both viz engines). Native-cairo
  OPERATOR_MULTIPLY is faster but forks the CRT look into two implementations -> drift risk. Only switch
  to native if the perf benchmark FAILS the budget. (SHOULD: measure first.)

## PERF BUDGET (propose)
- Budget: <= ~40 ms/frame at 1472x832 on CPU => a 25-frame beat paints in ~1s + encode; a typical
  ~6-12s beat (150-300 frames) < ~15s -- an order of magnitude under any GPU engine, no soak risk.
  Benchmark mandala vs viz_mxc_cpu on the same beat; record ms/frame in the build doc.

## TESTS (new tests/test_video_viz_mandala.py, mirror rainbow + add)
- registration+CAPABILITIES row; required_inputs=() fits all 5; accepts_still False; ambient gate;
  oracle motion-exempt; family map; render contract (encode monkeypatched) audio-present+absent;
  frame-count exact; determinism (paint twice -> array_equal); cold-import (no cairo/torch at module
  import); **assert_usable-missing-pycairo path** (monkeypatch import to raise -> asserts the loud
  message); **fallback_engine == "viz_mxc_cpu"**; **visual-acceptance smoke** (a few real cairo frames:
  nonblack ratio > threshold, frame-to-frame delta > 0 on audio, deterministic hash).

## OPEN (r2 -> panel)
- Does the engine base already provide assert_usable I can override, or is eligibility purely
  capability-driven (so a missing-dep engine needs an explicit gate)? verify in the base + registry.
- Confirm engine_consumes_still / the still-mint gate keys off accepts_still (it does for viz_mxc_cpu).
