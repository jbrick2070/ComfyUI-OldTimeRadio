VERDICT: yes-with-fixes. Close the stale r2/open-choice text and the concrete wiring/render-contract gaps below before handing to a builder.

MUST-FIX BEFORE BUILD:
1. [WIRING -- LOCKED at r3] Wrong import target: `__init__.py:~139` points at the repo root `__init__.py`, where line ~139 is node mapping comments, not video-engine registration. Concrete fix: say `nodes/_otr_video_engines/__init__.py`, add guarded `from . import eng_viz_mandala as _eng_viz_mandala` after the `eng_viz_rainbow` block. Verified: `nodes/_otr_video_engines/__init__.py:132-141`; root `__init__.py:119-160`.

2. [DECISIONS / WIRING] Missing `render_aspect="wide"` requirement. Every registered video engine must declare `render_aspect` and non-portrait engines are expected wide; otherwise existing tests fail and the dropdown label/aspect logic drifts. Concrete fix: add `render_aspect="wide"` to the engine contract and tests. Verified: `tests/test_still_aspect_and_labels.py:32-52`, `nodes/_otr_video_engines/eng_viz_rainbow.py:49`, `nodes/otr_video_director.py:41-58`.

3. [WIRING -- NEW helper] `apply_crt_post_rgb(rgb, scan, vig, rng_key)` is under-specified for deterministic temporal grain. Existing `scope_draw._rng` keys on `(key, fi, salt)`, and existing grain varies by frame; without `fi`, implementors may create static grain or use non-deterministic state. Concrete fix: lock signature as `apply_crt_post_rgb(rgb, scanlines, vignette, rng_key, fi, grain_intensity=...)`, no in-place mutation, same key+fi stable, different fi allowed to differ. Verified: `nodes/_otr_shared/scope_draw.py:40-43`, `nodes/_otr_shared/scope_draw.py:373-377`.

4. [WIRING -- surface->rgb] Missing `surface.flush()` before reading Cairo surface data. The prototype does this immediately before `get_data`; the plan’s final handoff only mentions stride/BGRA. Concrete fix: add `surface.flush()` after painting and before `surface.get_data()` in the surface-to-RGB path. Verified: `docs/2026-06-30-viz-rainbow/mandala_proto.py:178-183`.

5. [OPEN FOR r2] Stale unresolved section remains in the final r4 input and contradicts later locked decisions. It still says “Pick by the measured budget” and asks to set a numeric perf budget, while later sections say PIL roundtrip is decided. Concrete fix: delete `OPEN FOR r2` or rewrite it as “Resolved build locks”: `<=40 ms/frame @1472x832`, `25-frame beat <1s paint`, PIL roundtrip, raw-frame visual smoke.

SHOULD-FIX:
1. [DECISIONS #2] “Lazy import cairo INSIDE render_clip only” conflicts with the later `assert_usable` import-probe. Concrete fix: say “no module-scope cairo import; import-probe in assert_usable and import inside render_clip.”

2. [GROUNDING / title] “production engine” is misleading after r3 locked this as opt-in selectable, not a saved-widget default. Concrete fix: retitle as “production-quality selectable engine” or “selectable mandala engine.”

3. [TESTS] Missing-cairo test is still slightly vague. Concrete fix: specify patching `builtins.__import__` for `name == "cairo"` or use a subprocess, so the test remains unsKIPPED even when pycairo is installed.

OPTIONAL / NICE-TO-HAVE:
- Add a tiny contract test for `apply_crt_post_rgb`: shape/dtype, no mutation, same seed+frame equal, different frame or key can differ.

CUT THESE:
1. [OPEN FOR r2] Cut the whole unresolved historical section after folding the final decisions. It is safe because r2/r3 already locked the choices and leaving it creates ambiguity.
2. [GROUNDING] Cut or soften exact local versions “pycairo 1.29.0 / cairo 1.18.4” from build requirements unless the build intends to pin them. Keep them as provenance only.

VERIFY-AT-BUILD checklist:
1. Verify pycairo import works on the build box, and missing pycairo raises the planned loud `pip install pycairo` `EngineUnusable` path.
2. Verify ffmpeg resolution via `_sd.find_ffmpeg(os.environ.get("OTR_FFMPEG","ffmpeg"))` has a separate loud failure from missing cairo.
3. Verify Cairo `ARGB32` readback uses `surface.get_stride()`, `surface.flush()`, BGRA-to-RGB conversion, contiguous HxWx3 uint8, and opaque alpha.
4. Verify raw-frame visual smoke: nonblack ratio, frame-to-frame delta, deterministic hash.
5. Verify workflow reachability without saved-widget mutation: load `workflows/otr_scifi_16gb_full.json`, run `OTR_WorkflowValidator`, JSON round-trip, link/widget audit, and confirm `viz_mxc_mandala` appears in the registry-driven dropdown.