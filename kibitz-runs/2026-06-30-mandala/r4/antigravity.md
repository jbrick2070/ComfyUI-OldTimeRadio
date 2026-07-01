VERDICT: yes-with-fixes.
One line why: The signature for the new CRT post-processing helper in `scope_draw.py` is under-specified and lacks parameters necessary for dynamic, frame-to-frame noise generation and audio reactivity, and the cairo data-extraction step lacks a mandatory flush command.

MUST-FIX BEFORE BUILD:
1. [WIRING -- LOCKED at r3: L97] Under-specified signature for `apply_crt_post_rgb`. The plan specifies adding `apply_crt_post_rgb(rgb, scan, vig, rng_key)` to `nodes/_otr_shared/scope_draw.py`. However, this signature lacks:
   - The frame index `fi`, which is required by `_rng(key, fi, salt)` to generate dynamic, frame-varying grain/noise (using a static `fi=0` would produce frozen noise).
   - The audio volume `vol` (or an explicit `noise_intensity` value) required to calculate the dynamic noise intensity (`intensity = int(4 + float(vol) * 10)`).
   Fix: Define the helper signature as:
   `apply_crt_post_rgb(rgb, scanlines, vignette, fi, rng_key, vol=0.0)`
   Where the helper converts `rgb` to a PIL Image, applies `scanlines` (composite) and `vignette` (multiply), and seeds `_rng` using both `rng_key` and `fi` with dynamic intensity scaled by `vol`. Update the naming of `scan` and `vig` to `scanlines` and `vignette` to prevent naming divergence.

2. [WIRING -- LOCKED at r3: L94] Missing cairo `surface.flush()` call. In pycairo, `surface.flush()` must be explicitly called immediately after painting and before reading from `surface.get_data()` to ensure that cairo commits all pending drawing operations to the underlying memory buffer. Without this, some platforms/backends may return blank or partially-rendered frames.
   Fix: Add an explicit directive in the `surface->rgb` conversion step:
   `surface.flush()` must be called immediately before extracting the data buffer via `surface.get_data()`.

SHOULD-FIX:
1. [DECISIONS -- LOCKED at r1: L43 / WIRING -- LOCKED at r3: L99] Radius and spoke scaling multipliers are not locked. The plan dictates that outer core rings must be `<= ~0.33*min(w,h)` and spokes must be capped to avoid 16:9 clipping. However, it does not lock the specific scaled-down multipliers/coefficients (originally `rr = (0.10 + 0.055 * i) * min(w, h) * (1.0 + bass * 0.30)` for rings, and `outer = (0.52 + bass * 0.1) * min(w, h)` for spokes).
   Fix: Lock the exact scaled formulas to prevent implementor visual drift [ASSUMPTION]:
   - Concentric rings: `rr = (0.05 + 0.0275 * i) * min(w, h) * (1.0 + bass * 0.30)` (exactly 0.5x scaling).
   - Radial spokes outer limit: `outer = (0.33 + bass * 0.06) * min(w, h)`.
   - Outer solid spectrum band: `r_in = (0.25 + bass * 0.04) * min(w, h)` and `r_out = r_in + (0.01 + 0.06 * (0.4 + mag)) * min(w, h)`.

2. [DECISIONS -- LOCKED at r1: L23] Module-level type annotations check. Ensure `cairo` is not referenced in module-level type annotations (e.g., use `'cairo.Context'` as a string literal instead of `cairo.Context` if type annotations are added) to avoid import-time `NameError` on machines without `pycairo` installed.

OPTIONAL / NICE-TO-HAVE:
1. [TESTS: L61] Add a subprocess-isolated cold-import test in `test_video_viz_mandala.py` (mirroring `test_video_viz_rainbow.py` L131-140) to verify that importing `nodes._otr_video_engines.eng_viz_mandala` does not pull `cairo` into `sys.modules` at module import time, preserving the cold-import invariant (V-12).

CUT THESE:
None — the plan is highly focused and minimal.

VERIFY-AT-BUILD checklist:
1. Confirm that running the test suite on a system without `pycairo` installed passes the `assert_usable` test (unskipped via monkeypatched import machinery) and raises `EngineUnusable` with `EngineUsabilityReason.MISSING_MODEL`.
2. Confirm that running the test suite on a system with `pycairo` installed passes the visual smoke tests (determinism checks, non-black ratio checks, and frame-to-frame delta checks).
3. Confirm that `apply_crt_post_rgb` returns an array of shape `(H, W, 3)` with type `uint8` and does not mutate the input array in-place.
