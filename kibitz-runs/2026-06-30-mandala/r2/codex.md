VERDICT: no. Plan relies on a fallback path the current render driver explicitly disables, and several build choices are still open.

MUST-FIX BEFORE BUILD:
1. [DECISIONS 3] Fallback behavior is wrong for this codebase. `fallback_engine="viz_mxc_cpu"` will not be honored by production rendering: `render_shot` says `fallback_of` is accepted but ignored and raises `RenderError` on any failure (`C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py:1531`). Tests lock this in as “no fallbacks” (`.../tests/test_video_render_driver_additive.py:159`, `.../tests/test_video_render_driver_additive.py:733`). Concrete fix: either change the plan to `fallback_engine=None` and fail loud like `viz_mxc_cpu`, or explicitly scope a render-driver fallback restoration plus updates to the no-fallback tests and runtime fallback ledger behavior.

2. [DECISIONS 1 / CONTRACT / OPEN FOR r3] The registration path must be concrete. A new adapter file alone will not register; the package imports adapters explicitly (`.../nodes/_otr_video_engines/__init__.py:132`). The registry consistency invariant requires every registered engine to have a `CAPABILITIES` row and vice versa (`.../tests/test_capability_profiles.py:215`). Concrete fix: specify `nodes/_otr_video_engines/eng_viz_mandala.py`, `@register class VizMxcMandalaEngine`, `name="viz_mxc_mandala"`, import it in `nodes/_otr_video_engines/__init__.py`, add `registry.CAPABILITIES["viz_mxc_mandala"]`, and update `render_driver.ENGINE_FAMILY`, `content_oracle._FAMILY_FALLBACK`, and `_uses_ambient_master_audio`.

3. [OPEN FOR r2 2 / DECISIONS 8] “Reuse scope_draw scanlines/vignette/grain” is not an API as written. `scope_draw` exposes `build_scanlines` and `build_vignette`, but grain is embedded in paint functions and uses private `_rng` (`.../nodes/_otr_shared/scope_draw.py:40`, `.../nodes/_otr_shared/scope_draw.py:368`). Concrete fix: choose the PIL roundtrip path now and add a small public helper such as `apply_crt_post_rgb(...)`, or state that `eng_viz_mandala` owns its own deterministic grain implementation. Do not leave “native-cairo OPERATOR_MULTIPLY” as an undecided branch.

4. [OPEN FOR r2 3 / DECISIONS 6] The bounding rule is not implementable yet. The prototype’s outer band can exceed the 1472x832 vertical half-height: `r_in` plus `r_out` expansion reaches roughly `0.58 * min(w,h)` from center (`.../docs/2026-06-30-viz-rainbow/mandala_proto.py:67`). Concrete fix: lock exact radius caps before coding, e.g. `outer_max = min(cx, cy, w-cx, h-cy) - margin` for core rings, with only the named outer spectrum band allowed to bleed.

SHOULD-FIX:
1. [TESTS] The deterministic visual smoke must hash raw frames, not the encoded mp4. `encode_silent_mp4` shells out to ffmpeg (`.../nodes/_otr_shared/scope_draw.py:399`), so container metadata/encoder behavior can pollute a visual hash. Concrete fix: monkeypatch `encode_silent_mp4`, capture yielded RGB frames, and hash `np.ascontiguousarray(frame).tobytes()`.

2. [OPEN FOR r2 1] The perf gate has no number. Concrete fix: set a local budget before implementation, e.g. max ms/frame at 1472x832 and max seconds for 25 frames, and make the benchmark compare mandala against `viz_mxc_cpu` under the same monkeypatched/non-ffmpeg conditions where possible.

3. [DECISIONS 2] The pycairo-missing test needs to guard against accidental module-scope `import cairo`. The package import currently swallows adapter import exceptions (`.../nodes/_otr_video_engines/__init__.py:138`), which can silently unregister the engine if cairo is imported too early. Concrete fix: add a cold-import subprocess test for `eng_viz_mandala` and a separate `assert_usable` missing-cairo test that patches import resolution.

4. [TESTS / CONTRACT] Mirror tests should include the no-image path. `viz_mxc_cpu` asserts `accepts_still=False` and image skipping behavior (`.../tests/test_video_viz_rainbow.py:50`). Concrete fix: keep the same assertion for `viz_mxc_mandala`, because otherwise all-procedural episodes may start minting unused stills.

OPTIONAL / NICE-TO-HAVE:
- Add a small pure helper for cairo surface conversion so stride/opacity tests can hit it directly without rendering a full clip.
- Include one tiny 96x64 mandala frame fixture test for nonblack ratio and frame-to-frame delta.

CUT THESE (over-engineering):
1. [OPEN FOR r2 2] Cut native-cairo CRT compositing for the first build. PIL roundtrip is already proven by `viz_mxc_cpu` and keeps the implementation testable through existing `scope_draw` paths.
2. [OPEN FOR r2 1] Cut broad soak benchmarking from the unit-test layer. Keep one local benchmark script/log gate; unit tests should validate contracts and deterministic frames, not timing on shared hardware.