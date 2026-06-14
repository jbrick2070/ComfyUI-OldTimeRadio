<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. As written it has incompatible FPS/frame-count assumptions, an unplumbed floor flag, ambiguous tail detection, and interface gaps that will break the blend/assertion path.

MUST-FIX BEFORE BUILD:
1. [Pipeline + wiring / Part2 / Beat map + eligibility] FPS is contradictory. The plan hardcodes scopes/planner/output at 25fps, but grounded `OTR_SignalLostVideo` defaults to `fps=24`, while `OTR_SilentComposite` defaults to 25. If the floor/procgen remains 24fps, the “all three inputs == total” assertion before blend can fail. Concrete fix: make the workflow force `OTR_SignalLostVideo.fps=25` for this pipeline, or change its default to 25, and pass the same `fps` into `plan_timeline_segments`, `_analyze_audio`, scopes encoding, SilentComposite, and PostUpscale checks. If non-25 is supported, remove all hardcoded `fps=25` calls and derive from manifest/input consistently.

2. [Part4] `draw_scopes` is not actually plumbed through the grounded floor renderer. Grounding shows `SignalLostVideoRenderer.INPUT_TYPES` has no `draw_scopes`, `_CRTRenderer.__init__(w,h,title)` has no flag, and `_render_crt(fi)` calls `renderer.render(fi, total_frames, fps, volume[fi], freqs[fi], waves[fi])` with no flag. Concrete fix: add the optional input to `SignalLostVideoRenderer.INPUT_TYPES`, pass it through `render_video`, store it on `_CRTRenderer` or pass it into `render()`, and wrap sections 2/3/5/6 plus any dependent locals.

3. [Part1 vs Part4] The plan contradicts its own v1 compatibility requirement. Part1 says rewrite the floor scopes into new circular green-only helpers and “both the floor (v1) and the new node call them”; Part4 says `draw_scopes=True` is “v1 compat.” Grounded v1 floor currently includes non-green center ring/particles/wave/freq bars using `CRT_CYAN`, `CRT_AMBER`, red/yellow gradients. Replacing those with circular green-only helpers is not v1-compatible visually. Concrete fix: either keep the existing floor drawing unchanged when `draw_scopes=True` and only skip it when false, or explicitly drop the “v1 compat” claim and update tests/expectations.

4. [Beat map + eligibility] Tail-gap detection is underspecified and wrong if implemented as “final segment(s) past the last clip.” Grounded `plan_timeline_segments` segments do not include a `kind` field marking head/inter/tail; missing beat clips are emitted as `floor`/`black` with `shot_id`/`engine_id`, indistinguishable from gaps except by reconstructing the manifest timing. If the last real clip is not the last beat, “past the last clip” will suppress valid missing-beat/floor content as if it were credits tail. Concrete fix: compute `tail_start_frame` from manifest beat extents, not last emitted `"clip"` segment. In positioned mode use `max(round(start_s*fps)+target_frame_count)` over all manifest rows; in sequential mode use `sum(target_frame_count)` over all rows. Suppress only gap segments with `cursor >= tail_start_frame` that come from the target-total tail fill.

5. [Part2 / Audio re-analysis] Optional `audio` behavior is not buildable as stated. The plan says audio may be absent and should produce a silent black overlay, but `_analyze_audio(audio_np, sr, total, fps)` requires concrete audio samples and sample rate. Concrete fix: define the absent-audio branch explicitly: either emit all-black frames without calling `_analyze_audio`, or synthesize `volume=[0]*total`, `freqs=[np.zeros(32)]*total`, `waves=[np.zeros(200)]*total`.

6. [Pipeline + wiring / Part2 / Part3] The blend node cannot assert “all three inputs == total” from the interfaces described. `OTR_SceneAwareScopes` returns only `scopes_mp4_path`, and `OTR_PostUpscaleProcgenBlend` inputs are only paths; no manifest or expected frame count is passed. Concrete fix: either change the assertion to ffprobe equality among `source_mp4_path`, `procgen_mp4_path`, and `scopes_mp4`, or add an `expected_total_frames`/manifest input or return a report from `OTR_SceneAwareScopes` and pass it downstream.

7. [Part2 / Beat map + eligibility] The `fps` optional input conflicts with hardcoded 25fps planning. Part2 exposes `fps`, but Beat map calls `plan_timeline_segments(... fps=25)` and Part2 says output is `@ 25fps`. If a user sets `fps != 25`, total frames, audio analysis, and encode length can diverge. Concrete fix: remove the `fps` input and hard-lock 25, or thread the input through planner/analyzer/encoder and use manifest fps consistently.

8. [Part2 / Beat map + eligibility] Center-gap scope geometry is missing. The document gives gutter geometry for portrait clips but says floor/black head/inter-beat gaps should “DRAW scopes (centre ok)” without defining center `cx/cy/r` or whether one/two scopes are drawn. Concrete fix: specify exact center-gap geometry, e.g. `cx=out_w//2`, `cy=out_h//2`, `r=int(min(out_w*0.16, out_h*0.30))`, and reuse the same amplitude cap.

9. [Audio re-analysis] The determinism claim is false against grounding. The document says “stable-hash seeded RNG … same as v1; also fixes the floor’s unseeded np.random,” but grounded `_CRTRenderer.render()` uses `np.random.randint(...)` directly when `vol > 0.3`. Concrete fix: state this is a new floor change, and implement deterministic per-frame RNG, e.g. seeded from `(episode/stem, fi)` or a stable hash passed into the renderer. Do not claim existing v1 is seeded.

10. [Part2 / Pipeline + wiring] New node registration and imports are omitted. A new ComfyUI node is not buildable from `CATEGORY/FUNCTION/RETURN_TYPES` alone; it must be added to whatever module-level `NODE_CLASS_MAPPINGS` / display mappings exist. Grounding does not show those files. Concrete fix: add an explicit build step: register `OTR_SceneAwareScopes` in the package node mappings and verify `plan_timeline_segments` is importable without side effects. [ASSUMPTION] exact registration file name must be verified in repo.

SHOULD-FIX:
1. [Part1] `env` for `draw_fft_scope(..., env)` / `draw_scope(..., env)` is undefined. Concrete fix: define required keys/types, or replace `env` with explicit parameters such as `vol`, `t`, `brightness`, `rng`.

2. [Part3] The 3-input filtergraph ignores existing `blend_mode` / `blend_opacity` semantics. Grounding shows `OTR_PostUpscaleProcgenBlend` already exposes those options. The proposed graph hardcodes first blend to `screen` and second to `lighten` with no opacity. Concrete fix: either document that `scopes_mp4` mode forces canonical `screen/lighten` and ignores `blend_mode`, or preserve `blend_opacity` on the procgen blend and add a fixed/no-user opacity decision for scopes.

3. [Part3] The proposed scopes chain lacks explicit timing normalization. It says `[2:v]<scale>...`, but not `fps`, `setpts`, `trim`, `setsar`, or `shortest`. If scopes is generated correctly this may work, but it is fragile. Concrete fix: normalize the third input similarly to procgen: scale/crop or scale/pad to delivery, `fps=<fps>`, `setpts=PTS-STARTPTS`, `setsar=1`, and use `shortest=1` on blends after frame-count validation.

4. [Part2 / Beat map + eligibility] Aspect probing should be cached by path. The plan says ffprobe each `source=="clip"` segment; repeated beats from the same file will re-probe unnecessarily. Concrete fix: memoize `{path: (w,h,status)}` and log suppressions once per path.

5. [Part2] Empty or malformed manifest behavior is undefined. Grounded `plan_timeline_segments` can return `([], 0)` if there are no valid clips and no `target_total_frames`. Encoding zero frames through ffmpeg stdin will likely fail. Concrete fix: fail early with a clear error when `total <= 0`, unless audio fallback length is explicitly supported.

6. [Part2] “silent black overlay” vs “black frame + green-only scopes” is ambiguous for absent audio. Concrete fix: state whether absent audio means no scopes at all, or scopes drawn with zero volume/frequency.

7. [Part3] Output colorspace flags are only specified for scopes, not repeated for the final blend. Concrete fix: ensure the extended PostUpscale command keeps the existing bt709/CFR/yuv420p encode flags, not only `[out]format=yuv420p`.

8. [Beat map + eligibility] `VERIFY: path is a pre-composite asset` is left open but affects correctness. If paths are already pillarboxed/canvas-sized, `h>w` will suppress portrait incorrectly. Concrete fix: add a build-time assertion/log that prints probed dimensions and selected aspect for every unique clip path, and fail/suppress according to the documented policy.

OPTIONAL / NICE-TO-HAVE:
- Add golden-frame tests with exact frame indices around head gap start/end, portrait clip start/end, landscape clip start/end, and tail start.
- Add one ffprobe-based test asserting scopes mp4 has no audio stream, expected resolution, `yuv420p`, CFR, and expected frame count.
- Add a debug JSON report from `OTR_SceneAwareScopes` listing segment ranges, eligibility reason, aspect probe result, and geometry.

CUT THESE (over-engineering):
1. [Part1] Cut “both the floor and the new node call the new helpers.” It is safer to leave the grounded floor renderer unchanged except for the `draw_scopes` guard, and implement circular green-only helpers for `OTR_SceneAwareScopes` only. This preserves v1 compatibility and reduces regression risk in the existing procgen floor.

2. [Part2] Cut user-facing “scope params” for the first build. Fixed geometry/brightness from the spec is enough to validate scene-aware eligibility and blend behavior. Exposing knobs now increases test matrix and makes golden frames unstable.

3. [Pipeline + wiring] Cut strict “== total” comparison against manifest inside PostUpscale unless you pass `total` into that node. Equality among the actual ffprobed input videos is sufficient for the blend node; manifest-total validation belongs in `OTR_SceneAwareScopes` or an upstream report.