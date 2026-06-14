<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The design cannot be built as written because required data is not wired/provided, the audio-analysis contract contradicts the grounded implementation, and the 3-input blend/filtergraph is underspecified relative to the existing RGB-pinned blend path.

MUST-FIX BEFORE BUILD:
1. [The shape/2] `OTR_SceneAwareScopes` promises `length == the source video` and [Audio re-analysis] says `total_frames = ffprobe the SOURCE video's exact frame count`, but the proposed INPUT_TYPES do not include any source video path. The node cannot know the frame count it must render. Concrete fix: add a required `source_mp4_path`/`silent_video_path` input, ffprobe that for frame count/dimensions, and validate the produced `scopes_only.mp4` has the same frame count. If the node must run parallel to RTXUpscale, use the SilentComposite output path as the frame-count source and add a later validation that RTXUpscale preserved frame count.

2. [Pipeline + wiring] The stated wire `SilentComposite -> SceneAwareScopes` for `clip_manifest_json` is not possible with the grounded `OTRSilentComposite` IO: it returns only `("silent_video_path", "report")`; `clip_manifest_json` is an input, not an output. Concrete fix: either wire the original manifest producer directly into `OTR_SceneAwareScopes`, or extend `OTR_SilentComposite` to return the normalized manifest JSON as a third output and update `RETURN_TYPES`/`RETURN_NAMES`.

3. [Audio re-analysis] The spec says `_analyze_audio(audio_np, sr, total_frames, fps=25)` must use `round(fi*sr/fps)..round((fi+1)*sr/fps)` boundaries, but the grounded implementation uses `spf = sample_rate // fps` and fixed chunks. The statement “same `_analyze_audio` => frame-identical to a 25fps floor” is therefore false unless the floor is changed too. Concrete fix: either update `_analyze_audio` globally and verify all existing floor renders tolerate the changed analysis, or add a new exact-boundary helper for `OTR_SceneAwareScopes` and remove the “frame-identical to floor” claim. If frame identity is required, both floor and scopes must call the same helper with the same `total_frames` and `fps`.

4. [The shape/3] The 3-input ffmpeg blend is not compatible with the grounded current green-only filter by just appending `[tmp][scopes]blend=screen[out]`. The existing path pins `[main]` and procgen to `gbrp`, zeroes procgen R/B, performs one blend, then converts to `yuv420p`. A naive second blend after that risks running in YUV or failing labels/formats. Concrete fix: build a new filtergraph for three inputs: scale/crop/setpts procgen and scopes to source size, zero R/B on both overlays, `format=gbrp` on main/procgen/scopes, first blend to a `gbrp` tmp without final yuv conversion, second blend in `gbrp`, then only final `format=yuv420p`.

5. [The shape/3] The extended blend must define stream duration behavior for the third input. Current grounded comments mention `blend=...:shortest` for the two-input path. If `scopes_only.mp4` is one frame short/long, the output can truncate or desync depending on `shortest`/EOF behavior. Concrete fix: make the scopes node length-validated before blend, and set explicit blend options for both blends, e.g. same `shortest` behavior as current path after confirming all inputs have identical frame counts.

6. [The shape/2] The node says it generates `1920x1080` but also takes `canvas_w/h`, and [Pipeline + wiring] says it “needs only manifest+audio+canvas”. The grounded SilentComposite canvas defaults are `1472x832`, while the post-upscale blend target is `1920x1080`. Rendering scopes at the composite canvas and blending after upscale will misplace/distort gutters. Concrete fix: separate `output_w/output_h` from any pre-upscale canvas parameters. Derive render size from the ffprobed source video used for blending, defaulting to 1920x1080 only if no source dims are available.

7. [The shape/4] `draw_scopes: bool` is underspecified against the grounded `_CRTRenderer.render()`. The actual render path has centre frequency ring, orbiting particles, grid, mirrored waveform, wide frequency bars, title/timestamp/bottom bar, scanlines/vignette/noise. The plan only names `_waveform_mirror`, `_freq_bars_wide`, and “centre ring” but does not state what happens to orbiting particles/grid or their dependencies when scopes are disabled. Concrete fix: define exactly which render sections are controlled by `draw_scopes`; guard all dependent variables/loops; verify `draw_scopes=False` renders title/CRT/floor without references to removed ring state.

8. [The shape/1] The proposed helpers `draw_fft_scope(draw, cx, cy, r, freq_window, env)` / `draw_scope(draw, cx, cy, r, wave, env)` do not map cleanly to the grounded helpers, which currently take rectangular geometry plus `vol` and `t`: `_waveform_mirror(draw, wave, x, y, w, h, vol, t)` and `_freq_bars_wide(draw, freq, x, y, w, h, vol)`. Concrete fix: specify the adapter math from `(cx, cy, r)` to the existing rectangular draw areas, or change the helper signatures to the geometry actually needed.

9. [OPEN / VERIFY-AT-BUILD] “master-WAV-missing fallback: skip scope drawing rather than fail the node” conflicts with [The shape/2], where `audio` is a required input. In Comfy-style INPUT_TYPES, a missing required `AUDIO` generally prevents execution. Concrete fix: make `audio` optional and generate a black silent overlay of the correct length when absent/invalid, or keep it required and delete the fallback requirement.

10. [Invariants] The document requires deterministic output and [Audio re-analysis] mentions stable-hash seeded RNG, but the grounded `_CRTRenderer.render()` uses unseeded `np.random.randint(...)` for noise when `vol > 0.3`. Concrete fix: if any of this renderer path remains in v1/v2 scope generation, replace global RNG use with a deterministic per-frame RNG seeded from episode/frame parameters, or explicitly exclude the noisy postprocess from the new scopes.

SHOULD-FIX:
1. [Beat map + eligibility] The plan says `source in {"floor","black"}` should draw scopes in the centre, including the credits tail. Grounding for `plan_timeline_segments` explicitly says the tail floor slice is the rolling-credits post-roll. Centre scopes can cover the credits/title floor content. Concrete fix: add a credits-safe placement rule for tail gaps, or annotate gap reason in the segment plan so the tail can use side/gutter scopes, lower opacity, or suppression.

2. [Beat map + eligibility] The aspect resolver has an unresolved dependency: “VERIFY-AT-BUILD: that `path` points at a pre-composite asset with meaningful native dims.” If `path` is already pillarboxed/canvas-sized, `h>w` will misclassify portrait clips as landscape and suppress valid scopes. Concrete fix: validate this against real manifest rows before implementation, or add manifest-level native dimensions upstream.

3. [Beat map + eligibility] The `engine_id -> aspect` registry is listed as open for non-HuMo/LTX/Wan engines. Since unknown/unprobeable clips suppress drawing, missing registry entries will silently reduce scene-aware coverage. Concrete fix: either complete the registry before enabling the feature, or log per-shot suppression reasons so missing aspect data is visible.

4. [Audio re-analysis] The plan uses source video frame count instead of audio duration, but does not define what happens when the audio tensor is shorter/longer than the source video. Concrete fix: specify zero-padding past audio end and truncation past video end, and make normalization stable when all chunks are zero.

5. [The shape/2] `scopes_only.mp4` is described as silent, bt709/yuv420p/CFR, but the implementation requirements do not say to set color metadata. Concrete fix: require ffmpeg encode flags for `-pix_fmt yuv420p`, CFR/fps, and bt709 color metadata, matching the existing delivery conventions.

6. [The shape/3] The behavior of `bypass=True` with `scopes_mp4` present is unspecified. Concrete fix: define bypass as copying `source_mp4_path` without procgen or scopes, preserving current compatibility.

7. [Pipeline + wiring] The branch notation implies `SceneAwareScopes` can run parallel to RTXUpscale, but if its render size is derived from the final upscaled source, it depends on RTXUpscale output. Concrete fix: either accept the sequencing dependency or derive size/frame count from SilentComposite and enforce that RTXUpscale preserves both.

OPTIONAL / NICE-TO-HAVE:
- Add a small diagnostic JSON/report output from `OTR_SceneAwareScopes`: total frames, drawn frames, suppressed portrait/landscape/unknown counts, and first few suppression reasons.
- Add frame-count assertions before and after `OTR_PostUpscaleProcgenBlend`.
- Add golden-frame tests for one portrait clip, one landscape clip, one head gap, and one tail gap.

CUT THESE (over-engineering):
1. [Beat map + eligibility] Cut the initial `engine_id -> aspect` registry unless real manifests contain unprobeable-but-known-aspect clip paths. The safe fallback is already “unknown -> suppress,” and ffprobe should cover normal video assets. [ASSUMPTION] This is safe if generated clip paths are normal video/image files with readable dimensions.

2. [The shape/2] Cut `canvas_w/h` as render-size inputs for `OTR_SceneAwareScopes`. The overlay is post-upscale, so render dimensions should come from the actual source video being blended. Keep only separate pre-upscale constants if needed for gutter math, or derive them from manifest/composite metadata later.

3. [The shape/1] Cut the requirement for two separately named public helpers if one generic green scope renderer can cover waveform/frequency modes through parameters. The build risk is in refactoring `_CRTRenderer.render()`; minimizing helper surface reduces integration points without losing the goal.