<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The design still has unresolved build decisions and the core frame/beat/scopes mapping is not implementable as written against the grounded manifest and renderer code.

MUST-FIX BEFORE BUILD:
1. [Proposed v2 design / “Per frame fi”] The mapping uses `dur_s`, but the grounded manifest/segment data exposes `target_frame_count`, not `dur_s`; `plan_timeline_segments()` returns frame-counted segments with `n_frames`. Floating `t = fi/fps_delivery` also will not exactly match the composite’s `round(start_s*fps)` placement at beat seams. Concrete fix: build the scope visibility map by reusing `plan_timeline_segments(manifest, floor_available=True, target_total_frames=<source_frame_count>, fps=<delivery_fps>)`, then walk cumulative frame ranges `[cursor, cursor+n_frames)` using integer frame indices only.

2. [Proposed v2 design / “Gap beats”] “Active beat whose `[start_s, start_s+dur_s)` contains `t`” will incorrectly draw scopes during missing-clip floor fills, because missing rows still have `engine_id` and `target_frame_count`. The grounded planner emits these as `source: "floor"` or `"black"`. Concrete fix: scope eligibility must be `segment["source"] == "clip"` plus portrait aspect; suppress for `source in {"floor", "black"}` including head gaps, inter-beat gaps, missing clips, and tail.

3. [Proposed v2 design / “same `_draw_fft_scope`/`_draw_scope` module”] The grounded renderer excerpt does not contain `_draw_fft_scope` or `_draw_scope`. The visible audio-reactive drawing is embedded in `_CRTRenderer.render()` and private helpers `_waveform_mirror()` / `_freq_bars_wide()`, and they currently draw full-width layout, not gutter-scoped geometry. Concrete fix: first refactor the actual scope drawing into explicit reusable functions/classes with parameters for `x, y, w, h`, color policy, and visibility; update `_CRTRenderer.render()` to call them conditionally, and have `OTR_SceneAwareScopes` call the same refactored API.

4. [Proposed v2 design / “Split ONLY the scopes out of the floor”] There is no specified flag/branch for stopping the floor from drawing the existing waveform/frequency-bar scopes. Grounding shows `_CRTRenderer.render()` always draws the mirrored waveform and frequency bars when `wave`/`freq` are present. Concrete fix: add an explicit renderer/node option such as `draw_scopes: bool` defaulting to legacy behavior, and set it false in the v2 workflow. Do not rely on “v2 means the floor STOPS drawing them” without an implemented parameter.

5. [INVARIANTS / “green-only” + Proposed v2 design / “Same code”] The grounded scope helpers draw non-green colors: `_freq_bars_wide()` uses red/yellow/orange RGB, `_waveform_mirror()` uses CRT_GREEN and CRT_CYAN, and the central renderer draws amber/cyan/magenta/red elements. Reusing existing drawing code directly violates “green-only output.” Concrete fix: the late overlay must either draw scopes with `R=0,B=0` only, or render to an overlay and zero R/B before compositing, matching the grounded `OTR_PostUpscaleProcgenBlend` green-only behavior.

6. [Proposed v2 design / “Output: mp4 … audio `-c:a copy`”] At the proposed insertion point, the upstream video is still effectively part of the silent-video path: `OTR_SilentComposite` strips audio, then `RTXUpscale`, then `OTR_PostUpscaleProcgenBlend`; final audio is added later by `OTR_MasterAudioMux`. If `OTR_SceneAwareScopes` blindly copies audio, it may copy no stream or fail depending on ffmpeg mapping. Concrete fix: define the new node as video-only/silent-preserving and run it before `OTR_MasterAudioMux`; map optional input audio only if present, but do not require it. The frozen master audio remains consumed only for analysis and muxed only by `OTR_MasterAudioMux`.

7. [Proposed v2 design / “Audio analysis is RECOMPUTED … total frames”] The plan does not state where `total_frames` comes from for `_analyze_audio()`. Grounding shows `_analyze_audio(audio_np, sr, total_frames, fps)` requires an explicit frame count. If computed from audio duration, it can differ from the actual CFR source video length by rounding or master-length padding. Concrete fix: ffprobe/read the actual source video frame count and delivery fps, pass that exact frame count to `_analyze_audio()`, and clamp/zero-pad chunks when audio is shorter.

8. [Open architecture questions / Q2 fps] The “Confirm no 24/25 drift” claim is not closed. The grounded composite uses 25fps frame placement; the floor may be 24fps and is accepted as background skew, but scope visibility must match the 25fps composite, not the 24fps floor. Concrete fix: scope timeline must use only the delivery video’s CFR fps and the integer segment plan from `plan_timeline_segments()`. Do not derive visibility from the floor render rate.

9. [INVARIANTS / “deterministic per seed”] Grounding shows `_CRTRenderer.render()` uses `np.random.randint(...)` directly when `vol > 0.3`, with no per-frame seed. That violates the stated determinism invariant for the unchanged floor path. Concrete fix: replace global RNG usage with a deterministic per-frame RNG seeded from episode seed + frame index, or remove the random noise path. If v2 scopes use any randomization, apply the same rule there.

10. [Open architecture questions / Q4 engine_id -> aspect] Aspect resolution is still an open question, so scene awareness is underspecified. Concrete fix: define a deterministic resolver before build: for each `segment["source"] == "clip"`, classify portrait/landscape by probing the actual manifest `path` native dimensions when available; fall back to a versioned engine registry mapping only when probing fails; unknown engines must default to suppress, not draw. [ASSUMPTION] This assumes clip paths still point to pre-composite assets with meaningful native dimensions; verify that generated paths are not already pillarboxed canvas outputs.

SHOULD-FIX:
1. [Proposed v2 design / “New node placed AFTER the upscale/blend”] The node needs a clear output ordering relative to `OTR_MasterAudioMux`. Concrete fix: explicitly specify pipeline as `SilentComposite -> RTXUpscale -> PostUpscaleProcgenBlend -> SceneAwareScopes -> MasterAudioMux`.

2. [Open architecture questions / Q1] The document still asks “NEW node vs EXTEND” while also proposing a new node. Concrete fix: choose one. Smallest safe choice is a new node, because the grounded procgen blend is currently pure ffmpeg and does not consume manifests or audio analysis.

3. [Open architecture questions / Q5 “Two green layers”] “Both screen green-only -> additive, should be safe” is too loose. Concrete fix: define exact compositing math and pixel format for the late pass. If using ffmpeg, mirror the grounded `format=gbrp -> blend -> format=yuv420p` approach; if using PIL, implement screen/add with explicit RGB-channel math and zero R/B before merge.

4. [Proposed v2 design / “Pillow-only/CPU”] A PIL node that decodes, draws, and re-encodes full 1920x1080 video must specify codec, CFR, pix_fmt, color range/matrix, and audio mapping to avoid changing downstream assumptions. Concrete fix: match the existing video contract: bt709/yuv420p/CFR, delivery fps, video stream length unchanged.

5. [Open architecture questions / Q3 beat-boundary flicker] The pop at portrait/landscape seams is unresolved. Concrete fix: either explicitly accept hard cuts for v2 or implement a small frame-domain fade mask, e.g. 3-5 frames fade in/out based on adjacent segment eligibility. Do not leave behavior undefined.

6. [Grounded `_analyze_audio`] `_analyze_audio()` uses `spf = sample_rate // fps`. This is exact for common 44.1k/48k at 25fps, but not generally frame-accurate for arbitrary fps/sample rates. Concrete fix: for the new node, prefer chunk boundaries `round(fi * sr / fps)` to `round((fi+1) * sr / fps)` if you need general delivery-fps correctness.

7. [Proposed v2 design / “master audio availability”] “The frozen master WAV is in the episode folder” is not a resolved API. Concrete fix: add an explicit `audio` input to the node and require the workflow to wire the same frozen master audio object/path used by `OTR_MasterAudioMux`, rather than path-discovery inside the node.

OPTIONAL / NICE-TO-HAVE:
- Emit a debug JSON/report listing each segment’s frame range, source, engine_id, resolved aspect, and scope eligibility.
- Add a visual debug mode that draws translucent gutter bounds for portrait segments only.
- Cache audio analysis keyed by audio hash + fps + frame count if repeated builds are common.

CUT THESE (over-engineering):
1. [Open architecture questions / Q3] Cut crossfade/hold for the first build unless hard seam pops are unacceptable in review. Integer-accurate on/off visibility is enough to prove scene awareness.

2. [Proposed v2 design / “Landscape -> suppress the scopes (or draw an ultra-dim edge variant)”] Cut the ultra-dim edge variant. It weakens the requirement “landscape b-roll does not have gutters” and adds another scene-state mode without solving a core bug.

3. [Open architecture questions / Q1] Cut extending `OTR_PostUpscaleProcgenBlend` for now. It is grounded as a pure-ffmpeg green blend node with no manifest/audio-analysis inputs; forcing scene-aware PIL drawing into it increases blast radius. A separate node is the smaller, safer change.

4. [Open architecture questions / Q6] Cut maintaining two user-facing scope paths long-term. Keep a temporary compatibility flag if needed, but the target workflow should hard-switch floor scopes off and late scene-aware scopes on to avoid double-render regressions.