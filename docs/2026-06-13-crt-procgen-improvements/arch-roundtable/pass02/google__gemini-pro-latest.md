<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The pipeline wiring assumes outputs that don't exist, the audio chunking contradicts the "no audio-spine touch" invariant, and the double-blend will cause color-shifting garbage without a format pin.

MUST-FIX BEFORE BUILD:
1. [Pipeline + wiring] `SilentComposite` does not output the manifest. Its `RETURN_TYPES` are `("STRING", "STRING")` (`silent_video_path`, `report`). You cannot route `SilentComposite -> SceneAwareScopes`. Fix: Route the manifest directly from the upstream generator to `SceneAwareScopes`, or add the manifest as a third output to `OTRSilentComposite`.
2. [Audio re-analysis] Contradiction and invariant violation. The plan dictates changing chunk boundaries to `round(fi*sr/fps)`, but also demands "same `_analyze_audio`" and the invariants state "no audio-spine touch". Furthermore, `sr//fps` IS exact for 25fps (e.g., 48000//25 = 1920). Fix: Abandon the `round()` logic. Use the exact existing `spf = sample_rate // fps` logic in `_analyze_audio` to strictly maintain the invariant and guarantee frame-identical output.
3. [Audio re-analysis] The plan requires `total_frames = ffprobe the SOURCE video's exact frame count`, but `OTR_SceneAwareScopes` does not take a video input (only `manifest`, `audio`, `canvas_w/h`, `fps`). Fix: Compute `total_frames` using the `cursor` returned by `plan_timeline_segments(manifest, ...)`, which perfectly matches the assembled video length without needing a video file to probe.
4. [`OTR_PostUpscaleProcgenBlend` EXTENDED] The double blend `[tmp][scopes]blend=screen[out]` will produce color-shifting garbage because `scopes_mp4` is `yuv420p`. The grounding explicitly warns that blending without pinning to planar RGB (`gbrp`) breaks the math. Fix: Apply `format=gbrp` to the scopes input before the second blend: `[2:v]format=gbrp[scopes_gbrp]; [tmp][scopes_gbrp]blend=screen,format=yuv420p[out]`.

SHOULD-FIX:
5. [`OTR_PostUpscaleProcgenBlend` EXTENDED] Audio mapping. Adding a third silent video input (`scopes_mp4`) to ffmpeg without explicit audio mapping risks ffmpeg dropping or confusing the audio stream. Fix: Explicitly add `-map 0:v -map 0:a? -c:a copy` to ensure the master audio from the source video is copied.
6. [`OTR_SignalLostVideo` (floor)] The plan says "add `draw_scopes: bool