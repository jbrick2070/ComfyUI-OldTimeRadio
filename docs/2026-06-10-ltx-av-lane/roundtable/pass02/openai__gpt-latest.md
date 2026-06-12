<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no — input/output contracts still leave coder guesses, and grounding shows concrete schema/canvas/timing contradictions.

MUST-FIX BEFORE BUILD:

1. `nodes/_otr_video_engines/eng_ltx_av.py` — request extraction must be specified exactly. `schemas.py` defines `VideoRequest.audio_ref: Optional[AudioRef]`, `AudioRef.path`, and `VideoRequest.asset_refs: dict[str,str]`; `_present_input_tokens()` only treats `"init_image"` as present when it is in `asset_refs`, not `conditioning_refs`. Fix: shared core must copy HuMo’s precedent:
   - `audio_path = _ref_path(get("audio_ref"))`, where `_ref_path` accepts bare string, dict `{"path": ...}`, or object `.path` (`eng_humo.py _ref_path`).
   - `init_image = (request.asset_refs or {}).get("init_image", "")` (`eng_humo.py _init_image_ref`).
   - Never look for `asset_refs["audio_ref"]`; never use `conditioning_refs` for required `init_image`.
   Add unit tests for dict request and Pydantic-shaped `AudioRef(path=...)`.

2. `render_driver.py` / `schemas.py` — duration field is currently schema-invalid. `schemas.Timing` has `target_duration_s`, `target_frame_count`, and `start_s`; it does not have `dur_s`, and all models forbid extras. But `render_driver.build_request_from_shot()` writes `req["timing"]["dur_s"]`. Fix: request assembly must populate `timing.target_duration_s` from line/shot `dur_s` and stop requiring `timing.dur_s` in new adapter code. If legacy dicts must be tolerated, the adapter duration resolver may read `dur_s` only as a fallback, but the canonical contract must be `Timing.target_duration_s`.

3. `nodes/_otr_video_engines/eng_ltx_av.py` — audio slice assumptions must be narrowed. `render_driver._slice_master_audio()` only guarantees the fallback sliced-master output is WAV PCM s16le, 44100 Hz, mono via `ffmpeg ... -c:a pcm_s16le -ar 44100 -ac 1`. `_voice_audio_for_line()` may return existing `audio_wav_path`, `wav_path`, arbitrary `*wav_path`, `music_wav_path`, `clip_path`, or `video_clip_path` without normalization. Fix: plan must say either:
   - adapter normalizes every incoming `audio_ref` to temp WAV PCM s16le/44100/mono before staging, or
   - render_driver normalizes before constructing `AudioRef`.
   Also mark VERIFY-AT-BUILD for LTX-AV node accepted sample rate/channel/layout.

4. `nodes/_otr_video_engines/eng_ltx_av.py` — frame derivation must not copy `eng_ltx_video.py`’s floor snap. `eng_ltx_video.render_clip()` currently does `((length - 1) // 8) * 8 + 1`, which can make video shorter than the requested beat. For LTX-AV, fix formula:
   - `duration_s = max(timing.target_duration_s, timing.target_frame_count / fps if target_frame_count else 0, ffprobe_audio_duration_s if used)`
   - `needed = ceil(duration_s * 25)`
   - `frames = 8 * ceil((max(1, needed) - 1) / 8) + 1`
   - then call `assert_ltx_dims(width, height, frames)`.
   This ensures the clip covers the audio slice.

5. `nodes/_otr_video_engines/eng_ltx_av.py` — >20s policy is not buildable as written. Claims ledger says “>20s beat clamp policy” is still unverified. Fix: define native cap after M0. If LTX-2.3 hard ceiling is 20.0s at 25fps, the largest valid `8n+1` not exceeding 500 frames is 497; if M0 proves 505 frames is accepted, use 505. For v1, choose one explicit policy:
   - recommended: render native LTX segment up to verified cap, then append silent video-only freeze/Ken Burns tail to the full `final_frames = next_8n1(ceil(full_duration_s*25))`, with LOUD log.
   - do not silently truncate visible video below audio duration.
   Mark cap value VERIFY-AT-BUILD.

6. `nodes/_otr_video_engines/eng_ltx_av.py` — canonicalize must enforce V-1 with an actual stream check. `schemas.CanonicalClip` requires `has_audio=False`, `pixel_format="yuv420p"`, bt709 fields, fps, and frame_count. `wrapper_bridge._bt709_encode_args()` emits `-an`, but if LTX-AV node returns a joint video+audio file, audio must be dropped in `eng_ltx_av.canonicalize()` before returning. Fix:
   - if raw is IMAGE frames: encode through `wrapper_bridge.encode_frames_to_silent_mp4()` (`-an`).
   - if raw is MP4/video file: create canonical temp MP4 with `-map 0:v:0 -an`, normalized yuv420p/bt709/fps.
   - run ffprobe in canonicalize and fail if audio stream count != 0.
   Required proof: `ffprobe -select_streams a ...` returns zero streams. Also add a test that feeds a fake AV MP4 and asserts final stream count is video-only.

7. `render_driver.py` / `nodes/_otr_video_engines/eng_ltx_av.py` — landscape canvas plumbing currently misses the new engines. Grounding shows `build_request_from_shot()` only applies `OTR_VIDEO_LANDSCAPE_CANVAS` for `("ltx_video", "wan_i2v")`; otherwise `build_request()` defaults canvas to `(480,832)`. Fix: include `ltx_av_talk` and `ltx_av_music` in the landscape override, or have the adapter replace absent/portrait canvas with parsed `OTR_VIDEO_LANDSCAPE_CANVAS`. `ltx_av_talk` must use the full landscape frame, default `1472x832`, not HuMo’s portrait canvas.

8. `nodes/_otr_shared/av_dims.py` / `eng_ltx_av.py` — operator canvas overrides must fail loud, not round or silently fallback. Current `render_driver.py` parser silently falls back to `1472,832` on malformed env, and `eng_ltx_video.py` floors dims to `/32`. New LTX-AV path must not copy that. Fix: parse `OTR_VIDEO_LANDSCAPE_CANVAS` strictly as `WxH`; on malformed or non-/32 dims, raise via `assert_ltx_dims(width,height,frames)` naming nearest valid values. Example: `1450x832` must fail, not become `1440x832`.

9. `nodes/_otr_video_engines/eng_ltx_av.py` — init image size/aspect is not specified. Grounding only proves render_driver indexes a portrait path from `ledger["images"]["images"]`; it does not prove FLUX output dimensions. Synthetic request sets `init_w=480`, `init_h=832`, but that is not a real upstream guarantee. Fix: plan must say adapter either:
   - reads actual image dimensions and preprocesses/pads/crops to the LTX canvas using `motion_common.resolve_aspect_transform()` semantics, preserving uniform scale/no stretch, or
   - VERIFY-AT-BUILD that the IA2V node accepts arbitrary portrait input and performs equivalent pad/crop/fit.
   For build safety, make preprocessing the adapter’s job unless M0 proves wrapper behavior.

10. `role_compat.py` / `render_driver.py` — `ltx_av_music` cannot be selectable/requestable until `music_visual` supplies `audio_ref`. Grounding shows `ROLE_AVAILABLE_INPUTS[Role.MUSIC_VISUAL]` lacks `audio_ref`; an engine with `required_inputs=("text_prompt","audio_ref")` will be excluded by `engine_fits_role()`. Fix in same commit as registering `ltx_av_music`: add `"audio_ref"` to `MUSIC_VISUAL`, and verify `build_request_from_shot()` attaches music beat audio from `music_wav_path`/master slice.

SHOULD-CONSIDER:

1. `render_driver._slice_master_audio()` — cache key is `(start_s,dur_s,master_path)` only. If a later frozen master reuses the same path, stale temp slices can be reused. Consider including audio ledger hash / `AudioRef.content_hash` / file mtime+size in the slice key.

2. `eng_ltx_av.py` — add a single pure helper for duration/frame resolution and test these cases: exact 1.0s → 25 frames, 1.01s → 33 frames, 20.0s cap behavior, missing duration but target_frame_count present, audio longer than timing.

3. `eng_ltx_av.py` tests — include request-shape tests proving `conditioning_refs["init_image"]` is ignored for required-input satisfaction, matching `schemas.VideoRequest._present_input_tokens()`.

4. `canonicalize` tests — assert returned dict can instantiate `CanonicalClip` and that ffprobe agrees with `has_audio=False`, `pix_fmt=yuv420p`, fps 25, bt709 metadata.

5. Store normalized audio slices and preprocessed init images under an episode/session temp directory rather than global temp where possible, to simplify cleanup and reproducibility.

OPEN-QUESTIONS:

1. VERIFY-AT-BUILD: exact LTX-AV ComfyUI node classes, input names, and whether output is IMAGE frames, video file, audio tensor, or joint AV file.

2. VERIFY-AT-BUILD: LTX-2.3 actual temporal ceiling at 25fps — accepts 497, 500, 505, or time-based 20.0s?

3. VERIFY-AT-BUILD: LTX-AV audio node accepted audio format — WAV only? sample rate? mono/stereo? internal resampling?

4. VERIFY-AT-BUILD: real FLUX portrait output size/aspect from the image chain; grounding does not provide it.

5. VERIFY-AT-BUILD: whether IA2V conditioning requires init image pre-resized to canvas dimensions or accepts arbitrary source image with internal resize/pad.

6. VERIFY-AT-BUILD: whether per-line music fields used by `_voice_audio_for_line()` are always audio files; grounding allows `clip_path` / `video_clip_path`, which may be non-WAV or contain video.