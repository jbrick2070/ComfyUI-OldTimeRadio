<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan is missing critical input/output specifications: extraction pattern, audio format contract, frame derivation formula, canonicalize output contract, init image preprocessing, canvas dimension usage, and music_visual audio_ref supply.

MUST-FIX BEFORE BUILD:
1. [ARCHITECTURE / Input extraction] The plan does not define how the shared core extracts `audio_ref` and `init_image` paths from the `VideoRequest`. It must specify: use a helper mirroring `eng_humo._ref_path` for `audio_ref` (accepting `AudioRef` dict with `path` key or a bare string) and read `init_image` from `request.asset_refs['init_image']` (a string path). (See schemas.py `AudioRef.path` and `asset_refs` dict.)
2. [ARCHITECTURE / Audio slice contract] The plan does not state the audio format the adapter can rely on. The upstream `render_driver._slice_master_audio` produces PCM s16le, 44100 Hz, mono WAV. The plan must declare that the adapter expects this format; any resampling needed by LTX-2.3 is the adapter's responsibility. Add: "The `audio_ref` passed to the adapter is a mono 44100 Hz 16-bit PCM WAV (guaranteed by the render driver)."
3. [ARCHITECTURE / Frames derivation] The plan lacks the exact frame count formula. It must define: `frames = next_8n1(ceil(duration_s * 25))`, capped at 497 (20s ceiling). For beats >20s, the plan must specify a policy (e.g., cap at 497 and let the compositor handle the tail, or chunk into multiple clips). Currently undefined.
4. [ARCHITECTURE / Canonicalize output] The plan does not detail the canonicalize output contract. It must specify that `canonicalize` returns a dict matching `CanonicalClip` fields (`has_audio=False`, `bt709`, `yuv420p`, `frame_count`, etc.) as in `eng_humo._clip_from_raw`. Also, if the LTX node outputs audio, `canonicalize` must strip it (re-encode with `-an`) and verify via ffprobe that stream count == 1 (video only).
5. [ARCHITECTURE / Canvas dimensions] The plan does not explicitly state that the adapter must use the request's canvas dimensions (`w`, `h`) for output resolution, and that `ltx_av_talk` renders at the full landscape canvas (1472x832). Add: "The adapter reads `request.canvas.w/h`, validates with `av_dims`, and renders at those dimensions. `ltx_av_talk` uses the landscape canvas (default 1472x832)."
6. [ARCHITECTURE / Init image preprocessing] The plan does not specify how the init image is scaled to fit the canvas before LTX conditioning. It must state that the adapter uses the `aspect_policy` (default `"pad"`) and `motion_common.resolve_aspect_transform` to scale the init image to the canvas dimensions, then feeds the resized image to the LTX I2V node.
7. [Additive touch list / role_compat] The plan makes the `role_compat.py` edit (adding `audio_ref` to `MUSIC_VISUAL` supply) conditional on pass04. This edit is required for `ltx_av_music` to fit the music_visual role; it must be unconditional. Remove the condition.
8. [ARCHITECTURE / Music audio_ref supply] The plan does not specify how the `audio_ref` for music_visual beats is provided. The render driver must attach the per-beat master audio slice as `audio_ref` for music_visual beats (similar to announcer beats). The plan should state that the driver will do this (to be implemented in pass04).

SHOULD-FIX:
- The plan should require that `canonicalize` runs ffprobe to assert `has_audio=False` and correct color properties, as part of the test suite.
- The plan should define the behavior when `audio_ref` is missing for `ltx_av_talk` (e.g., fail with a clear error before render).

OPTIONAL / NICE-TO-HAVE: none.

CUT THESE: none.