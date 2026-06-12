<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The input extraction paths are wrong, the frame math truncates video shorter than the audio, and the init image will crash the LTX node without explicit graph-level scaling.

MUST-FIX BEFORE BUILD:

1.  [REQUEST SIDE] **Invalid Input Extraction**: The plan claims `schemas.py` maps the request to `(audio_ref, init_image)`. This is false. `init_image` is not a top-level attribute on `VideoRequest`; it lives inside the `asset_refs` dict.
    *   **Fix**: The shared core must extract inputs exactly as: `init_image = request.get("asset_refs", {}).get("init_image", "")` and `audio_path = request.get("audio_ref", {}).get("path", "")` (mirroring `eng_humo.py` lines 304-315).

2.  [FRAMES DERIVATION] **Audio Truncation via Snap-Down Math**: The legacy `eng_ltx_video.py` formula `((length - 1) // 8) * 8 + 1` snaps *down* (e.g., 26 frames becomes 25). If the video is shorter than the audio slice, the muxer leaves frozen/black frames at the tail of the beat.
    *   **Fix**: The adapter must snap *up* to cover the audio. Use: `frames = ((target_frames + 6) // 8) * 8 + 1`.

3.  [FRAMES DERIVATION] **Missing >20s Beat Clamp**: The plan asks what to do for >20s beats but fails to define a policy. An un-capped 30s music beat will OOM the 14.5GB VRAM ceiling.
    *   **Fix**: Define a hard `MAX_FRAMES` constant (e.g., 257) in `eng_ltx_av.py`. Clamp the target frames before the 8n+1 math. The standard OTR policy applies: the compositor will automatically hold the last frame for the remainder of the beat.

4.  [INIT IMAGE] **Missing Tensor Padding**: `motion_common.resolve_aspect_transform` only calculates the math; it does not alter the image. `eng_humo.py` passes this math to a custom C++ wrapper (`WanHuMoImageToVideo`), but native LTX nodes expect the `IMAGE` tensor to exactly match the latent dimensions.
    *   **Fix**: The adapter's declarative `_build_graph` must explicitly wire ComfyUI core nodes (e.g., `ImageScale`, `ImagePadForOutpaint`) using the `aspect_plan` values BEFORE feeding the image to the LTX VAE/Conditioning node.

5.  [CANVAS] **Late Canvas Validation**: `av_dims.py` will raise an exception on non-/32 overrides (e.g., `OTR_VIDEO_LANDSCAPE_CANVAS=1920x1080`, since 1080 % 32 = 24). If checked only in `prepare`, it wastes the AS-3 GPU lease before failing.
    *   **Fix**: Call `av_dims.assert_ltx_dims` inside `assert_usable` so an invalid operator override fails closed on the CPU *before* acquiring the GPU lease.

SHOULD-FIX:

1.  [AUDIO SLICE CONTRACT] **Implicit Audio Format**: `render_driver.py` (`_slice_master_audio`) guarantees the slice is a 44.1kHz, mono, 16-bit PCM WAV.
    *   **Fix**: Add a VERIFY-AT-BUILD step to confirm the LTX audio encoder node accepts 44.1kHz mono WAV natively. If it requires 16kHz or 48kHz, the adapter graph must wire a ComfyUI audio resample node.
2.  [OUTPUT SIDE] **Audio Drop Mechanics**: The plan asks where audio is dropped.
    *   **Fix**: Do not allow the wrapper to output an MP4. The graph must terminate at `VAEDecode` (yielding an `IMAGE` batch) and pass it to `wrapper_bridge.encode_frames_to_silent_mp4()`. This strictly enforces the `has_audio=False` V-1 invariant without needing new `ffmpeg` stripping commands.

[ASSUMPTION]:
*   Assuming the LTX-2.3 audio encoder node accepts standard ComfyUI `AUDIO` format (the output of the `LoadAudio` node used in `eng_humo.py`). If it requires a custom audio loader, the graph spec will need to reflect that.