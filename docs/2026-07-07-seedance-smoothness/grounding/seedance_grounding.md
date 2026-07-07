# Seedance Grounding Excerpts

## Adapter Inputs

`nodes/_otr_video_engines/eng_cloud_video.py`

- `CloudSeedance2Engine.name = "cloud_seedance_2"`
- `family = "audio_conditioned_video"`
- `required_inputs = ("init_image", "audio_ref", "text_prompt")`
- `_partner_inputs()` sends:
  - `model.model`
  - `model.prompt`
  - `model.resolution`
  - `model.ratio`
  - `model.duration`
  - `model.generate_audio = False`
  - `model.reference_images.image_1`
  - `model.reference_audios.audio_1`
  - top-level `seed`
  - top-level `watermark = False`

There is no checked Seedance adapter field for temperature, CFG, sampler, motion strength, prompt extension, camera-smoothing, or optical-flow smoothing.

## Existing Env Knobs

`nodes/_otr_video_engines/eng_cloud_video.py`

- `OTR_CLOUD_SEEDANCE_MODEL`
- `OTR_CLOUD_SEEDANCE_RESOLUTION`
- `OTR_CLOUD_SEEDANCE_RATIO`
- `OTR_CLOUD_SEEDANCE_DURATION`

`nodes/_otr_shared/cloud_model_ids.py`

- default model selector for `cloud_seedance_2` is currently `Seedance 2.0 Fast`
- accepted aliases normalize provider ids to labels for `Seedance 2.0`, `Seedance 2.0 Fast`, and `Seedance 2.0 Mini`

## Duration Behavior

`nodes/_otr_video_engines/eng_cloud_video.py`

- `_duration_seconds()` reads `OTR_CLOUD_SEEDANCE_DURATION` if present.
- Otherwise it derives `round(timing.target_frame_count / canvas.fps)`.
- Seedance duration is clamped to `4..15` seconds.

## Current Prompt Source

`nodes/_otr_video_engines/render_driver.py`

- `build_request_from_shot()` chooses the final `req["text_prompt"]`.
- M4 creative prompts win first for character-bearing beats.
- For LTX/Wan/LTX-audio scene prompts, the driver can compose from style-pack motion registers or brief+beat.
- Seedance currently receives whatever final `req["text_prompt"]` exists; there is no Seedance-specific conditioning layer.

## Current Sci-Fi Radio Motion Register

`nodes/visual_styles/sci_fi_radio.json`

```json
{
  "announcer": "Continuous shot, same console throughout. Tuning dial needle sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille trembles. Dust motes drift. Slow handheld dolly forward.",
  "music_open": "Continuous shot, same console throughout. Dial whip-pans across frequencies. Tube filaments ignite from cold to white-hot. Speaker grille vibrates aggressively. Dynamic dolly push forward.",
  "music_close": "Continuous shot, same console throughout. Dial settles. Tube filaments cool from white through deep amber. Smoke trails from cooling tubes. Slow dolly pull back.",
  "music_inter": "Continuous shot, same console throughout. Dial steady, glowing. Oscilloscope dances to the rhythm. VU meters bounce. Tubes pulse with the bass. Slow orbit around the speaker."
}
```

## Latest Live Signal

`C:\Users\jeffr\Documents\ComfyUI\user\comfyui_8000.log`

- `2026-07-07 11:00:04`: `cloud_seedance_2` started rendering `shot_b000_music_open`.
- `2026-07-07 11:02:41`: it advanced to `shot_b001`.

The still/init-image contract appears satisfied in the current run. The user-visible issue is motion smoothness, not request shape.

## Proposed Review Focus

The useful answer should recommend:

- exact prompt language for smoother Seedance camera movement
- whether to place it as a Seedance-only prompt conditioner
- whether to soften the high-energy sci-fi radio motion verbs
- whether to test `Seedance 2.0` vs `Seedance 2.0 Fast`
- what to verify in an A/B, without inventing provider fields

## Partner Node Schema Fact Check

Installed ComfyUI Partner Node source:

`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_api_nodes\nodes_bytedance.py`

`ByteDance2ReferenceNode` is the current `cloud_seedance_2` target. Its dynamic combo model fields are:

- `prompt`
- `resolution`
- `ratio`
- `duration` (integer, 4..15 seconds)
- `generate_audio`
- `reference_images` (`image_1`..`image_9`)
- `reference_videos` (`video_1`..`video_3`)
- `reference_audios` (`audio_1`..`audio_3`)
- `auto_downscale` (reference videos only)
- `auto_upscale` (reference videos only)
- `reference_assets`

The submitted API request model is `Seedance2TaskCreationRequest` in
`comfy_api_nodes\apis\bytedance.py`:

- `model`
- `content`
- `generate_audio`
- `resolution`
- `ratio`
- `duration`
- `seed`
- `watermark`

There is no Seedance 2 Reference field for temperature, CFG, sampler, motion strength,
camera smoothing, optical flow, or camera fixed.

Older ByteDance Seedance 1.x nodes expose `camera_fixed` and append
`--camerafixed true/false` to the prompt. The current Seedance 2 Reference node does
not expose that input. `ByteDance2FirstLastFrameNode` exists and can force first/last
frames, but it is a different partner node path and does not carry the same multimodal
reference-audio shape as `ByteDance2ReferenceNode`.

## Downstream Conform Fact Check

`nodes/_otr_shared/cloud_media_canonical.py::canonicalize_video()` strips provider audio,
rescales/pads to the requested delivery canvas, enforces the requested fps with `fps=N`,
and returns the canonical output duration. It does not retime the provider clip to the
beat target.

`nodes/otr_silent_composite.py::assemble_silent_timeline()` is the later stage that
assembles each real clip to exactly `target_frame_count` frames. Long clips are truncated
to the beat; short non-face clips are loop-filled when `OTR_CLIP_FILL` is enabled.
Therefore a smoothness QA must compare:

- raw provider duration/frame count/fps
- canonical cloud duration/frame count/fps
- manifest `target_frame_count`
- delivered segment frames and whether the clip was truncated or loop-filled

Important correction: this is not time-compression. `_encode_segment()` applies
`-frames:v N` after the fps conform chain, so a cloud clip longer than the beat is
trimmed to its needed head frames rather than sped up to fit.

## Active Cloud Duration Specs

From current OTR adapters plus installed partner-node schemas:

- `cloud_seedance_2`: Seedance 2 Reference duration is integer `4..15` seconds.
- `cloud_wan_i2v`: Wan 2.7 I2V duration is integer `2..15` seconds.
- `word_razzle` / `cloud_pixverse_i2v`: Pixverse duration enum is `5` or `8` seconds; current adapter selects `5` unless the beat exceeds 5 seconds.
- `cloud_kling_avatar`: no explicit duration widget; generated video follows `sound_file`, validated by partner node as `2..300` seconds.

For under-minimum beat lengths, the expected policy is: request the provider's
minimum valid duration, use the front portion needed by the audio-derived beat, and
make prompts look good from frame 1 instead of saving important action for the end.
