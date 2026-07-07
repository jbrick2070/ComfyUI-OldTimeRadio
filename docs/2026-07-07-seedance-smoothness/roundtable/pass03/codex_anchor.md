# Pass 03 Codex Anchor - Wiring Review

## Grounded Wiring Position

The proposed Seedance smoothness change does not add a node, widget, input,
output, link, or workflow-visible option. It edits only the value assigned to the
existing `model["prompt"]` field inside
`CloudSeedance2Engine._partner_inputs()`.

Therefore `workflows/otr_scifi_16gb_full.json` should not be edited in this
pass. Editing the workflow would be churn, not wiring.

## Existing Runtime Path

The live engine already reaches the current workflow through the existing cloud
video engine selection and request-shaping path. The specific Seedance adapter:

- `nodes/_otr_video_engines/eng_cloud_video.py::CloudSeedance2Engine`
- `name = "cloud_seedance_2"`
- `node_key = "cloud_seedance_2"`
- `family = "audio_conditioned_video"`
- `required_inputs = ("init_image", "audio_ref", "text_prompt")`

The previous code fix already made request shaping include the required
`init_image`, `audio_ref`, and `text_prompt` for scene-still Seedance renders.
This pass only changes the prompt text sent through that existing request.

## Inputs That Must Stay Unchanged

The adapter must continue to send exactly the supported installed Partner Node
shape:

- top-level: `seed`, `watermark`
- `model.model`
- `model.prompt`
- `model.resolution`
- `model.ratio`
- `model.duration`
- `model.generate_audio`
- `model.reference_images.image_1`
- `model.reference_audios.audio_1`

It must not add `reference_videos`, `auto_downscale`, `auto_upscale`,
temperature, CFG, sampler, camera-fixed, motion strength, or optical-flow
fields.

## Duration Wiring

No workflow change is needed for provider-minimum durations. The existing helper
derives seconds from `timing.target_frame_count / canvas.fps` and clamps
Seedance to `4..15s`. The downstream silent composite already trims the clip to
the beat frame count.

Under-minimum beats should continue to request 4s and trim to fit the audio.
The prompt conditioner must make those first frames visually useful.

## Test Wiring

The right regression location is `tests/test_cloud_video_adapters.py`, because
that file already tests:

- Seedance partner input shape
- Wan partner input shape
- Kling partner input shape
- request seed clamping
- cloud canonicalization

The tests should prove the prompt changes only for Seedance and that the partner
input key set remains unchanged.

## Open Wiring Question For Panel

Is there any hidden workflow JSON obligation here, or is this truly an internal
adapter behavior change?
