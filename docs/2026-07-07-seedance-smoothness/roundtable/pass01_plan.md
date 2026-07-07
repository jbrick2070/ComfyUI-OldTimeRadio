# Pass 01 Plan - Seedance Smoothness

## Problem

The current `cloud_seedance_2` path now ingests the required still and audio. The remaining symptom is that Seedance camera movement looks slightly laggy or not buttery smooth.

This is a motion-quality issue, not the prior `init_image` request-shape failure.

## Grounded Facts

Current adapter:

- `nodes/_otr_video_engines/eng_cloud_video.py::CloudSeedance2Engine`
- active partner row: `ByteDance2ReferenceNode`
- required OTR inputs: `init_image`, `audio_ref`, `text_prompt`
- sent model fields: `model`, `prompt`, `resolution`, `ratio`, `duration`, `generate_audio=False`, `reference_images.image_1`, `reference_audios.audio_1`
- sent top-level fields: `seed`, `watermark=False`

Actual Seedance 2 Reference schema:

- supports reference images, videos, audios, and asset ids;
- supports `auto_downscale`/`auto_upscale` only for reference videos;
- duration is integer 4..15 seconds;
- seed is documented by the installed node as non-deterministic;
- no temperature, CFG, sampler, motion strength, optical flow, camera smoothing, or camera-fixed input.

Downstream:

- `canonicalize_video()` strips provider audio, pads/scales, enforces fps, and records output duration.
- `OTR_SilentComposite` later assembles each clip to exactly `target_frame_count`, truncating long clips and loop-filling short non-face clips.
- Long cloud clips are not speed-compressed into shorter beats; the assembler takes exactly the needed head frames.
- Therefore a smoothness QA must distinguish provider motion from canonicalization/assembly cadence.

Active cloud duration specs:

- `cloud_seedance_2`: `4..15s`
- `cloud_wan_i2v`: `2..15s`
- `word_razzle` / Pixverse: `5s` or `8s`
- `cloud_kling_avatar`: follows the audio file; partner validation allows `2..300s`

Policy: if an audio-derived beat is shorter than the provider minimum, request the
minimum valid duration and trim to the beat. That is probably the correct cloud
behavior. The prompt conditioner must make the clip visually usable from frame 1,
with continuous gentle motion throughout, not a staged move that only pays off in
the unused tail.

Current style risk:

```json
"music_open": "Continuous shot, same console throughout. Dial whip-pans across frequencies. Tube filaments ignite from cold to white-hot. Speaker grille vibrates aggressively. Dynamic dolly push forward."
```

This asks for exactly the kind of quick camera/subject motion that can look jerky in short generated video.

## Buildable Direction

1. Add a pure Seedance-only prompt conditioner in the request-building path or immediately before partner inputs.

   Candidate output clause:

   ```text
   One continuous uncut shot. Smooth stabilized camera on a slow dolly with gentle ease-in/ease-out motion. Preserve the wide 16:9 reference-image composition. No whip pans, no handheld shake, no sudden reframing, no jump cuts, no rapid zooms. Gentle parallax only; all motion gradual and physically continuous.
   ```

2. Add a conservative verb softener for Seedance prompts only:

   - `whip-pans` -> `slowly sweeps`
   - `aggressively` -> `subtly`
   - `dynamic dolly push` -> `slow controlled dolly push`
   - `white-hot` -> `bright warm glow`
   - `handheld dolly` -> `stabilized dolly`

3. Add observability to make failures diagnosable:

   - original prompt hash or excerpt
   - conditioned prompt hash or excerpt
   - prompt variant name
   - requested provider duration and whether it exceeded beat duration due to provider minimum
   - requested Seedance duration
   - target frame count
   - canonical frame count/duration/fps
   - downstream delivered frame status from `timeline_quality_report`

4. Run a cheap A/B before broader code:

   - A: current prompt
   - B: conditioned prompt, same still/audio/model/resolution/ratio
   - C only if A/B is inconclusive: `Seedance 2.0` instead of `Seedance 2.0 Fast`
   - Optional D only if aspect artifacts appear: explicit `16:9` instead of `adaptive`

5. Defer reference-video guidance.

   Reference videos are real inputs, but they require an extra guide clip and trigger a different cost envelope. Use them only if prompt conditioning plus cadence audit fails.

## Open Questions For Pass 02

- Should the prompt conditioner live in `render_driver.py` where `text_prompt` is finalized, or in `CloudSeedance2Engine._text_prompt_input()` where only Seedance partner calls see it?
- What exact metadata should be added without changing workflow JSON?
- Should any cloud prompt conditioner explicitly say "motion begins immediately and remains gentle throughout" so provider-minimum over-rendering survives head-trim?
