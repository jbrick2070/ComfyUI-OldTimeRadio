# Seedance Smoothness Plan

## S0 Goal

Improve the perceived smoothness of the OTR `cloud_seedance_2` lane after the July 7 live run looked "a little laggy / not so smooth".

The specific symptom is not a missing still, not a black frame, and not a cloud auth failure. The symptom is: Seedance camera/object movement is not "buttery"; camera motion feels slightly laggy, choppy, or over-energetic.

The target is smoother motion without violating the current hard rules:

- no fallback engine swaps
- no black clips
- no wrong-shaped requests
- no portrait pillarbox for wide bookend / b-roll shots
- no invented Partner Node fields
- no workflow JSON drift unless a real widget/input change is needed

## S1 Current Grounded Behavior

`cloud_seedance_2` is registered as an `audio_conditioned_video` engine and currently requires:

- `init_image`
- `audio_ref`
- `text_prompt`

The adapter sends one Partner Node model dictionary containing:

- `model`
- `prompt`
- `resolution`
- `ratio`
- `duration`
- `generate_audio=False`
- `reference_images.image_1`
- `reference_audios.audio_1`

It also sends top-level:

- `seed`
- `watermark=False`

The adapter currently exposes environment overrides for:

- `OTR_CLOUD_SEEDANCE_MODEL`
- `OTR_CLOUD_SEEDANCE_RESOLUTION`
- `OTR_CLOUD_SEEDANCE_RATIO`
- `OTR_CLOUD_SEEDANCE_DURATION`

There is no grounded `temperature`, motion-strength, CFG, prompt-extend, sampler, or optical-flow knob in the checked code.

## S2 Latest Live Signals

The current log tail showed Seedance advancing:

- `2026-07-07 11:00:04`: `cloud_seedance_2` started `shot_b000_music_open`
- `2026-07-07 11:02:41`: it advanced to `shot_b001`

That means the recent still/init-image contract fix is working for this run. The smoothness problem is now about conditioning quality or provider output character, not missing inputs.

## S3 Likely Root Causes To Test

1. The prompt language may ask for too much energetic camera/object motion in a short clip.
   Example current sci-fi radio motion register:
   "Dial whip-pans across frequencies... Speaker grille vibrates aggressively. Dynamic dolly push forward."

2. Music/opening shots may be getting high-energy audio references, which can make the provider overreact when paired with aggressive motion verbs.

3. Duration rounding may ask Seedance for coarse provider duration buckets that do not line up cleanly with the downstream beat/frame target.

4. The default model is `Seedance 2.0 Fast`, which may favor speed over temporal polish. This is a hypothesis only; verify with an A/B.

5. Canonicalization conforms cloud output to the OTR canvas/fps after provider generation. If provider fps/duration differs from target, the conform step may contribute perceived cadence issues.

## S4 Candidate Build Plan

### S4.1 Prompt Conditioning

Add a Seedance-specific motion conditioner before `cloud_seedance_2` receives `text_prompt`.

For Seedance only, rewrite/append a compact stabilizer clause:

> continuous single take, stable smooth camera motion, no abrupt cuts, no jump cuts, no whip pans, no shaky camera, gentle parallax, consistent subject and background, preserve the reference image composition

Also soften high-energy style-pack motion words when the engine is Seedance:

- `whip-pans` -> `slowly sweeps`
- `vibrates aggressively` -> `subtle rhythmic vibration`
- `dynamic dolly push forward` -> `slow controlled dolly push`
- `tube filaments ignite from cold to white-hot` -> `tube filaments gradually warm and glow`

This should be engine-specific, not a global style-pack rewrite, because LTX and still-pan may benefit from punchier motion words.

### S4.2 Duration Discipline

Do not use a global `OTR_CLOUD_SEEDANCE_DURATION` as the first fix. It flattens all beats to one length and can desync visual clip intent from the beat budget.

Instead, audit actual provider output duration and frame count per clip:

- requested duration
- actual duration
- canonicalized fps
- canonicalized frame count
- target frame count

If the mismatch is large, add trace warnings first. Only then consider snapping Seedance durations to provider-friendly buckets.

### S4.3 Model A/B

A/B the same still/audio/prompt/seed across:

- `Seedance 2.0 Fast`
- `Seedance 2.0`

Acceptance:

- smoother camera continuity
- less stutter during fast audio peaks
- no material cost/runtime blow-up
- no resolution regression

Keep `Fast` as default until the A/B proves the full model is visibly smoother.

### S4.4 Audio Conditioning

Do not remove audio conditioning. Seedance requires `audio_ref` in this adapter and the current engine identity is "required_audio_ref".

If audio-reactive jitter is confirmed, add a bounded audio-preconditioning option for Seedance only:

- short fade in/out on the sliced beat WAV
- gentle peak normalization
- optional low-pass or transient softening

This must write a deterministic per-beat derivative, not mutate the frozen master.

### S4.5 Acceptance Test

Build a narrow A/B harness that reuses the same already-minted still and beat audio:

- A: current prompt
- B: Seedance-stabilized prompt
- same model
- same seed
- same duration
- same init image
- same audio slice

Then optionally:

- C: stabilized prompt + `Seedance 2.0`

Record results in the clip manifest / trace so the operator can eyeball A/B clips.

## S5 Non-Goals

- Do not add a fake `temperature` parameter unless the installed Partner Node schema exposes it.
- Do not touch `cloud_kling_lipsync`; that lane was nuked.
- Do not soften global visual styles unless Seedance-only conditioning proves too narrow.
- Do not silently switch existing workflow defaults.

## S6 Questions For Panel

1. What exact positive prompt wording is most likely to make Seedance camera movement smooth and continuous?
2. Should the fix be a Seedance-only prompt conditioner, a style-pack update, or both?
3. Is duration/frame mismatch likely enough to prioritize before prompt conditioning?
4. Is `Seedance 2.0` likely worth A/B testing against `Seedance 2.0 Fast` for smoothness?
5. Are there any real Partner Node fields this plan is missing, or should we avoid all unverified knobs?
