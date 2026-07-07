# Pass 02 Plan - Seedance Prompt Conditioner

## Goal

Make Seedance cloud video motion smoother without changing request shape,
workflow JSON, or downstream timeline behavior.

## Implementation

Edit `nodes/_otr_video_engines/eng_cloud_video.py`.

Add a Seedance-only helper near the Seedance constants or immediately above
`CloudSeedance2Engine`:

```python
def _condition_seedance_prompt(prompt: str) -> tuple[str, dict]:
    ...
```

The helper should:

- accept a non-empty prompt that already passed `_text_prompt_input()`;
- apply a small explicit case-insensitive regex softener set;
- append one idempotent smooth-motion clause;
- return the conditioned prompt plus metadata useful for logging/tests.

Initial softener set:

- `whip-pan`, `whip-pans`, `whip pan`, `whip pans` -> `slowly sweeps`
- `aggressively` -> `subtly`
- `dynamic dolly push` -> `slow controlled dolly push`
- `handheld dolly` -> `stabilized dolly`
- standalone `handheld` -> `stabilized`
- `white-hot` or `white hot` -> `bright warm glow`
- `rapid zoom` / `rapid zooms` -> `slow controlled push`

Smooth-motion clause:

```text
One continuous uncut shot. Smooth stabilized camera on a slow dolly with gentle ease-in and ease-out. Motion begins immediately in the first frame and remains gentle and continuous throughout. Preserve the reference-image composition and framing. No whip pans, handheld shake, sudden reframing, jump cuts, or rapid zooms. Gentle parallax only; all motion gradual and physically continuous.
```

Idempotence rule: if the clause, or a stable marker sentence from it, is already
present, do not append it again.

In `CloudSeedance2Engine._partner_inputs()`:

1. Read `prompt = self._text_prompt_input(request)`.
2. Derive `conditioned_prompt, meta = _condition_seedance_prompt(prompt)`.
3. Compute `duration = self._duration_seconds(...)` once and assign it into the
   model dict.
4. Assign `model["prompt"] = conditioned_prompt`.
5. Log one bounded structured line through the module logger:
   - `engine=cloud_seedance_2`
   - `prompt_variant=seedance_smooth_v1`
   - `changed=<bool>`
   - `prompt_original_sha8=<sha>`
   - `prompt_conditioned_sha8=<sha>`
   - `prompt_original_excerpt=<first 160 chars>`
   - `prompt_conditioned_excerpt=<first 160 chars>`
   - `seedance_requested_duration_s=<int>`

The log is diagnostic only; do not add a new request field or workflow input.

## Duration Policy

Keep the current provider-minimum clamp.

For `cloud_seedance_2`, valid duration is `4..15s`. If a beat is shorter than
4 seconds, request 4 seconds from Seedance and let `OTR_SilentComposite` trim
the clip to the audio-derived `target_frame_count`.

This is better than forcing unsupported durations or trying to retime the video.
The prompt conditioner is what makes this safe: motion starts immediately, so
the kept head segment contains useful smooth movement.

Current active cloud duration specs:

- `cloud_seedance_2`: `4..15s`
- `cloud_wan_i2v`: `2..15s`
- `word_razzle` / Pixverse: `5s` or `8s`
- `cloud_kling_avatar`: follows `audio_ref`; partner validation allows `2..300s`

## Tests

Add focused tests in `tests/test_cloud_video_adapters.py`.

- `_condition_seedance_prompt()` softens grounded risky text from
  `music_open`: no surviving `whip-pans`, `aggressively`, or `white-hot`.
- The helper is idempotent.
- `CloudSeedance2Engine._partner_inputs()` sends the conditioned prompt and no
  unsupported Seedance fields.
- A short beat below 4 seconds still requests `duration == 4`.
- Wan/Kling/Pixverse prompt paths are unchanged by the Seedance helper.

## Manual QA

Run a tiny qualitative A/B, not a full episode:

- A: current prompt conditioning disabled only if a temporary local toggle is
  added for testing.
- B: conditioned prompt, same still/audio/model/resolution/ratio.
- C only if A/B is inconclusive:
  `OTR_CLOUD_SEEDANCE_MODEL="Seedance 2.0"` instead of the default
  `Seedance 2.0 Fast`, then unset it.

Because Seedance documents seed as non-deterministic, compare 2-3 samples per
variant before calling the result real.

## No Workflow JSON Change

This plan adds no node, widget, input, output, or wire. The real workflow
`workflows/otr_scifi_16gb_full.json` should remain unchanged for this pass.
