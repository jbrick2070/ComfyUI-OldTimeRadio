# Pass 02 Judgment - Coding Plan

## Verdict

Proceed, with corrections.

R2 converged on a small Seedance-only code change. The useful change is not a
new cloud input, model knob, workflow widget, or downstream timeline rewrite.
It is a prompt-conditioning helper called at the Seedance partner-input
boundary, plus focused tests and modest logs.

## Accepted

- Put the conditioner in `nodes/_otr_video_engines/eng_cloud_video.py`.
- Call it from `CloudSeedance2Engine._partner_inputs()` immediately before
  assigning `model["prompt"]`.
- Leave `render_driver.py` alone. Prompt conditioning must not affect
  Wan/Kling/Pixverse.
- Use `_text_prompt_input(request)` first so missing or blank prompts still fail
  loud through the existing required-prompt path.
- Use a pure helper:

  ```python
  def _condition_seedance_prompt(prompt: str) -> tuple[str, dict]:
      ...
  ```

- Use case-insensitive regex replacements for risky motion language.
- Add wording that makes provider-minimum over-rendering safe under head-trim:
  motion must begin immediately in the first frame and stay gentle.
- Do not send unsupported Seedance fields. No temperature, CFG, sampler,
  camera-fixed, motion strength, optical flow, or reference-video fields in this
  pass.
- Keep duration behavior as-is: derive from beat frames, clamp to Seedance
  `4..15s`, render the provider minimum when the beat is shorter, and let the
  silent assembler trim to the beat.

## Rejected Or Deferred

- Do not build a cross-node observability side channel. It would require a
  workflow JSON/data-flow change and is not needed for this fix.
- Do not add `reference_videos`, `auto_downscale`, or `auto_upscale`. Reference
  video guidance is a later, costlier option if prompt conditioning fails.
- Do not change `OTR_CLOUD_SEEDANCE_RATIO`.
- Do not add a workflow widget or edit
  `workflows/otr_scifi_16gb_full.json`; this pass changes no node schema.
- Do not use a global duration override as the smoothness fix.

## Corrections To R1

- `timeline_quality_report()` exists in `nodes/otr_silent_composite.py`, but it
  is downstream and already local to silent composite assembly. It should not be
  mixed into Seedance adapter metadata in this pass.
- The current assembler trims long clips by taking the needed head frames; it
  does not speed-compress a 4s provider clip into a 2s beat. This supports the
  operator proposal: for beats below a cloud provider minimum, render the
  minimum valid duration and chop to the audio beat.
- The prompt should say "Preserve the reference-image composition and framing"
  instead of hard-coding "16:9", because the adapter can send `ratio=adaptive`.

## R2 Spend

- Pass 01 spend: about `$0.1132`
- Pass 02 spend: about `$0.1296`
- Running total: about `$0.2428`
