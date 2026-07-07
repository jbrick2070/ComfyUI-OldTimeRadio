# Codex Anchor - Pass 02 Coding Plan

## Grounded Position

The first code pass should not add new Seedance provider knobs. The active
`ByteDance2ReferenceNode` does not expose temperature, CFG, camera-fixed, camera
smoothing, optical flow, or sampler controls.

It should also not change the under-minimum duration policy. For cloud models, rendering
the provider minimum and trimming the front portion to the audio-derived beat is acceptable
and already matches how `OTR_SilentComposite` works. The real coding risk is that prompts
currently request late or abrupt motion that may not read well when only the head of a
minimum-length cloud clip is used.

## Preferred Implementation Shape

1. Add a pure Seedance prompt-conditioning helper near `CloudSeedance2Engine` in
   `nodes/_otr_video_engines/eng_cloud_video.py`.

   Rationale: this keeps the behavior scoped to the one cloud engine whose schema was
   audited, avoids workflow JSON/widget changes, and avoids affecting local LTX/Wan/HuMo
   prompts.

2. In `CloudSeedance2Engine._partner_inputs()`, send the conditioned prompt instead of
   raw `_text_prompt_input(request)`.

3. The helper should:

   - soften high-energy camera verbs;
   - append a smoothness clause;
   - preserve wide composition language;
   - say motion begins immediately and remains gentle throughout, so head-trimmed provider-minimum clips still look intentional;
   - be idempotent.

4. Add additive observability without changing workflow JSON:

   - `seedance_prompt_conditioned`
   - `seedance_prompt_variant`
   - `seedance_duration_requested_s`
   - `seedance_duration_policy` (`exact` or `provider_min_head_trim`)
   - canonical cloud `actual_duration_s` and `frame_count` already return in the clip dict; carry any missing fields into manifest rows only if tests prove they are absent.

5. Add tests for:

   - prompt conditioning softens `whip-pans` / `aggressively` / `dynamic dolly`;
   - prompt conditioning is idempotent;
   - `_partner_inputs()` requests provider minimum duration for a sub-4s Seedance beat and records `provider_min_head_trim`;
   - no unsupported Seedance fields are emitted.

## Not First-Pass Work

- Do not wire reference videos.
- Do not switch to `ByteDance2FirstLastFrameNode`.
- Do not change `OTR_CLOUD_SEEDANCE_RATIO`.
- Do not add a workflow widget unless the operator explicitly wants a visible toggle.
- Do not modify `OTR_SilentComposite` timing unless a real output audit shows it is causing a problem.
