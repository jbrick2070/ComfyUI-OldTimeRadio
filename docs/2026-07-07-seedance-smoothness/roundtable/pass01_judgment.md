# Pass 01 Judgment - Seedance Smoothness

Date: 2026-07-07

## Panel

- OpenAI: `openai/gpt-5.5-20260423` (partial output; visible claim accepted only where grounded)
- Gemini: `google/gemini-3.1-pro-preview-20260219` (partial output; visible claim accepted only where grounded)
- Tencent Hy3: `tencent/hy3-20260706:free` (retry with hidden reasoning disabled)

Actual spend for pass 01: about USD 0.1132.

## Codex Grounding Verdict

The panel converges on a better root-cause frame than the initial plan:

1. Prompt conditioning is probably useful because current sci-fi radio motion text contains high-energy verbs like `whip-pans`, `aggressively`, and `dynamic dolly`.
2. It is not enough to tune prompt text in isolation. Seedance 2 Reference has a hard 4-second minimum duration, while OTR beats can be shorter or longer, and downstream `OTR_SilentComposite` trims or loop-fills clips to exact beat frame counts.
3. The active Seedance 2 Reference node does not expose temperature, CFG, sampler, camera-fixed, camera-smoothing, or optical-flow knobs.
4. "More inputs" is real only in a narrow sense: the active node supports more reference images, reference videos, reference audios, and asset ids. Reference video is a possible guide-motion lever, but it adds cost and a new source clip dependency, so it is not the first fix.

Correction to the partial GPT/Gemini wording: the current OTR assembly path does not
speed-squash a 4-second cloud clip into a shorter beat. It trims the needed head
frames. That makes provider-minimum over-rendering acceptable, provided prompts do
not defer the main motion to the end of the generated clip.

## Accepted

- Add a Seedance-only prompt-conditioning layer, but make the integration point explicit and testable.
- Measure raw provider output versus canonicalized output versus final delivered timeline frames.
- Preserve the "render provider minimum, trim to audio target" policy for under-minimum cloud beats.
- Keep aspect wide; do not change `OTR_CLOUD_SEEDANCE_RATIO` as part of prompt conditioning.
- Cut speculative audio preconditioning until A/B evidence shows audio-reactive jitter.
- Treat seed comparisons as approximate because the partner node says results are non-deterministic regardless of seed.

## Rejected Or Deferred

- Do not invent provider fields such as temperature, CFG, camera speed, or motion strength.
- Do not switch to a reference-video workflow until prompt/conform evidence says we need it.
- Do not switch to `ByteDance2FirstLastFrameNode` for the audio-reactive lane yet; it is a different partner path and does not preserve the same reference-audio shape.
- Do not rely on operator eyeballing alone; record enough metadata to know whether the issue is provider motion, canonicalization, or assembly.

## Next Plan Delta

Pass 02 should produce a buildable coding plan for:

- a pure Seedance prompt conditioner;
- additive observability for prompt variant, raw/canonical/target frame counts, and delivered timeline status;
- a tiny A/B harness or manifest audit that can be run on real outputs;
- no workflow JSON change unless a new visible widget is introduced.
