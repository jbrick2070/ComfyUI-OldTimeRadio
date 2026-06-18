# Roundtable pass01 judgment -- z_image params

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend ~$0.10. Converged in 1 pass
(strong agreement; remaining opens are all VERIFY-AT-BUILD, unresolvable without the installed node).

## ACCEPTED (CONFIRMED against the code/docs -> folded into pass01_plan.md)
- Stale sidecar: the stub gates `assert_usable` on `OTR_ZIMAGE_SIDECAR` + `render_image` raises
  NotImplementedError -> rewrite to in-process file-gated (verified in z_image_turbo.py). [3/3]
- Missing env vars OTR_ZIMAGE_UNET/CLIP/VAE + file-based fail-closed gate (mirror lumina). [3/3]
- Missing `reclaim_idle_models` in a finally block (flux + lumina both do it). [GPT, DeepSeek]
- `render_image(self, request, prepared=None)` signature (stub requires prepared). [GPT]
- Scheduler: AuraFlow graph uses `normal` (lumina), not `simple` -- draft was wrong. [GPT]
- Negative-prompt env var `OTR_ZIMAGE_NEGATIVE` + a concrete default (color/material only first). [3/3]
- Resolution: pick ONE rule = honor request dims exactly, env default when absent, NO snap/upscale. [3/3]
- Tier caveat: bf16 6B + Qwen3-4B TE fits the 16GB 5080 but not a true 8GB box; FP8/GGUF deferred. [GPT]
- Cold-import discipline + basename loader args. [GPT]

## ACCEPTED CUTS (over-engineering, all 3 agree)
- "Light post grade" out of the engine spec (raw PNG out; grade is post-pipeline). [3/3]
- Snap/upscale out of v1 (honor request dims). [3/3]
- Natural-language rewrite out of v1 default (reuse compose_still_prompt as-is); kept as a
  documented optional `OTR_ZIMAGE_NATURALIZE` lever only if the as-is A/B fails. [GPT cut vs DeepSeek
  "flag" -> resolved: documented future flag, not in v1 path.]
- Broad shift sweep 1-6 -> two fixed candidates (3.0, 6.0). [GPT]

## DOWNGRADED to VERIFY-AT-BUILD (UNVERIFIABLE from the repo; node/weights not installed)
- Exact `CLIPLoader` type string for Qwen3-4B (placeholder `<z-image/qwen>`). [3/3]
- Latent node `EmptySD3LatentImage` vs `EmptyLatentImage` (Z-Image VAE channel count). [Gemini]
- `ModelSamplingAuraFlow` correctness + shift for Z-Image (S3-DiT != guaranteed AuraFlow). [3/3]
- `UNETLoader weight_dtype` (default vs fp8/bf16). [DeepSeek]
These need a live `/object_info` on the installed Z-Image nodes -- the project's VERIFY-AT-BUILD rule.

## REJECTED / NONE
No panel claim was a misread of the code that needed rejecting; all were either real defects in the
DRAFT (the draft was a plan, not the existing code) or genuine verify-at-build unknowns. No conflicts
except the naturalize-now-vs-cut, resolved above.

## CONVERGENCE
One grounded pass produced a complete, consistent must-fix set with no surviving architectural
disagreement; the only opens are verify-at-build node specifics a second panel pass cannot resolve.
Stop. Final plan = pass01_plan.md.
