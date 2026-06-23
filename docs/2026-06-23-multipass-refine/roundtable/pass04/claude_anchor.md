# Claude anchor review -- R4 (convergence / residual defects)

## VERDICT
yes (build-ready) -- with two verify-at-build checks, no new architecture-level must-fix.

The R3 defects are all resolved: pure scorer on RAW intents (no build_sq_data during scoring), cast_seed
as the deterministic base, explicit torch/random re-seed before each generate_outline, diversity_hint hook
on OutlineRequest, Comfy+OpenRouter both clamped, build_sq_data once on the winner, telemetry merged,
try/except per candidate + never-fail fallthrough, no re-validate.

## Residual (verify-at-build, not blocking)
1. **RNG re-seed efficacy is empirical.** Whether `torch.manual_seed`+`random.seed` before `generate_outline`
   actually diversifies a local-LLM (Ollama/llama-server) outline depends on the backend honoring sampling
   RNG; some HTTP backends ignore process-local seeds. If diversity is weak, the `diversity_hint` prompt
   overlay is the real lever and the seed is just the tie-break. Measure candidate-distinctness in step 5.
2. **Metric discrimination.** If `count_ungrounded_crisis` on raw intents is near-0 across candidates (the
   weak model rarely writes the literal generic nouns in the OUTLINE intents, only in the dialogue), the
   scorer won't discriminate at the outline layer -- which would argue for the v1 post-compose grade after
   all. The prerequisite soak + step-5 measurement decide this.

## Convergence call
CONVERGED. The plan is build-ready and explicitly gated behind a measurement prerequisite that can still
CUT it. No new must-fix.
