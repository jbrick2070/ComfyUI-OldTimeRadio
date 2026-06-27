# Recipe-selector expression -- judgment (panel: Claude anchor + Gemini + Codex)

The live OpenRouter roundtable was swapped (operator preference) for a read-only
code-grounded panel: Gemini and Codex each reviewed the real eng_ltx_av.py + the
anchor plan; Claude wrote the anchor review, grounded both panels against the
code, and judged. Convergence was clean.

## CONVERGED / ACCEPTED (folded into the build)
- Recipe FOLLOWS THE MODEL via a single resolver self._recipe(); strict family
  prefixes (ltx-2.3-22b-distilled-1.1- / ltx-2.3-22b-dev-); explicit
  OTR_LTX_AV_RECIPE override (auto|sharp_lora|distilled_native|m0_base).
- HARD RAISE on ambiguous unet, bad override, or a retired OTR_LTX_AV_SHARP.
  Both panels grounded this against fallback_engine = None (NO FALLBACKS, L659)
  and the fail-closed assert_usable gate. CONFIRMED.
- m0_base is OVERRIDE-ONLY (never auto). CONFIRMED.
- OTR_LTX_AV_SHARP cleanbreak-DELETE (nothing in repo sets it) -- present-but-set
  RAISES with a "use OTR_LTX_AV_RECIPE" message. CONFIRMED.
- Uniformity across the 3 director slots holds: the recipe is role-invariant; only
  the use_i2v branch (init_image present) adds the strength widget. CONFIRMED.

## Gemini MUST-FIX (3) -- all CONFIRMED against the code, all folded in
1. keep set (render_clip): the binary `"lora" if sharp else "modelsampling"` keeps
   a node that does NOT exist under distilled_native -> _keep_set(terminal, rcfg)
   keeps only the head node that the recipe actually built ({unet, decode} for
   distilled_native).
2. sigmas injection (render_clip): classes.setdefault("sigmas", _SigmasFromValues)
   must fire for BOTH sharp_lora and distilled_native (both run the fixed distilled
   sigmas) -> gated on rcfg["manual_sigmas"], not on a LoRA bool.
3. _node_candidates: distilled_native adds NEITHER lora nor modelsampling/sched.
- Gemini's note that _retain_model_patchers safely skips missing nodes: CONFIRMED
  (L541 `if not out: continue`) -- no change needed.

## Implementation
A static _recipe_config(recipe) struct (use_lora / use_modelsampling /
manual_sigmas / sampler / cfg / i2v_strength) is consumed UNIFORMLY by all four
sites + the keep set, so no site uses a binary 'sharp' bool. distilled_native ==
sharp_lora minus the LoRA (verified by test_recipe_config_distilled_native_is_
sharp_minus_lora). 11 unit tests green.

## Wiring (CLAUDE.md S0)
LTX-AV unet + recipe are env/code-driven; otr_scifi_16gb_full.json has NO
UnetLoaderGGUF/LTX-AV nodes (the engine runs in-process via OTR_VideoRenderBatch +
wrapper_bridge). No widget to change; the validator contract test still passes.
