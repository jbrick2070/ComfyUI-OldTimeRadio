<!-- Claude code-grounded anchor review, R1 (expression / arc) -->
# Claude anchor review -- R1 (recipe-selector expression)

## VERDICT
SOUND, build-ready with 3 tightenings. "Recipe-follows-model + explicit
OTR_LTX_AV_RECIPE override + fail-loud-on-ambiguous" is the right shape and maps
cleanly onto the four existing `_sharp_enabled()` call sites. No architectural
rethink.

## Grounding (CONFIRMED against the real eng_ltx_av.py)
- `_sharp_enabled()` is consulted at EXACTLY 4 sites: `_weight_paths` (L246),
  `_node_candidates` (L348), `_build_graph` (L371), `render_clip` (L484).
  CONFIRMED -- a single `self._recipe()` resolver swapped in at those 4 sites is
  the whole refactor.
- The real filenames disambiguate cleanly: dev = `ltx-2.3-22b-dev-Q3_K_M.gguf`
  (token "dev", no "distilled"); distilled = `ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf`
  (token "distilled", no "dev"). CONFIRMED the substring rule resolves both.
- `assert_usable` is the fail-closed gate that already runs `_weight_paths`
  BEFORE any GPU forward (L252+). CONFIRMED -- resolving the recipe there makes an
  ambiguous unet raise at the gate, not mid-render.
- `_build_graph` already branches `use_i2v` internally (i2v character vs t2v
  music/announcer). CONFIRMED -- the recipe must NOT re-branch on role; the
  model-head/sigmas/sampler/cfg are role-INVARIANT, only i2v adds `strength`.
- `distilled_native` == `sharp_lora` minus the LoRA node + minus the LoRA weight
  requirement (model head -> unet directly). CONFIRMED this is exactly what the
  bakeoff distilled legs ran (no LoRA, distilled sigmas, euler_cfg_pp, cfg 1.0).

## MUST-FIX
1. **One resolver, read fresh each render, memo-free.** `self._recipe()` must read
   `OTR_LTX_AV_RECIPE` + `OTR_LTX_AV_UNET` at call time (env-driven flips between
   beats must take effect). Do NOT cache on the instance across renders.
2. **Resolve in `assert_usable` so ambiguous fails CLOSED before GPU.** The same
   resolver feeds `_weight_paths` (LoRA required only for sharp_lora) and the
   graph build. An ambiguous unet with no override raises
   `EngineUnusable(MALFORMED_CONFIG)` naming the unet + the fix string.
3. **Recipe drives ONE config object** (name + use_lora + use_modelsampling +
   sigmas_source + sampler + cfg + i2v_strength) consumed uniformly by all 4
   sites and all 3 roles. No role-specific recipe branching.

## SHOULD-FIX
- **Cleanbreak-delete `OTR_LTX_AV_SHARP`** (verified: nothing in repo/scripts sets
  it). If kept at all, only as a deprecation shim that RAISES "renamed to
  OTR_LTX_AV_RECIPE" -- not a silent =0->m0_base map (CLAUDE.md cleanbreak framing).
- **Detection must reject double-token names** (`...dev-distilled...`) -> ambiguous
  -> raise, never pick one silently. The "and not <other>" clause already does this;
  pin it with a test.
- Fail-loud message: name the unet, the detected/forced recipe, and the override
  knob, so the operator can self-serve.

## Open for the panel
Q1 substring-detect-vs-family-map robustness; Q2 ambiguous = raise vs loud-default;
Q3 m0_base override-only; Q4 OTR_LTX_AV_SHARP delete-vs-shim; Q5 role uniformity.
