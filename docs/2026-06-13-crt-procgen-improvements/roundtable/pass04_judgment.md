# pass04 judgment -- broaden-panel CONVERGENCE pass (5 fresh frontier families)

Panel: gpt-5.5 (kept) + grok-4.3 + kimi-k2.6 + mistral-large-2512 + qwen3-235b-a22b
(4 of 5 NEVER used in rounds 1-3). ~$0.085 this pass (cumulative ~$0.48). Reviewed the
round-3-hardened pass03_plan.md.

## VERDICT: CONVERGED.
All 5 fresh families independently RE-DERIVED the same architecture already in the plan
(the constructor refactor, EMA-precompute-in-__init__, gutter-rect masked layers,
text-after-vignette, half-open intervals, stable-hash RNG). A different-family panel
arriving at the plan's own specs = strong validation, not new design. The findings that
remained are fine IMPLEMENTATION PRECISION (shrinking each round) -- folded below. A 5th
round would only polish.

## NEW MATERIAL items folded into §4C (grounded)
- **Geometry COMPUTE from w,h, not 1920-hardcoded** (gpt + qwen). The renderer also runs
  1280x720 / 832x480 / 4K; baked 1920 constants would break. Folded the `portrait_w =
  round(h*480/832)` -> gutter/centre/radius formulas.
- **Hero title is CENTRE-anchored + EXEMPT from center-sanctity during its window**
  (kimi, best catch). A 2-3x title cannot be both hero-scale AND gutter-clamped; it owns
  the evacuated centre at the open (no portrait yet), then DOCKS. The clamp applies to the
  scopes + the docked ident only. Resolved a real contradiction in the prior draft.
- **`volume`/`freqs` are Python LISTS -> `np.array` in __init__** before the EMA
  precompute / vectorized lookback (mistral). Would crash `np.zeros_like(list)`.
- **The base->vignette->text layering refactor lands in S1** (kimi), not S4 -- S3 depends
  on it.
- **Suppress the section-1 ident ELEMENTS, keep the layout/coords** (qwen) -- they are the
  dock target.
- **dock_frames bounded** `min(fps*0.5, first_dialogue_f - music_open_end_f)` (mistral,
  kimi); **rng.integers endpoint=False** keeps the half-open range (kimi); `salt` defined
  as a literal effect tag; empty-`volume` guard.
- **Signal-driven vignette choke CUT from v1** (gpt) -- it multiplies the whole frame =
  the exact v1.5.1 readable-text risk; per-element brightness is enough. Floored formula
  kept in CUTS if ever re-added.
- VERIFY-AT-BUILD sharpened: confirm `start_s` is in SECONDS (not 25fps frames) + the exact
  opening-music `speaker_role` string.

## DISCARDED (invalid / misread -- the grounding step doing its job)
- **grok: "cut the dual-EMA, just use the dormant scalar + volume[fi]"** -- WRONG: the
  scalar EMA is stateful/updated-per-frame, which is the exact determinism/out-of-order
  bug round 2 fixed by precomputing. The dual arrays are pure. Keep.
- **kimi: "cut the 2-beat gap-fill smoke effect"** -- MISREAD: it is a verification TEST
  (title card stays at the open), not a visual effect. Keep as a test.
- mistral/grok: "drop precomputed graticules / draw per-frame" -- minor style; keep the
  precompute (cleaner, cheap). Non-material.

## STILL OPEN (unchanged -- the operator's call)
Landscape-beat gutters. gpt pushed hardest to either plumb per-beat suppression or cut
the scopes on landscape rather than ship "faint edge telemetry." Per-beat gating is still
infeasible at procgen render time (floor renders before clips exist). Logged as the
operator decision; v1 default = dim clamped edge scopes.

## Convergence trend (why we stop here)
Round 2 = structural wiring bugs (np.roll, hash, vignette-multiply). Round 3 = interface
precision (signature, intervals, geometry numbers). Round 4 (fresh families) = generalize
geometry + 1 title-anchor contradiction + list->array. Findings shrank each round and a
fresh 5-family panel found no new DESIGN issue. CONVERGED.
