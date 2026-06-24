# R3 anchor review (Claude, code-grounded) — wiring / integration / sequencing

Focus: does the build wire into the real pipeline order without breaking it?

## VERDICT

One sequencing fix makes it coherent; otherwise the wiring holds.

## MUST-FIX

1. **Selector runs INSIDE `generate_outline`, post-macro, not in the writer.**
   CONFIRMED: the macro premise is produced by Stage 1 INSIDE `generate_outline`
   (`_MACRO_SYSTEM_PROMPT`); the writer has no premise before calling it. So
   `select_style(premise, meta, cast_seed)` + ending_tag/template resolution must
   happen inside `generate_outline` AFTER the macro returns and BEFORE the Stage-3
   beat prompts — not pre-filled by the writer. Revise §C: the writer threads only
   `enable_style_grammar` + `cast_seed` onto the request; `generate_outline`
   resolves the style + ending internally and stamps telemetry. Keeps the whole
   lever inside one function — cleaner + easier to gate byte-identical.

2. **`cast_seed` must reach `generate_outline`.** CONFIRMED: `OutlineRequest`
   currently carries no seed (only diversity_hint/prior_critique/script_brief).
   Add `cast_seed: int = 0` to the frozen request; the writer already has
   `cast_seed` (resolved at L2730) and builds the request — pass it. Default 0 =>
   selector still deterministic; flag OFF => unused => byte-identical.

3. **Keep the early style label AND the late ending tag distinct.** CONFIRMED:
   `meta.style` is set early (pre-outline) and feeds back-compat surfaces (HUD,
   FLUX prompt, info card). Do NOT repurpose it. When the flag is ON, the
   deterministic `select_style` result can OVERRIDE `meta.style` (so downstream
   visuals match), but that is a behavior change — gate it; OFF keeps the existing
   early picker value untouched.

## SHOULD-FIX

4. **Workflow JSON / widget.** If a `pitch_room`-style env-only gate is used
   (`OTR_ENABLE_STYLE_GRAMMAR`), NO widget => NO JSON change (lowest risk, matches
   the T1/T2 pattern). If a node widget is added, it must land in
   `otr_scifi_16gb_full.json` in the SAME commit + re-validate. Recommend env-only
   first.

5. **Build/commit sequencing.** Order: (1) catalog data contract + self-check
   (pure, no behavior). (2) OutlineRequest fields + selector (dark). (3) injection
   + announcer gate + telemetry (dark). (4) writer flag plumbing. (5) L1/L2
   bundle-on docs. Each chunk suite + Bug Bible green; byte-identical asserted;
   commit+push. The live A/B is the last, GPU step.

## UNVERIFIABLE (verify-at-build)

- Exact phase-skeleton shape for the final-beat coordinate — confirm against the
  live `generate_outline` Stage-2/3 loop when building.
- That overriding `meta.style` on flag-ON doesn't desync a downstream consumer.
