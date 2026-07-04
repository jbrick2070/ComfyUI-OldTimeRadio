# R1 Final: Phase-2 Prompt Meat Arc

Status: Codex-grounded R1. Corrected external reviewer pass was attempted;
Antigravity and Claude file handoffs did not return usable output on the
corrected prompt in time. The prior Antigravity claim to hide/cut public-domain
lanes is rejected as contrary to the operator rule.

## Verdict

Yes with fixes. The concept is coherent only if exposed choices are real,
loud, and testable. Hidden fallbacks are the main architecture risk.

## Accepted Direction

- Keep one shared downstream production ledger.
- First phase keeps the broad current many-pass story architecture.
- Story/source models become cloneable prompt/content packs.
- Source brains differ by lane: science news, media archive/RSS, public-domain
  folder, custom schema.
- Expose planned source/story/pipeline choices instead of hiding them.
- If an exposed lane cannot run, it fails loudly with source/model/pipeline
  names. It does not downgrade to science RSS, custom premise, or sci-fi style
  picker.
- Add `simple_4_prompt_experimental` as a visible experiment:
  story -> ledger fill -> schema repair -> final ledger consistency audit.

## Must Fix

1. Treat public domain as visible but fail-loud until real.
2. Separate source bank, story model, story pipeline, and visual style.
3. Do not clone the whole downstream workflow for each story model.
4. Make story packs carry full prompt bodies/examples/rubrics where useful,
   instead of scattering everything into tiny string variables.
5. Extract all science/news/sci-fi wording from prompt sites into packs or
   science-only paths.

## Cut

- Do not build the lean/custom pipeline runtime in first transplant.
- Do not perform visual render-driver deep fallback surgery in the source/story
  transplant.
- Do not edit `workflows/otr_scifi_16gb_full.json` until transplant gates pass.
