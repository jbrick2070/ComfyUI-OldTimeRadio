# Roundtable pass 02 -- judgment (coding sprint plan)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend $0.10. The panel hardened MEASURABILITY and EDGE CASES; it did not challenge the fix direction.

## Accepted (folded into CODING_SPRINT_PLAN v2)
- **Fixed seed-set for the smoke (all 4).** OS-entropy cast/style RNGs make before/after non-comparable. Use a fixed 12-leg input+seed list pinned via `OTR_CAST_SEED`/`OTR_STYLE_SEED` (the existing C7 reproducibility env vars) recorded in `SPRINT_BASELINE.md`; reuse for every sprint. Rounding rules stated.
- **Per-task commits (GPT, Grok).** Resolve the "after EVERY change" vs batched-T1.4 contradiction: each `.py`-touching task is its own green chunk (suite+Bug Bible+commit+push+HEAD verify).
- **Name the per-line variable `beat_target_words` + guard None (GPT, Gemini, DeepSeek).** Gemini's "stuck at 200" fear assumes the EPISODE target (864); it is the per-BEAT target (~20-64), so the cap is fine -- but the ambiguity is real. Rename + add a numeric fallback + a test for None and for 864.
- **F2 single source of truth + empty-case (GPT, Gemini, DeepSeek).** Pick the costly slot AND create its `must_turn` contract from one character-only list; assert `picked_slot_id in must_turn_contract_slot_ids`; define the zero-eligible-character-beat case (force the contract on the chosen last character slot, never leave `d001`=announcer). Test it.
- **F3 + F7 = prompt-first + bounded recompose, never strip-in-place (GPT, Gemini, DeepSeek).** Stripping "He paces..." leaves fragments; a reject/reroll pass would violate C3. Design: a negative OUTPUT-FORMAT constraint prevents it; a deterministic detector, when it fires, triggers the EXISTING single-line recompose seam (already how line-hygiene works), with a logged fallback to the original. This is a hygiene repair, not a new QA gate -- C3 honored.
- **F7 targets THIRD-PERSON narration / stage-direction-summary only (DeepSeek).** First-person self-narration ("I'm clamping...") is a separate craft tic (F6's domain), NOT broken; exclude it from the 0-narration target. Detector = line opens with He/She/They/<speaker-name> + narration verb, or is a 3rd-person summary.
- **Define the scan metrics operationally (GPT, DeepSeek).** Add a fixed hedge-phrase list + which `ending_change` categories count as "resolved success" (for F3); add a "length-pass fired?" field; state actual/target is the CHARACTER word total (exclude announcer/music).
- **F5 acceptance = artifact check, not "judge read" (GPT, DeepSeek).** Each character card carries a nonempty `speech_signature` (<=3-5 words, Gemini's bloat cap) and the composer prompt includes it; subjective distinctness is nice-to-have.
- **F8 `arc_shape` is additive `meta`, NOT a macro-schema key (GPT, Gemini).** Resolves the "add to macro" vs "keep macro schema" contradiction and the T3.1/T3.2 ordering question: pick `arc_shape` in a small seeded pre-step, pass it as CONTEXT into the macro + dramatic_state prompts, record `meta.arc_shape`. Macro JSON schema untouched (C2).
- **Sprint 0 tooling is a committed chunk (GPT, Grok).** `scripts/story_quality_scan.py` + `SPRINT_BASELINE.md` commit under build discipline.
- **Move shared narration regexes to one module (GPT).** Avoid an import cycle between `_otr_line_composer` and `_otr_line_hygiene`.

## Rejected / corrected
- **Grok: "F7 creates a new module + wires into `_otr_story_spine.py` -> violates C4 (logic outside node 1)." MISREAD.** `_otr_line_hygiene.py` ALREADY EXISTS and `_otr_story_spine.py` is an INTERNAL module called by `OTR_LedgerScriptWriter` (node 1), not a separate ComfyUI node. Editing the existing hygiene module stays inside node 1. F7 edits the existing module (not a new one). C4 holds.
- **Gemini: "drop the regex scrub, prompt-only."** Too absolute -- prompt-only won't reliably hit the 0-narration / 0-hedge targets (DeepSeek agrees re the outro hedge). Keep prompt-first + bounded recompose.
- **GPT: cut F6 (unconditional rider).** Keep it -- low risk, attacks over-tidy endings -- but give it a measurable acceptance (rider present in every character-line prompt; no length regression) and move it into Sprint 1 next to F1 (shared file region; GPT/DeepSeek both noted the adjacency).

## Deferred OUT of the executable plan (documented, not built this pass)
- **F9 (reorder outline after dramatic_state) -- CUT by all 4.** F2 fixes binding cheaply; the reorder is a large structural change with C4-spirit/regression risk. Moved to a "future structural option," not in Sprints 1-3.
- **F10 (persistent anti-repeat JSON) -- defer.** Statefulness/determinism/CI-contamination/race risk for the softest Tier-3 target. The deeper lever is RSS-source dedup (code out of scope). Revisit with explicit test-isolation + opt-out if variety becomes a release blocker.

## Net effect
Sprint plan tightens to Sprint 0 (measurement) + Sprint 1 (F1,F2,F3,F6) + Sprint 2 (F4,F5,F7) + Sprint 3 (F8 only). F9/F10 documented as deferred. Converged: no model challenged the fixes; all critiques were about rigor. One more pass (bugs) remains per the operator's plan.
