# Roundtable pass 04 -- judgment (bug/risk final)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend $0.12. This pass found real IMPLEMENTATION bugs/edge-cases; none change direction. All folded into SPRINT_READY_PLAN (final).

## Accepted bug fixes (convergent, real)
1. **F1 None crash (GPT, Gemini, DeepSeek).** `min(200,max(40,beat_target_words*4))` TypeErrors on None. Use `max_new_tokens = 200 if beat_target_words is None else min(200, max(40, int(beat_target_words)*4))`.
2. **F1 `beat_lo/beat_hi` not in scope (all 4).** Interpolating `{beat_lo}-{beat_hi}` would NameError. -> **Default F1 to "drop the number"** (no interpolation, removes the hard 20-30 cap, zero scope dependency). Band interpolation only IF the values are confirmed in `_build_user_prompt` scope at build time.
3. **F8 pipeline STALL (Gemini, Grok, GPT).** New arc shapes (`investigation_without_answer`, `slow_dread`) fail the opposed-wants/resolution post-validator -> valid generations rejected. -> Branch the validator: only the confrontation shapes require opposed wants; non-confrontation shapes use a shape-appropriate check.
4. **F2/F7 zero-eligible/empty-cast crash (all 4).** "force onto last character slot" IndexErrors with 0 character slots. -> Early-return: if no character voiced slot, do NOT emit the contract (episode is invalid, which is acceptable+rare); add all-announcer/empty-cast tests.
5. **F3 hedge repair has no guaranteed success (GPT, Gemini, DeepSeek).** Single recompose may still hedge. -> After recompose, re-check; if still hedged, use a deterministic fallback outro template that states `ending_change` with NO HEDGE_LIST phrase (the announcer already has deterministic fallbacks). Guarantees 0/12. Still C3-safe (template, not a reject pass).
6. **RESOLVED classification must be ONE shared helper (GPT, DeepSeek).** Composer-side and scan-side "is resolved" must not diverge. -> `is_resolved_ending_change()` shared by `compose_announcer_outro` repair and `story_quality_scan.py`.
7. **F7 detector false-positives (all 4).** "He is lying" / "They know the code" are legit. -> Target NARRATION/stage-direction verbs (paces, stops, gazes, contemplates, questions-as-summary) describing the SPEAKER's own behavior, NOT cognition/state verbs; mandatory false-positive tests BEFORE wiring to recompose; exactly 1 recompose attempt, distinct log marker (not "reroll"), test no-multiple-retries.
8. **F5 speech_signature generation + backfill (GPT, Gemini, DeepSeek).** -> Generate it in the existing cast LLM call (add to the JSON it already returns); deterministic default ("plain spoken") backfill for any missing/legacy card.
9. **F4 path + use existing cast.gender (GPT, DeepSeek, Gemini, Grok).** -> Smallest fix: inject the speaker's gender/pronouns from the EXISTING `cast[].gender` field at compose_line prompt assembly; do NOT change the casting contract (avoids C2 worry).
10. **F6 SPLIT (Gemini -- genuine craft correction).** Ungating "the situation must be different after this line" to EVERY beat causes over-acting/continuity loss. -> Split: "perform indirectly, do not summarize/explain the turn" becomes unconditional (always good); "the situation must be different" stays gated to turn/costly beats.
11. **episode_valid defined in the scan (GPT).** `episode_valid = freeze_valid AND dramatic_contract_valid`, exact validators named, so baseline and acceptance agree.
12. **length_ratio denominator (GPT).** Numerator = all VOICED words (character + announcer, exclude music) to match the 864 target which budgets the announcer; keeps it consistent with the 0.70 baseline (which was total_word_count/target).
13. **Sprint 0 pins the 12 invocations, not just seeds (GPT).** Record the exact 12 news inputs + `OTR_CAST_SEED`/`OTR_STYLE_SEED` in `SPRINT_BASELINE.md`.
14. **Sequencing: F6 lands right after F1 (GPT, Gemini, Grok).** Same prompt region -> Sprint 1 order = T1.1(F1) -> T1.4(F6) -> T1.2(F2) -> T1.3(F3).

## Corrected / cut
- **"move shared narration regexes to one module" -- CUT (GPT, Grok).** Import-cycle risk, premature. Keep the small verb/regex set LOCAL to `_otr_line_hygiene.py` (duplicate with a test); refactor only if duplication grows.
- **"no mid-sentence truncation" hard unit gate -- downgraded (GPT, DeepSeek).** A token cap can't guarantee it; the unit test asserts the cap is set, the smoke checks via `length_pass_fired`/ratio + spot-read.
- **Gemini "0.85 impossible if cap ~150 words" -- not a problem.** beat_target_words is per-BEAT (~20-64); 200 tokens (~150 words) >> 64, so the cap isn't binding per beat; 14x~62=868 reaches 864 (verified by the budget widening to ~64/beat). No change.
- **Gemini "requiring pronouns in casting contract violates C2" -- mooted** by choosing the prompt-assembly path (#9).
- **Grok "keep counts identical" redundant sentence -- dropped;** kept the explicit FIRST-PERSON-excluded clause (real clarity, DeepSeek pass-2).

## Convergence
Converged. Four passes total: direction (pass 1) -> sprint (pass 2) -> wiring (pass 3, consumer-audited) -> bugs (pass 4). The bug pass produced only implementation-hardening, no new direction. SPRINT_READY_PLAN (final) folds all 14 fixes. Ready for `/otr-handoff`.
