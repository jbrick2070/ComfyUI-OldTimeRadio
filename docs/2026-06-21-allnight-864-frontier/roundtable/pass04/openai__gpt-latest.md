<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still has acceptance-blocking ambiguities/contradictions in T1.2, F3, F7, and the measurement contract that can make the first build either crash, measure the wrong thing, or fail the stated targets.

MUST-FIX BEFORE BUILD:
1. [T1.2 / F2] The zero-eligible path contradicts the candidate rule. T1.2 says build costly-slot candidates from CHARACTER-only voiced beats, but then says “Empty case: force the contract onto the chosen last character slot.” If there are zero CHARACTER-only voiced beats, there may be no “last character slot” to force onto. Concrete fix: define one deterministic fallback:
   - either skip/disable the costly-choice contract when no character voiced slot exists and mark `episode_valid` rules accordingly, or
   - choose from a broader explicitly defined fallback list, e.g. last voiced CHARACTER line after ledger generation, and if none exists, do not emit the contract.
   Add the zero-eligible test to assert the exact fallback behavior and that no announcer/music/sfx slot receives `must_turn`.

2. [Measurement contract / T1.2] `episode_valid` is not defined precisely enough to be buildable or comparable. The document says the scan reports `episode_valid`, and T1.2 accepts `episode_valid>=11/12`, but does not specify whether this means `_otr_slot_drama_contract.validate_episode_contracts`, freeze validation, both, or a new script-local rule. Concrete fix: define `episode_valid` as an explicit boolean formula in `scripts/story_quality_scan.py`, e.g. `freeze_valid && dramatic_contract_valid`, and name the exact validators/errors counted. Otherwise Sprint 0 baseline and Sprint 1 acceptance can disagree.

3. [T1.3 / F3] The hedge repair has no guaranteed success path. The plan allows one deterministic post-check and a single recompose if `HEDGE_LIST` appears while ending is RESOLVED, but acceptance requires `outro_hedge_vs_resolved=0/12`. If the recomposed outro still contains a hedge, the plan leaves the bad line in place. Concrete fix: after the single recompose, run the same detector again; if it still fails, use a deterministic fallback outro template that states the resolved `ending_change` without any `HEDGE_LIST` phrase, or fail the build/test explicitly. Add a test for “first compose hedges, recompose hedges again.”

4. [T1.3 / Measurement contract] RESOLVED classification is underspecified and risks scanner/engine mismatch. The measurement contract says RESOLVED is a “keyword rule in the script,” while T1.3 says the composer uses `meta.dramatic_state.ending_change` and system rule “if resolved.” If the prompt-side classification and scan-side classification differ, the engine can think it is unresolved while the scanner fails it as resolved+hedged. Concrete fix: implement one shared deterministic helper for `is_resolved_ending_change()` and use it in both the composer repair and `story_quality_scan.py`.

5. [T1.1 / F1] The `None` path is not concretely safe. T1.1 says “rename the cap input `beat_target_words` + None guard” but also requires `max_new_tokens=min(200,max(40,beat_target_words*4))`. As written, `None * 4` crashes if the guard is not before the expression. Concrete fix: specify the exact guard, e.g. `if beat_target_words is None: max_new_tokens = 200 else: max_new_tokens = min(200, max(40, int(beat_target_words) * 4))`, and test `None`.

6. [T1.1 / F1] “No mid-sentence truncation” is not achieved by the stated token cap alone. `max_new_tokens` can still cut generation mid-sentence if the model reaches the cap. Concrete fix: either remove that acceptance/test claim, or add a deterministic post-generation sentence-boundary trim/recompose rule and test it. If trimming is added, confirm it does not violate `test_audio_byte_identical` for unchanged text.

7. [T2.3 / F7] The narration detector is too vague and likely to false-positive normal dialogue. “Line opens with He/She/They/<speaker> + narration verb, or 3rd-person summary” can match legitimate spoken lines such as “He is lying,” “They know the code,” or a character referring to another character. Concrete fix: define the exact regex/verb list and exclusion rules before build. Require tests for:
   - first-person narration allowed,
   - legitimate third-person reference allowed,
   - speaker-name substring does not trigger,
   - true self-narration triggers,
   - empty recomposition result falls back to original.
   Do not route any detector hit to `_hy_recompose` until false-positive tests exist.

8. [T2.3 / C3 no QA gate] The F7 detector/recompose path must be constrained to a hygiene repair, not a hidden QA reroll. The plan says this is allowed, but does not define the maximum attempts or logging fields. Concrete fix: specify exactly one recompose attempt, original-text fallback, and a log marker distinct from scoring/reject/reroll. Add a test that multiple retries cannot occur.

9. [T2.1 / F4] The implementation path is ambiguous. T2.1 says “pass normalized `{gender}`/pronouns into the compose_line CHARACTER context … or require them in the `_otr_casting` contract.” Those are materially different changes, with different schema/test impact. Concrete fix: choose one path before build. Smallest safe fix: normalize pronouns at prompt assembly from existing cast data; only add cast contract requirements if current cast data is insufficient. [ASSUMPTION] Verify current cast records actually contain enough gender/pronoun data.

10. [T2.2 / F5] “Every card has nonempty `speech_signature`” lacks a fallback for existing/legacy cast entries. Additive `cast[].speech_signature` is allowed, but any code path that reuses cast without generating this field can fail the acceptance test. Concrete fix: define a deterministic default/backfill, e.g. derive from role/archetype or set `"plain spoken"` when missing, and test legacy cast cards.

11. [Sprint 0 / Measurement contract] The fixed 12-leg smoke is not reproducible until the source of the 12 legs is specified. The document pins `target_words=864` and two seeds, but does not specify the exact inputs/prompts/RSS items/story seeds used for each leg. Concrete fix: `SPRINT_BASELINE.md` must record the 12 exact invocations or input payloads, not only seed list and VRAM/port state. Otherwise later sprint comparisons can drift.

12. [Open verify-at-build items / T1.1] Availability of `beat_lo`/`beat_hi` is still open, but T1.1 depends on it. Concrete fix: before editing `_build_user_prompt`, verify those values are in scope or pass them explicitly through the call chain. If unavailable, use the documented fallback “or drop the number” and update tests accordingly.

SHOULD-FIX:
1. [T1.2 / F2] The plan mixes “slot,” “beat,” and `line_id` language. Acceptance says `picked_slot_id in must_turn_contract_slot_ids`, but freeze invariants are line-oriented and the writer wiring is elsewhere. Concrete fix: define whether the costly-choice identifier is a beat id, slot id, or `line_id`, and convert in exactly one place. [ASSUMPTION] Verify `_otr_slot_drama_contract.validate_episode_contracts` checks the same id namespace.

2. [T1.3 / F3] `ending_change` is described as “always present,” but additive `meta.*` implies tolerant readers and possibly missing metadata. Concrete fix: null-guard `meta.dramatic_state.ending_change`; missing/unclassified should be treated as unresolved/unknown by both composer and scanner, not crash.

3. [T1.3 / F3] Final-character-line threading depends on outro composition order. The plan notes the uncertainty but does not make the fallback testable. Concrete fix: add two tests: final character line available, and unavailable uses final-beat summary without crash. [ASSUMPTION] Verify outro is composed after character lines in the current writer.

4. [T1.4 / F6] “Every CHARACTER beat” needs to map to prompt construction, not ledger output. Concrete fix: test at prompt-builder level for intro/outro/music exclusion and all character-line inclusion, including recomposed hygiene prompts if they reuse a separate prompt path.

5. [T3.1 / F8] Seed source for `arc_shape` is not specified. The plan says “SEEDED pre-step” but not whether it uses `OTR_STYLE_SEED`, `OTR_CAST_SEED`, episode seed, or another deterministic source. Concrete fix: tie it to the same reproducibility contract and record the selected `meta.arc_shape` in `SPRINT_BASELINE.md`.

6. [T3.1 / F8] Adding `arc_shape` into macro/dramatic prompts while keeping macro JSON schema unchanged can still cause the LLM to emit unexpected fields. Concrete fix: test that macro parsing ignores/strips any extra `arc_shape` field from LLM output and records only additive `meta.arc_shape`.

7. [Build invariant] “Bug Bible” is referenced as a required gate but not named as a command/path. Concrete fix: specify the exact command/file so each chunk has a repeatable gate.

8. [Sprint 0] “Selective CIM kill, port 8000/8011 clear” is operationally risky without a process match rule. Concrete fix: kill only processes matching the known Comfy/headless command line or PID file, not arbitrary owners of those ports. [ASSUMPTION] Verify the local harness cannot kill unrelated developer services.

9. [Acceptance targets] The length metric excludes announcer+music from numerator but divides by full `target_words=864`. If `target_words` is intended as total episode words, this depresses the ratio by design. Concrete fix: confirm target is “character word target,” or change denominator to character-target words. [ASSUMPTION] Verify current generator treats `target_words` as character-only.

10. [Sequencing & first commit] The document says “F1->F6 (shared region)” but first commits are Sprint 0, T1.1, T1.2, T1.3; T1.4/F6 comes later. If F1 and F6 touch the same prompt region, delaying F6 increases merge/retest churn. Concrete fix: either move T1.4 immediately after T1.1 or state why the shared-region dependency is safe.

OPTIONAL / NICE-TO-HAVE:
- [Measurement contract] Include raw counts behind each aggregate metric in `SPRINT_BASELINE.md`, not only pass/fail.
- [T2.1 / F4] Add a small fixed table of gender/pronoun examples to prevent regressions independent of names.
- [T3.1 / F8] Record both requested `arc_shape` and final accepted dramatic template name if those can diverge.

CUT THESE (over-engineering):
1. [T2.3 / F7] “Move shared narration regexes into one module” should be cut from the sprint unless there are already multiple real consumers. It adds import-cycle risk and is not required to meet `narration_self_address_lines=0`. Keep the detector local to `_otr_line_hygiene.py` first; extract later only if duplication appears.

2. [Sprint 0] Recording “VRAM/port state” in `SPRINT_BASELINE.md` is not necessary for story-quality acceptance unless the harness is known to be resource-flaky. Keep the reset/health checks operational, but do not make VRAM/port snapshots part of the quality baseline comparison. [ASSUMPTION] Safe if the smoke outputs are fully captured by seeds and input payloads.

3. [T3.1 / F8] Five arc shapes may be more than needed for the first variety proof. The acceptance only requires “distribution not single-valued across the smoke.” Start with three templates if implementation pressure is high; the metadata and seeded selection mechanism are the real deliverable.