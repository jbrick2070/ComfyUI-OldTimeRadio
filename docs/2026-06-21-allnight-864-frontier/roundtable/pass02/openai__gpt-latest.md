<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Acceptance is not reproducibly measurable as written, Sprint 1 has unresolved edge cases, and Sprint 3 contains a schema/ordering contradiction.

MUST-FIX BEFORE BUILD:
1. [Acceptance targets / Sprint 0] The smoke is stochastic and underspecified: “6-12 episode headless smoke” makes the >=90% costly-choice target ambiguous and not comparable before/after. With 6 episodes, >=90% means 6/6; with 12, it means 11/12. Concrete fix: define one fixed smoke size, one fixed seed-set/input set, and exact rounding rules for each percentage target. Record the seed/input list in `SPRINT_BASELINE.md` and reuse it for every sprint.

2. [Acceptance targets / T0.1] The length target says the ratio must improve “with the length-pass NOT firing as the cause,” but T0.1’s scan outputs only actual/target ratio, valid rate, outro agreement, and hygiene count. It does not require detecting whether the length-pass fired. Concrete fix: add a scan field for length-pass/recompose activation per episode, and make the Sprint 1 length acceptance require ratio >=0.85 with zero or explicitly bounded length-pass activations.

3. [Build discipline / Sprint tasks] The invariant says after EVERY change run the full suite, Bug Bible, commit, push, verify HEAD==origin, etc., but the sprint tasks batch multiple edits before the suite/commit steps, e.g. T1.1-T1.3 followed by T1.4. That contradicts the process requirement. Concrete fix: either weaken the invariant to “per green chunk” or split each sprint into explicit commit chunks with test commands and push/HEAD verification after each chunk. Do not leave both requirements in force.

4. [T1.1] The proposed token-cap expression `max_new_tokens=min(200, max(40, target_words*4))` assumes `target_words` is always present and numeric. [ASSUMPTION] If this composer can be called without `target_words`, this will fail at runtime. Concrete fix: guard it: derive a numeric `effective_target_words` with the current default/fallback before multiplying, and add a unit test for missing/None target plus the 864 case.

5. [T1.1] The prompt change is under-specified: it says replace “about 20-30 words” with `about {beat_lo}-{beat_hi} words` “or drop the number,” but the acceptance says “Update `_build_user_prompt` Test 3 (asserts literal target string).” These are two different expected prompts. Concrete fix: choose exactly one behavior. If using a band, define how `beat_lo`/`beat_hi` are computed and assert that. If dropping the number, update the test to assert absence of the stale literal and presence of the new nonnumeric instruction.

6. [T1.2] The costly-choice fallback remains unsafe. The plan says keep `"d001"` empty-fallback while also excluding announcer/music and requiring the picked slot to have a `must_turn` contract. If there are no character-only voiced beats, `"d001"` may not be a character beat or may not have the contract. Concrete fix: define the empty-character case explicitly: either synthesize/assign a valid character slot with a `must_turn` contract, or skip costly-choice binding and mark the audit reason deterministically. Add a test for zero eligible character voiced beats.

7. [T1.2] The fix says “Pick/check the costly slot from CHARACTER-only voiced beats” so it “matches the contract loop,” but it does not state whether the contract loop is changed or whether the picked slot is forced to receive `must_turn`. Merely picking from the same ID set does not guarantee a `must_turn` contract exists. Concrete fix: make the selected costly slot and contract creation share one source of truth; assert in code/tests that `picked_slot_id in must_turn_contract_slot_ids`.

8. [T1.3 / T0.1] “Outro-vs-ending agreement” is not operationally defined. The acceptance forbids hedge phrases when `dramatic_state.ending_change` is a “resolved success,” but the plan does not define the allowed values/taxonomy of `ending_change` or how the scan determines “resolved success.” Concrete fix: enumerate the exact `ending_change` values/categories considered resolved success and implement the scan against those values plus a fixed hedge phrase list.

9. [T2.4] The deterministic hygiene scrub is specified as regex detection but not as a safe rewrite. Removing or altering a line can create empty lines, too-short lines, broken char-band validation, or changed speaker semantics. Concrete fix: define the replacement strategy: e.g. rewrite only the offending prefix/self-name, preserve speaker and nonempty text, re-run the existing line validation, and fall back to the original only with an explicit logged failure count. Add tests for empty result, speaker-name substring, and legitimate non-self references.

10. [T3.1] Internal contradiction: it says “Add an `arc_shape` choice in the macro stage” while also saying “Keep macro schema `{title,premise,setting,time_of_day,central_tension}`.” If `arc_shape` is output by the macro stage, the schema changes; if the schema is unchanged, it must be stored elsewhere. Concrete fix: either store `arc_shape` only under additive `meta.*` outside the macro schema, or explicitly change the macro schema and confirm this does not violate the ledger/schema invariant.

11. [T3.1 / T3.2] Ordering is inconsistent. T3.1 adds `arc_shape` in the macro/dramatic-state template path, while T3.2 later says to pre-derive `dramatic_state` including `arc_shape` before outline. If T3.1 lands before T3.2, the source and timing of `arc_shape` are unclear. Concrete fix: define the final generation order before implementation: macro -> pre-derived dramatic_state including arc_shape -> outline, or macro -> outline -> dramatic_state. Then assign F8/F9 edits to that order.

12. [T3.3] The “small local JSON of recently-used descriptors/news-ids” introduces mutable persistent state, nondeterminism, possible test contamination, and possible parallel-run races. It also affects repeatability of byte-identical outputs for identical inputs depending on prior runs. Concrete fix: either make the anti-repeat list purely input/seed/meta-driven for the smoke, or specify file path, reset behavior, locking/atomic writes, test isolation, and an opt-out for deterministic regression tests.

SHOULD-FIX:
1. [Sprint 0 / Exit] Adding `scripts/story_quality_scan.py` and `SPRINT_BASELINE.md` is a repo change, but Sprint 0 says “no engine edits” and does not state whether this is committed/pushed under the build discipline. Concrete fix: make Sprint 0 a documented green commit with scan tests or explicitly mark it as documentation/tooling-only but still committed and pushed.

2. [T0.1] The scan’s hygiene metric says “third-person narration or self-address by name,” but T2.4 later defines only “line opens with He/She/They/<speaker> + narration verb” and “never speak your own name.” That misses narration not at the start and self-address variants. Concrete fix: define the exact regex/test cases used by both the scan and hygiene scrub so acceptance and implementation measure the same thing.

3. [T2.1] The task offers two different implementations: require gender/pronouns in the cast description, or pass raw `{gender}` into the CHARACTER block. These have different blast radii and test surfaces. Concrete fix: choose one. Prefer the smaller change: pass normalized gender/pronouns into the line-composer character context if already available; otherwise update the casting contract and add a deterministic unit test.

4. [T2.1] Acceptance “SOM-CORBEN-class clash gone in a targeted re-render; no ‘Mister <female-name>’” is too narrow and name-dependent. Concrete fix: add tests for male, female, nonbinary/unknown if supported, and title/pronoun consistency independent of a specific name.

5. [T2.2] Acceptance is “judge read,” which is not a build gate and will not be reproducible. Concrete fix: either downgrade F5 to manual/nice-to-have or add a concrete artifact check, e.g. each character card contains a nonempty speech-signature field/text fragment and the composer prompt includes it.

6. [T2.3] “Ungate the rider” can affect prompt length and line output globally, but the acceptance only says fewer explain-the-turn lines and no length regression. Concrete fix: add a unit test that the rider appears in the relevant prompt and that intro/outro/non-character paths are not unintentionally affected.

7. [T2.4] “Reuse `_NARRATION_LEAK_REGEXES` `:1310-1330`” conflicts with “new deterministic check in `_otr_line_hygiene.py`” unless those regexes are actually importable from there. [ASSUMPTION] If the existing regexes live in another module/private scope, direct reuse may create import cycles or duplication. Concrete fix: move shared regexes to one hygiene module or duplicate intentionally with tests.

8. [T1.3] “Thread final character line into the outro user prompt” has a hidden data dependency on the final rendered character line being available at outro composition time. [ASSUMPTION] If outro composition currently happens before all lines are finalized, this requires reordering. Concrete fix: verify call order before editing; if unavailable, pass only `ending_change` and the final beat/slot summary, or compose outro after final character line is finalized.

9. [Acceptance targets] “Audio must stay byte-identical where text is unchanged” is listed, but none of the sprint exits says how to prove the “where text is unchanged” condition for prompt-only changes. Concrete fix: name the exact existing audio byte-identical test command/golden fixture in each sprint exit or reference the specific test file.

10. [T3.2] “Costly choice is placed by the outline, not retrofitted” is a behavior assertion, but no scan/test is specified to distinguish outline placement from later repair. Concrete fix: add a trace/meta field under additive `meta.*` or a test seam showing the outline included the costly slot before contract repair.

OPTIONAL / NICE-TO-HAVE:
- [Open questions #5] Do not leave fixed seeds as an open question; resolve it in Sprint 0.
- [Sequencing] Add an explicit “stop after Sprint 1 if ship-first metrics pass” decision point if Sprint 2/3 are not required for the release.
- [Acceptance targets] State whether actual/target counts include intro/outro/music/announcer lines or only character dialogue.

CUT THESE (over-engineering):
1. [T3.2] Cut or defer F9 reorder for this pass. The plan already expects F2 to fix costly-choice validity cheaply, and T3.2 is explicitly “larger reorder” with high regression risk. Safe to cut because Sprint 1 acceptance already targets >=90% validity without F9.

2. [T3.3] Cut the persistent local JSON anti-repeat list unless variety is a release blocker. It adds state management, determinism, and race risks for a soft Tier 3 target. Safe to cut because the document labels variety as “Tier 3, softer,” and no hard release target depends on it.

3. [T2.2] Cut the “judge read” speech-register acceptance as a build gate. Keep only a prompt/card inclusion test if needed. Safe to cut because subjective voice distinctness is not automatable and can block the sprint without a reproducible failure.

4. [T2.3] Cut unconditional anti-decorative rider from Sprint 2 if F1 already fixes length and F7 fixes hygiene. It is a broad output-style change with weak acceptance. Safe to cut because it is craft-quality, not required for the hard Tier 1 acceptance targets.