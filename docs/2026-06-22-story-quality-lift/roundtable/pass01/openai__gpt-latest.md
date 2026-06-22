<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document identifies real defects, but it is not build-ready because the quality goal, no-op claim, gate strategy, and audio/golden expectations are still internally unsettled.

MUST-FIX BEFORE BUILD:

1. [0, 3] “Make the weak-end story measurably better” is not measurable.
   Defect: Acceptance only checks that specific defects are caught/stripped; it does not define a story-quality score, threshold, regression corpus, or pass/fail delta from C+.
   Concrete fix: Add a measurable campaign target, e.g. “same weak local config, no bypass, minimum N blinded/story-scan score improvement, no new critical defects, and zero regressions on a strong-model fixture.” Keep the 5 corpus-line checks as defect-level tests, not the definition of story lift.

2. [0, 1, 3] The “no-op on a good script” claim is unsupported.
   Defect: The plan says every gate must be one a strong model already passes, but provides no strong-end/opus fixture, no baseline evidence, and no acceptance condition proving zero mutation/reroll on good scripts.
   Concrete fix: Add a strong-model regression fixture and require: no workflow JSON drift, no schema drift, zero frozen-text changes, zero forced rerolls, or explicitly whitelisted metadata-only observations.

3. [1, 2 DEFECT 1, 4 Q1-Q2] The top fix is not specified safely enough for a deterministic freeze floor.
   Defect: The plan requires covering trailing, embedded-between-quotes, and embedded-undelimited bare stage directions, but still leaves open the core false-positive question. Putting an unresolved classifier into `_otr_ledger_scrub._strip_stage_directions` risks deleting legitimate spoken narration.
   Concrete fix: Split the fix into tiers:
   - deterministic floor: only strip high-confidence quoted patterns, e.g. `"<spoken>." <physical-action-clause>` and `"<spoken>." <physical-action-clause> "<spoken>"`;
   - composer reroll: detect broader suspicious embedded-undelimited cases like b017 and reroll with a LOUD hint;
   - only promote undelimited stripping to freeze floor after explicit negative fixtures pass.

4. [2 DEFECT 1, 3] The audio-byte-identical invariant conflicts with expected frozen-text changes unless the test scope is clarified.
   Defect: DEFECT 1 acceptance requires stripping five corpus lines from frozen text. If those lines are part of any audio golden path, TTS output cannot remain byte-identical. The document says byte-identical must stay green but also allows deliberate golden recapture; it does not say which path applies here.
   Concrete fix: Define three separate acceptance lanes:
   - hygiene/unit tests: no audio golden involved;
   - existing frozen audio baseline: must remain byte-identical unless the changed fixture is explicitly in scope;
   - Chandra re-smoke: if spoken text changes, require operator-gated golden recapture or compare only pre-TTS/frozen ledger text.

5. [2 DEFECT 2, 4 Q3] The antagonist-arc fix is still a design fork, not a plan.
   Defect: The proposed approach lists critic axis vs deterministic stance tracker vs outline-stage guard. Acceptance only says “at least one stance-reversal is caught,” not that the arc is repaired or that a coherent turn beat is inserted.
   Concrete fix: Pick one minimal lever before build. Recommended smallest change: add a line-scoped `stance_coherence` critic axis routed through existing `_otr_reroll.run_targeted_reroll`, with required output fields: character, object/person of stance, prior stance, new stance, missing turn beat, affected line ids, reroll hint. Acceptance must require the final frozen ledger has no unresolved critical stance reversal, not merely that one was detected.

6. [2 DEFECT 3, 4 Q4] The role-stamp repair point is unresolved even though the defect is framed as contained.
   Defect: The document says “decide whether to fix at init, at set_lines, or in the role_mismatch repair guard.” That is not build-ready. Grounding says `init_lines_from_outline` would not produce `announcer` + cast `char_id`, and the likely writer is `_otr_ledger_reviewer.py` role_mismatch repair.
   Concrete fix: Fix at the role_mismatch repair guard first: reject `expected="announcer"` when `char_id` is a cast id; require `char_id=="announcer"` for `speaker_role="announcer"`. Add a final ledger consistency assert before freeze: cast `char_id` => `speaker_role="character"`; `speaker_role="announcer"` => `char_id=="announcer"`.

7. [2 DEFECT 4, 3, 4 Q5] DEFECT 4 does not currently serve the stated lift goal.
   Defect: The approach is “strengthen prompt and/or measurement,” with acceptance only “measured even if not gated.” A non-gating scan does not lift the weak story and adds surface area.
   Concrete fix: Either cut DEFECT 4 from this pass, or define it as telemetry only and remove it from story-lift acceptance. Do not add a proportion/setup gate until DEFECT 2 is stable.

8. [3] “Caught by the gate” is too weak as an acceptance condition.
   Defect: For DEFECT 2/3, acceptance says the mis-stamp and at least one stance reversal are caught on re-smoke. A gate that catches but then ships via repair-then-ship, bypass, or unchanged fallback does not solve the user-facing story.
   Concrete fix: Require “caught, rerolled or deterministically repaired, and absent from the final frozen ledger,” with explicit failure behavior if max rerolls are exhausted.

SHOULD-FIX:

1. [1, 2 DEFECT 2, 2 DEFECT 4] The plan risks building multiple overlapping quality systems.
   Defect: It says reuse existing reroll machinery, but DEFECT 2 suggests a new stance tracker/critic/outline guard and DEFECT 4 suggests prompt changes plus `story_quality_scan.py`.
   Concrete fix: One gate path only for this pass: existing critic/reroll convergence for story issues; deterministic ledger assert only for structural correctness; deterministic scrub only for text hygiene.

2. [2 DEFECT 1] The three stage-direction sub-patterns need negative fixtures, not only the five positive corpus lines.
   Defect: The false-positive problem is acknowledged but no negative cases are required.
   Concrete fix: Add fixtures for legitimate spoken first-person action/narration, quoted titles, quoted scare words, and dialogue containing lowercase clauses after punctuation that are not stage directions.

3. [2 DEFECT 2] “Central object/another character” is underspecified.
   Defect: A stance detector cannot know whether Manfred’s stance is toward Mali, her work, the signal, the press leak, or Sherlock unless the plan defines the target.
   Concrete fix: Constrain v1 to stance toward the episode’s central dramatic object plus protagonist, derived from outline/cast metadata if available; otherwise critic must name the target explicitly in `meta`.

4. [2 DEFECT 3] The plan assumes the exact origin of the b011 stamp can be traced during build.
   Defect: Grounding identifies the likely writer, but the document still leaves open “unless the outline beat itself stamped the role.”
   Concrete fix: Add one debug/audit assertion around role changes: when `speaker_role` changes, stamp previous role, new role, source component, and reason into `meta` or test logs. Keep schema unchanged.

5. [1, 3] The no-workflow-change invariant needs a specific regression target.
   Defect: “Add a no-drift regression assert” does not say what is compared.
   Concrete fix: Hash or canonicalize `workflows/otr_scifi_16gb_full.json` before/after test execution and fail on any diff.

6. [3] The no-bypass re-smoke is necessary but not sufficient.
   Defect: The source smoke used `OTR_BYPASS_FREEZE_HALT=1`; the plan assumes the same defects reproduce under normal halt behavior. [ASSUMPTION]
   Concrete fix: First milestone should be a no-bypass baseline re-smoke before code changes, so fixes are not designed around bypass-only behavior.

OPTIONAL / NICE-TO-HAVE:

- [3] Add a compact “defect dashboard” emitted per episode: stripped stage-direction count, role consistency violations, stance-coherence criticals, scope-jump observations.
- [2 DEFECT 1] Store original stripped text in `meta` for audit if current schema conventions allow it. [ASSUMPTION]
- [2 DEFECT 2] Add a tiny hand-authored 6-line stance reversal fixture independent of Chandra so tests are not tied to one generated episode.

CUT THESE (scope / over-engineering):

1. [2 DEFECT 4] Cut DEFECT 4 as a gate for this pass.
   Why safe: The document itself marks it as candidate CUT and only requires measurement. It is not needed to fix the most visible hygiene leak, the structural role inconsistency, or the Manfred flip-flop.

2. [4 Q3] Cut deterministic stance-tracker and outline-stage guard alternatives for DEFECT 2.
   Why safe: They are unresolved design branches. A critic-axis routed through existing scoped reroll is the smallest change aligned with the “reuse existing machinery” invariant.

3. [4 Q2] Cut a general sentence-segmenter for DEFECT 1 v1.
   Why safe: The proven failures are narrow. High-confidence quote-boundary patterns plus reroll detection for undelimited cases are smaller and less likely to corrupt legitimate dialogue.

4. [2 DEFECT 4] Cut `story_quality_scan.py` scope-jump work unless it directly gates or informs reroll.
   Why safe: Measurement-only telemetry does not satisfy “make the weak-end story measurably better” unless tied to an acceptance threshold or repair loop.

5. [3] Cut “campaign output” process language from the build spec.
   Why safe: “This window is planner-only” describes roundtable workflow, not production behavior. Keep the actual deliverable as a dependency-ordered coder plan with tests.