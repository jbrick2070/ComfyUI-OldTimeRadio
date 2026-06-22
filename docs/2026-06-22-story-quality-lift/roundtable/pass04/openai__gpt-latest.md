<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Close, but not build-ready as-is because DEFECT 1 has two predicate/well-formedness contradictions that can prevent required b010/b012 strips, and DEFECT 2 still contains schema/prompt forks that allow incompatible implementations.

MUST-FIX BEFORE BUILD:
1. [§2 Tier 3 / Well-formedness] Defect: “require final char in `_TERMINAL_PUNCT`” contradicts the required balanced-quote floor outcome. b005/b010/b012 will likely end with a closing quote after stripping, e.g. `"Not before I amplify it. The world deserves to hear this."`; the literal final char is `"`/curly close, not `.`, so a strict final-char check can abort the strip and fail the stated acceptance. Concrete fix: change the rule to “after optional closing structural double quote(s), the last spoken character must be in `_TERMINAL_PUNCT`.” Add fixtures for straight and curly closing quotes.

2. [§2 Tier 3 / `is_third_person_action_clause`] Defect: the predicate says strip only when there is “no `_PRONOUN_ROOTS` token,” but required positives include b010 `clutches her wedding ring tightly` and b012 `taps his cane impatiently`. If `_PRONOUN_ROOTS` includes possessive/object pronouns such as `her`/`his`, the required fixture outcomes become impossible. Concrete fix: specify the actual pronoun rule so b010 and b012 return TRUE. Minimal wording: “Do not reject possessive/object pronouns after an early narration verb; reject only first-person/self-reference/speaker-name cases needed to protect legitimate spoken action.” Add explicit unit assertions:
   - `clutches her wedding ring tightly` => strip candidate TRUE.
   - `taps his cane impatiently` => strip candidate TRUE.
   - protected first-person spoken-action negatives remain FALSE.

3. [§2 Tier 2 / composer reroll site] Defect: the plan moves stage-business detection into `compose_line_draft`, but says “keep the existing one-reroll guard” from `compose_line` and “this is the ONLY tier that can reroll.” This is under-specified and can produce either zero rerolls, two rerolls, or recursion at the wrong layer. Concrete fix: state the exact control flow:
   - disable/remove the existing `compose_line` stage-business reroll block for stage directions, or make it delegate only to the new draft-level detector;
   - pass a single `_stage_dir_repair_attempted`/equivalent guard into `compose_line_draft`, or return a typed reroll request from draft to `compose_line`;
   - prove with a test that one malformed line gets at most one stage-business reroll total.

4. [§4 `StanceIssue` schema] Defect: the schema still has an implementation fork: `target [pass cast/protagonist/central-object context into the critic prompt, or relax target to a free-form string validated in tests]`. Two builders can choose different schemas/prompts and produce incompatible reports/tests. Concrete fix: choose one. Leanest: `target: str` free-form, with the critic prompt given protagonist/central-object labels as context but no enum validation. Update tests to validate presence/round-trip, not a closed target enum.

5. [§4 `StanceIssue.missing_turn_beat`] Defect: “id OR reason string” is not a concrete typed contract. Concrete fix: define it as one field with one type, e.g. `missing_turn_beat: str` where values may be a beat id or a reason string, or split into `missing_turn_beat_id: Optional[str]` and `missing_turn_reason: Optional[str]`. Do not leave it as an informal union.

6. [§4 Detection / telemetry only] Defect: the plan adds FailedDimension `"stance"` while also saying “telemetry only — no reroll/gate in v1.” Given W4, `failed_dimension` is consumed by `_otr_reroll.py`; if stance issues are converted into `RerollTarget`s, this silently reintroduces a repair path that the section says is cut. Concrete fix: explicitly state that `StanceIssue` entries are stored only under `meta.story_critic_report` and are not converted into reroll targets, freeze gates, or `needs_full_rerun` in v1. Add a test asserting a stance issue does not invoke targeted reroll or change freeze verdict. [ASSUMPTION] This depends on how critic report issues are currently bridged into reroll targets.

7. [§3 / §7 pre-freeze sweep] Defect: the role-coercion sweep placement is still a verify question, but it is not non-blocking: if it runs before `cast_lock.py:473` or after scrub/TTS routing, b011 can still be misrouted. Concrete fix: promote this from “small, non-blocking” to mandatory build verification. The plan must require the sweep to run after all known speaker_role mutators, including cast_lock and `_otr_speaker_role.py` backfill, and before scrub/hash/TTS routing. If exact phase order cannot be named in the plan, add a build-time assertion immediately before routing: `cast char_id => speaker_role == "character"` and `speaker_role == "announcer" => char_id == "announcer"`.

SHOULD-FIX:
1. [§3 `coerce_speaker_role_for_char_id`] Defect: `cast_ids` source is not specified for every application site. `production_ledger.set_lines` may not naturally have cast ids in scope. Concrete fix: specify that `cast_ids` is derived from the current ledger cast table at the call site, excluding `"announcer"` and music/sfx sentinels; if unavailable, the function must no-op and the final pre-freeze sweep remains authoritative.

2. [§2 Tier 2 / `undelimited_action_clause`] Defect: b017 is load-bearing, but the detection rule for undelimited embedded action clauses is not concretely bounded. A builder could implement an overbroad lowercase-comma heuristic or a too-narrow quote-only detector. Concrete fix: define minimum fixture-driven behavior: b017 must hit `reason_code="undelimited_action_clause"` before formatting normalization; benign lowercase-after-punctuation negative must not hit.

3. [§6 Acceptance] Defect: “b015/b017 Tier-2 rerolled or CI-fail” is weaker than the production story-quality goal; production can still “ship LOUD” with undelimited leaks after reroll exhaustion. Concrete fix: explicitly state this is an intentional v1 limitation, or add a production freeze warning artifact/counter that makes the leak visible in final metadata. Do not imply full leak=0 beyond the balanced-quote class.

4. [§5 DEFECT 4 CUT] Defect: “CUT” is too terse for a previously grounded escalation seam. If abrupt semantic escalation was part of original acceptance, this is a regression. Concrete fix: either state “DEFECT 4 is explicitly out of scope for this build; no tests/acceptance cover escalation scope,” or retain a prompt-only outline rider with no gate. [ASSUMPTION] Original pass acceptance is not shown here.

5. [§7 Verify-at-build] Defect: current verify items are questions, not concrete steps. Concrete fix: replace each with an actionable check as listed below.

OPTIONAL / NICE-TO-HAVE:
- Add a small comment near the shared quote segmentation helper explaining why single quotes are ignored, to protect titles/scare-quotes like `'The Chronicle'`.
- Add a metric name convention for `compose_flags` values so future tests do not depend on free-form string parsing beyond prefix matching.

CUT THESE:
1. [§4 FailedDimension `"stance"`] Safe to cut if stance is telemetry-only and never used for reroll/gate. The critic report can carry typed `StanceIssue` without extending `FailedDimension`, reducing risk of accidentally routing stance into `_otr_reroll.py`. If you keep it, add the no-reroll test from MUST-FIX #6.

2. [§2 reason_code granularity beyond fixtures] If build time is tight, cut extra diagnostic codes not used by acceptance. Keep only the four listed codes if fixtures assert them; do not add more “obvious” variants.

3. [§2 “+ obvious neighbors” for `_NARRATION_VERBS`] Cut or replace with an explicit list. “Obvious neighbors” is not buildable and can cause over-strip regressions. Keep only verbs required by corpus/fixtures plus explicitly named inflections.

VERIFY-AT-BUILD checklist:
1. [§3 / §7] Confirm actual freeze cascade phase order: final role-coercion sweep executes after cast_lock’s legitimate announcer re-stamp and all other known speaker_role mutators, and before scrub/hash/TTS routing. Add/run a test ledger with `char_id=c02, speaker_role=announcer` and verify final frozen row is `speaker_role=character`.

2. [§3] Confirm cast_lock announcer sentinel remains valid: row with `char_id=="announcer"` may be `speaker_role=="announcer"` and is not coerced to character.

3. [§1 / §3 / §6] Confirm `compose_flags` accepts arbitrary `"kind:detail"` strings downstream. Build check: add representative flags `stage_dir_stripped:x` and `role_coerce:x` to a fixture row and run freeze/TTS prep without validation failure.

4. [§4] Confirm critic report pydantic/model strictness. Build check: serialize and deserialize `meta.story_critic_report` containing one `StanceIssue`; assert no unknown-field rejection and exact round-trip of all new fields.

5. [§4] Confirm stance telemetry does not trigger reroll/gate. Build check: a critic report with only stance issue leaves reroll target list/freeze verdict unchanged. [ASSUMPTION] Exact bridge from critic report to reroll target is not shown in grounding.

6. [§1 / §3] Confirm `OTR_TEST_MODE` is set by conftest and is the correct gate for CI-only asserts. Build check: invariant assert fires under tests for `char_id` cast id + `speaker_role=announcer`, and does not fire in a normal non-test render path.

7. [§2] Confirm shared quote helper behavior: straight and curly double quotes segment as structural quotes; single quotes/apostrophes are ignored. Include `'The Chronicle'` and `'alive' frequency` negatives.

8. [§2] Confirm `_strip_stage_directions` idempotence: second pass over b005/b010/b012-stripped outputs returns `(same_text, False)`.

9. [§2] Confirm existing unanchored delimited scrub still removes bracket/asterisk/qualifying paren directions outside dialogue but does not remove parentheticals inside quoted spoken dialogue unless the existing rule already did so.