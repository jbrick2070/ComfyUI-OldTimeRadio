# Claude anchor review -- R1 (high-level arc / coherence)

Driver-panelist review of `pass00_plan.md`, written BEFORE reading any panel
output. Every claim below is labelled against the real Windows files.

## VERDICT

**PROCEED, with three MUST-FIXes.** The plan's core reasoning is correct and I
re-verified it by running the real code rather than reading it: alpha binds
before the ceiling, so shipping `EFFECTIVE_EMOTION_MASS_CAP=0.56` alone would
deliver 0.400 on a neutral line and 0.374 on the emotional line, neither of
which is the 0.560 rung the operator approved. Pinning alpha to 1.0 and moving
the ceiling to 0.56 reproduces that rung exactly.

The strongest evidence for the change is also the strongest evidence against
the objection I expected to have to answer. Across 57 character lines from the
6 most recent real episode ledgers, raw vector sums run 0.733 .. 1.333 --
**zero lines fall below 0.56** -- so the ceiling binds on 100% of production
lines and the shipped sound is exactly the ladder rung he heard. This is not an
extrapolation from one audition line.

## MUST-FIX

1. **[CONFIRMED] The fingerprint in the working tree is already dead.**
   `config/cast_pools.py` currently claims `engine_impl_version:
   c18df292a41d3ddc`, computed from the PRE-change `eng_indextts2.py`. Editing
   two constants in that file moves `live_engine_impl_version("indextts2")`
   (`nodes/_otr_voice_route.py:159` lists it in `RUNTIME_FINGERPRINT_SOURCES`),
   so the value must be recomputed AFTER the constants land and BEFORE the
   record is written. Order is: constants -> fingerprint -> re-render -> record.
   Writing the record first produces a route that demotes itself.

2. **[CONFIRMED] A second instrument can still destroy cited evidence.**
   `scripts/otr_lemmy_cross_engine_audition.py` writes `MANIFEST.json` (lines
   242 and 269) with **no** overwrite refusal, and
   `otr/episodes/lemmy_cross_engine/MANIFEST.json` is cited by sha256 in THREE
   provisional route records in `config/cast_pools.py` (lines 1052, 1082,
   1107 -- kokoro, chatterbox, dia). This is the exact defect d910b4ae fixed in
   `otr_g1_lemmy_audition.py`, still live in its sibling. The plan asks whether
   a second instrument exists; it does, and the answer is not "verify later".

3. **[CONFIRMED] The docstrings that justify the two-knob split become false.**
   `current_emo_mass_cap` (line 285) argues the cap knob exists because the
   2x2's alpha axis was degenerate -- a concluded experiment.
   `emotion_payload` (line 340) says "alpha is the taste knob above it"; after
   this change alpha is not a taste knob at all. Both are load-bearing comments
   that a future reader would act on. Rewrite them in the same commit, not
   after.

## SHOULD-FIX

4. **[CONFIRMED] Keep alpha, do not delete it.** I grepped every non-test
   consumer. Nothing branches on the value: `scripts/_otr_indextts2_worker.py:93`
   defaults it to 1.0 when absent, and `nodes/_otr_voice_node_common.py:1230`
   only formats it into the observability line. So pinning to 1.0 is behaviour-
   safe, and deletion would touch the cache key, the receipt, the profile
   schema and the acceptance checker for no gain. Keep it as an env override
   defaulted to 1.0 and say so in the docstring.

5. **[CONFIRMED] The uniformity consequence is real and should stay in the
   record.** Old effective mass varied 0.293 .. 0.400 across those 57 lines;
   new is 0.560 on every one. The vector SHAPE still varies, the total budget
   no longer does. The operator approved that sound, but the qualification
   record should say it rather than imply per-line variation survives.

6. **[CONFIRMED] The cache key is already correct.** `render_time_params`
   (line 411) returns both `emo_alpha` and `emo_mass_cap`, and
   `_begin_line_runtime` (`nodes/_otr_voice_node_common.py:490-556`) merges it
   last so the live value beats the profile default. Changing either constant
   changes every indextts2 line's key, so no audio rendered at 0.4/0.4 can
   replay as 1.0/0.56. Nothing to do -- but it was worth proving, not assuming.

7. **[CONFIRMED] The acceptance checker follows automatically.**
   `scripts/otr_voice_identity_acceptance.py:150` defaults `expect_mass_cap` to
   the imported `EFFECTIVE_EMOTION_MASS_CAP`, so it tracks the constant with no
   edit. `expect_alpha` is an opt-in string comparison and defaults to off.

## Verified since the plan was written

* The 8 stale tests are exactly the 8 named in the plan -- confirmed by running
  the three files: 3 fail in `test_cast_lock_policy_repin.py`, 4 in
  `test_voice_identity_fix.py`, 1 in `test_otr_dialogue_policy.py`.
* `EXPECTED_FAILED_NODEIDS` in `tests/conftest.py:176` is empty and
  `docs/known-failures.md` says the suite is green, so fixing these 8 needs no
  known-failure bookkeeping.

## UNVERIFIABLE at plan time

* Whether the re-rendered audition still sounds right to the operator. Only his
  ear settles that, and the ladder is the evidence that it will.
