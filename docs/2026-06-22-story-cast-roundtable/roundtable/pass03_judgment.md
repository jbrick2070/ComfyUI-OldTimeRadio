# R3 judgment log (Claude as judge)
Panel: GPT-5.5 + Gemini-3.1-pro (DeepSeek empty/finish_reason=length). Spend ~$0.0469.
Running total ~$0.15.

ACCEPTED (grounded):
- role_mismatch = the `_otr_ledger_reviewer.py:500` `or row.get("tts_model")` fallback;
  the one-liner is the fix (no upstream trace needed first). [GPT MUST-1 confirms anchor]
- FIX 3 re-point to beat-planning confirmed; do NOT add SceneArcContext. [GPT MUST-4 +
  Gemini]
- Sequencing is load-bearing: STEP 1 role schema BEFORE STEP 3 voice fail-closed (else
  node 80 crashes on a kokoro-role); migration BEFORE validation (else legacy cue rows
  fail). [Gemini MUST-1 + GPT MUST-2/3] -> build order rewritten.
- CORRECTED reroll invariant: not "scoped count strictly decreases" (false-halts when
  fixing N surfaces N+1); instead targeted ids must clear + newly-failed neighbors join
  next scope + halt only on cycle cap OR global-count increase. [Gemini MUST-2] -- sharp
  catch, my anchor's invariant was naive.
- failed_dimension critic-output + reroll-parser must change TOGETHER; invalid enum ->
  fallback/named error. [GPT MUST-7 + Gemini MUST-3]
- Voice fail-closed at node-80 OUTPUT boundary (before TTS 81/82), cue rows never to
  character TTS. [GPT MUST-8]
- No workflow-JSON change + add a no-drift regression check. [GPT MUST-10 confirms anchor]

VERIFY-AT-BUILD (R4):
- The beat/outline planner (STEP 6 -- the only un-grounded step + the biggest arc lever).
- cast_seed canonical read path (STEP 3).
- existing-ledger migration need (STEP 2).

CONVERGENCE: R3 converged on a sequenced, code-internal, grounded build plan with two
real corrections (invariant, ordering). Advance to R4 convergence check.
