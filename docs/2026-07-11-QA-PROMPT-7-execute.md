# OTR QA #7 -- EXECUTE THE CLASS FIX (paste into agy AND into codex)

REVIEWER ONLY. Do NOT edit source, do NOT git add/commit/push. Write to
`qa7_<yourname>.md`. Pull first. CONFIRMED or [ASSUMPTION] on every claim.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha

## You gave me the map. Now give me the patch.

Between you, you produced the blocking-gate table: ~18 gates, ~10 of which block on notes
rather than defects. That is the map of the single class that has caused nearly every kill
on this build. I am going to fix them all in one pass. I need the patches to be right,
because a gate downgraded wrongly ships a broken episode, and a gate kept wrongly kills a
good one.

Since the last round, two more instances landed live -- and both were exactly what you
predicted:

**8. The auditor would not stop blocking on craft.** No matter how plainly the seam said
"repetition is a craft note, not a defect", the LLM auditor kept setting status=defect for
"line 2 repeats line 0's claim" -- in a 30-word script with a two-fact dossier. So I
stopped ASKING it to classify and started USING the classification it already returns: its
own schema carries `severity` ("critical" iff an invented fact or unresolvable
contradiction), `invented_fact_flags`, and `sfw_pass`. The lane now blocks on THOSE and
records the rest as notes. Sonnet cleared its audit for the first time in its existence.

**9. A gate blocking on the system's own contract.** `_spoken_error` rejects all-caps
tokens as shouted emphasis -- but this lane's characters are NAMED in all caps by schema
(`Literal["ANNOUNCER","ORUM","THESSALY","VESH"]`). The moment the Warden addressed the
Literalist by name, the gate rejected the line FOR OBEYING THE SCHEMA. Both of you flagged
`_spoken_error` as role-blind; you were right, and it was worse than role-blind.

**The principle, now proven nine times:** when an LLM must classify something, do not
argue with it in prose -- take the STRUCTURED field it already gives you and let Python
decide what blocks. And never block on a thing your own contract requires.

## JOB 1 -- the downgrade patches (the deliverable)

For EVERY gate you marked DOWNGRADE, give me the exact patch:
- file:line, the current blocking condition, the replacement condition.
- **What structured signal justifies the new gate?** (severity field? error vs warning?
  an objectively checkable property?) If a downgrade rests on prose or vibes, say so and
  mark it KEEP instead -- I would rather block wrongly than ship a broken episode.
- What must be RECORDED instead (the note, the receipt, the journal entry) so nothing is
  lost silently. Every downgrade must leave a trail.
- The test that proves the gate still catches a REAL defect after the downgrade. **This is
  the part I will judge hardest.** A downgraded gate with no test is a hole.

Priority order: the three tail finalizers (warning-as-fatal, all three lanes), `_spoken_error`
(role-blind, all lanes), Fable2's +/-20% word variance, the Codex unvoiced-cast-row gate,
the credits-roll metadata checks.

For the tail finalizers specifically: agy enumerated ten warnings from
`_otr_ledger_freeze.py`. Give me a verdict per warning -- BLOCK or NOTE -- with the reason.
And answer plainly: **is `freeze_verdict == "frozen_clean"` achievable for a content-owned
lane at all**, or does the cascade emit a benign warning by construction, making the gate
unpassable by design? Gemini published, so it must be achievable at least sometimes -- what
made the difference?

## JOB 2 -- the guard suite, ranked and specified

Codex ranked seven guards; agy specified `tests/test_lane_guardrails.py`. Converge into ONE
suite. For each guard: the exact test name, what it walks (AST? the pack JSONs? the pydantic
models?), what it asserts, and which of the nine kills it would have caught at commit time.
Two already exist (the AST no-Python-authoring guard; the seam-example-must-validate guard).
Do not re-propose those -- extend them if they have gaps.

Rank by kills-prevented. I will write the top ones tonight.

## JOB 3 -- the 720w patch, final

This is the last thing standing between us and the bake-off. All three lanes are at or near
publishing 30w. Hand me the finished patch, jointly:
1. Every file:line to raise the effective writer cap to 16384; every test that pins 8192,
   and whether each asserts runtime behavior or a bare constant.
2. The reservation formula for each whole-script pass at 720w (Codex P5/P7/P9, Gemini P4/P6,
   Sonnet P5/P6) such that prompt + output fit. Show the arithmetic.
3. Which passes set `prompt_must_fit=True`.
4. The proof the default (env unset) stays byte-identical at 8192 -- name the test.
5. The non-token risks at 720w you have already flagged: Sonnet's hardcoded 2-line-per-role
   topology, Gemini's per-scene multiplier, Fable2 refusing >=120 words, caption CPS limits.
   Which of these BLOCKS the bake-off, and which merely degrades it?

## Output (`qa7_<yourname>.md`)

JOB 1 DOWNGRADE PATCHES (with the test for each -- no test, no downgrade)
JOB 2 THE UNIFIED GUARD SUITE (ranked by kills-prevented)
JOB 3 THE 720W PATCH (final, executable)
