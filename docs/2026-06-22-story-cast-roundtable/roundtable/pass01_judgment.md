# R1 judgment log (Claude as judge)
Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.0505.

ACCEPTED (panel-convergent + grounded):
- Split (A) story-craft from (B) cast/voice CONTRACT-code bug. [all 3 + anchor]
- Decouple prose from metadata to kill flat lines (constraint overload on small
  models). [Gemini lead; GPT/DeepSeek concur] -- the answer to the anchor's
  "is the critic miscalibrated?" question.
- Operational "flat" definition (knowledge/pressure/relationship/decision/obstacle).
  [GPT+DeepSeek]
- Convergent reroll = critic emits correction_instruction + targeted patch +
  re-judge only patched/continuity. [all 3]
- Cast contract as code: engine names never in role field, voice_preset required
  fail-closed, archetype in separate field, split speaker_role vs cue_type. [all 3]
- Per-speaker_role fulfillment rules (don't over-constrain announcer/music/sfx). [GPT]
- Per-character dialogue voice bible separate from portrait description. [GPT]
- Quantified acceptance + minimal-matrix-first scope cut (1 small+1 frontier, 1 tier).
  [GPT cut]
- Reconcile episode-count numbers; move soak tallies to appendix. [GPT+DeepSeek]

JUDGE OVERRIDE / REJECTED:
- Kill stage-direction leakage at generation [GPT MUST-8, DeepSeek MUST-6] -> REJECTED
  as a must-fix. Gemini is right: the 136 post-scrubs are the system WORKING (cheap,
  deterministic, 100%). Downgraded to optional `performance_direction` field.
- "No proposed writer structure = defect" [GPT/DeepSeek] -> the problem statement
  intentionally posed the questions (that is R1's input); the synthesis now supplies
  the design. Not a defect of pass00.

VERIFY-AT-BUILD (R2 grounding targets -- not yet code-confirmed):
1. Is `OTR_LedgerScriptWriter` compose path per-line-constrained (vs already
   prose-first)? Gemini's MUST-1 hinges on this.
2. Does `OTR_StoryCritic` re-score the whole episode each reroll cycle (whack-a-mole)?
   Do reroll targets carry stable line_ids?
3. Where does the `role_mismatch` originate -- which code writes an engine name into
   the role field?
4. Why are 2/4 characters `voice_preset=None` -- the cast builder's preset-assign path.

CONVERGENCE: R1 converged cleanly on the prose/metadata decouple + the craft/contract
split. No live re-loop needed. Advance to R2 (coding) with the four verify items as
the first grounding reads.
