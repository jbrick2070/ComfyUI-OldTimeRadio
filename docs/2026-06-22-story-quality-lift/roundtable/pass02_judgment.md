# R2 JUDGMENT (coding plan) -- accepted / rejected / verify-at-build

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.1257 (+ ~$0.083 wasted on a first run that
ran UNGROUNDED due to a comma-collapse in the grounding arg; killed + relaunched grounded).

## ACCEPTED (folded into pass02_plan), with grounding
- **DEFECT 3 circular-dependency** (Gemini mf#1) -- VERIFIED: `init_lines_from_outline` derives char_id
  FROM role (761-766), so coercion there is impossible/no-op; b011's c02+announcer is a POST-init
  mutation (role_mismatch repair). Moved coercion to the repair guard + set_lines + a pre-freeze sweep.
- **DEFECT 1 classifier = extend `_NARRATION_VERBS`** (GPT mf#5) -- VERIFIED against the real file: the
  frozenset (136-144) holds paces/gazes/stares/sighs/... and LACKS adjusts/clutches/taps/tightens/
  overrides; `detect_narration_self_address` (148-169) is first-word-anchored. Plan extends the verb set
  + reuses `_PRONOUN_ROOTS`/`_DIALOGUE_STARTER` guards for the outside-quote classifier.
- **Double-quote-only scanner + odd-count hard abort** (Gemini mf#3, GPT mf#6) -- ignore single quotes/
  apostrophes (scare-quotes 'The Chronicle'); odd `"` count -> return unchanged (routes b015 to reroll).
- **DEFECT 2 repair = needs_full_rerun ONLY + input mutation + bound** (Gemini cut#1 + mf#2, GPT mf#1/#12,
  DeepSeek mf#2) -- cut "outline re-intent"; hint must ride `meta["coherence_hints"]` re-injected into the
  new ledger (JSON frozen -> no new port); max 1 rerun. Flagged as the R3 make-or-break.
- **b015/b017 acceptance contradiction** (GPT mf#3) -- floor guarantees 0 for the balanced class; reroll
  best-effort for the rest; survived leak -> CI FAIL, production ships LOUD.
- **One coercion helper + all-write-point audit** (GPT mf#8, DeepSeek mf#5); **CI invariant excludes
  music/sfx** (GPT sf#6); **detect signature** `detect_stage_business_for_reroll(text, speaker_name)`
  (GPT mf#4); **ordering + idempotence** (GPT mf#7); **well-formedness `.strip(" ,;-")` + terminal-punct
  assert** (Gemini sf#1, GPT sf#2/#8); **critic StanceIssue typed contract + meta caveat** (GPT mf#2);
  **NO-OP fixture + counters + concrete asserts** (GPT mf#10, Gemini sf#2, DeepSeek sf#1); **no-bypass
  re-smoke is manual not a build gate** (GPT mf#11); **scan imports same detector** (GPT sf#3). All folded.

## REJECTED / CORRECTED (judge)
- **All three "no" verdicts -- maturity, not a code defect.** pass01 deliberately deferred the detection
  primitive + repair mechanism to R2 (its stated open questions). R2 closes them; that is the campaign
  working as designed, not a flaw to action separately.
- **GPT mf#2 "do not rely on meta unless the critic issue schema supports it"** -- partially corrected:
  the LEDGER schema is frozen, but the CRITIC REPORT models are not the ledger schema, so a typed
  `StanceIssue` on the report (riding `meta.story_critic_report`) is allowed. Folded with that distinction.
- **"Coerce at init_lines_from_outline" (my pass01 + GPT mf#8's three-site list)** -- rejected at init per
  Gemini's circular-dependency catch; init is consistent by construction.

## VERIFY-AT-BUILD / R3
- needs_full_rerun meta-survival across the reset + `_otr_outline` reading `meta["coherence_hints"]` with
  NO JSON change (DEFECT 2 make-or-break; decides auto-repair vs detection-only v1).
- Exact Tier-1 prompt-builder function; audit ALL `speaker_role` write points; `FailedDimension` is a
  Literal (adding "stance" non-breaking); ledger row `meta` is mutable/serializable; JSON no-drift hash target.

## CONVERGENCE CALL
R2 CONVERGED on the codeable algorithm: DEFECT 1 fully specified (extend `_NARRATION_VERBS` + double-quote
scanner + classifier + well-formedness + idempotence); DEFECT 3 coercion sites fixed (Gemini catch);
DEFECT 2 = detection v1 + a needs_full_rerun repair STRETCH gated on the R3 meta-survival check. Remaining
items are all WIRING/integration -> exactly R3. Advance.
