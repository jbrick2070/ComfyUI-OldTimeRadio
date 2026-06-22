# OTR Story + Cast Fulfillment -- R1-hardened plan (pass01)
2026-06-22. Hardened from pass00 via the R1 panel (GPT-5.5 + Gemini-3.1-pro +
DeepSeek-v4-pro, ~$0.05) + Claude's code-grounded anchor + judge. Forward-only; the
diagnosis lives in pass00. This is the FULFILLMENT DESIGN that R2 will turn into a
coding plan.

## 0. The reframe (R1's central finding)
The night's failures are TWO different problems that must NOT share a fix:
- **(A) Story CRAFT** -- arc unevenness, flat lines, voice-drift, blind reroll. This
  is a prompt/critic/writer-structure problem.
- **(B) Cast + voice CONTRACT** -- engine-vs-role conflation, `voice_preset=None`.
  This is a CODE correctness bug. "You cannot prompt an LLM out of a bad system
  integration" (Gemini). It is NOT the writer's job.
R1 owns the high-level shape of both; R2 codes them.

## 1. The high-leverage creative move (A): DECOUPLE PROSE FROM METADATA
Root cause of flat lines (panel-convergent, log-consistent): the writer is asked to
satisfy prose AND `arc_phase` + `trait` + `beat_intent` + `line_job` +
`hidden_pressure` + JSON shape for EVERY line at once. Small models (mistral,
gemma-12b) discharge the JSON keys with wooden dialogue. The constraint load is the
cause, not the cure.
- **New shape:** (1) writer generates the SCENE as natural script prose, steered by
  the episode `dramatic_state` + a human-readable beat outline (the slot `line_job`s
  as guidance, not per-line JSON straitjacket); (2) a SECOND, cheap mapping pass
  (or a deterministic parser + a small model) tags the prose into `lines[]` with
  `line_id` / `char_id` / `arc_phase` / `dialogue_slot_id`. Prose first, metadata
  second.
- This is the likely answer to "is the critic miscalibrated?": probably not -- the
  WRITER APPROACH manufactures the flatness the critic then catches. Fix the approach
  before touching the critic bar.
- [verify-at-build / R2] confirm the current `OTR_LedgerScriptWriter` compose path is
  in fact per-line-constrained (the SlotContract heartbeats suggest yes).

## 2. Define "flat" operationally (A)
A line is FLAT if it does none of: change knowledge, shift pressure, move a
relationship, force/avoid a decision, or raise/clear an obstacle -- i.e. it does not
advance its slot's `line_job` or escalate its `hidden_pressure` relative to prior
lines in the same `arc_phase`. The critic must apply THIS test (not taste) so the
fix is checkable and the reroll is actionable. Per-`speaker_role` rules: announcer =
frame/transition; music_*/sfx = mood/punctuation -- they are NOT held to discharge
dialogue pressure (do not over-constrain non-dialogue rows).

## 3. Make the reroll CONVERGE (A)
The reroll bounds out because it is blind + whole-episode. Three coupled fixes:
- **Critic emits a concrete `correction_instruction` per target:** `{line_id,
  failed_field, reason, minimal_fix}` (e.g. "b012 must show A's hidden pressure of
  guilt"). No bare "target: line_12".
- **Reroll = targeted patch** of only the flagged `line_id`s (immutable ids);
  approved lines are frozen, not re-rolled.
- **Critic re-judges only the patched lines + their continuity neighbours**, not the
  whole episode -- otherwise fixing 3 surfaces 3 new (whack-a-mole; matches the
  cycle1=3 -> cycle2=3 non-convergence). Track flagged-count monotonic decrease as
  the loop's success test; bail to repair-then-ship only if it stops decreasing.
- [verify-at-build / R2] confirm `OTR_StoryCritic` currently re-scores whole-episode
  each cycle, and whether targets carry stable ids.

## 4. Voice consistency (A)
Voice-drift is separate from the missing-preset bug. Add a compact per-character
**dialogue voice bible** (syntax, diction, taboo phrases, emotional mask under
pressure, one sample line) -- distinct from the portrait-heavy `character_description`
(which is for the image model, not dialogue). Inject the voice bible into both the
scene-prose pass and the reroll instruction.

## 5. Cast + voice CONTRACT fix (B -- CODE, not prompting)
- **Never expose TTS engine names to any role-assignment prompt or field.** The role
  field takes ONLY `allowed_roster` values. Hardcode the engine->role separation in
  the casting/reviewer code (this is the `role_mismatch` source).
- **Schema invariants, fail-closed:** `speaker_role in allowed_roster`;
  `tts_model in engine_roster`; `voice_preset` REQUIRED for every character +
  announcer (reject None, force a pick from the valid preset list); archetype
  (lead/foil/support) lives in a SEPARATE field, never `speaker_role`.
- **Split `allowed_roster`:** `speaker_role` (announcer/character) vs `cue_type`
  (music_*/sfx) -- they are not the same axis (GPT). [verify-at-build] confirm the
  reviewer audits these correctly.

## 6. Keep what already works
- **Stage-direction post-scrub STAYS** (regex, deterministic, free, 100%; the 136
  scrubs are the system working, not failing). Optional later: a separate
  `performance_direction` field. NOT a must-fix.

## 7. Acceptance (quantified) + scope discipline
- Targets for the fix run: **>=70% `frozen_clean`** on a fixed sample, **0 cast-
  contract violations**, **no `voice_preset=None`**, flat-line flags down materially,
  **no `OTR_BYPASS_FREEZE_HALT`** needed.
- Validate FIRST on a MINIMAL matrix -- 1 small (gemma-12b) + 1 frontier (grok or
  gpt) at ONE word tier -- to prove the method, THEN expand to the full
  5-writer x 4-tier x 4-creativity rotation. Do not gate the build on the full matrix.
- Small models may warrant a relaxed contract vs frontier; decide after the minimal
  run (tiered acceptance, not a single bar for all).

## 8. Doc hygiene
Reconcile counts: 18 episodes reached the freeze stage (0 clean / 6 warns / 11
repair-then-ship -- the 18th, "Frozen Fury", was the in-flight one), 17 published to
obs; the "~11 failures" are SEPARATE early pre-fix run errors (act-count rejects +
freeze halts before the bypass/act=auto fixes). Keep the soak tallies as an appendix,
not in the build spec.

## R2 hand-in (coding plan)
R2 shows the panel the real code -- `OTR_LedgerScriptWriter` (compose path),
`OTR_StoryCritic` + `OTR_Reroll`, `cast_lock` + `OTR_LedgerReviewer` -- and asks for
the implementation plan for: the prose/metadata decouple (S1), the operational flat
test + per-role rules (S2), the convergent targeted reroll (S3), the voice bible
(S4), and the cast-contract code fix (S5). Verify-at-build items above are the first
grounding targets.
