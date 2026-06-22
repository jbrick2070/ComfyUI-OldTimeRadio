# OTR Story + Cast Fix -- R2 coding plan (pass02)
2026-06-22. Hardened via R2 panel (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, ~$0.05)
+ Claude anchor + judge, grounded against the real source (grounding_r2.md). This is
the implementable coding plan; R3 wires it.

## Scope correction (do NOT build these -- already implemented)
Prose/metadata decouple, the per-target reroll `hint`, and stable-line_id targeted
patching ALL exist. The build targets the 5 real gaps only.

## FIX 1 -- Critic scope reduction (kills the whack-a-mole). SMALL, ship first.
Real gap: `_otr_story_critic.py` re-scores the whole character ledger every call, so a
reroll that fixes 3 surfaces 3 new (cycle1=3 -> cycle2=3, never converges).
- Add an optional review scope to the critic:
  `review_ledger(..., scope_line_ids: set[str] | None = None, neighbor_window: int = 1)`
  -> flags only `scope_line_ids` + their continuity neighbors when scope is given;
  whole-episode behaviour unchanged when `None`.
- In `_otr_reroll.py`: initial critic pass = whole episode (unchanged); each reroll
  cycle re-scores ONLY the just-patched `line_id`s + neighbors. Loop invariant: the
  scoped flagged-count must STRICTLY DECREASE per cycle, else stop as non-convergent
  (-> repair-then-ship). Track the prior target set, not just the count.
- Test: a fixture where line A is flat; assert one cycle clears it and the loop does
  NOT re-flag unrelated lines.

## FIX 2 -- Voice preset fail-closed. SMALL, ship first.
Real gap: `cast_lock.py` returns early when `cast_seed is None` ("voice_preset
preserved -- no replay"), and an unmatched `char_id` keeps `None`.
- Postcondition AFTER cast-lock, INDEPENDENT of `cast_seed`: every row with
  `speaker_role in {character, announcer}` must have a non-empty `voice_preset`. If
  missing -> assign the deterministic picker's fallback for that gender/timbre; if no
  fallback resolvable -> raise a NAMED error BEFORE freeze/TTS (no silent None).
- Persist `cast_seed` into `meta.cast_contract` at cast time so production always
  replays (keep the silent-skip only for explicit legacy test fixtures).
- Test: no shipped row has `voice_preset=None`; a seedless production ledger raises,
  not skips.

## FIX 3 -- Scene-arc context on compose (the quality lever). MEDIUM.
Real gap: `compose_line()` sees only "this beat + last N lines" -- no escalation view,
so lines (and re-composed lines) land flat. Do NOT rewrite to scene-level prose
(Gemini: the system is per-line `LineRequest` by design).
- Augment the existing per-line path: add a `SceneArcContext` onto `LineRequest`
  carrying {prev-beat outcome, THIS beat's required state-delta (what must change),
  the scene's escalation target, any prior unresolved `hidden_pressure`}. Inject it
  into the compose prompt. No metadata JSON from the LLM; no new pass, no new model.
- Measure flat-rate before/after on the minimal matrix; only THEN consider a
  scene-level draft pass if the prompt-context version under-delivers.

## FIX 4 -- Operational flat = rubric-guided critic (NOT a deterministic code test).
Judge call: flatness is literary judgment (Gemini), so it is NOT a `_is_flat()` code
function -- but it MUST be consistent + targetable.
- Put the explicit rubric in the CRITIC PROMPT: a `character` line is flat unless it
  does >=1 of {change knowledge, shift pressure, move a relationship, force/avoid a
  decision, raise/clear an obstacle} AND advances its slot `line_job`.
- Structure the critic output: extend the flag with `failed_dimension` (enum of the
  five) so the `hint` names WHICH dimension to add; the composer/reroll targets it.
- Per-`speaker_role` inclusion: character -> flat rubric; announcer -> frame/transition
  rubric only; `cue_type in music_*/sfx` -> excluded from flat review.

## FIX 5 -- role_mismatch + cast schema split. R3 WIRING (trace first).
Real gap: the engine-name-in-role WRITE is upstream of `cast_lock` (which only reads
`speaker_role or role`). R3 must:
- Trace every writer of `role`/`speaker_role`/`tts_model`/`voice_preset`/`archetype`
  across the orchestrator + casting pipeline; find the one stamping an engine name.
- Define the row schema + invariant matrix: `speaker_role in {character, announcer}`;
  `cue_type in {music_*, sfx}`; `tts_model in engine_roster`; `voice_preset` required
  for character/announcer, forbidden for cue rows; `archetype in {lead,foil,support}`
  in its OWN field (never `speaker_role`). Migration rule for existing `role` rows.
- Fail-closed validation at the FIRST ledger-construction boundary AND in
  `OTR_LedgerReviewer` (reject a `tts_model` value in any role field).

## Sequencing + acceptance
1. FIX 1 + FIX 2 (small, contained, test-backed) -> commit -> re-soak the minimal
   matrix (1 small + 1 frontier, 1 tier) to measure clean-freeze rate + 0 voice-None.
2. FIX 3 + FIX 4 (the quality levers) -> re-soak -> measure flat-rate drop + arc.
3. FIX 5 lands in R3 wiring (after the trace).
Acceptance unchanged: >=70% frozen_clean on the fixed sample, 0 cast violations, no
voice_preset=None, no OTR_BYPASS_FREEZE_HALT. Tiered bar for small vs frontier decided
after step 1.

## R3 hand-in (wiring)
Trace the role-field write path (FIX 5); confirm `LineRequest` fields available to
reuse for the `SceneArcContext` (FIX 3); confirm the critic call site(s) for the
scope arg (FIX 1); sequence the workflow-JSON / node changes; define the migration for
the cast row schema.
