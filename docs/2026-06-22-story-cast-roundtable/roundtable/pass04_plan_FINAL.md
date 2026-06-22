# OTR Story + Cast Fix -- FINAL hardened plan (pass04 / converged)
2026-06-22. Output of a 4-round roundtable: R1 GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro;
R2 same; R3 GPT + Gemini (DeepSeek empty); R4 anchor-only (panel call hung, killed --
convergence already reached). Claude = code-grounded panelist + sole judge throughout.
Total OpenRouter spend ~$0.15. All claims grounded against the real source
(grounding_r2.md / grounding_r3.md). NO workflow-JSON change in the whole plan.

## The reframe the roundtable produced
The night's "0/18 clean freezes" is NOT one problem. Three of the panel's headline
fixes were already implemented (prose/metadata decouple; the per-target reroll hint;
targeted patching) -- grounding caught that. The REAL, code-located defects:
- a 1-line cast bug (engine name read as a role),
- a whack-a-mole critic (re-scores the whole episode each reroll),
- voice presets that can ship None,
- a flat-line gate with no shared definition,
- and an arc that is uneven UPSTREAM in beat-planning (the composer is already
  context-rich, so the lever is the planner, not the line writer).

## BUILD ORDER (sequencing is load-bearing)
Roles correct -> voice fail-closed -> reroll convergence -> quality levers. Each step:
regression suite + Bug Bible, commit+push the green chunk (CLAUDE.md).

### STEP 1 -- role_mismatch (1 line) + speaker_role guarantee  [trivial]
`nodes/_otr_ledger_reviewer.py:500`: drop the `or row.get("tts_model")` fallback ->
`role = row.get("speaker_role") or ""`. Guarantee every line row gets an explicit
`speaker_role` at construction. Test: a `tts_model='kokoro'` row with empty
speaker_role no longer raises role_mismatch.

### STEP 2 -- cast schema: migrate THEN validate  [small]
Order: legacy normalization/migration -> validation -> cast_lock -> reviewer -> TTS.
Move legacy {music_open,music_close,music_inter,sfx} role values to `cue_type`; clear
`speaker_role` on cue rows; require `speaker_role in {character,announcer}` only for
spoken rows; `archetype in {lead,foil,support}` in its OWN field. (Validation before
migration would reject valid legacy cue rows.)

### STEP 3 -- voice preset fail-closed at node-80 OUTPUT  [small]
In `OTR_CastLock` (node 80), before rows go to node 81 (BatchCharacterVoices) / 82
(AnnouncerVoice): every `speaker_role in {character,announcer}` row MUST have a
non-empty `voice_preset` (deterministic picker fallback, else NAMED raise),
independent of `cast_seed`; cue rows never routed to character/announcer TTS. Persist
`cast_seed` to one canonical key (verify the read path). Test: no row reaches TTS with
voice_preset=None; seedless production ledger raises.

### STEP 4 -- critic scope + CORRECT reroll convergence  [contained -- high leverage]
Add `scope_line_ids: set[str] | None = None` to `run_story_critic` (def + thread to
reviewer). `_otr_freeze_cascade.py:754` -> None (whole-episode initial);
`_otr_reroll.py:621` -> the patched target set. Scoped => evaluate only those line_ids
+ continuity neighbors (keyed by stable line_id on the post-patch canonical order;
reject dup/missing id). CONVERGENCE INVARIANT (R3 -- the naive "strict decrease"
false-halts): each cycle the originally-targeted line_ids must CLEAR; newly-failed
neighbors are ADDED to the next cycle's scope; HALT to repair-then-ship only if
cycle_count > MAX_REROLL_CYCLES OR the GLOBAL flag count increases. Test: fixing line N
that surfaces a neighbor issue converges, does not false-halt.

### STEP 5 -- flat rubric + failed_dimension (producer+consumer together)  [quality]
Critic PROMPT carries the explicit flat rubric per speaker_role: a `character` line is
flat unless it does >=1 of {change knowledge, shift pressure, move relationship,
force/avoid decision, raise/clear obstacle} AND advances its slot `line_job`; announcer
= frame/transition; cue = excluded. Extend the critic flag schema with `failed_dimension`
(enum) AND update the `_otr_reroll.py` hint parser/consumer in the SAME change; invalid/
missing enum -> deterministic fallback or NAMED error (never silent). It is rubric-
guided LLM judgment, NOT a deterministic code test (flatness is literary).

### STEP 6 -- beat-planning arc audit  [the biggest lever -- GROUND FIRST]
The composer already carries arc context (dramatic_question, beat_objective/obstacle/
turn, beat_tension 1..5, next_turn, outline_spine) -- do NOT add a SceneArcContext. The
"uneven arc" (50/55) lives in the BEAT/OUTLINE PLANNER that POPULATES those fields.
FIRST SUB-TASK (go/no-go): read the beat/outline generator + measure whether
beat_tension actually escalates across the arc and whether objective/turn are strong
per beat. Only then define the planner change. Do NOT ship STEP 6 as a vague
aspiration.

## Acceptance + cadence
Build STEPs 1-4 (small, contained, high-confidence) FIRST -> re-soak the minimal matrix
(1 small e.g. gemma-12b + 1 frontier e.g. grok, ONE word tier): target >=70%
frozen_clean, 0 cast-contract violations, no voice_preset=None, no OTR_BYPASS_FREEZE_HALT.
Measure. THEN ground+decide STEP 6 with real data on whether the arc is still uneven
after the mechanical fixes; STEP 5 pairs with it. Remove the OTR_BYPASS_FREEZE_HALT
stopgap once STEP 4 converges clean. Decide the tiered small-vs-frontier bar after the
first re-soak.

## Invariants guarded
No workflow-JSON / node / widget change (add a no-drift regression check). Fail-closed,
never silent. `speaker_role` is the ONLY role source. Reroll preserves approved lines.
Regression suite + Bug Bible per green chunk; commit+push to v2.0-alpha same session.

## Residual / verify-at-build
STEP 6 planner read (go/no-go); STEP 3 cast_seed canonical key; STEP 1 the line-row
construction site that leaves speaker_role empty; STEP 2 whether any on-disk ledgers are
replayed (migration need).
