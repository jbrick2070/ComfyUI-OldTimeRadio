# OTR Story + Cast Fix -- R3 wiring-hardened plan (pass03)
2026-06-22. Hardened via R3 panel (GPT-5.5 + Gemini-3.1-pro; DeepSeek empty/length) +
Claude anchor + judge, grounded (grounding_r3.md). Implementation-ready; R4 = final
convergence check only.

## Invariant honored: NO workflow-JSON change
All fixes are internal node code (critic/reroll/reviewer run inside writer node 1;
voice in cast node 80). Add a regression check that `workflows/otr_scifi_16gb_full.json`
validates BYTE-UNCHANGED through this work (OTR_WorkflowValidator + round-trip). No
node/widget/wiring add -- nothing to wire, so nothing goes dead.

## BUILD ORDER (sequencing is load-bearing -- R3 catch)
Roles must be correct BEFORE voice fail-closed runs, and legacy rows must be migrated
BEFORE any fail-closed validation, or correct code crashes on bad legacy data.

### STEP 1 -- role_mismatch one-liner + speaker_role guarantee (was FIX 5 core)
- `nodes/_otr_ledger_reviewer.py:500`: `role = row.get("speaker_role") or row.get("tts_model") or ""`
  -> `role = row.get("speaker_role") or ""`. An engine name can no longer be read as a
  role. (This single fallback is THE source -- no upstream trace needed first.)
- Guarantee `speaker_role` is set on every row at line-row construction (character /
  announcer / cue). Where it is currently left empty is why the fallback ever fired.
- Test: a row with `tts_model='kokoro'` + empty `speaker_role` no longer produces a
  role_mismatch; rows ship with explicit speaker_role.

### STEP 2 -- cast schema migration THEN validation (ordering -- R3 catch)
Order: legacy normalization/migration -> schema validation -> cast_lock -> reviewer
audit -> TTS. Validation BEFORE migration would reject valid legacy cue rows.
- Migrate: legacy `role`/`speaker_role` in {music_open,music_close,music_inter,sfx} ->
  `cue_type`; clear `speaker_role` for cue rows; `speaker_role in {character,announcer}`
  required only for spoken rows; `archetype in {lead,foil,support}` in its own field.
- Update `OTR_LedgerReviewer` + deterministic repairs to the normalized schema.
- [verify R4] one migration for existing on-disk ledgers if any are replayed.

### STEP 3 -- voice fail-closed at node-80 OUTPUT (was FIX 2)
- In `OTR_CastLock` (node 80), make the OUTPUT contract fail-closed: before rows go to
  node 81 (OTR_BatchCharacterVoices) / 82 (OTR_AnnouncerVoice), assert every
  `speaker_role in {character,announcer}` row has non-empty `voice_preset` (assign the
  deterministic picker fallback, else NAMED raise). Independent of `cast_seed`. cue
  rows must NOT be routed to character/announcer TTS.
- Persist `cast_seed` to one canonical key; [verify R4] confirm every reader of the
  seed uses that key (`meta.cast_contract` path in cast_lock.py).
- Test: no row reaches TTS with `voice_preset=None`; seedless production ledger raises.

### STEP 4 -- critic scope + CORRECT reroll invariant (was FIX 1)
- Add `scope_line_ids: set[str] | None = None` to `run_story_critic` (def + thread to
  the reviewer). `_otr_freeze_cascade.py:754` passes None (whole-episode initial);
  `_otr_reroll.py:621` passes the patched target set. Scoped -> evaluate only those
  line_ids + continuity neighbors, neighbors keyed by stable `line_id` against the
  post-patch canonical order (validate no dup/missing id).
- CORRECTED loop invariant (R3 -- Gemini): NOT "scoped count strictly decreases".
  Instead: each cycle, the originally-targeted `line_id`s must CLEAR; any newly-failed
  neighbor is ADDED to the next cycle's scope; HALT to repair-then-ship only if
  cycle_count > MAX_REROLL_CYCLES OR the GLOBAL flag count increases. (Fixing N may
  surface N+1; that is progress, not a stall.)
- Test: a fixture where fixing line N surfaces a neighbor issue -> loop continues +
  converges, does not false-halt.

### STEP 5 -- flat rubric + failed_dimension (producer/consumer together) (was FIX 4)
- Critic PROMPT carries the explicit 5-dimension flat rubric (change knowledge / shift
  pressure / move relationship / force-or-avoid decision / raise-or-clear obstacle),
  per `speaker_role` (character only; announcer=frame; cue excluded).
- Extend the critic flag schema with `failed_dimension` (enum) AND update the
  `_otr_reroll.py` hint parser/consumer IN THE SAME CHANGE -- the hint names the
  dimension to add. Invalid/missing enum -> deterministic fallback dimension or NAMED
  parse error (never silent).

### STEP 6 -- beat-planning arc audit (re-pointed FIX 3)
- Do NOT add SceneArcContext (LineRequest already carries the arc fields). Audit the
  beat/outline planner that POPULATES beat_objective/obstacle/turn/beat_tension(1..5):
  does tension escalate across the arc? are objective/turn strong per beat? Fix weak
  beats at the PLAN; the context-rich compose then renders them. [verify R4: read the
  beat/outline generator before deciding the change.]

## Acceptance + cadence
Per-step: regression suite + Bug Bible (CLAUDE.md), commit+push the green chunk. After
STEPs 1-4 (the contained fixes) re-soak the minimal matrix (1 small + 1 frontier, 1
tier): target >=70% frozen_clean, 0 cast violations, no voice_preset=None, no
OTR_BYPASS_FREEZE_HALT. STEPs 5-6 (quality levers) re-soak + measure flat-rate/arc.
Remove the bypass stopgap once STEP 4 converges clean. Tiered bar (small vs frontier)
decided after the first re-soak.

## R4 verify-at-build targets
beat/outline planner (STEP 6); cast_seed canonical read path (STEP 3); existing-ledger
migration need (STEP 2); the critic neighbor-selection edge cases (STEP 4).
