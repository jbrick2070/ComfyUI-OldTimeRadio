# STEP 2 scope conflict -- cast/cue schema: "migrate THEN validate"

## Context
OTR is an offline ComfyUI text->radio-drama pipeline. A 4-round roundtable
(docs/2026-06-22-story-cast-roundtable/) converged a 6-step "Story + Cast Fix"
to stop the night soak's `0/18 frozen_clean` churn. STEP 1 (drop the
`tts_model` fallback that read a TTS engine name as a role; guarantee every line
row has a speaker_role) is SHIPPED + green (suite 4960/34, Bug Bible 16/7/3).

This document is a FOCUSED design review of STEP 2 ONLY, because grounding the
plan against the real code found that STEP 2 as literally written targets fields
that do not exist and would break a frozen schema invariant.

## STEP 2 as literally written (pass04_plan_FINAL.md)
> cast schema: migrate THEN validate. Move legacy {music_open, music_close,
> music_inter, sfx} role values to `cue_type`; clear `speaker_role` on cue rows;
> require `speaker_role in {character,announcer}` only for spoken rows;
> `archetype in {lead,foil,support}` in its OWN field. (Validation before
> migration would reject valid legacy cue rows.)

Acceptance for STEPs 1-4: >=70% frozen_clean on a minimal re-soak matrix, 0
cast-contract violations, no voice_preset=None.

## Grounded facts (verified against the real Windows source this session)
1. **`archetype` and `cue_type` exist NOWHERE in the codebase** (repo-wide grep,
   *.py). No producer, no consumer. They are net-new fields the plan invents.
2. **Cast rows carry no role field at all.** `OTR_LedgerScriptWriter._build_cast_rows`
   (~L1466-1487) emits exactly: char_id, name, character_description, gender,
   voice_preset, line_count, word_count. No speaker_role, no tts_model, no role,
   no archetype. (A separate streamed-partial path, story_orchestrator
   ._emit_partial_ledger -> set_lines, is what produced rows missing speaker_role;
   STEP 1 already closed that on the line side.)
3. **`music_*`/`sfx` are LINE-row `speaker_role` values, not cast values**, and
   they are load-bearing across the frozen ledger schema:
   - `OTR_LedgerScriptWriter.NON_VOICED_ROLES = {music_open,music_close,music_inter,sfx}`
     drives line construction (~L3214, L3705-3727).
   - `_otr_speaker_role.py`: `is_never_humo_role` / `is_music_role` /
     `VALID_SPEAKER_ROLES` route video/HuMo/captions off these exact strings.
   - `_otr_ledger_scrub.is_spoken_role` already distinguishes spoken
     (character/announcer) vs cue (music/sfx) rows from `speaker_role`.
   - Memory + CLAUDE.md: the ledger `{cast,lines,meta}` schema is FROZEN
     downstream (audio is the first consumer; `test_audio_byte_identical`).
4. **The plan's own invariant:** "No node/widget change; **speaker_role is the
   ONLY role source**." Moving music/sfx OUT of speaker_role into a new `cue_type`
   directly contradicts this and the frozen-schema rule.
5. **No cast-contract validator rejects cue rows.** The cast auditor
   (`_otr_ledger_reviewer.audit_cast_contract`) renders the cast table +ALL+
   lines to an LLM; cue rows are not cast members and there is no Python
   validator that raises on a music/sfx role. The "validation before migration
   would reject valid legacy cue rows" premise is unverified -- the plan's own
   residual flags STEP 2's migration need as UNVERIFIED.
6. **STEP-1 interaction (real, must be addressed):** after STEP 1,
   `_render_cast_contract_table` renders `role=''` for every cast member (cast
   rows have no speaker_role). The auditor now sees no contract role to compare a
   line's speaker_role against. `audit_cast_contract` also feeds the auditor ALL
   lines including cue rows (no spoken-only filter).

## The three candidate designs

### Option A -- Grounded-contained (proposed). Prompt-boundary only; zero schema/JSON change.
- In `_render_cast_contract_table`, derive a real cast-member role for the
  auditor: `role = row.get("speaker_role") or ("announcer" if
  name.strip().upper()=="ANNOUNCER" else "character")`. Cast members structurally
  ARE spoken roles; this gives the auditor a real contract role instead of ''.
- In `_render_lines_for_audit`, list ONLY spoken rows (`is_spoken_role`) -- the
  cast contract governs only character/announcer speakers; cue rows cannot drift
  against a cast, so excluding them prevents spurious speaker_unknown/role_mismatch.
- Drop the ungrounded `cue_type`/`archetype` fields entirely (nothing reads them).
- Net effect: serves "0 cast violations" using the EXISTING speaker_role
  taxonomy; honors "speaker_role is the ONLY role source"; no wire-schema/JSON/
  golden change; small + test-backed.

### Option B -- Literal schema migration (pass04 as written).
- Add `cue_type` + `archetype` fields; migrate music/sfx speaker_role on line
  rows into `cue_type`; clear speaker_role on cue rows.
- Cost: breaks the FROZEN {cast,lines,meta} schema and every NON_VOICED_ROLES /
  is_music_role / is_never_humo_role consumer (writer, scene_sequencer, captions,
  video routing); large blast radius for fields nothing currently reads;
  contradicts the plan's own "speaker_role is the ONLY role source" invariant.

### Option C -- Skip STEP 2 as a no-op.
- Treat the premise as absent in code; proceed to STEP 3 (voice fail-closed) +
  STEP 4 (critic convergence); revisit only if the re-soak still shows cast
  violations. Risk: leaves the STEP-1 `role=''` interaction (fact 6) unaddressed,
  which could itself raise role_mismatch flags.

## The question for the panel
Given the grounded facts, which design best achieves the acceptance criteria
(0 cast-contract violations, no regression to the frozen schema) at minimum risk?
Is Option A's "derive real cast role + audit spoken rows only" the right
root-cause fix, or is there a defect in A (e.g. a case where a cast member is
legitimately neither character nor announcer, or a cue row that legitimately
needs cast auditing)? Should `cue_type`/`archetype` be added now for a future
need, or is that speculative scope to cut? Is skipping (C) safe given fact 6?

## Invariants the answer MUST respect
- No workflow-JSON / node / widget change. Ledger {cast,lines,meta} schema FROZEN
  (audio byte-identical). speaker_role is the ONLY role source. Fail-closed, never
  silent. 100% local. Regression suite + Bug Bible green per chunk.
