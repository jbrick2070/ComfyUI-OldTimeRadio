# R2 judgment (coding plan / implementability)

Panel: GPT-5.5 + Gemini-3.1-pro substantive; DeepSeek FAILED (finish_reason=length, reasoning-token
cutoff -- not re-run). Spend $0.2836. Cumulative $0.5015.

## ACCEPTED (grounded CONFIRMED)
- **Placement paradox (GPT MF2 + Gemini MF1 + my anchor MF1/MF3).** CONFIRMED vs `OTR_LedgerScriptWriter.
  run`: `build_sq_data` runs after `generate_outline`, before `init_lines_from_outline`/compose; mutating
  the outline AFTER compose would not update stamped rows. RESOLUTION: best-of-N scores candidate OUTLINES
  immediately after `generate_outline`, BEFORE `build_sq_data`/compose; keep-best; compose the winner ONCE.
  The structural rubric signals are all OUTLINE-derivable (beat intents + beat_role + conflict slots), so
  no dialogue is needed to score sameness. Fixes the internal inconsistency in pass02.
- **Outline patch path (GPT MF3 + my anchor MF1).** CONFIRMED `Outline` has title/setting/time_of_day/
  central_tension/turning_point/button + `Beat` has beat_id/speaker_role/target_words/sfx_cue/
  dialogue_slot_id; in-place surgery bypasses `stamp_dialogue_slot_ids` + `validate_outline_against_budget`.
  RESOLUTION: N candidates = N independent `generate_outline` calls (seed-varied RNG + structural-diversity
  constraint in the prompt), each FULLY validated. No in-place beat surgery in v0. (A `StructuralRevision`
  schema is the v1 option only if fresh-generation diversity proves insufficient.)
- **Rubric as a pure function (GPT MF1 + my anchor MF2).** ACCEPTED: `score_story_candidate(outline,
  sq_by_beat, meta) -> StoryScore{ungrounded_crisis_density: float, distinct_conflict_objects: int,
  has_onstage_irreversible_choice: bool, ...}` with numeric thresholds + tie-break order. No LLM grade.

## RESOLVED design split (the operator's "B+ loop" vs cheap structural scoring)
- **v0 = OUTLINE-level best-of-N (cheap, pre-compose, deterministic, free locally).** Catches STRUCTURAL
  sameness, which is the actual defect. Score outlines, keep-best, compose winner once -> no double freeze,
  no N full composes/TTS.
- **v1 = the operator's holistic "B+ dialogue" loop (Tier-2).** Post-compose, an LLM (local) reads the
  composed story; if below the bar, REGENERATE a fresh outline (back to `generate_outline`, same
  premise/seed) + recompose; bounded cap + keep-best. More costly (full recompose per fail) but local/
  bounded. Built only after v0 proves the rubric discriminates.
- **never-fail = keep-best-after-cap** at both tiers; real errors still fail LOUD.

## REJECTED / DEFERRED
- Post-compose outline mutation with partial row patching: rejected (desyncs stamped rows/slots).
- StructuralRevision schema: deferred to v1.

## CONVERGENCE
R2 converged the coding plan: outline-level best-of-N after `generate_outline`, pure deterministic scorer,
fresh-generation candidates (no surgery), compose winner once. Carry to R3 (wiring): exact insertion point
in `run()`, the local-only backend gate, and how the winner flows into the existing compose/critic/freeze
without re-entry hazards.
