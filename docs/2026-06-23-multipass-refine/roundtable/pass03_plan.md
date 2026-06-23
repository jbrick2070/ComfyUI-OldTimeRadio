# Local-Only Multi-Pass Structural Story-Refine -- R2-hardened plan (R3 input: wiring)

**Operator:** Jeffrey Brick. 2026-06-23. Hardened through R2 (GPT-5.5 + Gemini-3.1-pro + Claude anchor).

## Prerequisite (unchanged, gates the whole thing)
MEASURE L1/L2 ON-vs-OFF first (the deferred soak). If the deterministic L1/L2 already collapses
cross-episode sameness, this loop is CUT. Build only on measurable residual sameness.

## v0 = OUTLINE-level best-of-N (the build target)
Operate at the OUTLINE stage, NOT post-compose (R2 grounding: `build_sq_data` runs after
`generate_outline`, before `init_lines_from_outline`/compose; post-compose outline mutation desyncs stamped
rows). Flow, all behind a default-OFF flag (`OTR_STORY_BEST_OF_N`), local-writer-only:

1. Generate **N candidate outlines** = N independent `generate_outline(...)` calls, seed-varied
   (`sha256(seed:n)` -> the cast/style/structural RNG) + a structural-diversity constraint in the prompt,
   each FULLY validated by `validate_outline_against_budget` + `stamp_dialogue_slot_ids` (no in-place beat
   surgery). Same premise/story-seed -- structure varies, premise does not.
2. **Score each candidate** with a PURE deterministic function:
   `score_story_candidate(outline, sq_by_beat, meta) -> StoryScore` where
   `StoryScore{ungrounded_crisis_density: float, distinct_conflict_objects: int, distinct_conflict_types:
   int, has_onstage_irreversible_choice: bool, character_want_clarity: int}`. `sq_by_beat` from
   `build_sq_data` (run per candidate so the grounded conflict + beat_role are scored). Map to a legible
   grade; >= B == pass. NO LLM grade in v0.
3. **Keep-best** comparator: (pass desc, ungrounded_crisis_density asc, distinct_conflict_objects desc,
   has_onstage_irreversible_choice desc), deterministic seed tie-break. Stamp the winner + all candidate
   scores to `meta.story_quality.best_of_n`.
4. **Compose the winner ONCE** through the existing path (`init_lines_from_outline` -> compose loop ->
   `run_story_critic` + `run_targeted_reroll` -> freeze). No double freeze; the existing line-level critic
   is the second gate. Audio renders ONCE on the frozen winner.

## v1 = the operator's holistic "B+" loop (after v0 proves out)
Post-compose, a LOCAL LLM reads the composed story and grades it (B+?). If below bar: REGENERATE a fresh
outline (same premise/seed, structural-diversity constraint), recompose, re-grade; bounded by a HARD CAP
(e.g. 5) + keep-best. Local-only so passes are free; never-fail = ship keep-best after the cap.

## never-fail reconciled with no-fallbacks (2026-06-16)
"No failures" = no QUALITY-FLOOR abort: after the cap, ship the best candidate (never halt the episode for
weak story). Genuine errors (crash / missing model / malformed ledger / SFW) STILL fail LOUD. The cap
bounds runtime; keep-best guarantees a shippable result.

## Wiring questions for R3
- Exact insertion point in `OTR_LedgerScriptWriter.run()`: wrap the `generate_outline` call (section ~2707)
  in the best-of-N selector; everything downstream (budget check, F2 build_sq_data, canon, compose) runs
  on the winner unchanged.
- Local-only gate: read the RESOLVED writer backend/model (`resolved["creative_writing_model"]` + the slot
  scheduler's backend), DISABLE (N=1) on any paid/OpenRouter writer. LOUD one-line log of N + winner score.
- Determinism: each candidate's RNG seed is `sha256(f"{episode_seed}:{n}")`; the improve/grade is pure
  (no LLM in v0). Flag OFF => exactly one `generate_outline` call => byte-identical.
- Telemetry: `meta.story_quality.best_of_n = {n, winner_index, scores[], winner_grade}` (gated; no key off).
- Zero workflow-JSON change unless a selector node is wired into `otr_scifi_16gb_full.json` in the same
  change (v0 is internal to the writer node -> no JSON change; confirm in R3).

## Invariants
No disguised reroll-the-same-beats gate (the rewrite is fresh-generated structure + the gate is
deterministic). Audio frozen; flag default-OFF; golden re-baseline on enable. 100% local for the loop;
seed-keyed determinism; LOUD on real errors; keep-best + cap; UTF-8 no BOM; SFW.
