# Local-Only Best-of-N Structural Story-Refine -- FINAL build-ready plan (R4 converged)

4-round live roundtable: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude code-grounded judge + panelist.
Spend: R1 $0.2179 + R2 $0.2836 + R3 $0.3803 + R4 (below). Artifacts: `pass00..pass04` + `pass0N_judgment.md`
+ per-model reviews. Origin idea: operator Jeffrey -- a pre-audio LLM that keeps refining the story
("is this a good story, 10th-grade B+? how do we make it better?") so a run never hard-fails on weak
quality, run ONLY on local writers (passes are free).

## What the roundtable changed (the value)
The naive "LLM grades and rerolls until good" is the exact thing this session's panel already rejected
(the weak model rephrases the same standoff). All three frontier models, grounded against the real code,
independently converged the idea into something sound AND caught real bugs in the intermediate design
(scoring with `build_sq_data` would zero out the very metric it scores on; there is no `episode_seed` to
key determinism on; the RNG was never wired into the LLM). The result is a deterministic, local-only,
structural **best-of-N OUTLINE selector** -- not a QA-reroll gate.

## PREREQUISITE (gates the whole build)
Run the L1/L2 ON-vs-OFF measurement soak (already deferred to the operator). Quantify residual
cross-episode sameness (`meta.story_quality.ungrounded_crisis` density + distinct conflict counts). **If
L1/L2 already collapses the sameness, this is CUT.** Build only on measurable residual sameness.

## v0 -- outline-level best-of-N (the build target)
Wrap the single `generate_outline` call in `OTR_LedgerScriptWriter.run()` (~L2707) in a selector; flag
default-OFF; local-writer-only. Everything downstream runs UNCHANGED on the winning `Outline`.

**Flag.** `OTR_STORY_BEST_OF_N`: unset/0/1 => DISABLED (one call, byte-identical); integer >=2 => N,
clamped to a max of 6. Read once at the top of `run()`.

**Local-only gate.** If `resolved["creative_writing_model"].startswith(("openrouter:", "comfy:"))` -> force
N=1 (both paid lanes). LOUD one-line log of the clamp + the resolved handle.

**Selector** `select_best_outline(generate_fn, outline_req, *, cast_seed, n, meta) -> Outline`:
1. For `i in range(n)`: derive `h = sha256(f"{cast_seed}:outline:{i}").hexdigest()`; seed the LLM RNG
   immediately before generation -- `torch.manual_seed(int(h,16) % 2**64)` + `random.seed(int(h,16) % 2**32)`;
   build a per-candidate `OutlineRequest` with a new optional `diversity_hint` (i==0 => "" so candidate 0 ==
   the exact current production outline; i>=1 => a short structural-variation instruction). Call
   `generate_outline(...)` inside `try/except OutlineFailedError` -> on failure log LOUD + `continue`.
2. **Score each candidate with a PURE function (NO `build_sq_data`, no mutation):**
   `score_outline(outline, meta, roster) -> StoryScore{ungrounded_crisis_density: float,
   distinct_conflict_nouns: int, premise_grounding: float}` where
   - `ungrounded_crisis_density` = sum over voiced beats of `count_ungrounded_crisis(beat.intent, grounded)`
     / total voiced-intent words (grounded palette = `premise_noun_palette(roster, premise, *premise_texts)`)
     -- computed on the RAW intents BEFORE any grounding (lower is better);
   - `distinct_conflict_nouns` = distinct premise-grounded content nouns across the beat intents (higher better);
   - `premise_grounding` = fraction of beat intents that reference a premise/roster noun (higher better).
   (`character_want_clarity` CUT -- no wants data at this stage.) Map to a legible grade; >= B is informational.
3. **Keep-best comparator** (only fields actually used): `(ungrounded_crisis_density asc,
   distinct_conflict_nouns desc, premise_grounding desc)`, deterministic tie-break on candidate index. If
   ALL N candidates failed to generate, fall through to ONE normal `generate_outline` (LOUD) -- never-fail.
4. Return the winning `Outline`. Downstream (budget check, F2 `build_sq_data` ONCE, canon, compose,
   `run_story_critic`+reroll, freeze, audio) runs unchanged on it -> NO double `build_sq_data`, NO double
   freeze, audio renders once.

**Diversity hook.** Add `diversity_hint: str = ""` to the `OutlineRequest` dataclass; render it in
`_otr_outline._build_user_prompt` only when non-empty (empty => byte-identical prompt). The selector sets it
per candidate (a structural-variation instruction, e.g. "open on the personal stake, not the institutional
threat"). NOT in-place beat surgery.

**Telemetry (gated, merged).** `sq = meta.setdefault("story_quality", {}); sq["best_of_n"] = {n, winner_index,
scores: [plain-dict per candidate], winner_grade, clamp_reason}` -- plain JSON primitives only; never replace
the `story_quality` dict (consistent with the L5a setdefault/update rule).

**Do NOT re-validate** candidates -- `generate_outline` already runs `validate_outline_against_budget` +
`stamp_dialogue_slot_ids`; just assert each returned candidate has slot ids + passes budget (test).

## v1 -- the operator's holistic "B+ until good" loop (DEFERRED, separate project)
Post-compose, a LOCAL LLM grades the composed story; if below bar, regenerate a fresh outline (same
premise/cast_seed-keyed) + recompose; HARD CAP + keep-best; local-only so passes are free; never-fail = ship
keep-best after the cap. Built only after v0 proves the deterministic rubric discriminates. (Cut from v0 by
unanimous panel: it needs recompose/regrade/freeze orchestration -- a distinct integration.)

## never-fail reconciled with the no-fallbacks rule (2026-06-16)
"No failures" = no QUALITY-FLOOR abort: ship keep-best after the cap, never halt an episode for weak story.
GENUINE errors (crash / missing model / malformed ledger / SFW / all-candidates-failed-AND-the-fallback-
call-failed) STILL fail LOUD. The cap bounds runtime; keep-best guarantees a shippable result.

## Invariants (may not break)
- Not a disguised reroll gate: candidates are FRESH-GENERATED structures + the selector gate is
  DETERMINISTIC (pure scorer), never "ask the same model to try again on the same beats".
- Audio spine frozen (`test_audio_byte_identical`); flag default-OFF => byte-identical (assert: exactly one
  `generate_outline` call + no `best_of_n` key when disabled); golden re-baseline only if a future change
  alters shipped text.
- 100% local for the loop (disabled on openrouter:/comfy:); determinism (cast_seed-keyed; pure scorer);
  LOUD on real errors; keep-best + hard cap; UTF-8 no BOM; SFW.
- Zero `otr_scifi_16gb_full.json` change: v0 is internal to the writer node. (If a future version exposes N
  as a widget, it goes IN the JSON in the same change.)

## Build order
1. PREREQUISITE measurement soak (operator GPU) -- decide go/no-go.
2. `diversity_hint` on OutlineRequest + `_build_user_prompt` render (flag-off byte-identical test).
3. `score_outline` pure scorer + `StoryScore` + tests (raw-intent metrics, no mutation).
4. `select_best_outline` selector + the `run()` wrapper + local-gate + flag parsing + telemetry +
   never-fail fallthrough; flag-off call-count==1 + no-key tests; full suite + Bug Bible; commit+push.
5. Small local re-soak with `OTR_STORY_BEST_OF_N=3` -> measure sameness drop vs baseline.
6. (DEFERRED, operator) v1 B+ loop.

## Verify-at-build checklist
1. `generate_outline`'s RNG is actually re-seeded per call by the global torch/random seed (no generator
   threading bleed) -- the one UNVERIFIABLE from R3.
2. `count_ungrounded_crisis` on RAW intents is non-zero on a real outline (else the metric can't discriminate).
3. Flag-off: exactly one `generate_outline` call; no `meta.story_quality.best_of_n` key; prompt byte-identical.
4. `build_sq_data` runs exactly once (on the winner); `_enrich_intent` never double-appends.
5. Local gate clamps N=1 on `openrouter:`/`comfy:` BEFORE the first candidate call.

## R4 build-fixes (small/specific; folded in -- no architecture change)
- **PREREQUISITE go/no-go is operator-judged + written:** proceed only on an explicit operator written
  go/no-go after reviewing the soak table (baseline vs L1/L2 density + distinct conflict counts over M
  episodes); no implicit threshold.
- **Drop "candidate 0 == exact current production outline":** reseeding the RNG changes the sampling stream,
  so candidate 0 cannot byte-match today's outline. Candidate 0 just uses `diversity_hint=""` (byte-identical
  PROMPT). When the loop is disabled (effective_n < 2) the existing single `generate_outline` path runs
  exactly once (no selector entry) -- THAT is the byte-identical path.
- **Local clamp short-circuits the selector:** after flag-parse + local clamp, call the selector ONLY when
  `effective_n >= 2`; else run the current single path once (prevents 2 paid calls on candidate-0+fallback).
- **Selector signature:** `select_best_outline(generate_outline_fn, outline_req, *, cast_seed, n, meta,
  roster)` where `generate_outline_fn(req)` wraps `_OTRO.generate_outline(creative_generate_fn, req,
  creative_repo_id=resolved["creative_writing_model"])`. `roster` = `outline_req.character_cast` (canonical).
- **`torch` is a LOCAL import** inside `select_best_outline` (the file forbids module-level torch, ~L440).
- **Division-by-zero guards:** `max(1, total_voiced_intent_words)`, `max(1, total_voiced_beats)`.
- **Comparator (exact key for `min`):** `key=lambda c: (c.score.ungrounded_crisis_density,
  -c.score.distinct_conflict_nouns, -c.score.premise_grounding, c.index)`.
- **`distinct_conflict_nouns` defined mechanically (no POS tagger):** tokenize voiced-beat intents with the
  `[A-Za-z][A-Za-z'-]{2,}` rule, casefold, count distinct tokens present in
  `premise_noun_palette(roster, outline.premise, *_OTRSQL12.premise_texts(meta))`.
- **Never-fail fallback is deterministic:** if every candidate raised, run one `generate_outline` with the
  `i=0` seed + `diversity_hint=""` (not "normal"); if THAT raises too, fail LOUD.
- **Flag parse:** `os.environ.get("OTR_STORY_BEST_OF_N","0")`; blank/non-int/<=1 => disabled (LOUD warn on
  non-int); int >=2 => `min(value, 6)`.
- **Telemetry exact shape:** `{"requested_n", "effective_n", "winner_index", "scores":[{candidate_index, ok,
  metric-fields | error_type}], "clamp_reason"}`; CUT `winner_grade`/">= B" from v0 telemetry (unused by the
  comparator). `scores` length == attempted n (failed candidates carry `ok:false`).
- Scorer imports `count_ungrounded_crisis`, `premise_noun_palette`, `premise_texts` from
  `_otr_story_quality_l12` (public; `premise_texts` was promoted public in the L1/L2 ship).
