# OTR Story-Quality Update Plan -- r3 hardened (wiring / integration / sequencing)

Panel: Codex gpt-5.5@high (yes-with-fixes) + agy gemini-3.5-pro (no -- same fixable
wiring). STRONG convergence: both independently caught the first-pass cap gap, the
`arc_shape` seam, undefined `run_on`, the ZeroDiv guard, JSON coercion, and the
leading/trailing-only `scrub_roster_vocative`. Claude grounded all (incl. the
`run_targeted_reroll -> compose_line` single-path + the `leak_floor:roster_vocative`
EntityPolicy dependency). Wiring is now pinned.

## 0. Invariants (carry r1/r2 + the wiring data-flow)
- v2-gate EVERY change; CPU/content; additive meta; suite+BugBible+B7 per chunk;
  prod/main GATED.
- **Data-flow (single source of truth):** `compute_episode_budget` ->
  (v2-gated) `meta["words_per_beat_range"]` stamp + first-pass `LineRequest` ->
  ledger -> reroll reconstruction -> scan. The reroll re-composes THROUGH
  `compose_line` (verified: `_otr_reroll.run_targeted_reroll` L514/L643), so G1's
  hint + keep-better fixes live in ONE place and cover both paths; only the
  `words_per_beat_range` INPUT must be threaded to first-pass AND reroll AND scan.
- **ONE shared cap helper** used by composer + reroll + scan:
  `derive_one_breath_cap(words_per_beat_range) -> int` =
  `min(max(eff_hi, 28), 60)` if range parses to (lo,hi) with hi>0 else `28`.
  JSON-coerce (meta round-trips ranges to LISTS): accept list/tuple with >=2 numeric
  elements, else (0,0)->28.

## G1 (LEAD) -- wiring pinned
- **G1.1 hint:** add a NEW `_QUALITY_COLLAPSE_HINT_V2` constant (<=240 chars):
  "Rephrase as natural spoken dialogue; split into two short sentences if needed;
  keep the specifics; drop listing/cramming; do not pad." `_quality_reroll_hint`
  selects it by `req.story_quality_v2_enabled`; the original constant is UNCHANGED
  (agy: a global edit breaks tests asserting the old hint).
- **G1.2 better-metric (v2-gated):** `import is_truncated` into
  `_otr_line_composer.py`; add pure `line_quality_defect_score(text, req) -> int`.
  When `req.story_quality_v2_enabled` is FALSE -> preserve the existing
  length-only `len(_after) < len(_q)` decision (byte-identical). When TRUE -> compare
  `score = len(flags) + 2*is_truncated(text) + (1 if _hard_clauses(text) > 3 else 0)`
  for both drafts (lower wins; ORIGINAL on tie). `_hard_clauses` (module-level,
  testable) counts `,;:` + coordinating conjunctions.
- **G1.3 dynamic cap (v2-gated):** add `LineRequest.words_per_beat_range:
  tuple[int,int]=(0,0)`. Pass it in the FIRST-PASS constructor
  (`OTR_LedgerScriptWriter` ~L4235 `_build_line_request_for_beat`,
  `=tuple(episode_budget.words_per_beat_range)`) AND reconstruct it in
  `_otr_reroll` L366 (JSON-coerced). Stamp `meta["words_per_beat_range"]` ONLY when
  `story_quality_v2_enabled` (else the off-ledger bytes change -- agy/Codex).
  `flag_one_breath` is called with `max_words=derive_one_breath_cap(range)` in the
  composer (L2319), the body-gate, AND the scan (L387). Absent/(0,0)=>28.
- **Acceptance:** `tests/test_story_quality_golden.py` (enrichment failures: plancks
  b03/b10, ledger_ink b04/b13, dance b04/b11) asserting `not is_truncated`,
  `_hard_clauses<=3`, word_count within budget.

## S2 -- coda floor (wiring pinned)
- `compose_news_coda(..., *, story_quality_v2_enabled: bool=False, arc_shape: str="")`
  (keyword-only defaults). Writer (L4770) passes both from
  `meta["story_quality_v2_enabled"]` + `str(meta.get("arc_shape") or "")` (arc_shape
  is available in writer/meta, L3563-3567).
- v2-ON only: LOCAL copy of `_NEWS_CODA_SYSTEM` + 1-2 in-context premise->bridge
  examples (never mutate the constant). **DROP attempts 2->3** (Codex: the curated
  fallback closes the gap without an extra LLM call) -- keep the existing 2 attempts.
- Fallback = an `arc_shape`-keyed CURATED template pool selected by
  `sha256(cast_seed)`, each validated by `validate_news_coda_bridge`; **if the
  arc_shape is absent OR yields zero valid templates -> fall back to the legacy
  `NEWS_CODA_POOL`** (guards ZeroDivision; preserves its tests). KEEP `NEWS_CODA_POOL`.

## S3 -- body-gate accept (wiring pinned)
- `import is_truncated` into `OTR_LedgerScriptWriter.py`. Define for each draft:
  `hard_leak = any(f in compose_flags for f in {"leak_floor:malformed_quote",
  "leak_floor:banned_source_entity"})`; `trunc = is_truncated(text)`;
  `run_on = flag_one_breath(text, max_words=derive_one_breath_cap(range))[0]`;
  `roster_caps` = a STANDALONE check (below). Accept the reroll iff grounding passes
  AND not (hard_leak OR trunc OR roster_caps); when both imperfect,
  `score = 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps` (lower wins; ORIGINAL on
  tie).
- **roster-caps as a STANDALONE body-gate check** (NOT via leak-floor): `scrub_roster
  _vocative` (L1231) is LEADING/TRAILING only, AND the leak-floor roster rule needs
  `EntityPolicy` which is NOT reconstructed on the reroll path (pre-existing gap) --
  so check directly: does the line contain an ALL-CAPS token-run that exactly matches
  an EPISODE CAST FULL NAME, anywhere? (cast list from the ledger; never any caps
  token -- NASA/UCLA safe). Add mid-line tests.

## S4 -- cliche replacement (wiring pinned)
- Add `find_cliche_phrase(text) -> str` (return the matched span; `flag_cliche` gives
  only `(bool, reason)`). Run the exact-span replacement BEFORE EVERY quality-gate
  return path (Codex: kept-reroll AND kept-original fall-through, else it misses the
  accepted-reroll branch). Curated safe-replacement map with CASE-MATCH (preserve a
  leading capital). Respect the single `_quality_repair_attempted` guard (no 2nd
  generic reroll). Else accept second-best + stamp `cliche_shipped_after_reroll`.

## S5 -- voices: scan-only "two principals" (no runtime counter)
Implement in `story_quality_scan.py` (L431): pick the two principals = the speakers
named in `character_a_wants` / `character_b_wants` (parse the name prefix) else top-2
by dialogue-line count; compute `register_overlap_ratio` over only those two. No
ledger/runtime change. The `speaks:` prompt directive already exists -- unchanged.

## S1 DEFERRED; S6 CUT.

## Build sequence (commit cadence)
1. **Shared helpers commit** (one): `is_truncated` imports (composer + writer),
   `find_cliche_phrase`, `_hard_clauses`, `line_quality_defect_score`,
   `derive_one_breath_cap` (+ JSON coercion). Suite+BugBible+B7 green, pushed.
2. **G1** (hint v2-const + v2-gated score + cap thread first-pass/reroll/scan + meta
   stamp + golden test). 3. **S2** (params + local prompt + arc-shape fallback).
4. **S3** (accept predicate + standalone roster-caps). 5. **S4** (span replace both
   paths). 6. **S5** (scan two-principals). Each: green commit + push to v2.0-alpha.

## Judgment log (r3)
- ACCEPTED (grounded, CONVERGENT both agents): first-pass cap thread (verified
  reroll-only in r2 was incomplete); arc_shape param seam (verified available
  L3563); `run_on`=flag_one_breath w/ shared cap; ZeroDiv->NEWS_CODA_POOL; JSON
  list-coercion; `scrub_roster_vocative` leading/trailing-only -> standalone mid-line
  cast-fullname check (verified L1231 + the EntityPolicy-not-reconstructed gap);
  meta-stamp + score-terms v2-gated for byte-identity (verified v2 meta is
  conditional L2635); `_QUALITY_COLLAPSE_HINT_V2` variant; S4 replace on both return
  paths (verified the early-return at L2515); drop the 3rd coda attempt (Codex).
- REJECTED: none -- r3 claims all grounded true (both agents read the seams
  correctly).
- VERIFY-AT-BUILD: first-pass cap == reroll cap == scan cap (one helper);
  flag-OFF byte-identical per sub-flag incl. NO new meta key when v2 off;
  golden-ledger asserts; the pre-existing EntityPolicy/speaker_gender reroll gaps
  (separate audit, not this plan).
