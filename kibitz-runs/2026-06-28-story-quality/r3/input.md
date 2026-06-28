# OTR Story-Quality Update Plan -- r2 hardened (coding plan / implementability)

Panel: Codex gpt-5.5@high + agy gemini-3.5-pro (both yes-with-fixes). Claude
grounded every claim vs the real `_otr_reroll.py` reconstruction (L366-403),
`_otr_line_composer.py`, `_otr_line_hygiene.py`, `OTR_LedgerScriptWriter.py`,
`story_quality_scan.py`. r2 turns the r1 directions into concrete interfaces.

## 0. Invariants (carry r1 + 3 r2 additions)
- (r1) v2-gate every shared-fn change; first-pass == reroll determinism; CPU/content
  only; additive meta keys only; suite+BugBible+B7 per chunk; prod/main GATED.
- **(NEW) Metric sync:** any gate threshold that changes at runtime MUST change the
  SAME way in `scripts/story_quality_scan.py`, or the scan lies. `flag_one_breath`
  is called with DEFAULTS by both `_quality_flags_for_line` (L2319) and the scan
  (L387) -- a dynamic cap must be threaded to BOTH.
- **(NEW) New shared-fn params are keyword-only with a back-compat default**
  (`story_quality_v2_enabled: bool = False`) so existing test callers
  (`tests/test_announcer_kill2_c3.py`) stay green and flag-OFF is byte-identical.
- **(NEW, pre-existing -- OUT OF SCOPE, verify-only) reroll reconstruction is
  PARTIAL:** `_otr_reroll.build_reroll_line_request` (L366-403) reconstructs the
  dramatic frame + `story_quality_v2_enabled` + `grounded_nouns`, but NOT
  `entity_policy`, `speaker_gender`, `beat_role`, `conflict_object`,
  `conflict_type` (agy r2 -- assumption-flagged on meta key names). If real, these
  cause pre-existing first-pass/reroll divergence -- a SEPARATE determinism bug, not
  this plan's lane. This plan only adds `words_per_beat_range` and MUST reconstruct
  it. Flag the others for a separate audit; do not fold in.

## G1 (LEAD) -- concrete interfaces
- **Hint (G1.1):** rewrite `_QUALITY_COLLAPSE_HINT` (L2293) to ~"Rephrase as natural
  spoken dialogue; split into two short sentences if needed; keep the specifics;
  drop listing/cramming; do not pad." MUST stay <=240 chars (the
  `_quality_reroll_hint` cap silently truncates). v2-gated path only.
- **Better-metric (G1.2):** `import is_truncated` (NOT currently imported in
  `_otr_line_composer.py` -- verified) + add a pure
  `line_quality_defect_score(text, req) -> int` = `len(_quality_flags_for_line) +
  2*is_truncated(text) + (1 if _hard_clauses(text) > 3 else 0)` where
  `_hard_clauses` counts `,;:` + coordinating conjunctions. Keep-better at L2503
  uses the SCORE for both drafts (lower wins; ORIGINAL on tie) instead of bare
  `len()`. So a 20-word fragment never beats a clean 35-word line.
- **Dynamic cap (G1.3):** `LineRequest.words_per_beat_range: tuple[int,int] = (0,0)`
  (sentinel); stamp `meta["words_per_beat_range"]` from
  `episode_budget.words_per_beat_range` (OTR_LedgerScriptWriter ~L3024); reconstruct
  it in `_otr_reroll` L366; `flag_one_breath` called with
  `max_words = min(max(eff_hi, 28), 60)` when range != (0,0) else 28; thread the
  SAME cap into `story_quality_scan.py` L387 (read `meta["words_per_beat_range"]`).
  (0,0)/absent => static 28 => legacy ledgers + flag-OFF byte-identical.
- **Acceptance (G1):** a NAMED golden set `tests/test_story_quality_golden.py` over
  the enrichment failures (plancks b03/b10, ledger_ink b04/b13, dance b04/b11): each
  line asserts `not is_truncated`, `_hard_clauses <= 3`, word_count within the beat
  budget. This is the gate -- not just scan counters.

## S2 -- coda floor (concrete)
- `compose_news_coda(..., *, story_quality_v2_enabled: bool = False)` (keyword-only,
  default False). Writer passes `meta["story_quality_v2_enabled"]` (L4770). v2-OFF =
  byte-identical (verify the matrix `OTR_ENABLE_STYLE_GRAMMAR=1 + STORY_QUALITY_V2=0`
  -- coda runs under `_style_grammar_on`, independent of v2).
- v2-ON only: copy `_NEWS_CODA_SYSTEM` to a LOCAL var + append 1-2 in-context
  premise->bridge examples (never mutate the module constant); attempts 2->3;
  fallback = an `arc_shape`-keyed CURATED premise-template pool (fixed bridge
  skeletons per betrayal/heist/investigation/slow_dread/setup_complication),
  selected by `sha256(cast_seed)`, validated by `validate_news_coda_bridge`. If
  zero valid templates -> FALL BACK to the existing `NEWS_CODA_POOL` (KEEP it as
  legacy data; guards the `ZeroDivisionError` agy flagged + preserves its tests).

## S3 -- body-gate accept (concrete)
- `import is_truncated` into `OTR_LedgerScriptWriter.py` (NOT imported -- verified).
- Accept `_bg_res.text` (L4528) iff grounding passes AND NOT
  (`is_truncated` OR a hard-leak compose_flag in {`leak_floor:malformed_quote`,
  `leak_floor:banned_source_entity`} OR a roster-caps hit). When both drafts
  imperfect, deterministic `score = 3*hard_leak + 2*is_truncated + 2*run_on +
  1*roster_caps`; lower wins; ORIGINAL on tie.
- Roster-caps rule: strip an embedded ALL-CAPS name ONLY when it exactly matches an
  episode cast FULL NAME (never any caps token -- NASA/UCLA safe); add mid-line
  position tests.

## S4 -- cliche replacement (concrete)
- `flag_cliche` returns only `(bool, reason)` (verified L666) -- the matched span `m`
  exists internally but isn't returned. Add `find_cliche_phrase(text) -> str` (or
  return the phrase) so the replacement matches an EXACT span.
- Respect the single quality-reroll guard (`_quality_repair_attempted`): do the
  replacement AFTER the existing one reroll (no 2nd generic reroll). Curated
  safe-replacement map with CASE-MATCHING (preserve leading-capital), else accept
  second-best + stamp `cliche_shipped_after_reroll`. Never drop into a fragment.

## S1 -- seed fidelity: **DEFER from this build** (Codex + agy)
Under-defined + weak-local/intermittent (grok stays on-premise) + the reroll seam is
already busy with G1/S2/S3/S4. If/when built: anchor source = `grounded_nouns`
(reconstructed in reroll, L400); WINDOW-level (last N character lines, zero anchors),
NOT per-line; SKIP entirely if `len(grounded_nouns) == 0` (agy: else infinite
reroll). Park as a follow-up pass.

## S5 -- voices: measurement only, REUSE the existing scan metric
`story_quality_scan.py` ALREADY emits `register_overlap_ratio` (L456, tests in
`test_story_quality_scan_r2.py`) -- do NOT add a runtime counter (Codex: byte-drift
risk for nil value). Define "two principals" = the `character_a_wants` /
`character_b_wants` speakers (parse name prefix) else top-2 by dialogue count (agy).
No reroll. (The prompt-side `speaks:` directive already exists -- unchanged.)

## S6 -- CUT (all three). (phantom_name false-positives noted; no reroll.)

## Build order: **G1 -> S2 -> S3 -> S4**; S1 DEFERRED; S5 measurement-only; S6 cut.

## Judgment log (r2)
- ACCEPTED (grounded): metric-sync to the scan (Codex, verified scan calls
  flag_one_breath defaults); `is_truncated` not imported anywhere it's needed
  (verified composer + reroll + writer); `flag_cliche` returns no phrase (verified
  L666); keyword-only-default for compose_news_coda + local-copy system prompt +
  ZeroDiv guard + keep NEWS_CODA_POOL (Codex+agy); STYLE_GRAMMAR=1/V2=0 test matrix;
  exact S3 hard-fail predicate + defect score; S4 case-matching + respect the reroll
  guard; S5 reuse existing register_overlap_ratio (Codex, verified in scan); DEFER S1
  (Codex); `words_per_beat_range` must be reconstructed in reroll (verified absent).
- PARTIAL / RE-SCOPED: agy's entity_policy / speaker_gender / beat_role /
  conflict_object reroll-reconstruction gaps are REAL omissions but PRE-EXISTING and
  assumption-flagged on meta keys -> moved to a SEPARATE determinism-audit note, NOT
  this plan (this plan only owns `words_per_beat_range`).
- REJECTED: none outright; agy's optional `sfx_cue` schema add is out of the
  content lane (ledger-schema change) -- declined.
- VERIFY-AT-BUILD: flag-OFF byte-identical per sub-flag; first-pass cap == reroll
  cap == scan cap; golden-ledger final-text asserts pass; the pre-existing reroll
  reconstruction gaps (separate audit).
