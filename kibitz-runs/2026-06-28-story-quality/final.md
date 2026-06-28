# OTR Story-Quality Update Plan -- FINAL (kibitz-converged, build-ready)

3-way kibitz: Codex `gpt-5.5`@high + Antigravity `gemini-3.5-pro` + Claude (Cowork)
as code-grounded panelist AND sole judge. 8 agent calls (2 agents x r1-r4), $0 (100%
local). Evidence: the 27-episode all-visualizer overnight soak + 3 operator-directed
enrichment renders (mistral-720, grok-720, grok-1340) + the budget/length probes.
CONVERGED at r4 (only residual refinements, no new architecture).

## What the panel corrected (grounded; the big wins of the kibitz)
- The DRAFT misstated existing code on 3 fixes -- all reframed from "add" to
  "fix/enforce": S1 dramatic binding ALREADY exists (compose_line L1416-1504 + the L2
  deflection contract + reroll reconstruction); S5 `speech_signature` is ALREADY
  threaded (`build_voice_card` `speaks:` L1129); G1 keep-better ALREADY exists
  (L2495-2528) and already covers one_breath/anchor.
- The true LEAD root cause (G1) is grounded: `_QUALITY_COLLAPSE_HINT` (L2293) literally
  says *"Rewrite ... under ~20 words, using at most one concrete detail"* -> it FORCES
  the compression that turns rich lines into noun-salad, AND (with the 14-beat
  skeleton x the ~22-28-word `one_breath` cap) hard-bounds every episode at ~210-310
  voiced words regardless of `target_words`. Length + craft share ONE root cause.
- Length is in scope as a SIDE EFFECT of G1 (operator 2026-06-28) -- never padded;
  the ~1363-word structural ceiling (`BEAT_WORD_HARD_MAX=80` x beats) is left intact.

## Scope / invariants
CPU/content only (writer/composer/critic CONTENT); NO workflow JSON, NO GPU, NO node
INPUT_TYPES/widget change. Every behavior change is `story_quality_v2`-gated and
flag-OFF byte-identical (`test_audio_byte_identical` green; NO new `meta` key stamped
when v2 off). Reuse the existing reroll loop (`compose_line`;
`run_targeted_reroll`->`compose_line` is one path -- a fix lands once, covers both).
Additive `meta` keys only. UTF-8 no BOM; SFW. Suite + Bug Bible + B7 per chunk;
commit+push per green chunk to `v2.0-alpha` ONLY; prod/main + tags GATED.

## The shared cap (single source of truth)
`derive_one_breath_cap(words_per_beat_range) -> int` in the stdlib LEAF
`nodes/_otr_line_hygiene.py` (so composer + reroll + scan import it with no cycle):
JSON-coerce the range (meta round-trips to a LIST; accept list/tuple with >=2 numeric
elements else `(0,0)`); return `min(max(eff_hi, 28), 60)` when hi>0, else `28`.
Used by `flag_one_breath(max_words=...)` in the composer (L2319), the body-gate, AND
`story_quality_scan.py` (L387). Absent/`(0,0)` => 28 (legacy + flag-off identical).

## G1 (LEAD, BUILD FIRST) -- stop the gates over-correcting (length + craft)
1. **Hint:** NEW `_QUALITY_COLLAPSE_HINT_V2` (<=240 chars): "Rephrase as natural spoken
   dialogue; split into two short sentences if needed; keep the specifics; drop
   listing/cramming; do not pad." `_quality_reroll_hint(flags, story_quality_v2_enabled)`
   (UPDATE its signature, L2336; pass the flag from L2462) selects it on the v2 path;
   the original constant is UNCHANGED.
2. **Better-metric (v2 path only):** `import is_truncated` into `_otr_line_composer.py`;
   add `_hard_clauses(text)` (count `,;:` + FANBOYS `for|and|nor|but|or|yet|so` on `\b`,
   case-insensitive; module-level + unit-tested) and `line_quality_defect_score(text,
   req)`. v2-OFF keeps the existing `len(_after) < len(_q)` decision byte-identical;
   v2-ON compares `len(flags) + 2*is_truncated + (1 if _hard_clauses>3 else 0)` (lower
   wins, ORIGINAL on tie) -- so a 20-word fragment never beats a clean 35-word line.
3. **Dynamic cap (v2 path only):** `LineRequest.words_per_beat_range: tuple[int,int] =
   (0,0)`; pass it in the FIRST-PASS constructor (`OTR_LedgerScriptWriter`
   `_build_line_request_for_beat` ~L4235, `= tuple(episode_budget.words_per_beat_range)`)
   AND reconstruct it in `_otr_reroll` L366 (coerced). Stamp `meta["words_per_beat_range"]`
   ONLY when v2 (else off-ledger bytes change). All three sites derive the cap from the
   one helper.

## S2 -- coda bridge floor (weak-local; grok PASSES the gate)
`compose_news_coda(..., *, story_quality_v2_enabled: bool=False, arc_shape: str="")`
(keyword-only defaults; existing callers/tests unaffected). Writer (L4770) passes both
from `meta`. v2-ON only: LOCAL copy of `_NEWS_CODA_SYSTEM` + 1-2 in-context
premise->bridge examples (never mutate the constant); keep 2 attempts (DROP the
proposed 3rd -- the better fallback closes the gap). Fallback = an `arc_shape`-keyed
CURATED template pool selected by `sha256(cast_seed)`, each validated by
`validate_news_coda_bridge`; if arc_shape absent OR zero valid templates -> fall back
to the legacy `NEWS_CODA_POOL` (guards ZeroDivision; KEEP it + its tests).

## S3 -- body-gate accept: one total ordering, scored on the TEXT
For original AND reroll, compute on the SHIPPED TEXT (the `use_exchange` path sets
`beat_compose_flags=()` and bypasses hygiene -- OTR_LedgerScriptWriter L4431-4433 --
so `compose_flags` would false-pass): run `verify_and_repair_line` with the live
`_episode_entity_policy` for leaks; `trunc=is_truncated`;
`run_on=flag_one_breath(text, max_words=derive_one_breath_cap(range))[0]`;
`roster_caps` = an ALL-CAPS token-run matching an episode CAST FULL NAME anywhere
(never any caps token -- NASA/UCLA safe). **A mid-CLAUSE roster name is usually the
grammatical SUBJECT/OBJECT ("when CLARISSE GORDON claim..."), so NEVER strip it in
place (that yields "...when claim..."); a mid-clause hit sets `needs_recompose`
(reroll), and only a leading/trailing VOCATIVE is scrub-safe** (flash-high A/B catch,
2026-06-28 -- a flaw the gemini-3.5-pro panel missed across r1-r4). ONE total order:
`score = 10*grounding_failed + 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps`
(lower wins; ORIGINAL on tie) -- grounding failure dominates, so a grounding-failed
reroll never beats a clean original.

## S4 -- cliche replacement (no fragment)
Add `find_cliche_phrase(text) -> str` (return the matched span; `flag_cliche` gives
only `(bool, reason)`, L666). Run the exact-span replacement BEFORE EVERY quality-gate
return path (kept-reroll AND kept-original fall-through, L2515). Curated
safe-replacement map (build-commit deliverable + a test) with CASE-MATCH; respect the
single `_quality_repair_attempted` guard (no 2nd generic reroll); else accept
second-best + stamp `cliche_shipped_after_reroll`.

## S5 -- voices: measurement only (reuse the existing scan metric)
`story_quality_scan.py` already emits `register_overlap_ratio` (L456). Define the two
principals = **top-2 by dialogue-line count** (the `character_a/b_wants` fields are
verb phrases with no reliable name prefix -- panel-grounded; DISCARD the name-parse).
No runtime/ledger change. The `speaks:` prompt directive already exists -- unchanged.

## DEFERRED: S1 (seed fidelity). CUT: S6 (phantom).
S1's binding exists + it is weak-local/intermittent (grok stays on-premise) + the
reroll seam is busy with G1-S4; revisit as a window-level (NOT per-line) detector over
`grounded_nouns`, skipped when `len(grounded_nouns)==0`. S6 cut (acronym false-
positives; detect-only).

## Build sequence (6 green commits, each suite+BugBible+B7, pushed to v2.0-alpha)
1. **Shared leaf helpers** (`_otr_line_hygiene.py`: `derive_one_breath_cap`,
   `_hard_clauses`, `find_cliche_phrase`; `is_truncated` imports where needed) +
   `tests/fixtures/` golden ledgers + `tests/test_story_quality_golden.py`. Run the
   v2-OFF byte-identical test HERE.
2. **G1** (hint v2-const + v2-gated score + cap thread first-pass/reroll/scan + v2 meta
   stamp). 3. **S2** (params + local prompt + arc-shape fallback + pool-validation test).
4. **S3** (text-scored total-order accept + standalone roster-caps). 5. **S4** (span
   replace both paths + replacement-map test). 6. **S5** (scan two-principals + update
   `test_story_quality_scan_r2` expected values).

## Build-commit content deliverables (deferred lists, each with a validation test)
(a) the `arc_shape`-keyed coda template pool (per betrayal/heist/investigation/
slow_dread/setup_complication), each asserted to pass `validate_news_coda_bridge`;
(b) the S4 cliche->safe-replacement map (exact phrases); (c) the golden fixtures = the
specific failing lines extracted to `tests/fixtures/` (plancks b03/b10, ledger_ink
b04/b13, dance b04/b11).

## Acceptance (the judge's resolution of the panel's one contradiction)
The golden test asserts, per fixture line: (1) `not is_truncated` (no fragment/mid-cut),
(2) `flag_one_breath(text, max_words=derive_one_breath_cap(range))[0] is False` at the
BUDGET cap (clean at the allowed length -- NOT a hard `_hard_clauses<=3`, which would
contradict G1's intent to allow fuller lines), (3) `budget_lo <= word_count <=
budget_hi`. `_hard_clauses>3` stays ONLY a tie-break term in `line_quality_defect_score`,
never a hard gate. Plus: `length_ratio` rises from ~0.5 toward ~0.7+ WITHOUT padding;
the gate counters do NOT regress (anchor_stuffing stays low); per-line word-count median
rises from ~15 toward the budget.

## VERIFY-AT-BUILD checklist
1. first-pass cap == reroll cap == scan cap (one `derive_one_breath_cap`). 2.
`OTR_STORY_QUALITY_V2=0` byte-identical per sub-flag, NO new meta key. 3. coda matrix
`OTR_ENABLE_STYLE_GRAMMAR=1`+`STORY_QUALITY_V2=0` = legacy behavior. 4. golden asserts
pass (per the acceptance section). 5. pre-existing reroll-reconstruction gaps
(`EntityPolicy`/`speaker_gender`/`beat_role`/`conflict_object`) stay OUT of scope --
verify no new S3 logic depends on them (S3 uses the live first-pass policy via
`verify_and_repair_line`, not the reroll reconstruction).

## Operator decision gate (before the coder window starts)
- Confirm v2 is the gate flag for all of the above (vs a new `story_quality_v3`
  sub-flag). - Confirm "length as a side effect" bar: accept lines filling toward the
  per-beat budget (cap up to ~60 words) with no padding. - The ~1363-word structural
  ceiling stays (raising `BEAT_WORD_HARD_MAX` is a SEPARATE, larger change, not this
  pass). - S1 stays deferred; S6 cut.
