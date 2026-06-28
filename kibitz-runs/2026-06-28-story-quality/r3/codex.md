VERDICT: yes-with-fixes. The plan is directionally implementable, but several values are not threaded through the actual call seams, so first-pass, reroll, and scan can disagree.

MUST-FIX BEFORE BUILD:
1. [G1.3] Dynamic one-breath cap is not threaded into first-pass composition. The plan adds `LineRequest.words_per_beat_range` and reconstructs it in reroll, but current first-pass `LineRequest` construction has no such propagation from `episode_budget` (`nodes\OTR_LedgerScriptWriter.py:3024`, `nodes\OTR_LedgerScriptWriter.py:4235-4305`). Fix: pass `words_per_beat_range=tuple(episode_budget.words_per_beat_range)` into the first-pass `LineRequest`, and use one shared helper for composer/reroll/scan cap derivation.

2. [G1.3] `meta["words_per_beat_range"]` must be v2-gated or it violates the stated flag-OFF byte-identical invariant. Current v2 meta is only stamped when enabled (`nodes\OTR_LedgerScriptWriter.py:2635-2636`); unconditionally adding the budget key after `compute_episode_budget` (`nodes\OTR_LedgerScriptWriter.py:3024-3038`) changes off-ledgers. Fix: stamp this meta key only when `meta["story_quality_v2_enabled"]` is true; absent must mean legacy static 28.

3. [G1.2] The new `line_quality_defect_score` extras can change v2-OFF behavior unless explicitly gated. `_quality_flags_for_line` has always-on cliche/stage/on-the-nose flags (`nodes\_otr_line_composer.py:2303-2311`), and the keep-better decision currently compares `len(_after_flags)` to `len(_q_flags)` (`nodes\_otr_line_composer.py:2502-2504`). Fix: when `req.story_quality_v2_enabled` is false, preserve the existing length-only decision; only add `is_truncated` and `_hard_clauses` scoring on the v2 path.

4. [S2] The arc-shape keyed coda fallback has no input seam. Current `compose_news_coda` accepts `creative_fn`, `news_close_brief`, `premise`, `intro_text`, `cast_seed`, and `creative_repo_id` only (`nodes\_otr_line_composer.py:3278-3279`), and the writer call passes only those (`nodes\OTR_LedgerScriptWriter.py:4770-4777`). Yet `arc_shape` is available in writer/meta (`nodes\OTR_LedgerScriptWriter.py:3563-3567`). Fix: add keyword-only `arc_shape: str = ""` to `compose_news_coda` and pass `_arc_shape` or `meta["arc_shape"]` from the writer.

5. [S3] The body-gate accept predicate names “roster-caps hit” but does not bind it to the real flag. The verifier emits `leak_floor:roster_vocative` for ALL-CAPS full-name scrubs (`nodes\_otr_line_hygiene.py:1362-1376`), while current body-gate acceptance only checks grounding (`nodes\OTR_LedgerScriptWriter.py:4528-4534`). Fix: define the S3 predicate against exact compose flags: hard leaks include `leak_floor:malformed_quote`, `leak_floor:banned_source_entity`; roster-caps is `leak_floor:roster_vocative`.

6. [S4] “Replacement AFTER the existing one reroll” will miss the accepted-reroll branch unless the plan names that return seam. Current quality reroll returns immediately when the reroll has fewer defects (`nodes\_otr_line_composer.py:2502-2515`). Fix: run the exact-span cliche replacement helper before every quality-gate return path: kept reroll and kept original fall-through.

SHOULD-FIX:
1. [G1.3/S3] Define a shared cap coercion helper for JSON-loaded ranges. Reroll/scan will read JSON arrays from meta, not tuples (`nodes\_otr_reroll.py:299`, `scripts\story_quality_scan.py:378-387`). Fix: coerce list/tuple/stringy ints to `(lo, hi)`; malformed/absent => `(0, 0)`.

2. [S3] The scoring term `run_on` is undefined as an interface. The available detector is `flag_one_breath(text, *, max_words=28, max_clause_markers=3)` (`nodes\_otr_line_hygiene.py:887-889`). Fix: define `run_on = flag_one_breath(text, max_words=shared_cap)[0]`, using the same G1 cap.

OPTIONAL / NICE-TO-HAVE:
1. [G1.2] Export or locally keep `_hard_clauses` testable; golden tests will otherwise need to import a private helper by convention.

CUT THESE (over-engineering):
1. [S2] Third coda LLM attempt is optional if curated fallback is implemented and validated. Current function already has two attempts plus deterministic fallback (`nodes\_otr_line_composer.py:3314-3328`); adding arc-shape fallback closes the reliability gap without extra model latency.