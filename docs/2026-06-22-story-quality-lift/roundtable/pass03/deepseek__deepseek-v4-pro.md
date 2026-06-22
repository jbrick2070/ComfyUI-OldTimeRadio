<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The auto-repair stretch for DEFECT 2 is unbuildable (cross-run channel absent; grounding R3 W1) and the coercion audit trail design mistakenly assumes a per-line `meta` dict that does not exist (grounding W5). Those must be corrected before implementation can proceed.

MUST-FIX BEFORE BUILD:
1. [Section 4 – DEFECT 2 repair stretch] The “Repair (STRETCH, gated on R3)” proposal relies on carrying a `meta["coherence_hints"]` through a `needs_full_rerun` reset, but grounding R3 W1 proves no cross-run channel exists; the writer overwrites with a blank ledger and no node reads the verdict. The auto-repair is therefore impossible. **Fix:** Remove the entire repair stretch. Keep detection + loud + telemetry only. Delete the “DETERMINISM TRAP” paragraph and the “BOUND: max 1 stance full-rerun” language.
2. [Section 3 – coercion audit] The plan says `meta["role_coercion"]={prev,new,source,reason}` on a “row meta dict”. Grounding W5 confirms there is no per-line `meta` dict; per-line breadcrumbs must use the existing `compose_flags` list (or episode‑level `meta` keyed by line_id). The current design would fail or require a schema change. **Fix:** Store per-line coercion audit in `compose_flags`, e.g., `"role_coerce:prev=announcer,new=character,reason=cast_char_id"`. Remove the reference to a per-line `meta` dict.
3. [Section 3 – coercion sweep ordering] The plan’s “FINAL pre-freeze consistency sweep” must run after all line-level mutations, including `cast_lock`’s legitimate announcer re‑stamp (grounding W3). The plan does not specify where in the freeze cascade to insert it. **Fix:** Explicitly state that the sweep is added inside `OTR_LedgerFreezeCascade` after all line-processing nodes have executed but before the final freeze/hash step, ensuring `cast_lock` has already run.

SHOULD-FIX:
1. [Section 7 – Tier‑1 prompt builder] Grounding R3 W2 already identifies the per‑line prompt builder as `_otr_line_composer.py::_build_user_prompt` (lines 1050‑1338). The plan should specify that location instead of leaving it as an R3 open item.
2. [Section 6 – clean fixture] The assertion `"coherence_hints" not in ledger.meta` should be removed, because the auto‑repair is cut and that key will never exist.
3. [Section 4 – detection integration] Adding `"stance"` to `FailedDimension` is safe (grounding W4), but the plan must also update the critic system‑prompt in `_otr_story_critic.py` (around lines 310‑329) so the model can emit stance issues. This should be noted explicitly.

OPTIONAL / NICE-TO-HAVE:
- Share the double‑quote segmentation logic between the reroll detection and the deterministic floor scrub to avoid parsing drift.
- Consider a small configurable MAX_WORD_COUNT for the Tier‑3 clause detector instead of a hard‑coded cap.

CUT THESE:
1. [Section 4] The entire “Repair (STRETCH, gated on R3)” subsection and its “coherence_hints” mechanism, because it is unbuildable.
2. [Section 4] The “DETERMINISM TRAP” paragraph about seed‑keyed rerun and meta injection; it is irrelevant without auto‑repair.

[ASSUMPTION] Assumes `compose_flags` can store arbitrary “kind:detail” strings without breaking downstream consumers; verify no strict format validation exists.