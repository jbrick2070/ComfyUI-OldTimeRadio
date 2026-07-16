# Rip the LLM audit gates; prove the contracts instead (r3: WIRING)

**Date:** 2026-07-13
**Base:** `c74d75b0` pushed + an UNCOMMITTED cross-lane rip in the worktree.
**Round:** r3 -- WIRING. Two live rolls already died on things a wiring review
should have caught. Find the rest before I spend another one.

## Operator law

> AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE.
> Only DETERMINISTIC validators may end an episode. An LLM verdict may trigger a
> bounded rewrite; it may never raise.

## SHIPPED (pushed, green: suite 7880, Bible 17/16/3)

1. `original_codex56sol`: ripped P4 (fair play), P7/P8 (blind listener + retake),
   P9 (final contract audit). 9 LLM passes -> 5. Fair play is now a DETERMINISTIC
   CONTRACT: the device anchor must be spoken on a clue-carrying line BEFORE the
   reveal line, as well as on the reveal line. Enforced at three rungs, each one
   able to repair it (`_validate_score_clue_ownership` -> P5 ladder;
   `_validate_score` + `_validate_script_grounding` -> the bounded intent/line
   patches). Removed from all three wiring surfaces (pack `prompt_stages`,
   `pipelines.json` `declared_seams`, pass `seam_refs`).
2. `G9` in `_otr_ledger_freeze.run_gap_audit`: word-boundary DEFAULT_PROFANITY_TERMS
   scan over spoken ledger text, on the one path every lane crosses. Phase 10
   raises. This is the SFW enforcement that never existed (the ledger scrub runs
   only on `run_story_spine=True` lanes and nothing reads its verdict).
3. `schema_shape_instruction` no longer emits an empty "exact keys=" line for
   every scalar (40 -> 26 lines on PossibilitySlate).

## LIVE FAILURES SINCE, AND THE ROOT FIXES (this is what r2 missed)

* **prompt `a89a46a4`** -- P5 died. The model wrote a valid announcer cast row
  and filed it under `char_id: "a"`. The rejection sent a **5,772-token repair
  prompt into a 4,592-token usable window** (context_cap 8192, max_new_tokens
  3600); PROMPT_GUARD truncated it, the tail carrying "return the complete
  artifact" fell off, and the model answered with a single cast row. Ladder
  exhausted.
  - FIX A: `_project_announcer_char_id` -- an id is a coordinate, not authored
    content. Canonicalize it at the schema-validated attempt boundary (same
    class as the existing topology/duplicate-clue projections).
  - FIX B: `_repair_inputs` -- the P5/P6 repair prompt no longer re-sends the
    full truth map + grounding contract. It sends the anchors and the clue
    inventory only (the failed artifact already carries the graph).
* **prompt `efafc6fa`** -- P1 died: "lost_objects and acoustic_device must be
  copied verbatim". The model wrote the right story about the right device and
  RE-WORDED the field that only echoes the immutable draw.
  - FIX: `_restore_slate_immutables` -- restoring an input is not authoring.
    Everything downstream is anchored to the draw anyway.
* **prompt `37a3cedf`** (in flight) -- P1/P2 now pass first attempt. P3 failed
  attempt 1 on "caller_threads must contain exactly one row per selected lost
  object" and went to typed repair.

## UNCOMMITTED IN THE WORKTREE -- the cross-lane rip (REVIEW THIS)

* `_otr_scifi_gemini.py`: `_spoken_error` gained an `allowed_all_caps` exemption
  (locked cast names + source acronyms) -- the outline seam ORDERS ALL-CAPS cast
  names and the validator killed any all-caps token, OUTSIDE every ladder. The
  spoken check moved INTO the P4/P6 post_validator (`validate_scene_draft`) so
  the ladder repairs it. Ripped: the `sfw_pass` clause, the P5-recheck call, and
  `SciFiGeminiRewriteExhaustedError`. The critique now buys ONE bounded rewrite
  and never raises. `SceneCritiqueV4.sfw_pass` and the seam's Safety clause are
  gone.
* `_otr_scifi_sonnet.py`: ripped the `severity == "critical" / invented_fact_flags
  / sfw_pass` veto and `SonnetAuditExhaustedError`. Added `ungrounded_lines`: a
  factual line must cite a real dossier fact id and may only speak numbers the
  source states -- a PROOF, and the only thing allowed to end the episode.
  `AuditVerdictV4` lost `severity`, `sfw_pass`, `invented_fact_flags`; the seam
  was updated to match.
* `_otr_original_radio.py`: `corroborate_hard_finding` was a RAW SUBSTRING scan
  over `finding.detail + script` -- so "gun" fired on "begun", and a judge that
  wrote "the scene mentions a gun" corroborated ITSELF. Now `lexicon_hits()`:
  word-boundary, script-only.
* `_otr_scifi_fable2.py`: **HALF-DONE.** The P8 LLM ledger audit call, `_triage`,
  and `_pass_audit` are DELETED and replaced by `_assert_no_weapons_or_smoking`
  (word-boundary scan of the spoken drama). The audit killed a COMPLETE,
  twice-persisted ledger on a model opinion, and its corroboration lexicons were
  the PERIOD lane's, so ordinary sci-fi words ("machine", "computer", "report")
  "proved" a hallucinated flag.

## WHAT I NEED FROM THIS ROUND (wiring)

1. **The fable2 rip is half-wired. Complete the wiring inventory.** `AuditFinding`
   / `AuditFindings` are now unreferenced; the `fable2_audit_system` seam still
   exists in the pack, in `pipelines.json` `declared_seams`, and in a pass's
   `seam_refs`; `_TEMP["ledger_audit"]` and `_MAX_NEW_TOKENS["ledger_audit"]` are
   dead; and FOUR test files still reference the removed surface
   (`test_fable2_runner_ladders.py` calls `F2._triage` and monkeypatches
   `F2._pass_audit`; `test_fable2_registry.py` and `test_fable2_prompt_snapshots.py`
   pin the seam; `test_fable2_artifacts.py` builds `AuditFinding`). Name EVERY
   surface that must change in the same commit, or the lane dies on import at
   registry load (three-way parity is enforced there). Did I miss one?
2. **Is `pass_4_technical_ledger_audit` (referenced in
   `tests/test_story_routing_stage2.py`) a pipeline pass that must go too?**
3. **The gemini spoken check moved into the P4/P6 post_validator.** Its ladder
   now repairs an all-caps defect -- but `invoke_gemini_structured`'s repair
   prompt has the same budget problem P5 just had. Does the gemini P4 repair
   prompt fit its window, or did I just move the kill from a raise to a truncated
   ladder exhaustion?
4. **Sonnet: `ungrounded_lines` raises `SonnetSpokenTextError` after `_assemble`?**
   Check the order. If the ledger is already persisted when it raises, I have
   rebuilt the exact defect I just removed from fable2.
5. **G9 vs the lanes:** does any lane legitimately speak a DEFAULT_PROFANITY_TERMS
   word (period drama: "hell", "damn")? G9 is now a hard ship-stop on ALL of them.
   If a shipped episode in `otr/obs/` would fail G9, say so -- that is a
   regression I introduced.
6. **Anything else in the worktree that is now dead, unwired, or half-wired.**

## Constraints

- Python may never author story content. Restoring an INPUT or canonicalizing an
  ID is not authoring; rewriting prose is.
- Deterministic validators are the authority and stay fatal.
- No shims. Root-cause only.
- `workflows/otr_canonical.json` must not change.
