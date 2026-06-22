# R3 JUDGMENT (wiring) -- accepted / rejected / verify-at-build

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.1053.

## ACCEPTED (folded into pass03_plan)
- **DEFECT 2 auto-repair CUT** -- unanimous + anchor + W1 (no cross-run channel; writer upstream + ignores
  the verdict; new_ledger wipes meta; regeneration_hint read by nobody; JSON frozen). Removed the
  coherence_hints mechanism, determinism-trap paragraph, max-1-rerun, and the coherence_hints assert
  (GPT cut#1/#2/mf#1/#7, Gemini cut#1/mf#2, DeepSeek mf#1/cut#1/#2). DEFECT 2 -> generation lever +
  detection only.
- **Tier-2 detector SITE corrected** (GPT mf#3, NEW): `compose_line_draft` (1689-1928) normalizes BEFORE
  the `compose_line` 2015-2060 reroll block, so the raw draft/quote boundaries are gone there. Moved the
  detector into `compose_line_draft` right after the LLM draft, before normalization.
- **Floor cannot route to reroll** (Gemini mf#2, NEW): it is downstream of the composer loop. Corrected:
  odd-quote line at the floor stays unscrubbed -> CI-fail / ships LOUD; only Tier 2 (compose time) rerolls.
- **Per-line audit via `compose_flags`, not per-line meta** (GPT mf#2, Gemini mf#1, DeepSeek mf#2, anchor
  mf#2, W5). + episode aggregate `meta["role_coercions"]`.
- **Tier 1 = strengthen the EXISTING rider at `_build_user_prompt`:1307-1315 ONLY** (GPT mf#4, Gemini sf#1,
  DeepSeek sf#1, anchor mf#4); beat prompt reserved for DEFECT 2 stance lever (no spoken-hygiene there).
- **FailedDimension "stance" + critic system-prompt prose (310-329) + StanceIssue + tests in one chunk**
  (GPT mf#6, Gemini mf#3, DeepSeek sf#3, anchor sf#1).
- **Coercion sweep placement** after all role mutations (incl cast_lock) before freeze/routing; cast_ids
  excludes announcer/music (GPT mf#5, DeepSeek mf#3, anchor mf#3). **Shared double-quote segmentation
  helper** between Tier 2 + Tier 3 (DeepSeek optional, GPT optional). **Reason codes** for fixtures.
  **CI-assert gated on OTR_TEST_MODE** (GPT sf#2). **Counters as harness returns + compose_flags + meta**
  (GPT sf#3). **Delimited-scrub interaction + announcer-line fixtures** (GPT sf#4, anchor sf#3). All folded.

## REJECTED / CORRECTED
- **All three "no" verdicts -- maturity, not new defects.** They fire because pass02 still carried the
  unbuildable auto-repair stretch + the per-line-meta assumption; R3 removes both. After folding, the
  plan is build-ready. No code-claim rejected as a misread this round -- the panel's wiring claims all
  matched W1-W5 (grounded).
- GPT sf#1 (StanceIssue.target provisioning) kept as a build option (pass context OR free-form string),
  not a blocker.

## VERIFY-AT-BUILD (small, non-blocking)
- Pre-freeze sweep insertion phase vs cast_lock ordering; `compose_flags` has no strict downstream format
  validation; the critic report model accepts a new `StanceIssue` (lenient pydantic); `OTR_TEST_MODE` is
  the right CI-assert gate.

## CONVERGENCE CALL
R3 CONVERGED. The one structural change (DEFECT 2 auto-repair -> generation+detection) is locked by W1;
the remaining items are interface/site corrections, all folded. pass03 is BUILD-READY. R4 = confirm no
new must-fix + lock final.
