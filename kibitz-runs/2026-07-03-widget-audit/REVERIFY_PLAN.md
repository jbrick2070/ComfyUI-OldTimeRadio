# Re-verification round -- PARKED until the credit-module rewrite is pushed
Trigger: operator says the credit-module-to-end change is committed+pushed on v2.0-alpha.
Status: r1-r3 complete and grounded; r4 NOT run; this replaces a plain r4 with a re-verify-against-new-code round.

## Sequence (Claude drives)
1. FREE RE-BASELINE: re-run the widget inventory script against the NEW workflows\otr_scifi_16gb_full.json (AST + consumption grep, no agent spend). Diff against docs\2026-07-03-widget-audit\widget_audit_raw.json -- any node whose widget vector or wiring changed gets flagged STALE in the findings table.
2. CLAUDE ANCHOR (delta review): re-read only the files the rewrite touched (git diff of the credit change) + the tail chain (nodes 84/86/93/85 order, links). Re-verdict every r3 finding as STILL-VALID / STALE / SUPERSEDED. Special attention: batch 3 stage-order spec (84->93->86->85) vs the NEW credits-at-end position -- decide the tail order (procgen -> credits -> captions -> mux vs captions before credits) as an explicit line item for the operator.
3. CODEX (kibitz r4): feed r3\final.md + the delta note as --doc, round r4 (convergence), --only codex. Prompt focus embedded in the doc header: "the code has changed since this review (credit module moved to end of chain); verify every file:line cite still holds, flag anything the rewrite invalidated, confirm no NEW must-fix".
4. AGY (manual packet v2): same re-verify directive, independent; writes to antigravity_reverify.md. Packet text below -- operator pastes it.
5. JUDGE + SYNTHESIZE: ground both, discard misreads, fold survivors into the UPDATED docs\2026-07-03-widget-audit\WIDGET_SURFACE_AUDIT.md (v2), commit+push everything (including the uncommitted r1-r3 finals) in one docs commit.

## Agy packet v2 (paste when triggered)
You are an independent reviewer. The repo has CHANGED since your last review (the credit module was moved to the end of the render chain). Read the REAL files; do not trust prior reviews including your own. REVIEW ONLY -- write your review to kibitz-runs\2026-07-03-widget-audit\antigravity_reverify.md and change nothing else.
Re-verify kibitz-runs\2026-07-03-widget-audit\r3\final.md claim by claim:
- Does every file:line cite still hold after the rewrite? List any that moved or vanished.
- Nodes 80-83 widget vectors: still exactly as stated in the saved workflow JSON?
- Tail chain: what is the NEW order of nodes 84/86/93/85 + the credits stage? Does the batch-3 rewire spec (84->93->86->85) still make sense, and where do credits sit relative to captions?
- widget_mapping.json + profile JSONs: still targeting node 93 for captions?
- Any NEW dead/confusing widgets introduced by the rewrite itself?
Format: VERDICT / STILL-VALID list / STALE list / NEW MUST-FIX / MISREADS.

## Standing constraints
- One coder window: nothing here runs while the operator is mid-rewrite.
- These kibitz-runs files stay UNCOMMITTED until step 5's single docs commit (except antigravity_manual.md, already pushed by agy @ f33cce58).
