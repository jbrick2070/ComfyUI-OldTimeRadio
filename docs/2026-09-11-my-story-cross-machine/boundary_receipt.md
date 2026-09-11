# One-act music boundary correction

The planning prompts now receive the actual number of music boundaries,
calculated by the same helper used by frame and assembly. One act means zero;
disabled breaks means zero; otherwise N acts means N-1 boundaries. Surplus
model cue proposals remain in the ledger with their disposition. No keyword
gate, schema change, widget change or new model call.

Live defect evidence: 4060 Steps 118/119/121, PBUG-20260910-04. Portable rule
BUG-11.62 is promoted in the separate survival-guide repo with coverage index
and executable regression anchors. Survival-guide commit
`77bd6012b3484cdf0136434f5ed41477c23e3733` is pushed and equals origin/main.

Validation:

- 72 focused runner tests passed, including 1/2/3/6 acts with breaks on/off,
  exact prompt counts, ledger anchors, frozen_clean and surplus preservation.
- Full suite: 14,372 collected; 14,134 passed, 54 same baseline failures,
  183 skipped, 1 xfailed. No new failure IDs or worsened assertions. An existing
  AMD dictionary assertion only changed display order. Comparison JSONs are nearby.
- Clean-OTR Bible: candidate 30 passed/10 pre-existing failures/11 skipped/3 xfailed;
  baseline 0327850a had 29/11/11/3. Shared failure assertions are identical.
  Comparison ignores temporary checkout paths/addresses, truncated fixture-path
  display and test-source line offsets; actual assertion contents remain compared.
- Canonical validator, JSON roundtrip and structural audits pass: 23 nodes,
  63 links, 37 writer widget slots. No workflow edit.

Exact canonical SHA256:
`d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c`.

Review: pre-edit independent wiring audit, then Gemini 3.8 Flash (High) CLI finished
diff. Its Bible coverage-prefix finding was fixed and verified. Review artifacts
are in kibitz-runs/2026-09-10-my-story-boundary-qa/r4/.

This closes the producer mismatch in code. It does not claim a new full media
run, source-fidelity qualification, Mac memory recovery, or P1 binder closure.
Other blockers remain in GO_FORWARD. The user subsequently proposed adaptive
first-pass briefing; that design is under review, with real capacity as a prerequisite.
