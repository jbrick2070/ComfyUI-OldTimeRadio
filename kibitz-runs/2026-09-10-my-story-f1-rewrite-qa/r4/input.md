# Sonnet QA: source checking that actually rewrites, with a fixed budget

Read-only finished-diff QA of the F1 source-repair chunk, relative to
7e4a0dd6738077d2c580304cb28dae3b6af5d1a8. Read the complete current production
and test files in snapshot.json; two files are new and absent from git diff.
Write only the requested review file. Do not edit code, run tests, load models,
coordinate machines or prepare hardware prompts.

The operator explicitly amended the earlier report-only reviewer architecture:
"I JUST NEED A CHECKER+REWRITER", "WE [CAN'T] HAVE AN INFINITE LOOP", and no
fourth/fifth continuing rewrite round. The implemented SOURCE correction gets
one combined check/rewrite call and at most one retry TOTAL, including helpers.
There is no report-only source model call, self-recheck or operation budget
reset. Existing author syntax/structural admission and existing spoken-hygiene
cleaner have their separately existing budgets. Inspect the total call behavior
for accidental multiplication, and flag a real root failure rather than ask for
another speculative layer. GO_FORWARD records the operator amendment.

Actual implementation:
- _otr_story_source.rewrite_story_source invokes structured_call once,
  max_attempts=2, ProviderCapacityMessages/max_new_tokens=None, and journals
  actual attempts before fit/acquisition/generation. Pass-id reentry is refused
  without more calls. Unknown capacity never becomes a measured refusal or PASS.
- P0/P1/P2/P3 see the exact four raw fields, not normalized/summary-only input.
  _otr_my_story._call accepts an authored artifact first, then invokes the
  source correction ONCE outside author post-validation. It uses the existing
  schema and structural validator, returns that exact validated corrected model,
  and retains the original if correction exhausts or cannot fit. P0 uses its
  technical owner; P1/P2/P3 use creative. Full failed artifacts survive typed repair.
- ledger_clean keeps its original nonspoken-text repair scope. It now sees raw
  source in its context. At its final seam one combined source correction returns
  actual text edits quoting exact source and original intervals, with no overlap,
  protected-row edits or empty spoken row. Python only interleaves authored text
  and unchanged slices, then uses the shared metrics setter.
- The writer tail retains the source journal across cleanup/reconciliation
  failure and rollback. Actual retained ordered/speaker text and final delivery
  receive hashes, not a new model check or semantic qualification claim.

233 focused tests passed, including two-call stubborn failures, full ending
context, wrong schema then correction, exact raw Unicode, missing-list-item
repair, actual saved accepted corrections, provider/cancellation persistence
after later cleanup fails, rollback history, and no model recheck.

No live hardware/media proof is claimed. Adaptive source organization and F2
visual integration remain subsequent coding chunks. Do not treat their absence
as a claim they are complete; identify any introduced regression that must be
fixed before this component commit. No node interface/widget/wiring change;
canonical workflow is unchanged and remains the only live-test graph.

Return concrete must-fixes with file/line/evidence, or a clear no-must-fix verdict.

