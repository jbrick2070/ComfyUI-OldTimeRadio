# Sonnet follow-up: the failed reread cannot authorize a rewrite

Review only the correction to the must-fix in
kibitz-runs/2026-09-10-my-story-c1-qa/r4/claude.md. That review read the actual
four source/test files and found the discarded reread validity flag. Do not
repeat the architecture review or reopen unrelated code. Read-only QA; write
only your requested review file. No tests, models, edits or hardware coordination.

Root confirmed the defect and corrected nodes/_otr_ledger_clean.py _repair_row:
every scoped proposal receives a model reread, even if the original judge was
malformed and the initial scope came from exact patterns. reread_ok=False
records reread_unresolved, preserves original edit scope/complaints, and uses
the existing next attempt without entering clean or best-progress acceptance.
If the budget exhausts, retain original/flag unless an EARLIER actually judged
candidate had shown progress. Real provider/OOM/cancellation/terminal-capacity
exceptions still propagate. Initial judge validity is recorded in the scope.

Read tests/test_ledger_clean_stage.py:
- test_malformed_reread_cannot_be_clean_or_best_progress
- test_pattern_repair_gets_a_real_reread_after_initial_judge_exhaustion
- test_cleaner_jobs_do_not_swallow_real_runtime_failures (now includes reread)

183 focused tests pass. The final full suite is rerunning; original run had
14,241 passes/51 existing failed IDs. A new missing slot-label comment in one
failure was corrected and its targeted test returned exactly to baseline.
The immutable four-file final tested hashes are in
docs/2026-09-11-my-story-cross-machine/c1_snapshot.json.

The first review's optional fidelity-lane pattern/model distinction is existing
explicit behavior, not a C1 change: the model continues to judge all lanes;
pattern-only language is ineligible on adaptation banks. No new work or whole-
source PASS is inferred. Current writer-tail calls were independently read at
nodes/_otr_writer_tail.py:1383-1404. Bible 11.64 now uses its valid legacy ID
PBUG-20260829-14; 341 entries, 146 unchanged metadata issues.

Confirm the actual correction closes the first review's must-fix and its
specific runtime-test coverage request, or name a concrete remaining blocker.
