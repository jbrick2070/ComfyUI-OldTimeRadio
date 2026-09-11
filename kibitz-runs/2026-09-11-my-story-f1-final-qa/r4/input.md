# Sonnet final QA after source-repair revisions

The operator explicitly requested Sonnet QA again after coding revisions. This
is a scoped finished-diff R4 follow-up, not a new architecture campaign. Read
the six current code/test files in snapshot.json relative to base commit
7e4a0dd6738077d2c580304cb28dae3b6af5d1a8. Read-only: write only your review file;
do not edit code, execute tests, start models, or coordinate hardware.

The implemented source correction returns actual replacement artifacts/text,
with at most TWO source calls per owning pass, including syntax/schema repair.
Reentry cannot reset the budget, and no model self-recheck or report-only pass
is added. Existing author structural admission and spoken-hygiene budgets are
separate. Unusable optional correction retains accepted usable input; genuine
provider/OOM/cancel propagates with a durable attempt record. All P0-P3 prompts
carry exact raw fields, and corrected models go through the existing schema
and structural validator before the exact validated result is applied.

The first Sonnet review is in
kibitz-runs/2026-09-10-my-story-f1-rewrite-qa/r4/claude.md. It found no blocker
and requested two improvements. Both are now implemented:
1. build_raw_documents is used by actual source receipt interval construction.
2. Each final spoken correction records retained and clean_outcome after
   cleanup reconciliation, comparing its actual editable row subset against
   the retained canonical text. Attempted application remains historical.

Also fixed two mechanical issues found by the full suite: projection creation
now constructs fresh dictionaries instead of assigning to a field named text
(the actual ledger still changes only through set_line_text_metrics); the LLM
slot comment uses the project's recognized creative/technical tag. The
qualified=False field now explicitly distinguishes application from proof.

Current focused execution: 236 passed in 8.67 seconds. Exact XML is
tmp/my_story_f1_final_focus.xml. This is an execution receipt from the driver,
not a request to execute or independently attest to the test count. The full
suite is being repeated on this final snapshot after fixing its one new
failure and its expanded pre-existing slot-tag failure. Baseline had 51
failures. Bible and canonical checks are also being reverified by the driver.

Verify the final integration, budgets, correct schemas/owners, application,
metrics, persistence after later cleanup exceptions, rollback retention, and
that source scope receipts describe actual input. Identify concrete remaining
must-fixes with file/line/evidence, or a clear no-must-fix verdict. Do not ask
for an extra source model checker or treat schema success as semantic proof.
Large-source preparation, visual integration and hardware proof are still
separate work; no claim that this component completes them. No node, widget,
or canonical interface changed.
