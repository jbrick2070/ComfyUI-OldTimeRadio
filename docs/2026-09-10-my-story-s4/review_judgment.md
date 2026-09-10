# My Story Sprint 4 finished-diff judgment

2026-09-10. Driver/judge: Codex. One external CLI reader: Cursor,
resolved `cursor-grok-4.6-high`, ask mode, exit 0. Read-only internal helpers
audited input/runner/drafts and writer/mux/workflow. This is the scoped code
review required by the existing handoff, not another design campaign.

Anchor written before dispatch: `driver_anchor.md`. Launcher receipt and
verbatim review are retained beside this judgment. No review service spend
was reported by the CLI; no cost is invented.

Disposition of the external review:

1. ACCEPT its sole must-fix: D1 now explicitly treats act/frame line counts as
   prompt guidance. Nonempty content, cast identity and ledger topology remain
   enforced. No new length gate was added.
2. REJECT the proposed retryable capacity error. The shared structured-call
   contract explicitly propagates terminal exceptions; returning a string would
   spend repairs trying to change the listener's cast. Refined the existing
   terminal check to compare the authoritative count before model arithmetic.
   Exclusive cast still takes precedence. Both planned-count cases are tested.
3. ACCEPT the joint draft check. The real queue admission and real writer now
   share one prompt/node identity and rolled-style input; the test proves one
   unchanged draft and matching digest after style resolution.
4. REJECT treating present-but-empty `script_json` as absent. D1 distinguishes
   these cases. Installed ComfyUI execution iterates supplied inputs rather
   than injecting optional defaults; the mux already defaults omitted input to
   `None`. Both legacy omission and malformed/present input are tested.
5. ACCEPT checking incremental ledger saves. A failed preamble or act save now
   aborts immediately; injected persistence failures cover both checkpoints.
6. ACCEPT currentizing D1 status, bank docs and the handoff. Optional native App
   and live publication are still separate Sprint 5 evidence, not a preflight PASS.

Additional grounding: the ledger's save makes a shallow top-level copy, so
nested metadata aliases retain identity. Announcer c01 and the announcer line
sentinel match the existing lane contract. No new freeze policy, draft writer,
or compatibility/replay subsystem was introduced.

Full-suite findings were stale test expectations for appended fields and bank
roster, and an AST-isolated validator fixture missing its new dependency. Those
fixtures were updated; failure allowlists were not changed. A separate audit
found the four story fields missing from both headless creative whitelists;
both now permit them, with real canonical conversion coverage for every field.

Final focused evidence: 138 My Story cases, 100 fixture/runner checks (before
the three added capacity/save cases), and 35 workflow-applier cases passed.
The final receipt owns the full-suite comparison and canonical audit results.
