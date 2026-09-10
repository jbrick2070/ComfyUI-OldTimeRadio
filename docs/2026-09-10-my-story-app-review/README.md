# My Story scope review and priority handoff

R1-R4 scope review is complete. Codex grounded and judged the reviews; two
read-only helpers checked contracts and the combined handoff. The final operator
clarifications assign cleanup to Opus, My Story to Codex, and remove replay-system
work, old-bundle compatibility and source-snapshot restoration from My Story.
Those final cuts are driver-applied operator instructions, not claims that every
reviewer saw the final wording.

- [Priority #1 and five-sprint sequence](../GO_FORWARD_PLAN.md)
- [Hardened My Story scope](../2026-09-10-my-story-app-scope.md)
- [Opus starter prompt](OPUS_START.md)
- [Separate cleanup handoff](../2026-09-10-cleanup-opus/HANDOFF.md)
- [Machine-readable campaign receipt](campaign_receipt.json)
- [Exact captured review evidence](review-evidence.zip)

| Round | Focus | Driver judgment |
|---|---|---|
| R1 | Audience, authoring flow and scope | [R1](r1_judgment.md) |
| R2 | Implementation contracts | [R2](r2_judgment.md) |
| R3 | Wiring, state and sequence | [R3](r3_judgment.md) |
| R4 | Convergence and combined handoff | [R4](r4_judgment.md) |

Actual external calls: **9 attempts, 8 successful reviews**. Gemini 3.8 Flash
(High) and Cursor Grok 4.6 High each completed all four rounds. The initial
Claude Code attempt failed authentication and was replaced by Cursor. The
reviewers supplied no usage/cost receipt; no spend total is claimed. These
counts exclude the separate cleanup campaign and internal helper work.

The ZIP preserves exact input, anchor, review, judgment and final bytes for each
round, including the previous-final to next-input chain. The receipt supplies
member hashes and original paths. Readable judgment copies normalize only line
endings and trailing whitespace. Captured plan links retain their original
`docs/` document base; use the delivered scope above for navigation.

This was documentation/scoping work only. No runtime, workflow, registry or
test changes, render, download or browser qualification were performed. Exact
independent bank design remains D1 work; scope convergence is not a bank
preflight PASS. The operational README/guides still await the scoped D0 update.
