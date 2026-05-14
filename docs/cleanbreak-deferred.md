# Voice-Path-Cleanbreak — Deferred Items

**S28 close (2026-05-13): no active deferrals.**

The S24 → S25 → S26 → S27 → S28 cleanbreak chain is complete; every
surface flagged across those sprints has been extincted. The cleaner
break ended the chain (see `docs/2026-05-13-S28-final-qa-review.md`).

## Historical resolutions (closed, retained for audit trail)

The following items were tracked in this file during S24–S27 and
reached a terminal decision before S28 opened. They are NOT active
deferrals; their resolution lives in the git log and in dedicated
ADRs / sprint-close docs.

| ID | Resolution date | Outcome | Where to look |
|----|-----------------|---------|---------------|
| C10 — LFC audit regex extension | 2026-05-13 (S25) | CLOSED — plan's premise was wrong; LFC is current live infrastructure, not a deleted lineage. No regex extension. | Git log at S25 close; commit messages reference IMP-37. |
| C8 — CastContract quarantine | 2026-05-13 (S25 CD-1) | CLOSED — Option 3 selected. Cast contract is production-wired (cast_repair → ledger_reviewer chain); quarantine plan dropped. | Git log at S25 CD-1; the broader-graph audit is captured in the commit body. |
| S14.2 — Validator auto-invoke | 2026-05-13 (Option B locked) | DEFERRED to S25+ implementation per the ADR at `docs/2026-05-13-S14_2-active-validation-ADR.md`. Mitigation already in place (S16.6 + S15.5.1). | ADR. |

If a future sprint reopens any of these (e.g. LFC retires, or
S14.2 ships), record the new state in the appropriate sprint
artifact — do NOT re-add them to this file. This file is reserved
for in-flight cleanbreak deferrals only, and S28 is the last
cleanbreak sprint.
