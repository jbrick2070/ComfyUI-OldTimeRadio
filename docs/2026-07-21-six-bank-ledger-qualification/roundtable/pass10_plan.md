# Pass 10 -- live P5 context/output exhaustion R2/R3 audit

## Trigger

The canonical `scifi_news` run `569b20e5-0e28-4472-a04d-637ab019f19f`
exhausted P5 before ledger or media production. One whole-artifact response hit
its output cap, one drifted to an unknown line ID, and the whole-artifact repair
could not retain the requested output reservation inside the local 8K context.
Sol stopped the campaign and reset the owned server/GPU state.

## Review lane

- reviewer: Antigravity, review-only
- exact model: `gemini-3.5-flash-high` (Gemini 3.5 Flash High)
- rounds: scoped R2 hotspot review, then same-conversation R3 convergence
- clean worktree: detached throwaway worktree at
  `e2f3b10d6856721bd24745975eec06f2197bd223`
- driver/coder/judge: Sol only

## Required scope

Inspect the P5 transport and all six routes for completion, ledger ownership,
spoken-hygiene bypasses, stale seals/hashes, row-local failures, readiness
mutations, and downstream consumer gaps. Decide whether the root repair belongs
in shared bank code or only the reachable producer lane. Preserve full
validation and truthful finite liveness while removing avoidable context and
output pressure.

## Grounding plan

Sol verifies every claim against the real Windows files, discards reviewer
misreads, fixes every confirmed reachable sibling, then runs exact-tokenizer,
focused, full-suite, Bug Bible, and canonical workflow gates before resuming
live qualification. Any new live defect repeats the targeted cross-bank audit.
