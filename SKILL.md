# OTR Sprint Workflow

**For Cowork and any other AI assistant working on this repo.** Read this BEFORE touching SPRINT.md or executing sprint commits.

## When to invoke this skill

Any time the user wants to:
- Apply round-robin review findings to the sprint plan
- Cut a branch and execute a sprint
- Run a commit's review-code-wire-pytest-regression-commit loop
- Close a sprint and hand off to the next
- Open a new sprint from a prior sprint's handoff

## Canonical files (single source of truth, no proliferation)

- `SPRINT.md` at repo root — the ONE rolling document for the active sprint. Edited in place. Never duplicated with `-v1/-v2/-v3` suffixes.
- `docs/closed-sprints/<YYYY-MM-DD>-<sprint-name>.md` — archived sprints, read-only.
- `SKILL.md` at repo root — this file.

## Workflow phases

### Phase 1 — Planning

User pastes an initial plan text or a problem statement. Build out the SPRINT.md template sections. **Do NOT create dated `-v1/-v2/-v3` files** — git history captures every revision.

### Phase 2 — Round-robin review

User pastes reviewer findings (from Gemini, ChatGPT, an adversarial Claude run, etc.) into the "Open findings (TRANSIENT)" section of SPRINT.md, then says "apply these findings."

1. Synthesize each finding as ACCEPT / REJECT / PARTIAL ACCEPT.
2. For each ACCEPT or PARTIAL: update the relevant commit specification in "Plan"; add a row to "Acceptance table" if new; add an entry to "Decisions log" with the finding ID, disposition, and rationale.
3. For each REJECT: add an entry to "Decisions log" with the reject rationale. Do NOT silently drop findings — name the rejection so future reviewers don't reopen it.
4. **DELETE the contents of "Open findings (TRANSIENT)" after applying.** Git history preserves the input. The file stays clean.
5. Commit: `git commit -m "<sprint>: round-<N> synthesis (<X accept / Y reject / Z partial>)"`.

### Phase 3 — Branch cut + execution

1. Verify "Status" preconditions met (clean checkout, branch ready, prior sprint closed).
2. Cut branch from the specified ref.
3. Set `Status: Phase = execution, Current commit = <first commit>` in SPRINT.md.
4. Execute each commit per the review-code-wire-pytest-regression-commit loop:
   - **Review** — read the commit's spec section from SPRINT.md
   - **Code** — implement per spec
   - **Wire** — connect to graph, call sites, or workflow JSON as specified
   - **Pytest** — run the commit's pytest table; ALL must pass
   - **Regression** — Bug Bible regression must hold
   - **Commit** — with the exact commit subject specified
5. After each commit lands, update `Status: Current commit = <next>` in SPRINT.md.
6. No operator gates between commits — the chain runs through to C-final.

### Phase 4 — Sprint close

1. Fill the "Sprint-close handoff" section in SPRINT.md.
2. Run wide pytest walk; confirm every row of "Acceptance table" green.
3. Final commit (typically `C-final`).
4. Push branch.
5. Archive: `git mv SPRINT.md docs/closed-sprints/<YYYY-MM-DD>-<sprint-name>.md`.
6. Initialize new SPRINT.md from the template at the bottom of this file. Copy the prior sprint's "Sprint-close handoff" into the new SPRINT.md's "Previous Sprint Handoff" section.

## Persistent project rules (enforce in every commit, every session)

- **No curse words.** Anywhere.
- **No "dummy".** Use "placeholder", "stub", or a descriptive name.
- **No-change-logs rule:** existing runtime log strings stay byte-stable. Existing `meta.*` attribute names stay byte-stable. New log lines added by a sprint follow the format conventions of neighboring lines; no surrounding existing line is modified.
- **Commit sizing rule:** if a commit would exceed one safe review-code-wire-pytest-regression-commit loop boundary, split it. Aim for ≤0.75 day per commit. Don't over-fragment beyond that.
- **Audio C7 baseline:** byte-identical pytest proxy holds at every commit boundary, except at explicit reset events specified in advance (with both pre and post b3sums captured).
- **Forbidden-pattern sweep:** zero runtime hits at every commit boundary. Tokenize-classified docstring/comment suppression for forensic mentions.
- **Pytest-only acceptance:** no ComfyUI Desktop runtime gates inside the sprint. Runtime quality verification is its own sprint downstream.
- **Hardware envelope:** RTX 5080 Laptop 16 GB, 14.5 GB VRAM ceiling, 8192 hard context limit, Windows-only, offline, Mistral-Nemo default LLM.
- **Git push:** Desktop Commander cmd shell. NEVER PowerShell for git (known hang issue, S30 B1b root cause).

## What does NOT belong in SPRINT.md

- Reviewer findings older than the current round (deleted after synthesis; git keeps them)
- Code change-logs (git log is the change log)
- Status updates from prior commits (just update the current Status line; git log shows history)
- Speculation about future sprints (use the "Sprint-close handoff" section to define the contract; let the next sprint's planning phase elaborate)

## SPRINT.md template (for new sprints)

When initializing a new SPRINT.md, use this skeleton:

```markdown
# Sprint: <NAME>

## Status
- Phase: planning
- Current commit: <none>
- Branch: <to-be-cut>
- Cut from: <ref>@<sha>

## Plan (commit chain)

| # | Commit | Day est. | What lands | Status |
|---|---|---|---|---|
| 1 | C0a | 0.1 | <subject> | pending |
| ... | | | | |

## Decisions log (cumulative)

| ID | Severity | Decision | Rationale | Lands at |
|---|---|---|---|---|

## Open findings (TRANSIENT — deleted after each synthesis round)

<paste reviewer text here>

## Acceptance table

| # | Check | Target |
|--:|---|---|

## Code surface citations (verbatim, so reviewers need no repo access)

### §A.1 — <topic>
```<lang>
<cited code>
```

## Reviewer instructions (paste into Gemini/ChatGPT/etc for round-robins)

> <prompt for adversarial reviewer>

## Sprint-close handoff (filled at C-final)

- What shipped:
- What's broken / known issues:
- Post-state contract for next sprint:
- Audio C7 baseline state:
- Forbidden-sweep markers added:
- New tests added (count + categories):

---

## Previous Sprint Handoff

<copy-pasted from the prior sprint's "Sprint-close handoff" section at archival time>

## Standing Project Context

<persisted from prior SPRINT.md or initialized fresh — hardware envelope, log conventions, etc.>
```
