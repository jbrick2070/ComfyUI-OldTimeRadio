# OTR Sprint Workflow Setup — One-File Pattern

You wanted: minimum file proliferation, maximum Cowork automation, no agent complexity.

This is the setup.

## What changes from your current pattern

**Current (creates clutter):**
```
docs/2026-05-12-story-brief-v2-research.md
docs/2026-05-12-story-brief-v2-design-refinements.md
docs/2026-05-13-story-brief-v2-go-forward-plan.md
docs/2026-05-15-sprint-c-cowork-review-findings.md
docs/2026-05-15-sprint-c-plan-final.md
docs/2026-05-15-sprint-c-plan-final-v2.md
docs/2026-05-15-sprint-c-plan-final-v3.md   <-- the active one
docs/2026-05-15-round-robin-3.md
docs/2026-05-15-round-robin-4.md
...
```

**New (clean):**
```
SPRINT.md              <-- the ONE active file, edited in place
SKILL.md               <-- workflow rules for Cowork
docs/closed-sprints/
  2026-05-08-sprint-s34-p0-p1-hotfix.md
  2026-05-15-sprint-c.md        <-- archived after C-final
  2026-05-22-sprint-a.md        <-- next sprint, archived after its close
  ...
```

Git history captures every revision of SPRINT.md. You can `git log SPRINT.md` and see every round-robin synthesis, every commit gate, every status update.

## Setup steps (one-time)

1. Drop `SKILL.md` at repo root (file delivered alongside this README).
2. Rename your current v3 plan:
   ```
   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   git mv docs\2026-05-15-sprint-c-plan-final-v3.md SPRINT.md
   ```
3. Delete the old versions from disk (they're in git):
   ```
   git rm docs\2026-05-15-sprint-c-plan-final.md
   git rm docs\2026-05-15-sprint-c-plan-final-v2.md
   git rm docs\2026-05-15-round-robin-3.md
   git rm docs\2026-05-15-round-robin-4.md
   ```
4. Create the archive folder:
   ```
   mkdir docs\closed-sprints
   ```
5. Commit:
   ```
   git add -A
   git commit -m "workflow: consolidate to SPRINT.md + SKILL.md one-file pattern"
   ```

## Per-round workflow (round-robin review)

**Your part (5 min):**
1. Open Cowork in this repo's directory.
2. Say: "Produce a review request from SPRINT.md."
3. Cowork prints the plan + the §9 reviewer instructions. Copy that.
4. Paste into Gemini (or ChatGPT, or a Claude conversation). Get findings back.
5. Paste the findings into Cowork: "Apply these findings to SPRINT.md."

**Cowork's part (auto, reading SKILL.md for the rules):**
- Synthesizes each finding: ACCEPT / REJECT / PARTIAL ACCEPT.
- Updates the Plan section in place.
- Adds entries to the Decisions Log.
- Deletes the transient findings text.
- Commits with `<sprint>: round-<N> synthesis (X accept / Y reject)`.

## Per-commit workflow (execution)

**Your part (~30 seconds per commit):**
1. Open Cowork.
2. Say: "Execute the next commit in SPRINT.md."

**Cowork's part (auto):**
- Reads SPRINT.md, finds Status → Current commit.
- Reads that commit's spec section.
- Reviews → codes → wires → runs the pytest table → runs Bug Bible regression → commits with the spec'd subject.
- Updates Status → Current commit = next.
- Stops at the next commit boundary so you can check before the next runs (or you can tell it "execute through C-final" and walk away).

## Sprint close workflow

When all commits have landed:

**Your part:**
1. Say: "Close the sprint."

**Cowork's part:**
- Fills "Sprint-close handoff" section in SPRINT.md (what shipped, broken items, post-state contract, baseline state, new tests count).
- Runs wide pytest walk, confirms acceptance table green.
- Lands C-final commit.
- Pushes branch.
- Archives: `git mv SPRINT.md docs/closed-sprints/2026-05-15-sprint-c.md`.
- Initializes new SPRINT.md from the template in SKILL.md.
- Copies prior sprint's handoff section into new "Previous Sprint Handoff".
- Commits the archival + new file.

## What survives across sprints inside SPRINT.md

- "Previous Sprint Handoff" section — copy-pasted from the just-closed sprint's "Sprint-close handoff".
- "Standing Project Context" section — your hardware envelope, log conventions, file paths, the no-curse / no-dummy / no-change-logs rules. Carried forward unchanged unless you decide to update them.

Everything else starts fresh per sprint.

## What about the round-robin reviewer docs you've been keeping?

Right now you keep them as separate dated files. New pattern: paste them into "Open findings (TRANSIENT)", let Cowork synthesize, let Cowork delete. They live in the git commit diff if you ever need them — you can recover any synthesis round's reviewer input with `git show <synthesis-commit>:SPRINT.md`.

If you genuinely want a permanent archive of reviewer findings (rather than just relying on git), one option: a single `docs/reviewer-archive.md` file you append to, dated entries, never re-edited. Still ONE file, not many.

## Why this works without going full agent

The three things you need an "agent" for are:
1. **Remembering the conventions across sessions** — solved by SKILL.md.
2. **Doing the file edits and commits without you typing them** — Cowork already does this; it just needs to know what to edit (SKILL.md tells it).
3. **Running the commit loop without supervision** — Cowork can execute "the next commit per SPRINT.md spec" because the spec is precise enough to be self-driving (your v3 plan IS that precise — pytest tables, code snippets, commit subjects, all there).

That covers ~95% of what a full agent would do. The remaining 5% is human judgment moments: deciding which reviewer findings to accept, deciding when to override a gate, deciding when to split a commit further. You keep those.

## Gotchas

- **Don't let Cowork run multiple commits past an unexpected failure.** SKILL.md says "stop the sprint" on any failure; trust that. If pytest fails or audio C7 drifts unexpectedly, you want to see it, not have the agent paper over it.
- **Branch cut hygiene.** Your row-20 precondition (clean Windows checkout, branch ready) is still your responsibility before saying "execute the next commit" at C0a. Cowork will check `git status --short` per the spec but the operator action of cleaning a poisoned index is yours.
- **Reviewer doc length.** If a reviewer dumps a 5000-word critique, paste it whole into "Open findings" — Cowork synthesizes from the full text, then deletes. Don't trim before pasting.

## TL;DR

```
ONE active file:  SPRINT.md
ONE workflow file: SKILL.md
History:          git log SPRINT.md
Archive:          docs/closed-sprints/<date>-<name>.md
Per round:        paste findings into SPRINT.md, tell Cowork "apply"
Per commit:       tell Cowork "execute next commit per SPRINT.md"
Sprint close:     tell Cowork "close the sprint"
```
