---
name: otr-handoff
description: Bidirectional OTR build baton for ComfyUI-OldTimeRadio -- RESUME at the start of a fresh window, HAND OFF at wrap-up, with anti-drift guardrails. Use for /otr-handoff, $otr-handoff, "resume/continue the OTR build", "pick up where we left off", "what's the current step", "hand off the build", "save the build context", "session handoff", "wrap up for a new chat", GO_FORWARD_PLAN, or HANDOFF_LOG. v2 (2026-08-07) -- replaces the stale 2026-06 skill (dead workflow name, no-push era, no log format, no Bible delta-scrape).
---

# OTR Handoff v2 -- the build baton

Two modes. Detect which one from the request: a fresh window resuming = RESUME;
a session wrapping up = HAND OFF. The baton is
`docs/GO_FORWARD_PLAN.md` (forward-only: queue + open bugs + budget ladder) plus
`docs/HANDOFF_LOG.md` (append-only history, newest at top). Nothing else is a
source of truth; `ROADMAP.md` is later-runway only.

Hard context that outranks this skill: `CLAUDE.md` at the ComfyUI root and in the
repo. If this skill and CLAUDE.md ever disagree, CLAUDE.md wins -- then fix this
skill in the same session.

## RESUME (fresh window)

Read, in order, before writing anything:

1. `docs/GO_FORWARD_PLAN.md` -- the operator-ordered QUEUE at the top, the
   MODEL & CREDIT BUDGET ladder, and "Window packing".
2. The top 2-3 entries of `docs/HANDOFF_LOG.md`.
3. Git reality via Desktop Commander: branch (must be `v2.0-alpha`), local HEAD
   vs `origin/v2.0-alpha`, dirty/untracked files. If HEAD != origin or there are
   uncommitted changes from a prior window, REPORT that first -- do not silently
   build on top of an unpushed or dirty state.

Then state, in the FIRST reply:

- **Which window this session is** (CODER / RENDER / PLANNER per the Window
  packing table). Never boot by letter alone -- boot by the queue: take the
  topmost item not blocked on the operator.
- **The current step** -- the exact queue item, its plan doc if one exists, and
  what "done" looks like for it (suite baseline, proof leg, or doc).
- **The MODEL & CREDIT BUDGET rung** and why (mandatory per GO_FORWARD).
- **Blockers** -- items marked blocked-on-operator are SKIPPED, not guessed at.

Then wait for go unless the kickoff line already said to execute.

Anti-drift on resume:

- **Grep before re-planning a carried-forward task.** A stale note is not a work
  item -- check the constant/code/doc it references before scheduling anything
  (the credits-scroll task was already shipped when a note re-surfaced it).
- **Remote/cloud Cowork session?** Read the "If the window is a REMOTE / cloud
  Cowork session" section of GO_FORWARD first -- file tools hit the container,
  and `/mnt/user-data/uploads/` is a lagging snapshot that reports phantom
  corruption. Route everything through Desktop Commander on the Windows paths.
- **Review routing is whatever GO_FORWARD says TODAY -- read it, do not assume.**
  This skill used to hard-code "any coding item carries the full
  `kibitz-plugin:kibitz` gate (r1-r4)". That gate is SUSPENDED as of the
  operator's 2026-08-11 directive, which routes a QUANDARY to **Codex CLI** and
  the post-coding QA on the finished diff to **Sonnet 5**, with no r1-r4 arc
  opened and no scoped tail ever reported as one. The 08-04 full-kibitz gate is
  suspended, not dead, and returns if the operator withdraws 08-11 -- so state
  the ROUTING YOU READ in the opening statement, with its date, rather than
  repeating either from memory.
- What 08-11 did NOT drop, and no routing change ever does: **Bug Bible
  regression every turn**, the BOM check on every touched file, the full suite,
  `build_variants.py --check`, AST parse on touched `.py`, and HEAD == origin
  after the push. The two-strikes floor also still stands underneath: a bug that
  survives two fixes gets a consult before the third swing.

## HAND OFF (wrap-up)

Pre-flight -- all three, before touching any doc:

1. **No handoff while background tasks are still running** (operator rule
   2026-08-06). Finish or explicitly kill/report them first.
2. State the final suite numbers (e.g. `9092/111/1`) and Bug Bible count as
   actually run this session -- or say plainly they were not run and why.
3. State box state: resident server or clean? VRAM back to baseline? The next
   window inherits the box, so the log entry must say what is running.

Then, in order:

### 1. Refresh GO_FORWARD_PLAN.md (forward-only)

- Remove or tombstone anything this session finished. If it is DONE, it does not
  belong there -- completed work lives only in the log entry.
- Update the queue: re-number if items closed, mark newly blocked items with WHO
  they are blocked on, fold newly surfaced work in where the operator's order
  puts it (or flag it for the operator if the order is unclear).
- Carry suite/Bible baselines forward so the next window can detect drift.
- Coder windows update queue STATE only; plan authorship stays with the planner
  window per Window packing.

### 2. Bible delta-scrape check (added 2026-08-07)

Before appending the log entry, run the Bible check:

- The Bible repo (`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`)
  carries `otr_coverage_index.yaml`: all 369 OTR bug records through 2026-08-07
  are already mapped to Bible ids (261-entry Bible). NEVER re-scrape indexed
  history -- the full scrape was paid once (~4M tokens); only the delta past the
  index date ever gets scraped.
- If this session recorded a NEW bug (a PROD_BUG_LOG/BUG_LOG entry or a
  confirmed live failure per the admission rule -- review observations and
  invented fixtures do NOT qualify): check it against the index + BUG_BIBLE.yaml.
  Genuinely uncovered -> promote it as a Bible entry AND append its index row in
  the same change (the README entry count moves with it -- Three-File Contract).
- Never pin or vendor a stale copy of the Bible or its tests; sync to the Bible
  repo's origin/main before running `tests\bug_bible_regression.py` (cd to the
  Bible repo root, relative path -- an absolute forward-slash path fails to
  collect).

### 3. Append the HANDOFF_LOG entry (newest at top)

Format -- match the existing entries exactly:

```
## YYYY-MM-DD -- HEAD <sha> (v2.0-alpha) -- <WINDOW KIND> (<one-line what happened>)

Did: <what the session actually did, with receipts -- commit shas, suite
  numbers, artifact paths, the one log line that proves a live leg>.
Current step: <where the queue stands now>.
Next: <the next window's first concrete action, including what it is blocked on>.
Models: <which rungs were actually used; whether the kibitz gate applied and
  what the panel really was -- a partial campaign is reported as a scoped tail
  with a scope receipt, NEVER worded as a full arc>.
Commits: <shas pushed, or "docs-only">.
```

- No long logs -- the important line, exit code, hash, or path only.
- ASCII punctuation throughout (`--`, plain quotes). UTF-8, no BOM.

### 4. Commit AND push the docs -- by pathspec

- `git add` the exact files (`docs/GO_FORWARD_PLAN.md docs/HANDOFF_LOG.md` plus
  anything else this step touched). NEVER `git add .` or `git add -A` -- other
  windows may have staged or dirty state in the shared index, and a blanket add
  once swept three staged deletions to origin.
- Commit with an ASCII-safe message, push `origin v2.0-alpha` (pushing to
  v2.0-alpha is always required -- local-only commits are the failure mode).
- Verify: HEAD == origin, no 0-byte files, no BOM on touched files, AST parse on
  any touched `.py`.
- If a stale `.git\index.lock` blocks the commit and `Get-Process git` is empty,
  remove the lock and retry once.

### 5. Print the next kickoff

End with the one-line kickoff the next window pastes, in the GO_FORWARD "NEVER
boot a window by letter" shape: resume + window kind + "read GO_FORWARD 'ON
DECK'/QUEUE and execute in stated order" + full-kibitz gate + "state your MODEL
& CREDIT BUDGET rung first".

## Standing guardrails (both modes)

- The handoff is a baton, not a history dump -- compact enough that the next
  window acts immediately without asking anything.
- Workflow source of truth is `workflows/otr_canonical.json` (the old
  `otr_scifi_16gb_full.json` name is dead). Any session that touched it lists
  the widget/link/schema validation actually run.
- Preserve user/unrelated changes; never imply a future window should revert
  them unless Jeffrey explicitly asked.
- One coder window in the code at a time, serialized through GO_FORWARD.
