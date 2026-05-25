# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (LLM-audit punch list)

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. Decompose every LLM call in
the script-writing + cleanup pipeline into single-job passes with
deterministic guards, so a structurally valid script can no longer ship
dramatically empty. This session closed the three actionable items on
the LLM-audit punch list (`docs/2026-05-24-story-pipeline-llm-audit.md`):
BUG-LOCAL-271, the 3C Doctor-row enrichment, and WORD_BUDGET_DRIFT. The
build is NOT finished: Sprints 3A, 4, 5, 6 remain. ROADMAP stays parked
until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **HEAD after this session = the `docs:` commit that carries this
  file.** Substantive code commits: `14818bb` (3C enrichment),
  `3e120df` (BUG-LOCAL-271), `dfd63ee` (WORD_BUDGET_DRIFT). Predecessor
  HEAD was `a294297`. Working tree clean except the parked
  `docs/s28_diff_tmp.txt` (never commit it).
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for PROGRESS; `BUG_LOG.md` owns BUGS. After a
  work session: append a dated Build Progress Log entry + update the
  Status Board in the same edit.
- **LLM backend is HF Transformers 5.5.0** (`model.generate()`). No
  llama.cpp, no GBNF. Sprint 2E deleted that scaffolding -- do not reopen.
- **Tests + git via Desktop Commander, `shell: "cmd"`.** Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~28s, 2808
  collected). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **The Cowork Linux sandbox `Bash` mount is STALE for this repo.** Use
  Desktop Commander (real Windows FS) for tests + git. The Read/Write/
  Edit file tools DO write the real Windows file correctly.
- `cmd` quoting gotchas: no `git log --format="%h %s"` (use `--oneline`
  or `%H%n%s`); no inline `python -c "..."` through cmd; commit messages
  via the file tool to `.git\COMMIT_EDITMSG`, then `git commit -F`.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- For console-output analysis, Jeffrey must paste it -- the AI has no
  real-time ComfyUI log access and `comfyui_8000.log` is stale.

## What's done & decided
- **LLM-audit punch list -- three items closed, three parallel subagents
  on disjoint files, lead-integrated:**
  - `14818bb` -- **3C Doctor-row enrichment 5/7 -> 7/7.**
    `production_ledger.py` (`init_lines_from_outline` + `set_lines`)
    now stamps `beat_intent` (from `Beat.intent`) and `target_words`
    (from `Beat.target_words`) onto every per-line record. The Script
    Doctor renderer `_render_lines_for_doctor` was already pre-wired to
    emit both when present -- no renderer logic change, comment refresh
    only. `OTR_LedgerScriptWriter.py` needed no change.
  - `3e120df` -- **BUG-LOCAL-271 FIXED.** The `wrong_char_id` cast
    auto-repair was dead: auditor emitted `expected`=char_id, repair
    resolved `expected` as a NAME. Fix (approach b): repair now accepts
    a char_id, validated against a case-fold `valid_char_ids` map from
    `cast_rows`; a line already on the right char_id counts as repaired
    (kills the small-model over-flagging); `_AUDITOR_SYSTEM_PROMPT`
    updated to document `wrong_char_id.expected` as a char_id.
  - `dfd63ee` -- **WORD_BUDGET_DRIFT FIXED.** The writer's word-budget
    check summed ALL beats vs a voiced-only target; announcer beats'
    fixed ~15-word overhead forced ratio 2.00 on small targets. Now
    sums voiced (`speaker_role == "character"`) beats only. The outline
    allocator was correct -- no `_otr_outline.py` change.
- **Regression green** (authoritative combined-tree run by the lead):
  full OTR suite 2787 passed / 21 skipped (2808 collected); Bug Bible
  16 passed / 7 skipped / 3 xfailed. 0 failed.
- No node `INPUT_TYPES` / widget / socket touched -> no workflow JSON
  re-wire. No LLM call added or removed.
- **Sprints 0, 1, 2A-2E, 3B-3G COMPLETE** (earlier sessions).
- **Build method: parallel subagents on disjoint file sets** -- used
  again this session (3 lanes).

## State of the art
- Build tracker `story_pipeline_sprint_plan_v4_audited.md` Status Board:
  Sprints 0, 1, 2A-2E, 3B-3G COMPLETE; 3A, 4, 5, 6 NOT STARTED.
- **BUG-LOCAL-271 is FIXED in code** but NOT yet live-verified -- a
  ComfyUI episode whose `audit_cast_contract:pre` resolves
  `wrong_char_id` violations (`repaired` count > 0) instead of
  escalating all to the Script Doctor. Bible promotion (Three-File
  Contract) waits on that live verification.
- **Finding C still open.** The pipeline can still ship a structurally
  clean but THIN episode -- the `ozempics_glitch`-class gap. The quality
  critic is Sprint 5, not yet built.
- The cast/style/seed-removal commits (`61dda9c` / `906a57f` /
  `d0ea595`) are still NOT live-validated.

## Immediate next steps
1. **Live-validate the open batch in ONE ComfyUI run on current HEAD:**
   confirm (a) BUG-271 -- `wrong_char_id` violations now repair instead
   of escalating; (b) the cast varies (not the pinned `HAYES VANCE`)
   and the writer node has no `seed` widget; (c) WORD_BUDGET_DRIFT no
   longer false-fires; (d) the episode completes + freezes clean.
   Operator-gated -- Jeffrey starts the run and pastes the console.
2. **Sprint 3A** -- rewrite `compose_line` (`nodes/_otr_line_composer.py`,
   ~1660 lines). Lead-driven, NOT a one-shot subagent. 2-3 days.
3. Then Sprint 4 (VRAM verify, live-hardware gated), then 5 (continuity
   ledger + story critic + targeted reroll -- the Finding C fix; depends
   on 3A), then 6 (critic -> render coupling, ships with 5).

## Open questions
- The 4 empty-ledger `906a57f` runs (`pending_20260525_133637..134632`)
  -- aborted operator iterations, or a real writer failure on the
  cast/style commits? Needs their console output to tell.
- Carried: BUG-LOCAL-265/-266/-267 + now -271 Bible-promotion via the
  Three-File Contract (all await live verification).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
