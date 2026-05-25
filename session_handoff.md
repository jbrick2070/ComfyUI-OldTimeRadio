# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (Wave 1 + seed work)

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. Decompose every LLM call in
the script-writing + cleanup pipeline into single-job passes with
deterministic guards, so a structurally valid script can no longer ship
dramatically empty. This session shipped the **Sprint 3B-3E wave**,
**live-validated 3B-3E**, then on Jeffrey's direction decoupled the cast
and style-picker RNGs from the `seed` widget and **removed the `seed`
widget entirely**. The build is NOT finished: 3A, 4, 5, 6 remain, plus
BUG-LOCAL-271. ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **HEAD = `d0ea595`, pushed; `origin/v2.0-alpha` == HEAD.** Working
  tree clean except the parked `docs/s28_diff_tmp.txt` (never commit it).
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for PROGRESS; `BUG_LOG.md` owns BUGS. After a
  work session: append a dated Build Progress Log entry + update the
  Status Board in the same edit. One code commit per sprint, one `docs:`
  commit for the plan.
- **LLM backend is HF Transformers 5.5.0** (`model.generate()`). No
  llama.cpp, no GBNF. Sprint 2E deleted that scaffolding -- do not reopen.
- **Tests + git via Desktop Commander, `shell: "cmd"`.** Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~30s, ~2806
  tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **The Cowork Linux sandbox `Bash` mount is STALE for this repo.** Use
  Desktop Commander (real Windows FS) for tests + git. The Read/Write/
  Edit file tools DO write the real Windows file correctly.
- `cmd` quoting gotchas: no `git log --format="%h %s"` (use `--oneline`
  or `%H%n%s`); no inline `python -c "..."` through cmd; commit messages
  via the file tool to `.git\COMMIT_EDITMSG`, then `git commit -F`.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- **The ComfyUI console log file `comfyui_8000.log` is STALE (May 19)** --
  today's runs are NOT in it. Episode ledgers under
  `output\otr\episodes\<ep>\audio\<ep>_ledger.json` are the on-disk
  record; for console-output analysis, Jeffrey must paste it.

## What's done & decided
- **Sprint 3B-3E wave COMPLETE + pushed** -- four parallel subagents on
  disjoint files, lead-integrated:
  - `3992607` 3B -- outline Stage 3: adjacency context in the beat
    prompt; `target_words` dropped from `_BeatFleshout`. (Plan asked for
    `next_beat_intent`; Stage 3 is sequential so a later beat's intent
    does not exist yet -- shipped `next_beat_speaker` instead.)
  - `74438ff` 3C -- Script Doctor split into `run_script_doctor_diagnosis`
    + `run_script_doctor_edits`; +1 LLM call (technical, tagged; sweep
    21/21). Doctor rows enriched 5/7 fields -- `beat_intent`/`target_words`
    are NOT on the ledger (follow-up: stamp them).
  - `5fe9931` 3D -- casting split: `precompute_ensemble_slots` /
    `llm_write_description` / `python_assign_voice_preset`. Python owns
    gender + voice; the LLM writes the description only.
  - `d230cd6` 3E -- title scratchpad + `EPISODE_TITLE: TBD` late binding;
    post-hoc title substitution removed.
  - `1514e11` docs commit (plan board + Build Progress Log + LLM audit).
- **3B-3E LIVE-VALIDATED.** Episode `signal_lost_vances_promise_20260525_125401`
  (id `pending_20260525_125025`, commit `1514e11`) ran end-to-end --
  froze `frozen_with_warns`, full episode produced. 3B/3C/3D/3E all
  confirmed working (3C's undiagnosed-edit drop fired correctly). See
  the "Live-run validation" section in
  `docs/2026-05-24-story-pipeline-llm-audit.md`.
- **BUG-LOCAL-269 + 270 -- cast + style RNG decoupled from the seed.**
  `61dda9c` (cast) + `906a57f` (style): the fixed `seed` widget pinned
  the cast (`HAYES VANCE / GULLIVER REEVES / JIMBO BLACK` every episode)
  and the style-picker's 5 sampled seed-flavors. Both now draw OS
  entropy per episode; `OTR_CAST_SEED` / `OTR_STYLE_SEED` env vars are
  the C7 reproducibility override. Twins of BUG-LOCAL-260.
- **Seed widget REMOVED.** `d0ea595`: with cast/style/LEMMY all decoupled,
  the `seed` widget drove nothing -- removed from `OTR_LedgerScriptWriter`
  INPUT_TYPES + the workflow JSON (`widgets_values` 19 -> 17). 6
  widget-vector tests re-aligned. Not a bug -- a cleanup.
- **Sprints 0, 1, 2A-2E, 3F, 3G COMPLETE** (earlier sessions).
- **Build method: parallel subagents on disjoint file sets** -- validated
  again this session (3B/3C/3D/3E ran as four concurrent subagents).

## State of the art
- Build tracker `story_pipeline_sprint_plan_v4_audited.md` Status Board:
  Sprints 0, 1, 2A-2E, 3B-3G COMPLETE; 3A, 4, 5, 6 NOT STARTED.
- **BUG-LOCAL-271 OPEN (fix pending).** The live run surfaced it: the
  cast auditor flags `wrong_char_id` with `expected` = a char_id, but
  `apply_deterministic_cast_repairs` resolves `expected` as a NAME --
  contract mismatch, every `wrong_char_id` violation goes unrepaired.
  Benign on the validated episode; the auto-repair is non-functional.
- **Finding C still open.** The validated episode is structurally clean
  but THIN -- 21 words of character dialogue, one phantom name -- exactly
  the `ozempics_glitch`-class gap. 3B-3E hardened the *structure*; the
  *quality* critic is Sprint 5, not yet built.
- Last regression (2026-05-25, post-seed-removal `d0ea595`): full OTR
  suite 2787 passed / 21 skipped; Bug Bible 16 passed / 7 skipped /
  3 xfailed. 0 failed.
- The cast/style/seed-removal commits (`61dda9c` / `906a57f` / `d0ea595`)
  are NOT yet live-validated. The runs after `vances_promise` on commit
  `906a57f` (`pending_20260525_133637..134632`) left empty-cast ledgers
  -- likely aborted operator iterations, not analyzed; no console output.

## Immediate next steps
1. **Fix BUG-LOCAL-271** -- align the cast auditor's `wrong_char_id.expected`
   with what `apply_deterministic_cast_repairs` consumes (auditor emits
   a NAME, or the repair accepts a char_id), and tighten the auditor
   prompt so it stops over-flagging already-correct char_ids.
2. **Live-validate the cast/style/seed-removal batch** -- one ComfyUI
   episode on HEAD `d0ea595`: confirm the cast varies (not HAYES VANCE),
   the writer node has no `seed` widget, the episode completes + freezes.
3. **Sprint 3A** -- rewrite `compose_line` (`nodes/_otr_line_composer.py`,
   ~1660 lines). Lead-driven, NOT a one-shot subagent. 2-3 days.
4. Then Sprint 4 (VRAM verify, live-hardware gated), then 5 (continuity
   ledger + story critic + targeted reroll -- the Finding C fix), then 6.

## Open questions
- BUG-LOCAL-271 fix direction (auditor prompt vs repair branch) -- pick one.
- The 4 empty-ledger `906a57f` runs -- aborted iterations, or a real
  writer failure on the cast/style commits? Needs their console output
  to tell. The `1514e11` validation run was clean.
- 3C's `beat_intent` / `target_words` Doctor-row enrichment is deferred
  -- those fields are not on the production ledger; stamping them is a
  follow-up in `OTR_LedgerScriptWriter.py` / `production_ledger.py`.
- Carried: BUG-LOCAL-265/-266/-267 Bible-promotion via the Three-File
  Contract.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
