# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25

## Core goal
The Story Pipeline Ledger Writer Hardening build (`story_pipeline_sprint_plan_v4_audited.md`).
This session finished **Wave 2** -- converting the four remaining
structured-JSON LLM call sites onto the shared `structured_call` retry
ladder -- and then did a follow-up **S32 B1 cleanup** removing the
unused paired-signature params. Both are committed and pushed. The
build is NOT finished: several sprints remain (see Immediate next
steps). ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for build PROGRESS (Sprint Status Board +
  dated Build Progress Log at the bottom). `BUG_LOG.md` owns BUGS.
  After every work session: append a dated Build Progress Log entry +
  update the Status Board in the same edit. ROADMAP untouched until
  the whole build lands.
- **Tests + git via Desktop Commander**, `shell: "cmd"`. Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~33s,
  ~2657 tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **cmd path gotcha.** Use `cd /d <backslash repo path>` then
  forward-slash relative paths for pytest/git. Commit message via the
  file tool to `.git\COMMIT_EDITMSG`, then `git commit -F` in cmd.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. Stage
  files explicitly by name, never `git add -A`.
- Run Bug Bible + full OTR suite after every code change, unprompted.

## What's done & decided
- **Wave 2 -- four structured_call conversions, all on `v2.0-alpha`,
  pushed.** Predecessor HEAD `f996544`.
  - `3f41fc8` -- `_otr_story_brief.run_story_brief_reflection`
  - `b4c6e83` -- `news_interpreter.build_news_briefs`
  - `6e2950d` -- `_otr_casting.cast_one_character`
  - `476eabc` -- `_otr_outline.generate_outline` (3 stages)
  - `2fc75c3` -- plan-tracking (Sprint 2A + 2D -> COMPLETE)
  Each helper's hand-rolled call->parse->validate->repair loop is now
  one `structured_call`; content validation rides `post_validator`;
  failures map to each function's existing contract. Every converted
  pass's structural retry now LOWERS temperature (Sprint 2B fix).
- **S32 B1 cleanup -- pushed.** `6940209` (13 files) + `b6db0e5`
  (plan). Removed the unused paired-signature params:
  `lock_cast.technical_fn`, `build_news_briefs.creative_fn`,
  `compose_line.technical_fn`. `pick_style` keeps both (uses both).
  Reverses Sprint 0 Decision 6.
- **Decisions (Jeffrey, this session):** casting fully converted with
  S32 B3 reversed (no per-attempt slot switching -- one `slot_fn`);
  deprecated params removed cleanly, not left as shims.
- **Dead code removed:** story_brief `_repair_pass` /
  `_build_repair_messages`; news + casting `_REPAIR_RAW_CAP_CHARS`;
  outline `_run_call_with_retry` / `_REPAIR_PROMPT_TEMPLATE`;
  `tests/test_story_brief_clamp_logging.py` deleted (SA-101 clamp log
  gone with `_repair_pass`).
- **HEAD = `b6db0e5`, local == origin. Working tree clean** except the
  parked `docs/s28_diff_tmp.txt`. No code mid-edit.

## State of the art
- `nodes/_otr_structured_call.py` -- the shared ladder, unchanged this
  session (extended in `fed6327`). All five structured-JSON passes
  (ledger reviewer x2, story_brief, news, casting, outline x3) route
  through it.
- Sprint Status Board (`story_pipeline_sprint_plan_v4_audited.md`):
  Sprint 0 IN PROGRESS (only the deferred CI AST `# LLM slot:` sweep
  left), 1 COMPLETE, 2A COMPLETE, 2B COMPLETE, 2C NOT STARTED, 2D
  COMPLETE, 2E NOT STARTED, 3A-3G NOT STARTED, 4 NOT STARTED, 5 NOT
  STARTED, 6 NOT STARTED.
- Last full regression (2026-05-25, post-S32-B1): OTR suite 2636
  passed / 21 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed;
  audio-byte-identical green.

## Immediate next steps
The build is NOT done. Remaining sprints, in plan order:

1. **Sprint 2E -- GBNF wire.** Wire `grammars/news_interpreter.gbnf`
   + `grammars/style_picker.gbnf` (the `GRAMMAR_PATH` constants exist;
   the loader never consumes them) into `structured_call` Attempt 4
   via the `grammar_path` parameter. Decision 4 resolved -> wire it.
   This is the natural next step -- it completes the structured_call
   story for the passes Wave 2 just converted.
2. **Sprint 2C -- typed repair prompts.** Bespoke per-failure-class
   `RepairPromptFactory` implementations. All Wave 2 conversions
   currently use `default_repair_prompt_factory`; 2C swaps in typed
   factories per pass.
3. **Sprint 0 -- finish.** The one remaining item: the deferred CI AST
   `# LLM slot:` sweep (audit gate that every LLM call site carries a
   creative/technical tag).
4. **Sprints 3A-3G** -- split compose_line, outline Stage 3, split
   Script Doctor, split Casting, title scratchpad, cast auditor
   confidence, reflection sanitize.
5. **Sprint 4** -- VRAM hardening (gate exists; verify only).
6. **Sprints 5 + 6** -- continuity + critic + reroll; critic->render
   coupling (6 ships with 5).

Pick up at Sprint 2E unless Jeffrey redirects.

## Open questions
- Carried, still unaddressed: Gemma-4 / 90-word test episode
  (operator-gated -- needs a live ComfyUI run); Bible-promotion of
  BUG-LOCAL-265/-266/-267 via the Three-File Contract.
- `CLAUDE.md` still carries a "Round-Robin Consultation" section as a
  general project rule; the round-robin waiver was scoped to the
  sprint plan only. Flag for Jeffrey if he wants CLAUDE.md amended.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
