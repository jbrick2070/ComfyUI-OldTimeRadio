# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. This session closed
**Sprint 2E** (GBNF -- resolved as DELETE, not wire). The build is
NOT finished: Sprint 2C is next, then Sprint 0's last item, then
3A-3G, 4, 5, 6. ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for build PROGRESS (Sprint Status Board +
  dated Build Progress Log at the bottom). `BUG_LOG.md` owns BUGS.
  After every work session: append a dated Build Progress Log entry +
  update the Status Board in the same edit. Two commits per session:
  one code commit, one `docs:` commit for the plan update.
- **LLM backend is HF Transformers 5.5.0** (`nodes/_otr_model_loader.py`
  `make_generate_fn` -> `model.generate()`). There is NO llama.cpp, NO
  GBNF / grammar-constrained decoding, no `transformers-cfg` /
  `outlines`. Do not reopen GBNF wiring -- Sprint 2E deleted that
  scaffolding for exactly this reason.
- **Tests + git via Desktop Commander**, `shell: "cmd"`. Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~27s,
  2656 tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **cmd path gotcha.** Use `cd /d <backslash repo path>` then
  forward-slash relative paths for pytest/git. No `tail`/`head` on
  Windows cmd. Commit message via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F` in cmd.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. Stage
  files explicitly by name, never `git add -A`.
- Run Bug Bible + full OTR suite after every code change, unprompted.

## What's done & decided
- **Sprint 2E COMPLETE -- GBNF DELETED.** Commits on `v2.0-alpha`:
  `b2b3f7d` (code, +52/-227) + `eab4b53` (plan update).
  - Open Decision 4 was overturned at implementation time: the v4
    plan said "wire GBNF into `structured_call` Attempt 4," but the
    loader has no grammar support and a real wire needs a new
    dependency (rejected -- offline-first + transformers 5.5.0 is a
    bleeding-edge stack). `_otr_style_picker.py` also never used
    `structured_call`, so its grammar could not be wired there.
  - Removed: `grammars/news_interpreter.gbnf`, `grammars/style_picker.gbnf`,
    the `grammars/` dir; `GRAMMAR_PATH` constant + `Path` import from
    `news_interpreter.py`; the `grammar_path` param, `_GRAMMAR_TEMPERATURE`,
    the never-reachable Attempt 4 block, and `_invoke_slot`'s TypeError
    fallback from `_otr_structured_call.py`.
  - `structured_call` is now a clean **3-rung ladder** (Attempt 1 base
    -> Attempt 2 structural retry -> Attempt 3 typed repair).
    `_DEFAULT_MAX_ATTEMPTS` 4 -> 3; the three `max_attempts=4` call
    sites (`_otr_ledger_reviewer` x2, `_otr_story_brief`) dropped to 3
    -- behaviour-identical (Attempt 4 never fired without a grammar).
  - Fixed stale GBNF docstrings in `_otr_style_picker.py` and a false
    "uses GBNF grammar-constrained generation" claim in `README.md`.
  - `tests/test_structured_call.py`: dropped the Attempt-4 test
    (`test_no_grammar_path_ends_ladder_at_attempt_three`), renumbered
    the coverage map -- now 17 tests.
- **HEAD = `eab4b53`, local == origin. Working tree clean** except the
  parked `docs/s28_diff_tmp.txt`. No code mid-edit.

## State of the art
- `nodes/_otr_structured_call.py` -- the shared 3-rung ladder.
  `structured_call(*, prompt, schema, slot_fn, base_temperature,
  structural_retry_temperature, repair_prompt_factory, post_validator,
  max_new_tokens, max_attempts, helper_name) -> T`. `RepairPromptFactory`
  Protocol + `default_repair_prompt_factory` live here. All five
  structured-JSON passes (ledger reviewer audit + doctor, story_brief,
  news, casting, outline x3 stages) route through it and currently
  pass NO `repair_prompt_factory` -- i.e. they all use
  `default_repair_prompt_factory`.
- Sprint Status Board (`story_pipeline_sprint_plan_v4_audited.md`):
  Sprint 0 IN PROGRESS (only the deferred CI AST `# LLM slot:` sweep
  left), 1 COMPLETE, 2A COMPLETE, 2B COMPLETE, 2C NOT STARTED, 2D
  COMPLETE, 2E COMPLETE, 3A-3G NOT STARTED, 4 NOT STARTED, 5 NOT
  STARTED, 6 NOT STARTED.
- Last full regression (2026-05-25, post-2E): OTR suite 2635 passed /
  21 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed;
  audio-byte-identical green. 0 failed.

## Immediate next steps
Pick up at **Sprint 2C -- typed repair prompts by failure class**
(plan section 2C):

1. Build bespoke `RepairPromptFactory` implementations, one per
   failure class: `json_syntax_repair`, `schema_field_repair`,
   `cast_membership_repair`, `too_many_words_repair`,
   `narration_leak_repair`, `forbidden_name_repair`. They replace
   `default_repair_prompt_factory` at the relevant `structured_call`
   call sites (the factory is passed via the `repair_prompt_factory`
   kwarg, currently omitted everywhere).
2. `cast_membership_repair` must NOT call the LLM when Levenshtein
   resolves the typo deterministically. Reuse the EXISTING matcher --
   `_levenshtein` + `auto_remap_phantom` in `_otr_ledger_reviewer.py`
   (threshold 3). Do not write a second Levenshtein.
3. Decide where the typed factories live (likely a new
   `nodes/_otr_repair_prompts.py` or inside `_otr_structured_call.py`)
   and which call site gets which factory; record the mapping in the
   plan.
4. Add regression tests; run Bug Bible + full OTR suite; commit
   (code + `docs:` plan update); push; verify HEAD match + AST parse.

Then: Sprint 0's deferred CI AST `# LLM slot:` sweep, then 3A-3G,
4, 5, 6 in plan order.

## Open questions
- Sprint 2C: where the typed factories should live, and whether each
  pass needs all six failure classes or a subset -- decide from the
  schema/`post_validator` of each call site when 2C starts.
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
